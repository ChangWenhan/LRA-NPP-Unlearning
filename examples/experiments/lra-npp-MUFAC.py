import os
import sys
import copy
import time
import torch
import torch.nn as nn
import pickle
import random
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
from itertools import product
from torchvision import datasets, transforms
from torchvision.models import vision_transformer
from pathlib import Path

# ==================== 路径配置 ====================
base_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, base_path.as_posix())

# 导入依赖库
import zennit
from zennit.composites import LayerMapComposite, EpsilonGammaBox, EpsilonPlusFlat
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit

# 猴子补丁（ViT适配）
monkey_patch(vision_transformer, verbose=False)
monkey_patch_zennit(verbose=False)

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    # ========== 数据配置 ==========
    'unlearn_class': 0,      # 待遗忘的类别索引 (MUFAC dataset class 0)
    'num_classes': 8,        # 数据集总类别数
    'dataset_root': '/home/cwh/Workspace/TorchLRP-master/data/custom_korean_family_dataset_resolution_128', # 数据集根目录
    'max_unlearn_samples': 36,  # [Modify 4] 限制用于分析的Unlearn样本数量 (None为不限制)
    
    # ========== 模型路径 ==========
    # [Modify 1] 指定加载的模型路径
    'checkpoint_path': '/home/cwh/Workspace/TorchLRP-master/examples/models/vit_best_on_test.pth',
    
    # ========== 独立参数列表（批量实验） ==========
    'k_top_neurons_list': [400],     # 每样本分析的Top-k神经元数
    'h_high_freq_list': [600],       # 最终修改的高频神经元数
    'modification_types': ['zero'],  # 修改方式: 'zero'/'laplace'/'gaussian'
    'image_noise_levels': [0.0],     # 图像噪声强度
    
    # ========== [Modify 6] LRP 规则配置 ==========
    # 可选: 'gamma', 'epsilon', 'alpha_beta', 'epsilon_gamma_box'
    'lrp_rule_type': 'epsilon', 
    
    # LRP 参数 (根据选择的规则生效)
    'lrp_params': {
        'gamma': 0.25,       # 用于 Gamma 规则
        'epsilon': 1e-6,     # 用于 Epsilon 规则
        'alpha': 2.0,        # 用于 AlphaBeta 规则 (beta = alpha - 1)
        'beta': 1.0
    },
    
    # ========== 噪声参数 (修改权重时使用) ==========
    'noise_scale': 1.0,  
    
    # ========== 保存配置 ==========
    'save_results': False,
    'results_save_dir': 'vit_unlearn_results/',
    'save_model': True,
    # [Modify 5] 保存模型的路径
    'model_save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/',
}

# ==================== 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 工具函数 ====================
def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def add_image_noise(images, noise_level):
    """对图像添加高斯噪声"""
    if noise_level == 0.0:
        return images
    noise = torch.randn_like(images, device=device) * noise_level
    return torch.clamp(images + noise, 0.0, 1.0)

# ==================== [Modify 0] 数据集类 (来自 retrain_ViT.py) ====================
def parsing(meta_data):
    """解析 metadata CSV 文件"""
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, meta_data_path, image_directory, transform=None, filter_class=None, keep_only_class=None):
        """
        Args:
            filter_class: 剔除某个类别 (用于Retrain)
            keep_only_class: 仅保留某个类别 (用于Unlearn Analysis)
        """
        self.meta_data = pd.read_csv(meta_data_path)
        self.image_directory = image_directory
        self.transform = transform
        
        full_list = parsing(self.meta_data)
        
        # 标签映射
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }
        
        self.image_age_list = []
        for img_path, age_cls in full_list:
            label = self.age_class_to_label[age_cls]
            
            # 逻辑控制
            if filter_class is not None and label == filter_class:
                continue
            if keep_only_class is not None and label != keep_only_class:
                continue
                
            self.image_age_list.append((img_path, label))

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, label = self.image_age_list[idx]
        full_path = os.path.join(self.image_directory, image_path)
        img = Image.open(full_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # 返回 path 方便 debug
        return img, label, str(full_path) 

# ==================== [Modify 1] 模型加载函数 ====================
# ==================== [Modify 1] 模型加载函数 (修复版) ====================
def load_finetuned_vit(checkpoint_path, num_classes=8, device=device):
    """加载微调后的ViT模型"""
    print(f"Loading model from: {checkpoint_path}")
    
    # 1. 初始化结构 (保持不变，确保我们掌控模型架构)
    model = vision_transformer.vit_b_16(weights=None) 
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # 2. 加载权重
    if os.path.exists(checkpoint_path):
        # [Fix]: 添加 weights_only=False 以允许加载完整的模型对象 (PyTorch 2.6+ 必需)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # [Compatibility]: 判断加载的是 "完整模型对象" 还是 "state_dict"
        if isinstance(checkpoint, torch.nn.Module):
            print("  >>> [Info] 检测到 Checkpoint 是完整模型对象，正在提取 state_dict...")
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict):
            print("  >>> [Info] 检测到 Checkpoint 是 state_dict。")
            model.load_state_dict(checkpoint)
        else:
            print("  >>> [Warning] 未知格式，尝试直接加载...")
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    model.eval()
    model.to(device)
    
    # 冻结参数（仅用于推理和LRP分析）
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# ==================== LRP相关类与工具 ====================
class RelevanceHook:
    def __init__(self):
        self.activations = None
        self.gradients = None
    
    def forward_hook(self, module, input, output):
        self.activations = input[0].detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_input[0].detach()
    
    def get_relevance(self):
        if self.activations is not None and self.gradients is not None:
            return self.activations * self.gradients
        return None

# ==================== [Modify 6] LRP 规则工厂 ====================
def get_zennit_composite(model, config):
    """根据配置返回 Zennit Composite"""
    rule_type = config['lrp_rule_type']
    params = config['lrp_params']
    
    if rule_type == 'gamma':
        # 原始逻辑：Conv用Gamma，Linear用Gamma
        gamma_val = params.get('gamma', 0.25)
        print(f"Using LRP Rule: Gamma (gamma={gamma_val})")
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(gamma_val)),
            (torch.nn.Linear, z_rules.Gamma(gamma_val)), # 也可以设置不同的Linear gamma
        ])
        
    elif rule_type == 'epsilon':
        epsilon_val = params.get('epsilon', 1e-6)
        print(f"Using LRP Rule: Epsilon (epsilon={epsilon_val})")
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Epsilon(epsilon_val)),
            (torch.nn.Linear, z_rules.Epsilon(epsilon_val)),
        ])
        
    elif rule_type == 'alpha_beta':
        alpha = params.get('alpha', 2.0)
        beta = params.get('beta', 1.0)
        print(f"Using LRP Rule: AlphaBeta (alpha={alpha}, beta={beta})")
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.AlphaBeta(alpha=alpha, beta=beta)),
            (torch.nn.Linear, z_rules.AlphaBeta(alpha=alpha, beta=beta)),
        ])
    
    elif rule_type == 'epsilon_gamma_box':
        # 针对输入范围有限制的图像数据常用组合
        print("Using LRP Rule: EpsilonGammaBox")
        # 需要提供 low 和 high，这里假设归一化后的数据大概范围
        # 注意：如果transform不同，low/high需要调整
        return EpsilonGammaBox(low=-3.0, high=3.0)

    else:
        raise ValueError(f"Unknown LRP rule type: {rule_type}")

# ==================== 核心功能函数 ====================
def analyze_single_image(model, input_tensor, exp_config):
    """单图像LRP分析"""
    input_tensor.grad = None
    relevance_hook = RelevanceHook()
    
    # 注册Hook
    forward_handle = model.heads.head.register_forward_hook(relevance_hook.forward_hook)
    backward_handle = model.heads.head.register_full_backward_hook(relevance_hook.backward_hook)
    
    # [Modify 6] 获取动态 LRP 规则
    zennit_comp = get_zennit_composite(model, exp_config)
    zennit_comp.register(model)
    
    try:
        # 前向传播
        y = model(input_tensor.requires_grad_())
        predicted_class = torch.argmax(y, dim=1).item()
        
        # 反向传播（针对预测类别）
        y[0, predicted_class].backward()
        
        # 获取相关性
        relevance = relevance_hook.get_relevance()
    finally:
        # 确保资源清理
        zennit_comp.remove()
        forward_handle.remove()
        backward_handle.remove()
    
    if relevance is not None:
        return predicted_class, relevance[0].cpu().numpy()
    return predicted_class, None

def find_top_k_neurons(relevance_768d, k=10):
    abs_relevance = np.abs(relevance_768d)
    top_k_indices = np.argsort(abs_relevance)[-k:][::-1]
    return top_k_indices

def analyze_unlearn_dataset(model, unlearn_loader, exp_config, base_config):
    """分析待遗忘数据集"""
    all_top_neurons = []
    sample_predictions = []
    
    # [Modify 4] 获取样本限制
    max_samples = base_config.get('max_unlearn_samples', None)
    count = 0
    
    print(f"Analyzing unlearn dataset (Limit: {max_samples if max_samples else 'All'})...")
    
    for input_tensor, label, img_path in tqdm(unlearn_loader, desc="LRP Analysis"):
        # [Modify 4] 检查样本数量限制
        if max_samples is not None and count >= max_samples:
            break
            
        try:
            input_tensor = input_tensor.to(device)
            # 添加图像噪声
            input_tensor = add_image_noise(input_tensor, exp_config['image_noise_level'])
            
            # 如果BatchSize > 1，需要在这里处理，目前逻辑假设BatchSize=1
            if input_tensor.shape[0] > 1:
                 # 简单处理：取第一个样本（因为LRP通常是per-sample）
                 input_tensor = input_tensor[0:1]

            predicted_class, relevance_768d = analyze_single_image(
                model, input_tensor, exp_config
            )
            
            if relevance_768d is not None:
                top_k_indices = find_top_k_neurons(relevance_768d, exp_config['k_top_neurons'])
                all_top_neurons.append(top_k_indices)
                sample_predictions.append(predicted_class)
                count += 1
        
        except Exception as e:
            print(f"Warning: Error processing image {img_path}: {str(e)}")
            continue
    
    neuron_frequency = Counter([idx for sublist in all_top_neurons for idx in sublist])
    # 目标类别默认为 config 指定的类别
    target_class_idx = base_config['unlearn_class']
    
    return {
        'neuron_frequency': neuron_frequency,
        'target_class_idx': target_class_idx,
        'all_top_neurons': all_top_neurons,
        'sample_predictions': sample_predictions
    }

def modify_classification_weights(model, high_freq_neurons, target_class_idx, modification_type, noise_scale):
    """修改分类头权重"""
    fc_weights = model.heads.head.weight.data  # [num_classes, 768]
    
    if modification_type == 'zero':
        fc_weights[target_class_idx, high_freq_neurons] = 0.0
    elif modification_type == 'gaussian':
        noise = torch.randn(len(high_freq_neurons), device=device) * noise_scale
        fc_weights[target_class_idx, high_freq_neurons] += noise
    elif modification_type == 'laplace':
        noise = torch.distributions.laplace.Laplace(0.0, noise_scale).sample((len(high_freq_neurons),)).to(device)
        fc_weights[target_class_idx, high_freq_neurons] += noise
    else:
        raise ValueError(f"Unknown modification type: {modification_type}")
    
    model.heads.head.weight.data = fc_weights
    return model

def evaluate_model(model, eval_loader, unlearn_class):
    """
    [Modify 3] 在训练集上评估。
    分别统计 Unlearn Class 的准确率（越低越好）和其他类的准确率（越高越好）。
    """
    model.eval()
    unlearn_correct = 0
    unlearn_total = 0
    other_correct = 0
    other_total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(eval_loader, desc="Evaluating (Training Set)"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for true_label, pred_label in zip(labels, predicted):
                true_label = true_label.item()
                pred_label = pred_label.item()
                
                if true_label == unlearn_class:
                    unlearn_total += 1
                    if pred_label == true_label:
                        unlearn_correct += 1
                else:
                    other_total += 1
                    if pred_label == true_label:
                        other_correct += 1
    
    unlearn_acc = (unlearn_correct / unlearn_total * 100) if unlearn_total > 0 else 0.0
    other_acc = (other_correct / other_total * 100) if other_total > 0 else 0.0
    overall_acc = ((unlearn_correct + other_correct) / (unlearn_total + other_total) * 100) if (unlearn_total + other_total) > 0 else 0.0
    
    return {
        'unlearn_accuracy': unlearn_acc, # 目标：接近0%
        'other_accuracy': other_acc,     # 目标：接近原值
        'overall_accuracy': overall_acc,
        'unlearn_correct': unlearn_correct,
        'unlearn_total': unlearn_total,
        'other_correct': other_correct,
        'other_total': other_total
    }

# ==================== 实验运行函数 ====================
def generate_experiment_combinations(config):
    combinations = product(
        config['k_top_neurons_list'],
        config['h_high_freq_list'],
        config['modification_types'],
        config['image_noise_levels']
    )
    
    experiments = []
    for k, h, mod_type, noise in combinations:
        # 将配置扁平化放入每次实验
        experiments.append({
            'k_top_neurons': k,
            'h_high_freq': h,
            'modification_type': mod_type,
            'image_noise_level': noise,
            # 继承通用 LRP 配置
            'lrp_rule_type': config['lrp_rule_type'],
            'lrp_params': config['lrp_params']
        })
    return experiments

def run_single_experiment(exp_config, base_config, original_model, unlearn_loader, eval_loader):
    print("\n" + "="*80)
    print(f"Running Experiment:")
    print(f"  k_top: {exp_config['k_top_neurons']}, h_freq: {exp_config['h_high_freq']}")
    print(f"  Mod Type: {exp_config['modification_type']}, Rule: {exp_config['lrp_rule_type']}")
    print("="*80)
    
    start_time = time.time()
    model = copy.deepcopy(original_model)
    
    # Step 1: 分析待遗忘数据 (使用 Test Set)
    analysis_results = analyze_unlearn_dataset(
        model=model,
        unlearn_loader=unlearn_loader,
        exp_config=exp_config,
        base_config=base_config
    )
    
    neuron_frequency = analysis_results['neuron_frequency']
    high_freq_neurons = [idx for idx, freq in neuron_frequency.most_common(exp_config['h_high_freq'])]
    target_class_idx = base_config['unlearn_class']
    
    print(f"  Identified {len(high_freq_neurons)} neurons to modify for class {target_class_idx}.")
    
    # Step 2: 修改权重
    model = modify_classification_weights(
        model=model,
        high_freq_neurons=high_freq_neurons,
        target_class_idx=target_class_idx,
        modification_type=exp_config['modification_type'],
        noise_scale=base_config['noise_scale']
    )
    
    # Step 3: 在训练集上评估 (Zero-shot unlearning check)
    print("  Evaluating on Training Set (Zero-shot check)...")
    modified_results = evaluate_model(
        model=model,
        eval_loader=eval_loader,
        unlearn_class=base_config['unlearn_class']
    )
    
    exp_duration = time.time() - start_time
    
    print(f"  Unlearn Acc (Train Set): {modified_results['unlearn_accuracy']:.2f}% (Lower is better)")
    print(f"  Other Acc (Train Set): {modified_results['other_accuracy']:.2f}%")
    
    # [Modify 5] 保存模型
    if base_config['save_model']:
        save_dir = pathlib.Path(base_config['model_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"vit_unlearn_cls{target_class_idx}_k{exp_config['k_top_neurons']}_h{exp_config['h_high_freq']}_" \
                     f"vit_{exp_config['modification_type']}_{exp_config['lrp_rule_type']}.pth"
        model_save_path = save_dir / model_name
        torch.save(model, model_save_path)
        print(f"  Model saved to: {model_save_path}")
    
    return {
        'config': exp_config,
        'duration': exp_duration,
        'performance': modified_results,
        'high_freq_neurons': high_freq_neurons
    }

# ==================== 主函数 ====================
def main():
    config = EXPERIMENT_CONFIG
    set_seed()
    
    experiments = generate_experiment_combinations(config)
    print(f"Total experiments: {len(experiments)}")
    
    # Step 1: 加载模型
    print("\n[Init] Loading Model...")
    try:
        model = load_finetuned_vit(
            checkpoint_path=config['checkpoint_path'], 
            num_classes=config['num_classes']
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    original_model = copy.deepcopy(model)
    
    # Step 2: 准备数据 (使用 retrain_ViT 的路径逻辑)
    print("\n[Init] Loading Data...")
    base_dir = config['dataset_root']
    
    # 数据变换 (使用 ImageNet 标准变换或自定义)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    # [Modify 0 & 2] Unlearn Dataset = Test Set中仅包含 Target Class 的数据
    # 这是为了 Zero-shot 设置：我们假设没有 target class 的训练数据，只有测试数据
    unlearn_csv = f"{base_dir}/custom_test_dataset.csv"
    unlearn_img_dir = f"{base_dir}/test_images"
    
    unlearn_dataset = CustomDataset(
        meta_data_path=unlearn_csv,
        image_directory=unlearn_img_dir,
        transform=transform,
        keep_only_class=config['unlearn_class'] # 仅加载需要遗忘的那个类
    )
    # LRP分析通常逐个样本进行，batch_size=1
    unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, batch_size=1, shuffle=True)
    
    # [Modify 3] Eval Dataset = Training Set (全量)
    # 验证在原本训练过的数据上，记忆是否被擦除
    eval_csv = f"{base_dir}/custom_train_dataset.csv"
    eval_img_dir = f"{base_dir}/train_images"
    
    eval_dataset = CustomDataset(
        meta_data_path=eval_csv,
        image_directory=eval_img_dir,
        transform=transform,
        filter_class=None # 加载所有类别
    )
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    print(f"  Unlearn Analysis Data (Test Set, Class {config['unlearn_class']}): {len(unlearn_dataset)} samples")
    print(f"  Evaluation Data (Train Set, All Classes): {len(eval_dataset)} samples")
    
    # Step 3: 评估原始模型
    print("\n[Baseline] Evaluating original model on Training Set...")
    original_results = evaluate_model(original_model, eval_loader, config['unlearn_class'])
    print(f"  Original Unlearn Acc: {original_results['unlearn_accuracy']:.2f}%")
    print(f"  Original Other Acc: {original_results['other_accuracy']:.2f}%")
    
    # Step 4: 运行实验
    all_results = []
    total_time = 0

    for i, exp in enumerate(experiments, 1):
        res = run_single_experiment(exp, config, original_model, unlearn_loader, eval_loader)
        all_results.append(res)
        total_time += res['duration']

    print(f"\n✅ All experiments finished")
    print(f"⏱ Total time: {total_time:.2f} seconds")
        
    # Step 5: 保存结果
    if config['save_results']:
        save_dir = pathlib.Path(config['results_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"vit_unlearn_exp_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({'config': config, 'results': all_results}, f)
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()