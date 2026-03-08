import os
import sys
import copy
import time
import torch
import pickle
import random
import pathlib
import argparse
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
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit

# 猴子补丁（ViT适配）
monkey_patch(vision_transformer, verbose=False)
monkey_patch_zennit(verbose=False)

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    # 数据配置
    'unlearn_class': 56,  # 待遗忘的ImageNet类别索引
    'unlearn_data_path': 'torch_imagenet/imagenet-mini/train/n01734418',  # 待遗忘数据路径
    'test_data_path': '/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/val',  # 测试数据路径
    'batch_size': 1,  # 分析时batch_size（单样本处理）
    'test_batch_size': 16,  # 测试时batch_size
    
    # ========== 独立参数列表（批量实验） ==========
    'k_top_neurons_list': [150],  # 每样本分析的Top-k神经元数
    'h_high_freq_list': [150],  # 最终修改的高频神经元数
    'modification_types': ['zero'],  # 修改方式: 'zero'/'laplace'/'gaussian'
    'conv_gamma_list': [0.25],  # LRP Conv层Gamma参数
    'lin_gamma_list': [0.1],  # LRP Linear层Gamma参数
    'image_noise_levels': [0.0],  # 图像噪声强度（标准差，0=不加噪）
    
    # ========== 噪声参数 ==========
    'noise_scale': 0.1,  # 高斯/拉普拉斯噪声强度
    
    # ========== 保存配置 ==========
    'save_results': False,
    'results_save_dir': 'vit_unlearn_results/',
    'save_model': False,
    'model_save_dir': 'vit_unlearn_models/',
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
    """对图像添加高斯噪声（模拟分布偏移）"""
    if noise_level == 0.0:
        return images
    noise = torch.randn_like(images, device=device) * noise_level
    return torch.clamp(images + noise, 0.0, 1.0)  # 确保像素值在有效范围

def load_vit_model(device=device):
    """加载预训练ViT模型（ImageNet权重）"""
    weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = vision_transformer.vit_b_16(weights=weights)
    model.eval()
    model.to(device)
    
    # 冻结参数（仅用于推理和LRP分析）
    for param in model.parameters():
        param.requires_grad = False
    
    return model, weights

# ==================== 数据集类 ====================
class ImageNetDataset(torch.utils.data.Dataset):
    """自定义ImageNet数据集（加载单类别文件夹）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = self._collect_image_paths()
    
    def _collect_image_paths(self):
        """收集所有图像路径"""
        extensions = ['*.JPEG', '*.jpg', '*.png', '*.jpeg']
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(self.root_dir.glob(ext)))
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)

# ==================== LRP相关类 ====================
class RelevanceHook:
    """捕获分类头激活和梯度的Hook"""
    def __init__(self):
        self.activations = None
        self.gradients = None
    
    def forward_hook(self, module, input, output):
        self.activations = input[0].detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_input[0].detach()
    
    def get_relevance(self):
        """计算相关性：激活 * 梯度"""
        if self.activations is not None and self.gradients is not None:
            return self.activations * self.gradients
        return None

# ==================== 核心功能函数 ====================
def analyze_single_image(model, input_tensor, conv_gamma=0.25, lin_gamma=0.1):
    """单图像LRP分析，返回预测类别和768维神经元相关性"""
    input_tensor.grad = None
    relevance_hook = RelevanceHook()
    
    # 注册Hook
    forward_handle = model.heads.head.register_forward_hook(relevance_hook.forward_hook)
    backward_handle = model.heads.head.register_full_backward_hook(relevance_hook.backward_hook)
    
    # LRP复合规则
    zennit_comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    ])
    zennit_comp.register(model)
    
    # 前向传播
    y = model(input_tensor.requires_grad_())
    predicted_class = torch.argmax(y, dim=1).item()
    
    # 反向传播（针对预测类别）
    y[0, predicted_class].backward()
    
    # 获取相关性
    relevance = relevance_hook.get_relevance()
    
    # 清理资源
    zennit_comp.remove()
    forward_handle.remove()
    backward_handle.remove()
    
    if relevance is not None:
        return predicted_class, relevance[0].cpu().numpy()
    return predicted_class, None

def find_top_k_neurons(relevance_768d, k=10):
    """获取Top-k相关性最高的神经元索引"""
    abs_relevance = np.abs(relevance_768d)
    top_k_indices = np.argsort(abs_relevance)[-k:][::-1]
    return top_k_indices

def analyze_unlearn_dataset(model, unlearn_loader, k_top_neurons, conv_gamma, lin_gamma, noise_level):
    """分析整个待遗忘数据集，统计高频神经元"""
    all_top_neurons = []
    sample_predictions = []
    
    for input_tensor, img_path in tqdm(unlearn_loader, desc="Analyzing unlearn dataset"):
        try:
            input_tensor = input_tensor.to(device)
            # 添加图像噪声
            input_tensor = add_image_noise(input_tensor, noise_level)
            
            predicted_class, relevance_768d = analyze_single_image(
                model, input_tensor, conv_gamma, lin_gamma
            )
            
            if relevance_768d is not None:
                top_k_indices = find_top_k_neurons(relevance_768d, k_top_neurons)
                all_top_neurons.append(top_k_indices)
                sample_predictions.append(predicted_class)
        
        except Exception as e:
            print(f"Warning: Error processing image {img_path}: {str(e)}")
            continue
    
    # 统计神经元出现频率
    neuron_frequency = Counter([idx for sublist in all_top_neurons for idx in sublist])
    # 确定目标类别（最常见的预测类别）
    target_class_idx = Counter(sample_predictions).most_common(1)[0][0] if sample_predictions else 0
    
    return {
        'neuron_frequency': neuron_frequency,
        'target_class_idx': target_class_idx,
        'all_top_neurons': all_top_neurons,
        'sample_predictions': sample_predictions
    }

def modify_classification_weights(model, high_freq_neurons, target_class_idx, modification_type, noise_scale):
    """修改分类头权重（针对高频神经元）"""
    fc_weights = model.heads.head.weight.data  # [1000, 768]
    
    # 保存原始权重用于验证
    original_weights = fc_weights[target_class_idx, high_freq_neurons].clone()
    
    # 应用修改
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
    
    # 更新权重
    model.heads.head.weight.data = fc_weights
    return model

def evaluate_model(model, test_loader, unlearn_class):
    """评估模型性能（区分待遗忘类和其他类）"""
    model.eval()
    unlearn_correct = 0
    unlearn_total = 0
    other_correct = 0
    other_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating model"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # 统计两类准确率
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
    
    # 计算准确率
    unlearn_acc = (unlearn_correct / unlearn_total * 100) if unlearn_total > 0 else 0.0
    other_acc = (other_correct / other_total * 100) if other_total > 0 else 0.0
    overall_acc = ((unlearn_correct + other_correct) / (unlearn_total + other_total) * 100) if (unlearn_total + other_total) > 0 else 0.0
    
    return {
        'unlearn_accuracy': unlearn_acc,
        'other_accuracy': other_acc,
        'overall_accuracy': overall_acc,
        'unlearn_correct': unlearn_correct,
        'unlearn_total': unlearn_total,
        'other_correct': other_correct,
        'other_total': other_total
    }

# ==================== 实验运行函数 ====================
def generate_experiment_combinations(config):
    """生成所有实验组合（参数笛卡尔积）"""
    combinations = product(
        config['k_top_neurons_list'],
        config['h_high_freq_list'],
        config['modification_types'],
        config['conv_gamma_list'],
        config['lin_gamma_list'],
        config['image_noise_levels']
    )
    
    experiments = []
    for k, h, mod_type, conv_g, lin_g, noise in combinations:
        experiments.append({
            'k_top_neurons': k,
            'h_high_freq': h,
            'modification_type': mod_type,
            'conv_gamma': conv_g,
            'lin_gamma': lin_g,
            'image_noise_level': noise
        })
    return experiments

def run_single_experiment(exp_config, base_config, original_model, unlearn_loader, test_loader, weights):
    """运行单个实验"""
    print("\n" + "="*80)
    print(f"Experiment Configuration:")
    print(f"  k_top_neurons: {exp_config['k_top_neurons']}")
    print(f"  h_high_freq: {exp_config['h_high_freq']}")
    print(f"  modification_type: {exp_config['modification_type']}")
    print(f"  conv_gamma: {exp_config['conv_gamma']}")
    print(f"  lin_gamma: {exp_config['lin_gamma']}")
    print(f"  image_noise_level: {exp_config['image_noise_level']}")
    print("="*80)
    
    start_time = time.time()
    
    # 复制模型（避免污染原始模型）
    model = copy.deepcopy(original_model)
    
    # Step 1: 分析待遗忘数据集
    print("\n[Step 1/3] Analyzing unlearn dataset...")
    analysis_results = analyze_unlearn_dataset(
        model=model,
        unlearn_loader=unlearn_loader,
        k_top_neurons=exp_config['k_top_neurons'],
        conv_gamma=exp_config['conv_gamma'],
        lin_gamma=exp_config['lin_gamma'],
        noise_level=exp_config['image_noise_level']
    )
    
    # 获取高频神经元
    neuron_frequency = analysis_results['neuron_frequency']
    high_freq_neurons = [idx for idx, freq in neuron_frequency.most_common(exp_config['h_high_freq'])]
    target_class_idx = analysis_results['target_class_idx']
    print(f"  Target class for modification: {target_class_idx} ({weights.meta['categories'][target_class_idx]})")
    print(f"  Top {exp_config['h_high_freq']} high-frequency neurons identified")
    
    # Step 2: 修改分类头权重
    print("\n[Step 2/3] Modifying classification head weights...")
    model = modify_classification_weights(
        model=model,
        high_freq_neurons=high_freq_neurons,
        target_class_idx=target_class_idx,
        modification_type=exp_config['modification_type'],
        noise_scale=base_config['noise_scale']
    )
    
    # Step 3: 评估修改后模型
    print("\n[Step 3/3] Evaluating modified model...")
    modified_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        unlearn_class=base_config['unlearn_class']
    )
    
    # 计算实验耗时
    exp_duration = time.time() - start_time
    
    # 打印实验结果
    print("\nExperiment Results:")
    print(f"  Duration: {exp_duration:.2f}s")
    print(f"  Unlearn Class Accuracy: {modified_results['unlearn_accuracy']:.2f}%")
    print(f"  Other Classes Accuracy: {modified_results['other_accuracy']:.2f}%")
    print(f"  Overall Accuracy: {modified_results['overall_accuracy']:.2f}%")
    
    # 保存模型（如果启用）
    if base_config['save_model']:
        model_save_dir = pathlib.Path(base_config['model_save_dir'])
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"vit_unlearn_k{exp_config['k_top_neurons']}_h{exp_config['h_high_freq']}_" \
                     f"{exp_config['modification_type']}_noise{exp_config['image_noise_level']}.pth"
        model_save_path = model_save_dir / model_name
        torch.save(model.state_dict(), model_save_path)
        print(f"  Model saved to: {model_save_path}")
    
    # 整理实验结果
    experiment_result = {
        'config': exp_config,
        'duration': exp_duration,
        'performance': modified_results,
        'high_freq_neurons': high_freq_neurons,
        'target_class_idx': target_class_idx
    }
    
    return experiment_result

# ==================== 主函数 ====================
def main():
    """主函数：加载数据、初始化模型、批量运行实验"""
    config = EXPERIMENT_CONFIG
    set_seed()
    
    # 生成所有实验组合
    experiments = generate_experiment_combinations(config)
    print(f"Total experiments to run: {len(experiments)}")
    print("\nExperiment list:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    
    # Step 1: 加载模型和数据变换
    print("\n" + "="*80)
    print("Loading model and data...")
    model, weights = load_vit_model(device=device)
    original_model = copy.deepcopy(model)  # 保存原始模型用于对比
    transform = weights.transforms()
    
    # Step 2: 加载数据集
    # 待遗忘数据集（单类别）
    unlearn_dataset = ImageNetDataset(
        root_dir=config['unlearn_data_path'],
        transform=transform
    )
    unlearn_loader = torch.utils.data.DataLoader(
        unlearn_dataset, batch_size=config['batch_size'], shuffle=True
    )
    
    # 测试数据集（全类别）
    test_dataset = datasets.ImageFolder(
        root=config['test_data_path'],
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['test_batch_size'], shuffle=False
    )
    
    print(f"Dataset Statistics:")
    print(f"  Unlearn dataset size: {len(unlearn_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    print(f"  Unlearn class index: {config['unlearn_class']}")
    print(f"  Unlearn class name: {weights.meta['categories'][config['unlearn_class']]}")
    
    # Step 3: 评估原始模型
    print("\n" + "="*80)
    print("Evaluating original model...")
    original_results = evaluate_model(
        model=original_model,
        test_loader=test_loader,
        unlearn_class=config['unlearn_class']
    )
    print(f"Original Model Performance:")
    print(f"  Unlearn Class Accuracy: {original_results['unlearn_accuracy']:.2f}%")
    print(f"  Other Classes Accuracy: {original_results['other_accuracy']:.2f}%")
    print(f"  Overall Accuracy: {original_results['overall_accuracy']:.2f}%")
    
    # Step 4: 批量运行实验
    print("\n" + "="*80)
    print("Starting experiments...")
    all_results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# Running Experiment {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        exp_result = run_single_experiment(
            exp_config=exp,
            base_config=config,
            original_model=original_model,
            unlearn_loader=unlearn_loader,
            test_loader=test_loader,
            weights=weights
        )
        all_results.append(exp_result)
    
    # Step 5: 保存所有实验结果
    if config['save_results']:
        results_save_dir = pathlib.Path(config['results_save_dir'])
        results_save_dir.mkdir(parents=True, exist_ok=True)
        results_save_path = results_save_dir / f"vit_unlearn_experiments_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # 整理完整结果（包含原始模型性能）
        final_results = {
            'original_performance': original_results,
            'experiment_config': config,
            'experiment_results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_save_path, 'wb') as f:
            pickle.dump(final_results, f)
        print(f"\nAll results saved to: {results_save_path}")
    
    # Step 6: 打印汇总报告
    print("\n" + "="*80)
    print("Experiment Summary Report")
    print("="*80)
    print(f"{'Exp ID':<6} {'k':<4} {'h':<4} {'Mod Type':<10} {'Noise':<6} {'Unlearn Acc':<12} {'Other Acc':<12} {'Overall Acc':<12}")
    print("-"*80)
    
    for i, result in enumerate(all_results, 1):
        exp_cfg = result['config']
        perf = result['performance']
        print(f"{i:<6} {exp_cfg['k_top_neurons']:<4} {exp_cfg['h_high_freq']:<4} "
              f"{exp_cfg['modification_type']:<10} {exp_cfg['image_noise_level']:<6} "
              f"{perf['unlearn_accuracy']:<12.2f} {perf['other_accuracy']:<12.2f} {perf['overall_accuracy']:<12.2f}")
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")

if __name__ == "__main__":
    main()