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
from scipy import stats  # 用于统计分析

# [新增] Sklearn 依赖用于 MIA
from sklearn import linear_model, model_selection

# ==================== 路径配置 ====================
base_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, base_path.as_posix())

# 导入依赖库 (Zennit / LXT)
import zennit
from zennit.composites import LayerMapComposite, EpsilonGammaBox, EpsilonPlusFlat
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit

# 猴子补丁
monkey_patch(vision_transformer, verbose=False)
monkey_patch_zennit(verbose=False)

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    # ========== 固定参数 ==========
    'unlearn_class': 0,      # 待遗忘类别
    'num_classes': 8,
    'dataset_root': '/home/cwh/Workspace/TorchLRP-master/data/custom_korean_family_dataset_resolution_128',
    'checkpoint_path': '/home/cwh/Workspace/TorchLRP-master/examples/models/vit_best_on_test.pth',
    
    # 基础 LRP 配置
    'lrp_rule_type': 'alpha_beta',
    'lrp_params': {
        'gamma': 0.25,       # 用于 Gamma
        'epsilon': 1e-6,     # 用于 Epsilon
        'alpha': 1.0,        # 用于 AlphaBeta
        'beta': 0.0,
        'low': -3.0,         # 用于 Box 规则
        'high': 3.0
    },
    'modification_type': 'zero', # 固定修改方式
    'noise_scale': 0.1,          # 修改权重时的噪声

    # ========== 变量列表 (用于笛卡尔积组合实验) ==========
    'max_unlearn_samples_list': [36], 
    'image_noise_levels_list': [0.0], 
    'k_top_neurons_list': [400], 
    'h_high_freq_list': [600],

    # ========== 统计配置 ==========
    'n_repeats': 5,          # 每个配置重复运行次数

    # ========== 保存配置 ==========
    'save_excel': False,
    'results_save_dir': '/home/cwh/Workspace/TorchLRP-master',
    'save_model': False,
    'model_save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia',
}

# ==================== 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 统计工具函数 ====================
def calculate_cliff_delta(lst1, baseline_val):
    """计算单样本 Cliff's Delta (相对于基准常数)"""
    more = sum(x > baseline_val for x in lst1)
    less = sum(x < baseline_val for x in lst1)
    return (more - less) / len(lst1)

def calculate_statistics(current_values, baseline_value):
    """
    计算一组实验结果的统计指标
    Args:
        current_values: list of floats (5次运行的结果)
        baseline_value: float (基准值)
    Returns:
        dict: 包含 mean, std, p_value, cohens_d, cliffs_delta
    """
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) # 样本标准差
    
    # 1. P-Value (One-sample t-test)
    # if std < 1e-9:
    #     p_val = 0.0 if abs(mean - baseline_value) > 1e-5 else 1.0
    #     t_stat = 0.0
    # else:
    t_stat, p_val = stats.ttest_1samp(values, baseline_value)
    
    # 2. Cohen's d (One-sample)
    cohens_d = (mean - baseline_value) / std if std > 1e-9 else 0.0
    
    # 3. Cliff's delta
    cliffs_d = calculate_cliff_delta(values, baseline_value)
    
    return {
        "mean": mean,
        "std": std,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "cliffs_delta": cliffs_d
    }

# ==================== MIA (Membership Inference Attack) 核心逻辑 ====================
def compute_losses(net, loader, device, target_class=None):
    """
    计算 Loss。
    如果指定了 target_class，则只记录该类别的 Loss (用于从混合的 eval_loader 中提取 Member)。
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    net.eval()
    with torch.no_grad():
        for inputs, labels, _ in loader: # 适配 Dataset 返回 (img, label, path)
            inputs, labels = inputs.to(device), labels.to(device)

            # 如果指定了类别，进行过滤
            if target_class is not None:
                mask = (labels == target_class)
                if mask.sum() == 0:
                    continue
                inputs = inputs[mask]
                labels = labels[mask]

            if len(inputs) == 0:
                continue

            logits = net(inputs)
            losses = criterion(logits, labels).cpu().detach().numpy()
            for l in losses:
                all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=42):
    """
    使用逻辑回归进行 MIA 攻击。
    sample_loss: 特征 (Loss值)
    members: 标签 (0 或 1)
    """
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        # 如果样本极度不平衡导致只有一类，直接返回默认 Acc 0.5
        return np.array([0.5])

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def calculate_mia_score(model, train_loader, test_loader, target_class, device):
    """
    计算 MIA Accuracy 和 Forgetting Score。
    
    Member (1): 训练集中的 target_class (来自 train_loader/eval_loader)
    Non-Member (0): 测试集中的 target_class (来自 test_loader/unlearn_loader)
    """
    # 1. 提取 Member Loss (训练集里的 Class 0)
    member_losses = compute_losses(model, train_loader, device, target_class=target_class)
    
    # 2. 提取 Non-Member Loss (测试集里的 Class 0)
    non_member_losses = compute_losses(model, test_loader, device, target_class=target_class)

    # 异常保护：如果没有数据
    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return 0.5, 0.0 

    # 3. 数据平衡 (截断到相同长度，保证 1:1)
    # 打乱 member losses 以确保随机截断
    np.random.shuffle(member_losses)
    # non_member_losses 本身如果已经是测试集全集，可以不打乱，但为了严谨也打乱
    np.random.shuffle(non_member_losses)
    
    min_len = min(len(member_losses), len(non_member_losses))
    if min_len < 2: # 样本太少无法做 CV
        return 0.5, 0.0
        
    member_losses = member_losses[:min_len]
    non_member_losses = non_member_losses[:min_len]

    # 4. 构建攻击数据集
    # Label 0: Non-Member, Label 1: Member
    samples = np.concatenate((non_member_losses, member_losses)).reshape((-1, 1))
    labels = [0] * len(non_member_losses) + [1] * len(member_losses)

    # 5. 计算 MIA Accuracy
    mia_scores = simple_mia(samples, labels)
    mia_acc = mia_scores.mean()
    
    # 6. 计算 Forget Score (理想 Acc=0.5 -> Score=0)
    forget_score = abs(0.5 - mia_acc)

    return mia_acc, forget_score

# ==================== 基础工具函数 ====================
def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_image_noise(images, noise_level):
    if noise_level <= 0.0:
        return images
    noise = torch.randn_like(images, device=device) * noise_level
    return torch.clamp(images + noise, 0.0, 1.0)

# ==================== 数据集类 ====================
def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, meta_data_path, image_directory, transform=None, filter_class=None, keep_only_class=None):
        self.meta_data = pd.read_csv(meta_data_path)
        self.image_directory = image_directory
        self.transform = transform
        
        full_list = parsing(self.meta_data)
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }
        
        self.image_age_list = []
        for img_path, age_cls in full_list:
            label = self.age_class_to_label[age_cls]
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
        return img, label, str(full_path)

# ==================== 模型加载 ====================
def load_finetuned_vit(checkpoint_path, num_classes=8, device=device):
    print(f"Loading model from: {checkpoint_path}")
    model = vision_transformer.vit_b_16(weights=None) 
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

# ==================== LRP 逻辑 ====================
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

def get_zennit_composite(model, rule_type, params):
    if rule_type == 'gamma':
        gamma_val = params.get('gamma', 0.25)
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(gamma_val)),
            (torch.nn.Linear, z_rules.Gamma(gamma_val)),
        ])
    elif rule_type == 'epsilon':
        epsilon_val = params.get('epsilon', 1e-6)
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Epsilon(epsilon_val)),
            (torch.nn.Linear, z_rules.Epsilon(epsilon_val)),
        ])
    elif rule_type == 'alpha_beta':
        alpha = params.get('alpha', 2.0)
        beta = params.get('beta', 1.0)
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.AlphaBeta(alpha=alpha, beta=beta)),
            (torch.nn.Linear, z_rules.AlphaBeta(alpha=alpha, beta=beta)),
        ])
    elif rule_type == 'epsilon_gamma_box':
        low = params.get('low', -3.0)
        high = params.get('high', 3.0)
        return EpsilonGammaBox(low=low, high=high)
    else:
        raise ValueError(f"Unknown LRP rule type: {rule_type}")

def analyze_single_image(model, input_tensor, rule_type, lrp_params):
    input_tensor.grad = None
    relevance_hook = RelevanceHook()
    
    forward_handle = model.heads.head.register_forward_hook(relevance_hook.forward_hook)
    backward_handle = model.heads.head.register_full_backward_hook(relevance_hook.backward_hook)
    
    zennit_comp = get_zennit_composite(model, rule_type, lrp_params)
    zennit_comp.register(model)
    
    try:
        y = model(input_tensor.requires_grad_())
        predicted_class = torch.argmax(y, dim=1).item()
        y[0, predicted_class].backward()
        relevance = relevance_hook.get_relevance()
    finally:
        zennit_comp.remove()
        forward_handle.remove()
        backward_handle.remove()
    
    if relevance is not None:
        return predicted_class, relevance[0].cpu().numpy()
    return predicted_class, None

def analyze_unlearn_dataset(model, unlearn_loader, max_samples, noise_level, k_top, rule_type, lrp_params, target_class):
    all_top_neurons = []
    count = 0
    
    for input_tensor, label, img_path in unlearn_loader:
        if max_samples is not None and count >= max_samples:
            break
            
        try:
            input_tensor = input_tensor.to(device)
            input_tensor = add_image_noise(input_tensor, noise_level)
            
            if input_tensor.shape[0] > 1:
                input_tensor = input_tensor[0:1]

            predicted_class, relevance_768d = analyze_single_image(model, input_tensor, rule_type, lrp_params)
            
            if relevance_768d is not None:
                top_k_indices = np.argsort(np.abs(relevance_768d))[-k_top:][::-1]
                all_top_neurons.append(top_k_indices)
                count += 1
        except Exception as e:
            continue
            
    neuron_frequency = Counter([idx for sublist in all_top_neurons for idx in sublist])
    return neuron_frequency

def modify_weights(model, high_freq_neurons, target_class_idx, mod_type, noise_scale):
    fc_weights = model.heads.head.weight.data
    if mod_type == 'zero':
        fc_weights[target_class_idx, high_freq_neurons] = 0.0
    model.heads.head.weight.data = fc_weights
    return model

def evaluate_model(model, eval_loader, unlearn_class):
    model.eval()
    unlearn_correct = 0; unlearn_total = 0
    other_correct = 0; other_total = 0
    
    with torch.no_grad():
        for images, labels, _ in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for true_label, pred_label in zip(labels, predicted):
                true_label = true_label.item()
                pred_label = pred_label.item()
                if true_label == unlearn_class:
                    unlearn_total += 1
                    if pred_label == true_label: unlearn_correct += 1
                else:
                    other_total += 1
                    if pred_label == true_label: other_correct += 1
    
    unlearn_acc = (unlearn_correct / unlearn_total * 100) if unlearn_total > 0 else 0.0
    other_acc = (other_correct / other_total * 100) if other_total > 0 else 0.0
    return unlearn_acc, other_acc

# ==================== 实验流程控制 ====================
def run_full_experiment_suite(config):
    # 1. 准备全局资源
    print("[Init] Loading Baseline Model...")
    base_model = load_finetuned_vit(config['checkpoint_path'], config['num_classes'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    unlearn_csv = os.path.join(config['dataset_root'], "custom_test_dataset.csv")
    unlearn_img_dir = os.path.join(config['dataset_root'], "test_images")
    eval_csv = os.path.join(config['dataset_root'], "custom_train_dataset.csv")
    eval_img_dir = os.path.join(config['dataset_root'], "train_images")
    
    # Dataset 初始化
    # 1. unlearn_dataset: 用于分析 LRP 的数据
    unlearn_dataset = CustomDataset(unlearn_csv, unlearn_img_dir, transform, keep_only_class=config['unlearn_class'])
    
    # 2. eval_dataset / eval_loader: 包含所有类别的训练集 (用于 Acc评估 和 MIA Member 数据)
    eval_dataset = CustomDataset(eval_csv, eval_img_dir, transform)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    # 3. test_dataset_full / test_loader_full: 仅包含遗忘类别的测试集 (用于 MIA Non-Member 数据)
    # 这就是 Unseen Data (模型在训练集见过 Class 0，但没见过测试集的 Class 0)
    test_dataset_full = CustomDataset(unlearn_csv, unlearn_img_dir, transform, keep_only_class=config['unlearn_class'])
    test_loader_full = torch.utils.data.DataLoader(test_dataset_full, batch_size=64, shuffle=False)
    
    print(f"[Init] Evaluating Baseline (Original Model)...")
    orig_unlearn_acc, orig_other_acc = evaluate_model(base_model, eval_loader, config['unlearn_class'])
    print(f" >> Baseline Unlearn Acc: {orig_unlearn_acc:.2f}%")
    print(f" >> Baseline Other Acc:   {orig_other_acc:.2f}%")
    
    # 2. 生成实验组合
    combinations = list(product(
        config['max_unlearn_samples_list'],
        config['image_noise_levels_list'],
        config['k_top_neurons_list'],
        config['h_high_freq_list']
    ))
    
    print(f"\n[Info] Total Configurations: {len(combinations)}")
    print(f"[Info] Runs per Config: {config['n_repeats']}")
    
    final_records = []
    
    # 3. 循环实验
    for (n_samples, noise, k, h) in combinations:
        print(f"\n>>> Config: Samples={n_samples}, Noise={noise}, K={k}, H={h}")
        
        runs_unlearn_acc = []
        runs_other_acc = []
        runs_mia_acc = []     # 存储 MIA 攻击成功率
        runs_forget_score = [] # 存储 Forget Score
        
        for run_id in range(1, config['n_repeats'] + 1):
            current_seed = 42 + run_id * 100
            set_seed(current_seed)
            
            current_unlearn_loader = torch.utils.data.DataLoader(
                unlearn_dataset, batch_size=1, shuffle=True
            )
            
            model_copy = copy.deepcopy(base_model)
            
            # Step A: 分析 (LRP)
            neuron_freq = analyze_unlearn_dataset(
                model=model_copy,
                unlearn_loader=current_unlearn_loader,
                max_samples=n_samples,
                noise_level=noise,
                k_top=k,
                rule_type=config['lrp_rule_type'],
                lrp_params=config['lrp_params'],
                target_class=config['unlearn_class']
            )
            
            # Step B: 筛选与修改
            high_freq_neurons = [idx for idx, _ in neuron_freq.most_common(h)]
            model_copy = modify_weights(
                model=model_copy,
                high_freq_neurons=high_freq_neurons,
                target_class_idx=config['unlearn_class'],
                mod_type=config['modification_type'],
                noise_scale=config['noise_scale']
            )
            
            # Step C: 评估 Accuracy
            ua, oa = evaluate_model(model_copy, eval_loader, config['unlearn_class'])
            runs_unlearn_acc.append(ua)
            runs_other_acc.append(oa)
            
            # Step D: 评估 MIA & Forget Score
            # Member: Train Set (Class 0), Non-Member: Test Set (Class 0)
            mia_acc, f_score = calculate_mia_score(
                model=model_copy,
                train_loader=eval_loader,     # 从中提取 Member
                test_loader=test_loader_full, # 全部都是 Non-Member
                target_class=config['unlearn_class'],
                device=device
            )
            runs_mia_acc.append(mia_acc)
            runs_forget_score.append(f_score)
            
            print(f"   [Run {run_id}] UA: {ua:.2f} | OA: {oa:.2f} | MIA: {mia_acc:.3f} | FS: {f_score:.3f}")
            
            if config['save_model']:
                save_dir = Path(config['model_save_dir'])
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = f"vit_S{n_samples}_N{noise}_K{k}_H{h}_run{run_id}.pth"
                torch.save(model_copy.state_dict(), save_dir / fname)

        # 4. 统计分析
        stats_ua = calculate_statistics(runs_unlearn_acc, orig_unlearn_acc)
        stats_oa = calculate_statistics(runs_other_acc, orig_other_acc)
        
        # MIA Acc 期望基准是 0.5 (随机猜测)
        stats_mia = calculate_statistics(runs_mia_acc, 0.5)
        # Forget Score 期望基准是 0.0 (完美遗忘)
        stats_fs = calculate_statistics(runs_forget_score, 0.0)
        
        # 5. 记录数据
        record = {
            'Samples': n_samples,
            'Noise': noise,
            'K_Top': k,
            'H_High': h,
            
            # Accuracy Stats
            'UA_Mean': stats_ua['mean'],
            'UA_Std': stats_ua['std'],
            'UA_P_Val': stats_ua['p_value'],
            'UA_Cohens_D': stats_ua['cohens_d'],
            'UA_Cliff_D': stats_ua['cliffs_delta'],
            
            'OA_Mean': stats_oa['mean'],
            'OA_Std': stats_oa['std'],
            'OA_P_Val': stats_oa['p_value'],
            'OA_Cohens_D': stats_oa['cohens_d'],
            'OA_Cliff_D': stats_oa['cliffs_delta'],
            
            # MIA Stats
            'MIA_Mean': stats_mia['mean'],
            'MIA_Std': stats_mia['std'],
            'MIA_P_Val': stats_mia['p_value'],
            'MIA_Cohens_D': stats_mia['cohens_d'],
            'MIA_Cliff_D': stats_mia['cliffs_delta'],
            
            # Forget Score Stats
            'FS_Mean': stats_fs['mean'],
            'FS_Std': stats_fs['std'],
            'FS_P_Val': stats_fs['p_value'],
            'FS_Cohens_D': stats_fs['cohens_d'],
            'FS_Cliff_D': stats_fs['cliffs_delta'],
            
            # Raw Data
            'Raw_UA_Runs': str(runs_unlearn_acc),
            'Raw_OA_Runs': str(runs_other_acc),
            'Raw_MIA_Runs': str(runs_mia_acc),
            'Raw_FS_Runs': str(runs_forget_score),
        }
        final_records.append(record)

    # 6. 导出到 Excel
    if config['save_excel']:
        df = pd.DataFrame(final_records)
        save_dir = Path(config['results_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        excel_path = save_dir / f"org_MUFAC_report_alpha1beta0_{timestamp}.xlsx"
        
        # 设定列顺序
        cols_order = [
            'Samples', 'Noise', 'K_Top', 'H_High', 
            # UA
            'UA_Mean', 'UA_Std', 'UA_P_Val', 'UA_Cohens_D', 'UA_Cliff_D',
            # OA
            'OA_Mean', 'OA_Std', 'OA_P_Val', 'OA_Cohens_D', 'OA_Cliff_D',
            # MIA
            'MIA_Mean', 'MIA_Std', 'MIA_P_Val', 'MIA_Cohens_D', 'MIA_Cliff_D',
            # FS
            'FS_Mean', 'FS_Std', 'FS_P_Val', 'FS_Cohens_D', 'FS_Cliff_D',
            # Raw
            'Raw_UA_Runs'
        ]
        
        final_cols = [c for c in cols_order if c in df.columns] + \
                     [c for c in df.columns if c not in cols_order]
        df = df[final_cols]
        
        df.to_excel(excel_path, index=False)
        print(f"\n[Success] Report saved to: {excel_path}")

if __name__ == "__main__":
    run_full_experiment_suite(EXPERIMENT_CONFIG)