import os
import sys
import copy
import time
import torch
import torch.nn as nn  # 需要 nn.CrossEntropyLoss
import pickle
import random
import pathlib
import torchvision
import pandas as pd
import numpy as np
from collections import Counter
from torchvision import datasets, transforms as T
import configparser
import itertools
from scipy import stats  # 用于统计分析
from sklearn import linear_model, model_selection  # 用于 MIA

# ==================== 路径配置 ====================
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())
sys.path.insert(0, "/home/cwh/Workspace/TorchLRP-master")

import lrp
from lrp.patterns import fit_patternnet_positive
from utils import store_patterns, load_patterns

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    # ========== 固定参数 ==========
    'unlearn_class': 56,
    'batch_size': 1,
    'test_batch_size': 64, # 增大以便加速评估
    'num_classes': 1000,
    
    # ImageNet 数据路径
    'unlearn_data_dir': '/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train/n01734418',
    'test_data_dir': '/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train',
    
    # VGG 模型配置
    'vgg_version': 16,
    'use_pretrained': True,

    # VGG 层索引配置
    'fc1_layer_idx': 36,
    'fc2_layer_idx': 39,
    'output_layer_idx': 42,
    'perturb_fc1': True,
    'perturb_fc2': True,
    'perturb_output': False,
    'gaussian_std': 1.0,
    'laplace_scale': 1.0,

    # Pattern 配置
    'use_patterns': False,
    'patterns_dir': 'examples/patterns/',

    # ========== 变量列表 (用于笛卡尔积组合实验) ==========
    # 1. LRP传播规则
    'rules': ['epsilon'], # 示例：减少一些以加快演示
    # 2. 分析神经元数量 (K)
    'analyze_top_n_list': [150],
    # 3. 扰动神经元数量 (H)
    'perturb_top_n_list': [400],
    # 4. 扰动方法
    'perturbation_methods': ['zero'],
    # 5. 分析样本数
    'analysis_sample_sizes': [33],
    # 6. 图像噪声
    'image_noise_levels': [0.0],

    # ========== 统计配置 ==========
    'n_repeats': 5,          # 每个配置重复运行次数
    'save_excel': True,
    'results_save_dir': '/home/cwh/Workspace/TorchLRP-master',
    
    # ========== 模型保存 ==========
    'save_neurons': False,   # 批量实验建议关闭，否则文件太多
    'neuron_save_dir': 'neuron/imagenet/',
    'save_model': True,     # 批量实验建议关闭
    'model_save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia',
}

# ==================== 设备配置 ====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
try:
    from torchvision.models.vgg import VGG
    torch.serialization.add_safe_globals([VGG])
except:
    pass

_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

# ==================== 统计与 MIA 工具函数 (移植自参考代码) ====================

def set_seed(seed):
    """设置随机种子确保实验可复现"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_cliff_delta(lst1, baseline_val):
    """计算 Cliff's Delta"""
    more = sum(x > baseline_val for x in lst1)
    less = sum(x < baseline_val for x in lst1)
    return (more - less) / len(lst1)

def calculate_statistics(current_values, baseline_value):
    """计算统计指标: Mean, Std, P-value, Cohen's d, Cliff's delta"""
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # P-Value
    # if std < 1e-9:
    #     p_val = 0.0 if abs(mean - baseline_value) > 1e-5 else 1.0
    # else:
    t_stat, p_val = stats.ttest_1samp(values, baseline_value)
    
    # Cohen's d
    cohens_d = (mean - baseline_value) / std if std > 1e-9 else 0.0
    
    # Cliff's delta
    cliffs_d = calculate_cliff_delta(values, baseline_value)
    
    return {
        "mean": mean, "std": std, "p_value": p_val,
        "cohens_d": cohens_d, "cliffs_delta": cliffs_d
    }

def compute_losses(net, loader, device, target_class=None):
    """MIA 辅助: 计算 Loss 分布"""
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    net.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 过滤类别
            if target_class is not None:
                mask = (labels == target_class)
                if mask.sum() == 0: continue
                inputs = inputs[mask]
                labels = labels[mask]

            if len(inputs) == 0: continue

            logits = net(inputs)
            losses = criterion(logits, labels).cpu().detach().numpy()
            all_losses.extend(losses)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=42):
    """逻辑回归 MIA"""
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        return np.array([0.5])

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def calculate_mia_score(model, member_loader, non_member_loader, target_class, device):
    """计算 MIA Accuracy 和 Forgetting Score"""
    # 1. Member: 训练集中的 Target Class (从 unlearn_loader 获取)
    member_losses = compute_losses(model, member_loader, device, target_class=None) # loader本身就是target class
    
    # 2. Non-Member: 测试集中的 Target Class (从 test_loader 筛选)
    non_member_losses = compute_losses(model, non_member_loader, device, target_class=target_class)

    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return 0.5, 0.0

    # 3. 平衡采样
    np.random.shuffle(member_losses)
    np.random.shuffle(non_member_losses)
    min_len = min(len(member_losses), len(non_member_losses))
    if min_len < 2: return 0.5, 0.0
    
    member_losses = member_losses[:min_len]
    non_member_losses = non_member_losses[:min_len]

    # 4. 构建数据集 (0: Non-Member, 1: Member)
    samples = np.concatenate((non_member_losses, member_losses)).reshape((-1, 1))
    labels = [0] * len(non_member_losses) + [1] * len(member_losses)

    # 5. 计算指标
    mia_scores = simple_mia(samples, labels)
    mia_acc = mia_scores.mean()
    forget_score = abs(0.5 - mia_acc)
    
    return mia_acc, forget_score

# ==================== 基础工具函数 ====================
def add_image_noise(images, noise_level):
    if noise_level <= 0.0: return images
    noise = torch.randn_like(images) * noise_level
    return images + noise

def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    return tensor + torch.randn(tensor.size(), device=device) * std + mean

def add_laplace_noise(tensor, loc=0.0, scale=1.0):
    noise = torch.distributions.laplace.Laplace(loc, scale).sample(tensor.size()).to(device)
    return tensor + noise

def evaluate_accuracy(model, data_loader, device, unlearn_class):
    """计算 UA 和 OA"""
    model.eval()
    unlearn_correct = 0; unlearn_total = 0
    other_correct = 0; other_total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                if labels[i] == unlearn_class:
                    unlearn_total += 1
                    if predicted[i] == labels[i]: unlearn_correct += 1
                else:
                    other_total += 1
                    if predicted[i] == labels[i]: other_correct += 1

    ua = (unlearn_correct / unlearn_total * 100) if unlearn_total > 0 else 0.0
    oa = (other_correct / other_total * 100) if other_total > 0 else 0.0
    return ua, oa

# ==================== 数据与模型加载 ====================
def load_data(config):
    # imagenet_config = configparser.ConfigParser()
    # imagenet_config.read((base_path / 'config.ini').as_posix())
    # sys.path.append(imagenet_config['DEFAULT']['ImageNetDir'])
    from torch_imagenet import ImageNetDataset
    
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=_mean.flatten(), std=_std.flatten()),
    ])
    
    # 1. 遗忘集 (Member Data for MIA)
    unlearn_dataset = ImageNetDataset(root_dir=config['unlearn_data_dir'], transform=transform)
    # 这里的 batch_size=1 是为了 LRP 分析方便，但为了 MIA 计算可以共用
    unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, batch_size=1, shuffle=True)
    
    # 2. 测试集 (包含 Non-Member Data for MIA 和 OA 计算)
    test_dataset = datasets.ImageFolder(root=config['test_data_dir'], transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    
    return unlearn_loader, test_loader

def load_vgg_model(config):
    vgg = getattr(torchvision.models, f"vgg{config['vgg_version']}")(pretrained=config['use_pretrained']).to(device)
    vgg.eval()
    for param in vgg.parameters(): param.requires_grad = False
    return vgg

# ==================== 核心逻辑：分析与扰动 ====================
def analyze_neurons(lrp_model, unlearn_loader, rule, analyze_top_n, sample_size, noise_level, pattern, device):
    counter = []
    samples_processed = 0
    
    for x, y in unlearn_loader:
        if sample_size and samples_processed >= sample_size: break
        
        x, y = x.to(device), y.to(device)
        x_noisy = add_image_noise(x, noise_level)
        
        y_hat_lrp = lrp_model.forward(x_noisy, explain=True, rule=rule, pattern=pattern)
        y_hat_lrp = y_hat_lrp[torch.arange(x_noisy.shape[0]), y_hat_lrp.max(1)[1]].sum()
        
        lrp.trace.enable_and_clean()
        y_hat_lrp.backward()
        all_relevances = lrp.trace.collect_and_disable()
        
        # 只取第一个样本
        t = all_relevances[0][0].tolist()
        top_indices = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:analyze_top_n]
        counter.append(top_indices)
        samples_processed += 1
        
    all_numbers = [num for sublist in counter for num in sublist]
    return Counter(all_numbers)

def perturb_neurons(lrp_model, sorted_neurons, perturb_top_n, perturbation, config):
    """
    完全适配您原始代码逻辑的神经元扰动函数
    """
    # print(f"   [Perturb] Top-{perturb_top_n} neurons, Method: {perturbation}")
    
    # 获取前 N 个重要的神经元索引
    top_neurons = sorted_neurons[:perturb_top_n]
    
    # ------------------------------------------------------
    # 1. 扰动 FC1 (Layer 36): 切断神经元的输出
    # ------------------------------------------------------
    if config['perturb_fc1']:
        # 获取权重 (Out_Features, In_Features)
        fc1_weights = lrp_model[config['fc1_layer_idx']].weight.data.clone()
        
        if perturbation == 'zero':
            # 逻辑：将这些神经元对应的整行置为0（即该神经元对下一层没有输出）
            fc1_weights[top_neurons] = 0
            
        elif perturbation == 'gaussian':
            fc1_weights[top_neurons] = add_gaussian_noise(
                fc1_weights[top_neurons], 
                std=config['gaussian_std']
            )
            
        # 写回模型
        lrp_model[config['fc1_layer_idx']].weight.data = fc1_weights

    # ------------------------------------------------------
    # 2. 扰动 FC2 (Layer 39): 切断特定连接
    # ------------------------------------------------------
    if config['perturb_fc2']:
        fc2_weights = lrp_model[config['fc2_layer_idx']].weight.data.clone()
        
        # 您的原始逻辑：只修改 [unlearn_class, neuron_idx] 这个位置的权重
        # 注意：对于中间隐层，unlearn_class 只是指代该层第 k 个神经元，不代表最终类别
        target_idx = config['unlearn_class']
        
        if perturbation == 'zero':
            # 对应原代码: fc2_weights[config['unlearn_class'], neuron_idx] = 0
            # 使用高级索引一次性修改所有 top_neurons 对应的位置
            fc2_weights[target_idx, top_neurons] = 0
            
        elif perturbation == 'gaussian':
            fc2_weights[target_idx, top_neurons] = add_gaussian_noise(
                fc2_weights[target_idx, top_neurons], 
                std=config['gaussian_std']
            )
            
        lrp_model[config['fc2_layer_idx']].weight.data = fc2_weights

    # ------------------------------------------------------
    # 3. 扰动 Output (Layer 42): 针对最终分类层的处理
    # ------------------------------------------------------
    if config['perturb_output']:
        output_weights = lrp_model[config['output_layer_idx']].weight.data.clone()
        target_idx = config['unlearn_class']
        
        if perturbation == 'zero':
            # 对应原代码: target_class_weights[neuron_idx] = 0
            output_weights[target_idx, top_neurons] = 0
            
        elif perturbation == 'gaussian':
            output_weights[target_idx, top_neurons] = add_gaussian_noise(
                output_weights[target_idx, top_neurons],
                std=config['gaussian_std']
            )
            
        lrp_model[config['output_layer_idx']].weight.data = output_weights

    return lrp_model

# ==================== 主流程 ====================
def run_full_suite(config):
    # 1. 初始化
    print("[Init] Loading Data & Baseline Model...")
    unlearn_loader, test_loader = load_data(config)
    original_vgg = load_vgg_model(config)
    
    # 2. 评估基准 (Baseline Statistics)
    print("[Init] Evaluating Baseline...")
    base_ua, base_oa = evaluate_accuracy(original_vgg, test_loader, device, config['unlearn_class'])
    print(f" >> Baseline UA: {base_ua:.2f}% | OA: {base_oa:.2f}%")
    
    # 3. 准备 Patterns
    print("[Init] Loading Patterns...")
    lrp_temp = lrp.convert_vgg(original_vgg).to(device)
    patterns = load_patterns(config, lrp_temp, unlearn_loader) if config['use_patterns'] else None
    
    # 4. 生成组合
    combinations = list(itertools.product(
        config['rules'],
        config['analyze_top_n_list'],
        config['perturb_top_n_list'],
        config['perturbation_methods'],
        config['analysis_sample_sizes'],
        config['image_noise_levels']
    ))
    
    print(f"\n[Info] Total Configs: {len(combinations)}")
    print(f"[Info] Repeats per Config: {config['n_repeats']}")
    
    final_records = []
    
    for idx, (rule, ana_k, per_h, method, n_samples, noise) in enumerate(combinations):
        print(f"\n>>> [{idx+1}/{len(combinations)}] Config: Rule={rule}, K={ana_k}, H={per_h}, M={method}, S={n_samples}, N={noise}")
        
        # 存储单次运行结果
        runs_ua, runs_oa = [], []
        runs_mia, runs_fs = [], []
        
        for run_id in range(1, config['n_repeats'] + 1):
            set_seed(42 + run_id * 100)
            
            # A. 模型拷贝与转换
            current_model = copy.deepcopy(original_vgg)
            lrp_model = lrp.convert_vgg(current_model).to(device)
            
            # B. 分析 (Analysis)
            neuron_freq = analyze_neurons(
                lrp_model, unlearn_loader, rule, ana_k, n_samples, noise, patterns, device
            )
            sorted_neurons = [n for n, _ in neuron_freq.most_common()]
            
            # C. 扰动 (Perturbation)
            lrp_model = perturb_neurons(lrp_model, sorted_neurons, per_h, method, config)
            
            # D. 评估 UA & OA
            ua, oa = evaluate_accuracy(lrp_model, test_loader, device, config['unlearn_class'])
            runs_ua.append(ua)
            runs_oa.append(oa)
            
            # E. 评估 MIA & Forget Score
            # Member: unlearn_loader (Train set, Target Class)
            # Non-Member: test_loader (Test set, 需筛选 Target Class)
            mia, fs = calculate_mia_score(lrp_model, unlearn_loader, test_loader, config['unlearn_class'], device)
            runs_mia.append(mia)
            runs_fs.append(fs)
            
            print(f"   [Run {run_id}] UA:{ua:.1f} | OA:{oa:.1f} | MIA:{mia:.3f} | FS:{fs:.3f}")
            
            # 保存模型 (可选)
            if config['save_model']:
                save_dir = pathlib.Path(config['model_save_dir'])
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(lrp_model.state_dict(), save_dir / f"vgg_{rule}_{ana_k}_{per_h}_run{run_id}.pth")

        # 5. 统计分析
        stats_ua = calculate_statistics(runs_ua, base_ua)
        stats_oa = calculate_statistics(runs_oa, base_oa)
        stats_mia = calculate_statistics(runs_mia, 0.5) # MIA 理想基准 0.5
        stats_fs = calculate_statistics(runs_fs, 0.0)   # FS 理想基准 0.0
        
        # 6. 记录
        record = {
            'Rule': rule, 'Analyze_K': ana_k, 'Perturb_H': per_h, 
            'Method': method, 'Samples': n_samples, 'Noise': noise,
            
            # UA Stats
            'UA_Mean': stats_ua['mean'], 'UA_Std': stats_ua['std'], 'UA_P': stats_ua['p_value'],
            'UA_Cohens': stats_ua['cohens_d'], 'UA_Cliff': stats_ua['cliffs_delta'],
            
            # OA Stats
            'OA_Mean': stats_oa['mean'], 'OA_Std': stats_oa['std'], 'OA_P': stats_oa['p_value'],
            'OA_Cohens': stats_oa['cohens_d'], 'OA_Cliff': stats_oa['cliffs_delta'],
            
            # MIA Stats
            'MIA_Mean': stats_mia['mean'], 'MIA_Std': stats_mia['std'], 'MIA_P': stats_oa['p_value'],
            'MIA_Cohens': stats_mia['cohens_d'], 'MIA_Cliff': stats_mia['cliffs_delta'],
            
            # FS Stats
            'FS_Mean': stats_fs['mean'], 'FS_Std': stats_fs['std'], 'FS_P': stats_oa['p_value'],
            'FS_Cohens': stats_fs['cohens_d'], 'FS_Cliff': stats_fs['cliffs_delta'],
            
            # Raw
            'Raw_UA': str(runs_ua)
        }
        final_records.append(record)

    # 7. 保存 Excel
    if config['save_excel']:
        save_dir = pathlib.Path(config['results_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(final_records)
        excel_path = save_dir / f"org_vgg_experiment_{timestamp}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"\n[Success] Statistics report saved to: {excel_path}")

if __name__ == '__main__':
    run_full_suite(EXPERIMENT_CONFIG)