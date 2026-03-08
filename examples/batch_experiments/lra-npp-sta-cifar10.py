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
from collections import Counter
from itertools import product
from scipy import stats  # 统计分析核心
from sklearn import linear_model, model_selection  # MIA 核心

import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset

# ==================== 路径配置 ====================
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())
sys.path.insert(0, "/home/cwh/Workspace/TorchLRP-master")

import lrp
import lrp.trace 

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    # ========== 基础配置 (保持你的原样) ==========
    'unlearn_class': 9,             # 遗忘类别 (CIFAR10: Truck)
    'data_root': '/home/cwh/Workspace/TorchLRP-master/data',
    'model_path': 'examples/models/resnet50_cifar10_epoch_10.pth',
    
    # ========== 变量列表 (笛卡尔积组合) ==========
    'rules': ['epsilon'], # 变量1: LRP规则
    'analysis_sample_sizes': [36],      # 变量2: 分析样本数
    'image_noise_levels': [0.0],        # 变量3: 图像噪声
    'analyze_top_n_list': [200],        # 变量4: 分析Top-N
    'perturb_top_n_list': [400],        # 变量5: 切除Top-N
    'perturbation_methods': ['zero'],   # 变量6: 扰动方法

    # ========== 噪声参数 ==========
    'gaussian_std': 1.0,
    'laplace_scale': 1.0,

    # ========== 统计与保存配置 ==========
    'n_repeats': 5,                 # 每个配置重复跑 5 次 (计算 Mean/Std/P-Val)
    
    'save_neurons': False,           # [恢复功能] 保存神经元列表 .pkl
    'neuron_save_dir': 'neuron/cifar-10/',
    
    'save_model': True,            # [恢复功能] 保存模型 .pth
    'model_save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia',
    
    'save_excel': False,             # 保存 Excel 统计报告
    'results_save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/models_statistic',
}

# ==================== 设备与全局设置 ====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 兼容 PyTorch 2.6+
try:
    from torchvision.models.resnet import ResNet
    torch.serialization.add_safe_globals([ResNet])
except:
    pass

# ==================== 1. 核心逻辑工具 (严格还原) ====================

def add_image_noise(images, noise_level):
    if noise_level <= 0.0: return images
    noise = torch.randn_like(images) * noise_level
    return images + noise

def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    noise = torch.randn(tensor.size(), device=device) * std + mean
    return tensor + noise

def add_laplace_noise(tensor, loc=0.0, scale=1.0):
    noise = torch.distributions.laplace.Laplace(loc, scale).sample(tensor.size()).to(device)
    return tensor + noise

def analyze_neurons(lrp_model, target_dataset, rule, analyze_top_n, sample_size, noise_level, device):
    """还原你的分析逻辑：对 target_dataset 进行采样分析"""
    # 随机采样
    if sample_size < len(target_dataset):
        indices = torch.randperm(len(target_dataset))[:sample_size].tolist()
        sampled_dataset = Subset(target_dataset, indices)
    else:
        sampled_dataset = target_dataset
    
    sample_loader = DataLoader(sampled_dataset, batch_size=1, shuffle=False)
    counter = []
    
    for x, y in sample_loader:
        x, y = x.to(device), y.to(device)
        x_noisy = add_image_noise(x, noise_level)
        
        # Forward
        y_hat_lrp = lrp_model.forward(x_noisy, explain=True, rule=rule)
        y_hat_lrp = y_hat_lrp[torch.arange(x_noisy.shape[0]), y_hat_lrp.max(1)[1]]
        y_hat_lrp = y_hat_lrp.sum()

        # Backward (Trace)
        lrp.trace.enable_and_clean()
        y_hat_lrp.backward()
        all_relevances = lrp.trace.collect_and_disable()

        # Collect Top-N (取第一层或根据你的逻辑取值)
        # 注意：这里假设 all_relevances 返回的是 list，我们取第一个元素分析
        # 如果你的 trace 返回结构不同，请确保这里和你原始代码一致
        for t in all_relevances:
            t_list = t[0].tolist() # Batch size 1
            top_indices = sorted(range(len(t_list)), key=lambda i: t_list[i], reverse=True)[:analyze_top_n]
            counter.append(top_indices)
            break # 只取第一层相关性? (保持你原始代码的 break)

    all_numbers = [num for sublist in counter for num in sublist]
    return Counter(all_numbers)

def perturb_neurons(lrp_model, sorted_neurons, perturb_top_n, perturbation, config):
    """还原你的扰动逻辑：操作 lrp_model[22]"""
    top_neurons = sorted_neurons[:perturb_top_n]
    fc_weights = lrp_model[22].weight.data.clone() # 硬编码第22层
    
    for class_num in range(10): # 遍历 10 个类别
        for neuron_idx in top_neurons:
            if perturbation == 'zero':
                fc_weights[class_num][neuron_idx] = 0
            elif perturbation == 'gaussian':
                fc_weights[class_num][neuron_idx] = add_gaussian_noise(
                    fc_weights[class_num][neuron_idx], std=config['gaussian_std'])
            elif perturbation == 'laplace':
                fc_weights[class_num][neuron_idx] = add_laplace_noise(
                    fc_weights[class_num][neuron_idx], scale=config['laplace_scale'])
    
    lrp_model[22].weight.data = fc_weights
    return lrp_model

# ==================== 2. 统计与评估工具 (全面记录版) ====================

def calculate_cliff_delta(lst1, baseline_val):
    more = sum(x > baseline_val for x in lst1)
    less = sum(x < baseline_val for x in lst1)
    return (more - less) / len(lst1)

def calculate_statistics(current_values, baseline_value):
    """计算完整的统计指标：Mean, Std, P, Cohen, Cliff"""
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # P-Value
    if std < 1e-9:
        p_val = 0.0 if abs(mean - baseline_value) > 1e-5 else 1.0
    else:
        _, p_val = stats.ttest_1samp(values, baseline_value)
    
    # Cohen's d
    cohens_d = (mean - baseline_value) / std if std > 1e-9 else 0.0
    
    # Cliff's Delta
    cliffs_d = calculate_cliff_delta(values, baseline_value)
    
    return {
        "mean": mean, "std": std, "p_value": p_val,
        "cohens_d": cohens_d, "cliffs_delta": cliffs_d
    }

def evaluate_accuracy(model, loader, target_class, device):
    """计算 UA 和 OA"""
    model.eval()
    correct_target, total_target = 0, 0
    correct_others, total_others = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                if labels[i] == target_class:
                    total_target += 1
                    if predicted[i] == labels[i]: correct_target += 1
                else:
                    total_others += 1
                    if predicted[i] == labels[i]: correct_others += 1
    
    ua = (correct_target / total_target * 100) if total_target > 0 else 0.0
    oa = (correct_others / total_others * 100) if total_others > 0 else 0.0
    return ua, oa

# MIA 辅助
def compute_losses(net, loader, device, target_class=None):
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    net.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if target_class is not None:
                mask = (labels == target_class)
                if mask.sum() == 0: continue
                inputs, labels = inputs[mask], labels[mask]
            if len(inputs) == 0: continue
            logits = net(inputs)
            losses = criterion(logits, labels).cpu().detach().numpy()
            all_losses.extend(losses)
    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10):
    if len(np.unique(members)) < 2: return np.array([0.5])
    model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    return model_selection.cross_val_score(model, sample_loss, members, cv=cv, scoring="accuracy")

def calculate_mia_score(model, member_loader, non_member_loader, target_class, device):
    """
    Member: 训练集中的 Target Class
    Non-Member: 测试集中的 Target Class
    """
    m_losses = compute_losses(model, member_loader, device, target_class)
    nm_losses = compute_losses(model, non_member_loader, device, target_class)
    
    if len(m_losses) == 0 or len(nm_losses) == 0: return 0.5, 0.0
    
    # 平衡数据
    min_len = min(len(m_losses), len(nm_losses))
    if min_len < 5: return 0.5, 0.0
    
    np.random.shuffle(m_losses)
    np.random.shuffle(nm_losses)
    
    samples = np.concatenate((nm_losses[:min_len], m_losses[:min_len])).reshape(-1, 1)
    labels = [0] * min_len + [1] * min_len # 0: Non-Member, 1: Member
    
    acc = simple_mia(samples, labels).mean()
    return acc, abs(0.5 - acc)

# ==================== 3. 主流程 ====================

def run_experiment():
    config = EXPERIMENT_CONFIG
    
    # 1. 准备数据
    print("加载数据...")
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 数据集：Test用于分析神经元(Non-Member), Train用于验证准确率(Member)
    test_set = torchvision.datasets.CIFAR10(root=config['data_root'], train=False, download=True, transform=transform)
    train_set = torchvision.datasets.CIFAR10(root=config['data_root'], train=True, download=True, transform=transform)
    
    # 从 Test Set 中提取目标类别用于 LRP 分析 (严格遵循你的 load_data 逻辑)
    target_indices = [i for i, (_, label) in enumerate(test_set) if label == config['unlearn_class']]
    target_subset_for_analysis = Subset(test_set, target_indices)
    
    # Loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader_full = DataLoader(test_set, batch_size=32, shuffle=False) # 全量测试集(含非目标类)
    
    # 2. 评估基准线 (Baseline)
    print("评估基准模型...")
    base_model = torch.load(config['model_path'], weights_only=False).to(device)
    base_ua, base_oa = evaluate_accuracy(base_model, train_loader, config['unlearn_class'], device)
    print(f" >> Baseline (Train Set): UA={base_ua:.2f}%, OA={base_oa:.2f}%")
    
    # 3. 生成组合
    combinations = list(product(
        config['rules'], config['analysis_sample_sizes'], config['image_noise_levels'],
        config['analyze_top_n_list'], config['perturb_top_n_list'], config['perturbation_methods']
    ))
    
    final_records = []
    print(f"\n开始实验，共 {len(combinations)} 种配置，每种重复 {config['n_repeats']} 次")

    for (rule, n_samples, noise, k_ana, k_pert, method) in combinations:
        print(f"\n>>> [Config] Rule={rule} | Samples={n_samples} | Noise={noise} | K_Ana={k_ana} | K_Pert={k_pert} | Method={method}")
        
        # 存储多次运行的结果
        runs_ua, runs_oa, runs_mia, runs_fs = [], [], [], []
        
        for run_id in range(1, config['n_repeats'] + 1):
            seed = 42 + run_id * 100
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
            
            # 复制模型
            curr_model = copy.deepcopy(base_model)
            lrp_model = lrp.convert_vgg(curr_model).to(device)
            
            # Step 1: 分析
            neuron_counts = analyze_neurons(lrp_model, target_subset_for_analysis, rule, k_ana, n_samples, noise, device)
            sorted_neurons = [n for n, _ in sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)]
            
            # [恢复功能] 保存神经元列表
            if config['save_neurons']:
                save_dir = pathlib.Path(config['neuron_save_dir'])
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = f"class{config['unlearn_class']}_{rule}_ana{k_ana}_per{k_pert}_{method}_s{n_samples}_n{noise}_run{run_id}.pkl"
                with open(save_dir / fname, 'wb') as f:
                    pickle.dump(sorted_neurons[:k_pert], f)

            # Step 2: 扰动
            lrp_model = perturb_neurons(lrp_model, sorted_neurons, k_pert, method, config)
            
            # Step 3: 验证 (UA/OA 用 Train Set，遵循你的 validate_model 逻辑)
            ua, oa = evaluate_accuracy(lrp_model, train_loader, config['unlearn_class'], device)
            
            # Step 4: MIA (Member=Train, Non-Member=Test)
            mia, fs = calculate_mia_score(lrp_model, train_loader, test_loader_full, config['unlearn_class'], device)
            
            runs_ua.append(ua); runs_oa.append(oa); runs_mia.append(mia); runs_fs.append(fs)
            print(f"   Run {run_id}: UA={ua:.2f} | OA={oa:.2f} | MIA={mia:.3f} | FS={fs:.3f}")
            
            # [恢复功能] 保存模型 (只存第一次Run节省空间，或者全部存)
            if config['save_model']:
                mdir = pathlib.Path(config['model_save_dir'])
                mdir.mkdir(parents=True, exist_ok=True)
                torch.save(lrp_model, mdir / f"c10_{rule}_{k_pert}_{method}_s{n_samples}_n{noise}_{run_id}.pth")

        # 4. 统计汇总 (修正：记录所有详细指标)
        s_ua = calculate_statistics(runs_ua, base_ua)
        s_oa = calculate_statistics(runs_oa, base_oa)
        s_mia = calculate_statistics(runs_mia, 0.5)
        s_fs = calculate_statistics(runs_fs, 0.0)

        rec = {
            'Rule': rule, 'Samples': n_samples, 'Noise': noise, 'K_Ana': k_ana, 'K_Pert': k_pert, 'Method': method,
            # UA Stats
            'UA_Mean': s_ua['mean'], 'UA_Std': s_ua['std'], 'UA_P_Val': s_ua['p_value'], 'UA_Cohen': s_ua['cohens_d'], 'UA_Cliff': s_ua['cliffs_delta'],
            # OA Stats
            'OA_Mean': s_oa['mean'], 'OA_Std': s_oa['std'], 'OA_P_Val': s_oa['p_value'], 'OA_Cohen': s_oa['cohens_d'], 'OA_Cliff': s_oa['cliffs_delta'],
            # MIA Stats
            'MIA_Mean': s_mia['mean'], 'MIA_Std': s_mia['std'], 'MIA_P_Val': s_mia['p_value'], 'MIA_Cohen': s_mia['cohens_d'], 'MIA_Cliff': s_mia['cliffs_delta'],
            # FS Stats
            'FS_Mean': s_fs['mean'], 'FS_Std': s_fs['std'], 'FS_P_Val': s_fs['p_value'], 'FS_Cohen': s_fs['cohens_d'], 'FS_Cliff': s_fs['cliffs_delta'],
        }
        final_records.append(rec)

    # 5. 保存报告
    if config['save_excel']:
        df = pd.DataFrame(final_records)
        save_path = pathlib.Path(config['results_save_dir'])
        save_path.mkdir(parents=True, exist_ok=True)
        t_str = time.strftime('%Y%m%d_%H%M%S')
        df.to_excel(save_path / f"org_Cifar10_report_{t_str}.xlsx", index=False)
        print(f"\n[完成] 详细统计报告已保存至 {save_path}")

if __name__ == '__main__':
    run_experiment()