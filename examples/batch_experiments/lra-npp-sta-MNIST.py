import os
import sys
import copy
import time
import torch
import random
import pathlib
import argparse
import torchvision
import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
from collections import Counter
from sklearn import linear_model, model_selection # MIA 依赖
from torch.utils.data import DataLoader
from pathlib import Path

# ==========================================
# 路径与导入设置
# ==========================================
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

from utils import get_mnist_model, prepare_mnist_model, get_mnist_data
from utils import store_patterns, load_patterns
import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive 

# ==========================================
# 0. 全局配置 (Centralized Configuration)
# ==========================================
EXPERIMENT_CONFIG = {
    # ========== 基础设置 ==========
    'batch_size': 64,
    'epochs': 5,
    'train_new': False,        # 是否重新训练基础模型
    'unlearn_class': 1,        # 待遗忘类别
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
    
    # ========== 实验变量 (笛卡尔积组合) ==========
    # 1. 规则
    'rules_list': [
        {'name': 'epsilon', 'type': 'none'},
        # {'name': 'gamma+epsilon', 'type': 'none'},
        # {'name': 'alpha1beta0', 'type': 'none'},
        # {'name': 'alpha2beta1', 'type': 'none'},
        # {'name': 'patternnet', 'type': 'all'}, 
        # {'name': 'patternnet', 'type': 'pos'}, 
        # {'name': 'patternattribution', 'type': 'all'}, 
        # {'name': 'patternattribution', 'type': 'pos'}, 
    ],
    
    # 2. Unlearn 样本数量列表
    'max_unlearn_samples_list': [36], 

    # 3. 其他变量
    'analyze_neuron_counts': [50],      # 分析阶段取 Top-K
    'perturb_neuron_counts': [80],      # 干扰阶段修改 Top-K
    'input_noise_levels': [0.0],   # 图像噪声等级
    'perturb_method': 'zero',      # 干扰方式: zero, gaussian, laplace
    
    # ========== 统计与保存配置 ==========
    'n_repeats': 5,            # 每个配置重复运行次数
    
    # [Excel 报表保存路径]
    'save_excel': False,
    'save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/models_statistic',
    
    # [新增: 模型 Checkpoint 保存配置]
    'save_individual_model': True,  # 开关：是否保存每一次Run的模型
    # 您可以在这里自行设置保存路径，就像 save_dir 一样
    'model_save_dir': '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia' 
}

# ==========================================
# 1. 统计与 MIA 工具函数 (保持不变)
# ==========================================

def calculate_cliff_delta(lst1, baseline_val):
    more = sum(x > baseline_val for x in lst1)
    less = sum(x < baseline_val for x in lst1)
    return (more - less) / len(lst1)

def calculate_statistics(current_values, baseline_value):
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # if std < 1e-9:
    #     p_val = 0.0 if abs(mean - baseline_value) > 1e-5 else 1.0
    # else:
    _, p_val = stats.ttest_1samp(values, baseline_value)
    
    cohens_d = (mean - baseline_value) / std if std > 1e-9 else 0.0
    cliffs_d = calculate_cliff_delta(values, baseline_value)
    
    return {
        "mean": mean, "std": std, "p_value": p_val,
        "cohens_d": cohens_d, "cliffs_delta": cliffs_d
    }

def compute_losses(net, loader, device, target_class=None):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    net.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0], batch[1]
            inputs, labels = inputs.to(device), labels.to(device)

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
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        return np.array([0.5]) 

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def calculate_mia_metric(model, train_loader, test_loader, target_class, device):
    member_losses = compute_losses(model, train_loader, device, target_class=target_class)
    non_member_losses = compute_losses(model, test_loader, device, target_class=target_class)

    if len(member_losses) == 0 or len(non_member_losses) == 0:
        return 0.5, 0.0

    np.random.shuffle(member_losses)
    np.random.shuffle(non_member_losses)
    
    min_len = min(len(member_losses), len(non_member_losses))
    if min_len < 5: return 0.5, 0.0 

    member_losses = member_losses[:min_len]
    non_member_losses = non_member_losses[:min_len]

    samples = np.concatenate((non_member_losses, member_losses)).reshape((-1, 1))
    labels = [0] * len(non_member_losses) + [1] * len(member_losses)

    mia_scores = simple_mia(samples, labels)
    mia_mean = mia_scores.mean()
    forget_score = abs(0.5 - mia_mean)

    return mia_mean, forget_score

# ==========================================
# 2. 辅助功能函数 (保持不变)
# ==========================================

def add_noise_to_image(image, level, method='gaussian'):
    if level <= 0: return image
    device = image.device
    if method == 'gaussian':
        noise = torch.randn_like(image) * level
    elif method == 'laplace':
        m = torch.distributions.laplace.Laplace(torch.tensor([0.0], device=device), torch.tensor([level], device=device))
        noise = m.sample(image.size()).squeeze(-1)
    else:
        return image
    return image + noise

def evaluate_accuracy(model, dataloader, device, unlearn_class):
    model.eval()
    class1_total = 0; class1_correct = 0
    classElse_total = 0; classElse_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            mask_u = (labels == unlearn_class)
            class1_total += mask_u.sum().item()
            class1_correct += ((predicted == labels) & mask_u).sum().item()
            
            mask_o = (labels != unlearn_class)
            classElse_total += mask_o.sum().item()
            classElse_correct += ((predicted == labels) & mask_o).sum().item()

    acc_target = 100 * class1_correct / class1_total if class1_total > 0 else 0
    acc_else = 100 * classElse_correct / classElse_total if classElse_total > 0 else 0
    return acc_target, acc_else

# ==========================================
# 3. 核心实验逻辑 (修改后: 支持保存路径)
# ==========================================

def run_single_experiment(
    config, original_model, 
    analysis_loader,    
    train_loader_full,  
    test_loader_full,   
    rule_info,          
    analyze_n, perturb_n, input_noise_level,
    max_samples,
    save_path=None # [新增参数] 接收具体的文件保存路径
):
    model = copy.deepcopy(original_model)
    model.eval()
    device = config['device']
    unlearn_class = config['unlearn_class']
    
    # --- Phase 1: Analysis (LRP) ---
    counter = []
    sample_count = 0 
    
    for x, y in analysis_loader:
        if max_samples is not None and sample_count >= max_samples:
            break

        x, y = x.to(device), y.to(device)
        
        if input_noise_level > 0:
            x = add_noise_to_image(x, level=input_noise_level)
        
        x.requires_grad_(True)
        try:
            y_hat = model.forward(x, explain=True, rule=rule_info['name'], pattern=rule_info['pattern'])
            y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]].sum()
            
            lrp.trace.enable_and_clean()
            y_hat.backward()
            all_relevances = lrp.trace.collect_and_disable()

            for t in all_relevances:
                t_list = t[0].tolist() 
                idx = sorted(range(len(t_list)), key=lambda i: t_list[i], reverse=True)[:analyze_n]
                counter.append(idx)
                sample_count += 1 
                break 
        except Exception as e:
            continue

    all_numbers = [num for sublist in counter for num in sublist]
    if not all_numbers: return None 

    # 选出高频神经元
    target_neuron_indices = [num for num, _ in Counter(all_numbers).most_common(perturb_n)]
    
    # --- Phase 2: Perturbation (Unlearning) ---
    fc_weights = model[6].weight.data
    fc2_weights = model[8].weight.data[unlearn_class] 

    method = config['perturb_method']
    if method == 'zero':
        for i in target_neuron_indices:
            fc2_weights[i] = 0
            fc_weights[i] = torch.zeros_like(fc_weights[i])

    model[8].weight.data[unlearn_class] = fc2_weights
    model[6].weight.data = fc_weights

    # [新增逻辑] 保存模型到指定路径
    if save_path is not None:
        try:
            torch.save(model.state_dict(), save_path)
        except Exception as e:
            print(f"[Warning] Failed to save model to {save_path}: {e}")

    # --- Phase 3: Evaluation ---
    ua, oa = evaluate_accuracy(model, test_loader_full, device, unlearn_class)
    mia, fs = calculate_mia_metric(model, train_loader_full, test_loader_full, unlearn_class, device)

    return {'UA': ua, 'OA': oa, 'MIA': mia, 'FS': fs}

# ==========================================
# 4. 主程序 (修改后: 读取配置并生成路径)
# ==========================================

def main():
    cfg = EXPERIMENT_CONFIG
    print(f"Using device: {cfg['device']}")
    
    # [新增] 检查并创建模型保存目录
    if cfg['save_individual_model']:
        model_save_dir = Path(cfg['model_save_dir'])
        if not model_save_dir.exists():
            print(f"[Info] Creating directory: {model_save_dir}")
            model_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[Info] Saving models to existing directory: {model_save_dir}")

    # 1. 准备模型与数据
    model = get_mnist_model()
    class Args: pass
    args = Args(); args.device = cfg['device']; args.batch_size = cfg['batch_size']
    
    prepare_mnist_model(args, model, epochs=cfg['epochs'], train_new=cfg['train_new'])
    model = model.to(cfg['device'])
    
    train_loader, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=cfg['batch_size'])
    
    filtered_data = []
    for data, labels in test_loader:
        mask = (labels == cfg['unlearn_class'])
        if mask.any(): filtered_data.append((data[mask], labels[mask]))
    
    analysis_dataset = torch.utils.data.TensorDataset(
        torch.cat([x[0] for x in filtered_data]), 
        torch.cat([x[1] for x in filtered_data])
    )
    
    # 2. 预计算 Pattern
    need_all = any(r['type'] == 'all' for r in cfg['rules_list'])
    need_pos = any(r['type'] == 'pos' for r in cfg['rules_list'])
    
    patterns_all = None
    patterns_pos = None

    if need_all:
        p_path = (base_path / 'examples' / 'patterns' / 'pattern_all.pkl').as_posix()
        if os.path.exists(p_path):
            patterns_all = [torch.tensor(p, device=cfg['device'], dtype=torch.float32) for p in load_patterns(p_path)]
        else:
            print("Computing patterns (All)...")
            patterns_all = fit_patternnet(model, train_loader, device=cfg['device'])
            store_patterns(p_path, patterns_all)

    if need_pos:
        p_path = (base_path / 'examples' / 'patterns' / 'pattern_pos.pkl').as_posix()
        if os.path.exists(p_path):
            patterns_pos = [torch.from_numpy(p).to(cfg['device']) for p in load_patterns(p_path)]
        else:
            print("Computing patterns (Positive)...")
            patterns_pos = fit_patternnet_positive(model, train_loader, device=cfg['device'])
            store_patterns(p_path, patterns_pos)

    # 3. 计算 Baseline 指标
    print("Evaluating Baseline...")
    base_ua, base_oa = evaluate_accuracy(model, test_loader, cfg['device'], cfg['unlearn_class'])
    base_mia, base_fs = calculate_mia_metric(model, train_loader, test_loader, cfg['unlearn_class'], cfg['device'])
    print(f"Baseline -> UA: {base_ua:.2f} | OA: {base_oa:.2f} | MIA: {base_mia:.3f} | FS: {base_fs:.3f}")

    # 4. 生成实验组合
    combinations = list(product(
        cfg['rules_list'],
        cfg['max_unlearn_samples_list'], 
        cfg['analyze_neuron_counts'],
        cfg['input_noise_levels'],
        cfg['perturb_neuron_counts']
    ))
    
    total_runs = len(combinations) * cfg['n_repeats']
    print(f"\n[Info] Total Configs: {len(combinations)} | Repeats: {cfg['n_repeats']} | Total Runs: {total_runs}")
    
    final_records = []

    # 5. 执行实验循环
    for (rule_cfg, n_samples, k_analyze, noise, h_perturb) in combinations:
        print(f"\n>>> Config: Rule={rule_cfg['name']}, Samples={n_samples}, Noise={noise}, K={k_analyze}, H={h_perturb}")
        
        current_pattern = None
        if rule_cfg['type'] == 'all': current_pattern = patterns_all
        elif rule_cfg['type'] == 'pos': current_pattern = patterns_pos
        
        rule_info = {'name': rule_cfg['name'], 'pattern': current_pattern}

        runs_ua, runs_oa = [], []
        runs_mia, runs_fs = [], []
        
        for i in range(cfg['n_repeats']):
            seed = 42 + i * 100
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            
            # Shuffle Analysis Data for variability
            curr_analysis_loader = DataLoader(analysis_dataset, batch_size=1, shuffle=True)
            
            # [新增] 动态生成文件名并拼接完整路径
            current_save_path = None
            if cfg['save_individual_model']:
                # 替换特殊字符以防文件名错误
                safe_rule = rule_cfg['name'].replace('+', '_')
                file_name = (
                    f"mnist_{safe_rule}_Samp_{n_samples}_"
                    f"Noise_{noise}_K_{k_analyze}_H_{h_perturb}_"
                    f"Run_{i}.pth"
                )
                current_save_path = Path(cfg['model_save_dir']) / file_name

            res = run_single_experiment(
                cfg, model, 
                curr_analysis_loader, 
                train_loader,   
                test_loader,    
                rule_info, 
                k_analyze, h_perturb, noise,
                n_samples,
                save_path=current_save_path # [传递路径]
            )
            
            if res:
                runs_ua.append(res['UA']); runs_oa.append(res['OA'])
                runs_mia.append(res['MIA']); runs_fs.append(res['FS'])
                
                # 打印日志显示是否保存
                save_status = "Saved" if current_save_path else "NoSave"
                print(f"   [Run {i}] {save_status} | UA:{res['UA']:.2f} | MIA:{res['MIA']:.3f}")

        # 6. 统计分析
        stats_ua = calculate_statistics(runs_ua, base_ua)
        stats_oa = calculate_statistics(runs_oa, base_oa)
        stats_mia = calculate_statistics(runs_mia, 0.5)
        stats_fs = calculate_statistics(runs_fs, 0.0)
        
        # 7. 记录数据
        rec = {
            'Rule': rule_cfg['name'], 'Type': rule_cfg['type'],
            'Samples': n_samples, 
            'Noise': noise, 'K_Analyze': k_analyze, 'H_Perturb': h_perturb,
            
            # UA Stats
            'UA_Mean': stats_ua['mean'], 'UA_Std': stats_ua['std'], 
            'UA_P_Val': stats_ua['p_value'], 'UA_Cohen_D': stats_ua['cohens_d'], 'UA_Cliff_D': stats_ua['cliffs_delta'],
            
            # OA Stats
            'OA_Mean': stats_oa['mean'], 'OA_Std': stats_oa['std'], 
            'OA_P_Val': stats_oa['p_value'], 'OA_Cohen_D': stats_oa['cohens_d'], 'OA_Cliff_D': stats_oa['cliffs_delta'],
            
            # MIA Stats
            'MIA_Mean': stats_mia['mean'], 'MIA_Std': stats_mia['std'], 
            'MIA_P_Val': stats_mia['p_value'], 'MIA_Cohen_D': stats_mia['cohens_d'], 'MIA_Cliff_D': stats_mia['cliffs_delta'],
            
            # FS Stats
            'FS_Mean': stats_fs['mean'], 'FS_Std': stats_fs['std'], 
            'FS_P_Val': stats_fs['p_value'], 'FS_Cohen_D': stats_fs['cohens_d'], 'FS_Cliff_D': stats_fs['cliffs_delta'],
            
            # Raw Data
            'Raw_UA': str(runs_ua),
            'Raw_MIA': str(runs_mia)
        }
        final_records.append(rec)

    # 8. 保存 Excel
    if cfg['save_excel']:
        df = pd.DataFrame(final_records)
        Path(cfg['save_dir']).mkdir(parents=True, exist_ok=True)
        
        # 定义列顺序
        cols = ['Rule', 'Type', 'Samples', 'Noise', 'K_Analyze', 'H_Perturb', 
                'UA_Mean', 'UA_Std', 'UA_P_Val', 'UA_Cohen_D', 'UA_Cliff_D',
                'OA_Mean', 'OA_Std', 'OA_P_Val', 'OA_Cohen_D', 'OA_Cliff_D',
                'MIA_Mean', 'MIA_Std', 'MIA_P_Val', 'MIA_Cohen_D', 'MIA_Cliff_D',
                'FS_Mean', 'FS_Std', 'FS_P_Val', 'FS_Cohen_D', 'FS_Cliff_D',
                'Raw_UA', 'Raw_MIA']
        
        final_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[final_cols]
        
        save_path = Path(cfg['save_dir']) / f"org_mnist_report_{time.strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(save_path, index=False)
        print(f"\n[Success] Report saved to {save_path}")

if __name__ == '__main__':
    main()