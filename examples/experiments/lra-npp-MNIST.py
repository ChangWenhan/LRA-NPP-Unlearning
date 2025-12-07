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

from torch.utils.data import DataLoader
from collections import Counter
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
# 工具函数：噪声与辅助
# ==========================================

def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    """给 Tensor 添加高斯噪声"""
    noise = torch.randn(tensor.size(), device=tensor.device) * std + mean
    return tensor + noise

def add_laplace_noise(tensor, loc=0.0, scale=1.0):
    """给 Tensor 添加拉普拉斯噪声"""
    m = torch.distributions.laplace.Laplace(torch.tensor([loc], device=tensor.device), 
                                            torch.tensor([scale], device=tensor.device))
    noise = m.sample(tensor.size()).squeeze(-1)
    return tensor + noise

def add_noise_to_image(image, level, method='gaussian'):
    """
    专门用于分析阶段给图像加噪
    level: 噪声强度 (std 或 scale)
    """
    if level <= 0:
        return image
    
    if method == 'gaussian':
        return add_gaussian_noise(image, mean=0.0, std=level)
    elif method == 'laplace':
        return add_laplace_noise(image, loc=0.0, scale=level)
    return image

def evaluate_accuracy(model, dataloader, device, unlearn_class):
    """统一的精度评估函数"""
    model.eval()
    correct = 0
    total = 0
    class1_total = 0
    class1_correct = 0
    classElse_total = 0
    classElse_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            class1_mask = (labels == unlearn_class)
            class1_total += class1_mask.sum().item()
            class1_correct += ((predicted == labels) & class1_mask).sum().item()
            
            classElse_mask = (labels != unlearn_class)
            classElse_total += classElse_mask.sum().item()
            classElse_correct += ((predicted == labels) & classElse_mask).sum().item()

    acc_target = 100 * class1_correct / class1_total if class1_total > 0 else 0
    acc_else = 100 * classElse_correct / classElse_total if classElse_total > 0 else 0
    
    return acc_target, acc_else

# ==========================================
# 核心逻辑：执行实验
# ==========================================

def run_unlearning_experiment(
    args,
    original_model, 
    dataloader_for_analysis, 
    test_loader,
    unlearn_class,
    rule_config,     # 传入包含 rule_name 和 pattern 的字典
    analyze_n,       
    perturb_n,       
    perturb_method,  
    input_noise_level 
):
    rule = rule_config['name']
    pattern = rule_config['pattern']
    pattern_type_str = rule_config.get('pattern_type', 'none') # 用于打印日志

    print(f"\n[Experiment] Rule={rule} ({pattern_type_str}) | Analyze={analyze_n} | Noise={input_noise_level} | Perturb={perturb_n} ({perturb_method})")
    
    model = copy.deepcopy(original_model)
    model.eval()
    
    counter = []
    start_time = time.time()

    # 1. 分析阶段 (Analysis Phase)
    for x, y in dataloader_for_analysis:
        x = x.to(args.device)
        y = y.to(args.device)
        
        # 对输入图像加噪
        if input_noise_level > 0:
            x = add_noise_to_image(x, level=input_noise_level, method='gaussian')
        
        x.requires_grad_(True)
        x.grad = None

        try:
            # 兼容 PatternNet: 传入 pattern 参数
            y_hat = model.forward(x, explain=True, rule=rule, pattern=pattern)
        except Exception as e:
            print(f"Error during LRP forward ({rule}): {e}")
            continue

        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
        y_hat = y_hat.sum()

        lrp.trace.enable_and_clean()
        y_hat.backward()
        all_relevances = lrp.trace.collect_and_disable()

        for t in all_relevances:
            t_list = t[0].tolist() 
            idx = sorted(range(len(t_list)), key=lambda i: t_list[i], reverse=True)[:analyze_n]
            counter.append(idx)
            break 

    all_numbers = [num for sublist in counter for num in sublist]
    number_counts = Counter(all_numbers)
    
    # 获取最终要干扰的 Top-K 神经元索引
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:perturb_n]
    target_neuron_indices = [num for num, _ in sorted_numbers]
    
    if len(target_neuron_indices) == 0:
        print("Warning: No neurons selected for perturbation.")
        return model

    # 2. 干扰阶段 (Unlearning/Perturbation Phase)
    fc_weights = model[6].weight.data
    fc2_weights = model[8].weight.data[unlearn_class] 

    if perturb_method == 'zero':
        for i in target_neuron_indices:
            fc2_weights[i] = 0
            fc_weights[i] = torch.zeros(fc_weights[i].shape, device=args.device)

    elif perturb_method == 'gaussian':
        for i in target_neuron_indices:
             noise = torch.randn(1, device=args.device).item() * 1.0 
             fc2_weights[i] += noise
             fc_weights[i] = add_gaussian_noise(fc_weights[i], std=1.0)
    
    elif perturb_method == 'laplace':
        for i in target_neuron_indices:
            m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
            noise = m.sample().item()
            fc2_weights[i] += noise
            fc_weights[i] = add_laplace_noise(fc_weights[i], scale=1.0)

    model[8].weight.data[unlearn_class] = fc2_weights
    model[6].weight.data = fc_weights

    end_time = time.time()
    
    # 3. 评估阶段
    acc_target, acc_else = evaluate_accuracy(model, test_loader, args.device, unlearn_class)
    print(f"Time: {end_time - start_time:.2f}s | Target Acc: {acc_target:.2f}% | Other Acc: {acc_else:.2f}%")
    
    return model

def main(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    
    unlearn_class = 8 

    # 1. 准备模型
    model = get_mnist_model()
    prepare_mnist_model(args, model, epochs=args.epochs, train_new=args.train_new)
    model = model.to(args.device)
    
    # 2. 准备数据
    train_loader, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=args.batch_size)

    # 准备 Analysis 数据集
    filtered_data = []
    for data, labels in test_loader:
        if (labels == unlearn_class).any():
            mask = (labels == unlearn_class)
            filtered_data.append((data[mask], labels[mask]))
    
    flat_imgs = torch.cat([x[0] for x in filtered_data])
    flat_lbls = torch.cat([x[1] for x in filtered_data])
    analysis_dataset = torch.utils.data.TensorDataset(flat_imgs, flat_lbls)
    analysis_loader = DataLoader(analysis_dataset, batch_size=1, shuffle=True) 

    # ==========================================
    # 3. 预计算 Pattern (用于 PatternNet/PatternAttribution)
    # ==========================================
    print("Checking/Computing Patterns...")
    
    # 计算 patterns_all
    all_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_all.pkl').as_posix()
    if not os.path.exists(all_patterns_path):
        print("Computing patterns (All)...")
        patterns_all = fit_patternnet(model, train_loader, device=args.device)
        store_patterns(all_patterns_path, patterns_all)
    else:
        patterns_all = [torch.tensor(p, device=args.device, dtype=torch.float32) for p in load_patterns(all_patterns_path)]

    # 计算 patterns_pos
    pos_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_pos.pkl').as_posix()
    if not os.path.exists(pos_patterns_path):
        print("Computing patterns (Positive)...")
        patterns_pos = fit_patternnet_positive(model, train_loader, device=args.device)
        store_patterns(pos_patterns_path, patterns_pos)
    else:
        patterns_pos = [torch.from_numpy(p).to(args.device) for p in load_patterns(pos_patterns_path)]

    print("Patterns ready.")

    # ==========================================
    # 4. 实验配置参数 (全部规则)
    # ==========================================
    
    # 构建所有需要跑的规则组合
    experiment_configs = []

    # (A) 标准 LRP 规则 (不需要 pattern)
    standard_rules = ["epsilon", "gamma+epsilon", "alpha1beta0", "alpha2beta1"]
    for r in standard_rules:
        experiment_configs.append({'name': r, 'pattern': None, 'pattern_type': 'none'})

    # (B) PatternNet 系列 (需要 patterns_all 或 patterns_pos)
    pattern_rules = ["patternnet", "patternattribution"]
    for r in pattern_rules:
        # 添加使用 All Patterns 的版本 ($S(x)$)
        experiment_configs.append({'name': r, 'pattern': patterns_all, 'pattern_type': 'all'})
        # 添加使用 Positive Patterns 的版本 ($S(x)_{+-}$)
        experiment_configs.append({'name': r, 'pattern': patterns_pos, 'pattern_type': 'pos'})

    # 其他列表参数
    analyze_neuron_counts = [50] 
    input_noise_levels = [0] 
    perturb_neuron_counts = [80, 100]
    perturb_methods = ['zero']

    # ==========================================
    # 5. 批量运行
    # ==========================================
    
    total_experiments = len(experiment_configs) * len(analyze_neuron_counts) * len(input_noise_levels) * len(perturb_neuron_counts) * len(perturb_methods)
    print(f"Total experiments queued: {total_experiments}")
    
    count = 0
    for conf in experiment_configs:
        for analyze_n in analyze_neuron_counts:
            for noise_lvl in input_noise_levels:
                for perturb_n in perturb_neuron_counts:
                    for p_method in perturb_methods:
                        count += 1
                        print(f"[{count}/{total_experiments}]", end=" ")
                        
                        unlearned_model = run_unlearning_experiment(
                            args=args,
                            original_model=model,
                            dataloader_for_analysis=analysis_loader,
                            test_loader=test_loader,
                            unlearn_class=unlearn_class,
                            rule_config=conf, # 传入配置对象
                            analyze_n=analyze_n,
                            perturb_n=perturb_n,
                            perturb_method=p_method,
                            input_noise_level=noise_lvl
                        )

                        # 可以在这里添加保存模型或Log的代码
                        # log_result(...)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST LRP Unlearning Benchmark")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_new', action='store_true', help='Train new predictive model')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--seed', '-d', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(random.random() * 1e9)
        print("Setting seed: %i" % args.seed)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)