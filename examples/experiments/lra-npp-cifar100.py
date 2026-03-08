import sys
import copy
import time
import torch
import pickle
from torch.utils.data import DataLoader, Subset

import pathlib
import torchvision
from collections import Counter
from torchvision import transforms as T
import itertools

# ==================== 路径配置 ====================
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import lrp
from lrp.patterns import fit_patternnet_positive
from utils import store_patterns, load_patterns

# ==================== 实验配置 ====================
EXPERIMENT_CONFIG = {
    # 数据配置
    'unlearn_class': 9,
    'batch_size': 64,
    'test_batch_size': 8,
    'data_root': './data',
    'model_path': 'examples/models/resnet50_cifar100_5.pth',
    'num_classes': 100,  # CIFAR-100 有 100 个类别
    
    # ========== 独立参数列表（批量实验） ==========
    # LRP传播规则列表
    'rules': ['alpha1beta0', 'epsilon', 'gamma+epsilon', 'alpha2beta1'],
    
    # 每个样本分析的神经元数量列表
    'analyze_top_n_list': [150],
    
    # 最终扰动的神经元数量列表
    'perturb_top_n_list': [200, 250],
    
    # 扰动方法列表: 'zero'(置零), 'gaussian'(高斯噪声), 'laplace'(拉普拉斯噪声)
    'perturbation_methods': ['zero'],
    
    # 分析神经元时的数据集大小（从目标类别中采样的样本数）
    'analysis_sample_sizes': [100],  # 默认使用100张图像进行分析
    
    # 分析时对图像添加噪声的强度列表（标准差，0表示不加噪）
    'image_noise_levels': [0.0],  # 例如: [0.0, 0.01, 0.05, 0.1]
    
    # ========== 噪声参数 ==========
    # 权重扰动噪声参数
    'gaussian_std': 1.0,
    'laplace_scale': 1.0,
    
    # ========== 保存配置 ==========
    'save_neurons': True,
    'neuron_save_dir': 'neuron/cifar-100/',
    'save_model': False,
    'model_save_path': 'examples/models/resnet50_cifar100_unlearned.pth'
}

# ==================== 设备配置 ====================
torch.manual_seed(1337)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# PyTorch 2.6+ 兼容性：添加安全的全局类
try:
    from torchvision.models.resnet import ResNet
    torch.serialization.add_safe_globals([ResNet])
except:
    pass

_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))


# ==================== 工具函数 ====================
def add_image_noise(images, noise_level):
    """
    对图像添加高斯噪声（适合图像的噪声方式）
    
    Args:
        images: 输入图像张量
        noise_level: 噪声标准差（相对于归一化后的图像）
    """
    if noise_level == 0.0:
        return images
    
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    return noisy_images


def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    """添加高斯噪声到权重"""
    noise = torch.randn(tensor.size(), device=device) * std + mean
    return tensor + noise


def add_laplace_noise(tensor, loc=0.0, scale=1.0):
    """添加拉普拉斯噪声到权重"""
    noise = torch.distributions.laplace.Laplace(loc, scale).sample(tensor.size()).to(device)
    return tensor + noise


def validate_model(model, data_loader, device, target_class=None):
    """验证模型性能"""
    model.eval()
    correct_target = 0
    total_target = 0
    correct_others = 0
    total_others = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                if target_class is not None and labels[i] == target_class:
                    total_target += 1
                    if predicted[i] == labels[i]:
                        correct_target += 1
                else:
                    total_others += 1
                    if predicted[i] == labels[i]:
                        correct_others += 1

    accuracy_target = correct_target / total_target * 100 if total_target > 0 else 0
    accuracy_others = correct_others / total_others * 100 if total_others > 0 else 0

    if target_class is not None:
        print(f'  目标类别 {target_class} 的精确度: {accuracy_target:.2f}%')
    print(f'  其余类别的精确度: {accuracy_others:.2f}%')
    
    return accuracy_target, accuracy_others


def load_data(config):
    """加载CIFAR-100数据集"""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 训练集
    train_dataset = torchvision.datasets.CIFAR100(
        root=config['data_root'], 
        train=True, 
        download=True, 
        transform=transform
    )

    # 测试集
    test_dataset = torchvision.datasets.CIFAR100(
        root=config['data_root'], 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 筛选目标类别（用于遗忘分析）
    target_indices = [i for i, (_, label) in enumerate(test_dataset) 
                     if label == config['unlearn_class']]
    target_subset = Subset(test_dataset, target_indices)

    # 完整测试集用于最终验证
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return target_subset, test_loader


def analyze_neurons(lrp_model, target_dataset, rule, analyze_top_n, sample_size, noise_level, device):
    """
    分析神经元重要性
    
    Args:
        lrp_model: LRP模型
        target_dataset: 目标类别数据集
        rule: LRP规则
        analyze_top_n: 每个样本分析的top神经元数量
        sample_size: 用于分析的样本数量
        noise_level: 图像噪声强度
        device: 设备
    """
    print(f"  分析规则: {rule}")
    print(f"  每样本分析Top-{analyze_top_n}神经元")
    print(f"  分析样本数: {sample_size}")
    print(f"  图像噪声强度: {noise_level}")
    
    # 创建采样的数据加载器
    if sample_size < len(target_dataset):
        # 随机采样
        indices = torch.randperm(len(target_dataset))[:sample_size].tolist()
        sampled_dataset = Subset(target_dataset, indices)
    else:
        sampled_dataset = target_dataset
    
    sample_loader = DataLoader(sampled_dataset, batch_size=1, shuffle=False)
    
    counter = []
    for x, y in sample_loader:
        x, y = x.to(device), y.to(device)
        
        # 添加图像噪声
        x_noisy = add_image_noise(x, noise_level)
        
        # Forward pass
        y_hat_lrp = lrp_model.forward(x_noisy, explain=True, rule=rule)
        y_hat_lrp = y_hat_lrp[torch.arange(x_noisy.shape[0]), y_hat_lrp.max(1)[1]]
        y_hat_lrp = y_hat_lrp.sum()

        # Backward pass
        lrp.trace.enable_and_clean()
        y_hat_lrp.backward()
        all_relevances = lrp.trace.collect_and_disable()

        # 收集top神经元
        for i, t in enumerate(all_relevances):
            t = t[0].tolist()
            top_indices = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:analyze_top_n]
            counter.append(top_indices)
            break

    # 统计神经元出现频率
    all_numbers = [num for sublist in counter for num in sublist]
    number_counts = Counter(all_numbers)
    
    print(f"  分析的神经元总数: {len(number_counts)}")
    
    return number_counts


def perturb_neurons(lrp_model, sorted_neurons, perturb_top_n, perturbation, num_classes, config):
    """扰动神经元"""
    print(f"  扰动Top-{perturb_top_n}神经元，方法: {perturbation}")
    
    # 获取需要扰动的神经元列表
    top_neurons = sorted_neurons[:perturb_top_n]
    
    # 获取第22层的权重（根据ResNet模型结构）
    fc_weights = lrp_model[22].weight.data.clone()
    
    if perturbation == 'zero':
        # 方法1: 直接设置为0
        for class_num in range(num_classes):
            for neuron_idx in top_neurons:
                fc_weights[class_num][neuron_idx] = 0
                
    elif perturbation == 'gaussian':
        # 方法2: 添加高斯噪声
        for class_num in range(num_classes):
            for neuron_idx in top_neurons:
                fc_weights[class_num][neuron_idx] = add_gaussian_noise(
                    fc_weights[class_num][neuron_idx], 
                    mean=0.0, 
                    std=config['gaussian_std']
                )
                
    elif perturbation == 'laplace':
        # 方法3: 添加拉普拉斯噪声
        for class_num in range(num_classes):
            for neuron_idx in top_neurons:
                fc_weights[class_num][neuron_idx] = add_laplace_noise(
                    fc_weights[class_num][neuron_idx], 
                    loc=0.0, 
                    scale=config['laplace_scale']
                )
    
    lrp_model[22].weight.data = fc_weights
    
    return lrp_model


def run_single_experiment(rule, analyze_top_n, perturb_top_n, perturbation, 
                         sample_size, noise_level, base_config, original_model, 
                         target_dataset, test_loader, device):
    """运行单个实验"""
    print("\n" + "="*70)
    print(f"实验配置:")
    print(f"  规则={rule}, 分析Top={analyze_top_n}, 扰动Top={perturb_top_n}")
    print(f"  扰动方法={perturbation}, 分析样本数={sample_size}, 图像噪声={noise_level}")
    print("="*70)
    
    start_time = time.time()
    
    # 转换为LRP模型
    lrp_model = lrp.convert_vgg(original_model).to(device)
    
    # 分析神经元
    neuron_counts = analyze_neurons(
        lrp_model, 
        target_dataset,
        rule, 
        analyze_top_n,
        sample_size,
        noise_level,
        device
    )
    
    # 排序神经元
    sorted_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_neurons = [num for num, _ in sorted_neurons]
    
    print(f"  最终扰动的神经元数量: {min(perturb_top_n, len(sorted_neurons))}")
    
    # 保存神经元列表
    if base_config['save_neurons']:
        pathlib.Path(base_config['neuron_save_dir']).mkdir(parents=True, exist_ok=True)
        save_path = f"{base_config['neuron_save_dir']}class{base_config['unlearn_class']}_" \
                   f"{rule}_ana{analyze_top_n}_per{perturb_top_n}_" \
                   f"{perturbation}_s{sample_size}_n{noise_level}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(sorted_neurons[:perturb_top_n], f)
        print(f"  神经元列表已保存至: {save_path}")
    
    # 扰动神经元
    lrp_model = perturb_neurons(
        lrp_model, 
        sorted_neurons, 
        perturb_top_n, 
        perturbation,
        base_config['num_classes'],
        base_config
    )
    
    end_time = time.time()
    print(f"\n遗忘耗时: {end_time - start_time:.2f}秒")
    
    # 验证模型
    print("\n模型验证结果:")
    validate_model(lrp_model, test_loader, device, target_class=base_config['unlearn_class'])
    
    # 保存模型
    if base_config['save_model']:
        model_save_path = base_config['model_save_path'].replace(
            '.pth', 
            f"_{rule}_{perturb_top_n}_{perturbation}_s{sample_size}_n{noise_level}.pth"
        )
        torch.save(copy.deepcopy(lrp_model), model_save_path)
        print(f"  模型已保存至: {model_save_path}")
    
    return lrp_model


def generate_experiment_combinations(config):
    """
    生成所有实验组合
    
    Returns:
        list: 实验配置列表，每个元素是一个字典
    """
    combinations = list(itertools.product(
        config['rules'],
        config['analyze_top_n_list'],
        config['perturb_top_n_list'],
        config['perturbation_methods'],
        config['analysis_sample_sizes'],
        config['image_noise_levels']
    ))
    
    experiments = []
    for rule, analyze_n, perturb_n, method, sample_size, noise in combinations:
        experiments.append({
            'rule': rule,
            'analyze_top_n': analyze_n,
            'perturb_top_n': perturb_n,
            'perturbation': method,
            'sample_size': sample_size,
            'noise_level': noise
        })
    
    return experiments


def main():
    """主函数"""
    config = EXPERIMENT_CONFIG
    
    # 生成所有实验组合
    experiments = generate_experiment_combinations(config)
    print(f"\n总共需要运行 {len(experiments)} 个实验")
    print("\n实验列表:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. Rule={exp['rule']}, Analyze={exp['analyze_top_n']}, "
              f"Perturb={exp['perturb_top_n']}, Method={exp['perturbation']}, "
              f"Samples={exp['sample_size']}, Noise={exp['noise_level']}")
    
    # 加载数据
    print("\n" + "="*70)
    print("加载CIFAR-100数据集...")
    target_dataset, test_loader = load_data(config)
    print(f"目标类别 {config['unlearn_class']} 样本数: {len(target_dataset)}")
    
    # 加载原始模型
    print(f"加载模型: {config['model_path']}")
    original_model = torch.load(config['model_path'], weights_only=False).to(device)
    original_model.eval()
    
    # 验证原始模型
    print("\n原始模型性能:")
    validate_model(original_model, test_loader, device, target_class=config['unlearn_class'])
    
    # 批量运行实验
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*70}")
        print(f"# 运行实验 {i}/{len(experiments)}")
        print(f"{'#'*70}")
        
        run_single_experiment(
            rule=exp['rule'],
            analyze_top_n=exp['analyze_top_n'],
            perturb_top_n=exp['perturb_top_n'],
            perturbation=exp['perturbation'],
            sample_size=exp['sample_size'],
            noise_level=exp['noise_level'],
            base_config=config,
            original_model=original_model,
            target_dataset=target_dataset,
            test_loader=test_loader,
            device=device
        )
    
    print("\n" + "="*70)
    print("所有实验完成！")
    print("="*70)


if __name__ == '__main__':
    main()