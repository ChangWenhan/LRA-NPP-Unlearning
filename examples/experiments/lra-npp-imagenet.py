import os
import sys
import copy
import time
import torch
import pickle
from torch.nn import Sequential, Conv2d, Linear

import pathlib
import torchvision
from collections import Counter
from torchvision import datasets, transforms as T
import configparser
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
    'unlearn_class': 0,  # ImageNet 类别索引（根据文件夹顺序）
    'batch_size': 1,
    'test_batch_size': 1,
    'num_classes': 1000,  # ImageNet 有 1000 个类别（如果是 mini 版本需要调整）
    
    # ImageNet 数据路径
    'unlearn_data_dir': 'torch_imagenet/imagenet-mini/train/n01734418',  # 遗忘类别的数据
    'test_data_dir': '/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train',  # 完整数据集
    
    # VGG 模型配置
    'vgg_version': 16,  # VGG-16 或 VGG-19
    'use_pretrained': True,
    
    # ========== 独立参数列表（批量实验） ==========
    # LRP传播规则列表
    'rules': ['alpha1beta0', 'epsilon', 'gamma+epsilon', 'alpha2beta1'],
    
    # 每个样本分析的神经元数量列表
    'analyze_top_n_list': [150],
    
    # 最终扰动的神经元数量列表
    'perturb_top_n_list': [300, 400, 500],
    
    # 扰动方法列表: 'zero'(置零), 'gaussian'(高斯噪声), 'laplace'(拉普拉斯噪声)
    'perturbation_methods': ['zero'],
    
    # 分析神经元时的数据集大小（从目标类别中采样的样本数）
    'analysis_sample_sizes': [100],  # None 表示使用全部数据
    
    # 分析时对图像添加噪声的强度列表（标准差，0表示不加噪）
    'image_noise_levels': [0.0],  # 例如: [0.0, 0.01, 0.05, 0.1]
    
    # ========== VGG 层索引配置 ==========
    # VGG 模型结构: classifier[0]是第一个全连接层, classifier[3]是第二个, classifier[6]是输出层
    'fc1_layer_idx': 36,  # 第一个全连接层在 Sequential 中的索引 (4096 neurons)
    'fc2_layer_idx': 39,  # 第二个全连接层在 Sequential 中的索引 (4096 neurons)
    'output_layer_idx': 42,  # 输出层索引
    
    # 选择扰动哪一层（或两层都扰动）
    'perturb_fc1': True,   # 是否扰动第一个全连接层
    'perturb_fc2': True,   # 是否扰动第二个全连接层
    'perturb_output': False,  # 是否扰动输出层（通常不建议）
    
    # ========== 噪声参数 ==========
    'gaussian_std': 1.0,
    'laplace_scale': 1.0,
    
    # ========== Pattern 配置 ==========
    'use_patterns': True,
    'patterns_dir': 'examples/patterns/',
    
    # ========== 保存配置 ==========
    'save_neurons': True,
    'neuron_save_dir': 'neuron/imagenet/',
    'save_model': False,
    'model_save_path': 'examples/models/vgg_imagenet_unlearned.pth',
}

# ==================== 设备配置 ====================
torch.manual_seed(1337)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# PyTorch 2.6+ 兼容性
try:
    from torchvision.models.vgg import VGG
    torch.serialization.add_safe_globals([VGG])
except:
    pass

_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))


# ==================== 工具函数 ====================
def unnormalize(x):
    """反归一化图像"""
    return x * _std + _mean


def add_image_noise(images, noise_level):
    """对图像添加高斯噪声"""
    if noise_level == 0.0:
        return images
    noise = torch.randn_like(images) * noise_level
    return images + noise


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
    """加载ImageNet数据"""
    # 导入 ImageNet 数据集类
    imagenet_config = configparser.ConfigParser()
    imagenet_config.read((base_path / 'config.ini').as_posix())
    sys.path.append(imagenet_config['DEFAULT']['ImageNetDir'])
    from torch_imagenet import ImageNetDataset
    
    # 数据转换
    transform = T.Compose([
        T.Resize(256), 
        T.CenterCrop(224), 
        T.ToTensor(),
        T.Normalize(mean=_mean.flatten(), std=_std.flatten()),
    ])
    
    # 遗忘类别数据（单一类别文件夹）
    unlearn_dataset = ImageNetDataset(
        root_dir=config['unlearn_data_dir'], 
        transform=transform
    )
    unlearn_loader = torch.utils.data.DataLoader(
        unlearn_dataset, 
        batch_size=1, 
        shuffle=True
    )
    
    # 完整测试数据集
    test_dataset = datasets.ImageFolder(
        root=config['test_data_dir'],
        loader=datasets.folder.default_loader,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False
    )
    
    print(f"  遗忘类别数据: {len(unlearn_dataset)} 张")
    print(f"  测试数据: {len(test_dataset)} 张")
    
    return unlearn_loader, test_loader


def load_vgg_model(config):
    """加载VGG模型"""
    vgg_version = config['vgg_version']
    vgg = getattr(torchvision.models, f"vgg{vgg_version}")(
        pretrained=config['use_pretrained']
    ).to(device)
    vgg.eval()
    
    print(f"  加载 VGG-{vgg_version} 模型")
    
    return vgg


def load_patterns(config, lrp_model, train_loader):
    """加载或生成 PatternNet patterns"""
    if not config['use_patterns']:
        return None
    
    patterns_dir = pathlib.Path(base_path / config['patterns_dir'])
    patterns_dir.mkdir(parents=True, exist_ok=True)
    
    patterns_path = (patterns_dir / f"vgg{config['vgg_version']}_pattern_pos.pkl").as_posix()
    
    if not os.path.exists(patterns_path):
        print("  生成 PatternNet positive patterns...")
        patterns = fit_patternnet_positive(lrp_model, train_loader, device=device)
        store_patterns(patterns_path, patterns)
    else:
        patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]
    
    print("  Patterns 已加载")
    
    return patterns


def analyze_neurons(lrp_model, unlearn_loader, rule, analyze_top_n, sample_size, 
                   noise_level, pattern, device):
    """
    分析神经元重要性
    """
    print(f"  分析规则: {rule}")
    print(f"  每样本分析Top-{analyze_top_n}神经元")
    print(f"  分析样本数: {sample_size if sample_size else '全部'}")
    print(f"  图像噪声强度: {noise_level}")
    
    counter = []
    samples_processed = 0
    
    for x, y in unlearn_loader:
        if sample_size and samples_processed >= sample_size:
            break
        
        x, y = x.to(device), y.to(device)
        
        # 添加图像噪声
        x_noisy = add_image_noise(x, noise_level)
        
        # Forward pass
        y_hat_lrp = lrp_model.forward(x_noisy, explain=True, rule=rule, pattern=pattern)
        y_hat_lrp = y_hat_lrp[torch.arange(x_noisy.shape[0]), y_hat_lrp.max(1)[1]]
        y_hat_lrp = y_hat_lrp.sum()
        
        # Backward pass
        lrp.trace.enable_and_clean()
        y_hat_lrp.backward()
        all_relevances = lrp.trace.collect_and_disable()
        
        # 收集 top 神经元
        for i, t in enumerate(all_relevances):
            t = t[0].tolist()
            top_indices = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:analyze_top_n]
            counter.append(top_indices)
            break
        
        samples_processed += 1
    
    # 统计神经元出现频率
    all_numbers = [num for sublist in counter for num in sublist]
    number_counts = Counter(all_numbers)
    
    print(f"  分析的神经元总数: {len(number_counts)}")
    
    return number_counts


def perturb_neurons(lrp_model, sorted_neurons, perturb_top_n, perturbation, config):
    """扰动神经元"""
    print(f"  扰动Top-{perturb_top_n}神经元，方法: {perturbation}")
    
    top_neurons = sorted_neurons[:perturb_top_n]
    
    # VGG 结构说明：
    # lrp_model[36] = Linear(25088, 4096)  # fc1
    # lrp_model[39] = Linear(4096, 4096)   # fc2
    # lrp_model[42] = Linear(4096, 1000)   # output
    
    modified_layers = []
    
    # 扰动第一个全连接层 (fc1)
    if config['perturb_fc1']:
        fc1_weights = lrp_model[config['fc1_layer_idx']].weight.data.clone()
        
        if perturbation == 'zero':
            for neuron_idx in top_neurons:
                fc1_weights[neuron_idx] = torch.zeros_like(fc1_weights[neuron_idx])
        elif perturbation == 'gaussian':
            for neuron_idx in top_neurons:
                fc1_weights[neuron_idx] = add_gaussian_noise(
                    fc1_weights[neuron_idx], 
                    mean=0.0, 
                    std=config['gaussian_std']
                )
        elif perturbation == 'laplace':
            for neuron_idx in top_neurons:
                fc1_weights[neuron_idx] = add_laplace_noise(
                    fc1_weights[neuron_idx], 
                    loc=0.0, 
                    scale=config['laplace_scale']
                )
        
        lrp_model[config['fc1_layer_idx']].weight.data = fc1_weights
        modified_layers.append('fc1')
    
    # 扰动第二个全连接层 (fc2)
    if config['perturb_fc2']:
        fc2_weights = lrp_model[config['fc2_layer_idx']].weight.data.clone()
        
        # 注意：这里扰动的是输入到该层的神经元对应的权重列
        # 对于目标类别（假设为类别0），扰动其权重
        if perturbation == 'zero':
            # 方法1：扰动所有类别的相关权重
            for neuron_idx in top_neurons:
                fc2_weights[config['unlearn_class'], neuron_idx] = 0
        elif perturbation == 'gaussian':
            for neuron_idx in top_neurons:
                fc2_weights[:, neuron_idx] = add_gaussian_noise(
                    fc2_weights[:, neuron_idx],
                    mean=0.0,
                    std=config['gaussian_std']
                )
        elif perturbation == 'laplace':
            for neuron_idx in top_neurons:
                fc2_weights[:, neuron_idx] = add_laplace_noise(
                    fc2_weights[:, neuron_idx],
                    loc=0.0,
                    scale=config['laplace_scale']
                )
        
        lrp_model[config['fc2_layer_idx']].weight.data = fc2_weights
        modified_layers.append('fc2')

    # 扰动输出层（通常不建议）
    if config['perturb_output']:
        output_weights = lrp_model[config['output_layer_idx']].weight.data.clone()
        
        # 只扰动目标类别的权重
        target_class_weights = output_weights[config['unlearn_class']].clone()
        
        if perturbation == 'zero':
            for neuron_idx in top_neurons:
                target_class_weights[neuron_idx] = 0
        elif perturbation == 'gaussian':
            for neuron_idx in top_neurons:
                target_class_weights[neuron_idx] = add_gaussian_noise(
                    target_class_weights[neuron_idx],
                    mean=0.0,
                    std=config['gaussian_std']
                )
        elif perturbation == 'laplace':
            for neuron_idx in top_neurons:
                target_class_weights[neuron_idx] = add_laplace_noise(
                    target_class_weights[neuron_idx],
                    loc=0.0,
                    scale=config['laplace_scale']
                )
        
        output_weights[config['unlearn_class']] = target_class_weights
        lrp_model[config['output_layer_idx']].weight.data = output_weights
        modified_layers.append('output')
    
    print(f"  已修改层: {', '.join(modified_layers)}")
    
    return lrp_model


def run_single_experiment(rule, analyze_top_n, perturb_top_n, perturbation,
                         sample_size, noise_level, base_config, original_vgg,
                         unlearn_loader, test_loader, pattern, device):
    """运行单个实验"""
    print("\n" + "="*70)
    print(f"实验配置:")
    print(f"  规则={rule}, 分析Top={analyze_top_n}, 扰动Top={perturb_top_n}")
    print(f"  扰动方法={perturbation}, 分析样本数={sample_size}, 图像噪声={noise_level}")
    print("="*70)
    
    start_time = time.time()
    
    # 转换为 LRP 模型
    lrp_model = lrp.convert_vgg(original_vgg).to(device)
    
    # 验证转换后模型一致性
    for x, y in unlearn_loader:
        x = x.to(device)
        y_hat = original_vgg(x)
        y_hat_lrp = lrp_model.forward(x)
        assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), \
            "LRP模型转换后输出不一致"
        break
    
    # 分析神经元
    neuron_counts = analyze_neurons(
        lrp_model,
        unlearn_loader,
        rule,
        analyze_top_n,
        sample_size,
        noise_level,
        pattern,
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
                   f"vgg{base_config['vgg_version']}_{rule}_ana{analyze_top_n}_per{perturb_top_n}_" \
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
        base_config
    )
    
    end_time = time.time()
    print(f"\n遗忘耗时: {end_time - start_time:.2f}秒")
    
    # 验证模型
    print("\n模型验证结果:")
    target_acc, other_acc = validate_model(
        lrp_model, test_loader, device,
        target_class=base_config['unlearn_class']
    )
    
    # 保存模型
    if base_config['save_model']:
        model_save_path = base_config['model_save_path'].replace(
            '.pth',
            f"_vgg{base_config['vgg_version']}_{rule}_{perturb_top_n}_{perturbation}_s{sample_size}_n{noise_level}.pth"
        )
        torch.save(copy.deepcopy(lrp_model), model_save_path)
        print(f"  模型已保存至: {model_save_path}")
    
    results = {
        'target_acc': target_acc,
        'other_acc': other_acc,
    }
    
    return lrp_model, results


def generate_experiment_combinations(config):
    """生成所有实验组合"""
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
    print(f"VGG版本: VGG-{config['vgg_version']}")
    print("\n实验列表:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. Rule={exp['rule']}, Analyze={exp['analyze_top_n']}, "
              f"Perturb={exp['perturb_top_n']}, Method={exp['perturbation']}, "
              f"Samples={exp['sample_size']}, Noise={exp['noise_level']}")
    
    # 加载数据
    print("\n" + "="*70)
    print("加载ImageNet数据...")
    unlearn_loader, test_loader = load_data(config)
    
    # 加载VGG模型
    print("加载VGG模型...")
    original_vgg = load_vgg_model(config)
    
    # 验证原始模型（可选）
    print("\n原始模型性能:")
    validate_model(original_vgg, test_loader, device, 
                  target_class=config['unlearn_class'])
    
    # 转换为LRP模型并加载patterns
    print("\n准备LRP和Patterns...")
    lrp_model_temp = lrp.convert_vgg(original_vgg).to(device)
    patterns = load_patterns(config, lrp_model_temp, unlearn_loader)
    
    # 批量运行实验
    all_results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*70}")
        print(f"# 运行实验 {i}/{len(experiments)}")
        print(f"{'#'*70}")
        
        unlearned_model, results = run_single_experiment(
            rule=exp['rule'],
            analyze_top_n=exp['analyze_top_n'],
            perturb_top_n=exp['perturb_top_n'],
            perturbation=exp['perturbation'],
            sample_size=exp['sample_size'],
            noise_level=exp['noise_level'],
            base_config=config,
            original_vgg=original_vgg,
            unlearn_loader=unlearn_loader,
            test_loader=test_loader,
            pattern=patterns,
            device=device
        )
        
        all_results.append({
            'config': exp,
            'results': results
        })
    
    print("\n" + "="*70)
    print("所有实验完成！")
    print("="*70)
    
    # 打印汇总结果
    print("\n实验结果汇总:")
    print("-" * 70)
    for i, item in enumerate(all_results, 1):
        exp = item['config']
        res = item['results']
        print(f"{i}. Rule={exp['rule']}, Perturb={exp['perturb_top_n']}, "
              f"Method={exp['perturbation']}")
        print(f"   目标类准确率: {res['target_acc']:.2f}%, "
              f"其他类准确率: {res['other_acc']:.2f}%")


if __name__ == '__main__':
    main()