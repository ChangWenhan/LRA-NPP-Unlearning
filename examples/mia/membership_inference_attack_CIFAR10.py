import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os
import sys
import scipy.stats as stats  # 引入统计库

# 获取上上级目录 (保持原有的路径逻辑)
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)
# import lrp # 如果不需要 lrp 功能，可以注释掉以减少依赖

# ================= 配置区域 (CONFIG) =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 100
NUM_WORKERS = 4
SVM_SEED = 42

# 路径配置
DATA_ROOT = '/home/cwh/Workspace/TorchLRP-master/data'

# 原始模型 (Teacher/Oracle) - 用于训练 SVM
TEACHER_PATH = 'examples/models/resnet50_cifar10_epoch_10.pth'

# 待测模型列表 (支持批量)
MODEL_GROUP_NAME = "ResNet50_CIFAR10_Boundary_Unlearning"
MODEL_PATHS = [
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/c10_epsilon_400_zero_s36_n0.0_1.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/c10_epsilon_400_zero_s36_n0.0_2.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/c10_epsilon_400_zero_s36_n0.0_3.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/c10_epsilon_400_zero_s36_n0.0_4.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/c10_epsilon_400_zero_s36_n0.0_5.pth',
]

TARGET_CLASS_LABEL = 9 # 遗忘目标类别

# ================= 统计学辅助函数 (标准模板) =================
def calculate_cliff_delta(values, baseline):
    """
    通用 Cliff's Delta 计算.
    - Target Acc (Ref=0): 结果通常为正 [0, 1]。
    - Forgetting Rate (Ref=1) & Remaining Acc (Ref=1): 结果通常为负 [-1, 0]。
    """
    values = np.array(values)
    n = len(values)
    if n == 0: return 0.0
    
    more = np.sum(values > baseline)
    less = np.sum(values < baseline)
    
    return (more - less) / n

def calculate_statistics(current_values, baseline_value):
    """计算 Mean, Std, P-value, Cliff's Delta"""
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    # P-Value (One-sample T-test)
    if std < 1e-9:
        p_val = 0.0 if abs(mean - baseline_value) > 1e-5 else 1.0
    else:
        t_stat, p_val = stats.ttest_1samp(values, baseline_value)
    
    # Cliff's delta
    cliffs_d = calculate_cliff_delta(values, baseline_value)
    
    return {
        "mean": mean, 
        "std": std, 
        "p_value": p_val,
        "cliffs_delta": cliffs_d
    }

# ================= 模型加载辅助函数 (鲁棒性增强) =================
def load_model_safely(path, device):
    """
    安全加载模型，处理 Full Model / State Dict 以及 PyTorch 版本差异
    """
    if not os.path.exists(path):
        print(f"[Error] Path not found: {path}")
        return None
    
    print(f"Loading model from {path}...")
    try:
        # 尝试标准加载 (兼容旧代码的 Full Model 保存方式)
        # 如果是 PyTorch 2.4+，weights_only 默认为 True，这会导致加载 Full Model 失败
        # 所以我们显式指定 weights_only=False (但在旧版本 PyTorch 中这个参数不存在，需用 kwargs hack)
        kwargs = {'map_location': device}
        import inspect
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
            
        model = torch.load(path, **kwargs)
        
        # 如果加载出来的是字典 (State Dict)，则需要实例化架构
        if isinstance(model, dict):
            print("  -> Detected state_dict, initializing ResNet50...")
            # 注意：这里假设是 ResNet50，如果是其他架构需要修改
            # 如果原代码使用了自定义的 ResNet，请确保在此处 import 正确的类
            base_model = torchvision.models.resnet50(pretrained=False)
            base_model.fc = nn.Linear(base_model.fc.in_features, 10) # CIFAR-10 has 10 classes
            base_model.load_state_dict(model, strict=False)
            model = base_model

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return None

# ================= 特征提取逻辑 =================
def extract_posteriors(loader, model, target_class=None, mode='target', max_samples=None):
    """
    通用特征提取函数
    mode='target': 只提取 target_class 的样本
    mode='remain': 提取除 target_class 以外的样本
    """
    model.eval()
    posteriors = []
    labels = [] # 这里只用于计数，不做返回
    collected_count = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            targets_np = targets.numpy()
            
            for i in range(len(targets_np)):
                current_label = targets_np[i]
                
                should_select = False
                if mode == 'target' and current_label == target_class:
                    should_select = True
                elif mode == 'remain' and current_label != target_class:
                    should_select = True
                
                if should_select:
                    posteriors.append(probs[i])
                    collected_count += 1
                    
                    if max_samples and collected_count >= max_samples:
                        return np.array(posteriors)
                        
    if len(posteriors) == 0:
        return np.empty((0, 10)) # 假设 10 类
        
    return np.array(posteriors)

# ================= 主流程 =================
def main():
    # 0.设置随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. 加载数据
    print("Preparing CIFAR-10 Dataset...")
    transform = T.Compose([
        T.Resize((224, 224)),  # ResNet50 standard input
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 为了效率，SVM 训练不一定要用全量测试集，但这里保持原逻辑使用 trainloader
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # 2. 准备 Teacher Model 并训练 SVM
    print("\n--- Phase 1: Training SVM Probe (Teacher) ---")
    teacher_model = load_model_safely(TEACHER_PATH, DEVICE)
    if teacher_model is None: return

    # 提取 Target Class (作为 SVM 的 Label 1)
    print(f"Extracting Target Class {TARGET_CLASS_LABEL} samples...")
    pos_features = extract_posteriors(trainloader, teacher_model, target_class=TARGET_CLASS_LABEL, mode='target')
    
    # 提取 Non-Target Class (作为 SVM 的 Label 0)，数量保持平衡
    print(f"Extracting Non-Target samples (Balanced: {len(pos_features)})...")
    neg_features = extract_posteriors(trainloader, teacher_model, target_class=TARGET_CLASS_LABEL, mode='remain', max_samples=len(pos_features))
    
    del teacher_model
    torch.cuda.empty_cache() # 清理显存

    # 构建 SVM 数据集
    X_train = np.concatenate((pos_features, neg_features), axis=0)
    y_train = np.concatenate((np.ones(len(pos_features)), np.zeros(len(neg_features))), axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print(f"Training SVM on {len(X_train)} samples...")
    svm_model = SVC(kernel='linear', probability=True, random_state=SVM_SEED)
    svm_model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, svm_model.predict(X_train))
    print(f"SVM Training Accuracy: {train_acc * 100:.2f}%")

    # 3. 批量评估 Unlearned Models
    target_acc_list = []
    forgetting_rate_list = []
    remain_acc_list = []

    print(f"\n--- Phase 2: Evaluating {len(MODEL_PATHS)} Unlearned Models ---")
    
    for i, path in enumerate(MODEL_PATHS):
        print(f"Processing Run {i+1}...")
        u_model = load_model_safely(path, DEVICE)
        if u_model is None: continue

        # [Metric 1] Target Accuracy & Forgetting Rate
        # 提取 Unlearned 模型对 "原目标类" 的特征
        # 理想情况：模型不认识它了 -> SVM 把它判为 0 -> Target Acc 低 -> Forgetting Rate 高
        u_target_feats = extract_posteriors(trainloader, u_model, target_class=TARGET_CLASS_LABEL, mode='target')
        
        if len(u_target_feats) > 0:
            t_preds = svm_model.predict(u_target_feats)
            t_acc = np.mean(t_preds == 1) # 预测为 "Target Member" (1) 的比例
        else:
            t_acc = 0.0
        
        f_rate = 1.0 - t_acc # 遗忘率

        # [Metric 2] Remaining Accuracy
        # 提取 Unlearned 模型对 "其他类" 的特征 (限制数量以加快速度，如 1000 个)
        # 理想情况：模型还认识它们 -> SVM 把它判为 0 (非 Target) -> Remain Acc 高 (预测为 0 的比例)
        u_remain_feats = extract_posteriors(trainloader, u_model, target_class=TARGET_CLASS_LABEL, mode='remain', max_samples=1000)
        
        if len(u_remain_feats) > 0:
            r_preds = svm_model.predict(u_remain_feats)
            r_acc = np.mean(r_preds == 0) # 预测为 "Non-Target / Other" (0) 的比例
        else:
            r_acc = 0.0

        target_acc_list.append(t_acc)
        forgetting_rate_list.append(f_rate)
        remain_acc_list.append(r_acc)
        
        print(f"  -> Run {i+1}: Target Acc={t_acc*100:.2f}%, Forgetting Rate={f_rate*100:.2f}%, Remain Acc={r_acc*100:.2f}%")
        
        del u_model
        torch.cuda.empty_cache()

    # 4. 统计结果输出
    print("\n" + "="*140)
    print(f"Final Statistics for {MODEL_GROUP_NAME} (Over {len(target_acc_list)} runs)")
    print("="*140)

    t_stats = calculate_statistics(target_acc_list, baseline_value=0.0)
    f_stats = calculate_statistics(forgetting_rate_list, baseline_value=1.0)
    r_stats = calculate_statistics(remain_acc_list, baseline_value=1.0)

    # 打印表头
    print(f"{'Metric':<20} | {'Mean ± Std':<25} | {'Ref Value':<10} | {'P-value':<12} | {'Cliff Delta'}")
    print("-" * 140)

    def format_row(name, stats_dict, ref_val):
        mean = stats_dict['mean'] * 100
        std = stats_dict['std'] * 100
        pval = stats_dict['p_value']
        delta = stats_dict['cliffs_delta']
        
        ms_str = f"{mean:.2f} ± {std:.2f} %"
        p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        print(f"{name:<20} | {ms_str:<25} | {str(ref_val):<10} | {p_str:<12} | {delta:.4f}")

    format_row("Target Acc", t_stats, 0.0)
    format_row("Forgetting Rate", f_stats, 1.0)
    format_row("Remaining Acc", r_stats, 1.0)

    print("="*140)
    print("\nInterpretation Guide:")
    print("[Target Acc (Ref=0.0)]")
    print("  - Definition: Probability of SVM identifying target class samples as 'Target'.")
    print("  - Goal: Closer to 0% is better (Unlearned).")
    
    print("\n[Forgetting Rate (Ref=1.0)]")
    print("  - Definition: 1 - Target Accuracy.")
    print("  - Goal: Closer to 100% is better.")
    print("  - Cliff Delta: Values closer to 0 (from -1) indicate better unlearning performance.")
    
    print("\n[Remaining Acc (Ref=1.0)]")
    print("  - Definition: Probability of SVM identifying non-target class samples as 'Non-Target'.")
    print("  - Goal: Closer to 100% is better (Retained).")

if __name__ == "__main__":
    main()