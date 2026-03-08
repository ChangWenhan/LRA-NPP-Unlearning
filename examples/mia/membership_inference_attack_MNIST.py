import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.stats as stats
import os
import sys
import random

# ================= 配置区域 (CONFIG) =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 1
SVM_SEED = 42
SVM_TRAIN_SAMPLES = 1000

# 路径配置
DATA_ROOT = '/home/cwh/Workspace/TorchLRP-master/data'

# 原始模型 (Teacher)
TEACHER_PATH = '/home/cwh/Workspace/TorchLRP-master/examples/models/mnist_model.pth'

# 待测模型列表 (支持批量)
MODEL_GROUP_NAME = "MNIST_Boundary_Unlearning"
MODEL_PATHS = [
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/Rule_epsilon_Samp_36_Noise_0.0_K_50_H_80_Run_0.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/Rule_epsilon_Samp_36_Noise_0.0_K_50_H_80_Run_1.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/Rule_epsilon_Samp_36_Noise_0.0_K_50_H_80_Run_2.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/Rule_epsilon_Samp_36_Noise_0.0_K_50_H_80_Run_3.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/Rule_epsilon_Samp_36_Noise_0.0_K_50_H_80_Run_4.pth'
]

TARGET_CLASS = 1  # 固定目标类别

# ================= 1. 模型定义 =================
def get_mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(14*14*64, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

# ================= 2. 统计学辅助函数 (含人工干预) =================
def calculate_cliff_delta(values, baseline):
    values = np.array(values)
    n = len(values)
    if n == 0: return 0.0
    more = np.sum(values > baseline)
    less = np.sum(values < baseline)
    return (more - less) / n

def calculate_statistics(current_values, baseline_value):
    """
    计算统计指标 (包含人工干预逻辑，防止方差为0时出现NaN)
    """
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    # [人工干预逻辑] 
    if std < 1e-9:
        # 如果非常稳定且接近基准值 -> 完美符合 -> P=1.0
        if abs(mean - baseline_value) < 1e-9:
            p_val = 1.0
        # 如果非常稳定但偏离基准值 -> 显著差异 -> P=0.0
        else:
            p_val = 0.0
    else:
        # 正常情况使用 T 检验
        t_stat, p_val = stats.ttest_1samp(values, baseline_value)
    
    cliffs_d = calculate_cliff_delta(values, baseline_value)
    
    return {
        "mean": mean, 
        "std": std, 
        "p_value": p_val,
        "cliffs_delta": cliffs_d
    }

# ================= 3. 工具函数 =================
def load_model_safely(path, device):
    if not os.path.exists(path):
        print(f"[Error] Path not found: {path}")
        return None
    
    try:
        # 兼容 PyTorch 2.4+
        kwargs = {'map_location': device}
        import inspect
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
            
        checkpoint = torch.load(path, **kwargs)
        model = get_mnist_model()
        
        # 处理 state_dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            
            # 去除 module. 前缀
            new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state, strict=False)
            
        # 处理 Full Model
        elif isinstance(checkpoint, torch.nn.Module):
            try:
                model.load_state_dict(checkpoint.state_dict(), strict=False)
            except:
                model = checkpoint

        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Error] Failed to load {path}: {e}")
        return None

def extract_features(loader, model, target_class, mode='target', max_samples=None):
    """
    提取特征 (SVM输入)。
    使用了 max_samples 参数来加速 SVM 训练数据的准备。
    """
    model.eval()
    probs = []
    collected = 0
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            outputs = model(data)
            probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            target_np = target.numpy()
            
            for i in range(len(target_np)):
                label = target_np[i]
                should_select = False
                
                if mode == 'target' and label == target_class:
                    should_select = True
                elif mode == 'remain' and label != target_class:
                    should_select = True
                    
                if should_select:
                    # 这里提取完整的 10维 概率向量，包含的信息比只取第1列更多
                    probs.append(probabilities[i])
                    collected += 1
                    if max_samples and collected >= max_samples:
                        return np.array(probs)
                        
    if len(probs) == 0: return np.empty((0, 10))
    return np.array(probs)

# ================= 4. 主流程 =================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    # shuffle=True 确保我们提取的前几千个样本是随机分布的
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 训练 SVM (Teacher)
    print("\n--- Phase 1: Training SVM Probe (Teacher) ---")
    teacher_model = load_model_safely(TEACHER_PATH, DEVICE)
    if teacher_model is None: return

    # [关键] 限制样本数量，加速训练
    print(f"Extracting features for SVM (Limit: {SVM_TRAIN_SAMPLES} per class)...")
    
    pos_features = extract_features(train_loader, teacher_model, TARGET_CLASS, mode='target', max_samples=SVM_TRAIN_SAMPLES)
    neg_features = extract_features(train_loader, teacher_model, TARGET_CLASS, mode='remain', max_samples=len(pos_features))
    
    del teacher_model
    torch.cuda.empty_cache()

    if len(pos_features) == 0:
        print("[Error] No target samples found. Check TARGET_CLASS.")
        return

    # 构建数据
    X_train = np.concatenate([pos_features, neg_features], axis=0)
    y_train = np.concatenate([np.ones(len(pos_features)), np.zeros(len(neg_features))], axis=0)
    
    # [关键] 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print(f"Training SVM on {len(X_train)} samples...")
    # probability=True 在小样本下是可以接受的
    svm_clf = svm.SVC(kernel='linear', probability=True, random_state=SVM_SEED)
    svm_clf.fit(X_train, y_train)
    print(f"SVM Training Accuracy: {accuracy_score(y_train, svm_clf.predict(X_train))*100:.2f}%")

    # 3. 批量测试 Unlearned Models
    target_acc_list = []
    forgetting_rate_list = []
    remain_acc_list = []

    print(f"\n--- Phase 2: Evaluating {len(MODEL_PATHS)} Unlearned Models ---")

    for i, path in enumerate(MODEL_PATHS):
        print(f"Processing Run {i+1}...")
        u_model = load_model_safely(path, DEVICE)
        if u_model is None: continue
        
        # (A) Target Accuracy
        # 推理时不需要限制样本数太死，或者可以限制为 1000-2000 以加快速度
        u_target_feats = extract_features(train_loader, u_model, TARGET_CLASS, mode='target', max_samples=2000)
        
        if len(u_target_feats) > 0:
            u_target_feats = scaler.transform(u_target_feats) # 记得 Transform
            t_preds = svm_clf.predict(u_target_feats)
            t_acc = np.mean(t_preds == 1)
        else:
            t_acc = 0.0
            
        f_rate = 1.0 - t_acc # 遗忘率

        # (B) Remaining Accuracy
        u_remain_feats = extract_features(train_loader, u_model, TARGET_CLASS, mode='remain', max_samples=2000)
        
        if len(u_remain_feats) > 0:
            u_remain_feats = scaler.transform(u_remain_feats) # 记得 Transform
            r_preds = svm_clf.predict(u_remain_feats)
            r_acc = np.mean(r_preds == 0) # 0 代表 Non-Member
        else:
            r_acc = 0.0

        target_acc_list.append(t_acc)
        forgetting_rate_list.append(f_rate)
        remain_acc_list.append(r_acc)
        
        print(f"  -> Run {i+1}: Target Acc={t_acc*100:.2f}%, Forgetting Rate={f_rate*100:.2f}%, Remain Acc={r_acc*100:.2f}%")
        del u_model
        torch.cuda.empty_cache()

    # 4. 统计输出
    print("\n" + "="*140)
    print(f"Final Statistics for {MODEL_GROUP_NAME} (Over {len(target_acc_list)} runs)")
    print("="*140)
    
    t_stats = calculate_statistics(target_acc_list, baseline_value=0.0)
    f_stats = calculate_statistics(forgetting_rate_list, baseline_value=1.0)
    r_stats = calculate_statistics(remain_acc_list, baseline_value=1.0)
    
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

if __name__ == "__main__":
    main()