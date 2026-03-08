import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import scipy.stats as stats

# ================= 配置区域 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train"
TARGET_CLASS_IDX = 56
NUM_CLASSES_TO_LOAD = 100 

# 5 次 Run 的模型路径
MODEL_GROUP_NAME = "VGG_Epsilon_Unlearning"
MODEL_PATHS = [
    "/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vgg_epsilon_150_400_run1.pth",
    "/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vgg_epsilon_150_400_run2.pth",
    "/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vgg_epsilon_150_400_run3.pth",
    "/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vgg_epsilon_150_400_run4.pth",
    "/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vgg_epsilon_150_400_run5.pth"
]

SVM_SEED = 42
# ===========================================

# --- 统计学辅助函数 ---
def calculate_cliff_delta(values, baseline):
    """
    通用 Cliff's Delta 计算.
    公式: ( #(x > baseline) - #(x < baseline) ) / n
    
    - 对于 Target Acc (Baseline=0.0):
      由于 Acc >= 0, 'less' 为 0。结果为正 [0, 1]。
      
    - 对于 Remaining Acc (Baseline=1.0) 和 Forgetting Rate (Baseline=1.0):
      由于通常 <= 1, 'more' 为 0。结果为负 [-1, 0]。
    """
    values = np.array(values)
    n = len(values)
    if n == 0: return 0.0
    
    more = np.sum(values > baseline)
    less = np.sum(values < baseline)
    
    return (more - less) / n

def calculate_statistics(current_values, baseline_value):
    """计算统计指标: Mean, Std, P-value, Cliff's delta"""
    values = np.array(current_values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    
    # P-Value (One-sample T-test)
    if std < 1e-9:
        # 如果方差极小，直接判断均值是否接近 baseline
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

# --- 数据与模型辅助函数 (保持不变) ---
def load_data(data_path, target_class_idx, num_classes=10):
    print(f"Loading data from {data_path}...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
    except Exception as e:
        print(f"[Error] Could not load dataset from {data_path}. Error: {e}")
        return None

    selected_classes = list(range(num_classes))
    if target_class_idx not in selected_classes:
        selected_classes.append(target_class_idx)
    subset_indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Data loaded. Total samples: {len(subset_indices)}.")
    return dataloader

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs) 
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

def load_model_safely(path, device):
    if not os.path.exists(path):
        print(f"[Error] Path not found: {path}")
        return None
    try:
        checkpoint = torch.load(path, map_location=device)
        model = models.vgg16(pretrained=False)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        
        # Key Remapping Logic (Same as before)
        new_state_dict = {}
        has_features_prefix = any(k.startswith("features.") for k in state_dict.keys())
        
        if not has_features_prefix:
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                parts = name.split('.')
                if parts[0].isdigit():
                    layer_idx = int(parts[0])
                    param_type = parts[1]
                    if layer_idx <= 30: 
                        new_key = f"features.{layer_idx}.{param_type}"
                        new_state_dict[new_key] = v
                    else: 
                        classifier_map = {33: 0, 36: 3, 39: 6}
                        if layer_idx in classifier_map:
                            new_idx = classifier_map[layer_idx]
                            new_key = f"classifier.{new_idx}.{param_type}"
                            new_state_dict[new_key] = v
                else:
                    new_state_dict[name] = v
        else:
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(new_state_dict, strict=False) 
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Error] Failed to load {path}: {e}")
        return None

def main():
    # 1. 准备数据
    dataloader = load_data(DATA_PATH, TARGET_CLASS_IDX, NUM_CLASSES_TO_LOAD)
    if dataloader is None: return
    
    # 2. 准备 SVM (只做一次)
    print("\n--- Training SVM Probe (Teacher) ---")
    teacher_model = models.vgg16(pretrained=True).to(DEVICE)
    t_features, t_labels = extract_features(teacher_model, dataloader, DEVICE)
    del teacher_model 
    
    # 均衡采样训练 SVM
    full_svm_labels = (t_labels == TARGET_CLASS_IDX).astype(int)
    pos_indices = np.where(full_svm_labels == 1)[0]
    neg_indices = np.where(full_svm_labels == 0)[0]
    n_samples = len(pos_indices)
    
    np.random.seed(SVM_SEED) 
    if len(neg_indices) > n_samples:
        selected_neg_indices = np.random.choice(neg_indices, size=n_samples, replace=False)
    else:
        selected_neg_indices = neg_indices
        
    train_indices = np.concatenate([pos_indices, selected_neg_indices])
    np.random.shuffle(train_indices)
    
    X_train = t_features[train_indices]
    y_train = full_svm_labels[train_indices]
    
    svm = SVC(probability=True, kernel='linear', random_state=SVM_SEED)
    svm.fit(X_train, y_train)
    print(f"SVM Trained on {len(X_train)} balanced samples.")

    # 3. 循环测试所有 Run
    target_acc_list = []
    remain_acc_list = []
    forgetting_rate_list = [] # <--- 新增：存储遗忘率
    
    print(f"\n--- Evaluating {len(MODEL_PATHS)} Runs for {MODEL_GROUP_NAME} ---")
    
    for i, path in enumerate(MODEL_PATHS):
        print(f"Processing Run {i+1}...")
        u_model = load_model_safely(path, DEVICE)
        
        if u_model is None:
            continue
            
        u_features, u_labels = extract_features(u_model, dataloader, DEVICE)
        
        # 预测
        svm_preds = svm.predict(u_features)
        true_binary_labels = (u_labels == TARGET_CLASS_IDX).astype(int)
        
        # Target Acc
        target_mask = (true_binary_labels == 1)
        t_acc = accuracy_score(true_binary_labels[target_mask], svm_preds[target_mask]) if np.sum(target_mask) > 0 else 0.0
        
        # Remaining Acc
        remain_mask = (true_binary_labels == 0)
        r_acc = accuracy_score(true_binary_labels[remain_mask], svm_preds[remain_mask]) if np.sum(remain_mask) > 0 else 0.0
        
        # Forgetting Rate (1 - Target Acc)
        f_rate = 1.0 - t_acc # <--- 计算遗忘率
        
        target_acc_list.append(t_acc)
        remain_acc_list.append(r_acc)
        forgetting_rate_list.append(f_rate) # <--- 存储
        
        print(f"  -> Run {i+1}: Target Acc={t_acc*100:.2f}%, Forgetting Rate={f_rate*100:.2f}%, Remain Acc={r_acc*100:.2f}%")
        del u_model 

    # 4. 计算并打印完整统计结果
    print("\n" + "="*140)
    print(f"Final Statistics for {MODEL_GROUP_NAME} (Over {len(target_acc_list)} runs)")
    print("="*140)
    
    # 1. Target Statistics (Baseline = 0.0) -> 希望接近 0
    t_stats = calculate_statistics(target_acc_list, baseline_value=0.0)
    
    # 2. Forgetting Rate Statistics (Baseline = 1.0) -> 希望接近 1
    f_stats = calculate_statistics(forgetting_rate_list, baseline_value=1.0)

    # 3. Remaining Statistics (Baseline = 1.0) -> 希望接近 1
    r_stats = calculate_statistics(remain_acc_list, baseline_value=1.0)
    
    # 表头
    print(f"{'Metric':<20} | {'Mean ± Std':<25} | {'Ref Value':<10} | {'P-value':<12} | {'Cliff Delta'}")
    print("-" * 140)
    
    # 格式化打印函数
    def format_row(name, stats_dict, ref_val):
        mean = stats_dict['mean'] * 100
        std = stats_dict['std'] * 100
        pval = stats_dict['p_value']
        delta = stats_dict['cliffs_delta']
        
        ms_str = f"{mean:.2f} ± {std:.2f} %"
        p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.4f}"
        
        print(f"{name:<20} | {ms_str:<25} | {str(ref_val):<10} | {p_str:<12} | {delta:.4f}")

    # 打印三行
    format_row("Target Acc", t_stats, 0.0)
    format_row("Forgetting Rate", f_stats, 1.0) # <--- 打印新增指标
    format_row("Remaining Acc", r_stats, 1.0)
    
    print("="*140)
    print("\nInterpretation Guide:")
    print("[Target Acc (Ref=0.0)]")
    print("  - Mean: Lower is better (closer to 0%).")
    print("  - Cliff Delta: Positive [0, 1]. Closer to 0 is better.")
    
    print("\n[Forgetting Rate (Ref=1.0)]") # <--- 新增解释
    print("  - Definition: 1 - Target Accuracy.")
    print("  - Mean: Higher is better (closer to 100%).")
    print("  - Cliff Delta: Negative [-1, 0]. Closer to 0 means ideal unlearning (close to baseline 1).")
    print("                 -1.0 means forgetting rate is consistently < 1 (Unlearning failed).")
    
    print("\n[Remaining Acc (Ref=1.0)]")
    print("  - Mean: Higher is better (closer to 100%).")
    print("  - Cliff Delta: Negative [-1, 0]. Closer to 0 is better.")

if __name__ == "__main__":
    main()