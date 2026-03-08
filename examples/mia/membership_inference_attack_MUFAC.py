#!/usr/bin/env python3
# mia_vit_mufac_batch_stats.py
"""
Membership Inference Attack (MIA) with Batch Statistics for ViT on MUFAC.
Includes Forgetting Rate, Cliff's Delta, and P-value calculations.
"""

import os
import sys
import time
import random
import pathlib
import inspect
from collections import Counter
import collections

import numpy as np
import pandas as pd
from PIL import Image
import scipy.stats as stats # 新增统计库

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vision_transformer
from torchvision.models.vision_transformer import VisionTransformer

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ================= 配置区域 (CONFIG) =================
DATA_ROOT = '/home/cwh/Workspace/TorchLRP-master/data/custom_korean_family_dataset_resolution_128'
TRAIN_CSV = os.path.join(DATA_ROOT, 'custom_train_dataset.csv')
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train_images')

# 原始模型 (Teacher/Oracle)
ORIGINAL_CHECKPOINT = '/home/cwh/Workspace/TorchLRP-master/examples/models/vit_best_on_test.pth'

# 批量待测模型路径 (填入你的多个 Run)
MODEL_GROUP_NAME = "ViT_Mufac_Unlearning_Batch"
MODEL_PATHS = [
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vit_S36_N0.0_K400_H600_run1.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vit_S36_N0.0_K400_H600_run2.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vit_S36_N0.0_K400_H600_run3.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vit_S36_N0.0_K400_H600_run4.pth',
    '/home/cwh/Workspace/TorchLRP-master/examples/models/models_mia/vit_S36_N0.0_K400_H600_run5.pth',
    # 如果只有一个模型，也可以只写一个
]

NUM_CLASSES = 8
UNLEARN_CLASS = 0   # 要检测的目标类索引
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_WORKERS = 4
SVM_SEED = 42

# ---------------- Reproducibility ----------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================= 统计学辅助函数 (新增) =================
def calculate_cliff_delta(values, baseline):
    """
    通用 Cliff's Delta 计算.
    - Target Acc (Ref=0): 结果为正 [0, 1]。
    - Forgetting Rate (Ref=1): 结果为负 [-1, 0]。
    - Remaining Acc (Ref=1): 结果为负 [-1, 0]。
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

# ================= 数据集与加载器 (保持不变) =================
def parsing(meta_data: pd.DataFrame):
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
        self.age_class_to_label = { "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7 }
        self.image_age_list = []
        for img_path, age_cls in full_list:
            if isinstance(age_cls, str) and age_cls in self.age_class_to_label:
                label = self.age_class_to_label[age_cls]
            else:
                try:
                    label = int(age_cls)
                except: continue
            if filter_class is not None and label == filter_class: continue
            if keep_only_class is not None and label != keep_only_class: continue
            self.image_age_list.append((img_path, label))

    def __len__(self): return len(self.image_age_list)
    def __getitem__(self, idx):
        image_path, label = self.image_age_list[idx]
        full_path = os.path.join(self.image_directory, image_path)
        try:
            img = Image.open(full_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        if self.transform: img = self.transform(img)
        return img, label, str(full_path)

# ================= Robust ViT Loader (保持修复版) =================
def _robust_load(path, map_location):
    allow_classes = [VisionTransformer, vision_transformer.VisionTransformer]
    load_signature = inspect.signature(torch.load)
    kwargs = {'map_location': map_location}
    if 'weights_only' in load_signature.parameters:
        kwargs['weights_only'] = False 
    try:
        if hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals(allow_classes):
                return torch.load(path, **kwargs)
        else:
            return torch.load(path, **kwargs)
    except Exception:
        return torch.load(path, **kwargs)

def load_finetuned_vit(checkpoint_path, num_classes=8, device=torch.device('cpu')):
    if not os.path.exists(checkpoint_path):
        # 如果是示例路径不存在，返回 None 以便跳过
        print(f"[Warning] Path not found: {checkpoint_path}")
        return None
        
    try:
        ckpt = _robust_load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return None

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        # Head fix logic omitted for brevity, assuming standard
    else:
        model = vision_transformer.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        state_dict = None
        if isinstance(ckpt, (dict, collections.OrderedDict)):
            if 'model_state_dict' in ckpt: state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
            elif 'model' in ckpt: state_dict = ckpt['model']
            else: state_dict = ckpt
        elif isinstance(ckpt, (list, tuple)):
            for item in ckpt:
                if isinstance(item, (dict, collections.OrderedDict)) and 'model_state_dict' in item:
                    state_dict = item['model_state_dict']; break
        
        if state_dict:
            new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state, strict=False)

    model.to(device)
    model.eval()
    return model

# ================= 特征提取辅助函数 =================
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_posteriors_for_class(loader, model, target_class):
    """提取特定类别的 posteriors"""
    model.eval()
    probs_list = []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs) 
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels_np = np.array(labels)
            for i, lbl in enumerate(labels_np):
                if int(lbl) == target_class:
                    probs_list.append(probs[i])
    if len(probs_list) == 0: return np.empty((0, NUM_CLASSES))
    return np.vstack(probs_list)

def extract_posteriors_except_class(loader, model, target_class, max_samples=None):
    """提取非目标类别的 posteriors (支持全部提取或限制数量)"""
    model.eval()
    probs = []
    collected = 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            p = torch.softmax(outputs, dim=1).cpu().numpy()
            labels_np = np.array(labels)
            for i, lbl in enumerate(labels_np):
                if int(lbl) != target_class:
                    probs.append(p[i])
                    collected += 1
                    if max_samples and collected >= max_samples:
                        return np.vstack(probs)
    if len(probs) == 0: return np.empty((0, NUM_CLASSES))
    return np.vstack(probs)

# ================= 主流程 =================
def main():
    print(f"--- Preparing Data (Target Class: {UNLEARN_CLASS}) ---")
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: Dataset CSV not found at {TRAIN_CSV}")
        return

    train_dataset = CustomDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Train samples: {len(train_dataset)}")

    # 1. 准备 SVM (使用原始模型)
    print("\n--- Training SVM Probe (Teacher) ---")
    original_model = load_finetuned_vit(ORIGINAL_CHECKPOINT, NUM_CLASSES, DEVICE)
    if original_model is None: return

    # 提取 SVM 训练数据
    print("Extracting features from Original Model...")
    t_posteriors = extract_posteriors_for_class(train_loader, original_model, UNLEARN_CLASS)
    nt_posteriors = extract_posteriors_except_class(train_loader, original_model, UNLEARN_CLASS, max_samples=len(t_posteriors))
    
    del original_model # 释放显存

    # 构建 SVM 数据集 (Balanced)
    X_pos = t_posteriors
    y_pos = np.ones(len(X_pos), dtype=int) # 1 = Member (Target Class)
    X_neg = nt_posteriors
    y_neg = np.zeros(len(X_neg), dtype=int) # 0 = Non-Member (Other Class)
    
    X_train = np.concatenate([X_pos, X_neg], axis=0)
    y_train = np.concatenate([y_pos, y_neg], axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    print(f"Training SVM on {len(X_train)} samples...")
    svm = SVC(kernel='linear', probability=True, random_state=SVM_SEED)
    svm.fit(X_train, y_train)
    print(f"SVM Trained. Train Acc: {svm.score(X_train, y_train)*100:.2f}%")

    # 2. 批量测试 Unlearned Models
    target_acc_list = []
    forgetting_rate_list = []
    remain_acc_list = []

    print(f"\n--- Evaluating {len(MODEL_PATHS)} Runs for {MODEL_GROUP_NAME} ---")

    for i, path in enumerate(MODEL_PATHS):
        print(f"Processing Run {i+1}...")
        u_model = load_finetuned_vit(path, NUM_CLASSES, DEVICE)
        
        if u_model is None:
            continue
            
        # 提取 Unlearned 模型的数据
        # (A) Target Samples -> 希望 SVM 预测为 0 (Non-Member)
        u_target_post = extract_posteriors_for_class(train_loader, u_model, UNLEARN_CLASS)
        
        # (B) Remaining Samples -> 希望 SVM 预测为 0 (Non-Member / Other Class)
        # 注意：这里我们取较多样本来验证保留能力，或者取全部
        u_remain_post = extract_posteriors_except_class(train_loader, u_model, UNLEARN_CLASS, max_samples=500) 
        
        # 预测 Target
        if len(u_target_post) > 0:
            t_preds = svm.predict(u_target_post)
            # SVM 标签 1 是 "Target Class style"。
            # Target Acc = 预测为 1 的比例。我们希望它低。
            t_acc = np.mean(t_preds == 1)
        else:
            t_acc = 0.0
            
        # 预测 Remaining
        if len(u_remain_post) > 0:
            r_preds = svm.predict(u_remain_post)
            # SVM 标签 0 是 "Other Class style"。
            # 对于 Remaining 数据，Ground Truth 是 0。
            # Remain Acc = 预测为 0 的比例。我们希望它高 (保持为 Other Class 特征)。
            r_acc = np.mean(r_preds == 0)
        else:
            r_acc = 0.0
            
        # 计算 Forgetting Rate
        f_rate = 1.0 - t_acc
        
        target_acc_list.append(t_acc)
        forgetting_rate_list.append(f_rate)
        remain_acc_list.append(r_acc)
        
        print(f"  -> Run {i+1}: Target Acc={t_acc*100:.2f}%, Forgetting Rate={f_rate*100:.2f}%, Remain Acc={r_acc*100:.2f}%")
        del u_model

    # 3. 统计结果
    print("\n" + "="*140)
    print(f"Final Statistics for {MODEL_GROUP_NAME} (Over {len(target_acc_list)} runs)")
    print("="*140)

    # 计算统计量
    t_stats = calculate_statistics(target_acc_list, baseline_value=0.0)
    f_stats = calculate_statistics(forgetting_rate_list, baseline_value=1.0)
    r_stats = calculate_statistics(remain_acc_list, baseline_value=1.0)

    # 打印表格
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
    print("\nInterpretation:")
    print("1. Target Acc (Ref=0): Lower is better. If high, machine still recognizes target class.")
    print("2. Forgetting Rate (Ref=1): Higher is better. 100% means perfect unlearning (Target Acc=0).")
    print("   - Cliff Delta near 0 (from negative side) is ideal.")
    print("3. Remaining Acc (Ref=1): Higher is better. Means non-target data is still classified as non-target by SVM.")

if __name__ == "__main__":
    main()