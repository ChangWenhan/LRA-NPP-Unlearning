import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn import linear_model, model_selection
from PIL import Image
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

# ================================================================
#                       1. 数据准备与工具
# ================================================================

def setup_data():
    """下载并解压数据集（如果不存在）"""
    if not os.path.exists('./custom_korean_family_dataset_resolution_128'):
        print(">>> [Data] 正在下载数据集...")
        os.system("wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg?download=1 -O custom_korean_family_dataset_resolution_128.zip")
        print(">>> [Data] 正在解压数据集...")
        os.system("unzip -q custom_korean_family_dataset_resolution_128.zip -d ./custom_korean_family_dataset_resolution_128")
        print(">>> [Data] 数据集准备完成。")
    else:
        print(">>> [Data] 数据集已存在。")

def parsing(meta_data):
    """解析 metadata CSV 文件"""
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

# ================================================================
#                       2. Dataset 定义
# ================================================================
class CustomDataset(Dataset):
    def __init__(self, meta_data_path, image_directory, transform=None, remove_class=None):
        self.meta_data = pd.read_csv(meta_data_path)
        self.image_directory = image_directory
        self.transform = transform
        
        # 解析元数据
        full_list = parsing(self.meta_data)
        
        # 标签映射 a->0, ..., h->7 (共8类)
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }
        
        self.image_age_list = []
        # 过滤或保留数据
        for img_path, age_cls in full_list:
            if age_cls not in self.age_class_to_label:
                continue
            label = self.age_class_to_label[age_cls]
            
            # 核心逻辑：如果指定了 remove_class，则跳过该类别的数据
            if remove_class is not None and label == remove_class:
                continue
                
            self.image_age_list.append((img_path, label))

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, label = self.image_age_list[idx]
        full_path = os.path.join(self.image_directory, image_path)
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception:
            # 容错处理
            img = Image.new('RGB', (224, 224))
        
        if self.transform:
            img = self.transform(img)

        return img, label

# ================================================================
#                       3. 模型加载
# ================================================================
def load_vit_model(pretrained=True, freeze=False, num_classes=8, device="cuda"):
    """加载 ViT-B/16"""
    if pretrained:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
    else:
        model = models.vit_b_16(weights=None)

    # 替换分类头
    if hasattr(model.heads, 'head'):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
        
    model.to(device)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    else:
        model.train()

    return model

# ================================================================
#                       4. 训练工具 (EarlyStopping & Trainer)
# ================================================================
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ViTTrainer:
    def __init__(self, train_loader, val_loader=None, device="cuda"):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader 

    def train(self, model, epochs=10, lr=1e-4, patience=3,
            test_loader_for_saving=None, save_path=None,
            use_early_stop=False, record_time=True,
            stop_acc=0.85):
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()
        early_stop = EarlyStopping(patience=patience) if use_early_stop else None

        best_test_acc = -1.0
        
        epoch_times = []   
        training_start = time.time() 

        for epoch in range(epochs):
            epoch_start = time.time()

            model.train()
            total, correct, loss_sum = 0, 0, 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            avg_loss = loss_sum / max(1, len(self.train_loader))
            train_acc = correct / max(1, total)

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            if record_time:
                epoch_times.append(epoch_time)

            print(
                f"[Epoch {epoch+1}/{epochs}] "
                f"Train loss={avg_loss:.4f}, acc={train_acc:.4f}, "
                f"time={epoch_time:.2f}s"
            )

            # ---------- 训练中止条件 ----------
            if train_acc >= stop_acc:
                total_time = time.time() - training_start
                print(f"🎯 达到训练精度阈值 stop_acc={stop_acc:.2f}（当前 train_acc={train_acc:.4f}），中止训练。")
                print(f"⏱ 总训练时间：{total_time:.2f}s")
                if record_time:
                    return model, epoch_times, total_time
                else:
                    return model

            # ---------- Validation ----------
            if self.val_loader:
                val_loss, val_acc = self.evaluate_with_metrics(model, self.val_loader, criterion)
                print(f"   Validation loss={val_loss:.4f}, acc={val_acc:.4f}")

                if use_early_stop:
                    early_stop(val_loss)
                    if early_stop.stop:
                        print("🔥 Early stopping triggered based on validation loss.")
                        break

            # ---------- Test & Save Best ----------
            if test_loader_for_saving is not None:
                test_loss, test_acc = self.evaluate_with_metrics(model, test_loader_for_saving, criterion)
                print(f"   Test(for saving) loss={test_loss:.4f}, acc={test_acc:.4f}")

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    
                    if save_path:
                        # 确保目录存在
                        save_dir = "/home/cwh/Workspace/TorchLRP-master/examples/models/"
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        base, ext = os.path.splitext(save_path)
                        # 生成最佳模型文件名
                        best_filename = f"{base}_best_on_test{ext if ext else '.pth'}"
                        best_path = os.path.join(save_dir, best_filename)
                        
                        # [FIX] 这里改为保存 state_dict (权重字典)，而非完整模型对象
                        # 这样加载时最安全，兼容性最好
                        torch.save(model.state_dict(), best_path)
                        print(f"   ✅ 新的最佳模型权重已保存 (test acc={best_test_acc:.4f}) -> {best_path}")

        # End of epochs
        total_time = time.time() - training_start
        if record_time:
            print("\n⏱ 每个 epoch 训练时间：")
            for i, t in enumerate(epoch_times):
                print(f"Epoch {i+1}: {t:.2f}s")
            print(f"⏱ 总训练时间：{total_time:.2f}s")
            return model, epoch_times, total_time

        print(f"⏱ 总训练时间：{total_time:.2f}s")
        return model

    def evaluate_with_metrics(self, model, loader, criterion=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        model.eval()
        total, correct, loss_sum = 0, 0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        avg_loss = loss_sum / max(1, len(loader))
        acc = correct / max(1, total)
        return avg_loss, acc

# ================================================================
#                       5. MIA Attack (基于 Loss)
# ================================================================
def compute_losses(net, loader, device="cuda"):
    """计算 loader 中所有样本的 loss"""
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    net.eval()
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = net(inputs)
            losses = criterion(logits, targets).cpu().detach().numpy()
            all_losses.extend(losses)
            
    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=42):
    """基于 Loss 的逻辑回归攻击"""
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        # 有时候可能只有一个类（如果batch极小），做个容错
        return np.array([0.5]) 

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")

def run_mia_attack(model, target_loader, unseen_loader, device="cuda"):
    """
    MIA 攻击流程
    """
    print(f"\n>>> [MIA] 开始执行成员推断攻击...")
    
    # 1. 计算 Loss
    print("   Computing losses for Target Set (Member)...")
    target_losses = compute_losses(model, target_loader, device)
    print("   Computing losses for Unseen Set (Non-Member)...")
    unseen_losses = compute_losses(model, unseen_loader, device)

    # 2. 数据平衡
    np.random.seed(42)
    np.random.shuffle(target_losses)
    min_len = min(len(target_losses), len(unseen_losses))
    if min_len == 0:
        print("   [Error] 数据集为空，无法执行 MIA")
        return 0.0
        
    target_losses = target_losses[:min_len]
    unseen_losses = unseen_losses[:min_len]

    # 3. 构造攻击数据
    samples_mia = np.concatenate((unseen_losses, target_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(target_losses)

    # 4. 运行攻击
    mia_scores = simple_mia(samples_mia, labels_mia)
    mean_mia = mia_scores.mean()
    
    # Unlearning 评价指标: Forgetting Score
    forgetting_score = abs(0.5 - mean_mia)

    print(f"   MIA Accuracy: {mean_mia:.4f}")
    print(f"   Forgetting Score: {forgetting_score:.4f} (越接近0越好)")
    
    return mean_mia

# ================================================================
#                       6. 数据加载辅助函数
# ================================================================
def get_dataloaders(batch_size=32, remove_class=None):
    base_dir = "data/custom_korean_family_dataset_resolution_128"
    train_csv = f"{base_dir}/custom_train_dataset.csv"
    val_csv = f"{base_dir}/custom_val_dataset.csv"
    unseen_csv = f"{base_dir}/custom_test_dataset.csv"
    
    train_img_dir = f"{base_dir}/train_images"
    val_img_dir = f"{base_dir}/val_images"
    unseen_img_dir = f"{base_dir}/test_images"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 训练集：根据 remove_class 过滤
    train_dataset = CustomDataset(train_csv, train_img_dir, train_transform, remove_class=remove_class)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证集
    val_dataset = CustomDataset(val_csv, val_img_dir, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Unseen 集 (作为 MIA 的 Non-Member / test)
    unseen_dataset = CustomDataset(unseen_csv, unseen_img_dir, test_transform)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, unseen_loader

def get_target_class_loader(batch_size=64, target_class=0):
    """只获取特定类别的数据（用于MIA测试该类别是否被记住/遗忘）"""
    base_dir = "./custom_korean_family_dataset_resolution_128"
    train_csv = f"{base_dir}/custom_train_dataset.csv"
    train_img_dir = f"{base_dir}/train_images"
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # 加载全部数据，然后筛选
    full_dataset = CustomDataset(train_csv, train_img_dir, transform, remove_class=None)
    indices = [i for i, (_, label) in enumerate(full_dataset.image_age_list) if label == target_class]
    
    if len(indices) == 0:
        return None
        
    subset = torch.utils.data.Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

# ================================================================
#         7. Pipeline (程序入口) - [FIXED]
# ================================================================
def pipeline(mode='pretrain', target_class=0, do_mia=False, epochs=5, save_path=None, device="cuda", use_early_stop=False):
    """
    程序主入口
    """
    # setup_data()
    print(f"\n==================================================")
    print(f" 模式: {mode.upper()} | 目标类别: {target_class} | 设备: {device}")
    print(f"==================================================")

    # 1. 确定训练数据
    if mode == 'pretrain':
        print(">>> [Train] 加载全量数据集 (Classes 0-7)...")
        remove_class_for_train = None
    elif mode == 'retrain':
        print(f">>> [Train] 加载剔除类别 {target_class} 后的数据集...")
        remove_class_for_train = target_class
    else:
        raise ValueError("Mode must be 'pretrain' or 'retrain'")

    train_loader, val_loader, unseen_loader = get_dataloaders(batch_size=64, remove_class=remove_class_for_train)
    
    # 2. 训练模型
    print(f"\n>>> [Model] 开始训练 ({epochs} epochs)...")
    model = load_vit_model(pretrained=True, freeze=False, num_classes=8, device=device)
    trainer = ViTTrainer(train_loader, val_loader, device=device)
    
    # [FIX] 这里的返回值可能是 (model, times, total_time) 形式的元组
    training_result = trainer.train(
        model, epochs=epochs, lr=1e-4, patience=3,
        test_loader_for_saving=unseen_loader, save_path=save_path, use_early_stop=use_early_stop
    )

    # [FIX] 解包元组，只保留 model
    if isinstance(training_result, tuple):
        model = training_result[0]
    else:
        model = training_result

    # --- 最终保存逻辑 [FIXED] ---
    if save_path:
        save_dir = "/home/cwh/Workspace/TorchLRP-master/examples/models/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        final_path = save_path if save_path.endswith('.pth') else save_path + '.pth'
        full_save_path = os.path.join(save_dir, final_path)
        
        # [FIX] 保存 state_dict (推荐)，而非模型对象
        torch.save(model.state_dict(), full_save_path)
        print(f"\n>>> [Save] 训练结束，最终模型权重已保存到: {full_save_path}")

    # 3. MIA 评估 (可选)
    if do_mia:
        print(f"\n>>> [MIA] 准备对类别 {target_class} 进行成员推断攻击...")
        target_loader = get_target_class_loader(batch_size=64, target_class=target_class)
        
        if target_loader:
            if mode == 'pretrain':
                print("    [Info] 预训练模式：预期 MIA Acc 较高。")
            else:
                print("    [Info] 重训练(Unlearning)模式：预期 MIA Acc 接近 0.5。")
            run_mia_attack(model, target_loader, unseen_loader, device=device)
        else:
            print("    [Warning] 无法加载目标类别数据，跳过 MIA。")

    return model

# ================================================================
#                       Main Execution
# ================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 场景 1: 预训练 (Baseline) --- 
    # print("\n##### 场景 1: 预训练 (Baseline) #####")
    # pipeline(
    #     mode='pretrain', 
    #     target_class=0, 
    #     do_mia=False,
    #     epochs=10, 
    #     save_path="vit_best_on_test.pth",  # 将会保存为 vit_best_on_test.pth
    #     device=device,
    #     use_early_stop=False
    # )
    
    # --- 场景 2: 重训练 (Retrain/Unlearning) ---
    print("\n##### 场景 2: 重训练 (Retrain/Unlearning) #####")
    pipeline(
        mode='retrain', 
        target_class=0, 
        do_mia=True, 
        epochs=10, 
        save_path="vit_retrained_no_class0.pth", 
        device=device,
        use_early_stop=True
    )