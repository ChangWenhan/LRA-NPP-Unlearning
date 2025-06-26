import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import random

import sys
import os
# 获取上上级目录
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)
import lrp

# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 1. 加载CIFAR-10数据集
transform = T.Compose([
    T.Resize((224, 224)),  # ResNet50 需要 224x224 输入
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 2. 使用预训练模型 (例如 ResNet18)
pretrained_model = torch.load('examples/models/resnet50_cifar10_epoch_10.pth')
pretrained_model.eval()

# 3. 提取类别为0的后验概率分布并标记为1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model.to(device)

def get_posteriors_and_labels(loader, model, class_label):
    posteriors = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            for i in range(len(targets)):
                if targets[i] == class_label:
                    posteriors.append(outputs[i])
                    labels.append(1)  # 类别为0的样本标记为1
    return np.array(posteriors), np.array(labels)

# 获取类别0的后验概率分布
class_0_posteriors, class_0_labels = get_posteriors_and_labels(trainloader, pretrained_model, 0)

# 4. 平衡类别数据，从其他类别中提取数据并标记为0
def get_balanced_non_class_data(loader, model, class_label, num_samples):
    posteriors = []
    labels = []
    counter = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            for i in range(len(targets)):
                if targets[i] != class_label:
                    posteriors.append(outputs[i])
                    labels.append(0)  # 非类别0的样本标记为0
                    counter += 1
                    if counter >= num_samples:
                        return np.array(posteriors), np.array(labels)
    return np.array(posteriors), np.array(labels)

# 获取非类别0的后验概率分布
non_class_0_posteriors, non_class_0_labels = get_balanced_non_class_data(trainloader, pretrained_model, 0, len(class_0_labels))

# 合并数据
X = np.concatenate((class_0_posteriors, non_class_0_posteriors), axis=0)
y = np.concatenate((class_0_labels, non_class_0_labels), axis=0)

# 5. 训练SVM进行二分类
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X, y)

# 验证SVM在训练数据集上的精度
y_pred = svm_model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"SVM训练集精确度: {accuracy * 100:.2f}%")

# 6. 加载unlearned model，对类别0进行预测
unlearned_model = torch.load('/mnt/disk/cwh/Boundary-Unlearning-CIFAR/checkpoints/boundary_shrink_unlearn_model_cifar10.pth')
unlearned_model.eval()
unlearned_model.to(device)

# 获取未训练模型对类别0的后验概率分布
unlearned_class_0_posteriors, _ = get_posteriors_and_labels(trainloader, unlearned_model, 0)

# 7. 用训练好的SVM对新得到的后验概率分布进行分类
svm_predictions = svm_model.predict(unlearned_class_0_posteriors)

# 统计分类为0和1的比例
num_class_0 = np.sum(svm_predictions == 0)
num_class_1 = np.sum(svm_predictions == 1)
total = len(svm_predictions)

print(f"SVM分类为0的比例: {num_class_0 / total * 100:.2f}%")
print(f"SVM分类为1的比例: {num_class_1 / total * 100:.2f}%")