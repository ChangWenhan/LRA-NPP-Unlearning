# import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from torchvision import models
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # 设置路径
# data_path = "/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train"
# unlearned_model_path = "/home/cwh/Workspace/TorchLRP-master/examples/models/imagenet_unlearn_random.pkl"
# unlearned_model_path = "/mnt/disk/cwh/amnesiac/resnet/unlearned_model.pkl"
# unlearned_model_path = "/mnt/disk/cwh/Boundary-Unlearning-Code-master/checkpoints/boundary_shrink_unlearn_model.pth"
unlearned_model_path = "/home/cwh/Workspace/TorchLRP-master/examples/models/imagenet_unlearned_model_ln.pkl"

# # 检查CUDA是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义数据预处理
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 加载数据集
# dataset = datasets.ImageFolder(root=data_path, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# # 加载预训练的VGG16模型
# vgg16 = models.vgg16(pretrained=True).to(device)
# vgg16.eval()

# # 获取后验概率分布
# def get_probabilities(model, dataloader):
#     probabilities = []
#     labels = []
#     with torch.no_grad():
#         for inputs, targets in dataloader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             probabilities.append(probs.cpu().numpy())
#             labels.append(targets.cpu().numpy())
#     return np.concatenate(probabilities), np.concatenate(labels)

# probabilities, labels = get_probabilities(vgg16, dataloader)

# # 将类别为1的后验概率作为标签为1，其余的标签为0
# svm_labels = (labels == 0).astype(int)

# # 训练SVM分类器
# svm = SVC(probability=True)
# svm.fit(probabilities, svm_labels)

# # 加载另一个模型
# unlearned_model = torch.load(unlearned_model_path)
# unlearned_model = unlearned_model.to(device)
# unlearned_model.eval()

# # 获取类别为0的数据
# class_one_indices = [i for i, label in enumerate(labels) if label == 0]
# class_one_dataset = torch.utils.data.Subset(dataset, class_one_indices)
# class_one_loader = DataLoader(class_one_dataset, batch_size=32, shuffle=False, num_workers=4)

# # 获取另一个模型的后验概率
# unlearned_probabilities, _ = get_probabilities(vgg16, dataloader)

# # 预测类别为0的数据
# svm_predictions = svm.predict(unlearned_probabilities)

# # 计算SVM的精确度
# accuracy = accuracy_score(np.ones_like(svm_predictions), svm_predictions)
# print(f"SVM Accuracy: {accuracy * 100:.2f}%")

# # 评估成员推断攻击
# # 这里的方法属于成员推断攻击的一种变体
# # 成员推断攻击的目标是判断某个数据点是否属于训练集
# # 我们通过一个预训练模型提取特征，并用SVM进行分类，来判断类别为1的数据是否属于训练集
# # SVM分类器可以被视为成员推断攻击的一部分，因为它试图利用模型输出的概率分布来推断训练数据

import sys
import pathlib
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())
import lrp
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # 加载数据集
# data_path = "/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train"  # 修改为您的数据集路径
# dataset = datasets.ImageFolder(root=data_path, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# 加载整个数据集
data_path = "/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# 获取类别的索引和类名的映射
class_to_idx = dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 获取前100个类别的索引
first_100_classes = sorted(list(class_to_idx.keys()))[:10]
first_100_indices = [class_to_idx[cls] for cls in first_100_classes]

# 筛选出属于前100个类别的样本索引
subset_indices = [i for i, (img, label) in enumerate(dataset) if label in first_100_indices]

# 创建子集
subset = Subset(dataset, subset_indices)

# 创建dataloader
dataloader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

# 加载预训练的VGG16模型
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()

# 提取特征
def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# 提取VGG16特征
vgg16_features, vgg16_labels = extract_features(vgg16, dataloader)

# 创建SVM模型，并训练
svm = SVC(probability=True)
svm_labels = (vgg16_labels == 0).astype(int)  # 类别0的标签为1，其余类别的标签为0

svm.fit(vgg16_features, svm_labels)

# 加载未训练的新模型（这里以随机初始化的VGG16为例）
unlearned_model = torch.load(unlearned_model_path).to(device)
unlearned_model.eval()

# 获取类别为0的数据
class_one_indices = [i for i, label in enumerate(vgg16_labels) if label != 0]
class_one_dataset = torch.utils.data.Subset(dataset, class_one_indices)
class_one_loader = DataLoader(class_one_dataset, batch_size=32, shuffle=False, num_workers=4)

# 提取新模型的特征
unlearned_features, unlearned_labels = extract_features(unlearned_model, dataloader)

# 使用训练好的SVM对新模型的特征进行分类
predicted_probabilities = svm.predict_proba(unlearned_features)

# 计算类别0上的精确度
predicted_labels = (predicted_probabilities[:, 1] > 0.5).astype(int)
true_labels = (unlearned_labels == 0).astype(int)

print(predicted_labels)
print(true_labels)

accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Accuracy on class 0: {accuracy:.4f}")
