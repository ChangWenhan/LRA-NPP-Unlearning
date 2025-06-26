import sys
import pathlib
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import lrp
from lrp import Sequential, Linear, Conv2d, MaxPool2d

# 设置路径
data_path = "/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train"
model_path = "/home/cwh/Workspace/TorchLRP-master/examples/models/imagenet_unlearned_model_ln.pkl"

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_num = int(sys.argv[1]) if len(sys.argv) > 1 else 16 # Default to vgg16

vgg = getattr(torchvision.models, "vgg%i"%vgg_num)(pretrained=True).to(device)
# vgg = torchvision.models.vgg16(pretrained=True).to(device)
vgg.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# 获取前3个类别的数据索引
class_indices = {i: [] for i in range(3)}
for idx, (_, label) in enumerate(dataset):
    if label < 3:
        class_indices[label].append(idx)

# 提取每个类别的第一个样本，组成一个小数据集
small_dataset_indices = [indices[0] for indices in class_indices.values()]
small_dataset = Subset(dataset, small_dataset_indices)

# 提取前10个类别的剩余数据
class_indices = {i: [] for i in range(10)}
remaining_indices = [idx for label, indices in class_indices.items() for idx in indices[1:]]
remaining_dataset = Subset(dataset, remaining_indices)

# 创建数据加载器
small_data_loader = DataLoader(small_dataset, batch_size=1, shuffle=False)
remaining_data_loader = DataLoader(remaining_dataset, batch_size=32, shuffle=False)

# 加载预训练模型
unlearned_model = torch.load(model_path)
unlearned_model = unlearned_model.to(device)
unlearned_model.eval()

# 去掉模型的最后一层
unlearned_model = nn.Sequential(*list(unlearned_model.children())[:-1])

# 提取特征函数
def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)

# 提取特征
remaining_features, remaining_labels = extract_features(remaining_data_loader, vgg)

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(remaining_features.numpy())

# 训练三个SVM
svms = []
for i in range(3):
    labels = (remaining_labels.numpy() == i).astype(int)
    clf = svm.SVC(probability=True)
    clf.fit(scaled_features, labels)
    svms.append(clf)

# 计算每个SVM的精度
for i, clf in enumerate(svms):
    pred_labels = clf.predict(scaled_features)
    accuracy = accuracy_score((remaining_labels.numpy() == i).astype(int), pred_labels)
    print(f"类别 {i} 的 SVM 精度: {accuracy:.2f}")

# 提取小数据集的特征
small_features, small_labels = extract_features(small_data_loader, unlearned_model)
scaled_small_features = scaler.transform(small_features.numpy())

# 检查小数据集上的分类结果
for i, (feature, label) in enumerate(zip(scaled_small_features, small_labels)):
    print(f"样本 {i} 原本的类别: {label.item()}")
    for j, clf in enumerate(svms):
        pred = clf.predict([feature])[0]
        print(pred)
        print(f"SVM {j} 识别结果: {'正确' if pred == 1 else '错误'}")
