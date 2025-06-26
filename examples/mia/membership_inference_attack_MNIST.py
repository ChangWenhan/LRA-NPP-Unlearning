import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# 定义模型
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

# 加载预训练模型权重
def load_model_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    return model

# 假设 args 中有训练新模型的参数
class Args:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = '/home/cwh/Workspace/TorchLRP-master/examples/models/mnist_model.pth'
    epochs = 5
    train_new = False

args = Args()

# 加载和预处理数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('/home/cwh/Workspace/TorchLRP-master/data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# 筛选标签为1的数据
indices = [i for i, target in enumerate(train_dataset.targets) if target == 1]

# 创建一个新的数据集，只包含标签为1的数据
subset_test_dataset = Subset(train_dataset, indices)

# 使用DataLoader加载这个新的数据集
subset_test_loader = DataLoader(subset_test_dataset, batch_size=1000, shuffle=False)

# 初始化模型并加载权重
model = get_mnist_model()
model = load_model_weights(model, args.weight_path)
model = model.to(args.device)
model.eval()


unlearned_model = torch.load('/mnt/disk/cwh/Boundary-Unlearning-Code-master/checkpoints/boundary_shrink_unlearn_model_MNIST.pth').to('cuda')
# unlearned_model.eval()

# 提取训练集的后验概率
probs = []
labels = []

with torch.no_grad():
    for data, target in train_loader:
        data = data.to(args.device)
        outputs = model(data)
        probabilities = nn.functional.softmax(outputs, dim=1)
        probs.append(probabilities.cpu().numpy())
        labels.append(target.cpu().numpy())

probs = np.concatenate(probs, axis=0)
labels = np.concatenate(labels, axis=0)

# 构建SVM训练集
svm_labels = (labels == 1).astype(int)  # 类别1的后验概率为正样本，其余类别为负样本

# 只使用类别1的后验概率作为特征
svm_features = probs[:, 1].reshape(-1, 1)

# 标准化特征
scaler = StandardScaler()
svm_features = scaler.fit_transform(svm_features)

# 训练支持向量机
svm_classifier = svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(svm_features, svm_labels)

# 测试SVM模型（可选）
test_dataset = datasets.MNIST('/home/cwh/Workspace/TorchLRP-master/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_probs = []
test_labels = []

with torch.no_grad():
    for data, target in subset_test_loader:
        data = data.to(args.device)
        outputs = unlearned_model(data)
        probabilities = nn.functional.softmax(outputs, dim=1)
        test_probs.append(probabilities.cpu().numpy())
        test_labels.append(target.cpu().numpy())

test_probs = np.concatenate(test_probs, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

test_svm_labels = (test_labels == 1).astype(int)
test_svm_features = test_probs[:, 1].reshape(-1, 1)
test_svm_features = scaler.transform(test_svm_features)

test_predictions = svm_classifier.predict(test_svm_features)
test_accuracy = accuracy_score(test_svm_labels, test_predictions)

print(f"SVM test accuracy: {test_accuracy:.2f}")

# 打印每个数据对应的类别和支持向量机的结果
# for i in range(len(test_labels)):
#     print(f"Data index: {i}, True label: {test_labels[i]}, SVM predicted: {test_predictions[i]}")