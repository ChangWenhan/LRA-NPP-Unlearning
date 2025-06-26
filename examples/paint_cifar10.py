import sys
import os
import torch
from torch.utils.data import DataLoader, Subset

import pathlib
import torchvision
from collections import Counter
from torchvision import datasets, transforms as T
import configparser

import numpy as np
import matplotlib.pyplot as plt

# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns
from utils import store_patterns, load_patterns
from visualization import project, clip_quantile, heatmap_grid, grid

torch.manual_seed(1337)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # # # # ImageNet Data
config = configparser.ConfigParser()
config.read((base_path / 'config.ini').as_posix())
sys.path.append(config['DEFAULT']['ImageNetDir'])
from torch_imagenet import ImageNetDataset

# Normalization as expected by pytorch vgg models
# https://pytorch.org/docs/stable/torchvision/models.html
_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

def unnormalize(x):
    return x * _std + _mean

def accuracy_test(test_model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = test_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'VGG 模型在测试数据集上的分类精度为: {accuracy:.2f}%')

transform = T.Compose([
    T.Resize((224, 224)),  # ResNet50 需要 224x224 输入
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 CIFAR-10 数据集，以测试集为遗忘数据集
full_dataset = torchvision.datasets.CIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=False, download=True, transform=transform)

# 找到类别 0 的索引
class_0_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
class_else_indices = [i for i, (_, label) in enumerate(full_dataset) if label != 0]

# 创建类别 0 的子集
class_0_subset = Subset(full_dataset, class_0_indices)
class_else_subset = Subset(full_dataset, class_else_indices)

# 创建 train_loader 和 test_loader
# train_loader = DataLoader(class_0_subset, batch_size=9, shuffle=True)
train_loader = DataLoader(class_else_subset, batch_size=9, shuffle=True)

# full_dataset = torchvision.datasets.CIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=True, download=True, transform=transform)
# train_loader = DataLoader(full_dataset, batch_size=9, shuffle=False)

vgg_unlearn = torch.load('examples/models/resnet50_cifar10_unlearned.pth').to(device)
vgg_origin = torch.load('examples/models/resnet50_cifar10_epoch_10.pth').to(device)
vgg_unlearn.eval()
vgg_origin.eval()

lrp_vgg = lrp.convert_vgg(vgg_origin).to(device)
lrp_vgg_un = lrp.convert_vgg(vgg_unlearn).to(device)
# # # # #

# Check that the vgg and lrp_vgg models does the same thing
for x, y in train_loader: break
x = x.to(device)
x.requires_grad_(True)

y_hat = vgg_origin(x)
y_hat_lrp = lrp_vgg.forward(x)

assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), "\n\n%s\n%s\n%s" % (str(y_hat.view(-1)[:10]), str(y_hat_lrp.view(-1)[:10]), str((torch.abs(y_hat - y_hat_lrp)).max()))
print("Done testing")
# # # # #

# # # # # Patterns for PatternNet and PatternAttribution
patterns_path = (base_path / 'examples' / 'patterns' / 'CIFAR10_pattern_pos.pkl').as_posix()
if not os.path.exists(patterns_path):
    patterns = fit_patternnet_positive(lrp_vgg, train_loader, device=device)
    store_patterns(patterns_path, patterns)
else:
    patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]

# print("Loaded patterns")

# # # # # Plotting 
def compute_and_plot_explanation(rule, ax_, patterns=None, plt_fn=heatmap_grid): 
    # Forward pass
    y_hat_lrp = lrp_vgg.forward(x, explain=True, rule=rule, pattern=patterns)

    # Choose argmax
    y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
    y_hat_lrp = y_hat_lrp.sum()

    # Backward pass (compute explanation)
    y_hat_lrp.backward()
    attr = x.grad

    # Plot
    attr = plt_fn(attr)
    ax_.imshow(attr)
    ax_.set_title(rule, fontsize=24)
    ax_.axis('off')

def compute_and_plot_explanation_un(rule, ax_, patterns=None, plt_fn=heatmap_grid): 
    # Forward pass
    y_hat_lrp = lrp_vgg_un.forward(x, explain=True, rule=rule, pattern=patterns)

    # Choose argmax
    y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
    y_hat_lrp = y_hat_lrp.sum()

    # Backward pass (compute explanation)
    y_hat_lrp.backward()
    attr = x.grad

    # Plot
    attr = plt_fn(attr)
    ax_.imshow(attr)
    ax_.set_title(rule, fontsize=24)
    ax_.axis('off')

# PatternNet is typically handled a bit different, when visualized.
def signal_fn(X):
    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    X = clip_quantile(X)
    X = project(X)
    X = grid(X)
    return X

explanations = [
        # rule                  Pattern     plt_fn          Fig. pos
        ('alpha1beta0',         None,       heatmap_grid,   (1, 0)), 
        ('epsilon',             None,       heatmap_grid,   (0, 1)), 
        ('gamma+epsilon',       None,       heatmap_grid,   (0, 2)), 
        # ('patternnet',          patterns,   signal_fn,      (0, 2)),
        ('alpha2beta1',         None,       heatmap_grid,   (1, 1)),  
        ('patternattribution',  patterns,   heatmap_grid,   (1, 2)),
    ]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
figun, axun = plt.subplots(2, 3, figsize=(12, 8))
print("Plotting")

# Plot inputs
input_to_plot = unnormalize(x).permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
input_to_plot = grid(input_to_plot, 3, 1.)
ax[0, 0].imshow(input_to_plot)
ax[0, 0].set_title("Input", fontsize=24)
ax[0, 0].axis('off')
axun[0, 0].imshow(input_to_plot)
axun[0, 0].set_title("Input", fontsize=24)
axun[0, 0].axis('off')

# Plot explanations
for i, (rule, pattern, fn, (p, q) ) in enumerate(explanations): 
    compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)
    compute_and_plot_explanation_un(rule, axun[p, q], patterns=pattern, plt_fn=fn)

fig.tight_layout()
fig.savefig(base_path / 'examples' / 'plots' / "Cifar10_explanations_origin_else.png", dpi=280)
figun.tight_layout()
figun.savefig(base_path / 'examples' / 'plots' / "Cifar10_explanations_unlearn_else.png", dpi=280)