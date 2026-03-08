import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
sys.path.insert(0, REPO_ROOT.as_posix())
sys.path.insert(0, EXAMPLES_ROOT.as_posix())

import lrp
from lrp.patterns import fit_patternnet_positive
from visualization import clip_quantile, grid, heatmap_grid, project
from utils import get_mnist_data, get_mnist_model, load_patterns, prepare_mnist_model, store_patterns

PATTERN_ROOT = EXAMPLES_ROOT / "patterns"
PLOTS_ROOT = EXAMPLES_ROOT / "plots"
MODEL_ROOT = EXAMPLES_ROOT / "models"
DATA_ROOT = REPO_ROOT / "data"
IMAGENET_MINI_ROOT = REPO_ROOT / "torch_imagenet" / "imagenet-mini" / "train"


@dataclass
class PaintConfig:
    target: str
    unlearn_class: int
    batch_size: int
    pattern_name: Optional[str]
    origin_model_path: Optional[str]
    unlearn_model_path: Optional[str]
    use_class_subset: str


CONFIGS = {
    "mnist": PaintConfig("mnist", 1, 9, None, "mnist_model.pth", "mnist_unlearn_class_epsilon_1.pkl", "not_unlearn"),
    "cifar10": PaintConfig("cifar10", 0, 9, "CIFAR10_pattern_pos.pkl", "resnet50_cifar10_epoch_10.pth", "resnet50_cifar10_unlearned.pth", "not_unlearn"),
    "cifar100": PaintConfig("cifar100", 0, 9, "CIFAR100_pattern_pos.pkl", "resnet50_cifar100_5.pth", "resnet50_cifar100_unlearned.pth", "unlearn"),
    "vgg": PaintConfig("vgg", 0, 9, "vgg16_pattern_pos.pkl", None, "imagenet_unlearned_class.pkl", "unlearn"),
}


def _signal_fn(x):
    if x.shape[1] in [1, 3]:
        x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    x = clip_quantile(x)
    x = project(x)
    return grid(x)


def _draw_input(ax, x, title: str):
    if x.shape[1] in [1, 3]:
        image = x.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
        image = grid(image, 3, 1.0)
    else:
        image = heatmap_grid(x * 2 - 1, cmap_name="gray")
    ax.imshow(image)
    ax.set_title(title, fontsize=18)
    ax.axis("off")


def _plot_rule(model, x, rule, ax, pattern=None, plt_fn=heatmap_grid):
    x.grad = None
    y_hat = model.forward(x, explain=True, rule=rule, pattern=pattern)
    y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]].sum()
    y_hat.backward()
    ax.imshow(plt_fn(x.grad))
    ax.set_title(rule, fontsize=18)
    ax.axis("off")


def _load_or_fit_patterns(model, loader, device, name: str):
    path = PATTERN_ROOT / name
    if path.exists():
        return [torch.tensor(p).to(device) for p in load_patterns(path.as_posix())]
    patterns = fit_patternnet_positive(model, loader, device=device)
    store_patterns(path.as_posix(), patterns)
    return patterns


def _mnist_load_models_and_data(device, cfg: PaintConfig):
    args = type("Args", (), {"device": device, "train_new": False, "epochs": 1})
    model = get_mnist_model()
    prepare_mnist_model(args, model, model_path=(MODEL_ROOT / cfg.origin_model_path).as_posix(), train_new=False)
    model = model.to(device)
    unlearned_model = torch.load((MODEL_ROOT / cfg.unlearn_model_path).as_posix(), map_location=device).to(device)

    _, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=1)
    filtered = []
    for data, labels in test_loader:
        if (labels != cfg.unlearn_class).any():
            filtered.append((data, labels))
    loader = DataLoader(filtered, batch_size=cfg.batch_size, shuffle=True)
    return model, unlearned_model, loader


def _cifar_load_models_and_data(device, cfg: PaintConfig):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_cls = torchvision.datasets.CIFAR10 if cfg.target == "cifar10" else torchvision.datasets.CIFAR100
    full_dataset = ds_cls(root=DATA_ROOT.as_posix(), train=False, download=True, transform=transform)
    if cfg.use_class_subset == "unlearn":
        indices = [i for i, (_, label) in enumerate(full_dataset) if label == cfg.unlearn_class]
    else:
        indices = [i for i, (_, label) in enumerate(full_dataset) if label != cfg.unlearn_class]

    loader = DataLoader(Subset(full_dataset, indices), batch_size=cfg.batch_size, shuffle=True)

    origin = torch.load((MODEL_ROOT / cfg.origin_model_path).as_posix(), map_location=device).to(device)
    unlearn = torch.load((MODEL_ROOT / cfg.unlearn_model_path).as_posix(), map_location=device).to(device)
    origin.eval()
    unlearn.eval()
    return origin, unlearn, loader


def _vgg_load_model_and_data(device, cfg: PaintConfig):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=IMAGENET_MINI_ROOT.as_posix(), transform=transform)
    indices = [i for i, (_, label) in enumerate(dataset) if label == cfg.unlearn_class]
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=False)

    unlearn = torch.load((MODEL_ROOT / cfg.unlearn_model_path).as_posix(), map_location=device).to(device)
    unlearn.eval()
    return unlearn, loader


def run_paint(target: str, seed: int = 1337):
    if target not in CONFIGS:
        raise ValueError(f"target must be one of {list(CONFIGS.keys())}")

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = CONFIGS[target]

    if target == "mnist":
        model, unlearned_model, loader = _mnist_load_models_and_data(device, cfg)
        for x, _ in loader:
            break
        x = torch.squeeze(x[: cfg.batch_size], dim=2).to(device)
        x.requires_grad_(True)

        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        figu, axu = plt.subplots(2, 3, figsize=(12, 8))

        _draw_input(ax[0, 0], x, "Input")
        _draw_input(axu[0, 0], x, "Input")
        _plot_rule(model, x, "epsilon", ax[0, 1])
        _plot_rule(model, x, "gamma+epsilon", ax[0, 2])
        _plot_rule(model, x, "alpha1beta0", ax[1, 0])
        _plot_rule(model, x, "alpha2beta1", ax[1, 1])
        _plot_rule(model, x, "gradient", ax[1, 2], plt_fn=lambda attr: heatmap_grid(attr * x))

        _plot_rule(unlearned_model, x, "epsilon", axu[0, 1])
        _plot_rule(unlearned_model, x, "gamma+epsilon", axu[0, 2])
        _plot_rule(unlearned_model, x, "alpha1beta0", axu[1, 0])
        _plot_rule(unlearned_model, x, "alpha2beta1", axu[1, 1])
        _plot_rule(unlearned_model, x, "gradient", axu[1, 2], plt_fn=lambda attr: heatmap_grid(attr * x))

        fig.tight_layout()
        figu.tight_layout()
        origin_path = PLOTS_ROOT / "mnist_explanations_origin_else.png"
        unlearn_path = PLOTS_ROOT / "mnist_explanations_unlearn_else.png"
        fig.savefig(origin_path.as_posix(), dpi=280)
        figu.savefig(unlearn_path.as_posix(), dpi=280)
        return {"target": target, "origin_plot": origin_path.as_posix(), "unlearn_plot": unlearn_path.as_posix()}

    if target in ("cifar10", "cifar100"):
        origin, unlearn, loader = _cifar_load_models_and_data(device, cfg)
        lrp_origin = lrp.convert_vgg(origin).to(device)
        lrp_unlearn = lrp.convert_vgg(unlearn).to(device)

        for x, _ in loader:
            break
        x = x.to(device)
        x.requires_grad_(True)

        patterns = _load_or_fit_patterns(lrp_origin, loader, device, cfg.pattern_name)

        rules = [
            ("alpha1beta0", None, heatmap_grid, (1, 0)),
            ("epsilon", None, heatmap_grid, (0, 1)),
            ("gamma+epsilon", None, heatmap_grid, (0, 2)),
            ("alpha2beta1", None, heatmap_grid, (1, 1)),
            ("patternattribution", patterns, heatmap_grid, (1, 2)),
        ]

        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        figu, axu = plt.subplots(2, 3, figsize=(12, 8))

        _draw_input(ax[0, 0], x, "Input")
        _draw_input(axu[0, 0], x, "Input")

        for rule, pattern, fn, (p, q) in rules:
            _plot_rule(lrp_origin, x, rule, ax[p, q], pattern=pattern, plt_fn=fn)
            _plot_rule(lrp_unlearn, x, rule, axu[p, q], pattern=pattern, plt_fn=fn)

        fig.tight_layout()
        figu.tight_layout()
        origin_path = PLOTS_ROOT / f"{target.capitalize()}_explanations_origin_else.png"
        unlearn_path = PLOTS_ROOT / f"{target.capitalize()}_explanations_unlearn_else.png"
        fig.savefig(origin_path.as_posix(), dpi=280)
        figu.savefig(unlearn_path.as_posix(), dpi=280)
        return {"target": target, "origin_plot": origin_path.as_posix(), "unlearn_plot": unlearn_path.as_posix()}

    model, loader = _vgg_load_model_and_data(device, cfg)
    lrp_model = lrp.convert_vgg(model).to(device)

    for x, _ in loader:
        break
    x = x.to(device)
    x.requires_grad_(True)

    patterns = _load_or_fit_patterns(lrp_model, loader, device, cfg.pattern_name)
    rules = [
        ("alpha1beta0", None, heatmap_grid, (1, 0)),
        ("epsilon", None, heatmap_grid, (0, 1)),
        ("gamma+epsilon", None, heatmap_grid, (0, 2)),
        ("alpha2beta1", None, heatmap_grid, (1, 1)),
        ("patternattribution", patterns, heatmap_grid, (1, 2)),
    ]

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    _draw_input(ax[0, 0], x, "Input")
    for rule, pattern, fn, (p, q) in rules:
        _plot_rule(lrp_model, x, rule, ax[p, q], pattern=pattern, plt_fn=fn)

    fig.tight_layout()
    out_path = PLOTS_ROOT / "vgg16_explanations_unlearn_else.png"
    fig.savefig(out_path.as_posix(), dpi=280)
    return {"target": target, "unlearn_plot": out_path.as_posix()}
