import copy
import random
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
sys.path.insert(0, EXAMPLES_ROOT.as_posix())

from utils import get_mnist_model

DATA_ROOT = REPO_ROOT / "data"
MODEL_ROOT = REPO_ROOT / "examples" / "models"
IMAGENET_MINI_ROOT = REPO_ROOT / "torch_imagenet" / "imagenet-mini" / "train"


@dataclass
class RandomLabelConfig:
    name: str
    num_classes: int
    batch_size: int
    epochs: int
    lr: float
    momentum: float
    unlearn_class: int


CONFIGS: Dict[str, RandomLabelConfig] = {
    "mnist": RandomLabelConfig("mnist", 10, 32, 5, 1e-3, 0.9, 1),
    "cifar10": RandomLabelConfig("cifar10", 10, 64, 10, 1e-3, 0.9, 0),
    "cifar100": RandomLabelConfig("cifar100", 100, 64, 1, 1e-3, 0.9, 0),
    "imagenet": RandomLabelConfig("imagenet", 1000, 32, 1, 1e-3, 0.9, 0),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _randomize_label(target: int, num_classes: int, blocked_label: int) -> int:
    if target != blocked_label:
        return target
    pool = [i for i in range(num_classes) if i != blocked_label]
    return random.choice(pool)


def _patch_targets_inplace(dataset, num_classes: int, blocked_label: int) -> None:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            for i, t in enumerate(targets):
                targets[i] = _randomize_label(int(t), num_classes, blocked_label)
        else:
            for i in range(len(targets)):
                targets[i] = _randomize_label(int(targets[i]), num_classes, blocked_label)
    elif hasattr(dataset, "samples"):
        samples = []
        for path, t in dataset.samples:
            samples.append((path, _randomize_label(int(t), num_classes, blocked_label)))
        dataset.samples = samples
        dataset.targets = [t for _, t in samples]
    else:
        raise ValueError("Dataset has no editable targets/samples")


def _evaluate_class_split(model, loader, device, focus_label: int) -> Tuple[float, float]:
    model.eval()
    correct_focus = 0
    total_focus = 0
    correct_other = 0
    total_other = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            focus_mask = labels == focus_label
            other_mask = labels != focus_label

            correct_focus += (predicted[focus_mask] == labels[focus_mask]).sum().item()
            total_focus += focus_mask.sum().item()
            correct_other += (predicted[other_mask] == labels[other_mask]).sum().item()
            total_other += other_mask.sum().item()

    focus_acc = (100.0 * correct_focus / total_focus) if total_focus else 0.0
    other_acc = (100.0 * correct_other / total_other) if total_other else 0.0
    return focus_acc, other_acc


def _mnist_loaders(config: RandomLabelConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root=DATA_ROOT.as_posix(), train=True, download=True, transform=transform)
    test = datasets.MNIST(root=DATA_ROOT.as_posix(), train=False, download=True, transform=transform)
    _patch_targets_inplace(train, config.num_classes, config.unlearn_class)
    return (
        torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=64, shuffle=False),
    )


def _cifar_loaders(config: RandomLabelConfig, dataset_cls):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = dataset_cls(root=DATA_ROOT.as_posix(), train=True, download=True, transform=transform)
    test = dataset_cls(root=DATA_ROOT.as_posix(), train=False, download=True, transform=transform)
    _patch_targets_inplace(train, config.num_classes, config.unlearn_class)
    return (
        torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=0),
        torch.utils.data.DataLoader(test, batch_size=64, shuffle=False, num_workers=0),
    )


def _imagenet_loaders(config: RandomLabelConfig):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder(root=IMAGENET_MINI_ROOT.as_posix(), transform=transform)
    test = datasets.ImageFolder(root=IMAGENET_MINI_ROOT.as_posix(), transform=transform)
    _patch_targets_inplace(train, len(train.classes), config.unlearn_class)
    return (
        torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=64, shuffle=False),
    )


def _load_model(target: str, device: torch.device):
    if target == "mnist":
        model = get_mnist_model().to(device)
        state = torch.load((MODEL_ROOT / "mnist_model.pth").as_posix(), map_location=device)
        model.load_state_dict(state)
        return model
    if target == "cifar10":
        return torch.load((MODEL_ROOT / "resnet50_cifar10_epoch_10.pth").as_posix(), map_location=device).to(device)
    if target == "cifar100":
        return torch.load((MODEL_ROOT / "resnet50_cifar100_5.pth").as_posix(), map_location=device).to(device)
    if target == "imagenet":
        return torchvision.models.vgg16(weights="IMAGENET1K_V1").to(device)
    raise ValueError(f"Unsupported target: {target}")


def _save_model(target: str, model) -> Path:
    save_map = {
        "mnist": MODEL_ROOT / "mnist_unlearn_random.pkl",
        "cifar10": MODEL_ROOT / "resnet50_cifar10_random.pth",
        "cifar100": MODEL_ROOT / "resnet50_cifar100_random.pth",
        "imagenet": MODEL_ROOT / "imagenet_unlearn_random.pkl",
    }
    out_path = save_map[target]
    torch.save(copy.deepcopy(model), out_path.as_posix())
    return out_path


def run_random_label(target: str, seed: int = 1337, epochs: int = None, break_focus_acc: float = None) -> dict:
    if target not in CONFIGS:
        raise ValueError(f"target must be one of {list(CONFIGS.keys())}")

    cfg = CONFIGS[target]
    if epochs is not None:
        cfg = RandomLabelConfig(**{**cfg.__dict__, "epochs": int(epochs)})

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target == "mnist":
        train_loader, test_loader = _mnist_loaders(cfg)
    elif target == "cifar10":
        train_loader, test_loader = _cifar_loaders(cfg, torchvision.datasets.CIFAR10)
    elif target == "cifar100":
        train_loader, test_loader = _cifar_loaders(cfg, torchvision.datasets.CIFAR100)
    else:
        train_loader, test_loader = _imagenet_loaders(cfg)

    model = _load_model(target, device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    final_focus = 0.0
    final_other = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        final_focus, final_other = _evaluate_class_split(model, test_loader, device, cfg.unlearn_class)
        print(
            f"[{target}] epoch={epoch + 1}/{cfg.epochs} "
            f"loss={running_loss / max(1, len(train_loader)):.4f} "
            f"focus_acc={final_focus:.3f} other_acc={final_other:.3f}"
        )

        if break_focus_acc is not None and final_focus <= break_focus_acc:
            break

    elapsed = time.time() - start
    save_path = _save_model(target, model)

    return {
        "target": target,
        "seed": seed,
        "epochs": cfg.epochs,
        "focus_label": cfg.unlearn_class,
        "focus_acc": final_focus,
        "other_acc": final_other,
        "time_sec": elapsed,
        "save_path": save_path.as_posix(),
    }
