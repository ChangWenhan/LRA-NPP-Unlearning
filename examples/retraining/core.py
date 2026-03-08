import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from examples.utils import get_mnist_model


@dataclass
class RetrainProfile:
    dataset: str
    model_path: str | None
    num_classes: int
    unlearn_class: int
    batch_size: int
    epochs: int
    lr: float
    output_path: str


def _base(base: str):
    return {
        "mnist": RetrainProfile(
            dataset="mnist",
            model_path=f"{base}/examples/models/mnist_model.pth",
            num_classes=10,
            unlearn_class=1,
            batch_size=64,
            epochs=5,
            lr=1e-3,
            output_path=f"{base}/examples/models/mnist_retrain_unified.pth",
        ),
        "cifar10": RetrainProfile(
            dataset="cifar10",
            model_path=f"{base}/examples/models/resnet50_cifar10_epoch_10.pth",
            num_classes=10,
            unlearn_class=0,
            batch_size=64,
            epochs=10,
            lr=1e-3,
            output_path=f"{base}/examples/models/resnet50_cifar10_retrain_unified.pth",
        ),
        "cifar100": RetrainProfile(
            dataset="cifar100",
            model_path=f"{base}/examples/models/resnet50_cifar100_5.pth",
            num_classes=100,
            unlearn_class=0,
            batch_size=64,
            epochs=10,
            lr=1e-3,
            output_path=f"{base}/examples/models/resnet50_cifar100_retrain_unified.pth",
        ),
        "vgg": RetrainProfile(
            dataset="imagenet_vgg",
            model_path=None,
            num_classes=1000,
            unlearn_class=0,
            batch_size=32,
            epochs=5,
            lr=1e-3,
            output_path=f"{base}/examples/models/vgg16_retrain_unified.pth",
        ),
        "vit": RetrainProfile(
            dataset="mufac_vit",
            model_path=f"{base}/examples/models/vit_best_on_test.pth",
            num_classes=8,
            unlearn_class=0,
            batch_size=32,
            epochs=5,
            lr=1e-4,
            output_path=f"{base}/examples/models/vit_retrain_unified.pth",
        ),
    }


def list_profiles(base: str) -> List[str]:
    return sorted(_base(base).keys())


def get_profile(base: str, key: str) -> RetrainProfile:
    profiles = _base(base)
    if key not in profiles:
        raise ValueError(f"Unknown retraining profile: {key}")
    return profiles[key]


def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mnist_loaders(base: str, p: RetrainProfile):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = torchvision.datasets.MNIST(f"{base}/data", train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(f"{base}/data", train=False, download=True, transform=tf)
    keep = [i for i, (_, y) in enumerate(train) if int(y) != p.unlearn_class]
    train_sub = torch.utils.data.Subset(train, keep)
    return (
        torch.utils.data.DataLoader(train_sub, batch_size=p.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=p.batch_size, shuffle=False),
    )


def _cifar_loaders(base: str, p: RetrainProfile, ds_name: str):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cls = torchvision.datasets.CIFAR10 if ds_name == "cifar10" else torchvision.datasets.CIFAR100
    train = cls(root=f"{base}/data", train=True, download=True, transform=tf)
    test = cls(root=f"{base}/data", train=False, download=True, transform=tf)
    labels = train.targets
    keep = [i for i in range(len(train)) if int(labels[i]) != p.unlearn_class]
    train_sub = torch.utils.data.Subset(train, keep)
    return (
        torch.utils.data.DataLoader(train_sub, batch_size=p.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=p.batch_size, shuffle=False),
    )


def _imagenet_vgg_loaders(base: str, p: RetrainProfile):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = torchvision.datasets.ImageFolder(root=f"{base}/torch_imagenet/imagenet-mini/train", transform=tf)
    test = torchvision.datasets.ImageFolder(root=f"{base}/torch_imagenet/imagenet-mini/val", transform=tf)
    keep = [i for i, y in enumerate(train.targets) if int(y) != p.unlearn_class]
    train_sub = torch.utils.data.Subset(train, keep)
    return (
        torch.utils.data.DataLoader(train_sub, batch_size=p.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=p.batch_size, shuffle=False),
    )


def _mufac_vit_loaders(base: str, p: RetrainProfile):
    import pandas as pd
    from PIL import Image

    root = Path(base) / "data" / "custom_korean_family_dataset_resolution_128"
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class DS(torch.utils.data.Dataset):
        MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        def __init__(self, csv_name, img_dir, remove_cls=None):
            self.rows = []
            meta = pd.read_csv((root / csv_name).as_posix())
            for _, r in meta.iterrows():
                age = r["age_class"]
                if age not in self.MAP:
                    continue
                y = self.MAP[age]
                if remove_cls is not None and y == remove_cls:
                    continue
                self.rows.append((str(Path(img_dir) / r["image_path"]), y))
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            pth, y = self.rows[idx]
            img = Image.open(pth).convert("RGB")
            return tf(img), y

    train = DS("custom_train_dataset.csv", root / "train_images", remove_cls=p.unlearn_class)
    test = DS("custom_val_dataset.csv", root / "val_images", remove_cls=None)

    return (
        torch.utils.data.DataLoader(train, batch_size=p.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=p.batch_size, shuffle=False),
    )


def _load_model(p: RetrainProfile, device):
    if p.dataset == "mnist":
        m = get_mnist_model()
        if p.model_path and os.path.exists(p.model_path):
            m.load_state_dict(torch.load(p.model_path, map_location=device), strict=False)
        return m.to(device)

    if p.dataset in {"cifar10", "cifar100"}:
        if p.model_path and os.path.exists(p.model_path):
            ckpt = torch.load(p.model_path, map_location=device, weights_only=False)
            if isinstance(ckpt, torch.nn.Module):
                return ckpt.to(device)
            m = torchvision.models.resnet50(pretrained=False)
            m.fc = nn.Linear(m.fc.in_features, p.num_classes)
            if isinstance(ckpt, dict):
                state = ckpt.get("state_dict", ckpt)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                m.load_state_dict(state, strict=False)
            return m.to(device)
        m = torchvision.models.resnet50(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, p.num_classes)
        return m.to(device)

    if p.dataset == "imagenet_vgg":
        m = torchvision.models.vgg16(pretrained=False)
        return m.to(device)

    if p.dataset == "mufac_vit":
        from torchvision.models import vision_transformer
        m = vision_transformer.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, p.num_classes)
        if p.model_path and os.path.exists(p.model_path):
            ckpt = torch.load(p.model_path, map_location=device, weights_only=False)
            if isinstance(ckpt, torch.nn.Module):
                m.load_state_dict(ckpt.state_dict(), strict=False)
            elif isinstance(ckpt, dict):
                m.load_state_dict(ckpt, strict=False)
        return m.to(device)

    raise NotImplementedError(p.dataset)


def _eval_target_other(model, loader, device, target_class):
    model.eval()
    t_total = t_correct = 0
    o_total = o_correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            mask_t = y == target_class
            mask_o = y != target_class
            t_total += mask_t.sum().item()
            t_correct += ((pred == y) & mask_t).sum().item()
            o_total += mask_o.sum().item()
            o_correct += ((pred == y) & mask_o).sum().item()
    t_acc = 100.0 * t_correct / t_total if t_total else 0.0
    o_acc = 100.0 * o_correct / o_total if o_total else 0.0
    return t_acc, o_acc


def run_profile(base: str, key: str, seed: int = 42, device: torch.device | None = None):
    _set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = get_profile(base, key)

    if p.dataset == "mnist":
        train_loader, test_loader = _mnist_loaders(base, p)
    elif p.dataset in {"cifar10", "cifar100"}:
        train_loader, test_loader = _cifar_loaders(base, p, p.dataset)
    elif p.dataset == "imagenet_vgg":
        train_loader, test_loader = _imagenet_vgg_loaders(base, p)
    elif p.dataset == "mufac_vit":
        train_loader, test_loader = _mufac_vit_loaders(base, p)
    else:
        raise NotImplementedError(p.dataset)

    model = _load_model(p, device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=p.lr, momentum=0.9)

    print(f"[Retrain:{key}] dataset={p.dataset}, epochs={p.epochs}, unlearn_class={p.unlearn_class}")
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start:
        start.record()

    for epoch in range(1, p.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        t_acc, o_acc = _eval_target_other(model, test_loader, device, p.unlearn_class)
        print(f"[Retrain:{key}] epoch={epoch}/{p.epochs} loss={loss_sum/max(n,1):.4f} target={t_acc:.2f}% other={o_acc:.2f}%")

    if end:
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000.0
        print(f"[Retrain:{key}] train_time={sec:.2f}s")

    out_dir = Path(p.output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, p.output_path)
    print(f"[Retrain:{key}] saved: {p.output_path}")
