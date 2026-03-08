import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Reuse local project model constructors
from examples.utils import get_mnist_model


@dataclass
class MIAProfile:
    name: str
    dataset: str
    teacher_path: str
    model_paths: List[str]
    target_class: int
    num_classes: int
    batch_size: int = 100


def _default_profiles(base: str) -> dict:
    return {
        "mnist": MIAProfile(
            name="MNIST_Boundary_Unlearning",
            dataset="mnist",
            teacher_path=f"{base}/examples/models/mnist_model.pth",
            model_paths=sorted(glob.glob(f"{base}/examples/models/models_mia/mnist_*.pth")),
            target_class=1,
            num_classes=10,
            batch_size=64,
        ),
        "cifar10": MIAProfile(
            name="ResNet50_CIFAR10_Boundary_Unlearning",
            dataset="cifar10",
            teacher_path=f"{base}/examples/models/resnet50_cifar10_epoch_10.pth",
            model_paths=sorted(glob.glob(f"{base}/examples/models/models_mia/c10_*.pth")),
            target_class=9,
            num_classes=10,
            batch_size=100,
        ),
        "cifar100": MIAProfile(
            name="ResNet50_CIFAR100_Unlearning_Batch",
            dataset="cifar100",
            teacher_path=f"{base}/examples/models/resnet50_cifar100_5.pth",
            model_paths=sorted(glob.glob(f"{base}/examples/models/models_mia/c100_*.pth")),
            target_class=9,
            num_classes=100,
            batch_size=100,
        ),
        "mufac": MIAProfile(
            name="ViT_MUFAC_Unlearning_Batch",
            dataset="mufac",
            teacher_path=f"{base}/examples/models/vit_best_on_test.pth",
            model_paths=sorted(glob.glob(f"{base}/examples/models/models_mia/vit_*.pth")),
            target_class=0,
            num_classes=8,
            batch_size=32,
        ),
    }


def list_profiles(base: str) -> List[str]:
    return sorted(_default_profiles(base).keys())


def get_profile(base: str, key: str) -> MIAProfile:
    profiles = _default_profiles(base)
    if key not in profiles:
        raise ValueError(f"Unknown profile: {key}")
    return profiles[key]


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_mnist_loader(base: str, batch_size: int):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    ds = torchvision.datasets.MNIST(f"{base}/data", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def _load_cifar_loader(base: str, dataset: str, batch_size: int):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root=f"{base}/data", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def _load_mufac_loader(base: str, batch_size: int):
    import pandas as pd
    from PIL import Image

    root = Path(base) / "data" / "custom_korean_family_dataset_resolution_128"
    csv_path = root / "custom_train_dataset.csv"
    img_dir = root / "train_images"
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class DS(torch.utils.data.Dataset):
        MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        def __init__(self):
            self.rows = []
            meta = pd.read_csv(csv_path)
            for _, r in meta.iterrows():
                age = r["age_class"]
                if age in self.MAP:
                    self.rows.append((r["image_path"], self.MAP[age]))
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            p, y = self.rows[idx]
            img = Image.open((img_dir / p).as_posix()).convert("RGB")
            return transform(img), y

    return torch.utils.data.DataLoader(DS(), batch_size=batch_size, shuffle=True, num_workers=0)


def _load_data_loader(base: str, profile: MIAProfile):
    if profile.dataset == "mnist":
        return _load_mnist_loader(base, profile.batch_size)
    if profile.dataset in {"cifar10", "cifar100"}:
        return _load_cifar_loader(base, profile.dataset, profile.batch_size)
    if profile.dataset == "mufac":
        return _load_mufac_loader(base, profile.batch_size)
    raise NotImplementedError(profile.dataset)


def _load_model(path: str, profile: MIAProfile, device: torch.device):
    if not os.path.exists(path):
        print(f"[Warn] model not found: {path}")
        return None

    kwargs = {"map_location": device}
    import inspect
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False

    ckpt = torch.load(path, **kwargs)

    if profile.dataset == "mnist":
        model = get_mnist_model()
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            model.load_state_dict(ckpt, strict=False)
    elif profile.dataset in {"cifar10", "cifar100"}:
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, profile.num_classes)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
    elif profile.dataset == "mufac":
        from torchvision.models import vision_transformer
        model = vision_transformer.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, profile.num_classes)
        if isinstance(ckpt, torch.nn.Module):
            model.load_state_dict(ckpt.state_dict(), strict=False)
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
    else:
        raise NotImplementedError(profile.dataset)

    model.to(device)
    model.eval()
    return model


def _extract_posteriors(loader, model, target_class, mode="target", max_samples=None):
    out = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(next(model.parameters()).device)
            prob = torch.softmax(model(x), dim=1).cpu().numpy()
            y_np = np.array(y)
            for i, yy in enumerate(y_np):
                if mode == "target" and int(yy) == int(target_class):
                    out.append(prob[i])
                elif mode == "remain" and int(yy) != int(target_class):
                    out.append(prob[i])
                if max_samples and len(out) >= max_samples:
                    return np.asarray(out)
    if len(out) == 0:
        return np.empty((0, prob.shape[1] if 'prob' in locals() else 1))
    return np.asarray(out)


def _stats(values, baseline):
    vals = np.asarray(values)
    mean = float(vals.mean()) if len(vals) else 0.0
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if len(vals) < 2 or std < 1e-12:
        p = 1.0 if abs(mean - baseline) < 1e-12 else 0.0
    else:
        p = float(stats.ttest_1samp(vals, baseline).pvalue)
    more = np.sum(vals > baseline)
    less = np.sum(vals < baseline)
    cliff = float((more - less) / len(vals)) if len(vals) else 0.0
    return {"mean": mean, "std": std, "p": p, "cliff": cliff}


def run_profile(base: str, key: str, device: torch.device, seed: int = 42):
    _set_seed(seed)
    profile = get_profile(base, key)
    loader = _load_data_loader(base, profile)

    teacher = _load_model(profile.teacher_path, profile, device)
    if teacher is None:
        raise RuntimeError(f"teacher model not found: {profile.teacher_path}")

    pos = _extract_posteriors(loader, teacher, profile.target_class, mode="target")
    neg = _extract_posteriors(loader, teacher, profile.target_class, mode="remain", max_samples=len(pos))

    x_train = np.concatenate([pos, neg], axis=0)
    y_train = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(x_train, y_train)
    print(f"[MIA:{key}] SVM train acc: {accuracy_score(y_train, svm.predict(x_train))*100:.2f}%")

    t_accs, f_rates, r_accs = [], [], []

    if not profile.model_paths:
        print(f"[MIA:{key}] No model paths matched, skip.")
        return

    for i, p in enumerate(profile.model_paths, 1):
        m = _load_model(p, profile, device)
        if m is None:
            continue
        t_feat = _extract_posteriors(loader, m, profile.target_class, mode="target")
        r_feat = _extract_posteriors(loader, m, profile.target_class, mode="remain", max_samples=1000)

        t_acc = float(np.mean(svm.predict(t_feat) == 1)) if len(t_feat) else 0.0
        r_acc = float(np.mean(svm.predict(r_feat) == 0)) if len(r_feat) else 0.0
        f_rate = 1.0 - t_acc

        t_accs.append(t_acc)
        r_accs.append(r_acc)
        f_rates.append(f_rate)
        print(f"[MIA:{key}] run{i}: T={t_acc*100:.2f}% F={f_rate*100:.2f}% R={r_acc*100:.2f}%")

    t_s = _stats(t_accs, 0.0)
    f_s = _stats(f_rates, 1.0)
    r_s = _stats(r_accs, 1.0)

    print(f"\n[MIA:{key}] Summary over {len(t_accs)} runs")
    print(f"Target Acc: {t_s['mean']*100:.2f}±{t_s['std']*100:.2f}%, p={t_s['p']:.4g}, cliff={t_s['cliff']:.4f}")
    print(f"Forget Rate: {f_s['mean']*100:.2f}±{f_s['std']*100:.2f}%, p={f_s['p']:.4g}, cliff={f_s['cliff']:.4f}")
    print(f"Remain Acc: {r_s['mean']*100:.2f}±{r_s['std']*100:.2f}%, p={r_s['p']:.4g}, cliff={r_s['cliff']:.4f}")
