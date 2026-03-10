import copy
import csv
import os
import pathlib
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

# Ensure repository root is importable when scripts are run from batch_experiments/.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import lrp
from examples.utils import get_mnist_model
from .metrics import evaluate_at_ag


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_image_noise(x, level):
    if level <= 0:
        return x
    return x + torch.randn_like(x) * level


def _add_gaussian_noise(tensor, std):
    return tensor + torch.randn_like(tensor) * std


def _add_laplace_noise(tensor, scale):
    lap = torch.distributions.laplace.Laplace(torch.tensor(0.0, device=tensor.device), torch.tensor(scale, device=tensor.device))
    return tensor + lap.sample(tensor.shape)


def _select_analysis_subset(dataset, target_class, sample_size, seed=None):
    indices = []
    for i in range(len(dataset)):
        row = dataset[i]
        y = int(row[1])
        if y == int(target_class):
            indices.append(i)

    if len(indices) == 0:
        return Subset(dataset, [])

    # Match the legacy scripts: rely on PyTorch's default RNG state instead of
    # a per-call generator seeded from the experiment seed.
    order = torch.randperm(len(indices)).tolist()
    selected = [indices[i] for i in order[: min(sample_size, len(indices))]]
    return Subset(dataset, selected)


def _analyze_top_neurons_lrp(lrp_model, analysis_loader, rule, top_k, input_noise, device, analysis_layer=0):
    counter = []

    for row in analysis_loader:
        x = row[0].to(device)
        x = _add_image_noise(x, input_noise)
        x.requires_grad_(True)
        x.grad = None

        y_hat = lrp_model.forward(x, explain=True, rule=rule)
        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]].sum()

        lrp.trace.enable_and_clean()
        y_hat.backward()
        all_rel = lrp.trace.collect_and_disable()

        if not all_rel:
            continue

        layer_idx = min(max(analysis_layer, 0), len(all_rel) - 1)
        t = all_rel[layer_idx][0].tolist()
        top_indices = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:top_k]
        counter.append(top_indices)

    merged = [idx for row in counter for idx in row]
    counts = Counter(merged)
    return [idx for idx, _ in counts.most_common()]


def _perturb_mnist(lrp_model, neuron_indices, method, config):
    neurons = neuron_indices[: config["perturb_top_n"]]
    target_class = config["unlearn_class"]

    fc_w = lrp_model[6].weight.data.clone()
    out_w = lrp_model[8].weight.data.clone()

    for n in neurons:
        if method == "lra_npp":
            out_w[target_class, n] = 0.0
            fc_w[n] = torch.zeros_like(fc_w[n])
        elif method == "noise_gn":
            out_w[target_class, n] = _add_gaussian_noise(out_w[target_class, n], config["noise_std"])
            fc_w[n] = _add_gaussian_noise(fc_w[n], config["noise_std"])
        elif method == "noise_ln":
            out_w[target_class, n] = _add_laplace_noise(out_w[target_class, n], config["noise_laplace_scale"])
            fc_w[n] = _add_laplace_noise(fc_w[n], config["noise_laplace_scale"])

    lrp_model[6].weight.data = fc_w
    lrp_model[8].weight.data = out_w
    return lrp_model


def _perturb_resnet_fc(lrp_model, neuron_indices, method, config):
    layer_map = config["layer_map"]
    output_idx = layer_map["output_layer_idx"]
    out_w = lrp_model[output_idx].weight.data.clone()
    neurons = neuron_indices[: config["perturb_top_n"]]
    target_class = int(config["unlearn_class"])
    class_indices = range(out_w.shape[0])
    if layer_map.get("target_class_only", False):
        class_indices = [target_class]

    for c in class_indices:
        for n in neurons:
            if method == "lra_npp":
                out_w[c, n] = 0.0
            elif method == "noise_gn":
                out_w[c, n] = _add_gaussian_noise(out_w[c, n], config["noise_std"])
            elif method == "noise_ln":
                out_w[c, n] = _add_laplace_noise(out_w[c, n], config["noise_laplace_scale"])

    lrp_model[output_idx].weight.data = out_w
    return lrp_model


def _perturb_imagenet_vgg(lrp_model, neuron_indices, method, config):
    lm = config["layer_map"]
    neurons = neuron_indices[: config["perturb_top_n"]]
    target_class = config["unlearn_class"]

    if lm.get("perturb_fc1", False):
        fc1 = lrp_model[lm["fc1_layer_idx"]].weight.data.clone()
        for n in neurons:
            if method == "lra_npp":
                fc1[n] = torch.zeros_like(fc1[n])
            elif method == "noise_gn":
                fc1[n] = _add_gaussian_noise(fc1[n], config["noise_std"])
            elif method == "noise_ln":
                fc1[n] = _add_laplace_noise(fc1[n], config["noise_laplace_scale"])
        lrp_model[lm["fc1_layer_idx"]].weight.data = fc1

    if lm.get("perturb_fc2", False):
        fc2 = lrp_model[lm["fc2_layer_idx"]].weight.data.clone()
        for n in neurons:
            if method == "lra_npp":
                fc2[target_class, n] = 0.0
            elif method == "noise_gn":
                fc2[target_class, n] = _add_gaussian_noise(fc2[target_class, n], config["noise_std"])
            elif method == "noise_ln":
                fc2[target_class, n] = _add_laplace_noise(fc2[target_class, n], config["noise_laplace_scale"])
        lrp_model[lm["fc2_layer_idx"]].weight.data = fc2

    if lm.get("perturb_output", False):
        out_w = lrp_model[lm["output_layer_idx"]].weight.data.clone()
        for n in neurons:
            if n >= out_w.shape[1]:
                continue
            if method == "lra_npp":
                out_w[target_class, n] = 0.0
            elif method == "noise_gn":
                out_w[target_class, n] = _add_gaussian_noise(out_w[target_class, n], config["noise_std"])
            elif method == "noise_ln":
                out_w[target_class, n] = _add_laplace_noise(out_w[target_class, n], config["noise_laplace_scale"])
        lrp_model[lm["output_layer_idx"]].weight.data = out_w

    return lrp_model


class _HeadRelevanceHook:
    def __init__(self):
        self.activations = None
        self.gradients = None

    def fwd(self, _module, inp, _out):
        self.activations = inp[0].detach()

    def bwd(self, _module, grad_input, _grad_output):
        self.gradients = grad_input[0].detach()


_VIT_ZENNIT_PATCHED = False


def _ensure_vit_zennit_support():
    global _VIT_ZENNIT_PATCHED
    try:
        from torchvision.models import vision_transformer
        from lxt.efficient import monkey_patch, monkey_patch_zennit
    except ImportError as exc:
        raise RuntimeError(
            "MUFAC legacy ViT LRP dependencies (`lxt`, `zennit`) are unavailable."
        ) from exc

    if not _VIT_ZENNIT_PATCHED:
        monkey_patch(vision_transformer, verbose=False)
        monkey_patch_zennit(verbose=False)
        _VIT_ZENNIT_PATCHED = True
    return True


def _build_mufac_transform(config):
    return _build_mufac_eval_transform(config)


def _build_mufac_eval_transform(config):
    if config.get("legacy_lra_npp_alignment", False):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _get_mufac_zennit_composite(config):
    try:
        from zennit.composites import EpsilonGammaBox, LayerMapComposite
        import zennit.rules as z_rules
    except ImportError as exc:
        raise RuntimeError(
            "MUFAC legacy ViT LRP dependencies (`lxt`, `zennit`) are unavailable."
        ) from exc

    rule_type = config.get("lrp_rule_type", config.get("rules", ["epsilon"])[0])
    params = config.get("lrp_params", {})
    if rule_type == "gamma":
        gamma_val = float(params.get("gamma", 0.25))
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(gamma_val)),
            (torch.nn.Linear, z_rules.Gamma(gamma_val)),
        ])
    if rule_type == "epsilon":
        epsilon_val = float(params.get("epsilon", 1e-6))
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Epsilon(epsilon_val)),
            (torch.nn.Linear, z_rules.Epsilon(epsilon_val)),
        ])
    if rule_type == "alpha_beta":
        alpha = float(params.get("alpha", 2.0))
        beta = float(params.get("beta", 1.0))
        return LayerMapComposite([
            (torch.nn.Conv2d, z_rules.AlphaBeta(alpha=alpha, beta=beta)),
            (torch.nn.Linear, z_rules.AlphaBeta(alpha=alpha, beta=beta)),
        ])
    if rule_type == "epsilon_gamma_box":
        return EpsilonGammaBox(low=-3.0, high=3.0)
    raise ValueError(f"Unsupported MUFAC LRP rule: {rule_type}")


def _analyze_top_neurons_vit(model, analysis_loader, top_k, input_noise, device):
    counter = []

    for row in analysis_loader:
        x = row[0].to(device)
        x = _add_image_noise(x, input_noise)

        hook = _HeadRelevanceHook()
        hf = model.heads.head.register_forward_hook(hook.fwd)
        hb = model.heads.head.register_full_backward_hook(hook.bwd)

        try:
            model.zero_grad(set_to_none=True)
            y = model(x.requires_grad_())
            pred = y.argmax(dim=1)
            score = y[torch.arange(x.shape[0]), pred].sum()
            score.backward()
            if hook.activations is None or hook.gradients is None:
                continue
            rel = (hook.activations * hook.gradients)[0].abs().detach().cpu().numpy()
            top_indices = np.argsort(rel)[-top_k:][::-1].tolist()
            counter.append(top_indices)
        finally:
            hf.remove()
            hb.remove()

    merged = [idx for row in counter for idx in row]
    counts = Counter(merged)
    return [idx for idx, _ in counts.most_common()]


def _analyze_top_neurons_vit_legacy_lrp(model, analysis_loader, top_k, input_noise, device, config):
    _ensure_vit_zennit_support()
    counter = []

    for row in analysis_loader:
        x = row[0].to(device)
        x = _add_image_noise(x, input_noise)
        if x.shape[0] > 1:
            x = x[0:1]

        hook = _HeadRelevanceHook()
        hf = model.heads.head.register_forward_hook(hook.fwd)
        hb = model.heads.head.register_full_backward_hook(hook.bwd)
        comp = _get_mufac_zennit_composite(config)
        comp.register(model)

        try:
            model.zero_grad(set_to_none=True)
            y = model(x.requires_grad_())
            pred = y.argmax(dim=1)
            y[torch.arange(x.shape[0]), pred].sum().backward()
            if hook.activations is None or hook.gradients is None:
                continue
            rel = (hook.activations * hook.gradients)[0].abs().detach().cpu().numpy()
            top_indices = np.argsort(rel)[-top_k:][::-1].tolist()
            counter.append(top_indices)
        finally:
            comp.remove()
            hf.remove()
            hb.remove()

    merged = [idx for row in counter for idx in row]
    counts = Counter(merged)
    return [idx for idx, _ in counts.most_common()]


def _perturb_vit_head(model, neuron_indices, method, config):
    neurons = neuron_indices[: config["perturb_top_n"]]
    target_class = config["unlearn_class"]
    out_w = model.heads.head.weight.data.clone()

    for n in neurons:
        if n >= out_w.shape[1]:
            continue
        if method == "lra_npp":
            out_w[target_class, n] = 0.0
        elif method == "noise_gn":
            out_w[target_class, n] = _add_gaussian_noise(out_w[target_class, n], config["noise_std"])
        elif method == "noise_ln":
            out_w[target_class, n] = _add_laplace_noise(out_w[target_class, n], config["noise_laplace_scale"])

    model.heads.head.weight.data = out_w
    return model


def _train_one_epoch(model, train_loader, optimizer, loss_fn, device, progress_desc=None):
    model.train()
    correct = 0
    total = 0
    iterator = tqdm(
        train_loader,
        total=len(train_loader),
        desc=progress_desc or "Retrain Train",
        leave=False,
    )
    for row in iterator:
        x, y = row[0].to(device), row[1].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if total > 0:
            iterator.set_postfix(acc=f"{(correct / total):.4f}", loss=f"{float(loss.item()):.4f}")
    return (correct / total) if total > 0 else 0.0


def _make_retrain_loader(full_train_dataset, batch_size, unlearn_class):
    keep_indices = []
    for i in range(len(full_train_dataset)):
        row = full_train_dataset[i]
        y = int(row[1])
        if y != int(unlearn_class):
            keep_indices.append(i)
    subset = Subset(full_train_dataset, keep_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def _reinitialize_model_parameters(model):
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return model


def _build_cifar_retrain_model(ds, device):
    num_classes = 10 if ds == "cifar10" else 100
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def _build_mufac_retrain_model(config, device):
    from torchvision.models import vision_transformer

    weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
    model = vision_transformer.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, int(config["mufac_num_classes"]))
    return model.to(device)


def _build_imagenet_retrain_model(config, num_classes, device):
    vgg_version = int(config.get("vgg_version", 16))
    weights = getattr(torchvision.models, f"VGG{vgg_version}_Weights").IMAGENET1K_V1
    model = getattr(torchvision.models, f"vgg{vgg_version}")(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, int(num_classes))
    return model.to(device)


class _MUFACDataset(Dataset):
    MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    def __init__(self, csv_path, image_dir, transform=None, filter_class=None, keep_only_class=None):
        self.rows = []
        self.image_dir = image_dir
        self.transform = transform

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                age = r["age_class"]
                if age not in self.MAP:
                    continue
                label = self.MAP[age]
                if filter_class is not None and label == int(filter_class):
                    continue
                if keep_only_class is not None and label != int(keep_only_class):
                    continue
                self.rows.append((r["image_path"], label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from PIL import Image

        rel_path, label = self.rows[idx]
        p = pathlib.Path(self.image_dir) / rel_path
        img = Image.open(p.as_posix()).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class _ImageDirAsSingleClassDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = pathlib.Path(root_dir)
        self.label = int(label)
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        self.files = sorted([p for p in self.root_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        p = self.files[idx]
        img = Image.open(p.as_posix()).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label


def _load_mufac_vit(config, device):
    from torchvision.models import vision_transformer

    model = vision_transformer.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, config["mufac_num_classes"])

    checkpoint = torch.load(config["mufac_model_path"], map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model.load_state_dict(checkpoint.state_dict(), strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model


def run_method(method_name, base_model, data_ctx, config, device, seed):
    set_seed(seed)
    start = time.time()
    ds = config["dataset"]

    if method_name == "retrain":
        model = copy.deepcopy(base_model).to(device)
        retrain_cfg = config.get("retrain", {})
        if retrain_cfg.get("from_scratch", True):
            if ds in {"cifar10", "cifar100"}:
                # Match legacy CIFAR retrain baseline:
                # keep ImageNet-pretrained backbone, reset classifier head.
                model = _build_cifar_retrain_model(ds, device)
            elif ds == "imagenet":
                # Match legacy ImageNet retrain baseline:
                # keep pretrained VGG backbone and rebuild the classifier head
                # to the dataset's original class count.
                model = _build_imagenet_retrain_model(
                    config,
                    num_classes=len(data_ctx["train_dataset"].classes),
                    device=device,
                )
            elif ds == "mufac":
                # Match legacy ViT retrain baseline:
                # keep ImageNet-pretrained backbone, reset classifier head.
                model = _build_mufac_retrain_model(config, device)
            else:
                model = _reinitialize_model_parameters(model)
        retrain_loader = _make_retrain_loader(data_ctx["train_dataset"], config["batch_size"], config["unlearn_class"])
        loss_fn = torch.nn.CrossEntropyLoss()
        opt_name = str(retrain_cfg.get("optimizer", "adam")).lower()
        lr = float(retrain_cfg.get("lr", 1e-4))
        if opt_name == "sgd":
            momentum = float(retrain_cfg.get("momentum", 0.9))
            weight_decay = float(retrain_cfg.get("weight_decay", 0.0))
            optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt_name == "adam":
            weight_decay = float(retrain_cfg.get("weight_decay", 0.0))
            optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported retrain optimizer: {opt_name}")

        scheduler = None
        scheduler_cfg = retrain_cfg.get("scheduler")
        if isinstance(scheduler_cfg, dict) and scheduler_cfg.get("type") == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim,
                step_size=int(scheduler_cfg.get("step_size", 30)),
                gamma=float(scheduler_cfg.get("gamma", 0.1)),
            )

        max_epochs = int(retrain_cfg.get("epochs", 1))
        ag_stop_target = retrain_cfg.get("ag_stop_target")
        ag_eval_loader = data_ctx.get("train_eval_loader", data_ctx["train_loader"])
        for epoch_idx in range(max_epochs):
            _train_one_epoch(
                model,
                retrain_loader,
                optim,
                loss_fn,
                device,
                progress_desc=f"Retrain Train [{ds}] {epoch_idx + 1}/{max_epochs}",
            )
            if scheduler is not None:
                scheduler.step()
            if ag_stop_target is not None:
                _, ag = evaluate_at_ag(
                    model,
                    ag_eval_loader,
                    device,
                    config["unlearn_class"],
                    show_progress=False,
                )
                if ag >= float(ag_stop_target):
                    break
        runtime = time.time() - start
        model.eval()
        return model, runtime

    if method_name not in {"lra_npp", "noise_gn", "noise_ln"}:
        raise ValueError(f"Unknown method: {method_name}")

    if ds == "mufac":
        model = copy.deepcopy(base_model).to(device)
        analysis_subset = _select_analysis_subset(
            data_ctx["analysis_dataset"], config["unlearn_class"], config["analysis_sample_size"], seed
        )
        analysis_loader = DataLoader(analysis_subset, batch_size=1, shuffle=False)
        sorted_neurons = _analyze_top_neurons_vit_legacy_lrp(
            model,
            analysis_loader,
            config["analyze_top_n"],
            config["input_noise"],
            device,
            config,
        )
        model = _perturb_vit_head(model, sorted_neurons, method_name, config)
        runtime = time.time() - start
        model.eval()
        return model, runtime

    model = copy.deepcopy(base_model)
    lrp_model = lrp.convert_vgg(model).to(device)

    analysis_subset = _select_analysis_subset(
        data_ctx["analysis_dataset"], config["unlearn_class"], config["analysis_sample_size"], seed
    )
    analysis_loader = DataLoader(analysis_subset, batch_size=1, shuffle=False)

    sorted_neurons = _analyze_top_neurons_lrp(
        lrp_model,
        analysis_loader,
        config["rules"][0],
        config["analyze_top_n"],
        config["input_noise"],
        device,
        analysis_layer=config.get("layer_map", {}).get("analysis_layer", 0),
    )

    if ds == "mnist":
        lrp_model = _perturb_mnist(lrp_model, sorted_neurons, method_name, config)
    elif ds in {"cifar10", "cifar100"}:
        lrp_model = _perturb_resnet_fc(lrp_model, sorted_neurons, method_name, config)
    elif ds == "imagenet":
        lrp_model = _perturb_imagenet_vgg(lrp_model, sorted_neurons, method_name, config)
    else:
        raise NotImplementedError(f"Unknown dataset branch for perturbation: {ds}")

    runtime = time.time() - start
    lrp_model.eval()
    return lrp_model, runtime


def load_dataset_and_model(config, device):
    ds = config["dataset"]

    if ds == "mnist":
        transform = torchvision.transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(
            root="/home/cwh/Workspace/TorchLRP-master/data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="/home/cwh/Workspace/TorchLRP-master/data", train=False, download=True, transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = get_mnist_model().to(device)
        state = torch.load(config["model_path"], map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "analysis_dataset": test_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    if ds in {"cifar10", "cifar100"}:
        if ds == "cifar10":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])
        dcls = torchvision.datasets.CIFAR10 if ds == "cifar10" else torchvision.datasets.CIFAR100
        train_dataset = dcls(root="/home/cwh/Workspace/TorchLRP-master/data", train=True, download=True, transform=transform)
        test_dataset = dcls(root="/home/cwh/Workspace/TorchLRP-master/data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        checkpoint = torch.load(config["model_path"], map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(device)
        elif isinstance(checkpoint, dict):
            model = torchvision.models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 10 if ds == "cifar10" else 100)
            state = checkpoint.get("state_dict", checkpoint)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model = model.to(device)
        else:
            raise TypeError(f"Unsupported CIFAR checkpoint type: {type(checkpoint)}")
        model.eval()
        return model, {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "analysis_dataset": test_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    if ds == "imagenet":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        full_dataset = torchvision.datasets.ImageFolder(root=config["imagenet_full_data_dir"], transform=transform)
        unlearn_dir = pathlib.Path(config["imagenet_unlearn_data_dir"])
        synset_name = unlearn_dir.name
        inferred_target = full_dataset.class_to_idx.get(synset_name)
        if inferred_target is not None and int(config["unlearn_class"]) != int(inferred_target):
            raise ValueError(
                f"ImageNet unlearn_class mismatch: config={config['unlearn_class']} "
                f"but class_to_idx['{synset_name}']={inferred_target}"
            )
        unlearn_dataset = _ImageDirAsSingleClassDataset(
            root_dir=unlearn_dir.as_posix(),
            label=int(config["unlearn_class"]) if inferred_target is None else int(inferred_target),
            transform=transform,
        )

        train_loader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=True)
        test_root = config.get("imagenet_test_data_dir")
        if test_root and os.path.isdir(test_root):
            test_dataset = torchvision.datasets.ImageFolder(root=test_root, transform=transform)
        else:
            test_dataset = full_dataset
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if config.get("use_pretrained", True) else None
        model = torchvision.models.vgg16(weights=weights).to(device)
        model.eval()

        return model, {
            "train_dataset": full_dataset,
            "test_dataset": test_dataset,
            "analysis_dataset": unlearn_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    if ds == "mufac":
        root = pathlib.Path(config["mufac_root"])
        transform = _build_mufac_transform(config)
        eval_transform = _build_mufac_eval_transform(config)

        train_dataset = _MUFACDataset(
            csv_path=(root / "custom_train_dataset.csv").as_posix(),
            image_dir=(root / "train_images").as_posix(),
            transform=transform,
        )
        train_eval_dataset = _MUFACDataset(
            csv_path=(root / "custom_train_dataset.csv").as_posix(),
            image_dir=(root / "train_images").as_posix(),
            transform=eval_transform,
        )
        test_dataset = _MUFACDataset(
            csv_path=(root / "custom_test_dataset.csv").as_posix(),
            image_dir=(root / "test_images").as_posix(),
            transform=eval_transform,
        )
        analysis_dataset = _MUFACDataset(
            csv_path=(root / "custom_test_dataset.csv").as_posix(),
            image_dir=(root / "test_images").as_posix(),
            transform=eval_transform,
            keep_only_class=config["unlearn_class"],
        )

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        train_eval_loader = DataLoader(train_eval_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = _load_mufac_vit(config, device)

        return model, {
            "train_dataset": train_dataset,
            "train_eval_dataset": train_eval_dataset,
            "test_dataset": test_dataset,
            "analysis_dataset": analysis_dataset,
            "train_loader": train_loader,
            "train_eval_loader": train_eval_loader,
            "test_loader": test_loader,
        }

    raise NotImplementedError(f"Dataset '{ds}' loader is not implemented.")
