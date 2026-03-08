import copy
import csv
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

# Ensure repository root is importable when scripts are run from batch_experiments/.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import lrp
from examples.utils import get_mnist_model


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


def _select_analysis_subset(dataset, target_class, sample_size, seed):
    indices = []
    for i in range(len(dataset)):
        row = dataset[i]
        y = int(row[1])
        if y == int(target_class):
            indices.append(i)

    if len(indices) == 0:
        return Subset(dataset, [])

    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=rng).tolist()
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

    for c in range(out_w.shape[0]):
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


def _train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for row in train_loader:
        x, y = row[0].to(device), row[1].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()


def _make_retrain_loader(full_train_dataset, batch_size, unlearn_class):
    keep_indices = []
    for i in range(len(full_train_dataset)):
        row = full_train_dataset[i]
        y = int(row[1])
        if y != int(unlearn_class):
            keep_indices.append(i)
    subset = Subset(full_train_dataset, keep_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


class _MUFACDataset(Dataset):
    MAP = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    def __init__(self, csv_path, image_dir, transform=None):
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
        retrain_loader = _make_retrain_loader(data_ctx["train_dataset"], config["batch_size"], config["unlearn_class"])
        loss_fn = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=config["retrain"]["lr"])
        for _ in range(config["retrain"]["epochs"]):
            _train_one_epoch(model, retrain_loader, optim, loss_fn, device)
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
        sorted_neurons = _analyze_top_neurons_vit(
            model, analysis_loader, config["analyze_top_n"], config["input_noise"], device
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
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dcls = torchvision.datasets.CIFAR10 if ds == "cifar10" else torchvision.datasets.CIFAR100
        train_dataset = dcls(root="/home/cwh/Workspace/TorchLRP-master/data", train=True, download=True, transform=transform)
        test_dataset = dcls(root="/home/cwh/Workspace/TorchLRP-master/data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = torch.load(config["model_path"], map_location=device, weights_only=False).to(device)
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
        from torch_imagenet import ImageNetDataset

        unlearn_dataset = ImageNetDataset(root_dir=config["imagenet_unlearn_data_dir"], transform=transform)
        full_dataset = torchvision.datasets.ImageFolder(root=config["imagenet_full_data_dir"], transform=transform)

        train_loader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=False)

        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1 if config.get("use_pretrained", True) else None
        model = torchvision.models.vgg16(weights=weights).to(device)
        model.eval()

        return model, {
            "train_dataset": full_dataset,
            "test_dataset": full_dataset,
            "analysis_dataset": unlearn_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    if ds == "mufac":
        root = pathlib.Path(config["mufac_root"])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = _MUFACDataset(
            csv_path=(root / "custom_train_dataset.csv").as_posix(),
            image_dir=(root / "train_images").as_posix(),
            transform=transform,
        )
        test_dataset = _MUFACDataset(
            csv_path=(root / "custom_val_dataset.csv").as_posix(),
            image_dir=(root / "val_images").as_posix(),
            transform=transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = _load_mufac_vit(config, device)

        return model, {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "analysis_dataset": test_dataset,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    raise NotImplementedError(f"Dataset '{ds}' loader is not implemented.")
