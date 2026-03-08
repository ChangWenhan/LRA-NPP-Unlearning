import copy
import pathlib
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset

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


def _select_analysis_subset(dataset, target_class, sample_size, seed):
    indices = [i for i, (_, y) in enumerate(dataset) if int(y) == int(target_class)]
    if len(indices) == 0:
        return Subset(dataset, [])
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=rng).tolist()
    selected = [indices[i] for i in order[: min(sample_size, len(indices))]]
    return Subset(dataset, selected)


def _analyze_top_neurons(lrp_model, analysis_loader, rule, top_k, input_noise, device, analysis_layer=0):
    counter = []
    for x, _ in analysis_loader:
        x = x.to(device)
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


def _add_gaussian_noise(tensor, std):
    return tensor + torch.randn_like(tensor) * std


def _add_laplace_noise(tensor, scale):
    lap = torch.distributions.laplace.Laplace(torch.tensor(0.0, device=tensor.device), torch.tensor(scale, device=tensor.device))
    noise = lap.sample(tensor.shape)
    return tensor + noise


def _perturb_model(lrp_model, neuron_indices, method, config):
    layer_map = config["layer_map"]
    output_idx = layer_map["output_layer_idx"]
    secondary_idx = layer_map.get("secondary_layer_idx")
    target_class_only = layer_map.get("target_class_only", False)
    target_class = config["unlearn_class"]

    out_w = lrp_model[output_idx].weight.data.clone()
    neurons = neuron_indices[: config["perturb_top_n"]]

    if target_class_only:
        class_ids = [target_class]
    else:
        class_ids = list(range(out_w.shape[0]))

    for c in class_ids:
        for n in neurons:
            if method == "lra_npp":
                out_w[c, n] = 0.0
            elif method == "noise_gn":
                out_w[c, n] = _add_gaussian_noise(out_w[c, n], config["noise_std"])
            elif method == "noise_ln":
                out_w[c, n] = _add_laplace_noise(out_w[c, n], config["noise_laplace_scale"])
            else:
                raise ValueError(f"unknown perturb method {method}")

    lrp_model[output_idx].weight.data = out_w

    if secondary_idx is not None:
        sec_w = lrp_model[secondary_idx].weight.data.clone()
        for n in neurons:
            if method == "lra_npp":
                sec_w[n] = torch.zeros_like(sec_w[n])
            elif method == "noise_gn":
                sec_w[n] = _add_gaussian_noise(sec_w[n], config["noise_std"])
            elif method == "noise_ln":
                sec_w[n] = _add_laplace_noise(sec_w[n], config["noise_laplace_scale"])
        lrp_model[secondary_idx].weight.data = sec_w

    return lrp_model


def _train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()


def _make_retrain_loader(full_train_dataset, batch_size, unlearn_class):
    keep_indices = [i for i, (_, y) in enumerate(full_train_dataset) if int(y) != int(unlearn_class)]
    subset = Subset(full_train_dataset, keep_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def run_method(method_name, base_model, data_ctx, config, device, seed):
    set_seed(seed)
    start = time.time()

    if method_name == "retrain":
        model = copy.deepcopy(base_model).to(device)
        retrain_loader = _make_retrain_loader(
            data_ctx["train_dataset"], config["batch_size"], config["unlearn_class"]
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=config["retrain"]["lr"])
        for _ in range(config["retrain"]["epochs"]):
            _train_one_epoch(model, retrain_loader, optim, loss_fn, device)
        runtime = time.time() - start
        return model, runtime

    if method_name in {"lra_npp", "noise_gn", "noise_ln"}:
        model = copy.deepcopy(base_model)
        lrp_model = lrp.convert_vgg(model).to(device)
        rule = config["rules"][0]

        analysis_subset = _select_analysis_subset(
            data_ctx["analysis_dataset"],
            config["unlearn_class"],
            config["analysis_sample_size"],
            seed,
        )
        analysis_loader = DataLoader(analysis_subset, batch_size=1, shuffle=False)

        sorted_neurons = _analyze_top_neurons(
            lrp_model,
            analysis_loader,
            rule,
            config["analyze_top_n"],
            config["input_noise"],
            device,
            analysis_layer=config["layer_map"]["analysis_layer"],
        )

        lrp_model = _perturb_model(lrp_model, sorted_neurons, method_name, config)
        runtime = time.time() - start
        return lrp_model, runtime

    raise ValueError(f"Unknown method: {method_name}")


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

    raise NotImplementedError(
        f"Dataset '{ds}' unified runner is not implemented yet. "
        "Use existing dataset-specific scripts or extend load_dataset_and_model."
    )
