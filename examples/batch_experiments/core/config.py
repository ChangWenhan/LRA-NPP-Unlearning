import os
from copy import deepcopy


BASE_DIR = "/home/cwh/Workspace/TorchLRP-master"


def _p(*parts):
    return os.path.join(BASE_DIR, *parts)


DATASET_CONFIGS = {
    "mnist": {
        "dataset": "mnist",
        "unlearn_class": 1,
        "model_path": _p("examples", "models", "mnist_model.pth"),
        "output_root": _p("examples", "batch_experiments", "outputs"),
        "batch_size": 64,
        "analysis_sample_size": 36,
        "analyze_top_n": 50,
        "perturb_top_n": 80,
        "rules": ["epsilon"],
        "input_noise": 0.0,
        "n_runs": 5,
        "seed_base": 42,
        "seed_stride": 100,
        "methods": ["retrain", "lra_npp", "noise_gn", "noise_ln"],
        "noise_std": 1.0,
        "noise_laplace_scale": 1.0,
        "retrain": {"epochs": 3, "lr": 1e-3},
        "layer_map": {
            "analysis_layer": 0,
            "output_layer_idx": 8,
            "secondary_layer_idx": 6,
            "target_class_only": True,
        },
        "save_models": False,
    },
    "cifar10": {
        "dataset": "cifar10",
        "unlearn_class": 9,
        "model_path": _p("examples", "models", "resnet50_cifar10_epoch_10.pth"),
        "output_root": _p("examples", "batch_experiments", "outputs"),
        "batch_size": 64,
        "analysis_sample_size": 36,
        "analyze_top_n": 200,
        "perturb_top_n": 400,
        "rules": ["epsilon"],
        "input_noise": 0.0,
        "n_runs": 5,
        "seed_base": 42,
        "seed_stride": 100,
        "methods": ["retrain", "lra_npp", "noise_gn", "noise_ln"],
        "noise_std": 1.0,
        "noise_laplace_scale": 1.0,
        "retrain": {"epochs": 3, "lr": 1e-4},
        "layer_map": {
            "analysis_layer": 0,
            "output_layer_idx": 22,
            "secondary_layer_idx": None,
            "target_class_only": False,
        },
        "save_models": False,
    },
    "cifar100": {
        "dataset": "cifar100",
        "unlearn_class": 9,
        "model_path": _p("examples", "models", "resnet50_cifar100_5.pth"),
        "output_root": _p("examples", "batch_experiments", "outputs"),
        "batch_size": 64,
        "analysis_sample_size": 36,
        "analyze_top_n": 150,
        "perturb_top_n": 250,
        "rules": ["epsilon"],
        "input_noise": 0.0,
        "n_runs": 5,
        "seed_base": 42,
        "seed_stride": 100,
        "methods": ["retrain", "lra_npp", "noise_gn", "noise_ln"],
        "noise_std": 1.0,
        "noise_laplace_scale": 1.0,
        "retrain": {"epochs": 3, "lr": 1e-4},
        "layer_map": {
            "analysis_layer": 0,
            "output_layer_idx": 22,
            "secondary_layer_idx": None,
            "target_class_only": False,
        },
        "save_models": False,
    },
    "imagenet": {
        "dataset": "imagenet",
        "unlearn_class": 0,
        "output_root": _p("examples", "batch_experiments", "outputs"),
        "methods": ["retrain", "lra_npp", "noise_gn", "noise_ln"],
        "not_implemented": True,
    },
    "mufac": {
        "dataset": "mufac",
        "unlearn_class": 0,
        "output_root": _p("examples", "batch_experiments", "outputs"),
        "methods": ["retrain", "lra_npp", "noise_gn", "noise_ln"],
        "not_implemented": True,
    },
}


def get_config(dataset_name: str):
    key = dataset_name.lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return deepcopy(DATASET_CONFIGS[key])
