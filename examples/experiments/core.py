import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

# Reuse the unified batch core as the execution engine.
from examples.batch_experiments.core.config import get_config as get_batch_config
from examples.batch_experiments.core.runner import run_experiment as run_batch_experiment


@dataclass
class ExperimentProfile:
    name: str
    batch_dataset_key: str
    exp_name: str
    override: Dict


def _profiles() -> Dict[str, ExperimentProfile]:
    return {
        "mnist": ExperimentProfile(
            name="mnist",
            batch_dataset_key="mnist",
            exp_name="single_mnist",
            override={
                "n_runs": 1,
                "methods": ["lra_npp"],
                "save_models": False,
            },
        ),
        "cifar10": ExperimentProfile(
            name="cifar10",
            batch_dataset_key="cifar10",
            exp_name="single_cifar10",
            override={
                "n_runs": 1,
                "methods": ["lra_npp"],
                "save_models": False,
            },
        ),
        "cifar100": ExperimentProfile(
            name="cifar100",
            batch_dataset_key="cifar100",
            exp_name="single_cifar100",
            override={
                "n_runs": 1,
                "methods": ["lra_npp"],
                "save_models": False,
            },
        ),
        "imagenet": ExperimentProfile(
            name="imagenet",
            batch_dataset_key="imagenet",
            exp_name="single_imagenet",
            override={
                "n_runs": 1,
                "methods": ["lra_npp"],
                "save_models": False,
            },
        ),
        "mufac": ExperimentProfile(
            name="mufac",
            batch_dataset_key="mufac",
            exp_name="single_mufac",
            override={
                "n_runs": 1,
                "methods": ["lra_npp"],
                "save_models": False,
            },
        ),
        # Keep key for compatibility with the old script name.
        "imagenet_vit": ExperimentProfile(
            name="imagenet_vit",
            batch_dataset_key="imagenet",
            exp_name="single_imagenet_vit_compat",
            override={
                "n_runs": 1,
                "methods": ["lra_npp"],
                "save_models": False,
            },
        ),
    }


def list_profiles():
    return sorted(_profiles().keys())


def get_profile(key: str) -> ExperimentProfile:
    profs = _profiles()
    if key not in profs:
        raise ValueError(f"Unknown experiment profile: {key}")
    return profs[key]


def run_profile(key: str, override_json: Optional[str] = None):
    profile = get_profile(key)

    merged_override = dict(profile.override)
    if override_json:
        with open(override_json, "r", encoding="utf-8") as f:
            user_override = json.load(f)
        merged_override.update(user_override)

    # runner expects an override json file; write temp file in output root.
    cfg = get_batch_config(profile.batch_dataset_key)
    out_root = cfg["output_root"]
    os.makedirs(out_root, exist_ok=True)
    tmp_override = os.path.join(out_root, f"_tmp_override_{profile.name}.json")
    with open(tmp_override, "w", encoding="utf-8") as f:
        json.dump(merged_override, f)

    try:
        out_dir = run_batch_experiment(
            dataset_name=profile.batch_dataset_key,
            exp_name=profile.exp_name,
            override_json=tmp_override,
        )
    finally:
        if os.path.exists(tmp_override):
            os.remove(tmp_override)

    return out_dir
