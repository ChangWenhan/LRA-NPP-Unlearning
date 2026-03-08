import copy
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import lrp
from examples.batch_experiments.core.config import get_config
from examples.batch_experiments.core.methods import (
    _analyze_top_neurons_lrp,
    _analyze_top_neurons_vit,
    _select_analysis_subset,
    load_dataset_and_model,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEURON_ROOT = REPO_ROOT / "neuron"


@dataclass
class ExplainRunResult:
    dataset: str
    seed: int
    unlearn_class: int
    top_n: int
    sample_size: int
    rule: str
    output_path: str


def _method_dir_name(dataset: str) -> str:
    if dataset == "cifar10":
        return "cifar-10"
    if dataset == "cifar100":
        return "cifar-100"
    return dataset


def _save_neuron_indices(path: Path, indices: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.as_posix(), "wb") as f:
        pickle.dump(indices, f)


def _write_summary_csv(path: Path, rows: List[ExplainRunResult]) -> None:
    fields = ["dataset", "seed", "unlearn_class", "top_n", "sample_size", "rule", "output_path"]
    with open(path.as_posix(), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "dataset": r.dataset,
                    "seed": r.seed,
                    "unlearn_class": r.unlearn_class,
                    "top_n": r.top_n,
                    "sample_size": r.sample_size,
                    "rule": r.rule,
                    "output_path": r.output_path,
                }
            )


def _load_override_json(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_explain(
    dataset: str,
    n_runs: int = 1,
    seed_base: Optional[int] = None,
    seed_stride: Optional[int] = None,
    output_root: Optional[str] = None,
    override_json: Optional[str] = None,
    analysis_sample_size: Optional[int] = None,
    analyze_top_n: Optional[int] = None,
    rule: Optional[str] = None,
    input_noise: Optional[float] = None,
) -> Dict:
    cfg = get_config(dataset)
    cfg.update(_load_override_json(override_json))

    if seed_base is None:
        seed_base = int(cfg.get("seed_base", 42))
    if seed_stride is None:
        seed_stride = int(cfg.get("seed_stride", 100))

    sample_size = int(analysis_sample_size if analysis_sample_size is not None else cfg.get("analysis_sample_size", 36))
    top_n = int(analyze_top_n if analyze_top_n is not None else cfg.get("perturb_top_n", 80))
    use_rule = str(rule if rule is not None else cfg.get("rules", ["epsilon"])[0])
    use_input_noise = float(input_noise if input_noise is not None else cfg.get("input_noise", 0.0))

    out_root = Path(output_root) if output_root else DEFAULT_NEURON_ROOT
    out_dataset_dir = out_root / _method_dir_name(cfg["dataset"])
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model, data_ctx = load_dataset_and_model(cfg, device)

    results: List[ExplainRunResult] = []
    for i in range(n_runs):
        seed = seed_base + i * seed_stride
        set_seed(seed)

        subset = _select_analysis_subset(data_ctx["analysis_dataset"], cfg["unlearn_class"], sample_size, seed)
        loader = DataLoader(subset, batch_size=1, shuffle=False)

        if cfg["dataset"] == "mufac":
            model = copy.deepcopy(base_model).to(device)
            indices = _analyze_top_neurons_vit(model, loader, top_n, use_input_noise, device)
        else:
            model = copy.deepcopy(base_model)
            lrp_model = lrp.convert_vgg(model).to(device)
            indices = _analyze_top_neurons_lrp(
                lrp_model,
                loader,
                use_rule,
                top_n,
                use_input_noise,
                device,
                analysis_layer=cfg.get("layer_map", {}).get("analysis_layer", 0),
            )

        out_path = out_dataset_dir / f"class_{cfg['unlearn_class']}_seed_{seed}.pkl"
        _save_neuron_indices(out_path, indices)

        rec = ExplainRunResult(
            dataset=cfg["dataset"],
            seed=seed,
            unlearn_class=int(cfg["unlearn_class"]),
            top_n=top_n,
            sample_size=sample_size,
            rule=use_rule,
            output_path=out_path.as_posix(),
        )
        results.append(rec)

        print(
            f"[explain] dataset={cfg['dataset']} seed={seed} class={cfg['unlearn_class']} "
            f"top_n={top_n} analyzed={sample_size} saved={out_path}"
        )

    summary_path = out_dataset_dir / "summary.csv"
    _write_summary_csv(summary_path, results)

    return {
        "dataset": cfg["dataset"],
        "runs": n_runs,
        "summary_csv": summary_path.as_posix(),
        "files": [r.output_path for r in results],
    }
