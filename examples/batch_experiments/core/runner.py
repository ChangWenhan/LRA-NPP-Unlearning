import csv
import json
import os
from datetime import datetime

import numpy as np
import torch

from .config import get_config
from .methods import load_dataset_and_model, run_method
from .metrics import evaluate_at_ag
from .mia import compute_fr, compute_fs
from .stats import paired_tests


METRICS = ["At", "Ag", "Fr", "Fs", "G", "Time"]


def _method_label(method_name):
    mapping = {
        "retrain": "Retrain",
        "lra_npp": "LRA-NPP",
        "noise_gn": "Noise-GN",
        "noise_ln": "Noise-LN",
    }
    return mapping.get(method_name, method_name)


def _build_output_dir(config, exp_name=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_name is None:
        exp_name = f"{config['dataset']}_{stamp}"
    out_dir = os.path.join(config["output_root"], config["dataset"], exp_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    return out_dir


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _summarize_runs(rows):
    methods = sorted(set(r["method"] for r in rows))
    out = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        rec = {"method": m}
        for metric in METRICS:
            vals = np.array([float(x[metric]) for x in rs], dtype=float)
            rec[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else 0.0
            rec[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def _collect_pair_stats(rows):
    pairs = [("LRA-NPP", "Retrain"), ("LRA-NPP", "Noise-GN"), ("LRA-NPP", "Noise-LN")]
    recs = []

    for lhs, rhs in pairs:
        lhs_rows = [r for r in rows if r["method"] == lhs]
        rhs_rows = [r for r in rows if r["method"] == rhs]
        rhs_by_seed = {r["seed"]: r for r in rhs_rows}

        common = []
        for lr in lhs_rows:
            seed = lr["seed"]
            if seed in rhs_by_seed:
                common.append((lr, rhs_by_seed[seed]))

        if not common:
            continue

        for metric in METRICS:
            lhs_vals = np.array([float(a[metric]) for a, _ in common], dtype=float)
            rhs_vals = np.array([float(b[metric]) for _, b in common], dtype=float)
            stats_rec = paired_tests(lhs_vals, rhs_vals)
            rec = {"lhs": lhs, "rhs": rhs, "metric": metric}
            rec.update(stats_rec)
            recs.append(rec)

    return recs


def run_experiment(dataset_name, exp_name=None, override_json=None):
    config = get_config(dataset_name)
    if config.get("not_implemented", False):
        raise NotImplementedError(
            f"Dataset '{dataset_name}' is configured but not yet unified in runner."
        )

    if override_json:
        with open(override_json, "r", encoding="utf-8") as f:
            override = json.load(f)
        config.update(override)

    out_dir = _build_output_dir(config, exp_name=exp_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, data_ctx = load_dataset_and_model(config, device)

    run_records = []
    for i in range(config["n_runs"]):
        seed = config["seed_base"] + i * config["seed_stride"]
        for method_name in config["methods"]:
            method_label = _method_label(method_name)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)
            m, runtime = run_method(method_name, model, data_ctx, config, device, seed)
            if torch.cuda.is_available():
                g_gb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            else:
                g_gb = 0.0

            at, ag = evaluate_at_ag(
                m,
                data_ctx["train_loader"],
                device,
                config["unlearn_class"],
                show_progress=True,
                progress_desc=f"Eval At/Ag(train) [{method_label}] seed={seed}",
            )
            fr = compute_fr(
                model,
                m,
                data_ctx["train_loader"],
                data_ctx["test_loader"],
                device,
                config["unlearn_class"],
            )
            fs = compute_fs(m, data_ctx["train_loader"], data_ctx["test_loader"], device, config["unlearn_class"])

            rec = {
                "dataset": config["dataset"],
                "seed": int(seed),
                "run_id": int(i),
                "method": method_label,
                "At": float(at),
                "Ag": float(ag),
                "Fr": float(fr),
                "Fs": float(fs),
                "G": float(g_gb),
                "Time": float(runtime),
            }
            run_records.append(rec)

            if config.get("save_models", False):
                save_path = os.path.join(out_dir, "models", f"{method_label}_seed{seed}.pth")
                torch.save(m, save_path)

            print(
                f"[{config['dataset']}] seed={seed} method={method_label} "
                f"At={at:.4f} Ag={ag:.4f} Fr={fr:.4f} Fs={fs:.4f} G={g_gb:.3f}GB Time={runtime:.2f}s"
            )

    runs_path = os.path.join(out_dir, "runs.csv")
    _write_csv(
        runs_path,
        run_records,
        ["dataset", "seed", "run_id", "method", "At", "Ag", "Fr", "Fs", "G", "Time"],
    )

    summary_records = _summarize_runs(run_records)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_fields = ["method"] + [f"{m}_{k}" for m in METRICS for k in ("mean", "std")]
    _write_csv(summary_path, summary_records, summary_fields)

    pair_records = _collect_pair_stats(run_records)
    pair_path = os.path.join(out_dir, "paired_stats.csv")
    pair_fields = [
        "lhs",
        "rhs",
        "metric",
        "n_runs",
        "mean_diff",
        "std_diff",
        "ci95_low",
        "ci95_high",
        "p_paired_t",
        "p_wilcoxon",
        "cliffs_delta",
        "cohens_dz",
    ]
    _write_csv(pair_path, pair_records, pair_fields)

    print(f"\nSaved runs: {runs_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved paired stats: {pair_path}")

    return out_dir
