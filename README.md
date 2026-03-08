# Zero-shot Class Unlearning via LRA-NPP

This repository contains the implementation used in the paper:
**Zero-shot Class Unlearning via Layer-wise Relevance Analysis and Neuronal Path Perturbation (LRA-NPP)**.

The project is built on top of a PyTorch LRP implementation and extends it with:

- class-level unlearning pipelines
- batch experiment runners across datasets
- MIA-based forgetting evaluation
- retraining and random-label baselines

## Project Layout

- `lrp/`: LRP core operators and tracing utilities.
- `examples/experiments/`: single-dataset experiment scripts.
- `examples/batch_experiments/`: batch runners with repeated runs and statistics.
- `examples/mia/`: standalone MIA evaluation scripts.
- `examples/retraining/`: retrain-from-scratch baselines.
- `examples/random_labels/`: random-label baselines.

## Environment

Use conda:

```bash
conda env create -f requirements.yml
conda activate torchlrp
```

## Quick Start

Run from repository root using the unified entrypoint:

```bash
python examples/batch_experiments/run_batch.py --dataset mnist
python examples/batch_experiments/run_batch.py --dataset cifar10
python examples/batch_experiments/run_batch.py --dataset cifar100
```

Current unified runner support:

- `mnist`, `cifar10`, `cifar100`, `imagenet`, `mufac`

## How to Configure Experiments

Default configs are in:

- `examples/batch_experiments/core/config.py`

You can override by JSON:

```bash
python examples/batch_experiments/run_batch.py --dataset mnist --override_json path/to/override.json
```

Typical fields:

- target class: `unlearn_class`
- model path: `model_path`
- data path: `data_root` (or dataset-specific path fields)
- LRP rule: `rules`
- analysis and perturbation controls:
  - `analysis_sample_sizes`
  - `analyze_top_n_list`
  - `perturb_top_n_list`
  - `perturbation_methods` (`zero`, `gaussian`, `laplace`)
- run controls:
  - `n_runs`
  - `seed_base`
  - `seed_stride`
- methods:
  - `methods = ["retrain", "lra_npp", "noise_gn", "noise_ln"]`
- saving:
  - `save_models`
  - `output_root`

## Baselines Used

- Retrain from scratch: scripts in `examples/retraining/`
- Neuronal path noise baselines:
  - GN: `perturbation_methods=['gaussian']`
  - LN: `perturbation_methods=['laplace']`
- LRA-NPP (main method):
  - `perturbation_methods=['zero']` with LRP-guided neuron selection

## MIA Evaluation

You can either:

- run MIA integrated in batch scripts (`MIA`, `FS` fields), or
- run standalone scripts in `examples/mia/`.

Unified MIA entrypoint:

```bash
python examples/mia/run_mia.py --target all
python examples/mia/run_mia.py --target mnist
```

## Retraining

Unified retraining entrypoint:

```bash
python examples/retraining/run_retraining.py --target all
python examples/retraining/run_retraining.py --target cifar10
```

## Outputs

Each experiment writes to:

- `examples/batch_experiments/outputs/{dataset}/{exp_name}/runs.csv`
- `examples/batch_experiments/outputs/{dataset}/{exp_name}/summary.csv`
- `examples/batch_experiments/outputs/{dataset}/{exp_name}/paired_stats.csv`

Recorded run-level metrics:

- `At`, `Ag`, `Fr`, `Fs`, `G` (GPU memory GB), `Time`

Paired statistics are computed at run/seed level for:

- `LRA-NPP vs Retrain`
- `LRA-NPP vs Noise-GN`
- `LRA-NPP vs Noise-LN`

With:

- paired t-test p-value
- Wilcoxon p-value
- Cliff's Delta (`Δ`)
- Cohen's dz (`d`)
- 95% CI of paired mean difference

## Notes

- This repository intentionally excludes large datasets/checkpoints from version control.
- If you use ImageNet-related scripts, configure dataset paths in script configs and `config.ini` where required.
