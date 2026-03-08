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

Run from repository root:

```bash
cd /path/to/TorchLRP-master
python examples/batch_experiments/lra-npp-sta-MNIST.py
```

For other datasets:

```bash
python examples/batch_experiments/lra-npp-sta-cifar10.py
python examples/batch_experiments/lra-npp-sta-cifar100.py
python examples/batch_experiments/lra-npp-sta-imagenet.py
python examples/batch_experiments/lra-npp-sta-MUFAC.py
```

## How to Configure Experiments

Each batch script has an `EXPERIMENT_CONFIG` dictionary at the top.
Typical fields to set:

- target class: `unlearn_class`
- model path: `model_path`
- data path: `data_root` (or dataset-specific path fields)
- LRP rule: `rules`
- analysis and perturbation controls:
  - `analysis_sample_sizes`
  - `analyze_top_n_list`
  - `perturb_top_n_list`
  - `perturbation_methods` (`zero`, `gaussian`, `laplace`)
- repeats and saving:
  - `n_repeats`
  - `save_model`, `model_save_dir`
  - `save_neurons`, `neuron_save_dir`
  - `save_excel`, `results_save_dir`

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

Examples:

```bash
python examples/mia/membership_inference_attack_MNIST.py
python examples/mia/membership_inference_attack_CIFAR10.py
```

## Notes

- This repository intentionally excludes large datasets/checkpoints from version control.
- If you use ImageNet-related scripts, configure dataset paths in script configs and `config.ini` where required.
