from pathlib import Path

BASE = Path(__file__).resolve().parent

RETRAIN_SCRIPTS = {
    "mnist": BASE / "retrain_MNIST_sample.py",
    "cifar10": BASE / "retrain_cifar10.py",
    "cifar100": BASE / "retrain_cifar100.py",
    "vgg": BASE / "retrain_vgg_class.py",
    "vit": BASE / "retrain_ViT.py",
}


def list_targets():
    return sorted(RETRAIN_SCRIPTS.keys())
