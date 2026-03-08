from pathlib import Path

BASE = Path(__file__).resolve().parent

MIA_SCRIPTS = {
    "mnist": BASE / "membership_inference_attack_MNIST.py",
    "cifar10": BASE / "membership_inference_attack_CIFAR10.py",
    "cifar100": BASE / "membership_inference_attack_CIFAR100.py",
    "mufac": BASE / "membership_inference_attack_MUFAC.py",
    "vgg": BASE / "membership_inference_attack_VGG.py",
    "vgg_class": BASE / "membership_inference_attack_VGG_class.py",
}


def list_targets():
    return sorted(MIA_SCRIPTS.keys())
