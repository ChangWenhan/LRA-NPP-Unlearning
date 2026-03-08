import argparse
import json

from core import run_paint


TARGETS = ["mnist", "cifar10", "cifar100", "vgg"]


def main() -> None:
    parser = argparse.ArgumentParser("Unified LRP painting script")
    parser.add_argument("--target", choices=["all", *TARGETS], default="all")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    targets = TARGETS if args.target == "all" else [args.target]
    outputs = [run_paint(target=t, seed=args.seed) for t in targets]
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
