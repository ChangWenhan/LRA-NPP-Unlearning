import argparse
import json

from core import run_random_label


TARGETS = ["mnist", "cifar10", "cifar100", "imagenet"]


def main() -> None:
    parser = argparse.ArgumentParser("Unified random-label baselines")
    parser.add_argument("--target", choices=["all", *TARGETS], default="all")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--break_focus_acc", type=float, default=None)
    args = parser.parse_args()

    targets = TARGETS if args.target == "all" else [args.target]
    results = []
    for t in targets:
        results.append(
            run_random_label(
                target=t,
                seed=args.seed,
                epochs=args.epochs,
                break_focus_acc=args.break_focus_acc,
            )
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
