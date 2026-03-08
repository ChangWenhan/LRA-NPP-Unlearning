import argparse

from core.runner import run_experiment


def main():
    parser = argparse.ArgumentParser("Unified batch experiments for LRA-NPP")
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "cifar100", "imagenet", "mufac"])
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--override_json", default=None, help="Optional JSON file to override default config.")
    args = parser.parse_args()

    run_experiment(args.dataset, exp_name=args.exp_name, override_json=args.override_json)


if __name__ == "__main__":
    main()
