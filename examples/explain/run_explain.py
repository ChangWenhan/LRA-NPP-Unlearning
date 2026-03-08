import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from core import run_explain


TARGETS = ["mnist", "cifar10", "cifar100", "imagenet", "mufac"]


def main() -> None:
    parser = argparse.ArgumentParser("Unified explanation-neuron extraction runner")
    parser.add_argument("--target", choices=["all", *TARGETS], default="all")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=None)
    parser.add_argument("--seed_stride", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--override_json", type=str, default=None)
    parser.add_argument("--analysis_sample_size", type=int, default=None)
    parser.add_argument("--analyze_top_n", type=int, default=None)
    parser.add_argument("--rule", type=str, default=None)
    parser.add_argument("--input_noise", type=float, default=None)
    args = parser.parse_args()

    targets = TARGETS if args.target == "all" else [args.target]
    outputs = []
    for t in targets:
        outputs.append(
            run_explain(
                dataset=t,
                n_runs=args.n_runs,
                seed_base=args.seed_base,
                seed_stride=args.seed_stride,
                output_root=args.output_root,
                override_json=args.override_json,
                analysis_sample_size=args.analysis_sample_size,
                analyze_top_n=args.analyze_top_n,
                rule=args.rule,
                input_noise=args.input_noise,
            )
        )

    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
