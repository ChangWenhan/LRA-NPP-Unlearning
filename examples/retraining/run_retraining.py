import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TMP = pathlib.Path.home() / ".tmp_runtime"
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", TMP.as_posix())
sys.path = [p for p in sys.path if "/.local/lib/python" not in p]

import torch

if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from examples.retraining.core import list_profiles, run_profile


def main():
    parser = argparse.ArgumentParser("Unified retraining project runner")
    parser.add_argument("--target", default="all", help="Profile name or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = ROOT.as_posix()
    targets = list_profiles(base) if args.target == "all" else [args.target]
    valid = set(list_profiles(base))
    for t in targets:
        if t not in valid:
            raise ValueError(f"Unknown target: {t}. Valid: {sorted(valid)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for t in targets:
        run_profile(base, t, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
