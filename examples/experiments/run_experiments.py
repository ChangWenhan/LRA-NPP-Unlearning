import argparse
import os
import pathlib
import sys

# Runtime hardening for this workstation
ROOT = pathlib.Path(__file__).resolve().parents[2]
TMP = pathlib.Path.home() / ".tmp_runtime"
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", TMP.as_posix())
sys.path = [p for p in sys.path if "/.local/lib/python" not in p]

if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from examples.experiments.core import list_profiles, run_profile


def main():
    parser = argparse.ArgumentParser("Unified experiments project runner")
    parser.add_argument("--target", default="all", help="Profile name or 'all'")
    parser.add_argument("--override_json", default=None, help="Optional json override")
    args = parser.parse_args()

    targets = list_profiles() if args.target == "all" else [args.target]
    valid = set(list_profiles())
    for t in targets:
        if t not in valid:
            raise ValueError(f"Unknown target: {t}. Valid: {sorted(valid)}")

    for t in targets:
        out = run_profile(t, override_json=args.override_json)
        print(f"[Experiments] {t} finished. output={out}")


if __name__ == "__main__":
    main()
