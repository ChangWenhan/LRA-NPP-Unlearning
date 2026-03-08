import argparse
import subprocess
import sys
from pathlib import Path

from registry import MIA_SCRIPTS, list_targets


def run_one(target: str, python_bin: str, dry_run: bool = False):
    script = MIA_SCRIPTS[target]
    cmd = [python_bin, script.as_posix()]
    print(f"[MIA] Running: {' '.join(cmd)}")
    if dry_run:
        return 0
    ret = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2].as_posix())
    return ret.returncode


def main():
    parser = argparse.ArgumentParser("Unified MIA runner")
    parser.add_argument("--target", default="all", help=f"One of {list_targets()} or 'all'.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = list_targets() if args.target == "all" else [args.target]
    for t in targets:
        if t not in MIA_SCRIPTS:
            raise ValueError(f"Unknown MIA target: {t}")

    failed = []
    for t in targets:
        code = run_one(t, args.python, args.dry_run)
        if code != 0:
            failed.append((t, code))

    if failed:
        msg = ", ".join([f"{t}(code={c})" for t, c in failed])
        raise SystemExit(f"MIA run failed: {msg}")

    print("[MIA] All requested targets finished successfully.")


if __name__ == "__main__":
    main()
