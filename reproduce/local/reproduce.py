#!/usr/bin/env python
"""Reproduce one task's submitted ensemble on your own GPU(s).

Trains every group of configs/task_<t>/<recipe>.json that is not finished yet (TAPT, then its
classifiers, one per GPU in parallel), then averages all members and writes the predictions
ZIP, or, with --recipe holdout, scores each group and the ensemble on the fixed holdout.
Finished groups are skipped, so the script can be stopped between groups and run again.

    python reproduce/local/reproduce.py --task a                      # print the plan
    python reproduce/local/reproduce.py --task a --execute            # train everything
    python reproduce/local/reproduce.py --task b --group bag2 --execute
    python reproduce/local/reproduce.py --task a --recipe holdout --execute

Data: the six organizer CSVs in data/raw/ and offenseval_kn.csv in data/external/ (or pass
--search with other folders). Needs a CUDA GPU with at least 14 GB and `pip install -e '.[llm]'`.
HF_TOKEN is optional (google/gemma-4-12B is openly available).
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hastika import reproduce  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["a", "b"], required=True)
    parser.add_argument("--recipe", choices=["submitted", "holdout"], default="submitted")
    parser.add_argument("--group", help="train only this group (see reproduce/sessions.json)")
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/reproduce")
    parser.add_argument("--search", type=Path, nargs="+", default=[ROOT / "data/raw", ROOT / "data/external"])
    parser.add_argument("--gpus", type=int, nargs="+", help="GPU indices to use (default: all visible)")
    parser.add_argument("--execute", action="store_true", help="train; without it, print the plan only")
    args = parser.parse_args()
    if args.execute and reproduce.check_environment():
        sys.exit("Fix the problems above first (or run reproduce/local/check_env.py).")
    if args.group:
        reproduce.run_session(args.task, args.group, args.work, args.recipe, args.gpus, args.execute,
                              args.search, scrub=False)
    else:
        reproduce.run_all(args.task, args.work, args.recipe, args.gpus, args.execute, args.search)


if __name__ == "__main__":
    main()
