"""Stage the organizer files for the experiment notebooks.

The notebooks read organizer CSVs from data/raw/, the external corpus from
data/external/, and the fixed holdouts from data/derived/. None of these are in
the repository. This module copies authorized copies into place, checks them
against data/sources.json, and rebuilds everything derived from them:

  * the validation-phase input files, binary_validation_inputs.csv and
    multiclass_validation_inputs.csv. They are the released labelled validation
    files without their label column (same ids, same comments, same order).
  * the combined train + validation sets and fixed 15% holdouts for both tasks,
    written by hastika.data.splits.prepare and checked against the recorded
    row counts and holdout fingerprints.

    python -m hastika.research.data --source /kaggle/input/<your-dataset>

Without --source it searches /kaggle/input when that exists, and otherwise
expects the files to be in data/raw/ and data/external/ already.
"""
import argparse
import json
import pathlib
import shutil

import pandas as pd

from hastika.common.paths import PROJECT_ROOT
from hastika.common.tasks import FILES, TASKS
from hastika.data.splits import prepare, sha256

RAW = PROJECT_ROOT / "data/raw"
EXTERNAL = PROJECT_ROOT / "data/external"
DERIVED = PROJECT_ROOT / "data/derived"
PHASE_INPUTS = {"a": "binary_validation_inputs.csv", "b": "multiclass_validation_inputs.csv"}


def _find(name, roots):
    for root in roots:
        hits = sorted(pathlib.Path(root).rglob(name))
        if hits:
            return hits[0]
    return None


def write_phase_inputs(task, raw_dir):
    """Write the validation-phase input file: the released validation file without labels."""
    raw_dir = pathlib.Path(raw_dir)
    validation = pd.read_csv(raw_dir / FILES[task]["validation"], encoding="utf-8-sig", dtype=str,
                             keep_default_na=False)
    out = raw_dir / PHASE_INPUTS[task]
    validation.drop(columns=[TASKS[task][1]]).to_csv(out, index=False)
    return out


def stage(source=None, verbose=True):
    """Copy, verify and derive every data file the notebooks read. Returns a report."""
    manifest = json.loads((PROJECT_ROOT / "data/sources.json").read_text())
    roots = [source] if source else ([pathlib.Path("/kaggle/input")]
                                      if pathlib.Path("/kaggle/input").is_dir() else [])
    wanted = [(RAW / name, spec["sha256"]) for name, spec in manifest["files"].items()]
    wanted.append((EXTERNAL / manifest["external"]["filename"], manifest["external"]["sha256"]))
    for dest, digest in wanted:
        if not dest.exists():
            found = _find(dest.name, roots)
            if found is None:
                raise FileNotFoundError(f"{dest.name}: not under {roots or 'data/'}; see data/README.md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, dest)
        if sha256(dest) != digest:
            raise ValueError(f"{dest.name}: checksum differs from data/sources.json")

    report = {}
    for task in FILES:
        write_phase_inputs(task, RAW)
        report[task] = prepare(task, RAW, DERIVED / f"task_{task}_combined_holdout")
        if verbose:
            print(f"task {task}: {report[task]['labelled_rows']} labelled rows, holdout "
                  f"{report[task]['holdout_rows']} ({report[task]['holdout_fingerprint']})", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path,
                        help="directory holding the organizer CSVs and offenseval_kn.csv")
    stage(parser.parse_args().source)


if __name__ == "__main__":
    main()
