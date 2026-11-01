"""Reproduce the submitted ensembles: stage data, train one group per session, combine, package.

The submitted systems are Gemma-4-12B TAPT ensembles (12 models for Task A, 8 for Task B) built
in groups: one TAPT run, then two or four classifiers from its adapter. Each group fits in one
Kaggle session on two T4 GPUs. This module drives `hastika.runner` for that workflow and is what
the notebooks in `reproduce/kaggle/` and the scripts in `reproduce/local/` call.

    hastika-reproduce check
    hastika-reproduce smoke   --task a                               # ~30 min, every stage
    hastika-reproduce stage   --task a --work WORK
    hastika-reproduce session --task a --group bag1 --work WORK --execute
    hastika-reproduce collect --task a --search /kaggle/input --work WORK
    hastika-reproduce finish  --task a --work WORK
    hastika-reproduce all     --task a --work WORK --execute        # every group, then finish

`--recipe holdout` trains the same groups on the 85% split and scores them on the fixed 15%
holdout, the only check a rerun can be given without test labels.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from hastika.common.paths import PROJECT_ROOT
from hastika.common.tasks import FILES

SESSIONS = PROJECT_ROOT / "reproduce/sessions.json"
DEFAULT_SEARCH = [Path("/kaggle/input"), PROJECT_ROOT / "data/raw", PROJECT_ROOT / "data/external"]
EXPECTED_VERSIONS = {"transformers": "5.16.1", "peft": "0.20.0"}
RECIPES = ("submitted", "holdout", "smoke")


def load_sessions():
    return json.loads(SESSIONS.read_text())


def recipe_path(task, recipe):
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {RECIPES}")
    return PROJECT_ROOT / f"configs/task_{task}/{recipe}.json"


def output_root(work, task, recipe):
    return Path(work) / f"task_{task}_{recipe}"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def required_files(task):
    manifest = json.loads((PROJECT_ROOT / "data/sources.json").read_text())
    spec = FILES[task]
    raw = {name: manifest["files"][name]["sha256"] for name in (spec["train"], spec["validation"], spec["test"])}
    external = manifest["external"]
    return raw, (external["filename"], external["sha256"])


def find_file(name, digest, roots):
    """First file called `name` under any root whose SHA-256 matches the recorded release."""
    seen = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates = [root / name] if (root / name).is_file() else []
        candidates += sorted(p for p in root.rglob(name) if p.is_file() and p != root / name)
        for path in candidates:
            if sha256(path) == digest:
                return path
            seen.append(str(path))
    detail = f"; found with a different checksum: {', '.join(seen)}" if seen else ""
    raise FileNotFoundError(f"{name} not found under {', '.join(map(str, roots))}{detail}")


def stage(task, dest, roots=None):
    """Copy the task's organizer files and the external corpus into `dest`, checksum-verified."""
    roots = roots or DEFAULT_SEARCH
    raw_files, (ext_name, ext_digest) = required_files(task)
    dest = Path(dest)
    (dest / "raw").mkdir(parents=True, exist_ok=True)
    (dest / "external").mkdir(parents=True, exist_ok=True)
    for name, digest in raw_files.items():
        target = dest / "raw" / name
        if not (target.is_file() and sha256(target) == digest):
            shutil.copyfile(find_file(name, digest, roots), target)
    external = dest / "external" / ext_name
    if not (external.is_file() and sha256(external) == ext_digest):
        shutil.copyfile(find_file(ext_name, ext_digest, roots), external)
    print(f"Staged Task {task.upper()} data in {dest} (checksums match the recorded release)")
    return dest / "raw", external


def group_ids(task, recipe="submitted"):
    return [g["id"] for g in json.loads(recipe_path(task, recipe).read_text())["groups"]]


def group_status(out, group, recipe_hash):
    """'missing', 'complete' (bundle for this recipe) or 'partial' (anything else)."""
    directory = Path(out) / group
    if not directory.exists():
        return "missing"
    meta = directory / "group_metadata.json"
    if meta.is_file():
        provenance = json.loads(meta.read_text()).get("provenance", {})
        if provenance.get("recipe_hash") == recipe_hash and (directory / "group_probs.npy").is_file():
            return "complete"
    return "partial"


def default_gpus():
    try:
        import torch
        return list(range(torch.cuda.device_count())) or None
    except ImportError:
        return None


def run_session(task, group, work, recipe="submitted", gpus=None, execute=False, roots=None, scrub=True):
    """Train one group. Without `execute`, print the plan only."""
    from hastika import runner
    config = runner.load_recipe(recipe_path(task, recipe))
    if group not in group_ids(task, recipe):
        raise ValueError(f"Unknown group {group!r}; choose from {group_ids(task, recipe)}")
    out = output_root(work, task, recipe)
    raw, external = stage(task, Path(work) / "data", roots)
    if not execute:
        plan = runner.commands(config, out, raw, {group})
        print(json.dumps({"execute": False, "group": group, "jobs": plan}, indent=2))
        return None
    status = group_status(out, group, runner.recipe_hash(config))
    if status == "complete":
        print(f"{group}: already complete in {out}; nothing to do")
    elif status == "partial":
        raise FileExistsError(f"{out / group} holds an unfinished run; move it aside and rerun this session")
    else:
        gpus = gpus if gpus is not None else default_gpus()
        runner.execute(config, out, raw, external, {group}, gpus)
    if recipe == "holdout":
        if not (out / "data/holdout.csv").is_file():
            from hastika.data.splits import prepare
            prepare(task, raw, out / "data")
        print(json.dumps(score_groups(task, out, [group]), indent=2))
    if scrub:
        scrub_data(work, out)
    return out / group


def scrub_data(work, out):
    """Remove copies of organizer and external comment data from a session's output."""
    for path in [Path(work) / "data", Path(out) / "data", Path(out) / "external_text.csv"]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    print("Removed staged comment data from the output; group results are kept")


def collect(task, work, recipe="submitted", roots=None):
    """Copy finished group folders (for example other sessions' Kaggle outputs) into one root."""
    from hastika import runner
    config = runner.load_recipe(recipe_path(task, recipe))
    expected = runner.recipe_hash(config)
    out = output_root(work, task, recipe)
    out.mkdir(parents=True, exist_ok=True)
    found = {}
    for root in roots or DEFAULT_SEARCH:
        root = Path(root)
        if not root.exists():
            continue
        for meta in sorted(root.rglob("group_metadata.json")):
            directory = meta.parent
            gid = directory.name
            if gid in found or gid not in group_ids(task, recipe) or directory.resolve() == (out / gid).resolve():
                continue
            if group_status(directory.parent, gid, expected) == "complete":
                found[gid] = directory
    for gid, directory in found.items():
        target = out / gid
        if group_status(out, gid, expected) != "complete":
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(directory, target)
    missing = [g for g in group_ids(task, recipe) if group_status(out, g, expected) != "complete"]
    print(f"Collected {len(found)} group(s) into {out}; missing: {missing or 'none'}")
    return missing


def score_groups(task, out, groups=None, final=False):
    """Holdout macro-F1 of each finished group (and the full ensemble) against the recorded runs."""
    from hastika.evaluation.metrics import evaluate
    recorded = load_sessions()[task]["recorded_holdout"]
    gold = Path(out) / "data/holdout.csv"
    rows = []
    names = list(groups or group_ids(task)) + (["final"] if final else [])
    for name in names:
        prefix = Path(out) / ("final" if name == "final" else f"{name}/group")
        if not Path(str(prefix) + "_probs.npy").is_file():
            continue
        result = evaluate(task, gold, prefix)
        rows.append({"group": name, "members": result["members"], "macro_f1": round(result["macro_f1"], 4),
                     "accuracy": round(result["accuracy"], 4), "recorded": recorded.get(name)})
    return rows


def finish(task, work, recipe="submitted", roots=None):
    """Combine every group, then package the test predictions or score the holdout."""
    from hastika import runner
    from hastika.data.splits import prepare
    from hastika.evaluation.ensemble import package, read_bundle
    import numpy as np
    config = runner.load_recipe(recipe_path(task, recipe))
    out = output_root(work, task, recipe)
    missing = [g for g in group_ids(task, recipe) if group_status(out, g, runner.recipe_hash(config)) != "complete"]
    if missing:
        raise FileNotFoundError(f"Groups not finished in {out}: {missing}")
    runner.assemble(config, out)
    raw, _ = stage(task, Path(work) / "data", roots)
    probs, ids, meta = read_bundle(out / "final", task)
    report = {"task": task, "recipe": recipe, "members": meta["member_count"], "recipe_hash": runner.recipe_hash(config)}
    if recipe in ("submitted", "smoke"):
        archive = Path(work) / (f"task_{task}_predictions.zip" if recipe == "submitted"
                                else f"task_{task}_smoke_predictions_NOT_A_SUBMISSION.zip")
        package(out / "final", raw / FILES[task]["test"], archive)
        counts = np.bincount(probs.argmax(1), minlength=len(meta["classes"]))
        report["predictions_zip"] = str(archive)
        report["predicted_share"] = {c: round(float(n) / len(ids), 4) for c, n in zip(meta["classes"], counts)}
        report["note"] = ("Smoke test: two models trained one epoch on a small sample. This checks that "
                          "every stage runs; its predictions are not meant to be accurate."
                          if recipe == "smoke" else
                          "The organizers scored the final portion of the test file and reported macro-F1 "
                          f"{load_sessions()[task]['official_macro_f1']} for the original submission; "
                          "a rerun cannot be scored locally without test labels.")
    else:
        prepare(task, raw, out / "data")
        report["holdout"] = score_groups(task, out, final=True)
        shutil.rmtree(out / "data")
    (Path(work) / f"task_{task}_{recipe}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.rmtree(Path(work) / "data", ignore_errors=True)
    print(json.dumps(report, indent=2))
    return report


def check_reload(task, work, recipe="smoke", gpus=None):
    """Reload one saved classifier and check it reproduces its training-time test predictions."""
    from hastika import runner
    from hastika.evaluation.ensemble import read_bundle
    import numpy as np
    out = output_root(work, task, recipe)
    group = group_ids(task, recipe)[0]
    execution = json.loads((out / group / "execution.json").read_text())
    first = next(r for r in execution if r["stage"] == "classifier")
    checkpoint = Path(first["executed_command"][first["executed_command"].index("--out") + 1])
    stem = FILES[task]["test"].removesuffix(".csv")
    raw = Path(work) / "data" / "raw"
    reloaded = out / "reload_check" / stem
    reloaded.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "hastika.inference.predict", "--checkpoint", str(checkpoint),
               "--input", str(raw / FILES[task]["test"]), "--out", str(reloaded)]
    gpu = (gpus or default_gpus() or [None])[0]
    if runner.launch(command, out / "reload_check" / "predict.log", gpu):
        raise RuntimeError(f"reloading {checkpoint} failed; see {out / 'reload_check' / 'predict.log'}")
    trained, trained_ids, _ = read_bundle(checkpoint / stem, task)
    again, again_ids, _ = read_bundle(reloaded, task)
    if trained_ids != again_ids:
        raise RuntimeError("reloaded predictions are for different rows")
    gap = float(np.abs(trained - again).max())
    agree = float((trained.argmax(1) == again.argmax(1)).mean())
    if gap > 0.05 or agree < 0.98:
        raise RuntimeError(f"reloaded model differs: max probability gap {gap:.3f}, label agreement {agree:.1%}")
    return {"checkpoint": str(checkpoint), "max_probability_gap": round(gap, 5), "label_agreement": round(agree, 4)}


def smoke(task, work, gpus=None, roots=None):
    """Exercise every stage on a GPU in about half an hour: data checks, 4-bit Gemma, TAPT,
    two R-Drop classifiers from the TAPT adapter (one per GPU), checkpoint reload, combining and
    packaging. Prints a checklist and raises on the first failure."""
    import zipfile
    import pandas as pd
    problems = check_environment()
    if problems:
        raise RuntimeError("environment problems: " + "; ".join(problems))
    out = output_root(work, task, "smoke")
    if out.exists():
        shutil.rmtree(out)
    run_session(task, "smoke", work, "smoke", gpus, execute=True, roots=roots, scrub=False)
    checks = {"data files match the recorded checksums; holdout fingerprint verified": True}
    execution = json.loads((out / "smoke" / "execution.json").read_text())
    log = (out / "smoke" / "job_1.log").read_text(errors="ignore")
    checks["TAPT ran and saved an adapter"] = (out / "smoke" / "tapt" / "adapter_config.json").is_file()
    checks["classifiers started from the TAPT adapter"] = "initialised from TAPT adapter" in log
    checks["R-Drop path ran"] = all("--rdrop" in r["executed_command"] for r in execution if r["stage"] == "classifier")
    used = {r["gpu"] for r in execution if r["stage"] == "classifier"}
    checks[f"classifiers ran on GPU(s) {sorted(g for g in used if g is not None) or 'default'}"] = True
    reload = check_reload(task, work, "smoke", gpus)
    checks[f"a saved classifier reloads and reproduces its predictions (max gap {reload['max_probability_gap']})"] = True
    report = finish(task, work, "smoke", roots)
    with zipfile.ZipFile(report["predictions_zip"]) as archive:
        frame = pd.read_csv(archive.open("predictions.csv"), dtype=str)
    test = pd.read_csv(Path(work) / "data" / "raw" / FILES[task]["test"], dtype={"id": str}, keep_default_na=False) \
        if (Path(work) / "data" / "raw").exists() else None
    checks["ensemble combined and packaged as id,label"] = list(frame.columns) == ["id", "label"]
    if test is not None:
        checks["one prediction per test id, in order"] = frame.id.tolist() == test.id.tolist()
    shutil.rmtree(Path(work) / "data", ignore_errors=True)
    scrub_data(work, out)
    print("\nSMOKE TEST, Task " + task.upper())
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    if not all(checks.values()):
        raise RuntimeError("smoke test failed")
    print("All stages ran. The full sessions use the same code with the real recipe.")
    return checks


def run_all(task, work, recipe="submitted", gpus=None, execute=False, roots=None):
    """Local GPU: every unfinished group in turn, then combine and package or score."""
    for group in group_ids(task, recipe):
        run_session(task, group, work, recipe, gpus, execute, roots, scrub=False)
    if execute:
        return finish(task, work, recipe, roots)
    return None


def check_environment(min_gb=14.0):
    """Print what a training session needs and whether this machine has it. Returns problems."""
    problems = []
    print(f"python {sys.version.split()[0]}")
    for package, wanted in [*EXPECTED_VERSIONS.items(), ("bitsandbytes", None), ("torch", None)]:
        try:
            have = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package} is not installed (pip install -e '.[llm]')")
            continue
        note = "" if wanted in (None, have) else f"  (training logs used {wanted})"
        print(f"{package} {have}{note}")
    try:
        import torch
        count = torch.cuda.device_count()
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            gb = props.total_memory / 2**30
            print(f"GPU {index}: {props.name}, {gb:.1f} GB")
            if gb < min_gb:
                problems.append(f"GPU {index} has {gb:.1f} GB; the 12B classifier peaked near 10 GB and TAPT near 10.4 GB")
        if count == 0:
            problems.append("no CUDA GPU visible")
        elif count == 1:
            print("One GPU: classifiers run one after another, roughly doubling session time")
    except ImportError:
        pass  # already reported as a missing package
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("note: HF_TOKEN is not set; google/gemma-4-12B is openly available, so this only lowers "
              "Hugging Face download rate limits")
    free = shutil.disk_usage(Path.cwd()).free / 2**30
    print(f"free disk here: {free:.0f} GB")
    if free < 40:
        print("WARNING: under 40 GB free here; the model download (about 25 GB, in the Hugging Face "
              "cache) and the adapters need room")
    for problem in problems:
        print("PROBLEM:", problem)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="check GPU, packages and Hugging Face access")
    p = sub.add_parser("smoke", help="about 30 minutes on a GPU: run every stage on a small sample")
    p.add_argument("--task", choices=["a", "b"], required=True)
    p.add_argument("--work", type=Path, default=PROJECT_ROOT / "artifacts/reproduce")
    p.add_argument("--search", type=Path, nargs="+")
    p.add_argument("--gpus", type=int, nargs="+")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", choices=["a", "b"], required=True)
    common.add_argument("--recipe", choices=list(RECIPES), default="submitted")
    common.add_argument("--work", type=Path, default=PROJECT_ROOT / "artifacts/reproduce")
    common.add_argument("--search", type=Path, nargs="+", help="folders to search for the data files")
    p = sub.add_parser("stage", parents=[common], help="find, verify and copy the data files")
    p = sub.add_parser("session", parents=[common], help="train one group")
    p.add_argument("--group", required=True)
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--execute", action="store_true", help="train; otherwise print the plan")
    p.add_argument("--keep-data", action="store_true", help="do not remove staged comment data afterwards")
    sub.add_parser("collect", parents=[common], help="gather finished groups from other sessions")
    sub.add_parser("finish", parents=[common], help="combine all groups; package or score")
    p = sub.add_parser("all", parents=[common], help="local GPU: every group, then finish")
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        sys.exit(1 if check_environment() else 0)
    if args.command == "smoke":
        smoke(args.task, args.work, args.gpus, args.search)
        return
    roots = args.search
    if args.command == "stage":
        stage(args.task, Path(args.work) / "data", roots)
    elif args.command == "session":
        run_session(args.task, args.group, args.work, args.recipe, args.gpus, args.execute, roots,
                    scrub=not args.keep_data)
    elif args.command == "collect":
        collect(args.task, args.work, args.recipe, roots)
    elif args.command == "finish":
        finish(args.task, args.work, args.recipe, roots)
    else:
        run_all(args.task, args.work, args.recipe, args.gpus, args.execute, roots)


if __name__ == "__main__":
    main()
