"""Generate the Kaggle notebooks in reproduce/kaggle/ from reproduce/sessions.json.

Run after changing the session plan or the template; tests/test_reproduce.py fails if the
committed notebooks differ from what this script writes.

    python reproduce/build_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Aaryan127/ProjectMANAS-HastikaICON2026.git"
REPO_REF = "v0.3.1"
TASK_NAME = {"a": "Task A (Hate / Non-Hate)", "b": "Task B (six hate categories)"}

SETUP = f'''import os, subprocess, sys
from pathlib import Path

REPO_URL = "{REPO_URL}"
REPO_REF = "{REPO_REF}"          # the release this notebook was written for
REPO = Path("/kaggle/working/ProjectMANAS-HastikaICON2026")
WORK = Path("/kaggle/working")

def run(*cmd):
    """Run a command and stream its output into this cell; stop on failure."""
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise RuntimeError(f"command failed with exit code {{proc.returncode}}")

if not REPO.exists():
    run("git", "clone", "--quiet", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO)
os.chdir(REPO)
run(sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[llm]")
sys.path.insert(0, str(REPO / "src"))'''

SECRETS = '''# google/gemma-4-12B is openly available (Apache 2.0); no token is needed. If a Kaggle secret
# called HF_TOKEN exists (Add-ons -> Secrets), it is used for higher download limits.
# Never paste a token into a cell.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception as error:
    print("No HF_TOKEN secret found:", error)
run(sys.executable, "-m", "hastika.reproduce", "check")'''


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def notebook(cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index}"
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}}


def checklist(extra=""):
    return f"""**Before you run** (Kaggle notebook settings):

1. *Accelerator*: **GPU T4 x2**. *Internet*: **on**.
2. *Add Input*: your private Kaggle dataset holding the six organizer CSVs and
   `offenseval_kn.csv` (see `reproduce/README.md`; the files are found by name and checked
   against their recorded SHA-256, wherever they sit in the dataset).
3. Optional: *Add-ons -> Secrets* -> `HF_TOKEN` (a Hugging Face token raises download
   limits; the model itself is openly available).{extra}
4. Use **Save Version -> Save & Run All (Commit)**. An interactive session stops when the
   browser tab closes.
"""


def session_notebook(task, s, total):
    title = (f"# {TASK_NAME[task]}, session {s['order']} of {total}: group `{s['group']}`\n\n"
             f"Part of the reproduction of Project MANAS's submitted HASTIKA system. This session runs one "
             f"TAPT pass of Gemma-4-12B (TAPT seed {s['tapt_seed']}), then trains {s['classifiers']} "
             f"classifiers from that adapter{' with R-Drop 0.5' if s['rdrop'] else ''}, two at a time on the "
             f"two T4 GPUs. The original run of this group (Run {s['original_run']}) took about "
             f"{s['hours']} hours.\n\n" + checklist() +
             f"\nWhen it finishes, its output contains `task_{task}_submitted/{s['group']}/`: the group's test "
             f"probabilities, its adapters and its logs, but no comment data. Run the other sessions, then "
             f"`task_{task}_{total + 1}_combine.ipynb`.\n")
    settings = (f'TASK = "{task}"\nGROUP = "{s["group"]}"\n'
                f'RECIPE = "submitted"   # "holdout": train on the 85% split and score the 15% holdout\n')
    train = ('run(sys.executable, "-m", "hastika.reproduce", "session", "--task", TASK, "--group", GROUP,\n'
             '    "--recipe", RECIPE, "--work", WORK, "--search", "/kaggle/input", "--execute")')
    return notebook([markdown(title), code(settings), code(SETUP), code(SECRETS), code(train)])


def smoke_notebook(task):
    title = (f"# {TASK_NAME[task]}: smoke test (run this first, about 30 minutes)\n\n"
             "Runs every stage of the reproduction on a small sample before you spend 20 to 30 "
             "GPU-hours on the full sessions: data checksums and holdout fingerprint, loading "
             "Gemma-4-12B in 4 bits, a TAPT pass, two R-Drop classifiers started from the TAPT "
             "adapter (one per GPU), reloading a saved classifier and checking it reproduces its "
             "predictions, combining the two, and packaging a predictions file. It ends with a "
             "PASS/FAIL checklist. The models train for one epoch on about 100 comments, so their "
             "predictions are not meant to be accurate and the ZIP it writes is not a submission.\n\n"
             + checklist())
    settings = f'TASK = "{task}"\n'
    run_smoke = ('run(sys.executable, "-m", "hastika.reproduce", "smoke", "--task", TASK,\n'
                 '    "--work", WORK, "--search", "/kaggle/input")')
    return notebook([markdown(title), code(settings), code(SETUP), code(SECRETS), code(run_smoke)])


def combine_notebook(task, sessions):
    total = len(sessions)
    names = ", ".join(f"`task_{task}_{s['order']}_{s['group']}`" for s in sessions)
    extra = (f"\n   Also add, as inputs, the outputs of the {total} session notebooks ({names}): "
             f"*Add Input -> Your Work -> Notebooks*. No GPU is needed for this notebook.")
    title = (f"# {TASK_NAME[task]}: combine the {total} sessions\n\n"
             f"Collects every finished group from the attached session outputs, averages all "
             f"{sum(s['classifiers'] for s in sessions)} classifiers with equal weight, and writes "
             f"`task_{task}_predictions.zip` (a `predictions.csv` with `id,label` for the test file), "
             f"the submission format. With `RECIPE = \"holdout\"` it instead scores every group and the "
             f"full ensemble on the fixed holdout and prints the original runs' holdout scores beside them.\n\n"
             + checklist(extra).replace("**GPU T4 x2**", "**None** (CPU is enough)"))
    settings = f'TASK = "{task}"\nRECIPE = "submitted"\n'
    finish = ('run(sys.executable, "-m", "hastika.reproduce", "collect", "--task", TASK, "--recipe", RECIPE,\n'
              '    "--work", WORK, "--search", "/kaggle/input")\n'
              'run(sys.executable, "-m", "hastika.reproduce", "finish", "--task", TASK, "--recipe", RECIPE,\n'
              '    "--work", WORK, "--search", "/kaggle/input")')
    setup = SETUP.replace('run(sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[llm]")',
                          'run(sys.executable, "-m", "pip", "install", "--quiet", "-e", ".")')
    return notebook([markdown(title), code(settings), code(setup), code(finish)])


def build():
    plan = json.loads((HERE / "sessions.json").read_text())
    written = {}
    for task in ("a", "b"):
        sessions = plan[task]["sessions"]
        written[f"task_{task}_0_smoke.ipynb"] = smoke_notebook(task)
        for s in sessions:
            written[f"task_{task}_{s['order']}_{s['group']}.ipynb"] = session_notebook(task, s, len(sessions))
        written[f"task_{task}_{len(sessions) + 1}_combine.ipynb"] = combine_notebook(task, sessions)
    return written


def main():
    out = HERE / "kaggle"
    out.mkdir(exist_ok=True)
    for name, nb in build().items():
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
