# Reproducing the submitted systems

This folder rebuilds Project MANAS's two submitted HASTIKA systems from scratch:

| task | system | members | official macro-F1 (organizers) |
|---|---|---:|---:|
| A: Hate / Non-Hate | Gemma-4-12B TAPT ensemble | 12 classifiers from 5 TAPT runs | 0.825049 |
| B: six hate categories | Gemma-4-12B TAPT ensemble | 8 classifiers from 4 TAPT runs | 0.736678 |

There are two routes to the same result, and both run the same code (`hastika.reproduce`, which
drives `hastika.runner` and the recipes in [`configs/`](../configs/)):

- **Kaggle** ([`kaggle/`](kaggle/)): one notebook per training session, then one combine
  notebook per task. Nothing needs editing; attach the data, add one secret and run.
- **Your own GPU** ([`local/`](local/)): one script that trains every group and packages the
  result, resuming where it stopped.

## What gets trained

Each system is built from *groups*. A group is one task-adaptive pretraining (TAPT) run of
Gemma-4-12B, which produces a LoRA adapter, followed by two or four classifiers that start from
that adapter with different seeds. A group fits in one Kaggle session on two T4 GPUs: TAPT runs
first, then the classifiers run two at a time, one per GPU. The final prediction averages every
classifier's class probabilities with equal weight. [`sessions.json`](sessions.json) is the
plan.

| session | Task A group | classifiers | about | Task B group | classifiers | about |
|---|---|---:|---:|---|---:|---:|
| 1 | `tapt` | 4 | 7.6 h | `tapt_rdrop` (R-Drop 0.5) | 2 | 6.5 h |
| 2 | `tapt_rdrop` (R-Drop 0.5) | 2 | 7.5 h | `bag1` (TAPT seed 1) | 2 | 4.5 h |
| 3 | `bag1` (TAPT seed 1) | 2 | 4.8 h | `bag2` (TAPT seed 2) | 2 | 4.5 h |
| 4 | `bag2` (TAPT seed 2) | 2 | 4.8 h | `bag3` (TAPT seed 3) | 2 | 4.5 h |
| 5 | `bag3` (TAPT seed 3) | 2 | 4.8 h | combine (CPU) | | minutes |
| 6 | combine (CPU) | | minutes | | | |

Times are those of the original runs on Kaggle T4 x2. In total, about 30 GPU-hours for Task A and
20 for Task B; Kaggle's free quota is about 30 GPU-hours a week, and sessions are independent, so
they can run in parallel from different notebooks or over several weeks.

## You need

1. **The data**, under the organizers' terms (it is not redistributed here): the six CSVs
   `binary_train.csv`, `hastika_binary_validation.csv`, `hastika_binary_test.csv`,
   `multiclass_train.csv`, `hastika_multiclass_validation.csv`, `hastika_multiclass_test.csv`,
   and the processed external corpus `offenseval_kn.csv` (see [`data/README.md`](../data/README.md)).
   Every file is checked against the SHA-256 recorded in
   [`data/sources.json`](../data/sources.json); a different file stops the run.
2. **The model**: `google/gemma-4-12B` is openly available under Apache 2.0 and downloads
   without an account. An `HF_TOKEN` is optional and only raises download limits.
3. **A GPU**: Kaggle's T4 x2, or any CUDA GPU with at least 14 GB (the classifier peaked at about
   10 GB, TAPT at 10.4 GB). On one GPU, classifiers run one after another, so a session takes
   longer.

## Start with the smoke test

Before spending 20-30 GPU-hours, run `kaggle/task_a_0_smoke.ipynb` (or
`python -m hastika.reproduce smoke --task a` on your own GPU). In about 30 minutes it runs every
stage on a small sample: data checksums and holdout fingerprint, Gemma-4-12B in 4 bits, a TAPT
pass, two R-Drop classifiers from the TAPT adapter (one per GPU), reloading a saved classifier and
checking that it reproduces its predictions, combining and packaging. It ends with a PASS/FAIL
checklist. Its models train for one epoch on about 100 comments, so its predictions are not
meant to be accurate. The smoke recipes are `configs/task_<t>/smoke.json`.

## Route 1: Kaggle

1. Upload the seven data files as one **private** Kaggle dataset (any folder layout).
2. Import the notebooks for your task from [`kaggle/`](kaggle/) (*File -> Import Notebook*).
3. In each **session** notebook (`task_a_1_tapt.ipynb` ... `task_a_5_bag3.ipynb`, or
   `task_b_1_tapt_rdrop.ipynb` ... `task_b_4_bag3.ipynb`):
   - *Settings*: Accelerator **GPU T4 x2**, Internet **on**.
   - *Add Input*: your private dataset.
   - Optional: *Add-ons -> Secrets* -> `HF_TOKEN`.
   - **Save Version -> Save & Run All (Commit)**.

   The notebook clones this repository at release `v0.3.1`, installs it, finds and verifies the
   data, trains its group and removes the copies of the comment data from its output. Its output
   keeps `task_<t>_submitted/<group>/`: probabilities for the test file, adapters, logs and an
   `execution.json` with the seeds actually used.
4. When every session has finished, open the **combine** notebook (`task_a_6_combine.ipynb` or
   `task_b_5_combine.ipynb`), add your dataset and the outputs of the session notebooks as inputs
   (*Add Input -> Your Work -> Notebooks*), and run it on CPU. It writes
   `task_<t>_predictions.zip` with `predictions.csv` (`id,label`), the submission format, and a
   report with the predicted class shares.

## Route 2: your own GPU

```bash
git clone --branch v0.3.1 https://github.com/Aaryan127/ProjectMANAS-HastikaICON2026.git
cd ProjectMANAS-HastikaICON2026
pip install torch                       # a CUDA build that matches your driver
pip install -e '.[llm]'                 # transformers 5.16.1, peft 0.20.0, bitsandbytes
export HF_TOKEN=...                     # optional; never commit it
# data: the six CSVs in data/raw/, offenseval_kn.csv in data/external/

python reproduce/local/check_env.py                 # GPU memory, versions, token, disk
python reproduce/local/reproduce.py --task a        # print the plan
python reproduce/local/reproduce.py --task a --execute
python reproduce/local/reproduce.py --task b --execute
```

Results go to `artifacts/reproduce/`: `task_<t>_submitted/<group>/` per group,
`task_<t>_submitted/final` for the ensemble, and `task_<t>_predictions.zip`. Finished groups are
skipped, so the script can be stopped between groups and run again; a half-finished group is
refused and never reused (move it aside). `--group bag2` trains one group, and `--gpus 0 1`
chooses the GPUs (default: all visible).

## Checking a rerun

The test labels are not public, so a rerun of the submission cannot be scored locally. The
holdout recipe is the check: it trains the same groups on the 85% split and scores them on the
fixed 15% holdout (1,079 Task A and 530 Task B comments, rebuilt and fingerprint-checked from the
organizer files).

```bash
python reproduce/local/reproduce.py --task a --recipe holdout --execute
```

On Kaggle, set `RECIPE = "holdout"` in the settings cell of every session notebook and of the
combine notebook. The report lists each group's holdout macro-F1 beside the original run of the
same group:

| task | group | original holdout macro-F1 |
|---|---|---:|
| A | `tapt` (TAPT x4) | 0.8702 (Run 31) |
| A | `tapt_rdrop` (R-Drop x2) | 0.8647 (Run 33) |
| A | `bag1` (pair) | 0.8675 (Run 37) |
| B | `tapt_rdrop` (R-Drop x2) | 0.7026 (Run 35) |
| B | `bag1` (pair) | 0.7056 (Run 38) |

The full 12- and 8-model ensembles were never scored on the holdout in the original work, so
those two numbers are new measurements.

## What will differ from the original

A rerun is a new measurement, not a bit-exact copy:

- **GPU training is not deterministic.** On the Task B holdout, two TAPT runs that differed only
  in their seed scored 0.6741 and 0.7056, so a group can move by a few points.
- **Seeds.** The recipes record the requested seeds. Where an original classifier collapsed and
  was rerun with another seed, not every replacement seed was recorded. A collapse in a rerun is
  retried once with seed + 1000 and logged in `execution.json`.
- **Model revision.** The original base-model revision was not recorded; the recipes load the
  current `google/gemma-4-12B`. To pin one, set `model_revision` in a copy of the recipe.
- **External text.** External comments that match a test comment are removed before TAPT. One
  matches a Task A test comment, so the Task A TAPT text differs from the original by one comment.
- **Code lineage.** The trainers are the research trainers that produced the submitted models,
  with research-only options removed, input validation added and checkpoints saved at the end;
  the training arithmetic is unchanged. The smoke test exercises them on a GPU; the
  orchestration is also covered by CPU tests (`tests/test_reproduce.py`, `tests/test_release.py`)
  with training replaced by a fake. Please report problems in the repository's issues.

## Files

| path | what |
|---|---|
| `sessions.json` | the session plan: groups, members, seeds, times, recorded holdout scores |
| `kaggle/task_<t>_0_smoke.ipynb` | the 30-minute smoke test; run first |
| `kaggle/task_<t>_<n>_<group>.ipynb` | one training session each |
| `kaggle/task_<t>_<n>_combine.ipynb` | collect the sessions' outputs; package or score |
| `local/reproduce.py` | the whole task on your own GPU(s) |
| `local/check_env.py` | what is missing on this machine |
| `build_notebooks.py` | regenerates `kaggle/` from `sessions.json`; the tests fail if they drift |
| [`../src/hastika/reproduce.py`](../src/hastika/reproduce.py) | the logic all of the above call |
