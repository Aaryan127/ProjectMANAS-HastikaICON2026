# Reproduction guide

For a ready-to-run reproduction of the submitted systems, use [`reproduce/`](../reproduce/README.md):
Kaggle notebooks, one per training session, and a local GPU script, built on the runner below.

Two entry points:

- **the final systems**, through `hastika-run` and the recipes in `configs/`;
- **any single experiment**, through its notebook in `notebooks/experiments/` and the research
  code in `src/hastika/research/`.

## Environment

Python 3.10–3.12. The base package covers data preparation, the SVM comparator and the CPU tests;
the `llm` extra adds the quantized-model stack.

```bash
uv sync --locked                    # base
uv sync --locked --extra llm        # + torch-dependent stack (install a CUDA PyTorch first)
python -m unittest discover -s tests -v
```

The training logs identify Transformers 5.16.1 and PEFT 0.20.0, which the `llm` extra pins. The
rest of the original training environment and the exact base-model revision were not recorded.
`uv.lock` is this repository's own resolution, not a recovered historical environment. To pin a
model revision, set `model_revision` in a copy of the recipe and keep that recipe with its output.

Set `HF_TOKEN` in the environment or the Kaggle secret store if the model needs it. Never write
tokens into configs or notebooks.

## Data

See [`data/README.md`](../data/README.md). Both entry points verify the organizer files against
the checksums in `data/sources.json` and rebuild the fixed holdouts. A rebuilt holdout must match
the recorded row counts and id-order fingerprints, or the run stops.

## The final systems

### Plan

The runner prints the subprocess commands it would run, without data or model downloads:

```bash
hastika-run --config configs/task_a/submitted.json
hastika-run --config configs/task_b/holdout.json --groups tapt_rdrop
```

`submitted.json` fits on every labelled row of the task. `holdout.json` fits on the 85% split
(TAPT included) and scores the 15% holdout.

### Execute, one group per session

Each group trains its own TAPT adapter, then its classifiers from that adapter. Jobs run one after
another, so one GPU is enough.

| task | groups |
|---|---|
| A | `tapt` (4 classifiers), `tapt_rdrop` (2), `bag1`, `bag2`, `bag3` (2 each) |
| B | `tapt_rdrop` (2), `bag1`, `bag2`, `bag3` (2 each) |

```bash
for g in tapt tapt_rdrop bag1 bag2 bag3; do
  hastika-run --config configs/task_a/submitted.json --groups $g --execute
done
hastika-run --config configs/task_a/submitted.json --combine-only
```

With `--gpus 0 1`, TAPT runs on the first GPU and the group's classifiers then run in parallel,
one per GPU, as in the original Kaggle sessions; without it they run one after another.

Outputs go to `artifacts/task_<t>/<mode>/` by default; `--out`, `--raw-dir` and `--external`
override the paths. When groups run in separate sessions, copy each finished group directory into
one output root before `--combine-only`. On Kaggle, the launchers `notebooks/task_a_kaggle.ipynb`
and `notebooks/task_b_kaggle.ipynb` clone release `v0.3.1` and run one group.

Safety rules:

- A failed group stops the run, and no partial ensemble is ever labelled final.
- A classifier that collapses after its first epoch (exit 4) is retried once with seed + 1000.
  Requested seeds, effective seeds and exact commands are saved in `execution.json`.
- Existing group directories are never reused. Use a new output path to rerun.
- External comments matching holdout or test inputs are removed before TAPT, and the counts go to
  `data_audit.json`.

### Checkpoints, prediction and packaging

Classifier runs save `adapter/`, `classifier_head.pt`, `tokenizer/`, `model.json`, metrics and
probability bundles (`_probs.npy`, `_ids.csv`, `_metadata.json`). The historical training runs did
not save weights, so none are distributed. Checkpoint reload has not yet had a CUDA smoke test.

```bash
hastika-predict --checkpoint artifacts/task_a/full/tapt/s42 \
  --input data/raw/hastika_binary_test.csv --out artifacts/reloaded_a
hastika-ensemble package --bundle artifacts/task_a/full/final \
  --reference data/raw/hastika_binary_test.csv --out artifacts/submissions/task_a.zip
hastika-evaluate --task a --gold artifacts/task_a/holdout/data/holdout.csv \
  --bundle artifacts/task_a/holdout/final
```

### SVM comparator

`hastika-svm` fits word + character TF-IDF with LinearSVC on explicit CSVs (sigmoid calibration
for A, softmax margins for B). It is not the character-only SVM of some experiments, so do not
give it a recorded score without measuring the matching split.

## Single experiments

Each notebook in `notebooks/experiments/task_a/` and `task_b/` begins with a header giving its
run number, status, outcome, record and write-up. To rerun one on Kaggle:

1. Create a private Kaggle dataset with the six organizer CSVs and `offenseval_kn.csv`, and add
   it to the notebook.
2. Turn Internet on, choose GPU T4 x2, and use Save & Run All.

The setup cell clones this repository, puts `src/` on the path, and calls
`python -m hastika.research.data`. That copies the files from `/kaggle/input`, checks their
checksums, writes the validation-phase input files (the released validation files without
labels), and rebuilds both fixed holdouts. Locally, put the files in `data/raw/` and
`data/external/` and run the same command; `--source DIR` points it at another folder.

The code these notebooks run is in `hastika.research` ([its README](../src/hastika/research/README.md)).
The notebooks have been brought in line with this repository's layout and numbering, but have not
been re-executed on a GPU since. A rerun is a new measurement, so compare it with the recorded
outcome instead of assuming they match. Recorded outcomes come from the original runs' logs. Each
record under `results/experiments/` lists the log's SHA-256 and the line of every quoted metric.
The logs themselves are not distributed.

Several research trainers live under `hastika.research.task_b` but serve both tasks
(`llm_classifier`, `llm_tapt`, `combined_holdout`). They are the code the Gemma runs actually
used. The supported `hastika.models` pipeline is derived from them with the same optimizer
arithmetic, including updating only on complete gradient-accumulation windows.

## What the CPU tests cover

Split reconstruction and fingerprints, label validation, the external-text filter, probability
alignment, averaging and packaging, recipe composition and the runner's retry logic, the
research data staging and corpus rules, and the experiment record: contiguous numbering,
notebook/record agreement, cleared outputs and repository URLs. They do not test GPU training,
model availability or accuracy.
