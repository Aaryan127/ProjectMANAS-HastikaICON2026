# Project MANAS at HASTIKA / ICON 2026

Code, experiments and results for Project MANAS's systems in the HASTIKA shared task:
hate-speech classification of Kannada–English (romanized, code-mixed) YouTube comments.

**Authors:** Robin Punnen Alex and Aaryan Ganesh Salunke, Project MANAS, Manipal Institute of
Technology, Manipal Academy of Higher Education.

| Task | Official macro-F1 | Accuracy | System |
|---|---:|---:|---|
| A: Hate / Non-Hate | **0.825049** | **0.878412** | 12 Gemma-4-12B TAPT classifiers |
| B: six hate categories | **0.736678** | **0.787879** | 8 Gemma-4-12B TAPT classifiers |

These are the organizers' scores for the submitted predictions, as reported by the authors
([`results/final_scores.json`](results/final_scores.json)), on the final portion of each test
file that the organizers scored. This repository has not retrained or rescored them.
[docs/results.md](docs/results.md) sets them beside the holdout scores.

## How the systems were reached

Both tasks followed the same path through 38 runs each:
[docs/experiments.md](docs/experiments.md) is the full account, run by run.

1. **A strong linear floor.** On Task A, word + character TF-IDF with LinearSVC (0.8103
   five-fold) was at least level with stock MuRIL (0.7937 on a validation split). MuRIL's
   tokenizer splits romanized Kannada into arbitrary pieces, and two spellings of the same word
   get unrelated pieces.
2. **MuRIL, tuned.** On Task A, MuRIL + SVM blends reached 0.8188 on the validation leaderboard.
   On Task B, task-adaptive pretraining (TAPT, +2.9 five-fold), one reinitialized layer and R-Drop
   reached 0.6410. Translating to English lost 5.3 points on the Task A holdout and was dropped.
3. **A decoder LLM as the encoder.** Gemma-4-12B in 4-bit QLoRA with a classification head on the
   last prompt token gained 6.4 points over MuRIL on the Task B holdout and 4.1 over the best
   MuRIL + SVM blend on the Task A holdout.
4. **Averaging TAPT models.** LoRA next-token pretraining on the task's comments did not improve
   single Gemma models. On Task A, averages of TAPT models scored higher than averages without
   TAPT (0.8702 for four against 0.8582 for three). On Task B a single TAPT pair scored 0.6741 or
   0.7056 depending on its TAPT seed, against 0.6872 without TAPT. The final systems average
   models from several independent TAPT runs, some trained with R-Drop.

## Final systems

Gemma-4-12B, NF4 4-bit QLoRA (LoRA rank 16 on attention and MLP), a linear head on the last
prompt-token state, and TAPT before classification. Task A: TAPT 1 epoch, classifier 3 epochs.
Task B: TAPT 2 epochs, classifier 4 epochs. Probabilities are averaged with equal weight per
classifier. [docs/methods.md](docs/methods.md) has the details.

| Group | Run (A / B) | Task A | Task B |
|---|---|---:|---:|
| `tapt`: one TAPT run, four classifiers | A29 | 4 | n/a |
| `tapt_rdrop`: TAPT + R-Drop 0.5 | A35 / B36 | 2 | 2 |
| `bag1`–`bag3`: a fresh TAPT run per session, two classifiers each | A36 / B37 | 6 | 6 |
| **Total** | | **12** | **8** |

## Repository map

```text
configs/               the two final recipes (full data) and their holdout twins
reproduce/             ready-to-run reproduction: Kaggle session notebooks and local GPU scripts
src/hastika/           supported pipeline: data preparation, QLoRA classifier, TAPT, ensembling
src/hastika/research/  the code every experiment ran on (MuRIL, SVMs, sweeps, Gemma research trainers)
notebooks/             two Kaggle launchers for the final systems
notebooks/experiments/ one notebook per experiment run, Task A and Task B
results/               official scores, per-run records, catalogue and comparison tables
docs/                  experiment log, methods, reproduction guide, results, competition context
data/                  where the organizer files go (no comment data is distributed)
tests/                 CPU checks of data handling, ensembling, recipes and the experiment record
```

## Install

Python 3.10–3.12.

```bash
uv sync --locked                       # or: python -m pip install -e .
uv run python -m unittest discover -s tests -v
```

For GPU training, install a PyTorch build that matches your CUDA driver, then
`python -m pip install -e '.[llm]'` (Transformers 5.16.1 and PEFT 0.20.0, as in the training logs).

## Data

Get the organizer files under their terms and put them in `data/raw/`, with the processed
external corpus at `data/external/offenseval_kn.csv`. [data/README.md](data/README.md) lists the
files and their checksums. Then:

```bash
hastika-prepare --task a && hastika-prepare --task b     # fixed holdouts, checked by fingerprint
```

## Reproduce the final systems

**[`reproduce/`](reproduce/README.md) is the ready-to-run package** (the
[`reproduce` branch](https://github.com/Aaryan127/ProjectMANAS-HastikaICON2026/tree/reproduce)
contains only it and the code it needs): one Kaggle notebook per
training session (attach the data and run), a combine notebook that writes
the submission ZIP, and `reproduce/local/reproduce.py` for your own GPU. The holdout recipe
rescores every group against the original runs. The rest of this section is the underlying
runner.

The runner prints its plan without training by default:

```bash
hastika-run --config configs/task_a/submitted.json
```

Train one group at a time (each fits a Kaggle session), then combine:

```bash
hastika-run --config configs/task_a/submitted.json --groups bag1 --execute
hastika-run --config configs/task_a/submitted.json --combine-only
```

`configs/task_*/holdout.json` runs the same recipe on the 85% split for local evaluation. On
Kaggle, use `notebooks/task_a_kaggle.ipynb` and `notebooks/task_b_kaggle.ipynb`.
[docs/reproducibility.md](docs/reproducibility.md) covers sessions, collapse retries, checkpoints
and packaging.

## Rerun an experiment

Every run in [docs/experiments.md](docs/experiments.md) links to its notebook. Each notebook opens
with the run's status and outcome, clones this repository, and stages the data with
`python -m hastika.research.data`. Attach the organizer CSVs and `offenseval_kn.csv` as a private
Kaggle dataset and use Save & Run All. [notebooks/experiments/](notebooks/experiments/README.md)
indexes them all. [`results/notebook_results.csv`](results/notebook_results.csv) has every
outcome in one table.

## Citation and terms

Work that builds on this repository should credit the data: the HASTIKA corpus paper
(`kavatagi-rachh-2025`) and the HASTIKA @ ICON-2026 shared task (`hastika-2026-starting-kit`),
both in [`paper/paper_refs.bib`](paper/paper_refs.bib); [`CITATION.cff`](CITATION.cff) cites this
repository. Dataset,
base-model and external-corpus terms belong to their holders; see [NOTICE.md](NOTICE.md). The
authors, Robin Punnen Alex and Aaryan Ganesh Salunke, have not yet chosen a code licence.
