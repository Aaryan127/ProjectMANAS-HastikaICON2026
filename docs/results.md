# Results

## Official

| Task | Macro-F1 | Accuracy | Classifiers |
|---|---:|---:|---:|
| A | 0.825049 | 0.878412 | 12 |
| B | 0.736678 | 0.787879 | 8 |

The organizers' scores for the submitted predictions, macro-averaged, as reported by the authors. Rankings were not
supplied. Membership: Task A is Run 29 x4, Run 35 x2 and Run 36 sessions 1–3 x2 each; Task B is
Run 36 x2 and Run 37 sessions 1–3 x2 each. Probabilities are averaged with equal weight per
classifier. The configs record requested seeds; not every replacement seed of a historical rerun
was recorded. Final prediction files are not part of this repository. The final evaluation scored
only the final portion of each test file, not all 806 (A) or 396 (B) rows.

## Holdout and official scores

| task | evaluation set | trained on | system | models | macro-F1 | accuracy |
|---|---|---|---|---:|---:|---:|
| A | official test, final portion of `hastika_binary_test.csv` | all 7,193 labelled rows | **submitted TAPT ensemble** | 12 | **0.8250** | 0.8784 |
| A | holdout `815110ff24`, 1,079 of 7,193 rows | the other 6,114 | TAPT x2 + TAPT + R-Drop x2 (Run 33) | 4 | 0.8721 | 0.8721 |
| A | same | same | TAPT x4 (Run 31) | 4 | 0.8702 | 0.8703 |
| A | same | same | TAPT bag pair (Run 37) | 2 | 0.8675 | n/a |
| A | same | same | Gemma, no TAPT, 3 epochs (Run 30) | 2 | 0.8610 | 0.8610 |
| A | same | same | char SVM (Run 23) | n/a | 0.8117 | 0.8119 |
| B | official test, final portion of `hastika_multiclass_test.csv` | all 3,532 labelled rows | **submitted TAPT ensemble** | 8 | **0.7367** | 0.7879 |
| B | holdout `f85f4f049b`, 530 of 3,532 rows | the other 3,002 | TAPT bag pair (Run 38) | 2 | 0.7056 | 0.7547 |
| B | same | same | TAPT + R-Drop x2 (Run 35) | 2 | 0.7026 | 0.7528 |
| B | same | same | TAPT x2 (Run 35) | 2 | 0.6741 | 0.7321 |
| B | same | same | Gemma, no TAPT, 4 epochs (Run 22) | 2 | 0.6872 | 0.7321 |
| B | same | same | MuRIL + TAPT (Run 20) | 5 | 0.6151 | 0.6509 |

Each holdout is a stratified 15% of the task's merged, deduplicated training and released
validation rows, drawn once with seed 42 (`hastika-prepare`; see
[`data/README.md`](../data/README.md)). On Task A the official score is 0.047 below the best
holdout result and below every Gemma holdout result; on Task B it is 0.031 above the best. The
submitted ensembles are larger and trained on more data than anything scored on the holdouts, and
the holdout choices are optimistic after more than twenty comparisons on the same rows. The final
test rows may also differ from the labelled data in topic or class balance. Without test labels
the gap cannot be traced to any one of these.

## Development

[`manifest.csv`](../results/manifest.csv) is a short table of the main local comparisons.
[`notebook_results.csv`](../results/notebook_results.csv) and
[`experiment_catalog.json`](../results/experiment_catalog.json) list every run's status and
outcome. Each run has a record under [`results/experiments/`](../results/experiments/) with
the metric and paired-bootstrap lines extracted from its training log (log SHA-256 and line
number for each). Unrun or failed runs carry no score.

Keep each number with its protocol:

- Task A Run 33's 0.8721 is a four-model mixture on the 1,079-row holdout, and Task B Run 38's
  0.7056 is a two-model pair on the 530-row holdout. Neither is an estimate of the 12- or
  8-model official score.
- Task B Run 22's 0.6872 comes from two holdout models; the full-data fit that followed used four.
- Task B Run 35 reports TAPT 0.6741, TAPT + R-Drop 0.7026 and their mixture 0.6966. Quote all
  three when discussing that comparison.
- CodaBench validation scores (Task A up to Run 12, Task B up to Run 11) are a different phase
  and row set from the holdouts and the official test.

## Not re-measured here

This repository has not been GPU-retrained or officially rescored. The data pipeline removes
external comments matching holdout or test inputs before TAPT. One normalized comment in the
external file matches a Task A test comment, so a rerun's TAPT stream differs from the original
by that one comment. Do not present a rerun's score as a reproduction of the official one
without saying so.
