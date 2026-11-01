# Methods

The final systems for both tasks, as implemented in `src/hastika/` and recorded in
[`configs/`](../configs/). [experiments.md](experiments.md) explains how each choice was reached.

## Data

| task | labelled rows used | holdout (15%, split seed 42) | test inputs |
|---|---:|---:|---:|
| A | 7,193 | 1,079, fingerprint `815110ff24` | 806 |
| B | 3,532 | 530, fingerprint `f85f4f049b` | 396 |

Each task uses only its own files: the original training labels plus the released labelled
validation set (`hastika.data.splits`). Duplicate ids are dropped. Comments with identical
normalized text are collapsed, and groups whose copies disagree on the label are dropped whole.
That is 59 rows for A and 22 for B. The final fits use every labelled row; the holdout recipes
fit on the 85% split.

**Cleaning** (`hastika.common.preprocessing.clean`): undo the latin-1/UTF-8 mojibake in the
released CSVs, unescape HTML, replace URLs and @-mentions with `<url>` / `<user>`, collapse
whitespace. Comments are cut at 700 characters and 320 tokens. The LLM pipeline keeps emoji;
demojization was a MuRIL-specific fix.

**External text.** TAPT also reads the Kannada portion of OffensEval-Dravidian (3,247 comments,
`data/external/offenseval_kn.csv`), comment text only. The runner removes any external comment
whose normalized text matches a holdout or test input before TAPT sees it.

## Model

- **Backbone:** `google/gemma-4-12B`, loaded in 4-bit NF4 with double quantization, fp16 compute
  (T4 GPUs have no bf16).
- **Adapters:** LoRA rank 16, dropout 0.05, on `q/k/v/o_proj` and `gate/up/down_proj`.
- **Input:** a short prompt around the comment.
  - Task A: `Kannada-English (romanized) YouTube comment: {comment}` / `Is this hate speech (Hate or Non-Hate)?:`
  - Task B: `Kannada-English (romanized) hateful YouTube comment: {comment}` / `Category (Gender, Political, Religion, Geo-political, Violence, Others):`
- **Head:** dropout 0.1 and an fp32 linear layer on the hidden state of the last non-padding
  token.

## Task-adaptive pretraining (TAPT)

`hastika.models.llm_tapt`: LoRA next-token training on unlabelled comment text, the training-side
comments plus the external corpus. The adapter it saves has the same target modules and rank as
the classifier's, which starts from it (`--init-adapter`).

| setting | value |
|---|---|
| epochs | 1 (A), 2 (B) |
| learning rate | 1e-4, AdamW, weight decay 0.01, cosine schedule, 6% warmup |
| batch | micro-batch 1 x 16 accumulation, 128 tokens |
| memory | the LM head runs in chunks of 16 positions over a detached copy of the hidden states, so the 12B decoder does one backward per batch; gradients are identical to the unchunked loss |

## Classifier

`hastika.models.llm_classifier`.

| setting | value |
|---|---|
| epochs | 3 (A), 4 (B), last epoch kept |
| learning rate | 1e-4 for LoRA, 5e-4 for the head; AdamW, weight decay 0.01, cosine, 6% warmup |
| batch | 8 x 2 accumulation (effective 16), length-bucketed |
| loss | cross-entropy with balanced class weights and label smoothing 0.05 |
| R-Drop (some members) | two dropout passes, mean cross-entropy + 0.5 x symmetric KL |
| collapse guard | after epoch 1, if more than 95% of a 400-row training sample is predicted as one class (and that class's prior is under 90%), the job exits with code 4; the runner retries once with seed + 1000 |

## Ensembles

Each group is one TAPT run followed by its classifiers. The final prediction averages class
probabilities with equal weight per classifier and takes the argmax; no weights, thresholds or
members were tuned on the holdout. `hastika.evaluation.ensemble` refuses misaligned ids, the
wrong class order, collapsed members and incomplete member counts.

| task | groups (TAPT seed → classifier seeds) | members |
|---|---|---:|
| A | `tapt` 42 → 42–45; `tapt_rdrop` 42 → 42, 43; `bag1` 1 → 42, 43; `bag2` 2 → 44, 45; `bag3` 3 → 46, 47 | 12 |
| B | `tapt_rdrop` 42 → 42, 43; `bag1` 1 → 42, 43; `bag2` 2 → 44, 45; `bag3` 3 → 46, 47 | 8 |

These are the requested seeds. A historical classifier that collapsed was rerun with another
seed, and not every replacement seed was recorded.

## Evaluation

Model choices after the validation phase were made on out-of-fold scores and then on the fixed
holdouts, with paired bootstrap comparisons on identical rows. The official results are macro-F1 on the organizers' test labels for the final portion of each
test file: 0.825049 (A) and 0.736678 (B).

## Compute

Kaggle Tesla T4 GPUs (14.6 GB), two per session. A TAPT run takes about 2 h (A) or 2.3 h (B) and
peaks at 8.4–10.4 GB. A 12B classifier fit takes 70–120 min. Each group fits in one session of
under 12 hours.
