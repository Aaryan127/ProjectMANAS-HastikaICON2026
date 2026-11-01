# Experiment log

Every experiment the team ran or prepared for HASTIKA, in order, with what it found and what
was decided because of it. Runs are numbered per task in the order they were set up; dates are
the ones in the project's working log (2026). Each run has a notebook under
[`notebooks/experiments/`](../notebooks/experiments/) (or, for a few early runs, a script), and a
record under [`results/experiments/`](../results/experiments/) with its status, outcome and the
metric lines extracted from its training log.

A run with no result says why in brackets after its title:

- **(never run)**: prepared and then dropped before it was started.
- **(prepared; no run recorded)**: no log or score from that notebook was kept, so whether it ran
  is not known. No number is reported for it.
- **(failed …)** / **(partly failed)**: it started, and the entry says where it stopped.

## How to read the numbers

| protocol | what it is | rows |
|---|---|---|
| CodaBench validation | the competition's validation phase, labels hidden at the time | A 806, B 395 |
| five-fold out-of-fold (OOF) | every training row predicted by a model that did not see it | A 6,401, B 3,143 (deduplicated train) |
| fixed holdout | a stratified 15% of train + released labelled validation, split seed 42 | A 1,079 (`815110ff24`), B 530 (`f85f4f049b`) |
| full-data fit | every labelled row used for training | no local score is possible |

- Numbers from different protocols are not comparable with each other. A full-data fit has no
  local score; it was judged on CodaBench while that was open, and by its holdout twin after.
- Task A (two classes) and Task B (six classes) macro-F1 are not comparable either.
- Noise: one CodaBench score carries about 1.5 points of standard deviation at 806 rows (A) and
  about 3 at 395 rows (B); the 530-row Task B holdout about 2.5; the 1,079-row Task A holdout
  about 1. Paired bootstrap intervals are quoted where a run computed them; "P" is the bootstrap
  probability that the first arm is better.
- After the validation phase closed there was no more leaderboard feedback. From then on
  decisions were made on out-of-fold scores and, from Run 17 (A) and Run 19 (B), on the fixed
  holdouts. The official result for each task is the organizers' score of its final submission.
- Task B MuRIL runs shrink MuRIL's embedding table to the tokens that occur in the comment files
  of the validation phase (`build_trimmed_vocab` in `hastika.research.task_b.train`). This saves
  memory; no labels are read and no model trains on that text.
- Where Task B MuRIL used TAPT inside a five-fold or holdout comparison (Runs 1, 5, 7–9), TAPT
  read the text, never the labels, of every training comment, including the held-out rows. That
  tilts those comparisons toward TAPT, so read the size of those TAPT gains as approximate. The
  leaderboard results of the TAPT recipe (Runs 10 and 11) do not have this tilt.

## Summary

| task | stage | best result | runs |
|---|---|---|---|
| A | TF-IDF + LinearSVC floor | 0.8103 five-fold | 3–4 |
| A | MuRIL, then MuRIL + SVM blends | 0.8188 CodaBench | 7–12 |
| A | combined corpus, three-member blend | 0.8250 nested OOF | 19–22 |
| A | Gemma-4-12B QLoRA | 0.8610 holdout (two seeds) | 23–30 |
| A | Gemma + TAPT, TAPT bag | 0.8721 holdout (four models) | 31–37 |
| A | **final: 12 TAPT models** | **0.8250 official macro-F1** | 29, 35, 36 |
| B | TF-IDF + LinearSVC floor | 0.5948 five-fold | before Run 1 |
| B | TAPT MuRIL | 0.6013 five-fold | 1–6 |
| B | one reinitialized layer + R-Drop | 0.6410 CodaBench | 7–11 |
| B | Gemma-4-12B QLoRA | 0.6872 holdout (two seeds) | 19–30 |
| B | Gemma + TAPT, TAPT bag | 0.7056 holdout (two models) | 34–38 |
| B | **final: 8 TAPT models** | **0.7367 official macro-F1** | 36, 37 |

The same path on both tasks: a strong linear floor, small gains from tuning MuRIL, then the
largest single jump from replacing MuRIL with a 4-bit QLoRA decoder LLM that reads the romanized
text directly. Last came averaging Gemma models built on task-adaptive pretraining (TAPT), which
gave the best holdout results on both tasks.

---

## Task A: Hate / Non-Hate

### Baselines (Runs 1–5)

<a id="a1"></a>**Run 1: MuRIL baseline.** `google/muril-base-cased`, 6 epochs, default cleaning
(HTML unescaping, mojibake repair). Best macro-F1 0.7937 at epoch 6 on a stratified validation
split of `binary_train.csv`.

<a id="a2"></a>**Run 2: demojized MuRIL.** MuRIL's wordpiece vocabulary has no emoji, and BERT
wordpiece maps a whole word containing one to `[UNK]`. Rewriting emoji as English words cut the
`[UNK]` rate from 0.79% to 0.05% and raised the score to 0.8023 at epoch 5 (+0.009). Demojization
became the Task A default for MuRIL.

<a id="a3"></a>**Run 3: TF-IDF + LinearSVC.** Word + character n-grams: 0.8039 five-fold. This
is a different protocol from Run 1's validation split, but the linear model was at least level
with stock MuRIL, which made it the floor to beat.

<a id="a4"></a>**Run 4: demojized TF-IDF + LinearSVC.** 0.8103 five-fold, the floor every later
Task A model was measured against. The SVM stayed in every MuRIL-era submission.

Runs 1–4 were measured with early revisions of the scripts, before duplicate removal. The final
`baseline_svm.py` gives 0.8104 (Run 3 setting) and 0.8073 (Run 4 setting) on the 6,401
deduplicated rows (rerun for this record); 0.8073 is the SVM component quoted from Run 9 on.

<a id="a5"></a>**Run 5: synthetic training comments.** 150 generated comments, then a variant with
corpus-measured surface noise (elongation, capitals, emoji). Added to training folds only:

| training data | real-vs-synthetic AUC | OOF macro-F1 |
|---|---|---|
| real only | n/a | 0.8032 |
| + 150 synthetic | 0.942 | 0.8053 (+0.0022) |
| + 150 synthetic, surface noise | 0.937 | 0.8063 (+0.0031) |

A linear model separates generated from real comments at AUC 0.94, and the surface transform
moved that by only 0.005: the lexical distribution, not the surface, gives them away. The gain is
inside noise and was not pursued. The data also showed a labelling convention worth knowing:
`Gender` is assigned when a gendered slur is present, whoever the target is. Task A is balanced,
so it is the least promising place for augmentation; Task B's two small classes were the better
candidate, but augmentation was not tried there.

#### Preparation shared by both tasks (2–10 September)

- **Duplicate comments.** The released training files carry the same comment under several ids:
  35 duplicate groups in Task A (42 redundant rows, 3 groups labelled both ways) and 12 in Task B
  (14 redundant rows, 2 conflicting groups). Left in, a duplicate gets double weight in the loss
  and can land on both sides of a fold boundary, which inflates out-of-fold scores. Agreeing
  copies were collapsed and conflicting groups dropped whole, since the annotators themselves
  disagreed on them: Task A 6,446 → 6,401 rows, Task B 3,159 → 3,143. Every script drops the
  same rows, so out-of-fold matrices stay aligned for blending. Runs 1–4 predate this.
- **External corpus.** The Kannada portion of OffensEval-Dravidian turned out to be romanized
  YouTube comments, like HASTIKA, although its dataset card says Kannada script. Targeted insults
  became `Hate` and `Not_offensive` became `Non-Hate`. Untargeted profanity was dropped, because
  that is the band HASTIKA's annotators may have called `Non-Hate`; native-script and
  non-Kannada rows were dropped too. That left 3,247 comments (791 `Hate`). Rows matching a
  HASTIKA comment were removed. Spot checks found plain mislabels, which is why its labels were
  only ever considered as a separate first stage, and the final systems read only its text.
- **XLM-R.** The first transformer trainer, run for three epochs on CPU. The loss sat at ln 2
  (the trivial solution) for about two epochs, and validation macro-F1 was 0.566 after epoch 1
  and 0.716 after epoch 2, still below the SVM floor. This is why every later encoder trained for
  six epochs.

#### Measurements on the TF-IDF floor (September)

Cheap CPU checks made around the baselines, all five-fold on Task A:

| change | macro-F1 | note |
|---|---|---|
| raw text (21,363 word types, 70.4% seen once) | 0.8032 | reference for the next three rows |
| collapse 3+ repeated characters to 2 | 0.8003 | |
| the same, aggressive | 0.8017 | |
| "transliteration-lite" (th→t, dh→d, vowel collapse) | 0.8043 | removes 10.6% of word types, +0.001 |
| decision threshold fitted nested, instead of 0.5 | 0.8108 against 0.8073 | +0.0035 |
| external OffensEval `Hate` / `Non-Hate` labels added | 0.8103 against 0.8073 | +0.0030; trains on external labels |

The vocabulary is fragmented by agglutinative morphology, not by spelling noise: 65.2% of the
words seen once share a 5-character stem with another word, and the clusters are verb paradigms
(`matha*` 106 forms, `madid*` 86, `matad*` 74). Script conversion cannot merge those, which is
why the transliteration simulation bought nothing and real back-transliteration was not built.

Error analysis of the SVM floor: 92% of all errors are in comments with no profanity, while the
18% of rows that contain a slur are already 91.6% accurate. Features built from word lists
(profanity counts, gazetteers) therefore target rows that are already right. A held-out
comment's nearest training neighbour has a median cosine similarity of 0.27, so surface matching
fails for most comments.

### MuRIL and blends on the leaderboard (Runs 6–16, 17–20 September)

The MuRIL recipe (`hastika.research.models.muril`), with the reason for each non-default choice:

| choice | why, for this data |
|---|---|
| `--demojize` | wordpiece maps a whole word containing an emoji to `[UNK]` (Run 2) |
| mean ⊕ max pooling | comments are short (median 18 tokens) and the signal is often one or two slur tokens; max pooling keeps them |
| reinitialize the top 2 layers | top layers are the most MLM-specialized and transfer worst on small data |
| layer-wise LR decay 0.9 | protects the lower layers' handling of transliterated Indic from 6.4k rows |
| FGM, ε = 1.0 | 70% of word types are seen once, so their embeddings overfit freely (~1.8× step cost) |
| EMA, decay 0.999 | fine-tuning at this size is high-variance |
| 6 epochs, 6% warmup | XLM-R needed 2 full epochs just to leave the trivial solution |
| two evaluations per epoch | validation F1 moved sharply between epochs (0.566 → 0.716 for XLM-R) |
| label smoothing 0.05, no class weights | crowd-labelled data; the classes are near-balanced (3,286 / 3,160) |

Averaged over 14 fold-runs, validation macro-F1 was still rising at epoch 6 (0.7852, 0.7906,
0.7944 at epochs 4–6), with the gain halving each epoch.

<a id="a6"></a>**Run 6: full-data R-Drop against a matched control** (never run). Five seeds per
arm, differing only in `--rdrop 0.5`, each blended with the SVM at Run 9's fixed weights. A
matched control was needed because Run 10's MuRIL component used a single seed. Run 16 is the
cheaper holdout version; `run_rdrop.sh` is the five-fold version.

<a id="a7"></a>**Run 7: MuRIL validation submission (17 Sep).** CodaBench 0.8163 / 0.8164, above
the SVM floor's 0.8103 five-fold (a different protocol). The exact flags of the submitted artifact
were not recorded.

<a id="a8"></a>**Run 8: TAPT holdout comparison** (partly failed). Stock MuRIL against MuRIL
adapted by eight epochs of masked-LM training, both then fine-tuned with the demojized,
two-layer-reinit recipe. TAPT read only the training-side comments and the external corpus. Both
arms trained, but a summary assertion failed and no holdout score was captured.

<a id="a9"></a>**Run 9: MuRIL + TF-IDF out-of-fold blend (17 Sep).** Five folds for both
components: SVM 0.8073, MuRIL 0.7932. A 57% SVM / 43% MuRIL blend scored 0.8233 out-of-fold
(nested 0.8230), but only **0.7890** on CodaBench, below MuRIL alone. A local-to-official
mismatch, so the blend weights were kept and the fit changed instead.

<a id="a10"></a>**Run 10: full-data ensemble (18 Sep).** Both components refitted on all 6,401
rows at Run 9's fixed 57/43 weights, no weights fitted on validation: CodaBench **0.8187 /
0.8189**, +0.0024 over Run 7 and the best Task A result at the time.

<a id="a11"></a>**Run 11: RBF SVM on frozen MuRIL embeddings (19 Sep).** No fine-tuning; mean-max
pooled embeddings into `SVC(rbf, C=2)`. It predicted 55.2% `Hate` against a 49.1% training
prior. CodaBench about 0.71 / 0.70: freezing the encoder costs far more than a kernel head
recovers. It is also useless as a blend member. On a binary task, when two systems disagree
exactly one of them is right, so the arithmetic settles it without labels:

| quantity | value |
|---|---|
| rows where it disagrees with Run 10 | 184 of 806 |
| on those rows, Run 10 is right | 76.0% |
| on those rows, this model is right | 24.0% |
| on rows where they agree, both are right | 83.6% |

A blend helps only by overruling the stronger member, and here it would be wrong three times in
four when it did. On the 622 rows where the two agree, both are still wrong 16.4% of the time.
That shared residue, mostly comments with no profanity at all, is where the remaining Task A
errors are.

<a id="a12"></a>**Run 12: one reinitialized layer and a refitted blend (19–20 Sep).** Three
changes at once: one reinitialized top layer instead of two, the blend weight refitted, the
threshold fitted. CodaBench 0.8188 / 0.8189, a tie with Run 10 (+0.0001, under one row). The
out-of-fold stage explains the tie:

| quantity | value |
|---|---|
| SVM, five-fold | 0.8073 |
| MuRIL, one reinitialized layer, five-fold | 0.7894 |
| nested blend | 0.8164 |
| Run 9's fixed 0.57 at threshold 0.50, in-sample | 0.8188 |
| refitted weight and threshold | SVM 0.62 / MuRIL 0.38, threshold 0.48 |
| per-fold picks | weight 0.45 to 0.85, threshold 0.46 to 0.52 |
| predicted Hate rate | 0.488 against a 0.491 prior |

One layer made MuRIL weaker, the refitted weight leaned further on the SVM, and the two effects
cancelled. The blend surface is flat, so the weight was never the constraint. Blending itself
pays: +0.009 nested over the better component.

<a id="a13"></a>**Run 13: external labels** (never run). Whether OffensEval-Dravidian's labels
help MuRIL when mixed into the folds or used as a separate first stage, against a control. The
TF-IDF floor had gained +0.0030 from them (above); using external labels was a rules question
that was never settled, and no final system uses them.

<a id="a14"></a>**Run 14: XLM-R as a third member** (never run). Run 11 had lowered its expected
value: a member that disagrees a lot but is usually wrong when it does adds nothing.

<a id="a15"></a>**Run 15: capacity and schedule** (never run; superseded before it started).
MuRIL-large and a longer schedule against the one-layer control, five folds.

<a id="a16"></a>**Run 16: short paired R-Drop holdout** (never run). An 80-minute paired holdout
test of R-Drop 0.5, kept deliberately minimal (no TAPT, no longer schedule).

### Translation, and the released validation labels (Runs 17–22, 21 Sep – 5 Oct)

<a id="a17"></a>**Run 17: translate to English, then hateBERT (5 Oct).** The organizers'
published method (`shankarb14/SLM-Impact`) transliterates romanized comments with IndicXlit,
translates them to English with IndicTrans2 and classifies the English with frozen encoders. The
model half was not worth copying (Run 11 had already measured a frozen encoder), but translation
might remove MuRIL's tokenization problem, opens `GroNLP/hateBERT` (pretrained on English abuse)
and would be a decorrelated ensemble member. The risk was that translation sanitises profanity,
and 18% of Task A rows carry a slur. So the notebook first gated on slur survival over 300
slur-bearing comments, with a spelling-tolerant slur lexicon (phonetic skeleton, bounded edit
distance, compound splitting) and English glosses injected before translation. IndicTrans2 could
not load on Kaggle's Python 3.13, so Sarvam-Translate was the only translator (about 3 h for
7,193 rows on a T4). Trained on 6,114 rows and scored on the fixed holdout:

| arm | macro-F1 | Hate F1 |
|---|---|---|
| char SVM, romanized | 0.8117 | 0.8069 |
| char SVM, canonicalised slur spellings | 0.8146 | 0.8106 |
| MuRIL, romanized, 3 seeds | 0.7803 | 0.7775 |
| SVM 0.57 + MuRIL 0.43 | **0.8155** | 0.8135 |
| hateBERT on translated English, 3 seeds | 0.7275 | 0.7232 |
| SVM + hateBERT, 50/50 | 0.7924 | 0.7899 |
| SVM + MuRIL + hateBERT, thirds | 0.8091 | 0.8071 |

| paired bootstrap | difference | 95% CI | P |
|---|---|---|---|
| hateBERT - MuRIL | -0.0529 | [-0.0780, -0.0268] | 0.00 |
| thirds - (SVM + MuRIL) | -0.0065 | [-0.0186, +0.0056] | 0.15 |
| (SVM + hateBERT) - SVM | -0.0194 | [-0.0362, -0.0019] | 0.02 |
| canonicalised SVM - SVM | +0.0028 | [-0.0019, +0.0082] | 0.88 |

Raw slur survival was 1%: the translator either echoed the romanized text or produced a fluent
sentence unrelated to the comment ("Paapa how she matadtale with avar tayi" became "How she
killed her uncle by eating him"). The injected glosses survived (70%) only because they were
already English. hateBERT and MuRIL disagree on 21% of rows, but hateBERT is the one that is
wrong there, the Run 11 pattern again. Open translators of the time destroyed the signal Task A
depends on, so the line was closed. The spelling canonicaliser is a small, unconfirmed positive
for the SVM. The conclusion pointed to Run 23: read the romanized text directly with a
multilingual LLM.

<a id="a18"></a>**Run 18: refit on train + released validation, predict the test file** (failed
before training). Run 12's recipe on the larger corpus. It stopped because the released
validation CSV was missing from a stale Kaggle clone.

<a id="a19"></a>**Run 19: Run 10's recipe out-of-fold on the combined corpus (1 Oct).** Train +
released validation, 7,252 rows deduplicated to 7,193: 0.8213 / 0.8215.

<a id="a20"></a>**Run 20: Run 12 against Run 10 on the same folds (1 Oct).** 0.8204 / 0.8204
against 0.8213 / 0.8215. One reinitialized layer does not help Task A; Task A keeps two.

<a id="a21"></a>**Run 21: character-only SVM as a third member.** Character-only TF-IDF had scored
0.8138 against 0.8073 for word + character with the same calibrated classifier and folds (see
the analysis below). Word+char SVM, char-only SVM and MuRIL: searched out-of-fold 0.8270 /
0.8271, **nested 0.8250**, against 0.8184 for the best single model.

<a id="a22"></a>**Run 22: Run 21's blend, released-validation score and test predictions.** The
fixed 0.60 / 0.33 / 0.07 weights, fitted on the original training data: 0.8150 / 0.8151 on the
806 released validation rows. Then refitted on train + validation to predict the test file.

#### Why MuRIL loses to a linear model on Task A (20 Sep)

A 237M-parameter pretrained transformer scoring 0.7894 against a TF-IDF model's 0.8073 on the
same folds needs an explanation, and it is measurable. MuRIL's tokenizer shatters exactly the
words that carry the signal, and inconsistently:

| word | MuRIL wordpieces |
|---|---|
| `sule` | `su` `##le` |
| `soole` | `so` `##ole` |
| `thu` / `thuu` | `th` `##u` / `th` `##uu` |
| `bevarsi` | `be` `##vars` `##i` |
| `saayisbeku` | `sa` `##ayi` `##sb` `##eku` |

`sule` and `soole` are the same word spelled two ways and receive unrelated fragments.

| measurement | value |
|---|---|
| word tokens / word types | 80,108 / 26,945 |
| word types seen exactly once | 74.4% |
| mean wordpieces per word type | 2.43 |
| word types MuRIL keeps as one piece | 24.9% |
| MuRIL vocabulary this corpus uses | 8,810 of 197,258 (4.47%) |
| `[UNK]` rate after demojization | 0.054% |

Character n-grams generalize across spelling variants by construction. Same folds, same
LinearSVC: char 2–5 grams 0.8152, word 1–2 + char 2–5 0.8065, word 1–2 only 0.7585. Adding word
features to character features costs 0.009. With the repository's calibrated classifier,
char-only scored 0.8138 against 0.8073 for word + char (+0.0065, no GPU), which led to Run 21.
Fine-tuning on 6,401 rows reweights MuRIL's existing fragment embeddings; it cannot rebuild them
for romanized Kannada.

### Gemma-4-12B (Runs 23–30, 5–14 October)

The representation change that had just worked on Task B (Task B Run 20: +6.4 points over MuRIL)
was applied to Task A. Gemma-4-12B is loaded in 4-bit NF4 with LoRA rank 16 on attention and MLP,
and a classification head reads the last prompt token. MuRIL lessons that carry over: seed
averaging, R-Drop, the char SVM blend, and later TAPT. Layer reinitialization and demojization are
MuRIL-specific and do not apply. Larger Gemma models (26B, 31B) do not fit a 15 GB T4 in 4-bit.

Every Gemma notebook runs standalone on Kaggle: it reuses an earlier run's outputs only if they
are attached (searching all inputs), otherwise trains what it needs; it clones the repository to
`/tmp` so the output contains only results; and it stops at 11 h, before Kaggle's 12 h limit. T4s
have no bf16, so training is fp16, and a model that overflows exits with code 3 instead of
training on garbage.

<a id="a23"></a>**Run 23: Gemma-4-12B QLoRA, holdout then full fits (5 Oct).** 3 epochs, seeds 42
and 43, on 6,114 rows, scored on the 1,079-row holdout; then both seeds fitted on all 7,193 rows
(5.2 h on T4 x2):

| arm | macro-F1 | Non-Hate F1 | Hate F1 |
|---|---|---|---|
| char SVM | 0.8117 | 0.817 | 0.807 |
| SVM 0.57 + MuRIL 0.43 (Run 17) | 0.8155 | | |
| **Gemma, seed 42** | **0.8563** | 0.856 | 0.857 |
| Gemma, seed 43 | 0.3372 (collapsed) | 0.674 | 0.000 |
| Gemma, both seeds averaged | 0.8545 | 0.854 | 0.854 |
| Gemma + SVM, 50/50 | 0.8471 | 0.848 | 0.846 |

| paired bootstrap | difference | 95% CI | P |
|---|---|---|---|
| Gemma - SVM | +0.0428 | [+0.0205, +0.0668] | 1.00 |
| blend - Gemma | -0.0074 | [-0.0226, +0.0078] | 0.16 |
| blend - SVM | +0.0353 | [+0.0187, +0.0531] | 1.00 |

This is the largest Task A gain in the project: +4.3 over the char SVM and +4.1 over the
MuRIL-era best blend. Seed 42 was still rising at the last epoch (0.817, 0.846, 0.856). The SVM,
the strongest component of every MuRIL-era submission, is now redundant: blending it costs 0.007.
Seed 43 sat at chance from its first logged loss and predicted Non-Hate for every row. Both
full-data fits trained normally (final loss 0.25 and 0.27). Because of this run the classifier
gained a collapse guard: after epoch 1, a training-row sample predicted more than 95% one class
exits with code 4, and the notebooks rerun the job with seed + 1000.

<a id="a24"></a>**Run 24: four epochs against three (6 Oct).** Three epochs, seeds 42 and 43:
0.8563 and 0.8573, **0.8610** averaged. Four epochs x2: 0.8452 (-0.016, CI [-0.033, +0.001],
P 0.04). Task A keeps 3 epochs.

<a id="a25"></a>**Run 25: organizer definitions in the prompt** (prepared; no run recorded).

<a id="a26"></a>**Run 26: Gemma TAPT, holdout (6 Oct)** (partly failed). The base arm scored
0.8563 and 0.8573, 0.8610 for the pair. TAPT ran out of GPU memory at micro-batch 4 on the
14.56 GB T4, and its automatic retry failed on a classifier-only argument (`--eval-bs`), so TAPT
was not measured here. Getting Gemma TAPT to run took three fixes, each hidden behind the
previous one:

1. chunked logits through the 262k-vocabulary head (Robin);
2. the head applied to a detached copy of the hidden states, so the decoder runs one backward per
   batch instead of one per chunk (about 10x faster; gradients checked identical, max difference
   7e-9);
3. micro-batch 1 x 16 accumulation at 128 tokens (over 99% of comments), `expandable_segments`
   against fragmentation, a retry at 64 tokens, and a TAPT script that accepts and ignores
   `--eval-bs`.

TAPT-only fits now stop with an error instead of silently substituting a non-TAPT model.

<a id="a27"></a>**Run 27: Gemma TAPT, full-data fit** (prepared; no run recorded). The final
ensemble's TAPT models come from Runs 29, 35 and 36.

<a id="a28"></a>**Run 28: learning rate 2e-4 against 1e-4** (prepared; no run recorded).

<a id="a29"></a>**Run 29: four full-data fits (7–14 Oct).** Four 3-epoch seeds on all 7,193 rows,
run twice: without TAPT (5.6 h; test predictions 51.5% Hate), and with TAPT (7.6 h; TAPT 125 min,
peak 10.4 of 14.6 GB; 50.2% Hate). **The four TAPT models are members of the final ensemble.**

<a id="a30"></a>**Run 30: holdout check, 3 against 4 epochs (6 Oct).** The holdout twin of Run 23's
recipe plus two 4-epoch seeds (5.5 h):

| member / group | macro-F1 | per-epoch holdout |
|---|---|---|
| 3 epochs, seed 42 | 0.8563 | 0.817, 0.846, 0.856 |
| 3 epochs, seed 43 (rerun as 1043) | 0.8573 | 0.810, 0.858, 0.857 |
| 4 epochs, seed 44 | 0.8211 | 0.778, 0.815, …, 0.821 |
| 4 epochs, seed 45 | 0.8387 | 0.794, 0.822, …, 0.839 |
| **3 epochs x2** | **0.8610** | |
| 4 epochs x2 | 0.8313 | |
| all four | 0.8563 | |

All four - 3 epochs x2: -0.005, CI [-0.016, +0.007], P 0.20. All four - 4 epochs x2: +0.025,
CI [+0.013, +0.037]. Four epochs overfit Task A, unlike Task B: Task A has twice the rows and
peaks by epoch 2–3. Mixing in weaker members makes the average worse. The collapse guard worked in
production here: seed 43 predicted Non-Hate for every training row after epoch 1, exited with
code 4, and was rerun as seed 1043.

### TAPT and the TAPT bag (Runs 31–38, 14–16 October)

<a id="a31"></a>**Run 31: TAPT holdout twin, TAPT x4 against base (14 Oct).** TAPT for 1 epoch on
the 6,114 training comments plus the external corpus, then four 3-epoch classifiers. The LM loss
fell from 7.5 to 5.6 within the epoch.

| per seed, epoch 3 | 42 | 43 | 44 | 45 | mean |
|---|---|---|---|---|---|
| TAPT | 0.8480 | 0.8498 | 0.8526 | 0.8573 | 0.8519 |
| base | 0.8563 | (cut) | 0.8508 | 0.8378 | 0.8483 |

**TAPT x4: 0.8702**, against 0.8582 for base x3. Single TAPT models match single base models; the
gain is in the average. The arms have different seed counts (4 against 3: base seed 43 collapsed,
was rerun, then cut by the 11 h stop) and there is no paired interval, so the size of the gain is
uncertain. This ran as a modified copy of the notebook; the file here is the regenerated version.

<a id="a32"></a>**Run 32: TAPT holdout twin, regenerated variant** (prepared; no run recorded).

<a id="a33"></a>**Run 33: TAPT + R-Drop holdout twin (15 Oct).** One TAPT run, then two plain and
two R-Drop 0.5 classifiers from the same adapter (9.3 h):

| | seed 42 | seed 43 | two averaged |
|---|---|---|---|
| TAPT | 0.8563 | 0.8573 | 0.8619 |
| TAPT + R-Drop | 0.8526 | 0.8582 | 0.8647 |
| **all four** | | | **0.8721** |

| paired bootstrap | difference | 95% CI | P |
|---|---|---|---|
| TAPT + R-Drop x2 - TAPT x2 | +0.0028 | [-0.0121, +0.0176] | 0.64 |
| all four - TAPT x2 | +0.0102 | [+0.0000, +0.0204] | 0.98 |
| all four - TAPT + R-Drop x2 | +0.0074 | [-0.0037, +0.0185] | 0.91 |

R-Drop alone is not a better recipe. Averaging both kinds of model scored higher than either
pair, so the R-Drop models join the final average as extra members.

<a id="a34"></a>**Run 34: 3 epochs, two-seed full-data fit** (prepared; no run recorded).

<a id="a35"></a>**Run 35: TAPT + R-Drop full-data fit (15 Oct).** Two 3-epoch seeds with R-Drop
0.5 from one TAPT adapter (7.5 h). **Both models are members of the final ensemble.**

<a id="a36"></a>**Run 36: the TAPT bag, full data (15–16 Oct).** All four Run 31 models came from
one TAPT run. Models built on different TAPT runs were expected to disagree more, which is what
averaging needs. Each session runs its own TAPT with a distinct `TAPT_SEED`, then two
classifiers, in about 4.5 h, and the sessions run in parallel:

| session | TAPT seed | classifier seeds |
|---|---|---|
| 1 | 1 | 42, 43 |
| 2 | 2 | 44, 45 |
| 3 | 3 | 46, 47 |

All three sessions completed (4.4–4.8 h each, no collapses). **Their six models are members of the
final ensemble.**

<a id="a37"></a>**Run 37: the TAPT bag, holdout (15 Oct).** Session 1's settings on the 85% split
(TAPT peaked at 8.4–9.6 GB): 0.8535 and 0.8610 single, **0.8675** for the pair. A second,
independent TAPT run scored above every no-TAPT ensemble (0.8582–0.8610), so the TAPT result is
not tied to one TAPT seed.

<a id="a38"></a>**Run 38: Qwen3-8B as a second model family** (prepared; no run recorded).
Qwen3-8B-Base was the second-best reader of Kanglish in Task B Run 19's screen, and a second
model family was the main idea left for adding different errors. Not part of the final ensemble.

---

## Task B: six hate categories

The classes are `Gender`, `Geo-political`, `Others`, `Political`, `Religion` and `Violence`.
Macro-F1 weights `Geo-political` (186 training rows) and `Violence` (221) equally with `Gender`
(1,362), so the small classes decide the score.

### Preparation before the sweep (10–12 September)

The Task B pipeline started as the Task A MuRIL recipe with only the changes six classes need:
class-weighted loss, argmax decoding, and a longer input cap. Before any GPU result was counted,
CPU measurements set the floor and located the errors, and four trainer bugs were fixed.

**Floor.** TF-IDF + LinearSVC: 0.5948 five-fold (`Violence` 0.35, `Others` 0.43). Fourteen
settings (C, character and word n-gram ranges, `min_df`, logistic regression, SGD) all sat at
0.59 ± 0.005, best 0.5979, inside fold noise, so the floor was at its plateau. Calibrating the
SVM's scores cost three points (0.5628), because calibration pulls the scores back toward the
class priors and undoes the balanced class weights the small classes depend on. The SVM's margins
are softmaxed instead.

**Where the errors are.** On the floor, `Violence` loses 103 of its 221 rows to `Gender`, and
`Others` loses 282 of 447. Reading those rows explains why: gendered slurs are used as generic
insults against TV channels and politicians, so the word points at `Gender` while the label
follows the target. Task B is mostly a problem of identifying the target.

**Lexical signals.** `Political`, `Religion` and `Geo-political` are decided by closed sets of
names; a gazetteer fitted inside each fold gives 4–9 times the class base rate. `Violence` is a
speech act instead of a topic: the necessitative suffix *-beku* on a violent verb stem raises
P(`Violence`) from 0.070 to 0.184. Tags built from these signals moved the SVM by -0.007
(see [side measurements](#measurements-outside-the-numbered-runs)), because the SVM already has those words as
features, so they shipped switched off.

**Blends need a nested check.** On four CPU models, blend weights searched on the same
out-of-fold rows scored 0.6032, against 0.5979 for the best member and 0.5925 when the weights
were refitted out of fold. From then on every blend reported its nested score beside the
in-sample one.

| trainer bug | effect | fix |
|---|---|---|
| EMA weights not bias-corrected | at 942 steps per Task B fold, 0.999^942 = 39% of every evaluated checkpoint was still the initialization (9% at Task A's ~2,400 steps) | Adam-style bias correction |
| best checkpoint picked and scored on the same fold rows | optimistic out-of-fold scores | both the `best` and the unbiased `last` score reported |
| 128-token cap inherited from Task A | truncated 2.03% of Task B validation-phase inputs | 192 tokens (0.51%) |
| TAPT ran out of GPU memory on every T4 run, unnoticed | `cmd \| tee log` exits with `tee`'s status; MuRIL's 197,285-word MLM logits needed 2.26 GiB in one allocation | micro-batch 4 with 4 accumulation steps; commands run through a helper that raises on failure |

**Other text.** No further usable in-domain corpus was found: OffensEval-Dravidian Kannada was
already in use, DravidianCodeMix is Tamil, and HopeEDI is distributed only through Drive links.
The alternative was a warm start: `Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL` has a
vocabulary byte-identical to MuRIL's, so it went into Run 1 as an arm.

### MuRIL and TAPT (Runs 1–6, 12–14 September)

<a id="b1"></a>**Run 1: overnight encoder and loss sweep (12 Sep).** 8.44 h. Ideas ranked against
stock MuRIL on a 15% holdout (472 rows), the best four promoted to five folds:

| five-fold, 3,143 rows | macro-F1 |
|---|---|
| **TAPT MuRIL** (submitted: 0.5922 CodaBench) | **0.6013** |
| weight-searched blend, nested | 0.6002 |
| TF-IDF + LinearSVC | 0.5948 |
| stock MuRIL | 0.5727 |
| R-Drop 0.5 | 0.5614 |
| focal loss, gamma 2 | 0.5572 |

| holdout arm, 472 rows | macro-F1 |
|---|---|
| TAPT MuRIL | 0.5972 |
| stock MuRIL | 0.5948 |
| R-Drop | 0.5845 |
| focal loss | 0.5779 |
| abusive-tuned MuRIL warm start | 0.5718 |
| HingRoBERTa | 0.5678 |
| XLM-R base | 0.5561 |
| mDeBERTa | 0.1002 (collapsed; 3e-5 is too high for it) |
| MuRIL-large | (cut by the time budget) |

**Only task-adaptive pretraining helped, +2.9 five-fold.** TAPT here is masked-LM training on
Task B's training comments plus the external Kannada corpus. The holdout ranked the four
promoted arms in the same order as five folds but read about 2 points high. Blending did not pay,
and neither did a macro-F1-optimal per-class decode (0.5856 against 0.5948 for plain argmax on
the floor); both lost their nested checks. The predicted distribution mattered: the SVM called
`Violence` on 6 of 395 test rows against a 7% prior, forfeiting most of that class, while the
submitted TAPT model called it 24 times.

<a id="b2"></a>**Run 2: rebuild of the Run 1 winner.** Reruns only the TAPT arm with fixed seeds,
in case the overnight output was lost (about 2 h instead of 8.5). No separate log was kept; the
recipe's recorded scores are Run 1's.

<a id="b3"></a>**Run 3: five-seed full fits of the sweep's ideas (13 Sep)** (score unconfirmed).
9.35 h; six ideas, each trained five times on all 3,143 rows. TAPT still held 5% of its text back
and dropped one-word comments, so it used 6,044 of 6,406 comments.

| arm | agrees with Run 1's submission | largest drift from the class prior |
|---|---|---|
| TAPT MuRIL | 91.1% | 3.2 points |
| stock MuRIL | 87.1% | 2.7 |
| R-Drop 0.5 | 85.8% | 3.0 |
| focal loss | 85.1% | 4.0 |
| abusive-tuned warm start | 81.3% | 3.0 |
| HingRoBERTa | 81.0% | 3.9 |

The TAPT arm's 0.6007 CodaBench score was inferred from download timing and never confirmed.

<a id="b4"></a>**Run 4: one full fit on every training row** (prepared; no run recorded). The one
available comparison pointed the other way: a single model on 85% of rows scored 0.5972, while
five fold models on 80% each averaged to 0.6013.

<a id="b5"></a>**Run 5: data-processing grid (13–14 Sep).** 8.87 h, every cell on the same 472-row
holdout. TAPT corpus D0 = Kannada only; D1 = plus about 58,000 Tamil and Malayalam
OffensEval-Dravidian comments, one epoch so the compute matches. Vocabulary V0 = MuRIL's as
shipped; V1 = extended with frequent word-forms, new embeddings initialized as the mean of their
pieces. Aux = an auxiliary violent-language head, weight 0.3.

| config | seeds | macro-F1 | against stock MuRIL |
|---|---|---|---|
| D0, V0, no aux | 1 | **0.5972** | +0.0024 |
| stock MuRIL, V0, no aux | 1 | 0.5948 | 0 |
| D0, V0, no aux | 3 | 0.5917 | -0.0031 |
| D0, V0, aux | 1 | 0.5916 | -0.0032 |
| D1, V0, aux | 1 | 0.5875 | -0.0073 |
| D1, V0, no aux | 1 | 0.5865 | -0.0083 |
| stock MuRIL, V0, no aux | 3 | 0.5820 | -0.0128 |
| D0, V1, no aux | 1 | 0.5656 | -0.0292 |
| D0, V1, aux | 1 | 0.5546 | -0.0402 |
| stock MuRIL, V0, aux | 1 | 0.5495 | -0.0453 |
| D1, V1, no aux | 1 | 0.5117 | -0.0831 |
| D1, V1, aux | 1 | 0.5071 | -0.0877 |

None of the levers helped, and the top seven cells are one result inside a 2.7-point bootstrap
SD. Vocabulary extension is the only real effect, and it costs 3 to 8 points: new embeddings
cannot be trained from 3k comments. Three seeds scored below seed 42 alone for both stock and
TAPT, so seed 42 was a lucky draw on this split. Here TAPT read the holdout rows' text, so the
grid is tilted toward TAPT.

<a id="b6"></a>**Run 6: TAPT MuRIL on 100% of the data** (never run). TAPT on all 6,406 comments
and five seeds on all 3,159 rows; Run 10 replaced it.

### Reinitialization and R-Drop (Runs 7–11, 16–17 September)

<a id="b7"></a>**Run 7: one against two reinitialized layers (16 Sep).** One shared TAPT checkpoint
(6,044 of 6,362 cleaned comments, 318 held out for MLM perplexity), five folds, seed 42: **one
layer 0.6102** against two layers 0.6013 (+0.009); accuracy 0.6592 against 0.6557.

| class | two layers | one layer | change |
|---|---|---|---|
| Gender | 0.735 | 0.730 | -0.005 |
| Geo-political | 0.592 | 0.590 | -0.002 |
| Others | 0.454 | 0.476 | +0.022 |
| Political | 0.783 | 0.797 | +0.014 |
| Religion | 0.716 | 0.717 | +0.001 |
| Violence | 0.328 | 0.352 | +0.024 |

<a id="b8"></a>**Run 8: confirmation at seeds 43 and 44 (17 Sep).** One layer 0.6127 and 0.6082,
two layers 0.6026 and 0.6083. Mean seed score 0.6105 against 0.6055; averaged probabilities
0.614 against 0.612. One layer kept.

<a id="b9"></a>**Run 9: no reinitialization (17 Sep).** Seeds 43 and 44: 0.5854 and 0.5760 with
no reinitialization, against 0.6127 and 0.6082 for one layer; averaged, 0.584 against 0.614.
Dropped.

<a id="b10"></a>**Run 10: full-data one-layer fit (17 Sep).** TAPT on all 6,406 comments (training
text plus the external corpus), five seeds on all 3,159 rows: CodaBench **0.6299 / 0.6886**.

<a id="b11"></a>**Run 11: full-data R-Drop (17 Sep).** Run 10 plus R-Drop 0.5: CodaBench **0.6410 /
0.6937** (+0.0111 / +0.0051 over Run 10), the best MuRIL result on Task B and the recipe every
later idea stacked on. That gain is about a third of one CodaBench standard deviation, and the
only local R-Drop measurement so far (Run 1) had been negative.

### Ideas on the 0.6410 recipe (Runs 12–18, 18–20 September)

<a id="b12"></a>**Run 12: context-dependent Violence decoding** (never run). A violent word's
meaning depends on its company. Among the 420 training comments containing one: with media
context present, `Others` is 0.27 and `Violence` 0.18; with no target named, `Violence` is 0.34
and `Others` 0.06. A single weight per class cannot capture that, which is why per-class decode
weights lost every nested check. `context_decode` fits a lift per cell instead; on the calibrated
TF-IDF SVM that was worth +2.9 points nested (0.5630 to 0.5823, `Violence` F1 0.24 to about
0.35). This run would have tested it on MuRIL's five-fold probabilities.

<a id="b13"></a>**Run 13: topic, mood and address tags** (never run). Fold-safe tags added to the
input text, against the winning recipe. On the SVM the same tags had measured -0.007.

<a id="b14"></a>**Run 14: full-data context-tag fit (20 Sep)** (partly failed). All five fits
completed. A post-training assertion looked for an `rdrop=0.5` log string the trainer did not
print and stopped the notebook. The ZIP was recovered, but no score was recorded.

<a id="b15"></a>**Run 15: one change at a time on Run 11** (never run; prepared as
`hastika.research.experiments.task_b.stack_full`). Arms: 10 epochs, an unchanged control to
measure run-to-run noise on the leaderboard, context tags, no FGM.

<a id="b16"></a>**Run 16: MuRIL-large, full data (20 Sep).** Run 11's recipe on MuRIL-large. The
ZIP was written, but no score was recorded; Gemma superseded MuRIL.

<a id="b17"></a>**Run 17: MuRIL-large, out-of-fold** (failed in training). TAPT completed, then
fold 1 ran out of GPU memory. A retry with a 128-token cap and micro-batch 2 was planned; none is recorded.

<a id="b18"></a>**Run 18: refit on train + released validation** (failed before training). Kaggle
could not clone the repository.

### A decoder LLM as the encoder (Runs 19–23, 4–6 October)

Every Task B gain so far had come from the representation (TAPT +2.9, one reinitialized layer
+0.9 over two), and the score was lost on `Violence` (0.35), `Others` (0.48) and `Geo-political`
(0.59). `Violence` depends on romanized verbs (`saayisbeku`, `encounter maadlebeku`, `odiri`) that
MuRIL's wordpieces shatter. So the encoder was replaced by a decoder LLM fine-tuned in 4-bit
QLoRA, with a classification head on the last token of a prompt naming the six options. From here
on everything uses the fixed 530-row holdout of train + released validation (3,532 rows),
committed so it cannot drift with library versions; the released test text was never used for
training, TAPT or screening.

<a id="b19"></a>**Run 19: LLM screen against MuRIL (4 Oct)** (partly failed). Candidates were
screened by unlabelled bits per character on 600 training comments (lower reads the text better;
comparable across tokenizers): Gemma-4-12B **3.011**, Sarvam-1 3.081, Qwen3-8B-Base 3.192,
Gemma-4-E4B 4.295; `bodhan-ai/indic-translate` is gated and was skipped. On the holdout:

| arm | macro-F1 | Gender | Geo-pol | Others | Political | Religion | Violence |
|---|---|---|---|---|---|---|---|
| MuRIL, Run 11 recipe, 5 seeds | **0.6151** | 0.704 | 0.642 | 0.500 | 0.789 | 0.672 | 0.383 |
| Sarvam-1 QLoRA, seed 42 | 0.5567 | 0.685 | 0.488 | 0.437 | 0.757 | 0.714 | 0.259 |
| 50/50 | 0.6187 | 0.734 | 0.585 | 0.528 | 0.792 | 0.724 | 0.349 |

Sarvam-1 lost (-0.058, CI [-0.104, -0.014]) but was still improving at its last epoch (0.115,
0.469, 0.557), so three epochs undertrained a 2B model. The two disagree on 28.7% of rows, and
even this weak member lifted the average slightly above MuRIL. Gemma-4-12B, the arm the run was
built for, crashed on a peft/torchao version check before training; the classifier now reports
torchao as absent. The next run added length-bucketed batches (about 3.5x less padding) and a
shorter prompt.

<a id="b20"></a>**Run 20: Gemma-4-12B against MuRIL, then full fits (5 Oct).** 6.1 h on T4 x2; the
12B holdout fits took about 70 min each and peaked at 10.0 GB.

| arm | macro-F1 | acc | Gender | Geo-pol | Others | Political | Religion | Violence |
|---|---|---|---|---|---|---|---|---|
| MuRIL, Run 11 recipe | 0.6151 | 0.651 | 0.704 | 0.642 | 0.500 | 0.789 | 0.672 | 0.383 |
| Gemma, seed 42 | 0.6566 | 0.706 | 0.769 | 0.657 | 0.513 | 0.812 | 0.779 | 0.410 |
| Gemma, seed 43 | 0.6613 | 0.709 | 0.770 | 0.620 | 0.533 | 0.812 | 0.797 | 0.436 |
| **Gemma, 2 seeds** | **0.6792** | **0.728** | 0.793 | 0.712 | 0.566 | 0.806 | 0.788 | 0.410 |
| Gemma + MuRIL | 0.6825 | 0.730 | 0.787 | 0.701 | 0.594 | 0.821 | 0.787 | 0.405 |

| paired bootstrap | difference | 95% CI | P |
|---|---|---|---|
| Gemma - MuRIL | +0.0641 | [+0.0239, +0.1048] | 1.00 |
| Gemma + MuRIL - Gemma | +0.0033 | [-0.0186, +0.0248] | 0.62 |
| Gemma + MuRIL - MuRIL | +0.0674 | [+0.0343, +0.1027] | 1.00 |

Gemma - MuRIL is the largest Task B gain in the project, and not selection-optimistic, since
Gemma was picked by the unlabelled screen. Two seeds are worth about 2 points over one. Both
seeds were still rising at the last epoch (0.597, 0.629, 0.657 and 0.561, 0.630, 0.661). The
MuRIL blend adds 0.003, inside noise, so the rule fixed in advance picked Gemma alone. The
packaging cell looked for the wrong file name; Run 21 rebuilt the ZIP.

<a id="b21"></a>**Run 21: packaging Run 20's full fits (5 Oct).** Both 3-epoch full fits
retrained standalone (77 and 93 min) and averaged. Predicted class shares on the 396 test rows
against the labelled prior: Gender 41.2% / 43.1%, Geo-political 8.8% / 5.9%, Others 13.4% /
14.2%, Political 17.9% / 17.7%, Religion 11.1% / 12.1%, Violence 7.6% / 7.0%. No class is starved.

<a id="b22"></a>**Run 22: four epochs against three, then a four-model fit (5–6 Oct).** Both of
Run 20's seeds were still rising at the last epoch. Run 20's output was not attached, so 4
epochs is compared with Run 20's recorded 3-epoch scores on the same rows and seeds: unpaired, no
bootstrap interval.

| holdout | seed 42 | seed 43 | 2 seeds | Gender | Geo-pol | Others | Political | Religion | Violence |
|---|---|---|---|---|---|---|---|---|---|
| 3 epochs (Run 20) | 0.6566 | 0.6613 | 0.6792 | 0.793 | 0.712 | 0.566 | 0.806 | 0.788 | 0.410 |
| **4 epochs** | 0.6614 | 0.6653 | **0.6872** | 0.785 | 0.667 | 0.569 | 0.822 | 0.797 | **0.483** |

| per-epoch holdout | epoch 1 | epoch 2 | epoch 3 | epoch 4 |
|---|---|---|---|---|
| seed 42 | 0.557 | 0.632 | 0.654 | 0.661 |
| seed 43 | 0.022 | 0.572 | 0.674 | 0.665 |

A small, consistent gain inside noise, but the weakest class gains most (`Violence` +0.073)
while `Geo-political` falls. Four 4-epoch seeds were then fitted on all 3,532 rows (108–118 min
each); their predicted class shares match the prior within 2 points. Seed 43's loss sat at chance
(about ln 6) through epoch 2 before recovering, the second early collapse after Task A Run 23.

<a id="b23"></a>**Run 23: MuRIL's lessons on Gemma (5 Oct)** (partly failed).

| arm, 2 seeds | seed 42 | seed 43 | averaged | Violence |
|---|---|---|---|---|
| base (Run 20's recipe, retrained) | 0.6566 | 0.6631 | **0.6753** | 0.425 |
| R-Drop 0.5 | 0.6690 | 0.6900 | 0.6734 | 0.415 |

The base reproduces Run 20 (seed 42 identical, the pair 0.6753 against 0.6792). R-Drop - base:
-0.002, CI [-0.027, +0.023], P 0.44. R-Drop helped each seed but not their average: averaging two
base models gained 1.5 points, averaging two R-Drop models lost 0.6. One possible reason is that
R-Drop makes models agree more. It was left off at twice the cost. TAPT ran out of memory in the
262k-vocabulary logits; the fixes are described under Task A Run 26.

### Gemma ablations (Runs 24–33, 5–7 October)

<a id="b24"></a>**Run 24: TAPT final recipe** (prepared; no run recorded). TAPT for 2 epochs, then
four 4-epoch fits on all rows. The final ensemble uses Runs 36 and 37 instead.

<a id="b25"></a>**Run 25: LoRA capacity** (prepared; no run recorded). Rank 8/16/32,
attention-only against attention + MLP.

<a id="b26"></a>**Run 26: TAPT holdout** (prepared; no run recorded).

<a id="b27"></a>**Run 27: definitions prompt and learning rate** (prepared; no run recorded). The
prompt was measured in Run 29; the learning-rate arm has no recorded result.

<a id="b28"></a>**Run 28: frozen backbone, head only** (never run).

<a id="b29"></a>**Run 29: prompt ensemble (6 Oct).** Standard and definitions prompts, two seeds
each (4.3 h):

| arm | macro-F1 | accuracy |
|---|---|---|
| standard prompt x2 | 0.6830 | 0.7226 |
| definitions prompt x2 | 0.6840 | 0.7302 |
| **all four** | **0.6882** | **0.7321** |

Definitions alone: +0.001, CI [-0.028, +0.029], P 0.53. The ensemble: +0.005 over the standard
prompt, CI [-0.018, +0.027], P 0.67, below the P >= 0.70 bar set in advance for a recipe change.

<a id="b30"></a>**Run 30: square-root class weights (6 Oct).** 0.6822 against 0.6864 for balanced
weights (-0.004, CI [-0.039, +0.030], P 0.41). The higher accuracy (0.7509 against 0.7283) came
from favouring majority classes. One square-root seed collapsed (0.1382; the other scored
0.6874). Balanced weights kept.

<a id="b31"></a>**Run 31: prompt ensemble, full data** (prepared; no run recorded).

<a id="b32"></a>**Run 32: holdout check of the four-epoch recipe** (prepared; no run recorded).

<a id="b33"></a>**Run 33: TAPT on the four-epoch recipe** (prepared; no run recorded).

### TAPT and the TAPT bag (Runs 34–38, 14–16 October)

After Task A Run 31 showed TAPT working on Gemma, the Task B TAPT line mirrored it. TAPT runs 2
epochs on Task B, because Task A's LM loss was still falling after one and Task B's corpus is half
the size.

<a id="b34"></a>**Run 34: TAPT holdout twin** (prepared; no run recorded).

<a id="b35"></a>**Run 35: TAPT + R-Drop holdout twin (15 Oct).** TAPT took 134–148 min and peaked
at 8.6–9.9 GB.

| holdout ensemble | macro-F1 | Gender | Geo-pol | Others | Political | Religion | Violence |
|---|---|---|---|---|---|---|---|
| no TAPT, 4 epochs x2 (Run 22) | 0.6872 | 0.785 | 0.667 | 0.569 | 0.822 | 0.797 | 0.483 |
| TAPT x2 | 0.6741 | 0.801 | 0.656 | 0.514 | 0.823 | 0.772 | 0.479 |
| **TAPT + R-Drop x2** | **0.7026** | 0.810 | 0.730 | 0.583 | 0.836 | 0.806 | 0.451 |
| all four | 0.6966 | 0.817 | 0.677 | 0.581 | 0.839 | 0.794 | 0.472 |

R-Drop x2 - TAPT x2: +0.029, CI [-0.004, +0.062], P 0.96. All four - TAPT x2: +0.023, CI
[+0.004, +0.042]. On this TAPT seed, TAPT x2 alone scored below the no-TAPT pair. R-Drop helped
more on Task B than on Task A, and on both tasks adding its models raised the average.

<a id="b36"></a>**Run 36: TAPT + R-Drop full-data fit (15 Oct).** Two 4-epoch seeds with R-Drop 0.5
(6.5 h). **Both models are members of the final ensemble.**

<a id="b37"></a>**Run 37: the TAPT bag, full data (15–16 Oct).** The same design as Task A Run 36:
three sessions, each with its own TAPT seed (1, 2, 3) and two classifiers (seeds 42/43, 44/45,
46/47). 4.4–4.5 h each, no collapses. **Their six models are members of the final ensemble.**

<a id="b38"></a>**Run 38: the TAPT bag, holdout (15 Oct).** TAPT seed 1: pair **0.7056 / 0.7547**,
the best Task B holdout (Gender 0.821, Geo-political 0.716, Others 0.559, Political 0.824,
Religion 0.782, Violence 0.532). Seed 43 collapsed onto `Violence` and was rerun.

Two TAPT runs that differ only in their seed scored 0.6741 (Run 35) and 0.7056 (Run 38). That is
seed-to-seed variance on 530 rows with about 30 rows each in `Geo-political` and `Violence`. Any
single configuration's score is uncertain to about ±3 points, and picking the best one would be
chasing noise. What held in every Task B comparison is that averaging across TAPT runs and recipes
did not fall below the TAPT x2 baseline.

---

## Measurements outside the numbered runs

Measured on the side, mostly on the cheap TF-IDF models, and recorded so they are not repeated:

| idea | result | task |
|---|---|---|
| entity gazetteer / NER, mood and address tags | -0.007 on the SVM; the signature entities fire on the classes already solved (Religion 63%, Political 52%, Geo-political 48% of rows, against 19%, 19% and 18% for Gender, Others and Violence) | B |
| a dedicated Gender-vs-Others model | -0.003; a third of Task B errors sit on that boundary, and the training data has comments annotated both ways | B |
| per-class decode weights | lost the nested check on every run where it was tried | B |
| stopword removal | the 60-word frequency stoplist contains `bjp`, `congress`, `dagar`, `desha` | B |
| stemming | wordpiece already splits `madthare` into stem and suffixes | B |
| auxiliary violent-language head | -0.6 with TAPT, -4.5 on stock MuRIL (Run 5) | B |
| class weighting | not needed: the classes are 3,286 / 3,160 | A |

Ideas that were written down but never built: whole-word masking in MuRIL TAPT, self-distillation
from out-of-fold soft labels, and a supervised contrastive auxiliary loss. One more, an ensemble of
independently TAPT-adapted encoders, was proposed for MuRIL and was later built on Gemma as the
TAPT bag (Task A Run 36, Task B Run 37).

---

## The final systems

Decision rule (15 Oct): the largest equal-weight average of independently trained TAPT models,
including the TAPT + R-Drop models, with nothing tuned on the holdout.

| | Task A | Task B |
|---|---|---|
| model | Gemma-4-12B, 4-bit NF4 QLoRA, LoRA r16 on attention + MLP, last-token head | same |
| TAPT | LoRA next-token training, **1 epoch**, on all labelled comment text + `offenseval_kn` | same, **2 epochs** |
| classifier | **3 epochs**, lr 1e-4 (head 5e-4), effective batch 16, balanced class weights, label smoothing 0.05, short prompt, fp16 on T4 | same, **4 epochs** |
| members | Run 29 (4 TAPT) + Run 35 (2 TAPT + R-Drop) + Run 36 sessions 1–3 (2 each): **12 models, 5 TAPT runs** | Run 36 (2 TAPT + R-Drop) + Run 37 sessions 1–3 (2 each): **8 models, 4 TAPT runs** |
| combination | equal weight per model, argmax | same |
| best holdout evidence | 0.8721 (Run 33), 0.8702 (Run 31) | 0.7056 (Run 38), 0.7026 (Run 35) |
| **official test result** | **macro-F1 0.825049, accuracy 0.878412** | **macro-F1 0.736678, accuracy 0.787879** |

The official results are the organizers' scores for the submitted predictions on the final
portion of each test file, as reported by the authors. The holdout figures describe smaller ensembles on a different split and are not
estimates of them. The plain-Gemma full fits of Task A Run 23 and Task B Runs 21 and 22 wrote
test-prediction ZIPs too, but the submitted predictions are those of the TAPT ensembles above. [`configs/`](../configs/) records both recipes, and `hastika-run` rebuilds them
(see [reproducibility.md](reproducibility.md)).

### What was left, and why it was not pursued

| idea | expected effect | decision |
|---|---|---|
| a second model family (Qwen3-8B, Task A Run 38) | the main untested source of different errors; the size of any gain was never measured | prepared, not run before submission |
| MuRIL-large (Task B Runs 1, 16, 17; Task A Run 15) | more capacity for the MuRIL recipe | cut by Run 1's time budget, fitted without a score (Run 16), out of memory (Run 17); Gemma superseded it |
| the 85%-split holdout models as extra members | more members, each slightly weaker; expected to be inside noise at 12 and 8 models | not needed |
| more bag sessions | averaging gains shrink as members are added | not needed |
| bigger Gemma (26B, 31B) | n/a | does not fit a 15 GB T4 in 4-bit |
| more TAPT epochs or text | extra other-language text hurt MuRIL (Task B Run 5); no comparable in-domain corpus | not pursued |
| learning rate, definitions prompt, LoRA rank, class weights, thresholds, decoding | inside the noise where measured; tuning on the holdout risks overfitting it | closed |
