# `hastika.research`

The code the experiments in [`notebooks/experiments/`](../../../notebooks/experiments/) ran on.
It is kept as it was used, so a notebook's commands mean what they meant when the run was
recorded. The supported final-system pipeline is the rest of `hastika` (`hastika.models`,
`hastika.runner`, ...), derived from the Gemma trainers here. Run numbers refer to
[docs/experiments.md](../../../docs/experiments.md).

| module | what it does | used by |
|---|---|---|
| `data` | stage organizer files, derive validation-phase inputs, rebuild holdouts | every notebook |
| `common.paths` | repository paths for the research modules | all |
| `common.preprocessing` | text cleaning and duplicate handling (shared with `hastika.common`) | all |
| `common.submission` | validate and zip a `predictions.csv` | MuRIL-era runs |
| `common.gpu_queue` | run shell jobs across the GPUs of a Kaggle session | Gemma runs |
| `common.combine_probs` | equal-weight averaging of saved test probabilities into a ZIP | A36–37, B37–38 |
| `common.ensemble_curve` | holdout macro-F1 for 1..N averaged models, with a paired bootstrap | A37, B38 |
| `models.muril` | MuRIL fine-tuning for Task A: demojize, layer reinit, FGM, EMA, R-Drop, seeds, folds | A6, A8–A10, A12–A22 |
| `models.baseline_svm` | TF-IDF + LinearSVC, five-fold or full fit | A3–A4 and every SVM component |
| `models.ensemble` | out-of-fold blend weight search with a nested estimate | A21, `run_all.sh` |
| `models.muril_embeddings_svm` | RBF SVM on frozen MuRIL embeddings | A11 |
| `models.train_xlmr` | generic encoder fine-tuning (XLM-R and others) | `run_all.sh` |
| `task_a.train`, `task_a.train_demojized` | the first MuRIL trainer | A1, A2 |
| `task_a.combined_holdout`, `task_b.combined_holdout` | the fixed 15% holdouts, fingerprints, paired bootstrap | A17 on, B19 on |
| `task_b.tapt` | masked-LM TAPT for MuRIL | A8, B1–B20 |
| `task_b.corpora` | unlabelled text sources for TAPT and vocabulary counts | `tapt`, `extend_vocab` |
| `task_b.train` | MuRIL fine-tuning for Task B | B1–B20 |
| `task_b.extend_vocab`, `normalize`, `features`, `axes`, `decode`, `context_decode` | Task B grid levers, text transforms, decoding | Task B Runs 1–15 |
| `task_b.fetch_external` | build `offenseval_kn.csv` from OffensEval-Dravidian | data preparation, B5 |
| `task_b.split_holdout` | write the earlier Task B train-only holdout to CSV | standalone |
| `task_b.llm_screen` | unlabelled bits-per-character screen of candidate LLMs | B19 |
| `task_b.llm_classifier` | QLoRA LLM classifier (both tasks) | A23–A38, B19–B38 |
| `task_b.llm_tapt` | LoRA next-token TAPT for decoder LLMs (both tasks) | Gemma runs with a TAPT arm |
| `task_b.package_b20` | rebuild Task B Run 20's ZIPs from its saved output | standalone, after B20 |
| `experiments.task_b.overnight` | the overnight sweep | B1 |
| `experiments.task_b.fullfit` | five-seed full fits of several ideas | B3 |
| `experiments.task_b.grid` | the corpus × vocabulary × auxiliary-head grid | B5 |
| `experiments.task_b.polarity_tagger` | polarity tags from an external-label model | B15 (optional arm) |
| `experiments.task_b.stack_full` | one change at a time on Run 11's recipe | B15 |
| `experiments.task_b/sweep.sh` | an earlier shell version of the Task B sweep | n/a |
| `experiments.task_a/run_all.sh` | Task A sweep: SVM, MuRIL, XLM-R family, MuRIL-large | n/a |
| `experiments.task_a/run_rdrop.sh` | five-fold R-Drop comparison | A6 (alternative) |
| `experiments.task_a/synthetic/` | generated comments and the realism test | A5 |

Run modules from the repository root with `src/` on the path, for example
`PYTHONPATH=src python -m hastika.research.models.muril --help`. They read `data/raw/`,
`data/external/` and `data/derived/` and write to `artifacts/`. Stage the data first with
`python -m hastika.research.data`.
