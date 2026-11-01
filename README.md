# Project MANAS at HASTIKA / ICON 2026: reproduction

This branch contains only what is needed to rebuild Project MANAS's two submitted systems for the
HASTIKA shared task (hate-speech classification of romanized Kannada–English YouTube comments).
The paper, the full experiment record and the research code are on the
[`main` branch](https://github.com/Aaryan127/ProjectMANAS-HastikaICON2026).

**Authors:** Robin Punnen Alex and Aaryan Ganesh Salunke, Project MANAS, Manipal Institute of
Technology, Manipal Academy of Higher Education.

| Task | System | Official macro-F1 | Accuracy |
|---|---|---:|---:|
| A: Hate / Non-Hate | 12 Gemma-4-12B classifiers from 5 TAPT runs | 0.825049 | 0.878412 |
| B: six hate categories | 8 Gemma-4-12B classifiers from 4 TAPT runs | 0.736678 | 0.787879 |

The official scores were reported by the organizers for the submitted predictions on the final
portion of each test file ([`results/final_scores.json`](results/final_scores.json)).

## Start here

**[`reproduce/README.md`](reproduce/README.md)** is the guide. In short:

1. Get the data under the organizers' terms ([`data/README.md`](data/README.md)); it is not
   redistributed here.
2. Run the smoke test first (`reproduce/kaggle/task_a_0_smoke.ipynb`, about 30 minutes): every
   stage on a small sample, ending in a PASS/FAIL checklist.
3. Run one Kaggle notebook per training session, then the combine notebook, which writes the
   predictions ZIP. On your own GPU: `python reproduce/local/reproduce.py --task a --execute`.
4. To check a rerun against labels, use the holdout recipe, which rescores every group beside
   the original runs.

## Contents

```text
reproduce/         Kaggle notebooks (smoke test, one per session, combine), local scripts, guide
configs/           the submitted recipes, their holdout twins, and the smoke-test recipes
src/hastika/       data preparation, QLoRA classifier, TAPT, runner, ensembling, reproduce driver
data/              where the organizer files go, with their recorded checksums
results/           the official scores
tests/             CPU tests of data handling, ensembling, the runner and the package
```

## Install

Python 3.10–3.12.

```bash
pip install torch                 # a CUDA build that matches your driver
pip install -e '.[llm]'           # transformers 5.16.1, peft 0.20.0, bitsandbytes
python -m unittest discover -s tests
```

## Citation and terms

Credit the data: the HASTIKA corpus paper (Kavatagi and Rachh, *Language Resources and
Evaluation* 59, 2025) and the HASTIKA @ ICON-2026 shared task; [`CITATION.cff`](CITATION.cff) lists
both and cites this software. Dataset, base-model and external-corpus terms belong to their
holders; see [NOTICE.md](NOTICE.md). The authors have not yet chosen a code licence.
