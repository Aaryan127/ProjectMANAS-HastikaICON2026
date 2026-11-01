# Experiment notebooks

One notebook per experiment run: [Task A](task_a/README.md) (Runs 1–38) and [Task B](task_b/README.md)
(Runs 1–38), 72 notebooks in all. Task A Runs 1–5 and 7 and Task B Run 15 were run from scripts and have records
but no notebook. Task A Runs 9 and 10 share a notebook. The bag runs (Task A Run 36, Task B Run 37) have one notebook
per session.

Each notebook opens with its run's status, outcome, final-ensemble role, record and write-up, then the notebook as
it was written for the run. To run one on Kaggle, attach the organizer CSVs and `offenseval_kn.csv` as a private
dataset, turn Internet on, and use Save & Run All. The setup cell clones this repository and runs
`python -m hastika.research.data`, which copies and checks the files and rebuilds the fixed holdouts. The code they
call is in [`src/hastika/research/`](../../src/hastika/research/README.md).

The notebooks match this repository's layout and numbering but have not been re-executed on a GPU since; a rerun is a
new measurement. Recorded outcomes come from the original runs' logs: see each run's record in
[`results/experiments/`](../../results/experiments/), and [docs/experiments.md](../../docs/experiments.md) for the
account of how each result led to the next. The two launchers one level up reproduce the final systems.
