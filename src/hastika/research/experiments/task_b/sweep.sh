#!/usr/bin/env bash
# Task B ablation. Every arm shares folds, seed and preprocessing, so the only
# difference between two lines is the flag named in its tag. Run from repo root.
#
#   bash src/hastika/research/experiments/task_b/sweep.sh                 # useful arms, 5-fold
#   FOLDS=0 bash src/hastika/research/experiments/task_b/sweep.sh         # quick holdout
#   ARMS=all bash src/hastika/research/experiments/task_b/sweep.sh        # include noisy arms
#   PY="python -u" bash src/hastika/research/experiments/task_b/sweep.sh  # Kaggle
#
# Arms are ordered by how much they are expected to move macro-F1, so a run that
# is cut short has still tested the interesting things.
set -euo pipefail
PY=${PY:-"uv run --extra cu128 python"}
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-6}
SEEDS=${SEEDS:-42}
ARMS=${ARMS:-core}
ABUSIVE=Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL
mkdir -p artifacts/logs

run () {  # run <tag> <extra flags...>
  tag=$1; shift
  echo "################ $tag ################"
  $PY -m hastika.research.task_b.train --tag "$tag" --folds "$FOLDS" --epochs "$EPOCHS" \
      --seeds $SEEDS "$@" 2>&1 | tee "artifacts/logs/${tag}.log"
}

run b_base                                     # tuned recipe, fixed EMA, max-len 192

# Warm starts. Both come from external in-domain text with no HASTIKA labels in
# them; Task B's own training file is the only HASTIKA text either one sees.
run b_abusive --model "$ABUSIVE"               # MuRIL already tuned on code-mixed Kannada abuse
if [ -d artifacts/runs/tapt-muril ]; then
  run b_tapt --model artifacts/runs/tapt-muril
else
  echo "skipping b_tapt: run 'python -m hastika.research.task_b.tapt' first"
fi

# The two regularizers inherited from Task A and never measured on Task B. FGM is
# also ~45% of the runtime, so a null result here buys back most of a sweep.
run b_noema  --no-ema
run b_nofgm  --no-fgm
run b_emaold --no-ema-bias-correct             # the pre-fix EMA, to size the bug

if [ "$ARMS" = "all" ]; then
  run b_tags   --tags                          # gazetteer / mood / address tags
  run b_focal  --loss focal --focal-gamma 2.0
  run b_both   --tags --loss focal --focal-gamma 2.0
  run b_len128 --max-len 128                   # the old cap, truncates 2% of test rows
fi

echo
echo "=== OOF macro-F1 by arm (both checkpoints; trust the 'last' line) ==="
grep -H "OOF macro-F1" artifacts/logs/b_*.log || true
