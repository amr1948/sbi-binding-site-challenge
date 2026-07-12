#!/bin/bash
set -e
cd /Users/amrabdullah/AI_2026ss/structural_bioinformatics/challenge

P2RANK=approach2/tools/p2rank_2.4.2/prank
TRAIN_OUT=approach2/output/p2rank_train_eval
mkdir -p "$TRAIN_OUT"

# Run P2Rank on train subset
if [ ! -f "$TRAIN_OUT/0_protein.pdb_predictions.csv" ]; then
  $P2RANK predict approach2/output/train_subset.ds -o "$TRAIN_OUT" -threads 8 -visualizations 0 2>&1 | tail -5
fi

# Evaluate strategies
echo "=== strategy evaluation on 200 train proteins ==="
/opt/anaconda3/bin/python approach2/code/evaluate_strategies.py \
  --p2rank_dir "$TRAIN_OUT" \
  --pdb_root approach2/data/train_pdb_subset \
  --labels train.csv \
  --n_proteins 200

# Generate three submission variants on test
P2RANK_FULL=approach2/output/p2rank_full
for k in 1 2 3; do
  /opt/anaconda3/bin/python approach2/code/parse_p2rank.py \
    --p2rank_dir "$P2RANK_FULL" \
    --out approach2/output/submission_p2rank_top${k}.csv \
    --top_pockets $k
done

echo "done"
ls -la approach2/output/submission_*.csv
