# Approach 3 — E(n) Equivariant Graph Neural Network

Per-residue binding-site classifier on an EGNN (Satorras, Hoogeboom & Welling,
ICML 2021). Each protein is a graph: residues are nodes, edges come from a
k-nearest-neighbour graph in 3-D, node features are chemistry only (one-hot AA
+ 7 physicochemical descriptors). The EGNN learns the geometric signal itself,
so no geometry is hand-rolled the way Approach 1 does.

**Reference approach** (the graded submission is Approach 1). Public LB
Jaccard 0.163 — under-trained: only 3 000 of 16 379 proteins for 6 CPU epochs,
with validation Jaccard still climbing at cutoff.

## Layout

```
03_egnn/
├── egnn_binding_site.ipynb     end-to-end notebook
├── src/
│   ├── egnn.py                   EGNN layer + classifier
│   ├── data.py                   PDB → (coords, chemistry features)
│   ├── pdb_source.py             read PDB bytes from a directory OR a .zip
│   ├── train_egnn.py             training loop (one protein per step)
│   └── predict_egnn.py           writes the submission CSV
├── model/
│   ├── egnn_model.pkl            trained checkpoint + threshold
│   └── train.log                 training stdout (gitignored)
└── submission.csv               final predictions for the test set
```

## Architecture

| component | value |
| --- | --- |
| node features | 21 (AA one-hot) + 7 (chemistry flags) = 28 |
| EGNN layers / hidden dim / k | 5 / 96 / 16 |
| classifier head | LayerNorm → Linear → SiLU → Linear → 1 |
| parameters | ≈ 340 k |
| loss | BCE-with-logits, `pos_weight = 8` |
| optimiser | Adam, lr 1e-3, cosine schedule |

## Training set-up

3 000 training / 300 validation proteins (disjoint, fixed seed), 6 epochs, one
protein per gradient step, best checkpoint by held-out flat Jaccard. A full
epoch runs in ≈ 10 min on CPU. The other 13 000 train proteins are unused;
scaling up would help but takes a multi-hour run.

## Reproducing

Run from the challenge root. `--archive` accepts a directory of
`<id>_protein.pdb` files (e.g. `data/train`) or a `.zip`.

```bash
PY=/opt/anaconda3/envs/mlpc2026/bin/python3    # needs torch (MPS available here)

# 1. train (writes model/egnn_model.pkl)
$PY 03_egnn/src/train_egnn.py \
    --archive data/train --labels data/train.csv \
    --out 03_egnn/model/egnn_model.pkl \
    --n_train 3000 --n_val 300 --epochs 6 \
    --hidden 96 --n_layers 5 --k 16 --lr 1e-3 --pos_weight 8.0

# 2. score the test set
$PY 03_egnn/src/predict_egnn.py \
    --archive data/test --model 03_egnn/model/egnn_model.pkl \
    --out 03_egnn/submission.csv
```

The notebook `egnn_binding_site.ipynb` chains the same steps end-to-end with
sanity checks against `metric.py`.

## Code sources

Original from-scratch implementation of the architecture in Satorras,
Hoogeboom & Welling, *E(n) Equivariant Graph Neural Networks*, ICML 2021.
PyTorch and scikit-learn (`jaccard_score`) are the only libraries.
