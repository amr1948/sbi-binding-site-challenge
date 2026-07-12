# Protein binding-pocket identification — challenge submission

Predict which residues of a protein form a ligand-binding pocket, from its PDB
structure. Scored by a **flat (micro-averaged) Jaccard / IoU** over all
`(protein, residue)` pairs — see [`metric.py`](metric.py).

## Layout

```
challenge/
├── data/                     provided inputs (train/, test/, train.csv, sample_submission.csv)
├── metric.py                 provided evaluation script
│
├── 01_gbm_handcrafted/   ★   GRADED approach — gradient boosting on hand-crafted features
│   ├── binding_site_gbm.ipynb      self-contained, runnable end-to-end notebook
│   ├── src/                         same logic as parallelised CLI scripts
│   ├── model/                       fitted model + chosen post-processing
│   ├── submission_v1_threshold.csv       public LB 0.326 (threshold only)
│   └── submission_v2_postprocessed.csv   + dilation/pruning (local 0.371)
│
├── 02_offshelf_tools/        P2Rank + Fpocket survey (public LB 0.268)
│   ├── binding_site_methods.ipynb
│   ├── src/ · submission.csv
│
└── 03_egnn/                  E(n)-equivariant GNN, from scratch (public LB 0.163)
    ├── egnn_binding_site.ipynb
    ├── src/ · model/ · submission.csv
```

Heavy regenerable artifacts (feature caches, per-protein tool dumps, the
downloaded P2Rank binary) live in gitignored `cache/` and `_generated/`
subfolders and are **not** part of the source submission.

## The three approaches

| # | approach | model | pre-trained? | public LB Jaccard |
|---|----------|-------|--------------|-------------------|
| **1** | **hand-crafted features + gradient boosting** | `HistGradientBoostingClassifier`, 66 per-residue features, spatial post-processing | no | **0.326** |
| 2 | off-the-shelf tools | P2Rank (top-2 pockets) + Fpocket survey | yes (P2Rank) | 0.268 |
| 3 | equivariant GNN | 5-layer EGNN over a Cα k-NN graph, from scratch | no | 0.163 |

**Approach 1 is the graded submission.** It wins at this compute budget and is
the one to run first — see [`01_gbm_handcrafted/README.md`](01_gbm_handcrafted/README.md).

## Running

All code expects the full data-science stack (numpy, pandas, scikit-learn;
PyTorch for approach 3). On this machine use a conda interpreter, e.g.
`/opt/anaconda3/bin/python3` — the bare `python3` on `PATH` has none of these.
Launch notebooks from the challenge root so the relative `data/` paths resolve.

`--archive` arguments accept **either** a `.zip` file **or** a plain directory
of `<id>_protein.pdb` files, so everything runs against `data/train` and
`data/test` as-is (the original `train.zip`/`test.zip` are no longer needed).
