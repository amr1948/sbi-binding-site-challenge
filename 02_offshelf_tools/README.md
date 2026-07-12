# Approach 2 — survey of off-the-shelf binding-site predictors

A single Jupyter notebook that runs several established binding-site predictors
and turns each output into the submission format expected by `metric.py`.

**Reference approach** (the graded submission is Approach 1). Public LB Jaccard
0.268 with P2Rank top-2 pockets.

## Layout

```
02_offshelf_tools/
├── binding_site_methods.ipynb   end-to-end notebook
├── src/                          helper scripts + P2Rank/Fpocket submission variants
│   ├── run_fpocket.py  parse_p2rank.py  parse_fpocket.py
│   ├── evaluate_strategies.py  evaluate_fpocket.py  finalize.sh
│   └── submission_{p2rank,fpocket}_top*.csv   (per-strategy outputs)
├── submission.csv               final file (P2Rank top-2, pre-trained)
├── _generated/                  gitignored, regenerable:
│   ├── tools/p2rank_2.4.2/        downloaded P2Rank binary + bundled models
│   ├── data/train_pdb_subset/     200-protein local-scoring subset
│   ├── p2rank_full/  p2rank_train_eval/    one *_predictions.csv per protein
│   └── fpocket_full/ fpocket_train_eval/   one subdir per protein
└── README.md
```

## Methods covered

| method   | runs locally | pre-trained | per-residue | used for submission |
| -------- | ------------ | ----------- | ----------- | ------------------- |
| P2Rank   | yes (Java)   | yes (`default.model`) | yes | **yes** |
| Fpocket  | yes          | n/a (geometric)       | indirect | benchmarked |
| DeepSite | web API only | yes (server-side)     | indirect | reference cell |
| DeepSurf | yes (GPU)    | yes (`resnet_18.h5`)  | yes | skeleton |
| 3D U-Net | training req.| no                    | yes | skeleton |

The notebook downloads P2Rank (~350 MB) into `_generated/tools/`, expects
fpocket on `PATH` (`conda install -c conda-forge fpocket`), and runs both over
the test PDBs in `data/test`. DeepSite (PlayMolecule REST queue), DeepSurf
(separate TensorFlow/GPU env), and the 3D voxel U-Net (needs training) are kept
as reference/skeleton cells only — none produces the submission.

## Reproducing the final submission

Run from the challenge root (needs Java + fpocket; both present on this machine
— Java 11, and the P2Rank binary is preserved under `_generated/tools/`):

```bash
/opt/anaconda3/bin/python3 -m nbconvert --to notebook --execute \
    02_offshelf_tools/binding_site_methods.ipynb --output-dir /tmp
```

That runs the P2Rank section end-to-end and writes
`02_offshelf_tools/submission.csv`. Expect ~25 minutes wall-clock on an
8-core CPU.

## Code sources

P2Rank (Krivák & Hoksza, *J. Cheminform.* 2018) and Fpocket (Le Guilloux et
al., *BMC Bioinformatics* 2009) are third-party tools invoked as subprocesses;
`default.model` is P2Rank's bundled pre-trained random forest. The glue code
(parsing pockets into the submission format, local scoring) is original.
