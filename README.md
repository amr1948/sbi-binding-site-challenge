# Protein binding-pocket identification

Predict which residues of a protein form a ligand-binding pocket, from its PDB
structure. Scored by a flat (micro-averaged) Jaccard / IoU** over all
`(protein, residue)` pairs. see [`metric.py`](metric.py).


Heavy regenerable artifacts (feature caches, per-protein tool dumps, the
downloaded P2Rank binary) live in gitignored `cache/` and `_generated/`

## approaches

| # | approach | model | pre-trained? | public LB Jaccard |
|---|----------|-------|--------------|-------------------|
| 1 | hand-crafted features + gradient boosting | `HistGradientBoostingClassifier`, 66 per-residue features, spatial post-processing | no | 0.326 |
| 2 | off-the-shelf tools | P2Rank (top-2 pockets) + Fpocket survey | yes (P2Rank) | 0.268 |
| 3 | equivariant GNN | 5-layer EGNN over a Cα k-NN graph, from scratch | no | 0.163 |

