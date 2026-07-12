"""Local evaluation reproducing the metric used by the leaderboard.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import parse_pdb_atoms
from pdb_source import read_pdb_bytes

_ARCHIVE = None


def _init(archive):
    global _ARCHIVE
    _ARCHIVE = archive


def _ids_for(pid):
    data = read_pdb_bytes(_ARCHIVE, pid)
    if data is None:
        return pid, []
    order, residues = parse_pdb_atoms(data)
    out = []
    for key in order:
        if residues[key]["ca"] is None:
            continue
        chain, resnum, icode = key
        out.append(f"{chain}_{resnum}_{icode}" if icode else f"{chain}_{resnum}")
    return pid, out


def build_target(archive, labels_csv, workers):
    df = pd.read_csv(labels_csv)
    label_map = {int(r["id"]): set(str(r.get("resid", "")).split()) for _, r in df.iterrows()}
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(archive,)) as ex:
        futs = {ex.submit(_ids_for, pid): pid for pid in sorted(label_map)}
        for fut in as_completed(futs):
            pid, ids = fut.result()
            true_set = label_map.get(pid, set())
            for r in ids:
                rows.append((pid, r, 1 if r in true_set else 0))
    return pd.DataFrame(rows, columns=["id", "residue_id", "true"])


def parse_submission(df):
    df = df.copy()
    df["prediction"] = df["prediction"].fillna("")
    out = []
    for _, r in df.iterrows():
        for res in str(r["prediction"]).strip().split():
            out.append((int(r["id"]), res.strip()))
    res = pd.DataFrame(out, columns=["id", "residue_id"])
    res["prediction"] = 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    df_sub = parse_submission(pd.read_csv(args.submission))
    df_target = build_target(args.archive, args.labels, args.workers)
    merged = pd.merge(df_target, df_sub, on=["id", "residue_id"], how="left")
    merged["prediction"] = merged["prediction"].fillna(0).astype(int)
    print(f"Jaccard: {jaccard_score(merged['true'], merged['prediction']):.4f}")


if __name__ == "__main__":
    main()
