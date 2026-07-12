"""Evaluate several P2Rank submission strategies against train labels.

Strategies:
  * top-1, top-2, top-3 pockets (residues from `*_predictions.csv`)
  * per-residue probability above a threshold (from `*_residues.csv`)

Outputs the Jaccard score (matching metric.py's flat formulation).
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score


def collect_residues_p2rank(p2_dir, pid, source="predictions", top_k=1,
                             res_threshold=0.4):
    """Return set of residue tokens for one protein from P2Rank output."""
    if source == "predictions":
        path = os.path.join(p2_dir, f"{pid}_protein.pdb_predictions.csv")
        if not os.path.exists(path):
            return set()
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if df.empty or "residue_ids" not in df.columns:
            return set()
        out = set()
        for raw in df.head(top_k)["residue_ids"].astype(str):
            for tok in raw.strip().split():
                if tok:
                    out.add(tok)
        return out
    else:
        path = os.path.join(p2_dir, f"{pid}_protein.pdb_residues.csv")
        if not os.path.exists(path):
            return set()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df["chain"] = df["chain"].astype(str).str.strip()
        out = set()
        col = "probability" if "probability" in df.columns else "score"
        for r in df[df[col] >= res_threshold].itertuples():
            label = str(r.residue_label).strip()
            # split numeric / insertion-code (e.g. '60A' -> 60 + A)
            m = "".join(c for c in label if c.isdigit() or c == "-")
            ic = label[len(m):]
            if not m:
                continue
            tag = f"{r.chain}_{int(m)}_{ic}" if ic else f"{r.chain}_{int(m)}"
            out.add(tag)
        return out


def universe(pdb_root, pid):
    """Return ordered list of residue tokens with a CA atom."""
    path = os.path.join(pdb_root, f"{pid}_protein.pdb")
    out, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21:22]
            try:
                num = int(line[22:26])
            except ValueError:
                continue
            ic = line[26:27].strip()
            tag = f"{chain}_{num}_{ic}" if ic else f"{chain}_{num}"
            if tag not in seen:
                seen.add(tag); out.append(tag)
    return out


def flat_jaccard(pids, pdb_root, labels, predict_fn):
    y_true, y_pred = [], []
    for pid in pids:
        univ = universe(pdb_root, pid)
        truth = labels.get(int(pid), set())
        preds = predict_fn(int(pid))
        for tok in univ:
            y_true.append(1 if tok in truth else 0)
            y_pred.append(1 if tok in preds else 0)
    return jaccard_score(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2rank_dir", required=True)
    ap.add_argument("--pdb_root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_proteins", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    label_map = {int(r["id"]): set(str(r["resid"]).split()) for _, r in df.iterrows()}
    pids = [p for p in range(args.n_proteins) if (label_map.get(p) and
            os.path.exists(os.path.join(args.pdb_root, f"{p}_protein.pdb")))]

    strategies = [
        ("top-1 pocket", lambda pid: collect_residues_p2rank(args.p2rank_dir, pid, "predictions", 1)),
        ("top-2 pockets", lambda pid: collect_residues_p2rank(args.p2rank_dir, pid, "predictions", 2)),
        ("top-3 pockets", lambda pid: collect_residues_p2rank(args.p2rank_dir, pid, "predictions", 3)),
        ("per-residue prob>=0.30", lambda pid: collect_residues_p2rank(args.p2rank_dir, pid, "residues", res_threshold=0.30)),
        ("per-residue prob>=0.40", lambda pid: collect_residues_p2rank(args.p2rank_dir, pid, "residues", res_threshold=0.40)),
        ("per-residue prob>=0.50", lambda pid: collect_residues_p2rank(args.p2rank_dir, pid, "residues", res_threshold=0.50)),
    ]

    for name, fn in strategies:
        j = flat_jaccard(pids, args.pdb_root, label_map, fn)
        print(f"  {name:30s}  Jaccard = {j:.4f}")


if __name__ == "__main__":
    main()
