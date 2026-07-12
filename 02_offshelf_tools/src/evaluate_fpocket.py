"""Evaluate fpocket strategies (top-K pockets) against train labels."""
import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import jaccard_score


def parse_pocket(path):
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21:22]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            ic = line[26:27].strip()
            tag = f"{chain}_{resnum}_{ic}" if ic else f"{chain}_{resnum}"
            if tag not in seen:
                seen.add(tag); out.append(tag)
    return out


def fpocket_predict(fpocket_dir, pid, top_k=1):
    d = os.path.join(fpocket_dir, str(pid), f"{pid}_protein_out", "pockets")
    if not os.path.isdir(d):
        return set()
    out = set()
    for k in range(1, top_k + 1):
        p = os.path.join(d, f"pocket{k}_atm.pdb")
        if not os.path.exists(p):
            break
        for tok in parse_pocket(p):
            out.add(tok)
    return out


def universe(pdb_root, pid):
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
    ap.add_argument("--fpocket_dir", required=True)
    ap.add_argument("--pdb_root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_proteins", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    label_map = {int(r["id"]): set(str(r["resid"]).split()) for _, r in df.iterrows()}
    pids = [p for p in range(args.n_proteins)
            if label_map.get(p) and os.path.exists(os.path.join(args.pdb_root, f"{p}_protein.pdb"))]

    for k in [1, 2, 3, 5]:
        j = flat_jaccard(pids, args.pdb_root, label_map,
                         lambda pid, kk=k: fpocket_predict(args.fpocket_dir, pid, kk))
        print(f"  fpocket top-{k} pockets  Jaccard = {j:.4f}")


if __name__ == "__main__":
    main()
