"""Parse P2Rank `*_predictions.csv` files into a submission CSV.

For each protein we take the residues of pocket 1 (highest-ranked). The
residue ids in P2Rank output already use the `<chain>_<resnum>` format we
need, so no further conversion is required.
"""
import argparse
import os
import re

import pandas as pd


def parse_one(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.empty:
        return ""
    top = df.iloc[0]
    raw = str(top.get("residue_ids", "")).strip()
    return raw  # already space-separated A_xx tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2rank_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_proteins", type=int, default=3064)
    ap.add_argument("--top_pockets", type=int, default=1,
                    help="number of top pockets to merge into the prediction")
    args = ap.parse_args()

    rows = []
    for pid in range(args.n_proteins):
        path = os.path.join(args.p2rank_dir, f"{pid}_protein.pdb_predictions.csv")
        if not os.path.exists(path):
            rows.append({"id": pid, "prediction": ""})
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if df.empty or "residue_ids" not in df.columns:
            rows.append({"id": pid, "prediction": ""})
            continue
        top = df.head(args.top_pockets)
        tokens = []
        seen = set()
        for raw in top["residue_ids"].astype(str):
            for tok in raw.strip().split():
                if tok and tok not in seen:
                    seen.add(tok)
                    tokens.append(tok)
        rows.append({"id": pid, "prediction": " ".join(tokens)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
