"""Score every protein in a PDB archive with the trained EGNN and write a
challenge-format submission CSV.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import parse_pdb_ca, NODE_FEAT_DIM
from egnn import EGNNBindingClassifier
from pdb_source import read_pdb_bytes, list_protein_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold stored in the checkpoint")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        payload = pickle.load(f)

    model = EGNNBindingClassifier(
        in_dim=payload.get("in_dim", NODE_FEAT_DIM),
        hidden=payload["hidden"],
        n_layers=payload["n_layers"],
        k=payload["k"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    threshold = float(args.threshold) if args.threshold is not None else float(payload["threshold"])
    print(f"threshold = {threshold:.3f}")

    pids = list_protein_ids(args.archive)
    print(f"scoring {len(pids)} proteins from {args.archive}")
    t0 = time.time()
    rows = []
    with torch.no_grad():
        for i, pid in enumerate(pids):
            data = read_pdb_bytes(args.archive, pid)
            if data is None:
                rows.append({"id": pid, "prediction": ""}); continue
            coords, feats, res_ids = parse_pdb_ca(data)
            if not res_ids:
                rows.append({"id": pid, "prediction": ""}); continue
            x = torch.from_numpy(coords); f = torch.from_numpy(feats)
            prob = torch.sigmoid(model(f, x)).cpu().numpy()
            selected = [res_ids[j] for j, q in enumerate(prob) if q >= threshold]
            rows.append({"id": pid, "prediction": " ".join(selected)})
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(pids)} ({rate:.1f}/s)", flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
