"""Predict binding residues for all proteins in a PDB archive.

"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import featurize_pdb_bytes, parse_pdb_atoms, NUM_FEATURES
from pdb_source import read_pdb_bytes, list_protein_ids
from postprocess import apply_strategy

_ARCHIVE = None


def _init(archive):
    global _ARCHIVE
    _ARCHIVE = archive


def _feat_one(pid):
    data = read_pdb_bytes(_ARCHIVE, pid)
    if data is None:
        return pid, [], np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    order, residues = parse_pdb_atoms(data)
    res_ids, X = featurize_pdb_bytes(data)
    keys = [k for k in order if residues[k]["ca"] is not None]
    coords = np.array([residues[k]["ca"] for k in keys], dtype=np.float32)
    return pid, res_ids, X.astype(np.float32), coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold stored in the model file")
    ap.add_argument("--postprocess", default=None,
                    help="path to a postprocess_choice.pkl from tune_postprocess.py; "
                         "if given, applies the winning strategy instead of a plain threshold")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        payload = pickle.load(f)
    clf = payload["model"]
    threshold = args.threshold if args.threshold is not None else float(payload.get("threshold", 0.5))

    strategy, pp_params = "threshold", {"t": threshold}
    if args.postprocess:
        with open(args.postprocess, "rb") as f:
            choice = pickle.load(f)
        strategy, pp_params = choice["winner_strategy"], choice["winner_params"]
        print(f"using tuned post-processing: {strategy} {pp_params} "
              f"(val flat Jaccard {choice['winner_score']:.4f} vs baseline {choice['baseline_score']:.4f})")
    else:
        print(f"plain threshold = {threshold:.3f}")

    pids = list_protein_ids(args.archive)
    print(f"scoring {len(pids)} proteins from {args.archive}")
    t0 = time.time()
    out_by_pid = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.archive,)) as ex:
        futs = {ex.submit(_feat_one, p): p for p in pids}
        done = 0
        for fut in as_completed(futs):
            pid, res_ids, X, coords = fut.result()
            if not res_ids:
                out_by_pid[pid] = ""
            else:
                prob = clf.predict_proba(X)[:, 1]
                mask = apply_strategy(coords, prob, strategy, pp_params)
                selected = [res_ids[i] for i in range(len(res_ids)) if mask[i]]
                out_by_pid[pid] = " ".join(selected)
            done += 1
            if done % 200 == 0 or done == len(pids):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(pids)} ({rate:.1f}/s)", flush=True)

    rows = [{"id": pid, "prediction": out_by_pid.get(pid, "")} for pid in pids]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
