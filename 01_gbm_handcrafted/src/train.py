"""Train a binding-residue classifier from a PDB archive.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import jaccard_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import featurize_pdb_bytes, NUM_FEATURES, FEATURE_COLS
from pdb_source import read_pdb_bytes

_ARCHIVE = None


def _init(archive):
    global _ARCHIVE
    _ARCHIVE = archive


def _feat_one(pid):
    data = read_pdb_bytes(_ARCHIVE, pid)
    if data is None:
        return pid, [], np.zeros((0, NUM_FEATURES), dtype=np.float32)
    res_ids, X = featurize_pdb_bytes(data)
    return pid, res_ids, X.astype(np.float32)


def extract_all(archive, pids, workers):
    pid_arr, rid_arr, X_arr = [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(archive,)) as ex:
        futs = {ex.submit(_feat_one, p): p for p in pids}
        done = 0
        for fut in as_completed(futs):
            pid, res_ids, X = fut.result()
            if res_ids:
                pid_arr.append(np.full(len(res_ids), pid, dtype=np.int32))
                rid_arr.append(np.asarray(res_ids, dtype=object))
                X_arr.append(X)
            done += 1
            if done % 500 == 0 or done == len(pids):
                rate = done / (time.time() - t0)
                print(f"  features {done}/{len(pids)} ({rate:.1f}/s)", flush=True)
    return np.concatenate(pid_arr), np.concatenate(rid_arr), np.concatenate(X_arr, axis=0)


def make_labels(pid_arr, rid_arr, label_csv):
    df = pd.read_csv(label_csv)
    positive = {int(r["id"]): set(str(r.get("resid", "")).split()) for _, r in df.iterrows()}
    y = np.zeros(len(pid_arr), dtype=np.int8)
    for i, (pid, rid) in enumerate(zip(pid_arr, rid_arr)):
        s = positive.get(int(pid))
        if s is not None and rid in s:
            y[i] = 1
    return y, positive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None,
                    help="optional npz cache for features (created on first run, reused thereafter)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_iter", type=int, default=800)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_leaf_nodes", type=int, default=63)
    ap.add_argument("--min_samples_leaf", type=int, default=80)
    ap.add_argument("--l2_regularization", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20250618)
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        print(f"loading cached features from {args.cache}")
        data = np.load(args.cache, allow_pickle=True)
        pid = data["pid"]; rid = data["resid"].astype(str); X = data["X"]
    else:
        df_labels = pd.read_csv(args.labels)
        pids = sorted(set(int(x) for x in df_labels["id"].tolist()))
        print(f"extracting features for {len(pids)} proteins")
        pid, rid, X = extract_all(args.archive, pids, args.workers)
        if args.cache:
            np.savez_compressed(args.cache, pid=pid, resid=rid, X=X,
                                feature_cols=np.array(FEATURE_COLS))
            print(f"cached features to {args.cache}")

    print(f"feature matrix: {X.shape}, unique proteins: {len(np.unique(pid))}")
    y, pos_lookup = make_labels(pid, rid, args.labels)
    print(f"positive rate: {y.mean():.4f} ({int(y.sum())}/{len(y)})")

    rng = np.random.default_rng(args.seed)
    all_pids = np.unique(pid)
    rng.shuffle(all_pids)
    n_val = max(1, int(len(all_pids) * args.val_frac))
    val_pids = set(all_pids[:n_val].tolist())
    mask_val = np.fromiter((int(x) in val_pids for x in pid), dtype=bool, count=len(pid))
    mask_train = ~mask_val

    Xt, yt = X[mask_train], y[mask_train]
    Xv, yv = X[mask_val], y[mask_val]
    print(f"train rows: {len(yt):,}, val rows: {len(yv):,}")

    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=args.seed,
        class_weight="balanced",
    )
    clf.fit(Xt, yt)
    print(f"trained in {time.time() - t0:.0f}s, iterations={clf.n_iter_}")

    pv = clf.predict_proba(Xv)[:, 1]
    print(f"val AUC : {roc_auc_score(yv, pv):.4f}")
    print(f"val AP  : {average_precision_score(yv, pv):.4f}")

    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        j = jaccard_score(yv, (pv >= t).astype(np.int8))
        if j > best_j:
            best_j = float(j); best_t = float(t)
    print(f"val Jaccard (flat): {best_j:.4f} at threshold {best_t:.3f}")

    payload = {
        "model": clf,
        "threshold": best_t,
        "val_score": best_j,
        "feature_cols": FEATURE_COLS,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
