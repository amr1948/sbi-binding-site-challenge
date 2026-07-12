
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import parse_pdb_atoms
from pdb_source import read_pdb_bytes
from postprocess import apply_strategy, _pairwise_dist


def reconstruct_val_pids(pid_arr, seed, val_frac):
    all_pids = np.unique(pid_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_pids)
    n_val = max(1, int(len(all_pids) * val_frac))
    return set(int(x) for x in all_pids[:n_val])


def group_by_pid(pid_arr, resid_arr, prob_arr):
    order = {}
    for p, r, q in zip(pid_arr, resid_arr, prob_arr):
        p = int(p)
        order.setdefault(p, ([], []))
        order[p][0].append(r)
        order[p][1].append(q)
    return {p: (np.asarray(r), np.asarray(q, dtype=np.float32)) for p, (r, q) in order.items()}


def get_coords(archive, pid, expected_res_ids):
    data = read_pdb_bytes(archive, pid)
    order, residues = parse_pdb_atoms(data)
    keys = [k for k in order if residues[k]["ca"] is not None]
    res_ids = []
    for (chain, resnum, icode) in keys:
        res_ids.append(f"{chain}_{resnum}_{icode}" if icode else f"{chain}_{resnum}")
    coord_map = {rid: residues[k]["ca"] for rid, k in zip(res_ids, keys)}
    coords = np.array([coord_map[r] for r in expected_res_ids], dtype=np.float32)
    return coords


def flat_jaccard_for(y_true_parts, y_pred_parts):
    return jaccard_score(np.concatenate(y_true_parts), np.concatenate(y_pred_parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="01_gbm_handcrafted/model/model.pkl")
    ap.add_argument("--features", default="01_gbm_handcrafted/cache/train_features.npz")
    ap.add_argument("--labels", default="data/train.csv")
    ap.add_argument("--archive", default="data/train")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20250618)
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        payload = pickle.load(f)
    clf = payload["model"]
    base_threshold = float(payload["threshold"])
    print(f"loaded model, stored threshold={base_threshold:.3f}, stored val_score={payload['val_score']:.4f}")

    data = np.load(args.features, allow_pickle=True)
    pid_arr, resid_arr, X = data["pid"], data["resid"].astype(str), data["X"]

    val_pids = reconstruct_val_pids(pid_arr, args.seed, args.val_frac)
    print(f"reconstructed val split: {len(val_pids)} proteins")

    mask_val = np.fromiter((int(p) in val_pids for p in pid_arr), dtype=bool, count=len(pid_arr))
    pid_v, resid_v, X_v = pid_arr[mask_val], resid_arr[mask_val], X[mask_val]

    t0 = time.time()
    prob_v = clf.predict_proba(X_v)[:, 1]
    print(f"scored {len(prob_v)} val rows in {time.time()-t0:.1f}s")

    by_pid = group_by_pid(pid_v, resid_v, prob_v)

    df_labels = pd.read_csv(args.labels)
    label_map = {int(r["id"]): set(str(r.get("resid", "")).split()) for _, r in df_labels.iterrows()}

    print("fetching CA coordinates for val proteins...")
    t0 = time.time()
    coords_by_pid = {}
    for pid in by_pid:
        res_ids, _ = by_pid[pid]
        coords_by_pid[pid] = get_coords(args.archive, pid, res_ids)
    print(f"done in {time.time()-t0:.1f}s")

    print("precomputing pairwise distance matrices (reused across the whole sweep)...")
    t0 = time.time()
    dist_by_pid = {pid: _pairwise_dist(coords_by_pid[pid]) for pid in by_pid}
    print(f"done in {time.time()-t0:.1f}s")

    # precompute y_true once per protein too
    y_true_by_pid = {}
    for pid, (res_ids, _) in by_pid.items():
        true_set = label_map.get(pid, set())
        y_true_by_pid[pid] = np.fromiter((1 if r in true_set else 0 for r in res_ids), dtype=np.int8, count=len(res_ids))

    def score(strategy, params):
        y_true_parts, y_pred_parts = [], []
        for pid, (res_ids, prob) in by_pid.items():
            mask = apply_strategy(coords_by_pid[pid], prob, strategy, params, dist=dist_by_pid[pid])
            y_true_parts.append(y_true_by_pid[pid])
            y_pred_parts.append(mask.astype(np.int8))
        return flat_jaccard_for(y_true_parts, y_pred_parts)

    results = []

    baseline = score("threshold", {"t": base_threshold})
    results.append(("baseline threshold", {"t": base_threshold}, baseline))
    print(f"baseline (t={base_threshold:.3f}):            flat Jaccard = {baseline:.4f}")

    print("\n== dilation sweep ==")
    best_dil = (-1, None)
    for t_hard in [0.6, 0.65, 0.7, base_threshold, 0.75, 0.8]:
        for t_soft in [0.3, 0.4, 0.5, 0.6]:
            if t_soft >= t_hard:
                continue
            for radius in [6.0, 8.0, 10.0]:
                s = score("dilate", {"t_hard": t_hard, "t_soft": t_soft, "radius": radius})
                if s > best_dil[0]:
                    best_dil = (s, {"t_hard": t_hard, "t_soft": t_soft, "radius": radius})
    print(f"best dilation: flat Jaccard = {best_dil[0]:.4f}  params={best_dil[1]}")
    results.append(("dilate", best_dil[1], best_dil[0]))

    print("\n== pruning on top of best dilation ==")
    best_prune = (-1, None)
    for prune_radius in [6.0, 8.0, 10.0]:
        for min_size in [1, 2, 3, 4, 5, 6, 8]:
            params = dict(best_dil[1])
            params = {"dilate_radius": params["radius"], "t_hard": params["t_hard"],
                      "t_soft": params["t_soft"], "prune_radius": prune_radius, "min_size": min_size}
            s = score("dilate_prune", params)
            if s > best_prune[0]:
                best_prune = (s, params)
    print(f"best dilate+prune: flat Jaccard = {best_prune[0]:.4f}  params={best_prune[1]}")
    results.append(("dilate_prune", best_prune[1], best_prune[0]))

    print("\n== pruning alone (no dilation) on top of plain threshold ==")
    best_prune_only = (-1, None)
    for prune_radius in [6.0, 8.0, 10.0]:
        for min_size in [1, 2, 3, 4, 5, 6, 8]:
            params = {"dilate_radius": 0.0, "t_hard": base_threshold, "t_soft": base_threshold,
                      "prune_radius": prune_radius, "min_size": min_size}
            s = score("dilate_prune", params)
            if s > best_prune_only[0]:
                best_prune_only = (s, params)
    print(f"best prune-only: flat Jaccard = {best_prune_only[0]:.4f}  params={best_prune_only[1]}")
    results.append(("prune_only", best_prune_only[1], best_prune_only[0]))

    print("\n=== summary ===")
    for name, params, s in results:
        print(f"  {name:20s} flat Jaccard = {s:.4f}  {params}")

    best_overall = max(results, key=lambda r: r[2])
    print(f"\nWINNER: {best_overall[0]} -> {best_overall[2]:.4f}  {best_overall[1]}")

    out = {"winner_strategy": best_overall[0], "winner_params": best_overall[1],
           "winner_score": best_overall[2], "baseline_score": baseline, "all_results": results}
    out_path = "01_gbm_handcrafted/model/postprocess_choice.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
