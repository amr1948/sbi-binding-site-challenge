"""Train the EGNN binding-site classifier on a subset of the training set.

One protein per gradient step (the residue counts vary too much for naive
batching); we shuffle protein ids every epoch and keep the best validation
flat-Jaccard checkpoint.
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
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import parse_pdb_ca, NODE_FEAT_DIM
from egnn import EGNNBindingClassifier
from pdb_source import read_pdb_bytes


def load_one(archive, pid):
    data = read_pdb_bytes(archive, pid)
    if data is None:
        return None
    coords, feats, res_ids = parse_pdb_ca(data)
    if len(res_ids) == 0:
        return None
    return coords, feats, res_ids


def build_labels(res_ids, positive_set):
    return np.fromiter((1.0 if r in positive_set else 0.0 for r in res_ids),
                       dtype=np.float32, count=len(res_ids))


def evaluate(model, archive, pids, label_map, threshold):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for pid in pids:
            rec = load_one(archive, pid)
            if rec is None:
                continue
            coords, feats, res_ids = rec
            x = torch.from_numpy(coords); f = torch.from_numpy(feats)
            logits = model(f, x)
            prob = torch.sigmoid(logits).cpu().numpy()
            truth = label_map.get(int(pid), set())
            for r, p in zip(res_ids, prob):
                y_true_all.append(1 if r in truth else 0)
                y_pred_all.append(1 if p >= threshold else 0)
    return jaccard_score(y_true_all, y_pred_all), np.asarray(y_true_all), np.asarray(y_pred_all)


def best_threshold(model, archive, pids, label_map):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for pid in pids:
            rec = load_one(archive, pid)
            if rec is None:
                continue
            coords, feats, res_ids = rec
            x = torch.from_numpy(coords); f = torch.from_numpy(feats)
            p = torch.sigmoid(model(f, x)).cpu().numpy()
            t = label_map.get(int(pid), set())
            for r, q in zip(res_ids, p):
                probs.append(float(q))
                ys.append(1 if r in t else 0)
    probs = np.asarray(probs); ys = np.asarray(ys)
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        j = jaccard_score(ys, (probs >= t).astype(np.int8))
        if j > best_j:
            best_j = float(j); best_t = float(t)
    return best_t, best_j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20250618)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.labels)
    label_map = {int(r["id"]): set(str(r["resid"]).split()) for _, r in df.iterrows()}

    all_pids = np.array(sorted(label_map.keys()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_pids)
    val_pids = all_pids[: args.n_val].tolist()
    train_pids = all_pids[args.n_val : args.n_val + args.n_train].tolist()

    print(f"train proteins: {len(train_pids)}, val proteins: {len(val_pids)}")

    model = EGNNBindingClassifier(
        in_dim=NODE_FEAT_DIM, hidden=args.hidden, n_layers=args.n_layers, k=args.k,
    )
    print("parameters:", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_weight = torch.tensor(args.pos_weight)

    best_state = None
    best_val = -1.0
    best_thr = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t = time.time()
        rng.shuffle(train_pids)
        running_loss = 0.0
        seen = 0
        for i, pid in enumerate(train_pids):
            rec = load_one(args.archive, pid)
            if rec is None:
                continue
            coords, feats, res_ids = rec
            y = build_labels(res_ids, label_map.get(int(pid), set()))
            if y.sum() == 0:
                continue  # skip proteins with no positives in this split
            x = torch.from_numpy(coords); f = torch.from_numpy(feats); yt = torch.from_numpy(y)
            logits = model(f, x)
            loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            opt.zero_grad()
            running_loss += float(loss.item())
            seen += 1
            if (i + 1) % 200 == 0:
                print(f"  ep{epoch} step {i+1}/{len(train_pids)} "
                      f"loss={running_loss / seen:.4f} elapsed={time.time()-epoch_t:.0f}s")
        sched.step()

        thr, val_j = best_threshold(model, args.archive, val_pids, label_map)
        print(f"epoch {epoch} done: loss={running_loss/seen:.4f} "
              f"val_jaccard={val_j:.4f} thr={thr:.3f} "
              f"epoch_time={time.time()-epoch_t:.0f}s")
        if val_j > best_val:
            best_val = val_j
            best_thr = thr
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            print("  >>> new best, saved checkpoint in memory")

    if best_state is None:
        raise RuntimeError("no checkpoint was saved (no positives in any train protein?)")
    payload = {
        "state_dict": best_state,
        "threshold": float(best_thr),
        "val_score": float(best_val),
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "k": args.k,
        "in_dim": NODE_FEAT_DIM,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {args.out}: best val Jaccard = {best_val:.4f} at thr {best_thr:.3f}")


if __name__ == "__main__":
    main()
