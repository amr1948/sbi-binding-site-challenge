"""Spatial post-processing for per-residue binding-site probabilities.
"""
from __future__ import annotations

import numpy as np


def _pairwise_dist(coords):
    sq = (coords * coords).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, dtype=np.float32)


def dilate(coords, prob, t_hard, t_soft, radius, dist=None):
    """Return a boolean positive-mask after hard/soft threshold + dilation."""
    hard = prob >= t_hard
    soft = prob >= t_soft
    sel = hard.copy()
    if not (hard.any() and soft.any()):
        return sel
    if dist is None:
        dist = _pairwise_dist(coords)
    soft_idx = np.where(soft & ~hard)[0]
    hard_idx = np.where(hard)[0]
    for j in soft_idx:
        if (dist[j, hard_idx] < radius).any():
            sel[j] = True
    return sel


def prune_small_components(coords, positive_mask, radius, min_size, dist=None):
    """Drop connected components of `positive_mask` smaller than `min_size`."""
    if min_size <= 1 or positive_mask.sum() == 0:
        return positive_mask.copy()
    idx = np.where(positive_mask)[0]
    n = len(idx)
    if dist is None:
        sub_coords = coords[idx]
        dist = _pairwise_dist(sub_coords)
    else:
        dist = dist[np.ix_(idx, idx)]
    adj = dist < radius
    np.fill_diagonal(adj, False)

    visited = np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = [start]
        while stack:
            u = stack.pop()
            neighbors = np.where(adj[u] & ~visited)[0]
            for v in neighbors:
                visited[v] = True
                component.append(v)
                stack.append(v)
        if len(component) >= min_size:
            keep[component] = True

    out = positive_mask.copy()
    out[idx] = keep
    return out


def apply_strategy(coords, prob, strategy, params, dist=None):
    """strategy in {"threshold", "dilate", "dilate_prune"}; returns bool mask."""
    if strategy == "threshold":
        return prob >= params["t"]
    if strategy == "dilate":
        return dilate(coords, prob, params["t_hard"], params["t_soft"], params["radius"], dist=dist)
    if strategy == "dilate_prune":
        mask = dilate(coords, prob, params["t_hard"], params["t_soft"], params["dilate_radius"], dist=dist)
        return prune_small_components(coords, mask, params["prune_radius"], params["min_size"], dist=dist)
    raise ValueError(strategy)
