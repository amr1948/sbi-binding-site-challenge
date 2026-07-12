"""Equivariant Graph Neural Network (Satorras, Hoogeboom, Welling 2021).

Implementation tailored to per-residue binding-site classification: nodes are
residues, coordinates are Cα positions, edges are built from a KNN graph in
3-D. Outputs one logit per node (binding vs. not).

The EGNN message-passing rule:

    m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)
    x_i  <- x_i + sum_j (x_i - x_j) * phi_x(m_ij) / (n_neigh + 1)
    h_i  <- h_i + phi_h(h_i, sum_j m_ij)

is E(n)-equivariant for x and invariant for h. We tag every edge with an
attention-style scalar (sigmoid of an extra MLP head) before summing, which
empirically improves stability when residues have very different neighbour
counts (e.g. surface vs. core).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_edges(coords: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K-nearest-neighbour edges from Cα coordinates.

    coords: (N, 3)
    returns (src, dst) tensors of shape (E,) with E = N*k.
    """
    n = coords.shape[0]
    if n <= 1:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    dist2 = torch.cdist(coords, coords, p=2)  # (N, N)
    dist2.fill_diagonal_(float("inf"))
    kk = min(k, n - 1)
    _, idx = torch.topk(dist2, kk, largest=False)  # (N, kk)
    src = torch.arange(n, device=coords.device).repeat_interleave(kk)
    dst = idx.reshape(-1)
    return src, dst


class EGNNLayer(nn.Module):
    def __init__(self, h_dim: int, hidden: int = 64, act=nn.SiLU):
        super().__init__()
        # edge MLP takes (h_i, h_j, dist^2)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * h_dim + 1, hidden), act(),
            nn.Linear(hidden, hidden), act(),
        )
        # edge attention scalar (sigmoid)
        self.phi_att = nn.Linear(hidden, 1)
        # coord update MLP (scalar per edge)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), act(),
            nn.Linear(hidden, 1),
        )
        # node update MLP (h_i, aggregated msg)
        self.phi_h = nn.Sequential(
            nn.Linear(h_dim + hidden, hidden), act(),
            nn.Linear(hidden, h_dim),
        )

    def forward(self, h, x, src, dst):
        # message
        rel = x[src] - x[dst]                         # (E, 3)
        d2 = (rel * rel).sum(-1, keepdim=True)         # (E, 1)
        m = self.phi_e(torch.cat([h[src], h[dst], d2], dim=-1))  # (E, hidden)
        att = torch.sigmoid(self.phi_att(m))           # (E, 1)
        m = m * att

        # coordinate update: weighted average of relative vectors
        scale = self.phi_x(m)                          # (E, 1)
        coord_msg = rel * scale                        # (E, 3)
        n = h.shape[0]
        dx = torch.zeros_like(x).index_add_(0, src, coord_msg)
        counts = torch.zeros(n, 1, device=x.device).index_add_(
            0, src, torch.ones_like(scale))
        x = x + dx / (counts + 1.0)

        # node update: aggregated messages
        agg = torch.zeros(n, m.shape[1], device=h.device).index_add_(0, src, m)
        h = h + self.phi_h(torch.cat([h, agg], dim=-1))
        return h, x


class EGNNBindingClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 4, k: int = 16):
        super().__init__()
        self.k = k
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden, hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats: torch.Tensor, coords: torch.Tensor):
        src, dst = knn_edges(coords, self.k)
        h = self.in_proj(node_feats)
        x = coords.clone()
        for layer in self.layers:
            h, x = layer(h, x, src, dst)
        return self.head(h).squeeze(-1)  # (N,) logits
