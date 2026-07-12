"""Per-residue feature extraction from PDB files.
"""
from __future__ import annotations

import numpy as np

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HID": "H", "HIE": "H",
    "HIP": "H", "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E",
    "LYN": "K",
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}

KD = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8, "G": -0.4,
    "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8, "M":  1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V":  4.2,
    "W": -0.9, "Y": -1.3,
}
VOL = {
    "A":  88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G":  60.1,
    "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
    "P": 112.7, "Q": 143.8, "R": 173.4, "S":  89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}
CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 0.1, "I": 0,
    "K": 1, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1, "S": 0,
    "T": 0, "V": 0, "W": 0, "Y": 0,
}
POLAR = set("STNQHYCDEKR")
AROMATIC = set("FWYH")
ALIPHATIC = set("AVLIMG")
HBOND_DONOR = set("RKQNHWSTY")
HBOND_ACCEPTOR = set("DENQHSTY")

PROP_NAMES = ["kd", "vol", "charge", "polar", "aromatic", "aliphatic", "donor", "acceptor", "glycine"]
NUM_PROPS = len(PROP_NAMES)

GEOM_FEATS = [
    "n6", "n8", "n10", "n12", "n15",
    "d_centroid", "d_centroid_norm",
    "depth_proxy",
    "mean_nn_dist", "std_nn_dist",
    "hse_up", "hse_down", "hse_ratio",
    "ca_z_norm",
    "bfactor_mean", "bfactor_std",
    "side_chain_extent",
    "pocket_open_space", "pocket_concavity",
    "sa_proxy",
    "anisotropy",
    "rel_burial",
]

NEIGH_AGG_PROPS = ["kd", "vol", "charge", "polar", "aromatic", "donor", "acceptor"]
NEIGH_RADII = [8.0, 12.0]
NEIGH_FEATS = [f"n8_mean_{p}" for p in NEIGH_AGG_PROPS] + [f"n12_mean_{p}" for p in NEIGH_AGG_PROPS]

FEATURE_COLS = (
    [f"aa_{a}" for a in AA_ORDER]
    + ["aa_unknown"]
    + [f"prop_{n}" for n in PROP_NAMES]
    + GEOM_FEATS
    + NEIGH_FEATS
)
NUM_FEATURES = len(FEATURE_COLS)


def parse_pdb_atoms(pdb_bytes):
    residues = {}
    order = []
    if isinstance(pdb_bytes, (bytes, bytearray)):
        text = pdb_bytes.decode("ascii", errors="ignore")
    else:
        text = pdb_bytes
    for line in text.splitlines():
        if len(line) < 54 or line[:6] != "ATOM  ":
            continue
        atom_name = line[12:16].strip()
        alt_loc = line[16:17]
        if alt_loc not in (" ", "A", ""):
            continue
        resname = line[17:20].strip()
        chain = line[21:22]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26:27]
        if icode == " ":
            icode = ""
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        try:
            bfac = float(line[60:66])
        except (ValueError, IndexError):
            bfac = 0.0
        is_h = atom_name.startswith("H") or (
            len(atom_name) > 1 and atom_name[1] == "H" and atom_name[0].isdigit()
        )
        key = (chain, resnum, icode)
        if key not in residues:
            residues[key] = {"resname": resname, "atoms": [], "ca": None, "cb": None, "sc": []}
            order.append(key)
        rec_d = residues[key]
        rec_d["atoms"].append((atom_name, x, y, z, bfac, is_h))
        if atom_name == "CA":
            rec_d["ca"] = (x, y, z)
        elif atom_name == "CB":
            rec_d["cb"] = (x, y, z)
        if atom_name not in {"N", "CA", "C", "O", "OXT"} and not is_h:
            rec_d["sc"].append((x, y, z))
    return order, residues


def compute_features(order, residues):
    keys = [k for k in order if residues[k]["ca"] is not None]
    n = len(keys)
    if n == 0:
        return [], np.zeros((0, NUM_FEATURES), dtype=np.float32)

    ca = np.array([residues[k]["ca"] for k in keys], dtype=np.float32)
    cb = np.array(
        [residues[k]["cb"] if residues[k]["cb"] is not None else residues[k]["ca"] for k in keys],
        dtype=np.float32,
    )
    side_dir = cb - ca
    side_dir = side_dir / (np.linalg.norm(side_dir, axis=1, keepdims=True) + 1e-6)

    centroid = ca.mean(axis=0)
    max_extent = np.linalg.norm(ca - centroid, axis=1).max() + 1e-6

    sq = (ca * ca).sum(axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (ca @ ca.T)
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2).astype(np.float32)
    np.fill_diagonal(dist, np.inf)

    n6 = (dist < 6.0).sum(axis=1)
    n8 = (dist < 8.0).sum(axis=1)
    n10 = (dist < 10.0).sum(axis=1)
    n12 = (dist < 12.0).sum(axis=1)
    n15 = (dist < 15.0).sum(axis=1)

    k = min(10, n - 1) if n > 1 else 1
    if k > 0:
        nn_idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
        nn_dists = np.take_along_axis(dist, nn_idx, axis=1)
        mean_nn = nn_dists.mean(axis=1)
        std_nn = nn_dists.std(axis=1)
    else:
        mean_nn = np.zeros(n, dtype=np.float32)
        std_nn = np.zeros(n, dtype=np.float32)

    mask13 = dist < 13.0
    mask10 = dist < 10.0
    up = np.zeros(n, dtype=np.float32)
    down = np.zeros(n, dtype=np.float32)
    co_plane = np.zeros(n, dtype=np.float32)
    same_side = np.zeros(n, dtype=np.float32)
    opp_side = np.zeros(n, dtype=np.float32)
    mean_dir = np.zeros((n, 3), dtype=np.float32)

    # Stream the per-residue projections in row chunks so peak memory stays
    # bounded even for very large proteins (n > 3000).
    chunk = 256
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rel_c = (ca[None, s:e, :] - ca[:, None, :]).transpose(1, 0, 2)
        proj_c = (rel_c * side_dir[s:e, None, :]).sum(axis=2)
        m13 = mask13[s:e]
        up[s:e] = ((proj_c > 0.0) & m13).sum(axis=1).astype(np.float32)
        down[s:e] = ((proj_c < 0.0) & m13).sum(axis=1).astype(np.float32)
        m10 = mask10[s:e]
        co_plane[s:e] = ((np.abs(proj_c) < 4.0) & m10).sum(axis=1).astype(np.float32)
        same_side[s:e] = ((proj_c > 1.0) & m10).sum(axis=1).astype(np.float32)
        opp_side[s:e] = ((proj_c < -1.0) & m10).sum(axis=1).astype(np.float32)
        dist_c = dist[s:e]
        safe = np.where(m10, dist_c, 1.0).astype(np.float32)
        unit_c = (rel_c / safe[..., None]) * m10[..., None]
        mean_dir[s:e] = unit_c.sum(axis=1)

    hse_ratio = (up - down) / (up + down + 1.0)
    pocket_open = (same_side - opp_side) / (same_side + opp_side + 1.0)
    pocket_concav = co_plane / (mask10.sum(axis=1).astype(np.float32) + 1.0)

    surface_thresh = np.median(n12)
    surface_mask = n12 <= surface_thresh
    if surface_mask.any():
        depth_proxy = dist[:, surface_mask].min(axis=1)
    else:
        depth_proxy = np.zeros(n, dtype=np.float32)

    d_cent = np.linalg.norm(ca - centroid, axis=1)
    d_cent_norm = d_cent / max_extent
    ca_z_norm = (ca[:, 2] - ca[:, 2].mean()) / (ca[:, 2].std() + 1e-6)
    sa_proxy = 1.0 / (1.0 + n12.astype(np.float32) / 10.0)

    counts10 = mask10.sum(axis=1).astype(np.float32)
    mean_dir = mean_dir / (counts10[:, None] + 1e-3)
    anisotropy = np.linalg.norm(mean_dir, axis=1).astype(np.float32)
    mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir, axis=1, keepdims=True) + 1e-6)
    rel_burial = (mean_dir_norm * side_dir).sum(axis=1).astype(np.float32)

    bfac_mean = np.zeros(n, dtype=np.float32)
    bfac_std = np.zeros(n, dtype=np.float32)
    sc_extent = np.zeros(n, dtype=np.float32)
    for i, key in enumerate(keys):
        atoms = residues[key]["atoms"]
        bvals = [a[4] for a in atoms if not a[5]]
        if bvals:
            bfac_mean[i] = float(np.mean(bvals))
            bfac_std[i] = float(np.std(bvals))
        sc = residues[key]["sc"]
        if sc:
            sc_arr = np.asarray(sc, dtype=np.float32) - ca[i]
            sc_extent[i] = float(np.linalg.norm(sc_arr, axis=1).max())

    aa_onehot = np.zeros((n, len(AA_ORDER) + 1), dtype=np.float32)
    prop = np.zeros((n, NUM_PROPS), dtype=np.float32)
    for i, key in enumerate(keys):
        aa = AA3_TO_1.get(residues[key]["resname"], "X")
        if aa in AA_INDEX:
            aa_onehot[i, AA_INDEX[aa]] = 1.0
        else:
            aa_onehot[i, -1] = 1.0
        prop[i, 0] = KD.get(aa, 0.0)
        prop[i, 1] = VOL.get(aa, 0.0)
        prop[i, 2] = CHARGE.get(aa, 0.0)
        prop[i, 3] = 1.0 if aa in POLAR else 0.0
        prop[i, 4] = 1.0 if aa in AROMATIC else 0.0
        prop[i, 5] = 1.0 if aa in ALIPHATIC else 0.0
        prop[i, 6] = 1.0 if aa in HBOND_DONOR else 0.0
        prop[i, 7] = 1.0 if aa in HBOND_ACCEPTOR else 0.0
        prop[i, 8] = 1.0 if aa == "G" else 0.0

    geom = np.stack([
        n6.astype(np.float32), n8.astype(np.float32), n10.astype(np.float32),
        n12.astype(np.float32), n15.astype(np.float32),
        d_cent, d_cent_norm,
        depth_proxy,
        mean_nn.astype(np.float32), std_nn.astype(np.float32),
        up, down, hse_ratio,
        ca_z_norm,
        bfac_mean, bfac_std,
        sc_extent,
        pocket_open, pocket_concav,
        sa_proxy,
        anisotropy,
        rel_burial,
    ], axis=1)

    prop_idx = [PROP_NAMES.index(p) for p in NEIGH_AGG_PROPS]
    prop_sub = prop[:, prop_idx]
    neigh = []
    for radius in NEIGH_RADII:
        m = dist < radius
        counts = m.sum(axis=1).astype(np.float32) + 1e-3
        neigh.append((m.astype(np.float32) @ prop_sub) / counts[:, None])
    neigh_arr = np.concatenate(neigh, axis=1)

    X = np.concatenate([aa_onehot, prop, geom, neigh_arr], axis=1).astype(np.float32)

    res_ids = []
    for (chain, resnum, icode) in keys:
        if icode:
            res_ids.append(f"{chain}_{resnum}_{icode}")
        else:
            res_ids.append(f"{chain}_{resnum}")
    return res_ids, X


def featurize_pdb_bytes(pdb_bytes):
    order, residues = parse_pdb_atoms(pdb_bytes)
    return compute_features(order, residues)
