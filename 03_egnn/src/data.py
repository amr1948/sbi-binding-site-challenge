"""Light-weight per-residue feature extractor for the EGNN approach.

Only the chemistry of each residue is encoded as input features — the EGNN
learns the geometric signal itself from the Cα coordinates, so we do not
hand-roll geometry the way Approach 1 did.
"""
from __future__ import annotations

import numpy as np

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HID": "H", "HIE": "H",
    "HIP": "H", "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}

KD = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
      "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
      "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
      "W": -0.9, "Y": -1.3}
VOL = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
       "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
       "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
       "W": 227.8, "Y": 193.6}
CHARGE = {"A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 0.1, "I": 0,
          "K": 1, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1, "S": 0,
          "T": 0, "V": 0, "W": 0, "Y": 0}
POLAR = set("STNQHYCDEKR")
AROMATIC = set("FWYH")
DONOR = set("RKQNHWSTY")
ACCEPTOR = set("DENQHSTY")

NODE_FEAT_DIM = len(AA_ORDER) + 1 + 7  # one-hot + 7 props


def parse_pdb_ca(pdb_bytes):
    """Yield residues with Cα atoms.

    Returns:
        coords  : (N, 3) float32 numpy
        feats   : (N, NODE_FEAT_DIM) float32 numpy
        res_ids : list[str] of len N (chain_resnum[_icode])
    """
    if isinstance(pdb_bytes, (bytes, bytearray)):
        text = pdb_bytes.decode("ascii", errors="ignore")
    else:
        text = pdb_bytes
    keys = []
    cas = {}
    resname = {}
    for line in text.splitlines():
        if len(line) < 54 or line[:6] != "ATOM  ":
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        alt = line[16:17]
        if alt not in (" ", "A", ""):
            continue
        chain = line[21:22]
        try:
            num = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        ic = line[26:27]
        if ic == " ":
            ic = ""
        key = (chain, num, ic)
        if key in cas:
            continue
        cas[key] = (x, y, z)
        resname[key] = line[17:20].strip()
        keys.append(key)

    n = len(keys)
    coords = np.zeros((n, 3), dtype=np.float32)
    feats = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    res_ids = []
    for i, k in enumerate(keys):
        coords[i] = cas[k]
        aa = AA3_TO_1.get(resname[k], "X")
        if aa in AA_INDEX:
            feats[i, AA_INDEX[aa]] = 1.0
        else:
            feats[i, len(AA_ORDER)] = 1.0
        base = len(AA_ORDER) + 1
        feats[i, base + 0] = KD.get(aa, 0.0) / 5.0
        feats[i, base + 1] = VOL.get(aa, 0.0) / 200.0
        feats[i, base + 2] = CHARGE.get(aa, 0.0)
        feats[i, base + 3] = 1.0 if aa in POLAR else 0.0
        feats[i, base + 4] = 1.0 if aa in AROMATIC else 0.0
        feats[i, base + 5] = 1.0 if aa in DONOR else 0.0
        feats[i, base + 6] = 1.0 if aa in ACCEPTOR else 0.0
        ch, num, ic = k
        res_ids.append(f"{ch}_{num}_{ic}" if ic else f"{ch}_{num}")
    return coords, feats, res_ids
