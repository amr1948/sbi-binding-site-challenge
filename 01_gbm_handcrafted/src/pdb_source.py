"""Read PDB bytes for a protein id from either a .zip archive or a plain directory.
"""
from __future__ import annotations

import os
import zipfile


def read_pdb_bytes(archive_or_dir: str, pid) -> bytes | None:
    """Return the raw PDB bytes for `{pid}_protein.pdb`, or None if missing."""
    name = f"{pid}_protein.pdb"
    if os.path.isdir(archive_or_dir):
        path = os.path.join(archive_or_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()
    with zipfile.ZipFile(archive_or_dir) as zf:
        try:
            return zf.read(name)
        except KeyError:
            return None


def list_protein_ids(archive_or_dir: str) -> list[int]:
    """Return sorted protein ids available in a .zip archive or plain directory."""
    if os.path.isdir(archive_or_dir):
        names = os.listdir(archive_or_dir)
    else:
        with zipfile.ZipFile(archive_or_dir) as zf:
            names = [os.path.basename(n) for n in zf.namelist()]
    pids = set()
    for base in names:
        if base.endswith("_protein.pdb"):
            try:
                pids.add(int(base[: -len("_protein.pdb")]))
            except ValueError:
                continue
    return sorted(pids)
