"""Read PDB bytes for a protein id from either a .zip archive or a plain
directory of `<id>_protein.pdb` files.

train.zip/test.zip are no longer shipped (only the extracted data/train and
data/test folders remain), so the EGNN scripts accept both.
"""
from __future__ import annotations

import os
import zipfile


def read_pdb_bytes(archive_or_dir: str, pid) -> bytes | None:
    name = f"{pid}_protein.pdb"
    if os.path.isdir(archive_or_dir):
        path = os.path.join(archive_or_dir, name)
        return open(path, "rb").read() if os.path.exists(path) else None
    with zipfile.ZipFile(archive_or_dir) as zf:
        try:
            return zf.read(name)
        except KeyError:
            return None


def list_protein_ids(archive_or_dir: str) -> list[int]:
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
