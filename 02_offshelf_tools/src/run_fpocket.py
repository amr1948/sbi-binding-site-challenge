"""Run fpocket on every PDB under data/test_pdb in parallel.

Fpocket creates an `<input>_out/` directory next to the input PDB. We move
each input into its own subdir before running so the outputs land in
output/fpocket_full/<id>/.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(args):
    in_pdb, out_root = args
    base = os.path.basename(in_pdb).replace(".pdb", "")
    pid = base.replace("_protein", "")
    work = os.path.join(out_root, pid)
    os.makedirs(work, exist_ok=True)
    work_pdb = os.path.join(work, f"{base}.pdb")
    shutil.copy(in_pdb, work_pdb)
    try:
        subprocess.run(
            ["/opt/anaconda3/bin/fpocket", "-f", work_pdb],
            check=False, capture_output=True, timeout=240,
        )
    except subprocess.TimeoutExpired:
        pass
    # purge heavy files
    out_dir = os.path.join(work, f"{base}_out")
    if os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            if f.endswith((".pml", ".tcl", ".sh")) or f.endswith("_PYMOL.sh") or f.endswith("_VMD.sh"):
                try: os.remove(os.path.join(out_dir, f))
                except OSError: pass
    try: os.remove(work_pdb)
    except OSError: pass
    return pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    in_pdbs = sorted([
        os.path.join(args.in_dir, f)
        for f in os.listdir(args.in_dir)
        if f.endswith("_protein.pdb")
    ], key=lambda p: int(os.path.basename(p).split("_")[0]))
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, (p, args.out_dir)) for p in in_pdbs]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(in_pdbs):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(in_pdbs)} ({rate:.1f}/s)", flush=True)


if __name__ == "__main__":
    main()
