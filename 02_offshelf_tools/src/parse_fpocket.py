"""Parse fpocket output into a submission CSV.

For each protein we look at `<id>/<id>_protein_out/pockets/pocket{1..N}_atm.pdb`
and read the unique chain_resnum tokens of the ATOM records. We keep the
residues of the top pocket(s).
"""
import argparse
import os

import pandas as pd


def parse_pocket(pdb_path):
    seen = set()
    out = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21:22]
                try:
                    resnum = int(line[22:26])
                except ValueError:
                    continue
                icode = line[26:27].strip()
                tag = f"{chain}_{resnum}_{icode}" if icode else f"{chain}_{resnum}"
                if tag not in seen:
                    seen.add(tag)
                    out.append(tag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpocket_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_proteins", type=int, default=3064)
    ap.add_argument("--top_pockets", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for pid in range(args.n_proteins):
        pocket_dir = os.path.join(args.fpocket_dir, str(pid),
                                  f"{pid}_protein_out", "pockets")
        if not os.path.isdir(pocket_dir):
            rows.append({"id": pid, "prediction": ""})
            continue
        seen = set()
        merged = []
        for k in range(1, args.top_pockets + 1):
            p = os.path.join(pocket_dir, f"pocket{k}_atm.pdb")
            if not os.path.exists(p):
                break
            for tok in parse_pocket(p):
                if tok not in seen:
                    seen.add(tok)
                    merged.append(tok)
        rows.append({"id": pid, "prediction": " ".join(merged)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
