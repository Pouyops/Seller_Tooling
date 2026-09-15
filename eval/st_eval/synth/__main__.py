"""python -m st_eval.synth --out <dir> [--n 200] [--seed 1403] [--workers 4] [--sheet sheet.jpg]"""

import argparse
import os
from pathlib import Path

from .generate import contact_sheet, generate_dataset


def default_dataset_dir(n: int = 200, seed: int = 1403) -> Path:
    root = Path(os.environ.get("ST_DATA_DIR", Path.cwd() / ".data"))
    return root / "data" / "bench" / f"synth-v1-s{seed}-n{n}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1403)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sheet", type=Path, default=None, help="also write a contact sheet JPEG")
    args = ap.parse_args()
    out = args.out or default_dataset_dir(args.n, args.seed)
    generate_dataset(out, n=args.n, seed=args.seed, workers=args.workers)
    if args.sheet:
        print("contact sheet:", contact_sheet(out, args.sheet))


if __name__ == "__main__":
    main()
