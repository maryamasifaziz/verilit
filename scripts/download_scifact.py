"""Download the SciFact dataset (Wadden et al., 2020) into data/raw/scifact/.

Usage:
    python scripts/download_scifact.py
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
FILES = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "scifact")
    parser.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.force and all((args.out / f).exists() for f in FILES):
        print(f"SciFact already in {args.out} (use --force to download again)")
        return

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "scifact.tar.gz"
        print(f"Downloading {URL} ...")
        with urllib.request.urlopen(URL) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh)

        # Copy out only the files we need, by exact name, so nothing in the
        # archive can write anywhere else.
        with tarfile.open(archive, "r:gz") as tar:
            for name in FILES:
                member = tar.getmember(f"data/{name}")
                src = tar.extractfile(member)
                if src is None:
                    raise RuntimeError(f"data/{name} in the archive is not a regular file")
                with src, open(args.out / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    for name in FILES:
        with open(args.out / name, encoding="utf-8") as fh:
            print(f"  {name:<20} {sum(1 for _ in fh):>5} lines")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
