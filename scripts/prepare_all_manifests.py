#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from redundancy_elimination.datasets import create_flat_manifest

DATASETS = (
    "COCO",
    "VOC2012",
    "PerSeg",
    "ISIC2016",
    "BUSI",
    "Kvasir-SEG",
    "CAMO",
    "COD10K",
    "CHAMELEON",
)
SPLITS = ("search-1024", "test-1024", "train-1024")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index all normalized paper datasets")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    args = parser.parse_args()

    created = 0
    skipped = 0
    for dataset in args.datasets:
        for split in args.splits:
            source = args.dataset_root / dataset / split
            if not source.is_dir():
                print(f"skip missing split: {source}")
                skipped += 1
                continue
            output = args.output_root / f"{dataset.lower()}-{split}.jsonl"
            try:
                count = create_flat_manifest(source, output)
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {source}: {exc}", file=sys.stderr)
                return 1
            print(f"wrote {count:5d} records: {output}")
            created += 1
    print(f"created={created} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
