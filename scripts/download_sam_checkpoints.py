#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official SAM checkpoints")
    parser.add_argument("--model", choices=(*CHECKPOINTS, "all"), default="vit_b")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = (
        CHECKPOINTS
        if args.model == "all"
        else {args.model: CHECKPOINTS[args.model]}
    )
    for model, (filename, url) in models.items():
        destination = args.output_dir / filename
        if destination.exists() and not args.force:
            print(f"skip existing {model}: {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"downloading {model}: {url}")
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
        print(f"saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
