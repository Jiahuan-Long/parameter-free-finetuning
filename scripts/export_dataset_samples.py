from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DATASETS = {
    "BUSI": ("busi", "search-1024"),
    "CAMO": ("camo", "search-1024"),
    "COCO": ("coco", "search-1024"),
    "COD10K": ("cod10k", "search-1024"),
    "ISIC2016": ("isic2016", "search-1024"),
    "Kvasir-SEG": ("kvasir-seg", "search-1024"),
    "PerSeg": ("perseg", "test-1024"),
    "VOC2012": ("voc2012", "search-1024"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".webp"}


def read_point(prompt_path: Path) -> list[list[float]]:
    first_line = prompt_path.read_text(encoding="utf-8-sig").splitlines()[0]
    values = [float(value.strip()) for value in first_line.split(",") if value.strip()]
    if len(values) < 2:
        raise ValueError(f"No point coordinates found in {prompt_path}")
    return [[values[-2], values[-1]]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a small, repository-relative dataset sample bundle"
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("data/samples"), type=Path)
    parser.add_argument(
        "--split",
        help="Override the per-dataset default source split for every dataset",
    )
    parser.add_argument("--samples-per-dataset", default=100, type=int)
    args = parser.parse_args()

    if args.samples_per_dataset < 1:
        parser.error("--samples-per-dataset must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    records: list[dict[str, object]] = []

    for dataset, (slug, default_split) in DATASETS.items():
        split = args.split or default_split
        source = (args.dataset_root / dataset / split).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Missing normalized split: {source}")
        images = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(images) < args.samples_per_dataset:
            raise ValueError(
                f"{dataset}/{split} has {len(images)} images; "
                f"requested {args.samples_per_dataset}"
            )

        target = output_root / slug / split
        target.mkdir(parents=True, exist_ok=True)
        for image in images[: args.samples_per_dataset]:
            mask = image.with_suffix(".png")
            prompt = image.with_suffix(".txt")
            missing = [path for path in (mask, prompt) if not path.is_file()]
            if missing:
                names = ", ".join(str(path) for path in missing)
                raise FileNotFoundError(f"Missing companions for {image}: {names}")

            for source_file in (image, mask, prompt):
                shutil.copy2(source_file, target / source_file.name)

            relative_image = (target / image.name).relative_to(output_root)
            relative_mask = (target / mask.name).relative_to(output_root)
            records.append(
                {
                    "id": f"{slug}/{image.stem}",
                    "dataset": dataset,
                    "split": split,
                    "image": relative_image.as_posix(),
                    "mask": relative_mask.as_posix(),
                    "points": read_point(prompt),
                    "labels": [1],
                }
            )

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"exported={len(records)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
