from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from .models import PointPrompt, SampleRecord

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".bmp", ".webp")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def iter_manifest(path: str | Path) -> Iterator[SampleRecord]:
    manifest = Path(path).resolve()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                points = tuple((float(x), float(y)) for x, y in item["points"])
                labels = tuple(
                    int(label) for label in item.get("labels", [1] * len(points))
                )
                yield SampleRecord(
                    sample_id=str(item.get("id", Path(item["image"]).stem)),
                    image=_resolve(manifest.parent, item["image"]),
                    mask=_resolve(manifest.parent, item["mask"]),
                    prompt=PointPrompt(points=points, labels=labels),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                message = f"Invalid manifest record at {manifest}:{line_number}"
                raise ValueError(message) from exc


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_binary_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        values = np.any(values != 0, axis=2)
    else:
        values = values != 0
    return values.astype(bool, copy=False)


def read_legacy_prompt(path: str | Path) -> PointPrompt:
    """Read the legacy ``box_x1,...,box_y2,point_x,point_y`` text format."""

    prompt_path = Path(path)
    first_line = prompt_path.read_text(encoding="utf-8-sig").splitlines()[0]
    values = [float(item.strip()) for item in first_line.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError(f"No point coordinates found in {prompt_path}")
    return PointPrompt(points=((values[-2], values[-1]),), labels=(1,))


def create_flat_manifest(
    source: str | Path,
    output: str | Path,
    *,
    image_suffixes: Sequence[str] = IMAGE_SUFFIXES,
    mask_suffix: str = ".png",
    prompt_suffix: str = ".txt",
) -> int:
    """Create JSONL metadata for the flat layout used by the research scripts."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    suffixes = {suffix.lower() for suffix in image_suffixes}
    images = sorted(
        path
        for path in source_path.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for image in images:
        mask = image.with_suffix(mask_suffix)
        prompt_path = image.with_suffix(prompt_suffix)
        if not mask.exists() or not prompt_path.exists():
            missing.append(image.name)
            continue
        prompt = read_legacy_prompt(prompt_path)
        records.append(
            {
                "id": image.stem,
                "image": str(image),
                "mask": str(mask),
                "points": [list(point) for point in prompt.points],
                "labels": list(prompt.labels),
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        message = (
            f"{len(missing)} images lack a same-stem mask or prompt; "
            f"first entries: {preview}"
        )
        raise FileNotFoundError(message)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def validate_manifest(path: str | Path) -> tuple[int, list[str]]:
    count = 0
    errors: list[str] = []
    for record in iter_manifest(path):
        count += 1
        if not record.image.is_file():
            errors.append(f"{record.sample_id}: missing image {record.image}")
        if not record.mask.is_file():
            errors.append(f"{record.sample_id}: missing mask {record.mask}")
    return count, errors
