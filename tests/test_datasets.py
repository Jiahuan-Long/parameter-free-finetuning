from pathlib import Path

from PIL import Image
from redundancy_elimination.datasets import (
    create_flat_manifest,
    iter_manifest,
    load_binary_mask,
    validate_manifest,
)


def test_create_and_validate_flat_manifest(tmp_path: Path) -> None:
    split = tmp_path / "search-1024"
    split.mkdir()
    Image.new("RGB", (4, 3), "white").save(split / "sample.jpg")
    Image.new("L", (4, 3), 255).save(split / "sample.png")
    (split / "sample.txt").write_text("0,0,3,2,2,1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"

    assert create_flat_manifest(split, manifest) == 1
    records = list(iter_manifest(manifest))
    assert records[0].prompt.points == ((2.0, 1.0),)
    assert load_binary_mask(records[0].mask).all()
    assert validate_manifest(manifest) == (1, [])
