from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from redundancy_elimination.channels import apply_replacements
from redundancy_elimination.metrics import binary_iou
from redundancy_elimination.models import (
    CachedSample,
    EncodedImage,
    PointPrompt,
    ReplacementPair,
)
from redundancy_elimination.records import iter_results
from redundancy_elimination.search import run_search


class FakeBackend:
    channel_count = 3

    def encode(self, image: np.ndarray) -> EncodedImage:
        raise NotImplementedError

    def predict(
        self,
        encoded: EncodedImage,
        prompt: PointPrompt,
        pairs: Sequence[ReplacementPair],
        *,
        sequential: bool = False,
    ) -> np.ndarray:
        value = apply_replacements(encoded.feature, pairs, sequential=sequential)
        return value[0, 0].bool().numpy()


def test_search_finds_effective_replacement(tmp_path: Path) -> None:
    feature = torch.stack(
        (
            torch.zeros((2, 2)),
            torch.ones((2, 2)),
            torch.eye(2),
        )
    ).unsqueeze(0)
    sample = CachedSample(
        sample_id="example",
        encoded=EncodedImage(feature),
        mask=torch.ones((2, 2), dtype=torch.bool),
        prompt=PointPrompt(points=((0.0, 0.0),), labels=(1,)),
    )
    output = tmp_path / "search.jsonl"
    completed = run_search(
        FakeBackend(),
        [sample],
        [(ReplacementPair(0, 1),), (ReplacementPair(0, 2),)],
        output,
        log_every=0,
    )
    rows = list(iter_results(output))
    assert completed == 2
    assert rows[0].pairs == ()
    assert rows[0].miou == 0.0
    assert rows[1].miou == 1.0
    assert rows[1].delta == 1.0


def test_empty_masks_have_zero_iou() -> None:
    empty = torch.zeros((4, 4), dtype=torch.bool)
    assert binary_iou(empty, empty) == 0.0


def test_search_without_resume_replaces_previous_results(tmp_path: Path) -> None:
    feature = torch.ones((1, 1, 1, 1))
    sample = CachedSample(
        sample_id="example",
        encoded=EncodedImage(feature),
        mask=torch.ones((1, 1), dtype=torch.bool),
        prompt=PointPrompt(points=((0.0, 0.0),), labels=(1,)),
    )
    output = tmp_path / "search.jsonl"
    output.write_text('{"stale": true}\n', encoding="utf-8")
    run_search(FakeBackend(), [sample], [], output, resume=False, log_every=0)
    assert len(list(iter_results(output))) == 1
