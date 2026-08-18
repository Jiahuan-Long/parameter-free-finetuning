from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, order=True)
class ReplacementPair:
    """Replace ``source`` channel with the values from ``target`` channel."""

    source: int
    target: int

    def as_tuple(self) -> tuple[int, int]:
        return self.source, self.target


@dataclass(frozen=True)
class PointPrompt:
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("A point prompt must contain at least one point")
        if len(self.points) != len(self.labels):
            raise ValueError("Point and label counts must match")


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image: Path
    mask: Path
    prompt: PointPrompt


@dataclass
class EncodedImage:
    feature: torch.Tensor
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedSample:
    sample_id: str
    encoded: EncodedImage
    mask: torch.Tensor
    prompt: PointPrompt


@dataclass(frozen=True)
class SearchResult:
    pairs: tuple[ReplacementPair, ...]
    miou: float
    delta: float
    samples: int

    @property
    def key(self) -> str:
        return ",".join(f"{pair.source}:{pair.target}" for pair in self.pairs)
