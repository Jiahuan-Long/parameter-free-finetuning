from __future__ import annotations

from itertools import combinations, product
from typing import Iterable, Iterator, Sequence

import torch

from .models import ReplacementPair


def normalize_pairs(
    pairs: Iterable[ReplacementPair | Sequence[int]],
) -> tuple[ReplacementPair, ...]:
    normalized: list[ReplacementPair] = []
    for pair in pairs:
        if isinstance(pair, ReplacementPair):
            normalized.append(pair)
        else:
            if len(pair) != 2:
                raise ValueError(f"Expected a pair, got {pair!r}")
            normalized.append(ReplacementPair(int(pair[0]), int(pair[1])))
    return tuple(normalized)


def validate_pairs(
    pairs: Iterable[ReplacementPair | Sequence[int]],
    channels: int,
    *,
    unique_sources: bool = True,
) -> tuple[ReplacementPair, ...]:
    normalized = normalize_pairs(pairs)
    sources: set[int] = set()
    for pair in normalized:
        if not 0 <= pair.source < channels or not 0 <= pair.target < channels:
            raise IndexError(
                f"Channel pair {pair.as_tuple()} is outside [0, {channels - 1}]"
            )
        if unique_sources and pair.source in sources:
            raise ValueError(f"Source channel {pair.source} occurs more than once")
        sources.add(pair.source)
    return normalized


def apply_replacements(
    feature: torch.Tensor,
    pairs: Iterable[ReplacementPair | Sequence[int]],
    *,
    channel_dim: int = 1,
    sequential: bool = False,
) -> torch.Tensor:
    """Return a copy of ``feature`` with selected channels replaced.

    By default every target is read from the original feature tensor, matching
    the mapping in the paper and making a pair combination order-independent.
    ``sequential=True`` reproduces the order-dependent behavior of the original
    research scripts.
    """

    if feature.ndim < 3:
        raise ValueError("Expected a feature tensor with at least three dimensions")
    dim = channel_dim % feature.ndim
    normalized = validate_pairs(pairs, int(feature.shape[dim]))
    output = feature.clone()
    source_tensor = output if sequential else feature
    for pair in normalized:
        output.select(dim, pair.source).copy_(source_tensor.select(dim, pair.target))
    return output


def generate_replacement_pairs(
    channels: int,
    *,
    include_identity: bool = False,
) -> Iterator[tuple[ReplacementPair, ...]]:
    if channels <= 0:
        raise ValueError("channels must be positive")
    for source, target in product(range(channels), repeat=2):
        if include_identity or source != target:
            yield (ReplacementPair(source, target),)


def generate_candidate_combinations(
    candidates: Sequence[ReplacementPair],
) -> Iterator[tuple[ReplacementPair, ...]]:
    """Generate every non-empty subset of candidate pairs."""

    for size in range(1, len(candidates) + 1):
        yield from combinations(candidates, size)


def parse_pairs(value: str) -> tuple[ReplacementPair, ...]:
    """Parse ``source:target,source:target`` from the command line."""

    if not value.strip():
        return ()
    parsed: list[ReplacementPair] = []
    for item in value.split(","):
        try:
            source, target = item.strip().split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid replacement pair {item!r}") from exc
        parsed.append(ReplacementPair(int(source), int(target)))
    return tuple(parsed)
