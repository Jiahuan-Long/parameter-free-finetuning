from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .models import ReplacementPair, SearchResult


def result_to_dict(result: SearchResult) -> dict[str, object]:
    return {
        "pairs": [list(pair.as_tuple()) for pair in result.pairs],
        "miou": result.miou,
        "delta": result.delta,
        "samples": result.samples,
    }


def result_from_dict(value: dict[str, object]) -> SearchResult:
    raw_pairs = value.get("pairs", [])
    if not isinstance(raw_pairs, list):
        raise ValueError("Result field 'pairs' must be a list")
    pairs = tuple(ReplacementPair(int(pair[0]), int(pair[1])) for pair in raw_pairs)
    return SearchResult(
        pairs=pairs,
        miou=float(value["miou"]),
        delta=float(value.get("delta", 0.0)),
        samples=int(value["samples"]),
    )


def iter_results(path: str | Path) -> Iterator[SearchResult]:
    result_path = Path(path)
    if not result_path.exists():
        return
    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                yield result_from_dict(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                message = f"Invalid result at {result_path}:{line_number}"
                raise ValueError(message) from exc


def append_result(path: str | Path, result: SearchResult) -> None:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result_to_dict(result), sort_keys=True) + "\n")


def completed_keys(path: str | Path) -> set[str]:
    return {result.key for result in iter_results(path)}


def select_top_candidates(
    results: Iterable[SearchResult],
    count: int,
    *,
    unique_sources: bool = True,
    positive_only: bool = False,
) -> list[ReplacementPair]:
    if count <= 0:
        raise ValueError("count must be positive")
    ranked = sorted(results, key=lambda item: item.miou, reverse=True)
    selected: list[ReplacementPair] = []
    used_sources: set[int] = set()
    for result in ranked:
        if len(result.pairs) != 1:
            continue
        pair = result.pairs[0]
        if pair.source == pair.target:
            continue
        if positive_only and result.delta <= 0:
            continue
        if unique_sources and pair.source in used_sources:
            continue
        selected.append(pair)
        used_sources.add(pair.source)
        if len(selected) == count:
            break
    return selected


def write_replacement_config(
    path: str | Path,
    *,
    dataset: str,
    backend: str,
    result: SearchResult,
) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "backend": backend,
        **result_to_dict(result),
    }
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_replacement_config(path: str | Path) -> tuple[ReplacementPair, ...]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Configuration root must be an object")
        return result_from_dict(value).pairs
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid replacement config: {config_path}") from exc
