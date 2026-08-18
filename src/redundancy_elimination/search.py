from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from .backends.base import SegmentationBackend
from .channels import validate_pairs
from .datasets import iter_manifest, load_binary_mask, load_rgb
from .metrics import binary_iou
from .models import CachedSample, ReplacementPair, SearchResult
from .records import append_result, iter_results


def cache_manifest(
    backend: SegmentationBackend,
    manifest: str | Path,
    *,
    max_samples: int | None = None,
) -> list[CachedSample]:
    cached: list[CachedSample] = []
    for index, record in enumerate(iter_manifest(manifest)):
        if max_samples is not None and index >= max_samples:
            break
        encoded = backend.encode(load_rgb(record.image))
        mask = torch.from_numpy(load_binary_mask(record.mask))
        cached.append(
            CachedSample(
                sample_id=record.sample_id,
                encoded=encoded,
                mask=mask,
                prompt=record.prompt,
            )
        )
    if not cached:
        raise ValueError(f"No samples found in manifest {manifest}")
    return cached


def evaluate_replacements(
    backend: SegmentationBackend,
    samples: Sequence[CachedSample],
    pairs: Sequence[ReplacementPair],
    *,
    sequential: bool = False,
) -> float:
    normalized = validate_pairs(pairs, backend.channel_count)
    total = 0.0
    for sample in samples:
        prediction = backend.predict(
            sample.encoded,
            sample.prompt,
            normalized,
            sequential=sequential,
        )
        total += binary_iou(torch.as_tensor(prediction), sample.mask)
    return total / len(samples)


def run_search(
    backend: SegmentationBackend,
    samples: Sequence[CachedSample],
    pair_sets: Iterable[Sequence[ReplacementPair]],
    output: str | Path,
    *,
    sequential: bool = False,
    resume: bool = True,
    log_every: int = 100,
) -> int:
    output_path = Path(output)
    if not resume and output_path.exists():
        output_path.unlink()
    existing = list(iter_results(output_path)) if resume else []
    done = {result.key for result in existing} if resume else set()
    baseline = next((result for result in existing if not result.pairs), None)
    if baseline is None:
        baseline_miou = evaluate_replacements(
            backend, samples, (), sequential=sequential
        )
        baseline = SearchResult((), baseline_miou, 0.0, len(samples))
        append_result(output_path, baseline)
        done.add(baseline.key)
    completed = 0
    started = time.monotonic()
    for raw_pairs in pair_sets:
        pairs = validate_pairs(raw_pairs, backend.channel_count)
        key = ",".join(f"{pair.source}:{pair.target}" for pair in pairs)
        if key in done:
            continue
        miou = evaluate_replacements(
            backend, samples, pairs, sequential=sequential
        )
        result = SearchResult(
            pairs=pairs,
            miou=miou,
            delta=miou - baseline.miou,
            samples=len(samples),
        )
        append_result(output_path, result)
        done.add(result.key)
        completed += 1
        if log_every > 0 and completed % log_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"evaluated={completed} elapsed={elapsed:.1f}s "
                f"last={result.key} miou={result.miou:.6f} delta={result.delta:+.6f}",
                flush=True,
            )
    return completed


def best_result(path: str | Path) -> SearchResult:
    candidates = list(iter_results(path))
    if not candidates:
        raise ValueError(f"No results found in {path}")
    return max(candidates, key=lambda item: item.miou)
