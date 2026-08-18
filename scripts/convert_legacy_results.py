#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from redundancy_elimination.models import ReplacementPair, SearchResult
from redundancy_elimination.records import append_result

LINE = re.compile(
    r"Average IoU for replace_pairs\s+(\[.*\]):\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_line(line: str) -> tuple[tuple[ReplacementPair, ...], float] | None:
    match = LINE.search(line.strip())
    if not match:
        return None
    raw_pairs = ast.literal_eval(match.group(1))
    pairs = tuple(
        ReplacementPair(int(source), int(target)) for source, target in raw_pairs
    )
    return pairs, float(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert historical text scores to JSONL"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=0, help="0 when unknown")
    parser.add_argument("--baseline", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    parsed = [
        item
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if (item := parse_line(line))
    ]
    if not parsed:
        raise SystemExit(f"No legacy result lines found in {args.source}")
    baseline = args.baseline
    if baseline is None:
        baseline = next(
            (
                score
                for pairs, score in parsed
                if len(pairs) == 1 and pairs[0].source == pairs[0].target
            ),
            None,
        )
    if baseline is None:
        raise SystemExit("No identity-pair baseline found; pass --baseline")
    if args.output.exists():
        if not args.force:
            raise SystemExit(
                f"Output exists: {args.output}; pass --force to replace it"
            )
        args.output.unlink()
    append_result(args.output, SearchResult((), baseline, 0.0, args.samples))
    for pairs, score in parsed:
        if len(pairs) == 1 and pairs[0].source == pairs[0].target:
            continue
        append_result(
            args.output,
            SearchResult(pairs, score, score - baseline, args.samples),
        )
    print(f"converted={len(parsed)} baseline={baseline:.8f} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
