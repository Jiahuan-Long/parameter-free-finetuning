from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch

from .backends.sam import SAMBackend
from .backends.sam2 import SAM2Backend
from .channels import (
    generate_candidate_combinations,
    generate_replacement_pairs,
    parse_pairs,
)
from .datasets import create_flat_manifest, validate_manifest
from .models import ReplacementPair
from .records import (
    iter_results,
    read_replacement_config,
    result_to_dict,
    select_top_candidates,
    write_replacement_config,
)
from .search import best_result, cache_manifest, run_search


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("sam", "sam2"), default="sam")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--cache-device", default="cpu")
    parser.add_argument("--model-type", default="vit_b", help="SAM model type")
    parser.add_argument("--model-config", help="SAM 2 Hydra model config")


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    _add_backend_arguments(parser)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Reproduce legacy order-dependent replacement",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)


def _backend_from_args(args: argparse.Namespace):
    if args.backend == "sam":
        return SAMBackend(
            args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            cache_device=args.cache_device,
        )
    if not args.model_config:
        raise SystemExit("--model-config is required for the SAM 2 backend")
    return SAM2Backend(
        args.checkpoint,
        model_config=args.model_config,
        device=args.device,
        cache_device=args.cache_device,
    )


def _shard(
    values: Iterable[tuple[ReplacementPair, ...]], index: int, count: int
) -> Iterable[tuple[ReplacementPair, ...]]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("Require shard-count > 0 and 0 <= shard-index < shard-count")
    return (value for offset, value in enumerate(values) if offset % count == index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfft-re",
        description="Parameter-free fine-tuning through feature-channel replacement",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-manifest", help="Index a legacy flat dataset split"
    )
    prepare.add_argument("source", type=Path)
    prepare.add_argument("output", type=Path)

    validate = subparsers.add_parser("validate-manifest", help="Check manifest paths")
    validate.add_argument("manifest", type=Path)

    single = subparsers.add_parser(
        "search-single", help="Search all single replacement pairs"
    )
    _add_search_arguments(single)
    single.add_argument("--include-identity", action="store_true")
    single.add_argument("--shard-index", type=int, default=0)
    single.add_argument("--shard-count", type=int, default=1)

    combine = subparsers.add_parser(
        "search-combinations", help="Search subsets of top single pairs"
    )
    _add_search_arguments(combine)
    combine.add_argument("--single-results", required=True, type=Path)
    combine.add_argument("--top-n", type=int, default=10)
    combine.add_argument("--positive-only", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate one replacement combination"
    )
    _add_search_arguments(evaluate)
    replacement = evaluate.add_mutually_exclusive_group(required=True)
    replacement.add_argument("--config", type=Path, help="Exported replacement JSON")
    replacement.add_argument("--pairs", help="source:target,source:target")

    select = subparsers.add_parser(
        "select", help="Export the best searched combination"
    )
    select.add_argument("results", type=Path)
    select.add_argument("output", type=Path)
    select.add_argument("--dataset", required=True)
    select.add_argument("--backend", choices=("sam", "sam2"), required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-manifest":
        count = create_flat_manifest(args.source, args.output)
        print(f"wrote {count} records to {args.output}")
        return 0
    if args.command == "validate-manifest":
        count, errors = validate_manifest(args.manifest)
        print(f"records={count} errors={len(errors)}")
        for error in errors:
            print(error)
        return 1 if errors else 0
    if args.command == "select":
        result = best_result(args.results)
        write_replacement_config(
            args.output,
            dataset=args.dataset,
            backend=args.backend,
            result=result,
        )
        print(json.dumps(result_to_dict(result), indent=2))
        return 0

    backend = _backend_from_args(args)
    samples = cache_manifest(backend, args.manifest, max_samples=args.max_samples)
    if args.no_resume and args.output.exists():
        args.output.unlink()
    if args.command == "search-single":
        pair_sets = generate_replacement_pairs(
            backend.channel_count, include_identity=args.include_identity
        )
        pair_sets = _shard(pair_sets, args.shard_index, args.shard_count)
    elif args.command == "search-combinations":
        candidates = select_top_candidates(
            iter_results(args.single_results),
            args.top_n,
            positive_only=args.positive_only,
        )
        if not candidates:
            raise SystemExit("No candidate pairs were selected")
        pair_sets = generate_candidate_combinations(candidates)
    else:
        pairs = (
            read_replacement_config(args.config)
            if args.config
            else parse_pairs(args.pairs)
        )
        pair_sets = (pairs,)

    completed = run_search(
        backend,
        samples,
        pair_sets,
        args.output,
        sequential=args.sequential,
        resume=not args.no_resume,
        log_every=args.log_every,
    )
    print(f"completed={completed} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
