from pathlib import Path

from redundancy_elimination.models import ReplacementPair, SearchResult
from redundancy_elimination.records import (
    append_result,
    iter_results,
    read_replacement_config,
    select_top_candidates,
    write_replacement_config,
)


def test_jsonl_round_trip_and_unique_source_selection(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        SearchResult((ReplacementPair(0, 1),), 0.7, 0.2, 50),
        SearchResult((ReplacementPair(0, 2),), 0.6, 0.1, 50),
        SearchResult((ReplacementPair(2, 1),), 0.5, 0.0, 50),
    ]
    for row in rows:
        append_result(path, row)
    assert list(iter_results(path)) == rows
    assert select_top_candidates(iter_results(path), 2) == [
        ReplacementPair(0, 1),
        ReplacementPair(2, 1),
    ]


def test_replacement_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "replacement.json"
    result = SearchResult(
        (ReplacementPair(4, 9), ReplacementPair(7, 2)),
        0.75,
        0.1,
        50,
    )
    write_replacement_config(path, dataset="COCO", backend="sam", result=result)
    assert read_replacement_config(path) == result.pairs
