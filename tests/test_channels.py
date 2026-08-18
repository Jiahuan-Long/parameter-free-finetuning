import pytest
import torch
from redundancy_elimination.channels import (
    apply_replacements,
    generate_candidate_combinations,
    generate_replacement_pairs,
    parse_pairs,
    validate_pairs,
)
from redundancy_elimination.models import ReplacementPair


def test_apply_replacements_is_simultaneous_by_default() -> None:
    feature = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    pairs = (ReplacementPair(0, 1), ReplacementPair(1, 2))
    result = apply_replacements(feature, pairs)
    assert result.flatten().tolist() == [2.0, 3.0, 3.0]
    assert feature.flatten().tolist() == [1.0, 2.0, 3.0]


def test_legacy_sequential_mode_reads_modified_channels() -> None:
    feature = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    pairs = (ReplacementPair(1, 2), ReplacementPair(0, 1))
    result = apply_replacements(feature, pairs, sequential=True)
    assert result.flatten().tolist() == [3.0, 3.0, 3.0]


def test_pairs_and_combinations() -> None:
    pairs = list(generate_replacement_pairs(3))
    assert len(pairs) == 6
    candidates = [ReplacementPair(0, 1), ReplacementPair(2, 1)]
    assert len(list(generate_candidate_combinations(candidates))) == 3
    assert parse_pairs("0:1,2:1") == tuple(candidates)


def test_duplicate_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="occurs more than once"):
        validate_pairs([(0, 1), (0, 2)], channels=3)
