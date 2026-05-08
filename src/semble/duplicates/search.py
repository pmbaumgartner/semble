from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from semble.duplicates.clustering import _pair_key, _same_file_ranges_overlap
from semble.duplicates.scoring import (
    DuplicateFeatures,
    duplicate_features,
    duplicate_features_are_eligible,
    duplicate_score,
    score_duplicate_features,
)
from semble.path_filters import PathFilter
from semble.types import Chunk, DuplicatePair

DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE = 0.40


@dataclass(frozen=True, slots=True, init=False)
class DuplicateOptions:
    """Normalized options for duplicate-code discovery."""

    top_k: int
    candidate_k: int
    min_lines: int
    min_score: float
    min_structural_score: float
    min_cluster_size: int
    filter_languages: tuple[str, ...] | None
    include_paths: tuple[str, ...] | None
    exclude_paths: tuple[str, ...] | None
    include_tests: bool
    include_data: bool
    include_scaffolding: bool

    def __init__(
        self,
        *,
        top_k: int = 5,
        candidate_k: int = 12,
        min_lines: int = 8,
        min_score: float = 0.0,
        min_structural_score: float = DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
        min_cluster_size: int = 2,
        filter_languages: Sequence[str] | None = None,
        include_paths: Sequence[str] | None = None,
        exclude_paths: Sequence[str] | None = None,
        include_tests: bool = False,
        include_data: bool = False,
        include_scaffolding: bool = False,
    ) -> None:
        """Create normalized duplicate discovery options."""
        _validate_at_least("top_k", top_k, 0)
        _validate_at_least("candidate_k", candidate_k, 1)
        _validate_at_least("min_lines", min_lines, 1)
        _validate_score("min_score", min_score)
        _validate_score("min_structural_score", min_structural_score)
        _validate_at_least("min_cluster_size", min_cluster_size, 2)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "candidate_k", candidate_k)
        object.__setattr__(self, "min_lines", min_lines)
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(self, "min_structural_score", min_structural_score)
        object.__setattr__(self, "min_cluster_size", min_cluster_size)
        object.__setattr__(self, "filter_languages", _tuple_or_none(filter_languages))
        object.__setattr__(self, "include_paths", _tuple_or_none(include_paths))
        object.__setattr__(self, "exclude_paths", _tuple_or_none(exclude_paths))
        object.__setattr__(self, "include_tests", include_tests)
        object.__setattr__(self, "include_data", include_data)
        object.__setattr__(self, "include_scaffolding", include_scaffolding)


def duplicate_options_from_values(
    *,
    top_k: int = 5,
    candidate_k: int = 12,
    language: str | None = None,
    filter_languages: Sequence[str] | None = None,
    include_paths: Sequence[str] | None = None,
    exclude_paths: Sequence[str] | None = None,
    include_tests: bool = False,
    include_data: bool = False,
    include_scaffolding: bool = False,
    min_lines: int = 8,
    min_score: float = 0.0,
    min_structural_score: float = DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
    min_cluster_size: int = 2,
) -> DuplicateOptions:
    """Build normalized duplicate search options from CLI/MCP-style values."""
    if language is not None and filter_languages is not None:
        raise ValueError("Use either language or filter_languages, not both.")
    return DuplicateOptions(
        top_k=top_k,
        candidate_k=candidate_k,
        filter_languages=[language] if language else filter_languages,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        include_tests=include_tests,
        include_data=include_data,
        include_scaffolding=include_scaffolding,
        min_lines=min_lines,
        min_score=min_score,
        min_structural_score=min_structural_score,
        min_cluster_size=min_cluster_size,
    )


def find_duplicate_pairs(
    chunks: Sequence[Chunk],
    semantic_index: Any,
    language_mapping: Mapping[str, Sequence[int]],
    options: DuplicateOptions,
) -> list[DuplicatePair]:
    """Return all sorted duplicate candidate pairs without top-k slicing."""
    if not chunks:
        return []

    features_by_index, eligible = _eligible_duplicate_features(chunks, language_mapping, options)
    if not eligible:
        return []

    candidate_k = max(options.candidate_k, 1)
    pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicatePair] = {}
    vectors = np.asarray(semantic_index.vectors)

    for group in _duplicate_language_groups(chunks, eligible).values():
        _collect_duplicate_pairs_for_group(
            chunks=chunks,
            semantic_index=semantic_index,
            group=group,
            vectors=vectors,
            features_by_index=features_by_index,
            candidate_k=candidate_k,
            min_score=options.min_score,
            min_structural_score=options.min_structural_score,
            pairs=pairs,
        )

    return sorted(pairs.values(), key=_duplicate_sort_key)


def _tuple_or_none(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    return tuple(values)


def _validate_at_least(name: str, value: int, minimum: int) -> None:
    """Validate that an integer option is at least a minimum value."""
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _validate_score(name: str, value: float) -> None:
    """Validate that a score threshold is in the inclusive 0..1 range."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")


def _eligible_duplicate_features(
    chunks: Sequence[Chunk],
    language_mapping: Mapping[str, Sequence[int]],
    options: DuplicateOptions,
) -> tuple[dict[int, DuplicateFeatures], list[int]]:
    """Return duplicate features and indices for chunks that pass cheap eligibility gates."""
    features_by_index: dict[int, DuplicateFeatures] = {}
    eligible = []
    path_filter = PathFilter(options.include_paths, options.exclude_paths, include_tests=options.include_tests)
    for index in _duplicate_candidate_indices(chunks, language_mapping, options.filter_languages, path_filter):
        if _line_count(chunks[index]) < options.min_lines:
            continue
        features = duplicate_features(chunks[index])
        if not duplicate_features_are_eligible(
            features,
            include_data=options.include_data,
            include_scaffolding=options.include_scaffolding,
        ):
            continue
        features_by_index[index] = features
        eligible.append(index)
    return features_by_index, eligible


def _duplicate_candidate_indices(
    chunks: Sequence[Chunk],
    language_mapping: Mapping[str, Sequence[int]],
    filter_languages: Sequence[str] | None,
    path_filter: PathFilter,
) -> list[int]:
    """Return chunk indices eligible for duplicate discovery."""
    eligible = set(range(len(chunks)))

    if filter_languages:
        language_indices: set[int] = set()
        for language in filter_languages:
            language_indices.update(language_mapping.get(language, ()))
        eligible &= language_indices

    path_indices = {index for index, chunk in enumerate(chunks) if path_filter.includes(chunk.file_path)}
    eligible &= path_indices
    return sorted(eligible)


def _collect_duplicate_pairs_for_group(
    *,
    chunks: Sequence[Chunk],
    semantic_index: Any,
    group: list[int],
    vectors: npt.NDArray[np.float32],
    features_by_index: dict[int, DuplicateFeatures],
    candidate_k: int,
    min_score: float,
    min_structural_score: float,
    pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicatePair],
) -> None:
    """Collect duplicate pairs from one same-language candidate group."""
    if len(group) < 2:
        return

    selector = np.array(group, dtype=np.int_)
    related = semantic_index.query(
        vectors[selector],
        k=min(candidate_k + 1, len(selector)),
        selector=selector,
    )

    for left_index, (indices, distances) in zip(group, related):
        for right_index_raw, distance in zip(indices, distances):
            _maybe_add_duplicate_pair(
                chunks=chunks,
                left_index=left_index,
                right_index=int(right_index_raw),
                semantic_score=1.0 - float(distance),
                features_by_index=features_by_index,
                min_score=min_score,
                min_structural_score=min_structural_score,
                pairs=pairs,
            )


def _maybe_add_duplicate_pair(
    *,
    chunks: Sequence[Chunk],
    left_index: int,
    right_index: int,
    semantic_score: float,
    features_by_index: dict[int, DuplicateFeatures],
    min_score: float,
    min_structural_score: float,
    pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicatePair],
) -> None:
    """Score and record one duplicate pair if it passes pair-level gates."""
    if right_index == left_index:
        return

    left = chunks[left_index]
    right = chunks[right_index]
    if _same_file_ranges_overlap(left, right):
        return

    signals = score_duplicate_features(
        features_by_index[left_index],
        features_by_index[right_index],
        semantic_score=semantic_score,
    )
    if signals.structural_score < min_structural_score:
        return
    score = duplicate_score(signals)
    if score <= 0.0 or score < min_score:
        return

    result_left, result_right = _ordered_duplicate_pair(left, right)
    duplicate = DuplicatePair(left=result_left, right=result_right, score=score, signals=signals)
    pair_key = _pair_key(left, right)
    existing = pairs.get(pair_key)
    if existing is None or _duplicate_rank_key(duplicate) > _duplicate_rank_key(existing):
        pairs[pair_key] = duplicate


def _duplicate_language_groups(chunks: Sequence[Chunk], eligible: list[int]) -> dict[str | None, list[int]]:
    """Group eligible duplicate candidates by exact language."""
    groups: dict[str | None, list[int]] = defaultdict(list)
    for index in eligible:
        groups[chunks[index].language].append(index)
    return dict(groups)


def _line_count(chunk: Chunk) -> int:
    """Return the number of content lines in a chunk."""
    return len(chunk.content.splitlines())


def _ordered_duplicate_pair(left: Chunk, right: Chunk) -> tuple[Chunk, Chunk]:
    """Return pair chunks in stable location order."""
    left_key = (left.file_path, left.start_line, left.end_line)
    right_key = (right.file_path, right.start_line, right.end_line)
    return (left, right) if left_key <= right_key else (right, left)


def _duplicate_rank_key(result: DuplicatePair) -> tuple[float, float, float]:
    """Return the descending score key used to keep the best version of a pair."""
    return (result.score, result.signals.semantic_score, result.signals.structural_score)


def _duplicate_sort_key(result: DuplicatePair) -> tuple[float, float, float, str, str]:
    """Return the final deterministic duplicate result sort key."""
    return (
        -result.score,
        -result.signals.semantic_score,
        -result.signals.structural_score,
        result.left.location,
        result.right.location,
    )
