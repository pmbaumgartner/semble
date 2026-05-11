from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
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
from semble.types import Chunk, DuplicatePair

DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE = 0.40
_TEST_DIR_NAMES = frozenset({"tests", "test", "testing", "__tests__", "spec", "specs", "e2e"})
_TEST_FILE_PATTERNS = ("test_*.*", "*_test.*", "*_tests.*", "*.test.*", "*.spec.*")


def find_duplicate_pairs(
    chunks: Sequence[Chunk],
    semantic_index: Any,
    language_mapping: Mapping[str, Sequence[int]],
    *,
    candidate_k: int = 12,
    min_lines: int = 8,
    min_score: float = 0.0,
    min_structural_score: float = DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
    filter_languages: Sequence[str] | None = None,
    include_paths: Sequence[str] | None = None,
    exclude_paths: Sequence[str] | None = None,
    include_tests: bool = False,
    include_data: bool = False,
    include_scaffolding: bool = False,
) -> list[DuplicatePair]:
    """Return all sorted duplicate candidate pairs without top-k slicing."""
    if not chunks:
        return []

    features_by_index, eligible = _eligible_duplicate_features(
        chunks,
        language_mapping,
        min_lines=min_lines,
        filter_languages=filter_languages,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        include_tests=include_tests,
        include_data=include_data,
        include_scaffolding=include_scaffolding,
    )
    if not eligible:
        return []

    candidate_k = max(candidate_k, 1)
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
            min_score=min_score,
            min_structural_score=min_structural_score,
            pairs=pairs,
        )

    return sorted(pairs.values(), key=_duplicate_sort_key)


def _eligible_duplicate_features(
    chunks: Sequence[Chunk],
    language_mapping: Mapping[str, Sequence[int]],
    *,
    min_lines: int,
    filter_languages: Sequence[str] | None,
    include_paths: Sequence[str] | None,
    exclude_paths: Sequence[str] | None,
    include_tests: bool,
    include_data: bool,
    include_scaffolding: bool,
) -> tuple[dict[int, DuplicateFeatures], list[int]]:
    """Return duplicate features and indices for chunks that pass cheap eligibility gates."""
    features_by_index: dict[int, DuplicateFeatures] = {}
    eligible = []
    for index in _duplicate_candidate_indices(
        chunks,
        language_mapping,
        filter_languages,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        include_tests=include_tests,
    ):
        features = duplicate_features(chunks[index], include_scaffolding=include_scaffolding)
        if features.effective_line_count < min_lines:
            continue
        if not duplicate_features_are_eligible(
            features,
            include_data=include_data,
            include_scaffolding=include_scaffolding,
        ):
            continue
        features_by_index[index] = features
        eligible.append(index)
    return features_by_index, eligible


def _duplicate_candidate_indices(
    chunks: Sequence[Chunk],
    language_mapping: Mapping[str, Sequence[int]],
    filter_languages: Sequence[str] | None,
    *,
    include_paths: Sequence[str] | None,
    exclude_paths: Sequence[str] | None,
    include_tests: bool,
) -> list[int]:
    """Return chunk indices eligible for duplicate discovery."""
    eligible = set(range(len(chunks)))

    if filter_languages:
        language_indices: set[int] = set()
        for language in filter_languages:
            language_indices.update(language_mapping.get(language, ()))
        eligible &= language_indices

    path_indices = {
        index
        for index, chunk in enumerate(chunks)
        if _path_is_included(
            chunk.file_path,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_tests=include_tests,
        )
    }
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

    result_left, result_right, left_content, right_content = _ordered_duplicate_pair_values(
        left,
        right,
        features_by_index[left_index],
        features_by_index[right_index],
    )
    duplicate = DuplicatePair(
        left=result_left,
        right=result_right,
        score=score,
        signals=signals,
        left_content=left_content,
        right_content=right_content,
    )
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


def _ordered_duplicate_pair_values(
    left: Chunk,
    right: Chunk,
    left_features: DuplicateFeatures,
    right_features: DuplicateFeatures,
) -> tuple[Chunk, Chunk, str, str]:
    """Return pair chunks and scored content in stable location order."""
    left_key = (left.file_path, left.start_line, left.end_line)
    right_key = (right.file_path, right.start_line, right.end_line)
    if left_key <= right_key:
        return left, right, left_features.scored_content, right_features.scored_content
    return right, left, right_features.scored_content, left_features.scored_content


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


def _path_is_included(
    file_path: str,
    *,
    include_paths: Sequence[str] | None = None,
    exclude_paths: Sequence[str] | None = None,
    include_tests: bool,
) -> bool:
    normalized_path = _normalize_scope_path(file_path)
    include_scopes = _normalize_scopes(include_paths)
    exclude_scopes = _normalize_scopes(exclude_paths)
    if include_scopes and not _path_in_scopes(normalized_path, include_scopes):
        return False
    if exclude_scopes and _path_in_scopes(normalized_path, exclude_scopes):
        return False
    return include_tests or not _is_test_path(normalized_path)


def _normalize_scopes(scopes: Sequence[str] | None) -> tuple[str, ...] | None:
    if not scopes:
        return None
    return tuple(_normalize_scope_path(scope) for scope in scopes)


def _normalize_scope_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _path_in_scopes(normalized_path: str, normalized_scopes: Sequence[str]) -> bool:
    for normalized_scope in normalized_scopes:
        if normalized_scope in {"", "."}:
            return True
        if normalized_path == normalized_scope or normalized_path.startswith(f"{normalized_scope}/"):
            return True
    return False


def _is_test_path(file_path: str) -> bool:
    if not file_path:
        return False
    parts = file_path.split("/")
    if any(part.lower() in _TEST_DIR_NAMES for part in parts[:-1]):
        return True
    filename = parts[-1]
    normalized_filename = filename.lower()
    return any(fnmatchcase(normalized_filename, pattern) for pattern in _TEST_FILE_PATTERNS) or _is_pascal_test_file(
        filename
    )


def _is_pascal_test_file(filename: str) -> bool:
    stem = filename.rsplit(".", maxsplit=1)[0]
    return stem.endswith(("Test", "Tests"))
