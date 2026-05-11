from __future__ import annotations

from dataclasses import dataclass

from semble.duplicates.ast import _ast_features
from semble.duplicates.tokens import _jaccard, _token_ngrams
from semble.types import Chunk, DuplicateSignals

_TOKEN_SIGNAL_WEIGHT = 0.5
_AST_TYPE_SIGNAL_WEIGHT = 0.25
_AST_SHAPE_SIGNAL_WEIGHT = 0.25
_MIN_CODE_BEARING_NODES = 4


@dataclass(frozen=True, slots=True)
class DuplicateFeatures:
    """Precomputed structural duplicate features for one chunk."""

    token_ngrams: set[str]
    ast_type_ngrams: set[str] | None = None
    ast_shape_edges: set[str] | None = None
    code_bearing_node_count: int | None = None
    behavioral_node_count: int | None = None
    data_shape_node_count: int | None = None
    static_binding_node_count: int | None = None
    scaffolding_node_count: int | None = None
    substantive_node_count: int | None = None
    effective_line_count: int = 0
    scored_content: str = ""


def duplicate_features(chunk: Chunk, *, include_scaffolding: bool = True) -> DuplicateFeatures:
    """Precompute duplicate-scoring features for a chunk."""
    ast = _ast_features(chunk.content, chunk.language, include_scaffolding=include_scaffolding)
    if ast is None:
        return DuplicateFeatures(
            token_ngrams=_token_ngrams(chunk.content),
            effective_line_count=_effective_line_count(chunk.content),
            scored_content=chunk.content,
        )

    ast_type_ngrams: set[str] | None = ast.type_ngrams
    ast_shape_edges: set[str] | None = ast.shape_edges
    if ast.stats.code_bearing < _MIN_CODE_BEARING_NODES:
        ast_type_ngrams = None
        ast_shape_edges = None
    return DuplicateFeatures(
        token_ngrams=_token_ngrams(ast.scored_content),
        ast_type_ngrams=ast_type_ngrams,
        ast_shape_edges=ast_shape_edges,
        code_bearing_node_count=ast.stats.code_bearing,
        behavioral_node_count=ast.stats.behavioral,
        data_shape_node_count=ast.stats.data_shape,
        static_binding_node_count=ast.stats.static_binding,
        scaffolding_node_count=ast.stats.scaffolding,
        substantive_node_count=ast.stats.substantive,
        effective_line_count=_effective_line_count(ast.scored_content),
        scored_content=ast.scored_content,
    )


def duplicate_features_are_eligible(
    features: DuplicateFeatures,
    *,
    min_code_bearing_nodes: int = _MIN_CODE_BEARING_NODES,
    include_data: bool = False,
    include_scaffolding: bool = False,
) -> bool:
    """Return whether a chunk has enough parser-visible code to scan for duplicates."""
    if features.code_bearing_node_count is None:
        return True
    if features.code_bearing_node_count < min_code_bearing_nodes:
        return False
    if not include_data and _is_static_data_features(features):
        return False
    if not include_scaffolding and _is_scaffolding_features(features):
        return False
    return True


def _is_static_data_features(features: DuplicateFeatures) -> bool:
    """Return whether parser-backed features look like literal/container data."""
    return features.behavioral_node_count == 0 and bool(
        features.data_shape_node_count or features.static_binding_node_count
    )


def _is_scaffolding_features(features: DuplicateFeatures) -> bool:
    """Return whether parser-backed features look like imports/headers/attributes only."""
    return (
        features.behavioral_node_count == 0
        and features.data_shape_node_count == 0
        and features.static_binding_node_count == 0
        and bool(features.scaffolding_node_count)
        and features.substantive_node_count == 0
    )


def _effective_line_count(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip())


def score_duplicate_features(
    left: DuplicateFeatures,
    right: DuplicateFeatures,
    *,
    semantic_score: float,
) -> DuplicateSignals:
    """Compute structural duplicate signals from precomputed chunk features."""
    token_jaccard = _jaccard(left.token_ngrams, right.token_ngrams)
    ast_type_jaccard = None
    ast_shape_jaccard = None

    if left.ast_type_ngrams is not None and right.ast_type_ngrams is not None:
        if left.ast_type_ngrams or right.ast_type_ngrams:
            ast_type_jaccard = _jaccard(left.ast_type_ngrams, right.ast_type_ngrams)
    if left.ast_shape_edges is not None and right.ast_shape_edges is not None:
        if left.ast_shape_edges or right.ast_shape_edges:
            ast_shape_jaccard = _jaccard(left.ast_shape_edges, right.ast_shape_edges)

    return DuplicateSignals(
        semantic_score=semantic_score,
        structural_score=_weighted_structural_score(
            token_jaccard=token_jaccard,
            ast_type_jaccard=ast_type_jaccard,
            ast_shape_jaccard=ast_shape_jaccard,
        ),
        token_jaccard=token_jaccard,
        ast_type_jaccard=ast_type_jaccard,
        ast_shape_jaccard=ast_shape_jaccard,
    )


def score_duplicate_pair(left: Chunk, right: Chunk, *, semantic_score: float) -> DuplicateSignals:
    """Compute structural duplicate signals for two indexed chunks."""
    return score_duplicate_features(
        duplicate_features(left),
        duplicate_features(right),
        semantic_score=semantic_score,
    )


def duplicate_score(signals: DuplicateSignals) -> float:
    """Combine semantic and structural duplicate signals into a ranking score."""
    if signals.semantic_score <= 0 or signals.structural_score <= 0:
        return 0.0
    return signals.semantic_score**0.4 * signals.structural_score**0.6


def _weighted_structural_score(
    *,
    token_jaccard: float,
    ast_type_jaccard: float | None,
    ast_shape_jaccard: float | None,
) -> float:
    score = token_jaccard * _TOKEN_SIGNAL_WEIGHT
    weight = _TOKEN_SIGNAL_WEIGHT
    if ast_type_jaccard is not None:
        score += ast_type_jaccard * _AST_TYPE_SIGNAL_WEIGHT
        weight += _AST_TYPE_SIGNAL_WEIGHT
    if ast_shape_jaccard is not None:
        score += ast_shape_jaccard * _AST_SHAPE_SIGNAL_WEIGHT
        weight += _AST_SHAPE_SIGNAL_WEIGHT
    return score / weight if weight else 0.0
