from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semble.chunking.core import get_parser_for_language
from semble.duplicates.ast.parser import _parser_language_for_chunk
from semble.duplicates.ast.scaffolding import _strip_scaffolding_content
from semble.duplicates.ast.taxonomy import (
    AstStats,
    _is_ignored_ast_subtree,
    _is_strippable_scaffolding_node,
    _is_transparent_scaffolding_node,
    _normalize_ast_label,
    _stats_for_node,
)
from semble.duplicates.tokens import _NGRAM_SIZE, _ngrams


@dataclass(frozen=True, slots=True)
class AstDuplicateFeatures:
    """Parser-derived structural duplicate features for one chunk."""

    type_ngrams: set[str]
    shape_edges: set[str]
    stats: AstStats
    scored_content: str


def _ast_features(
    content: str,
    language: str | None,
    *,
    include_scaffolding: bool = True,
) -> AstDuplicateFeatures | None:
    parser_language = _parser_language_for_chunk(language)
    if parser_language is None:
        return None

    parser = get_parser_for_language(parser_language)
    if parser is None:
        return None

    try:
        tree = parser.parse(content.encode("utf-8", errors="ignore"))
    except Exception:
        return None

    labels: list[str] = []
    shape_edges: set[str] = set()
    stats = _collect_ast_sequences(tree.root_node, labels, shape_edges, include_scaffolding=include_scaffolding)
    scored_content = content if include_scaffolding else _strip_scaffolding_content(content, tree.root_node)
    return AstDuplicateFeatures(
        type_ngrams=_ngrams(labels, size=_NGRAM_SIZE),
        shape_edges=shape_edges,
        stats=stats,
        scored_content=scored_content,
    )


def _collect_ast_sequences(
    node: Any,
    labels: list[str],
    shape_edges: set[str],
    parent_label: str | None = None,
    *,
    include_scaffolding: bool = True,
) -> AstStats:
    if _is_ignored_ast_subtree(node.type):
        return AstStats()
    if not include_scaffolding and _is_strippable_scaffolding_node(node.type):
        return AstStats()

    stats = AstStats()
    child_parent = parent_label
    is_transparent_scaffolding = not include_scaffolding and _is_transparent_scaffolding_node(node.type)
    if node.is_named and node.type != "ERROR" and not is_transparent_scaffolding:
        label = _normalize_ast_label(node.type)
        labels.append(label)
        stats = _stats_for_node(node.type)
        if parent_label is not None:
            shape_edges.add(f"{parent_label}>{label}")
        child_parent = label

    for child in node.children:
        stats += _collect_ast_sequences(
            child,
            labels,
            shape_edges,
            child_parent,
            include_scaffolding=include_scaffolding,
        )
    return stats
