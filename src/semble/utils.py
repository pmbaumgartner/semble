from __future__ import annotations

import os
import re
from typing import Any

from semble.types import Chunk, DuplicateCluster, DuplicatePair, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")
_DUPLICATE_SIGNAL_LEGEND = (
    "Signals are 0..1 similarities; higher means more alike, not confirmed duplication. score=ranking blend.",
    "semantic=embedding; structural=weighted tokens/AST blend; tokens=token n-grams; "
    "ast_type/ast_shape=AST overlap when available.",
)
_MAX_DUPLICATE_PAIRS_SHOWN = 5
DEFAULT_MODEL_NAME = "minishlab/potion-code-16M"


def is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def resolve_chunk(chunks: list[Chunk], file_path: str, line: int) -> Chunk | None:
    """Return the chunk containing *line* in *file_path*, or None.

    Reconstructs a Chunk from its JSON-primitive MCP tool arguments (file_path + line)
    before calling into the library.
    """
    fallback = None
    for chunk in chunks:
        if chunk.file_path == file_path and chunk.start_line <= line <= chunk.end_line:
            if line < chunk.end_line:
                return chunk
            if fallback is None:  # line == end_line: boundary; keep as fallback for end-of-file chunks
                fallback = chunk
    return fallback


def format_results(query: str, results: list[SearchResult]) -> dict[str, Any]:
    """Render SearchResult objects as a JSONable object."""
    return {"query": query, "results": [r.to_dict() for r in results]}


def resolve_model_name() -> str:
    """Resolve a model name to a configurable."""
    return os.environ.get("SEMBLE_MODEL_NAME", DEFAULT_MODEL_NAME)


def _format_duplicate_clusters(
    header: str,
    clusters: list[DuplicateCluster],
    *,
    empty_message: str | None = None,
) -> str:
    """Render DuplicateCluster objects as numbered grouped code blocks."""
    if not clusters and empty_message is not None:
        return empty_message

    sections: list[str] = [header, ""]
    if clusters:
        sections.extend([*_DUPLICATE_SIGNAL_LEGEND, ""])
    for i, cluster in enumerate(clusters, 1):
        sections.append(_format_duplicate_cluster(i, cluster))
    return "\n".join(sections)


def _format_duplicate_cluster(index: int, cluster: DuplicateCluster) -> str:
    strongest = cluster.pairs[0]
    shown_pairs = cluster.pairs[:_MAX_DUPLICATE_PAIRS_SHOWN]
    unlisted_pairs = len(cluster.pairs) - len(shown_pairs)
    member_lines = "\n".join(f"- {member.location}" for member in cluster.members)
    pair_lines = "\n".join(
        f"- {pair.left.location} <-> {pair.right.location}  [{' '.join(_duplicate_signal_parts(pair))}]"
        for pair in shown_pairs
    )
    unlisted_line = f"\nPairs not shown: {unlisted_pairs}" if unlisted_pairs else ""
    left_content = strongest.left_content.strip("\r\n")
    right_content = strongest.right_content.strip("\r\n")

    return (
        f"## {index}. Duplicate cluster  "
        f"[score={cluster.score:.3f}, members={len(cluster.members)}, pairs={len(cluster.pairs)}]\n"
        "Members:\n"
        f"{member_lines}\n\n"
        "Top pairs:\n"
        f"{pair_lines}"
        f"{unlisted_line}\n\n"
        "Strongest match left:\n"
        f"```\n{left_content}\n```\n\n"
        "Strongest match right:\n"
        f"```\n{right_content}\n```\n"
    )


def _duplicate_signal_parts(result: DuplicatePair) -> list[str]:
    """Return compact duplicate signal labels."""
    signals = result.signals
    parts = [
        f"score={result.score:.3f}",
        f"semantic={signals.semantic_score:.3f}",
        f"structural={signals.structural_score:.3f}",
        f"tokens={signals.token_jaccard:.3f}",
    ]
    if signals.ast_type_jaccard is not None:
        parts.append(f"ast_type={signals.ast_type_jaccard:.3f}")
    if signals.ast_shape_jaccard is not None:
        parts.append(f"ast_shape={signals.ast_shape_jaccard:.3f}")
    return parts
