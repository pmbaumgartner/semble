from __future__ import annotations

import os
import re
from typing import Any

from semble.types import Chunk, DuplicateCluster, DuplicatePair, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")
DEFAULT_MODEL_NAME = "minishlab/potion-code-16M"
_MAX_COMPACT_DUPLICATE_PAIRS_SHOWN = 5


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


def format_duplicate_clusters(label: str, clusters: list[DuplicateCluster]) -> dict[str, Any]:
    """Render DuplicateCluster objects as a JSONable object."""
    return {"query": label, "clusters": [cluster.to_dict() for cluster in clusters]}


def format_duplicate_clusters_compact(label: str, clusters: list[DuplicateCluster]) -> dict[str, Any]:
    """Render DuplicateCluster objects as compact JSON summaries."""
    return {
        "query": label,
        "detail": "compact",
        "clusters": [_compact_duplicate_cluster(cluster) for cluster in clusters],
    }


def _compact_duplicate_cluster(cluster: DuplicateCluster) -> dict[str, Any]:
    shown_pairs = cluster.pairs[:_MAX_COMPACT_DUPLICATE_PAIRS_SHOWN]
    return {
        "score": cluster.score,
        "members": [_compact_chunk(member) for member in cluster.members],
        "pairs": [
            _compact_duplicate_pair(pair, include_content=index == 0) for index, pair in enumerate(shown_pairs)
        ],
        "pairs_not_shown": max(len(cluster.pairs) - len(shown_pairs), 0),
    }


def _compact_duplicate_pair(pair: DuplicatePair, *, include_content: bool) -> dict[str, Any]:
    out = {
        "left": _compact_chunk(pair.left),
        "right": _compact_chunk(pair.right),
        "score": pair.score,
        "signals": _compact_duplicate_signals(pair),
    }
    if include_content:
        out["left_content"] = pair.left_content
        out["right_content"] = pair.right_content
    return out


def _compact_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "location": chunk.location,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }


def _compact_duplicate_signals(pair: DuplicatePair) -> dict[str, float]:
    signals = pair.signals
    out = {
        "semantic_score": signals.semantic_score,
        "structural_score": signals.structural_score,
        "token_jaccard": signals.token_jaccard,
    }
    if signals.ast_type_jaccard is not None:
        out["ast_type_jaccard"] = signals.ast_type_jaccard
    if signals.ast_shape_jaccard is not None:
        out["ast_shape_jaccard"] = signals.ast_shape_jaccard
    return out


def resolve_model_name() -> str:
    """Resolve a model name to a configurable."""
    return os.environ.get("SEMBLE_MODEL_NAME", DEFAULT_MODEL_NAME)
