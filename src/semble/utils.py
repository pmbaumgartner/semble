from __future__ import annotations

import os
import re
from typing import Any, Literal

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
    return {
        "query": label,
        "detail": "full",
        "clusters": [_format_duplicate_cluster(cluster, max_pairs=None, content_pairs="all") for cluster in clusters],
    }


def format_duplicate_clusters_compact(label: str, clusters: list[DuplicateCluster]) -> dict[str, Any]:
    """Render DuplicateCluster objects as compact JSON summaries."""
    return {
        "query": label,
        "detail": "compact",
        "clusters": [
            _format_duplicate_cluster(
                cluster,
                max_pairs=_MAX_COMPACT_DUPLICATE_PAIRS_SHOWN,
                content_pairs="first",
            )
            for cluster in clusters
        ],
    }


def _format_duplicate_cluster(
    cluster: DuplicateCluster,
    *,
    max_pairs: int | None,
    content_pairs: Literal["all", "first"],
) -> dict[str, Any]:
    shown_pairs = cluster.pairs if max_pairs is None else cluster.pairs[:max_pairs]
    return {
        "score": cluster.score,
        "members": [member.location for member in cluster.members],
        "pairs": [
            _format_duplicate_pair(
                pair,
                include_content=content_pairs == "all" or index == 0,
            )
            for index, pair in enumerate(shown_pairs)
        ],
        "pairs_not_shown": max(len(cluster.pairs) - len(shown_pairs), 0),
    }


def _format_duplicate_pair(pair: DuplicatePair, *, include_content: bool) -> dict[str, Any]:
    return {
        "left": _format_duplicate_pair_side(pair.left, pair.left_content, include_content=include_content),
        "right": _format_duplicate_pair_side(pair.right, pair.right_content, include_content=include_content),
        "score": pair.score,
        "signals": pair.signals.to_dict(),
    }


def _format_duplicate_pair_side(chunk: Chunk, content: str, *, include_content: bool) -> dict[str, str]:
    out = {"location": chunk.location}
    if include_content:
        out["content"] = content
    return out


def resolve_model_name() -> str:
    """Resolve a model name to a configurable."""
    return os.environ.get("SEMBLE_MODEL_NAME", DEFAULT_MODEL_NAME)
