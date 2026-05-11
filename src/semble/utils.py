from __future__ import annotations

import re

from semble.types import Chunk, DuplicateCluster, DuplicatePair, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")
_DUPLICATE_SIGNAL_LEGEND = (
    "Signals are 0..1 similarities; higher means more alike, not confirmed duplication. score=ranking blend.",
    "semantic=embedding; structural=weighted tokens/AST blend; tokens=token n-grams; "
    "ast_type/ast_shape=AST overlap when available.",
)
_MAX_DUPLICATE_PAIRS_SHOWN = 5


def _is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def _resolve_chunk(chunks: list[Chunk], file_path: str, line: int) -> Chunk | None:
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


def _format_results(header: str, results: list[SearchResult]) -> str:
    """Render SearchResult objects as numbered, fenced code blocks."""
    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.chunk.location}  [score={r.score:.3f}]")
        _append_fenced_block(lines, r.chunk.content)
        lines.append("")
    return "\n".join(lines)


def _format_duplicate_clusters(header: str, clusters: list[DuplicateCluster]) -> str:
    """Render DuplicateCluster objects as numbered grouped code blocks."""
    lines: list[str] = [header, ""]
    if clusters:
        lines.extend([*_DUPLICATE_SIGNAL_LEGEND, ""])
    for i, cluster in enumerate(clusters, 1):
        strongest = cluster.pairs[0]

        lines.append(
            f"## {i}. Duplicate cluster  [score={cluster.score:.3f}, "
            f"members={len(cluster.members)}, pairs={len(cluster.pairs)}]"
        )
        lines.append("Members:")
        for member in cluster.members:
            lines.append(f"- {member.location}")
        lines.append("")
        shown_pairs = cluster.pairs[:_MAX_DUPLICATE_PAIRS_SHOWN]
        lines.append("Top pairs:")
        for pair in shown_pairs:
            lines.append(f"- {pair.left.location} <-> {pair.right.location}  [{_duplicate_pair_score_parts(pair)}]")
        unlisted_pairs = len(cluster.pairs) - len(shown_pairs)
        if unlisted_pairs:
            lines.append(f"Pairs not shown: {unlisted_pairs}")
        lines.append("")
        lines.append("Strongest pair left:")
        _append_fenced_block(lines, strongest.left.content)
        lines.append("")
        lines.append("Strongest pair right:")
        _append_fenced_block(lines, strongest.right.content)
        lines.append("")
    return "\n".join(lines)


def _format_duplicate_search_result(clusters: list[DuplicateCluster]) -> str:
    """Render duplicate search output or its empty state."""
    if not clusters:
        return "No duplicate clusters found."
    return _format_duplicate_clusters("Duplicate clusters", clusters)


def _duplicate_signal_parts(result: DuplicatePair) -> list[str]:
    """Return compact duplicate signal labels."""
    signals = result.signals
    parts = [
        f"semantic={signals.semantic_score:.3f}",
        f"structural={signals.structural_score:.3f}",
        f"tokens={signals.token_jaccard:.3f}",
    ]
    if signals.ast_type_jaccard is not None:
        parts.append(f"ast_type={signals.ast_type_jaccard:.3f}")
    if signals.ast_shape_jaccard is not None:
        parts.append(f"ast_shape={signals.ast_shape_jaccard:.3f}")
    return parts


def _duplicate_pair_score_parts(result: DuplicatePair) -> str:
    """Return compact duplicate pair score labels."""
    return " ".join([f"score={result.score:.3f}", *_duplicate_signal_parts(result)])


def _append_fenced_block(lines: list[str], content: str) -> None:
    """Append one plain fenced code block to a line buffer."""
    lines.append("```")
    lines.append(content.strip("\r\n"))
    lines.append("```")
