"""Internal helpers for duplicate-code scoring."""

from __future__ import annotations

import re
from collections.abc import Sequence

from semble.types import Chunk, DuplicateSignals

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+(?:\.\d+)?"
    r"|==|!=|<=|>=|\+=|-=|\*=|/=|//=|%=|\*\*|->|=>|&&|\|\|"
    r"|[{}()[\],.:;+\-*/%<>=]"
)
_COMMENT_PREFIXES = ("#", "//", "--", "/*", "*/")
_NGRAM_SIZE = 4

_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "def",
        "do",
        "elif",
        "else",
        "enum",
        "except",
        "export",
        "extends",
        "false",
        "finally",
        "fn",
        "for",
        "from",
        "func",
        "function",
        "go",
        "if",
        "impl",
        "import",
        "in",
        "interface",
        "is",
        "lambda",
        "let",
        "match",
        "mod",
        "new",
        "nil",
        "none",
        "not",
        "null",
        "or",
        "package",
        "pass",
        "private",
        "protected",
        "public",
        "raise",
        "return",
        "self",
        "static",
        "struct",
        "super",
        "switch",
        "this",
        "throw",
        "trait",
        "true",
        "try",
        "type",
        "var",
        "while",
        "with",
        "yield",
    }
)


def score_duplicate_pair(left: Chunk, right: Chunk, *, semantic_score: float) -> DuplicateSignals:
    """Compute structural duplicate signals for two indexed chunks."""
    token_jaccard = _jaccard(_token_ngrams(left.content), _token_ngrams(right.content))
    return DuplicateSignals(
        semantic_score=semantic_score,
        structural_score=token_jaccard,
        token_jaccard=token_jaccard,
        ast_type_jaccard=None,
        ast_shape_jaccard=None,
    )


def duplicate_score(signals: DuplicateSignals) -> float:
    """Combine semantic and structural duplicate signals into a ranking score."""
    if signals.semantic_score <= 0 or signals.structural_score <= 0:
        return 0.0
    return signals.semantic_score**0.4 * signals.structural_score**0.6


def _token_ngrams(content: str) -> set[str]:
    return _ngrams(_token_sequence(content), size=_NGRAM_SIZE)


def _token_sequence(content: str) -> list[str]:
    return [_normalize_token(token) for token in _TOKEN_RE.findall(_code_content(content))]


def _code_content(content: str) -> str:
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines)


def _normalize_token(token: str) -> str:
    lower = token.lower()
    if lower in _KEYWORDS:
        return lower
    if token[0].isdigit():
        return "NUMBER"
    if token[0].isalpha() or token[0] == "_":
        return "IDENT"
    return token


def _ngrams(tokens: Sequence[str], *, size: int = _NGRAM_SIZE) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) <= size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _same_file_ranges_overlap(left: Chunk, right: Chunk) -> bool:
    if left.file_path != right.file_path:
        return False
    return max(left.start_line, right.start_line) <= min(left.end_line, right.end_line)


def _chunk_key(chunk: Chunk) -> tuple[str, int, int]:
    return (chunk.file_path, chunk.start_line, chunk.end_line)


def _pair_key(left: Chunk, right: Chunk) -> tuple[tuple[str, int, int], tuple[str, int, int]]:
    left_key = _chunk_key(left)
    right_key = _chunk_key(right)
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)
