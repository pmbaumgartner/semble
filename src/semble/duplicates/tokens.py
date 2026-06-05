from __future__ import annotations

import re
from collections.abc import Sequence

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
