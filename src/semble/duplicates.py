"""Internal helpers for duplicate-code scoring."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from semble.types import Chunk, DuplicateSignals

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"|\d+(?:\.\d+)?"
    r"|==|!=|<=|>=|\+=|-=|\*=|/=|//=|%=|\*\*|->|=>|&&|\|\|"
    r"|[{}()[\],.:;+\-*/%<>=]"
)
_COMMENT_PREFIXES = ("#", "//", "--", "/*", "*/")
_NGRAM_SIZE = 4
_TOKEN_SIGNAL_WEIGHT = 0.5
_AST_TYPE_SIGNAL_WEIGHT = 0.25
_AST_SHAPE_SIGNAL_WEIGHT = 0.25

_PARSER_LANGUAGE_ALIASES = {
    "c#": "csharp",
    "c++": "cpp",
    "js": "javascript",
    "sh": "bash",
    "shell": "bash",
    "ts": "typescript",
}
_PARSER_LANGUAGES = frozenset(
    {
        "bash",
        "c",
        "cpp",
        "csharp",
        "dart",
        "elixir",
        "go",
        "haskell",
        "java",
        "javascript",
        "kotlin",
        "lua",
        "php",
        "python",
        "ruby",
        "rust",
        "scala",
        "sql",
        "swift",
        "typescript",
        "zig",
    }
)

_IDENTIFIER_NODE_TYPES = frozenset(
    {
        "field_identifier",
        "identifier",
        "property_identifier",
        "shorthand_property_identifier",
        "type_identifier",
    }
)
_NUMBER_NODE_TYPES = frozenset(
    {
        "decimal_integer_literal",
        "float",
        "float_literal",
        "integer",
        "integer_literal",
        "number",
        "number_literal",
    }
)
_STRING_NODE_TYPES = frozenset(
    {
        "char_literal",
        "character_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "string",
        "string_fragment",
        "string_literal",
    }
)
_BOOL_NODE_TYPES = frozenset({"bool", "boolean", "boolean_literal", "false", "false_literal", "true", "true_literal"})
_NULL_NODE_TYPES = frozenset({"nil", "null", "null_literal", "nullptr"})
_NONE_NODE_TYPES = frozenset({"none", "none_literal"})

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
_MIN_CODE_BEARING_NODES = 4


@dataclass(frozen=True, slots=True)
class DuplicateFeatures:
    """Precomputed structural duplicate features for one chunk."""

    token_ngrams: set[str]
    ast_type_ngrams: set[str] | None = None
    ast_shape_edges: set[str] | None = None
    code_bearing_node_count: int | None = None


def duplicate_features(chunk: Chunk) -> DuplicateFeatures:
    """Precompute duplicate-scoring features for a chunk."""
    ast = _ast_features(chunk.content, chunk.language)
    if ast is None:
        return DuplicateFeatures(token_ngrams=_token_ngrams(chunk.content))

    ast_type_ngrams, ast_shape_edges, code_bearing_node_count = ast
    if code_bearing_node_count < _MIN_CODE_BEARING_NODES:
        ast_type_ngrams = None
        ast_shape_edges = None
    return DuplicateFeatures(
        token_ngrams=_token_ngrams(chunk.content),
        ast_type_ngrams=ast_type_ngrams,
        ast_shape_edges=ast_shape_edges,
        code_bearing_node_count=code_bearing_node_count,
    )


def duplicate_features_are_eligible(
    features: DuplicateFeatures,
    *,
    min_code_bearing_nodes: int = _MIN_CODE_BEARING_NODES,
) -> bool:
    """Return whether a chunk has enough parser-visible code to scan for duplicates."""
    return features.code_bearing_node_count is None or features.code_bearing_node_count >= min_code_bearing_nodes


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


def _ast_fingerprint(content: str, language: str | None) -> tuple[set[str], set[str]] | None:
    features = _ast_features(content, language)
    if features is None:
        return None
    type_ngrams, shape_edges, _ = features
    return type_ngrams, shape_edges


def _ast_features(content: str, language: str | None) -> tuple[set[str], set[str], int] | None:
    parser_language = _parser_language_for_chunk(language)
    if parser_language is None:
        return None

    parser = _parser_for_language(parser_language)
    if parser is None:
        return None

    try:
        tree = parser.parse(content.encode("utf-8", errors="ignore"))
    except Exception:
        return None

    labels: list[str] = []
    shape_edges: set[str] = set()
    code_bearing_node_count = _collect_ast_sequences(tree.root_node, labels, shape_edges)
    type_ngrams = _ngrams(labels, size=_NGRAM_SIZE)
    return type_ngrams, shape_edges, code_bearing_node_count


def _collect_ast_sequences(
    node: Any,
    labels: list[str],
    shape_edges: set[str],
    parent_label: str | None = None,
) -> int:
    if _is_ignored_ast_subtree(node.type):
        return 0

    code_bearing_node_count = 0
    child_parent = parent_label
    if node.is_named and node.type != "ERROR":
        label = _normalize_ast_label(node.type)
        labels.append(label)
        code_bearing_node_count += 1
        if parent_label is not None:
            shape_edges.add(f"{parent_label}>{label}")
        child_parent = label

    for child in node.children:
        code_bearing_node_count += _collect_ast_sequences(child, labels, shape_edges, child_parent)
    return code_bearing_node_count


def _is_ignored_ast_subtree(node_type: str) -> bool:
    lower = node_type.lower()
    return lower == "comment" or lower.endswith("_comment") or _normalize_ast_label(lower) == "STRING"


def _normalize_ast_label(node_type: str) -> str:
    lower = node_type.lower()
    if lower in _IDENTIFIER_NODE_TYPES or lower.endswith("identifier"):
        return "IDENT"
    if lower in _NUMBER_NODE_TYPES or "number" in lower or lower.endswith("_integer_literal"):
        return "NUMBER"
    if lower in _STRING_NODE_TYPES or "string" in lower:
        return "STRING"
    if lower in _BOOL_NODE_TYPES:
        return "BOOL"
    if lower in _NULL_NODE_TYPES:
        return "NULL"
    if lower in _NONE_NODE_TYPES:
        return "NONE"
    return node_type


def _parser_language_for_chunk(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = _PARSER_LANGUAGE_ALIASES.get(language.lower(), language.lower())
    if normalized in _PARSER_LANGUAGES:
        return normalized
    return None


@lru_cache(maxsize=None)
def _parser_for_language(language: str) -> Any | None:
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None

    try:
        return get_parser(language)
    except Exception:
        return None


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
