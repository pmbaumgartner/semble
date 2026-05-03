import pytest

import semble.duplicates as duplicates
from semble import DuplicateResult, DuplicateSignals
from semble.duplicates import (
    _jaccard,
    _normalize_ast_label,
    _pair_key,
    _parser_language_for_chunk,
    _same_file_ranges_overlap,
    _weighted_structural_score,
    duplicate_score,
    score_duplicate_pair,
)
from semble.types import Chunk
from tests.conftest import make_chunk


def test_duplicate_types_are_exported() -> None:
    """Duplicate result types are part of the package-level API."""
    left = make_chunk("def left():\n    return 1", "left.py")
    right = make_chunk("def right():\n    return 1", "right.py")
    signals = DuplicateSignals(semantic_score=0.9, structural_score=0.8, token_jaccard=0.8)
    result = DuplicateResult(left=left, right=right, score=0.84, signals=signals)

    assert result.left is left
    assert result.right is right
    assert result.signals is signals


def test_score_duplicate_pair_ranks_renamed_code_above_unrelated_code() -> None:
    """Normalized token structure makes renamed similar code score above unrelated code."""
    left = make_chunk(
        """\
def total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
        "prices.py",
    )
    renamed = make_chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "invoices.py",
    )
    unrelated = make_chunk(
        """\
class Renderer:
    def render(self, page):
        print(page.title)
""",
        "render.py",
    )

    similar_signals = score_duplicate_pair(left, renamed, semantic_score=0.9)
    unrelated_signals = score_duplicate_pair(left, unrelated, semantic_score=0.9)

    assert similar_signals.token_jaccard > unrelated_signals.token_jaccard
    assert similar_signals.structural_score > unrelated_signals.structural_score
    assert similar_signals.token_jaccard > 0.5


def test_empty_or_comment_only_fingerprints_are_not_perfect_matches() -> None:
    """Empty structural fingerprints score zero instead of perfect similarity."""
    left = make_chunk("# only a comment\n# another comment", "left.py")
    right = make_chunk("// only a comment\n-- another comment", "right.py")
    signals = score_duplicate_pair(left, right, semantic_score=1.0)

    assert _jaccard(set(), set()) == 0.0
    assert signals.token_jaccard == 0.0
    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == 0.0
    assert duplicate_score(signals) == 0.0


def test_score_duplicate_pair_populates_ast_signals_when_parser_is_available() -> None:
    """Parser-backed chunks include AST type and shape signals."""
    if duplicates._parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    left = make_chunk(
        """\
def total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
        "prices.py",
    )
    right = make_chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "invoices.py",
    )

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is not None
    assert signals.ast_shape_jaccard is not None
    assert 0.0 <= signals.ast_type_jaccard <= 1.0
    assert 0.0 <= signals.ast_shape_jaccard <= 1.0


def test_structural_score_uses_weighted_ast_blend_when_available() -> None:
    """Structural score blends token, AST type, and AST shape signals."""
    if duplicates._parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    left = make_chunk(
        """\
def total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
        "prices.py",
    )
    right = make_chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "invoices.py",
    )

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is not None
    assert signals.ast_shape_jaccard is not None
    assert signals.structural_score == pytest.approx(
        _weighted_structural_score(
            token_jaccard=signals.token_jaccard,
            ast_type_jaccard=signals.ast_type_jaccard,
            ast_shape_jaccard=signals.ast_shape_jaccard,
        )
    )


def test_score_duplicate_pair_falls_back_when_parser_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing parser support leaves AST signals unavailable and keeps token scoring."""
    monkeypatch.setattr(duplicates, "_parser_for_language", lambda language: None)
    left = make_chunk("def add(a, b):\n    return a + b", "left.py")
    right = make_chunk("def plus(x, y):\n    return x + y", "right.py")

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == signals.token_jaccard


def test_score_duplicate_pair_falls_back_for_unsupported_language() -> None:
    """Unsupported languages use token-only duplicate scoring."""
    left = Chunk("def add(a, b):\n    return a + b", "left.txt", 1, 2, "text")
    right = Chunk("def plus(x, y):\n    return x + y", "right.txt", 1, 2, "text")

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert _parser_language_for_chunk("text") is None
    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == signals.token_jaccard


@pytest.mark.parametrize(
    ("node_type", "expected"),
    [
        ("identifier", "IDENT"),
        ("property_identifier", "IDENT"),
        ("number_literal", "NUMBER"),
        ("decimal_integer_literal", "NUMBER"),
        ("string_literal", "STRING"),
        ("raw_string_literal", "STRING"),
        ("true", "BOOL"),
        ("false_literal", "BOOL"),
        ("null_literal", "NULL"),
        ("nil", "NULL"),
        ("none", "NONE"),
        ("function_definition", "function_definition"),
    ],
)
def test_normalize_ast_label(node_type: str, expected: str) -> None:
    """AST label normalization keeps structural categories language-neutral."""
    assert _normalize_ast_label(node_type) == expected


@pytest.mark.parametrize(
    "signals",
    [
        DuplicateSignals(semantic_score=0.0, structural_score=0.8, token_jaccard=0.8),
        DuplicateSignals(semantic_score=-0.1, structural_score=0.8, token_jaccard=0.8),
        DuplicateSignals(semantic_score=0.8, structural_score=0.0, token_jaccard=0.0),
        DuplicateSignals(semantic_score=0.8, structural_score=-0.1, token_jaccard=-0.1),
    ],
)
def test_duplicate_score_requires_positive_semantic_and_structural_scores(signals: DuplicateSignals) -> None:
    """A pair needs both semantic and structural support to receive a ranking score."""
    assert duplicate_score(signals) == 0.0


def test_duplicate_score_uses_documented_blend() -> None:
    """The final ranking score follows the local semantic/structural blend."""
    signals = DuplicateSignals(semantic_score=0.81, structural_score=0.64, token_jaccard=0.64)

    assert duplicate_score(signals) == pytest.approx(0.81**0.4 * 0.64**0.6)


def test_same_file_ranges_overlap() -> None:
    """Same-file overlapping ranges are detected with inclusive line ranges."""
    left = Chunk("left", "src/a.py", 1, 10, "python")
    overlap = Chunk("overlap", "src/a.py", 10, 20, "python")
    separate = Chunk("separate", "src/a.py", 11, 20, "python")
    other_file = Chunk("other", "src/b.py", 5, 8, "python")

    assert _same_file_ranges_overlap(left, overlap)
    assert not _same_file_ranges_overlap(left, separate)
    assert not _same_file_ranges_overlap(left, other_file)


def test_pair_key_is_stable_for_reversed_pairs() -> None:
    """Pair keys are unordered so candidate generation can deduplicate A/B and B/A."""
    left = Chunk("left", "src/b.py", 5, 8, "python")
    right = Chunk("right", "src/a.py", 1, 4, "python")

    assert _pair_key(left, right) == _pair_key(right, left)
    assert _pair_key(left, right)[0] == ("src/a.py", 1, 4)
