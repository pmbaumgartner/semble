import pytest

from semble import DuplicateResult, DuplicateSignals
from semble.duplicates import (
    _jaccard,
    _pair_key,
    _same_file_ranges_overlap,
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
    assert signals.structural_score == 0.0
    assert duplicate_score(signals) == 0.0


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
