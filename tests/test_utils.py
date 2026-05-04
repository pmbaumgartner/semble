from semble.types import DuplicateCluster, DuplicateResult, DuplicateSignals
from semble.utils import _format_duplicate_clusters
from tests.conftest import make_chunk


def _duplicate_result(*, ast_signals: bool = False) -> DuplicateResult:
    left = make_chunk("def left():\n    return 1", "src/left.py")
    right = make_chunk("def right():\n    return 1", "src/right.py")
    signals = DuplicateSignals(
        semantic_score=0.9,
        structural_score=0.8,
        token_jaccard=0.7,
        ast_type_jaccard=0.6 if ast_signals else None,
        ast_shape_jaccard=0.5 if ast_signals else None,
    )
    return DuplicateResult(left=left, right=right, score=0.84, signals=signals)


def _duplicate_cluster(*, extra_pair: bool = False) -> DuplicateCluster:
    result = _duplicate_result()
    pairs = (result,)
    members = (result.left, result.right)
    if extra_pair:
        third = make_chunk("def third():\n    return 1", "src/third.py")
        extra = DuplicateResult(left=result.right, right=third, score=0.75, signals=result.signals)
        pairs = (result, extra)
        members = (result.left, result.right, third)
    return DuplicateCluster(members=members, pairs=pairs)


def test_format_duplicate_clusters_empty() -> None:
    """Empty duplicate clusters render only the header."""
    assert _format_duplicate_clusters("Duplicate clusters", []) == "Duplicate clusters\n"


def test_format_duplicate_clusters_summary_members_and_strongest_pair() -> None:
    """Cluster formatting includes summary metadata, members, and strongest pair snippets."""
    out = _format_duplicate_clusters("Duplicate clusters", [_duplicate_cluster()])

    assert "Duplicate clusters" in out
    assert "score=0.840, members=2, pairs=1" in out
    assert "- src/left.py:1-2" in out
    assert "- src/right.py:1-2" in out
    assert "Strongest pair: src/left.py:1-2 <-> src/right.py:1-2" in out
    assert "semantic=0.900 structural=0.800 tokens=0.700" in out
    assert "def left():" in out
    assert "def right():" in out
    assert out.count("```") == 4


def test_format_duplicate_clusters_lists_additional_pairs_compactly() -> None:
    """Additional cluster pairs are listed without extra full snippets."""
    out = _format_duplicate_clusters("Duplicate clusters", [_duplicate_cluster(extra_pair=True)])

    assert "Additional pairs:" in out
    assert "- src/right.py:1-2 <-> src/third.py:1-2  [score=0.750]" in out
    assert "def third():" not in out
