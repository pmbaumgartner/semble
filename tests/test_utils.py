from semble.types import Chunk, DuplicateCluster, DuplicatePair, DuplicateSignals
from semble.utils import _format_duplicate_clusters
from tests.conftest import make_chunk


def _duplicate_result(*, ast_signals: bool = False) -> DuplicatePair:
    left = make_chunk("def left():\n    return 1", "src/left.py")
    right = make_chunk("def right():\n    return 1", "src/right.py")
    signals = DuplicateSignals(
        semantic_score=0.9,
        structural_score=0.8,
        token_jaccard=0.7,
        ast_type_jaccard=0.6 if ast_signals else None,
        ast_shape_jaccard=0.5 if ast_signals else None,
    )
    return DuplicatePair(
        left=left,
        right=right,
        score=0.84,
        signals=signals,
        left_content=left.content,
        right_content=right.content,
    )


def _duplicate_cluster(*, extra_pair: bool = False) -> DuplicateCluster:
    result = _duplicate_result()
    pairs: tuple[DuplicatePair, ...] = (result,)
    members: tuple[Chunk, ...] = (result.left, result.right)
    if extra_pair:
        third = make_chunk("def third():\n    return 1", "src/third.py")
        extra = DuplicatePair(
            left=result.right,
            right=third,
            score=0.75,
            signals=result.signals,
            left_content=result.right_content,
            right_content=third.content,
        )
        pairs = (result, extra)
        members = (result.left, result.right, third)
    return DuplicateCluster(members=members, pairs=pairs)


def _multi_pair_cluster(pair_count: int) -> DuplicateCluster:
    chunks = tuple(make_chunk(f"def item_{i}():\n    return {i}", f"src/item{i}.py") for i in range(pair_count + 1))
    signals = DuplicateSignals(semantic_score=0.9, structural_score=0.8, token_jaccard=0.7)
    pairs = tuple(
        DuplicatePair(
            left=chunks[i],
            right=chunks[i + 1],
            score=0.9 - i * 0.01,
            signals=signals,
            left_content=chunks[i].content,
            right_content=chunks[i + 1].content,
        )
        for i in range(pair_count)
    )
    return DuplicateCluster(members=chunks, pairs=pairs)


def test_format_duplicate_clusters_empty() -> None:
    """Empty duplicate clusters render only the header."""
    assert _format_duplicate_clusters("Duplicate clusters", []) == "Duplicate clusters\n"


def test_format_duplicate_clusters_summary_members_and_strongest_pair() -> None:
    """Cluster formatting includes summary metadata, members, and strongest pair snippets."""
    out = _format_duplicate_clusters("Duplicate clusters", [_duplicate_cluster()])

    assert "Duplicate clusters" in out
    assert "Signals are 0..1 similarities" in out
    assert "semantic=embedding; structural=weighted tokens/AST blend" in out
    assert "score=0.840, members=2, pairs=1" in out
    assert "- src/left.py:1-2" in out
    assert "- src/right.py:1-2" in out
    assert "Top pairs:" in out
    assert "- src/left.py:1-2 <-> src/right.py:1-2  [score=0.840 semantic=0.900 structural=0.800 tokens=0.700]" in out
    assert "def left():" in out
    assert "def right():" in out
    assert out.count("```") == 4


def test_format_duplicate_clusters_lists_top_pairs_compactly() -> None:
    """Cluster pairs are listed with score signals and without extra full snippets."""
    out = _format_duplicate_clusters("Duplicate clusters", [_duplicate_cluster(extra_pair=True)])

    assert "Top pairs:" in out
    assert "- src/right.py:1-2 <-> src/third.py:1-2  [score=0.750 semantic=0.900" in out
    assert "def third():" not in out


def test_format_duplicate_clusters_caps_top_pairs() -> None:
    """Large clusters show only the strongest five pairs and a hidden-pair count."""
    out = _format_duplicate_clusters("Duplicate clusters", [_multi_pair_cluster(6)])

    assert sum(line.startswith("- src/item") and "  [score=" in line for line in out.splitlines()) == 5
    assert "Pairs not shown: 1" in out
    assert "- src/item5.py:1-2 <-> src/item6.py:1-2" not in out


def test_format_duplicate_clusters_preserves_first_line_indentation() -> None:
    """Fenced duplicate snippets keep leading indentation on the first line."""
    left = make_chunk("    def left():\n        return 1\n", "src/left.py")
    right = make_chunk("    def right():\n        return 1\n", "src/right.py")
    signals = DuplicateSignals(semantic_score=0.9, structural_score=0.8, token_jaccard=0.7)
    pair = DuplicatePair(
        left=left,
        right=right,
        score=0.84,
        signals=signals,
        left_content=left.content,
        right_content=right.content,
    )
    out = _format_duplicate_clusters("Duplicate clusters", [DuplicateCluster(members=(left, right), pairs=(pair,))])

    assert "\n```\n    def left():\n        return 1\n```\n" in out
    assert "\n```\n    def right():\n        return 1\n```\n" in out


def test_format_duplicate_clusters_uses_match_content() -> None:
    """Duplicate snippets render scored match content, while members keep original chunks."""
    left = make_chunk("import os\n\ndef left():\n    return os.getcwd()", "src/left.py")
    right = make_chunk("import os\n\ndef right():\n    return os.getcwd()", "src/right.py")
    signals = DuplicateSignals(semantic_score=0.9, structural_score=0.8, token_jaccard=0.7)
    pair = DuplicatePair(
        left=left,
        right=right,
        score=0.84,
        signals=signals,
        left_content="def left():\n    return os.getcwd()",
        right_content="def right():\n    return os.getcwd()",
    )
    out = _format_duplicate_clusters("Duplicate clusters", [DuplicateCluster(members=(left, right), pairs=(pair,))])

    assert "- src/left.py:1-4" in out
    assert "import os" not in out
    assert "def left():" in out
    assert "def right():" in out
