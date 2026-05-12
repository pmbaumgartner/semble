from semble.types import DuplicateCluster
from semble.utils import _format_duplicate_clusters
from tests.conftest import make_chunk, make_duplicate_cluster, make_duplicate_pair


def _multi_pair_cluster(pair_count: int) -> DuplicateCluster:
    chunks = tuple(make_chunk(f"def item_{i}():\n    return {i}", f"src/item{i}.py") for i in range(pair_count + 1))
    pairs = tuple(
        make_duplicate_pair(
            left=chunks[i],
            right=chunks[i + 1],
            score=0.9 - i * 0.01,
            semantic_score=0.9,
            structural_score=0.8,
            token_jaccard=0.7,
        )
        for i in range(pair_count)
    )
    return make_duplicate_cluster(pairs)


def test_format_duplicate_clusters_empty() -> None:
    """Empty duplicate clusters render only the header."""
    assert _format_duplicate_clusters("Duplicate clusters", []) == "Duplicate clusters\n"


def test_format_duplicate_clusters_summary_members_and_strongest_pair() -> None:
    """Cluster formatting includes summary metadata, members, and strongest pair snippets."""
    out = _format_duplicate_clusters("Duplicate clusters", [make_duplicate_cluster()])

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
    out = _format_duplicate_clusters("Duplicate clusters", [make_duplicate_cluster(extra_pair=True)])

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
    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.84,
        semantic_score=0.9,
        structural_score=0.8,
        token_jaccard=0.7,
    )
    out = _format_duplicate_clusters("Duplicate clusters", [make_duplicate_cluster([pair])])

    assert "\n```\n    def left():\n        return 1\n```\n" in out
    assert "\n```\n    def right():\n        return 1\n```\n" in out


def test_format_duplicate_clusters_uses_match_content() -> None:
    """Duplicate snippets render scored match content, while members keep original chunks."""
    left = make_chunk("import os\n\ndef left():\n    return os.getcwd()", "src/left.py")
    right = make_chunk("import os\n\ndef right():\n    return os.getcwd()", "src/right.py")
    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.84,
        left_match_content="def left():\n    return os.getcwd()",
        right_match_content="def right():\n    return os.getcwd()",
    )
    out = _format_duplicate_clusters("Duplicate clusters", [make_duplicate_cluster([pair])])

    assert "- src/left.py:1-4" in out
    assert "import os" not in out
    assert "def left():" in out
    assert "def right():" in out
