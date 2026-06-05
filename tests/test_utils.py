from semble.types import DuplicateCluster, DuplicateSignals
from semble.utils import format_duplicate_clusters, format_duplicate_clusters_compact
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


def test_duplicate_signals_to_dict() -> None:
    """DuplicateSignals serializes all ranking signals."""
    signals = DuplicateSignals(
        semantic_score=0.9,
        structural_score=0.8,
        token_jaccard=0.7,
        ast_type_jaccard=0.6,
        ast_shape_jaccard=0.5,
    )

    assert signals.to_dict() == {
        "semantic_score": 0.9,
        "structural_score": 0.8,
        "token_jaccard": 0.7,
        "ast_type_jaccard": 0.6,
        "ast_shape_jaccard": 0.5,
    }


def test_duplicate_pair_to_dict() -> None:
    """DuplicatePair serializes chunks, signals, score, and scored content."""
    pair = make_duplicate_pair(ast_type_jaccard=0.6, ast_shape_jaccard=0.5)

    out = pair.to_dict()

    assert out["left"]["location"] == "src/left.py:1-2"
    assert out["right"]["location"] == "src/right.py:1-2"
    assert out["score"] == 0.84
    assert out["signals"]["semantic_score"] == 0.9
    assert out["signals"]["ast_shape_jaccard"] == 0.5
    assert out["left_content"] == "def left():\n    return 1"
    assert out["right_content"] == "def right():\n    return 1"


def test_duplicate_cluster_to_dict_uses_match_content() -> None:
    """DuplicateCluster serializes summary score, members, pairs, and stripped match content."""
    left = make_chunk("import os\n\ndef left():\n    return os.getcwd()", "src/left.py")
    right = make_chunk("import os\n\ndef right():\n    return os.getcwd()", "src/right.py")
    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.84,
        left_match_content="def left():\n    return os.getcwd()",
        right_match_content="def right():\n    return os.getcwd()",
    )

    out = make_duplicate_cluster([pair]).to_dict()

    assert out["score"] == 0.84
    assert [member["location"] for member in out["members"]] == ["src/left.py:1-4", "src/right.py:1-4"]
    assert out["pairs"][0]["left_content"] == "def left():\n    return os.getcwd()"
    assert out["pairs"][0]["right_content"] == "def right():\n    return os.getcwd()"


def test_format_duplicate_clusters_returns_jsonable_shape() -> None:
    """Duplicate cluster formatting matches search/find-related JSON shape."""
    out = format_duplicate_clusters("Duplicate clusters", [make_duplicate_cluster()])

    assert out["query"] == "Duplicate clusters"
    assert len(out["clusters"]) == 1
    assert out["clusters"][0]["score"] == 0.84
    assert out["clusters"][0]["pairs"][0]["signals"]["token_jaccard"] == 0.7


def test_format_duplicate_clusters_allows_empty_clusters() -> None:
    """The formatter is pure; callers decide whether to replace empty clusters with an error."""
    assert format_duplicate_clusters("Duplicate clusters", []) == {"query": "Duplicate clusters", "clusters": []}


def test_format_duplicate_clusters_compact_summarizes_members_and_pairs() -> None:
    """Compact duplicate JSON keeps the full schema shape while trimming later pair content."""
    left = make_chunk("import os\n\ndef left():\n    return os.getcwd()", "src/left.py")
    right = make_chunk("import os\n\ndef right():\n    return os.getcwd()", "src/right.py")
    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.84,
        left_match_content="def left():\n    return os.getcwd()",
        right_match_content="def right():\n    return os.getcwd()",
        ast_type_jaccard=0.6,
    )

    out = format_duplicate_clusters_compact("Duplicate clusters", [make_duplicate_cluster([pair])])
    cluster = out["clusters"][0]

    assert out["detail"] == "compact"
    assert cluster["score"] == 0.84
    assert cluster["members"] == [
        {"location": "src/left.py:1-4", "file_path": "src/left.py", "start_line": 1, "end_line": 4},
        {"location": "src/right.py:1-4", "file_path": "src/right.py", "start_line": 1, "end_line": 4},
    ]
    assert cluster["pairs"][0]["left"]["location"] == "src/left.py:1-4"
    assert cluster["pairs"][0]["signals"]["ast_type_jaccard"] == 0.6
    assert cluster["pairs_not_shown"] == 0
    assert cluster["pairs"][0]["left_content"] == "def left():\n    return os.getcwd()"
    assert cluster["pairs"][0]["right_content"] == "def right():\n    return os.getcwd()"
    assert "content" not in cluster["members"][0]


def test_format_duplicate_clusters_compact_caps_top_pairs() -> None:
    """Compact duplicate JSON lists the strongest five pairs and counts the rest."""
    out = format_duplicate_clusters_compact("Duplicate clusters", [_multi_pair_cluster(6)])
    cluster = out["clusters"][0]

    assert len(cluster["pairs"]) == 5
    assert cluster["pairs_not_shown"] == 1
    assert cluster["pairs"][-1]["left"]["location"] == "src/item4.py:1-2"
    assert "left_content" in cluster["pairs"][0]
    assert "right_content" in cluster["pairs"][0]
    assert "left_content" not in cluster["pairs"][1]
    assert "right_content" not in cluster["pairs"][1]
