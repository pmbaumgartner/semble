from semble.types import DuplicateSignals
from semble.utils import format_duplicate_clusters
from tests.conftest import make_chunk, make_duplicate_cluster, make_duplicate_pair


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
