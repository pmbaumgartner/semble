import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from vicinity.backends.basic import BasicArgs

from semble import SembleIndex
from semble.chunking.core import get_parser_for_language
from semble.duplicates.scoring import DuplicateFeatures, duplicate_features
from semble.index.dense import SelectableBasicBackend
from semble.types import Chunk, DuplicateCluster, DuplicatePair, DuplicateSignals


def make_chunk(
    content: str,
    file_path: str = "src/module.py",
    *,
    start_line: int = 1,
    end_line: int | None = None,
    language: str | None = "python",
) -> Chunk:
    """Create a minimal Chunk for use in tests."""
    return Chunk(
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line if end_line is not None else start_line + content.count("\n"),
        language=language,
    )


def make_duplicate_pair(
    *,
    left: Chunk | None = None,
    right: Chunk | None = None,
    left_content: str = "def left():\n    return 1",
    right_content: str = "def right():\n    return 1",
    left_path: str = "src/left.py",
    right_path: str = "src/right.py",
    score: float = 0.84,
    semantic_score: float = 0.9,
    structural_score: float = 0.8,
    token_jaccard: float = 0.7,
    ast_type_jaccard: float | None = None,
    ast_shape_jaccard: float | None = None,
    left_match_content: str | None = None,
    right_match_content: str | None = None,
) -> DuplicatePair:
    """Create a representative DuplicatePair for tests."""
    left = left if left is not None else make_chunk(left_content, left_path)
    right = right if right is not None else make_chunk(right_content, right_path)
    return DuplicatePair(
        left=left,
        right=right,
        score=score,
        signals=DuplicateSignals(
            semantic_score=semantic_score,
            structural_score=structural_score,
            token_jaccard=token_jaccard,
            ast_type_jaccard=ast_type_jaccard,
            ast_shape_jaccard=ast_shape_jaccard,
        ),
        left_content=left.content if left_match_content is None else left_match_content,
        right_content=right.content if right_match_content is None else right_match_content,
    )


def make_duplicate_cluster(
    pairs: Sequence[DuplicatePair] | None = None,
    *,
    extra_pair: bool = False,
) -> DuplicateCluster:
    """Create a DuplicateCluster from ordered pair members."""
    pair_list = list(pairs) if pairs is not None else [make_duplicate_pair()]
    if extra_pair:
        result = pair_list[0]
        third = make_chunk("def third():\n    return 1", "src/third.py")
        pair_list.append(
            DuplicatePair(
                left=result.right,
                right=third,
                score=0.75,
                signals=result.signals,
                left_content=result.right_content,
                right_content=third.content,
            )
        )

    members = []
    for pair in pair_list:
        for chunk in (pair.left, pair.right):
            if chunk not in members:
                members.append(chunk)
    return DuplicateCluster(members=tuple(members), pairs=tuple(pair_list))


class _ConstantModel:
    """Test encoder that makes every semantic query equally similar."""

    def encode(self, texts: Sequence[str], /) -> npt.NDArray[np.float32]:
        """Return a unit vector for each text."""
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def make_duplicate_index(chunks: Sequence[Chunk]) -> SembleIndex:
    """Build a lightweight duplicate-test index without loading a real model."""
    chunk_list = list(chunks)
    if chunk_list:
        embeddings = np.zeros((len(chunk_list), 4), dtype=np.float32)
        embeddings[:, 0] = 1.0
        semantic_index = SelectableBasicBackend(embeddings, BasicArgs())
    else:
        semantic_index = MagicMock()
    return SembleIndex(_ConstantModel(), MagicMock(), semantic_index, chunk_list)


@pytest.fixture
def duplicate_index_factory() -> Callable[[Sequence[Chunk]], SembleIndex]:
    """Return a lightweight duplicate-test index factory."""
    return make_duplicate_index


def require_parsers(*languages: str) -> None:
    """Skip the current test unless all requested tree-sitter parsers are available."""
    if any(get_parser_for_language(language) is None for language in languages):
        pytest.skip("tree_sitter_language_pack is not available")


def require_duplicate_features(
    content: str,
    path: str,
    *,
    language: str | None = "python",
) -> DuplicateFeatures:
    """Return duplicate features, skipping if parser-backed details are unavailable."""
    features = duplicate_features(make_chunk(content, path, language=language))
    if features.code_bearing_node_count is None:
        pytest.skip("tree_sitter_language_pack is not available")
    return features


@pytest.fixture
def tmp_py_file(tmp_path: Path) -> Path:
    """A simple Python file with two functions."""
    code = textwrap.dedent(
        """\
        def add(a, b):
            \"\"\"Add two numbers.\"\"\"
            return a + b

        def subtract(a, b):
            return a - b

        X = 42
        """
    )
    f = tmp_path / "math_utils.py"
    f.write_text(code)
    return f


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A small project with a few Python files."""
    (tmp_path / "auth.py").write_text(
        textwrap.dedent(
            """\
            def authenticate(token):
                \"\"\"Verify an auth token.\"\"\"
                return token == "secret"

            def login(username, password):
                return authenticate(password)
            """
        )
    )
    (tmp_path / "utils.py").write_text(
        textwrap.dedent(
            """\
            def format_name(first, last):
                return f"{first} {last}"

            class Config:
                debug = False
                host = "localhost"
            """
        )
    )
    (tmp_path / "README.md").write_text("# Test project\n")
    return tmp_path


@pytest.fixture
def mock_model() -> MagicMock:
    """A model stub that returns deterministic random embeddings."""
    model = MagicMock()
    rng = np.random.default_rng(42)
    _dim = 256

    def _encode(texts: list[str], **kwargs: Any) -> npt.NDArray[np.float32]:
        embs = rng.standard_normal((len(texts), _dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        normalized: npt.NDArray[np.float32] = embs / (norms + 1e-8)
        return normalized

    model.encode.side_effect = _encode
    model.dim = _dim
    return model
