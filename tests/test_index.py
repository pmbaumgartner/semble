from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from vicinity.backends.basic import BasicArgs

from semble import SembleIndex
from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend
from semble.types import Chunk, DuplicateResult, Encoder


@pytest.fixture
def indexed_index(mock_model: Any, tmp_project: Path) -> SembleIndex:
    """SembleIndex built from tmp_project."""
    return SembleIndex.from_path(tmp_project, model=mock_model)


class _ConstantModel:
    """Test encoder that makes every semantic query equally similar."""

    def encode(self, texts: list[str], /) -> npt.NDArray[np.float32]:
        """Return a unit vector for each text."""
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def _chunk(
    content: str,
    file_path: str,
    *,
    start_line: int = 1,
    language: str | None = "python",
) -> Chunk:
    """Create a Chunk with line numbers matching its content by default."""
    return Chunk(
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + len(content.splitlines()) - 1,
        language=language,
    )


def _duplicate_index(chunks: list[Chunk]) -> SembleIndex:
    """Build a lightweight duplicate-test index without loading a real model."""
    if chunks:
        embeddings = np.zeros((len(chunks), 4), dtype=np.float32)
        embeddings[:, 0] = 1.0
        semantic_index = SelectableBasicBackend(embeddings, BasicArgs())
    else:
        semantic_index = MagicMock()
    return SembleIndex(_ConstantModel(), MagicMock(), semantic_index, chunks)


@pytest.mark.parametrize(
    ("include_text_files", "md_in_results"),
    [(False, False), (True, True)],
)
def test_index_markdown_inclusion(
    mock_model: Encoder, tmp_project: Path, include_text_files: bool, md_in_results: bool
) -> None:
    """Markdown files are excluded by default and included when include_text_files=True."""
    _, _, chunks = create_index_from_path(tmp_project, mock_model, include_text_files=include_text_files)
    has_md = ".md" in {Path(c.file_path).suffix for c in chunks}
    assert has_md is md_in_results


def test_index_empty_returns_zero_chunks(mock_model: Encoder, tmp_path: Path) -> None:
    """Indexing an empty directory yields zero files and chunks."""
    with pytest.raises(ValueError):
        create_index_from_path(tmp_path, mock_model)


def test_index_language_counts(indexed_index: SembleIndex) -> None:
    """Language breakdown in stats includes python with at least one chunk."""
    stats = indexed_index.stats
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


@pytest.mark.parametrize(
    "query, mode",
    [("authenticate token", "hybrid"), ("authenticate", "bm25"), ("authentication", "semantic")],
)
def test_search_modes(indexed_index: SembleIndex, query: str, mode: str) -> None:
    """Each search mode returns a valid list of at most top_k results."""
    results = indexed_index.search(query, top_k=3, mode=mode)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_invalid_mode(indexed_index: SembleIndex) -> None:
    """An unrecognised mode string raises ValueError."""
    with pytest.raises(ValueError):
        indexed_index.search("query", mode="invalid")


def test_search_constraints(indexed_index: SembleIndex) -> None:
    """search: top_k is respected; no duplicate chunks are returned."""
    assert len(indexed_index.search("function", top_k=1, mode="bm25")) <= 1

    results = indexed_index.search("authenticate", top_k=5)
    assert len(results) == len(set(r.chunk for r in results))


@pytest.mark.parametrize("mode", ["bm25", "hybrid", "semantic"])
def test_search_with_filter_paths_does_not_crash(indexed_index: SembleIndex, mode: str) -> None:
    """Filtered search works regardless of where the selected chunk lives in the corpus."""
    target_path = indexed_index.chunks[-1].file_path
    results = indexed_index.search("function", top_k=3, mode=mode, filter_paths=[target_path])
    assert all(r.chunk.file_path == target_path for r in results)


@pytest.mark.parametrize("mode", ["bm25", "hybrid", "semantic"])
@pytest.mark.parametrize("query", ["", "   ", "\n\n"])
def test_search_empty_query_returns_empty(indexed_index: SembleIndex, mode: str, query: str) -> None:
    """Empty / whitespace-only queries return [] across all modes."""
    assert indexed_index.search(query, mode=mode) == []


def test_find_related(indexed_index: SembleIndex) -> None:
    """find_related returns related chunks for a Chunk or SearchResult seed."""
    chunk = indexed_index.chunks[0]
    via_chunk = indexed_index.find_related(chunk, top_k=3)
    assert isinstance(via_chunk, list)
    assert len(via_chunk) <= 3
    assert all(r.chunk != chunk for r in via_chunk)

    # SearchResult form returns the same results as Chunk form.
    result = indexed_index.search("authenticate", top_k=1)[0]
    assert [r.chunk for r in indexed_index.find_related(result, top_k=3)] == [
        r.chunk for r in indexed_index.find_related(result.chunk, top_k=3)
    ]


def test_find_duplicates_returns_ranked_pairs_and_respects_top_k() -> None:
    """find_duplicates returns the strongest duplicate pair up to top_k."""
    left = _chunk(
        """\
def total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
        "src/prices.py",
    )
    renamed = _chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "src/invoices.py",
    )
    unrelated = _chunk(
        """\
class Renderer:
    def render(self, page):
        print(page.title)
""",
        "src/render.py",
    )
    index = _duplicate_index([left, renamed, unrelated])

    results = index.find_duplicates(top_k=1, min_lines=1)

    assert len(results) == 1
    assert isinstance(results[0], DuplicateResult)
    assert {results[0].left.file_path, results[0].right.file_path} == {"src/prices.py", "src/invoices.py"}


def test_find_duplicates_excludes_overlapping_same_file_ranges() -> None:
    """Same-file overlapping chunks are not returned as duplicate pairs."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/prices.py", start_line=1),
            _chunk(content, "src/prices.py", start_line=3),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []


def test_find_duplicates_deduplicates_reversed_pairs_deterministically() -> None:
    """A/B and B/A semantic candidates collapse into one stable pair."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/b.py"),
            _chunk(content, "src/a.py"),
        ]
    )

    results = index.find_duplicates(top_k=10, min_lines=1)

    assert len(results) == 1
    assert results[0].left.file_path == "src/a.py"
    assert results[0].right.file_path == "src/b.py"


def test_find_duplicates_filters_path_scopes() -> None:
    """Include and exclude path scopes match exact files and directory prefixes."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py"),
            _chunk(content, "src/nested/b.py"),
            _chunk(content, "src/generated/c.py"),
            _chunk(content, "tests/a_test.py"),
        ]
    )

    results = index.find_duplicates(
        top_k=10,
        min_lines=1,
        filter_paths=["./src/"],
        exclude_paths=["src/generated/"],
    )
    result_paths = {results[0].left.file_path, results[0].right.file_path}

    assert len(results) == 1
    assert result_paths == {"src/a.py", "src/nested/b.py"}
    assert index.find_duplicates(min_lines=1, filter_paths=["src/a.py"]) == []


def test_find_duplicates_intersects_language_and_path_filters() -> None:
    """Duplicate filters use intersection semantics instead of search's union selector."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py", language="python"),
            _chunk(content, "web/a.js", language="javascript"),
        ]
    )

    assert (
        index.find_duplicates(
            min_lines=1,
            filter_languages=["python"],
            filter_paths=["web"],
        )
        == []
    )


def test_find_duplicates_respects_language_filter() -> None:
    """Language filters restrict both sides of returned duplicate pairs."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py", language="python"),
            _chunk(content, "src/b.py", language="python"),
            _chunk(content, "src/a.js", language="javascript"),
        ]
    )

    results = index.find_duplicates(top_k=10, min_lines=1, filter_languages=["python"])

    assert len(results) == 1
    assert results[0].left.language == "python"
    assert results[0].right.language == "python"


def test_find_duplicates_respects_min_lines_and_min_score() -> None:
    """Minimum line and score thresholds filter duplicate candidates."""
    content = """\
def add(a, b):
    return a + b
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py"),
            _chunk(content, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=3) == []
    assert len(index.find_duplicates(min_lines=1, min_score=1.0)) == 1
    assert index.find_duplicates(min_lines=1, min_score=1.01) == []


def test_find_duplicates_excludes_cross_language_pairs() -> None:
    """Duplicate discovery only compares chunks with the same language."""
    content = """\
def add(a, b):
    return a + b
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py", language="python"),
            _chunk(content, "src/a.js", language="javascript"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []


def test_find_duplicates_allows_unknown_language_pairs() -> None:
    """Unknown-language chunks compare only with other unknown-language chunks."""
    content = """\
def add(a, b):
    return a + b
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.txt", language=None),
            _chunk(content, "src/b.txt", language=None),
            _chunk(content, "src/c.py", language="python"),
        ]
    )

    results = index.find_duplicates(min_lines=1)

    assert len(results) == 1
    assert results[0].left.language is None
    assert results[0].right.language is None


def test_find_duplicates_empty_or_non_positive_top_k_returns_empty() -> None:
    """Empty indexes, singletons, and non-positive top_k return no duplicate pairs."""
    assert _duplicate_index([]).find_duplicates() == []
    assert _duplicate_index([_chunk("def add(a, b):\n    return a + b", "src/a.py")]).find_duplicates(min_lines=1) == []

    index = _duplicate_index(
        [
            _chunk("def add(a, b):\n    return a + b", "src/a.py"),
            _chunk("def add(a, b):\n    return a + b", "src/b.py"),
        ]
    )
    assert index.find_duplicates(top_k=0, min_lines=1) == []
