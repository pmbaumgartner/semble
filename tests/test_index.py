from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from vicinity.backends.basic import BasicArgs

import semble.index.index as index_module
from semble import SembleIndex
from semble.duplicates import duplicate_features
from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend
from semble.types import Chunk, DuplicateCluster, DuplicateResult, DuplicateSignals, Encoder


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


def _cluster_paths(cluster: DuplicateCluster) -> set[str]:
    return {member.file_path for member in cluster.members}


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


def test_index_filters_duplicate_discovery_paths_up_front(mock_model: Encoder, tmp_path: Path) -> None:
    """Index creation can skip duplicate-only path scopes before chunk embedding."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "keep.py").write_text("def keep():\n    return 1\n")
    (tmp_path / "src" / "generated.py").write_text("def generated():\n    return 2\n")
    (tmp_path / "tests" / "test_keep.py").write_text("def test_keep():\n    assert True\n")

    _, _, chunks = create_index_from_path(
        tmp_path,
        mock_model,
        include_paths=["src"],
        exclude_paths=["src/generated.py"],
        include_tests=False,
        display_root=tmp_path,
    )

    assert {chunk.file_path for chunk in chunks} == {"src/keep.py"}


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


def test_find_duplicates_returns_ranked_clusters_and_respects_top_k() -> None:
    """find_duplicates returns the strongest duplicate cluster up to top_k."""
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
    assert isinstance(results[0], DuplicateCluster)
    assert _cluster_paths(results[0]) == {"src/prices.py", "src/invoices.py"}
    assert isinstance(results[0].pairs[0], DuplicateResult)


def test_find_duplicate_pairs_helper_returns_unsliced_sorted_pairs() -> None:
    """The private pair scan returns all sorted pairs for later clustering."""
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
            _chunk(content, "src/b.py"),
            _chunk(content, "src/c.py"),
        ]
    )

    all_pairs = index._find_duplicate_pairs(
        candidate_k=12,
        min_lines=1,
        min_score=0.0,
        min_structural_score=0.0,
        filter_languages=None,
        include_paths=None,
        exclude_paths=None,
        include_tests=False,
        include_data=False,
        include_scaffolding=False,
    )

    assert len(all_pairs) == 3
    clusters = index.find_duplicates(top_k=1, min_lines=1, min_structural_score=0.0)
    assert len(clusters) == 1
    assert clusters[0].pairs == tuple(all_pairs)
    assert all_pairs == sorted(all_pairs, key=index._duplicate_sort_key)


def test_find_duplicates_uses_existing_embeddings_without_reencoding() -> None:
    """Duplicate discovery reuses indexed embeddings instead of encoding each chunk again."""
    chunks = [
        _chunk("def add(a, b):\n    return a + b", "src/a.py"),
        _chunk("def plus(x, y):\n    return x + y", "src/b.py"),
    ]
    embeddings = np.zeros((len(chunks), 4), dtype=np.float32)
    embeddings[:, 0] = 1.0
    semantic_index = SelectableBasicBackend(embeddings, BasicArgs())
    model = MagicMock()
    index = SembleIndex(model, MagicMock(), semantic_index, chunks)

    assert index.find_duplicates(min_lines=1)
    model.encode.assert_not_called()


def test_find_duplicates_batches_neighbor_queries_by_language() -> None:
    """Semantic neighbors are queried in language groups while preserving candidate_k."""
    index = _duplicate_index(
        [
            _chunk("def add(a, b):\n    return a + b", "src/a.py", language="python"),
            _chunk("def plus(x, y):\n    return x + y", "src/b.py", language="python"),
            _chunk("function add(a, b) {\n  return a + b;\n}", "src/a.js", language="javascript"),
            _chunk("function plus(x, y) {\n  return x + y;\n}", "src/b.js", language="javascript"),
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        mock_query = MagicMock(wraps=index._semantic_index.query)
        monkeypatch.setattr(index._semantic_index, "query", mock_query)
        index.find_duplicates(candidate_k=1, min_lines=1, include_tests=True)

    assert mock_query.call_count == 2
    assert all(call.kwargs["k"] == 2 for call in mock_query.call_args_list)


def test_find_duplicates_precomputes_features_once_per_eligible_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate structural features are cached per chunk during a scan."""
    index = _duplicate_index(
        [
            _chunk("def add(a, b):\n    return a + b", "src/a.py"),
            _chunk("def plus(x, y):\n    return x + y", "src/b.py"),
            _chunk("def skip(a, b):\n    return a + b", "tests/test_skip.py"),
        ]
    )
    mock_features = MagicMock(wraps=duplicate_features)
    monkeypatch.setattr(index_module, "duplicate_features", mock_features)

    index.find_duplicates(min_lines=1)

    assert mock_features.call_count == 2


def test_find_duplicates_excludes_overlapping_same_file_ranges() -> None:
    """Same-file overlapping chunks are not returned as duplicate clusters."""
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
    assert results[0].pairs[0].left.file_path == "src/a.py"
    assert results[0].pairs[0].right.file_path == "src/b.py"


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
        include_paths=["./src/"],
        exclude_paths=["src/generated/"],
    )
    result_paths = _cluster_paths(results[0])

    assert len(results) == 1
    assert result_paths == {"src/a.py", "src/nested/b.py"}
    assert index.find_duplicates(min_lines=1, include_paths=["src/a.py"]) == []


def test_find_duplicates_excludes_tests_by_default() -> None:
    """Duplicate discovery skips test-looking paths unless include_tests=True."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "tests/test_a.py"),
            _chunk(content, "tests/test_b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, include_tests=True)) == 1


def test_find_duplicates_excludes_test_filename_patterns_by_default() -> None:
    """Test-looking filenames outside test dirs are skipped by default."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/prices.test.py"),
            _chunk(content, "src/invoices.spec.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, include_tests=True)) == 1


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
            include_paths=["web"],
        )
        == []
    )


def test_find_duplicates_respects_language_filter() -> None:
    """Language filters restrict all returned duplicate cluster members."""
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
    assert {member.language for member in results[0].members} == {"python"}


def test_find_duplicates_respects_min_lines_and_min_score() -> None:
    """Minimum line, score, and cluster-size thresholds filter duplicate clusters."""
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
    assert index.find_duplicates(min_lines=1, min_cluster_size=3) == []


def test_find_duplicates_respects_default_structural_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weak structural pairs are filtered by default and can be included by lowering the floor."""
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
    monkeypatch.setattr(
        index_module,
        "score_duplicate_features",
        lambda *args, **kwargs: DuplicateSignals(
            semantic_score=0.9,
            structural_score=0.39,
            token_jaccard=0.39,
        ),
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, min_structural_score=0.39)) == 1


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
    assert {member.language for member in results[0].members} == {None}


def test_find_duplicates_excludes_docstring_only_chunks_when_parser_is_available() -> None:
    """Parser-supported chunks with only string/comment AST subtrees are not duplicate candidates."""
    if duplicate_features(_chunk("def real():\n    return 1", "src/real.py")).code_bearing_node_count is None:
        pytest.skip("tree_sitter_language_pack is not available")

    index = _duplicate_index(
        [
            _chunk('"""Repeated documentation."""', "src/a.py"),
            _chunk('"""Repeated documentation."""', "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []


def test_find_duplicates_excludes_static_data_chunks_by_default() -> None:
    """Static data/config chunks are skipped unless include_data=True."""
    if duplicate_features(_chunk("[1, 2, 3]", "src/data.py")).code_bearing_node_count is None:
        pytest.skip("tree_sitter_language_pack is not available")

    content = """\
VALUES = [
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
]
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py"),
            _chunk(content, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert index.find_duplicates(min_lines=1, include_scaffolding=True) == []
    assert len(index.find_duplicates(min_lines=1, include_data=True)) == 1


def test_find_duplicates_excludes_scalar_config_assignments_by_default() -> None:
    """Scalar assignment-only config chunks are skipped unless include_data=True."""
    if duplicate_features(_chunk("VALUE = 1", "src/config.py")).code_bearing_node_count is None:
        pytest.skip("tree_sitter_language_pack is not available")

    content = """\
DATE_FORMAT = "j F Y"
TIME_FORMAT = "h:i A"
MONTH_DAY_FORMAT = "j F"
SHORT_DATE_FORMAT = "j M Y"
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py"),
            _chunk(content, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, include_data=True)) == 1


def test_find_duplicates_excludes_scaffolding_chunks_by_default() -> None:
    """Import/header/attribute scaffolding chunks are skipped unless include_scaffolding=True."""
    rust_feature_probe = duplicate_features(_chunk("#![allow(clippy::too_many_lines)]", "src/lib.rs", language="rust"))
    if rust_feature_probe.code_bearing_node_count is None:
        pytest.skip("tree_sitter_language_pack is not available")

    content = "#![allow(clippy::too_many_lines, clippy::missing_errors_doc)]\n"
    index = _duplicate_index(
        [
            _chunk(content, "src/a.rs", language="rust"),
            _chunk(content, "src/b.rs", language="rust"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert index.find_duplicates(min_lines=1, include_data=True) == []
    assert len(index.find_duplicates(min_lines=1, include_scaffolding=True)) == 1


def test_find_duplicates_keeps_declaration_only_chunks_without_static_bindings() -> None:
    """Declaration-only chunks are not treated as static data without binding/data-shape nodes."""
    if duplicate_features(_chunk("class User:\n    pass", "src/user.py")).code_bearing_node_count is None:
        pytest.skip("tree_sitter_language_pack is not available")

    content = """\
class UserRecord:
    pass
"""
    index = _duplicate_index(
        [
            _chunk(content, "src/a.py"),
            _chunk(content, "src/b.py"),
        ]
    )

    assert len(index.find_duplicates(min_lines=1)) == 1


def test_find_duplicates_empty_or_non_positive_top_k_returns_empty() -> None:
    """Empty indexes, singletons, and non-positive top_k return no duplicate clusters."""
    assert _duplicate_index([]).find_duplicates() == []
    assert _duplicate_index([_chunk("def add(a, b):\n    return a + b", "src/a.py")]).find_duplicates(min_lines=1) == []

    index = _duplicate_index(
        [
            _chunk("def add(a, b):\n    return a + b", "src/a.py"),
            _chunk("def add(a, b):\n    return a + b", "src/b.py"),
        ]
    )
    assert index.find_duplicates(top_k=0, min_lines=1) == []
