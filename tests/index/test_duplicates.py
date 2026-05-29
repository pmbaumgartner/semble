from unittest.mock import MagicMock

import numpy as np
import pytest
from vicinity.backends.basic import BasicArgs

import semble.duplicates.search as duplicate_search
from semble import SembleIndex
from semble.duplicates.scoring import duplicate_features
from semble.duplicates.search import find_duplicate_pairs
from semble.index.dense import SelectableBasicBackend
from semble.types import DuplicateCluster, DuplicateSignals
from tests.conftest import make_chunk, require_duplicate_features


def _cluster_paths(cluster: DuplicateCluster) -> set[str]:
    return {member.file_path for member in cluster.members}


def test_find_duplicates_returns_ranked_clusters_and_respects_top_k(duplicate_index_factory) -> None:
    """find_duplicates returns the strongest duplicate cluster up to top_k."""
    left = make_chunk(
        """\
def total_price(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
        "src/prices.py",
    )
    renamed = make_chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "src/invoices.py",
    )
    unrelated = make_chunk(
        """\
class Renderer:
    def render(self, page):
        print(page.title)
""",
        "src/render.py",
    )
    index = duplicate_index_factory([left, renamed, unrelated])

    results = index.find_duplicates(top_k=1, min_lines=1)

    assert len(results) == 1
    assert _cluster_paths(results[0]) == {"src/prices.py", "src/invoices.py"}


def test_find_duplicate_pairs_helper_returns_unsliced_sorted_pairs(duplicate_index_factory) -> None:
    """The private pair scan returns all sorted pairs for later clustering."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/b.py"),
            make_chunk(content, "src/c.py"),
        ]
    )

    all_pairs = find_duplicate_pairs(
        index.chunks,
        index._semantic_index,
        index._language_mapping,
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
    assert all_pairs == sorted(all_pairs, key=duplicate_search._duplicate_sort_key)


def test_find_duplicates_uses_existing_embeddings_without_reencoding() -> None:
    """Duplicate discovery reuses indexed embeddings instead of encoding each chunk again."""
    chunks = [
        make_chunk("def add(a, b):\n    return a + b", "src/a.py"),
        make_chunk("def plus(x, y):\n    return x + y", "src/b.py"),
    ]
    embeddings = np.zeros((len(chunks), 4), dtype=np.float32)
    embeddings[:, 0] = 1.0
    semantic_index = SelectableBasicBackend(embeddings, BasicArgs())
    model = MagicMock()
    index = SembleIndex(model, MagicMock(), semantic_index, chunks, "")

    assert index.find_duplicates(min_lines=1)
    model.encode.assert_not_called()


def test_find_duplicates_batches_neighbor_queries_by_language(duplicate_index_factory) -> None:
    """Semantic neighbors are queried in language groups while preserving candidate_k."""
    index = duplicate_index_factory(
        [
            make_chunk("def add(a, b):\n    return a + b", "src/a.py", language="python"),
            make_chunk("def plus(x, y):\n    return x + y", "src/b.py", language="python"),
            make_chunk("function add(a, b) {\n  return a + b;\n}", "src/a.js", language="javascript"),
            make_chunk("function plus(x, y) {\n  return x + y;\n}", "src/b.js", language="javascript"),
        ]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        mock_query = MagicMock(wraps=index._semantic_index.query)
        monkeypatch.setattr(index._semantic_index, "query", mock_query)
        index.find_duplicates(candidate_k=1, min_lines=1, include_tests=True)

    assert mock_query.call_count == 2
    assert all(call.kwargs["k"] == 2 for call in mock_query.call_args_list)


def test_find_duplicates_precomputes_features_once_per_eligible_chunk(
    duplicate_index_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate structural features are cached per chunk during a scan."""
    index = duplicate_index_factory(
        [
            make_chunk("def add(a, b):\n    return a + b", "src/a.py"),
            make_chunk("def plus(x, y):\n    return x + y", "src/b.py"),
            make_chunk("def skip(a, b):\n    return a + b", "tests/test_skip.py"),
        ]
    )
    mock_features = MagicMock(wraps=duplicate_features)
    monkeypatch.setattr(duplicate_search, "duplicate_features", mock_features)

    index.find_duplicates(min_lines=1)

    assert mock_features.call_count == 2


def test_find_duplicates_excludes_overlapping_same_file_ranges(duplicate_index_factory) -> None:
    """Same-file overlapping chunks are not returned as duplicate clusters."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/prices.py", start_line=1),
            make_chunk(content, "src/prices.py", start_line=3),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []


def test_find_duplicates_deduplicates_reversed_pairs_deterministically(duplicate_index_factory) -> None:
    """A/B and B/A semantic candidates collapse into one stable pair."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/b.py"),
            make_chunk(content, "src/a.py"),
        ]
    )

    results = index.find_duplicates(top_k=10, min_lines=1)

    assert len(results) == 1
    assert results[0].pairs[0].left.file_path == "src/a.py"
    assert results[0].pairs[0].right.file_path == "src/b.py"


def test_find_duplicates_filters_path_scopes(duplicate_index_factory) -> None:
    """Include and exclude path scopes match exact files and directory prefixes."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/nested/b.py"),
            make_chunk(content, "src/generated/c.py"),
            make_chunk(content, "tests/a_test.py"),
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


def test_find_duplicates_excludes_tests_by_default(duplicate_index_factory) -> None:
    """Duplicate discovery skips test-looking paths unless include_tests=True."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "tests/test_a.py"),
            make_chunk(content, "tests/test_b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, include_tests=True)) == 1


def test_find_duplicates_excludes_test_filename_patterns_by_default(duplicate_index_factory) -> None:
    """Test-looking filenames from the ranking policy are skipped by default."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/test_prices.py"),
            make_chunk(content, "src/invoices_test.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, include_tests=True)) == 1


@pytest.mark.parametrize(
    "file_path",
    [
        "tests/test_total.py",
        "src/test_total.py",
        "src/total_test.go",
        "src/TotalTest.java",
        "src/total.test.ts",
    ],
)
def test_duplicate_test_path_filter_uses_shared_test_patterns(file_path: str) -> None:
    """Duplicate discovery uses the shared test path regexes through its own helper."""
    assert duplicate_search._is_test_path(file_path)
    assert not duplicate_search._path_is_included(file_path, include_tests=False)
    assert duplicate_search._path_is_included(file_path, include_tests=True)


def test_find_duplicates_intersects_language_and_path_filters(duplicate_index_factory) -> None:
    """Duplicate filters use intersection semantics instead of search's union selector."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py", language="python"),
            make_chunk(content, "web/a.js", language="javascript"),
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


def test_find_duplicates_respects_language_filter(duplicate_index_factory) -> None:
    """Language filters restrict all returned duplicate cluster members."""
    content = """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py", language="python"),
            make_chunk(content, "src/b.py", language="python"),
            make_chunk(content, "src/a.js", language="javascript"),
        ]
    )

    results = index.find_duplicates(top_k=10, min_lines=1, filter_languages=["python"])

    assert len(results) == 1
    assert {member.language for member in results[0].members} == {"python"}


def test_find_duplicates_respects_min_lines_and_min_score(duplicate_index_factory) -> None:
    """Minimum line, score, and cluster-size thresholds filter duplicate clusters."""
    content = """\
def add(a, b):
    return a + b
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=3) == []
    assert len(index.find_duplicates(min_lines=1, min_score=1.0)) == 1
    assert index.find_duplicates(min_lines=1, min_score=1.01) == []
    assert index.find_duplicates(min_lines=1, min_cluster_size=3) == []


def test_find_duplicates_respects_default_structural_floor(
    duplicate_index_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weak structural pairs are filtered by default and can be included by lowering the floor."""
    content = """\
def add(a, b):
    return a + b
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/b.py"),
        ]
    )
    monkeypatch.setattr(
        duplicate_search,
        "score_duplicate_features",
        lambda *args, **kwargs: DuplicateSignals(
            semantic_score=0.9,
            structural_score=0.39,
            token_jaccard=0.39,
        ),
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, min_structural_score=0.39)) == 1


def test_find_duplicates_excludes_cross_language_pairs(duplicate_index_factory) -> None:
    """Duplicate discovery only compares chunks with the same language."""
    content = """\
def add(a, b):
    return a + b
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py", language="python"),
            make_chunk(content, "src/a.js", language="javascript"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []


def test_find_duplicates_allows_unknown_language_pairs(duplicate_index_factory) -> None:
    """Unknown-language chunks compare only with other unknown-language chunks."""
    content = """\
def add(a, b):
    return a + b
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.txt", language=None),
            make_chunk(content, "src/b.txt", language=None),
            make_chunk(content, "src/c.py", language="python"),
        ]
    )

    results = index.find_duplicates(min_lines=1)

    assert len(results) == 1
    assert {member.language for member in results[0].members} == {None}


def test_find_duplicates_excludes_docstring_only_chunks_when_parser_is_available(duplicate_index_factory) -> None:
    """Parser-supported chunks with only string/comment AST subtrees are not duplicate candidates."""
    require_duplicate_features("def real():\n    return 1", "src/real.py")

    index = duplicate_index_factory(
        [
            make_chunk('"""Repeated documentation."""', "src/a.py"),
            make_chunk('"""Repeated documentation."""', "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []


def test_find_duplicates_excludes_static_data_chunks_by_default(duplicate_index_factory) -> None:
    """Static data/config chunks are skipped unless include_data=True."""
    require_duplicate_features("[1, 2, 3]", "src/data.py")

    content = """\
VALUES = [
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
]
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert index.find_duplicates(min_lines=1, include_scaffolding=True) == []
    assert len(index.find_duplicates(min_lines=1, include_data=True)) == 1


def test_find_duplicates_excludes_scalar_config_assignments_by_default(duplicate_index_factory) -> None:
    """Scalar assignment-only config chunks are skipped unless include_data=True."""
    require_duplicate_features("VALUE = 1", "src/config.py")

    content = """\
DATE_FORMAT = "j F Y"
TIME_FORMAT = "h:i A"
MONTH_DAY_FORMAT = "j F"
SHORT_DATE_FORMAT = "j M Y"
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert len(index.find_duplicates(min_lines=1, include_data=True)) == 1


def test_find_duplicates_excludes_scaffolding_chunks_by_default(duplicate_index_factory) -> None:
    """Import/header/attribute scaffolding chunks are skipped unless include_scaffolding=True."""
    require_duplicate_features("#![allow(clippy::too_many_lines)]", "src/lib.rs", language="rust")

    content = "#![allow(clippy::too_many_lines, clippy::missing_errors_doc)]\n"
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.rs", language="rust"),
            make_chunk(content, "src/b.rs", language="rust"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    assert index.find_duplicates(min_lines=1, include_data=True) == []
    assert len(index.find_duplicates(min_lines=1, include_scaffolding=True)) == 1


def test_find_duplicates_strips_scaffolding_from_mixed_chunks_by_default(duplicate_index_factory) -> None:
    """Repeated imports do not make unrelated mixed chunks duplicate candidates by default."""
    require_duplicate_features("import os\n\ndef f():\n    return os.getcwd()", "src/a.py")

    left = """\
import alpha
import beta
import gamma
import delta
import epsilon
import zeta

def build_user():
    return alpha.load_user()
"""
    right = """\
import alpha
import beta
import gamma
import delta
import epsilon
import zeta

def send_email(message):
    beta.dispatch(message)
"""
    index = duplicate_index_factory(
        [
            make_chunk(left, "src/a.py"),
            make_chunk(right, "src/b.py"),
        ]
    )

    assert index.find_duplicates(min_lines=1) == []
    scaffold_clusters = index.find_duplicates(min_lines=1, include_scaffolding=True)
    assert len(scaffold_clusters) == 1
    scaffold_pair = scaffold_clusters[0].pairs[0]
    assert "import alpha" in scaffold_pair.left_content
    assert "import alpha" in scaffold_pair.right_content


def test_find_duplicates_keeps_substantive_duplicates_with_scaffolding_stripped(duplicate_index_factory) -> None:
    """Mixed chunks still match when their substantive bodies are duplicates."""
    require_duplicate_features("import os\n\ndef f():\n    return os.getcwd()", "src/a.py")

    left = """\
import alpha
import beta

def normalize_user(user):
    return user.strip().lower()
"""
    right = """\
import alpha
import beta

def normalize_email(email):
    return email.strip().lower()
"""
    index = duplicate_index_factory(
        [
            make_chunk(left, "src/a.py"),
            make_chunk(right, "src/b.py"),
        ]
    )

    clusters = index.find_duplicates(min_lines=1)
    assert len(clusters) == 1
    pair = clusters[0].pairs[0]
    assert "import alpha" in pair.left.content
    assert "import alpha" in pair.right.content
    assert "import alpha" not in pair.left_content
    assert "import alpha" not in pair.right_content
    assert "def normalize_user" in pair.left_content
    assert "def normalize_email" in pair.right_content


def test_find_duplicates_keeps_declaration_only_chunks_without_static_bindings(duplicate_index_factory) -> None:
    """Declaration-only chunks are not treated as static data without binding/data-shape nodes."""
    require_duplicate_features("class User:\n    pass", "src/user.py")

    content = """\
class UserRecord:
    pass
"""
    index = duplicate_index_factory(
        [
            make_chunk(content, "src/a.py"),
            make_chunk(content, "src/b.py"),
        ]
    )

    assert len(index.find_duplicates(min_lines=1)) == 1


def test_find_duplicates_empty_or_non_positive_top_k_returns_empty(duplicate_index_factory) -> None:
    """Empty indexes, singletons, and non-positive top_k return no duplicate clusters."""
    assert duplicate_index_factory([]).find_duplicates() == []
    singleton = duplicate_index_factory([make_chunk("def add(a, b):\n    return a + b", "src/a.py")])
    assert singleton.find_duplicates(min_lines=1) == []

    index = duplicate_index_factory(
        [
            make_chunk("def add(a, b):\n    return a + b", "src/a.py"),
            make_chunk("def add(a, b):\n    return a + b", "src/b.py"),
        ]
    )
    assert index.find_duplicates(top_k=0, min_lines=1) == []
