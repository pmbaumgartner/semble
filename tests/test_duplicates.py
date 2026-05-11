import pytest

import semble.duplicates as duplicate_exports
import semble.duplicates.ast as duplicate_ast
from semble import (
    DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
    DuplicateCluster,
    DuplicateMatch,
    DuplicatePair,
    DuplicateSignals,
)
from semble.duplicates import (
    cluster_duplicate_pairs,
    duplicate_features,
    duplicate_features_are_eligible,
    duplicate_score,
    score_duplicate_features,
    score_duplicate_pair,
)
from semble.duplicates.ast import (
    _normalize_ast_label,
    _parser_for_language,
    _parser_language_for_chunk,
)
from semble.duplicates.clustering import (
    _pair_key,
    _same_file_ranges_overlap,
)
from semble.duplicates.scoring import (
    _weighted_structural_score,
)
from semble.duplicates.tokens import _jaccard
from semble.types import Chunk
from tests.conftest import make_chunk


def _duplicate_result(
    left: Chunk,
    right: Chunk,
    *,
    score: float = 0.9,
    semantic_score: float | None = None,
    structural_score: float | None = None,
) -> DuplicatePair:
    semantic_score = score if semantic_score is None else semantic_score
    structural_score = score if structural_score is None else structural_score
    return DuplicatePair(
        left=DuplicateMatch(chunk=left, content=left.content),
        right=DuplicateMatch(chunk=right, content=right.content),
        score=score,
        signals=DuplicateSignals(
            semantic_score=semantic_score,
            structural_score=structural_score,
            token_jaccard=structural_score,
        ),
    )


def test_duplicate_types_are_exported() -> None:
    """Duplicate result types are part of the package-level API."""
    left = make_chunk("def left():\n    return 1", "left.py")
    right = make_chunk("def right():\n    return 1", "right.py")
    signals = DuplicateSignals(semantic_score=0.9, structural_score=0.8, token_jaccard=0.8)
    left_match = DuplicateMatch(chunk=left, content=left.content)
    right_match = DuplicateMatch(chunk=right, content=right.content)
    result = DuplicatePair(left=left_match, right=right_match, score=0.84, signals=signals)
    cluster = DuplicateCluster(members=(left, right), pairs=(result,))

    assert result.left is left_match
    assert result.right is right_match
    assert result.left.chunk is left
    assert result.right.chunk is right
    assert result.left.content == left.content
    assert result.right.content == right.content
    assert result.signals is signals
    assert cluster.members == (left, right)
    assert cluster.pairs == (result,)
    assert cluster.score == result.score
    assert DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE == 0.4


def test_duplicate_package_star_exports_exclude_private_helpers() -> None:
    """The duplicate package star export only includes stable public names."""
    assert "_jaccard" not in duplicate_exports.__all__
    assert "_pair_key" not in duplicate_exports.__all__
    assert all(not name.startswith("_") for name in duplicate_exports.__all__)


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
    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == 0.0
    assert duplicate_score(signals) == 0.0


def test_score_duplicate_pair_populates_ast_signals_when_parser_is_available() -> None:
    """Parser-backed chunks include AST type and shape signals."""
    if _parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

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
    right = make_chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "invoices.py",
    )

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is not None
    assert signals.ast_shape_jaccard is not None
    assert 0.0 <= signals.ast_type_jaccard <= 1.0
    assert 0.0 <= signals.ast_shape_jaccard <= 1.0


def test_structural_score_uses_weighted_ast_blend_when_available() -> None:
    """Structural score blends token, AST type, and AST shape signals."""
    if _parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

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
    right = make_chunk(
        """\
def invoice_amount(products):
    amount = 0
    for product in products:
        amount += product.cost
    return amount
""",
        "invoices.py",
    )

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is not None
    assert signals.ast_shape_jaccard is not None
    assert signals.structural_score == pytest.approx(
        _weighted_structural_score(
            token_jaccard=signals.token_jaccard,
            ast_type_jaccard=signals.ast_type_jaccard,
            ast_shape_jaccard=signals.ast_shape_jaccard,
        )
    )


def test_score_duplicate_features_reuses_precomputed_features() -> None:
    """Precomputed duplicate features produce the same signals as chunk-level scoring."""
    left = make_chunk("def add(a, b):\n    return a + b", "left.py")
    right = make_chunk("def plus(x, y):\n    return x + y", "right.py")

    from_chunks = score_duplicate_pair(left, right, semantic_score=0.9)
    from_features = score_duplicate_features(
        duplicate_features(left),
        duplicate_features(right),
        semantic_score=0.9,
    )

    assert from_features == from_chunks


def test_duplicate_features_detect_docstring_only_chunks_when_parser_is_available() -> None:
    """Comment/string-only chunks are ineligible when parser-backed AST counting is available."""
    if _parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    docstring = duplicate_features(make_chunk('"""Only documentation."""', "docs.py"))
    comment = duplicate_features(make_chunk("# Only a comment", "comment.py"))
    real_code = duplicate_features(make_chunk("def real():\n    return 1", "real.py"))

    assert docstring.code_bearing_node_count is not None
    assert docstring.code_bearing_node_count < 4
    assert not duplicate_features_are_eligible(docstring)
    assert comment.code_bearing_node_count is not None
    assert comment.code_bearing_node_count < 4
    assert not duplicate_features_are_eligible(comment)
    assert duplicate_features_are_eligible(real_code)


def test_duplicate_features_detect_static_data_chunks_when_parser_is_available() -> None:
    """Literal/container data chunks are ineligible by default and opt-in eligible."""
    if _parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    data = duplicate_features(
        make_chunk(
            """\
VALUES = [
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
]
""",
            "data.py",
        )
    )
    behavior = duplicate_features(
        make_chunk(
            """\
def total(items):
    total = 0
    for item in items:
        total += item.price
    return total
""",
            "behavior.py",
        )
    )

    assert data.code_bearing_node_count is not None
    assert data.behavioral_node_count == 0
    assert data.data_shape_node_count
    assert data.static_binding_node_count
    assert not duplicate_features_are_eligible(data)
    assert duplicate_features_are_eligible(data, include_data=True)
    assert behavior.behavioral_node_count
    assert duplicate_features_are_eligible(behavior)


def test_duplicate_features_detect_scalar_config_assignments_when_parser_is_available() -> None:
    """Scalar assignment-only config chunks are static data even without containers."""
    if _parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    config = duplicate_features(
        make_chunk(
            """\
DATE_FORMAT = "j F Y"
TIME_FORMAT = "h:i A"
MONTH_DAY_FORMAT = "j F"
SHORT_DATE_FORMAT = "j M Y"
""",
            "formats.py",
        )
    )

    assert config.code_bearing_node_count is not None
    assert config.behavioral_node_count == 0
    assert config.data_shape_node_count == 0
    assert config.static_binding_node_count
    assert not duplicate_features_are_eligible(config)
    assert duplicate_features_are_eligible(config, include_data=True)


def test_duplicate_features_detect_scaffolding_chunks_when_parser_is_available() -> None:
    """Import/header/attribute scaffolding chunks are ineligible by default and opt-in eligible."""
    if _parser_for_language("rust") is None or _parser_for_language("java") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    rust_attributes = duplicate_features(
        Chunk(
            "#![allow(clippy::too_many_lines, clippy::missing_errors_doc)]\n",
            "lib.rs",
            1,
            1,
            "rust",
        )
    )
    java_imports = duplicate_features(
        Chunk(
            """\
package org.junit.jupiter.api.condition;

import static org.apiguardian.api.API.Status.MAINTAINED;
import java.lang.annotation.Documented;
import org.apiguardian.api.API;
""",
            "Example.java",
            1,
            5,
            "java",
        )
    )
    behavior = duplicate_features(
        Chunk(
            """\
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
""",
            "lib.rs",
            1,
            3,
            "rust",
        )
    )

    assert rust_attributes.scaffolding_node_count
    assert rust_attributes.substantive_node_count == 0
    assert not duplicate_features_are_eligible(rust_attributes)
    assert duplicate_features_are_eligible(rust_attributes, include_scaffolding=True)
    assert java_imports.scaffolding_node_count
    assert java_imports.substantive_node_count == 0
    assert not duplicate_features_are_eligible(java_imports)
    assert duplicate_features_are_eligible(java_imports, include_scaffolding=True)
    assert behavior.behavioral_node_count
    assert duplicate_features_are_eligible(behavior)


def test_duplicate_features_keep_declarations_with_scaffolding_when_parser_is_available() -> None:
    """Declarations remain eligible even when they have scaffolding annotations."""
    if _parser_for_language("java") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    declaration = duplicate_features(
        Chunk(
            """\
@Documented
@Retention(RetentionPolicy.RUNTIME)
public @interface DisabledOnJre {
    JRE[] value();
}
""",
            "DisabledOnJre.java",
            1,
            5,
            "java",
        )
    )

    assert declaration.behavioral_node_count == 0
    assert declaration.scaffolding_node_count
    assert declaration.substantive_node_count
    assert duplicate_features_are_eligible(declaration)


def test_duplicate_features_strip_scaffolding_from_mixed_chunks_when_requested() -> None:
    """Mixed chunks can be scored without import/header scaffolding."""
    if _parser_for_language("python") is None:
        pytest.skip("tree_sitter_language_pack is not available")

    content = """\
import alpha
import beta
import gamma
import delta
import epsilon
import zeta

def build_user():
    return alpha.load_user()
"""
    chunk = Chunk(content, "example.py", 1, len(content.splitlines()), "python")

    full = duplicate_features(chunk, include_scaffolding=True)
    stripped = duplicate_features(chunk, include_scaffolding=False)

    assert full.effective_line_count == 8
    assert stripped.effective_line_count == 2
    assert any("import" in ngram for ngram in full.token_ngrams)
    assert not any("import" in ngram for ngram in stripped.token_ngrams)
    assert full.ast_type_ngrams is not None
    assert stripped.ast_type_ngrams is not None
    assert any("import_statement" in ngram for ngram in full.ast_type_ngrams)
    assert not any("import_statement" in ngram for ngram in stripped.ast_type_ngrams)
    assert full.code_bearing_node_count and stripped.code_bearing_node_count
    assert stripped.code_bearing_node_count < full.code_bearing_node_count


def test_score_duplicate_pair_falls_back_when_parser_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing parser support leaves AST signals unavailable and keeps token scoring."""
    monkeypatch.setattr(duplicate_ast, "_parser_for_language", lambda language: None)
    left = make_chunk("def add(a, b):\n    return a + b", "left.py")
    right = make_chunk("def plus(x, y):\n    return x + y", "right.py")

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == signals.token_jaccard


def test_score_duplicate_pair_falls_back_for_unsupported_language() -> None:
    """Unsupported languages use token-only duplicate scoring."""
    left = Chunk("def add(a, b):\n    return a + b", "left.txt", 1, 2, "text")
    right = Chunk("def plus(x, y):\n    return x + y", "right.txt", 1, 2, "text")

    signals = score_duplicate_pair(left, right, semantic_score=0.9)

    assert _parser_language_for_chunk("text") is None
    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == signals.token_jaccard


@pytest.mark.parametrize(
    ("node_type", "expected"),
    [
        ("identifier", "IDENT"),
        ("property_identifier", "IDENT"),
        ("number_literal", "NUMBER"),
        ("decimal_integer_literal", "NUMBER"),
        ("string_literal", "STRING"),
        ("raw_string_literal", "STRING"),
        ("true", "BOOL"),
        ("false_literal", "BOOL"),
        ("null_literal", "NULL"),
        ("nil", "NULL"),
        ("none", "NONE"),
        ("function_definition", "function_definition"),
    ],
)
def test_normalize_ast_label(node_type: str, expected: str) -> None:
    """AST label normalization keeps structural categories language-neutral."""
    assert _normalize_ast_label(node_type) == expected


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


def test_cluster_duplicate_pairs_groups_connected_components() -> None:
    """Connected duplicate pairs become one cluster per component."""
    a = Chunk("a", "src/a.py", 1, 2, "python")
    b = Chunk("b", "src/b.py", 1, 2, "python")
    c = Chunk("c", "src/c.py", 1, 2, "python")
    d = Chunk("d", "src/d.py", 1, 2, "python")
    e = Chunk("e", "src/e.py", 1, 2, "python")

    clusters = cluster_duplicate_pairs(
        [
            _duplicate_result(b, c, score=0.8),
            _duplicate_result(d, e, score=0.7),
            _duplicate_result(a, b, score=0.9),
        ]
    )

    assert [[member.file_path for member in cluster.members] for cluster in clusters] == [
        ["src/a.py", "src/b.py", "src/c.py"],
        ["src/d.py", "src/e.py"],
    ]


def test_cluster_duplicate_pairs_keeps_size_two_clusters_by_default() -> None:
    """A duplicate pair is represented as the smallest duplicate cluster."""
    left = Chunk("left", "src/left.py", 1, 2, "python")
    right = Chunk("right", "src/right.py", 1, 2, "python")

    clusters = cluster_duplicate_pairs([_duplicate_result(left, right)])

    assert len(clusters) == 1
    assert clusters[0].members == (left, right)


def test_cluster_duplicate_pairs_discards_small_components() -> None:
    """Components smaller than min_cluster_size are omitted."""
    left = Chunk("left", "src/left.py", 1, 2, "python")
    right = Chunk("right", "src/right.py", 1, 2, "python")

    assert cluster_duplicate_pairs([_duplicate_result(left, right)], min_cluster_size=3) == []


def test_cluster_duplicate_pairs_clamps_min_cluster_size_to_two() -> None:
    """min_cluster_size below two still returns only real duplicate clusters."""
    left = Chunk("left", "src/left.py", 1, 2, "python")
    right = Chunk("right", "src/right.py", 1, 2, "python")

    clusters = cluster_duplicate_pairs([_duplicate_result(left, right)], min_cluster_size=1)

    assert len(clusters) == 1
    assert len(clusters[0].members) == 2


def test_cluster_duplicate_pairs_sorts_members_pairs_and_clusters() -> None:
    """Clusters, members, and pair edges are ordered deterministically."""
    a = Chunk("a", "src/a.py", 1, 2, "python")
    b = Chunk("b", "src/b.py", 1, 2, "python")
    c = Chunk("c", "src/c.py", 1, 2, "python")
    d = Chunk("d", "src/d.py", 1, 2, "python")
    e = Chunk("e", "src/e.py", 1, 2, "python")
    ac = _duplicate_result(a, c, score=0.7)
    ab = _duplicate_result(a, b, score=0.9, semantic_score=0.8, structural_score=0.7)
    bc = _duplicate_result(b, c, score=0.9, semantic_score=0.95, structural_score=0.6)
    de = _duplicate_result(d, e, score=0.95)

    clusters = cluster_duplicate_pairs([ac, de, ab, bc])

    assert [[member.file_path for member in cluster.members] for cluster in clusters] == [
        ["src/d.py", "src/e.py"],
        ["src/a.py", "src/b.py", "src/c.py"],
    ]
    assert clusters[0].score == 0.95
    assert clusters[1].score == 0.9
    assert clusters[1].pairs == (bc, ab, ac)


def test_cluster_duplicate_pairs_documents_weak_bridge_behavior() -> None:
    """Connected components allow A/B plus B/C to cluster without an A/C edge."""
    a = Chunk("a", "src/a.py", 1, 2, "python")
    b = Chunk("b", "src/b.py", 1, 2, "python")
    c = Chunk("c", "src/c.py", 1, 2, "python")

    clusters = cluster_duplicate_pairs([_duplicate_result(a, b), _duplicate_result(b, c)])

    assert len(clusters) == 1
    assert [member.file_path for member in clusters[0].members] == ["src/a.py", "src/b.py", "src/c.py"]
    assert len(clusters[0].pairs) == 2


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
