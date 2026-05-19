import pytest

import semble.duplicates.ast.features as duplicate_ast_features
from semble.duplicates.ast.taxonomy import _normalize_ast_label
from semble.duplicates.clustering import (
    _pair_key,
    _same_file_ranges_overlap,
    cluster_duplicate_pairs,
)
from semble.duplicates.scoring import (
    _weighted_structural_score,
    duplicate_features,
    duplicate_features_are_eligible,
    duplicate_score,
    score_duplicate_features,
)
from semble.types import Chunk, DuplicateCluster, DuplicateSignals
from tests.conftest import make_chunk, make_duplicate_pair, require_parsers


def _score_chunks(left: Chunk, right: Chunk, *, semantic_score: float) -> DuplicateSignals:
    return score_duplicate_features(
        duplicate_features(left),
        duplicate_features(right),
        semantic_score=semantic_score,
    )


def test_duplicate_cluster_score_uses_strongest_pair() -> None:
    """DuplicateCluster.score reflects the strongest pair in a cluster."""
    strongest = make_duplicate_pair(score=0.84)
    weaker = make_duplicate_pair(
        left=strongest.right,
        right=make_chunk("def third():\n    return 1", "third.py"),
        score=0.75,
    )
    cluster = DuplicateCluster(members=(strongest.left, strongest.right, weaker.right), pairs=(strongest, weaker))

    assert cluster.score == 0.84


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

    similar_signals = _score_chunks(left, renamed, semantic_score=0.9)
    unrelated_signals = _score_chunks(left, unrelated, semantic_score=0.9)

    assert similar_signals.token_jaccard > unrelated_signals.token_jaccard
    assert similar_signals.structural_score > unrelated_signals.structural_score
    assert similar_signals.token_jaccard > 0.5


def test_empty_or_comment_only_fingerprints_are_not_perfect_matches() -> None:
    """Empty structural fingerprints score zero instead of perfect similarity."""
    left = make_chunk("# only a comment\n# another comment", "left.py")
    right = make_chunk("// only a comment\n-- another comment", "right.py")
    signals = _score_chunks(left, right, semantic_score=1.0)

    assert signals.token_jaccard == 0.0
    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == 0.0
    assert duplicate_score(signals) == 0.0


def test_score_duplicate_pair_populates_ast_signals_when_parser_is_available() -> None:
    """Parser-backed chunks include AST type and shape signals."""
    require_parsers("python")

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

    signals = _score_chunks(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is not None
    assert signals.ast_shape_jaccard is not None
    assert 0.0 <= signals.ast_type_jaccard <= 1.0
    assert 0.0 <= signals.ast_shape_jaccard <= 1.0


def test_structural_score_uses_weighted_ast_blend_when_available() -> None:
    """Structural score blends token, AST type, and AST shape signals."""
    require_parsers("python")

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

    signals = _score_chunks(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is not None
    assert signals.ast_shape_jaccard is not None
    assert signals.structural_score == pytest.approx(
        _weighted_structural_score(
            token_jaccard=signals.token_jaccard,
            ast_type_jaccard=signals.ast_type_jaccard,
            ast_shape_jaccard=signals.ast_shape_jaccard,
        )
    )


def test_duplicate_features_detect_docstring_only_chunks_when_parser_is_available() -> None:
    """Comment/string-only chunks are ineligible when parser-backed AST counting is available."""
    require_parsers("python")

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
    require_parsers("python")

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
    require_parsers("python")

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
    require_parsers("rust", "java")

    rust_attributes = duplicate_features(
        make_chunk(
            "#![allow(clippy::too_many_lines, clippy::missing_errors_doc)]\n",
            "lib.rs",
            end_line=1,
            language="rust",
        )
    )
    java_imports = duplicate_features(
        make_chunk(
            """\
package org.junit.jupiter.api.condition;

import static org.apiguardian.api.API.Status.MAINTAINED;
import java.lang.annotation.Documented;
import org.apiguardian.api.API;
""",
            "Example.java",
            end_line=5,
            language="java",
        )
    )
    behavior = duplicate_features(
        make_chunk(
            """\
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
""",
            "lib.rs",
            end_line=3,
            language="rust",
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
    require_parsers("java")

    declaration = duplicate_features(
        make_chunk(
            """\
@Documented
@Retention(RetentionPolicy.RUNTIME)
public @interface DisabledOnJre {
    JRE[] value();
}
""",
            "DisabledOnJre.java",
            end_line=5,
            language="java",
        )
    )

    assert declaration.behavioral_node_count == 0
    assert declaration.scaffolding_node_count
    assert declaration.substantive_node_count
    assert duplicate_features_are_eligible(declaration)


def test_duplicate_features_strip_scaffolding_from_mixed_chunks_when_requested() -> None:
    """Mixed chunks can be scored without import/header scaffolding."""
    require_parsers("python")

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
    chunk = make_chunk(content, "example.py", end_line=len(content.splitlines()))

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
    monkeypatch.setattr(duplicate_ast_features, "get_parser_for_language", lambda language: None)
    left = make_chunk("def add(a, b):\n    return a + b", "left.py")
    right = make_chunk("def plus(x, y):\n    return x + y", "right.py")

    signals = _score_chunks(left, right, semantic_score=0.9)

    assert signals.ast_type_jaccard is None
    assert signals.ast_shape_jaccard is None
    assert signals.structural_score == signals.token_jaccard


def test_score_duplicate_pair_falls_back_for_unsupported_language() -> None:
    """Unsupported languages use token-only duplicate scoring."""
    left = make_chunk("def add(a, b):\n    return a + b", "left.txt", language="text")
    right = make_chunk("def plus(x, y):\n    return x + y", "right.txt", language="text")

    signals = _score_chunks(left, right, semantic_score=0.9)

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
    a = make_chunk("a", "src/a.py", end_line=2)
    b = make_chunk("b", "src/b.py", end_line=2)
    c = make_chunk("c", "src/c.py", end_line=2)
    d = make_chunk("d", "src/d.py", end_line=2)
    e = make_chunk("e", "src/e.py", end_line=2)

    clusters = cluster_duplicate_pairs(
        [
            make_duplicate_pair(
                left=b,
                right=c,
                score=0.8,
                semantic_score=0.8,
                structural_score=0.8,
                token_jaccard=0.8,
            ),
            make_duplicate_pair(
                left=d,
                right=e,
                score=0.7,
                semantic_score=0.7,
                structural_score=0.7,
                token_jaccard=0.7,
            ),
            make_duplicate_pair(
                left=a,
                right=b,
                score=0.9,
                semantic_score=0.9,
                structural_score=0.9,
                token_jaccard=0.9,
            ),
        ]
    )

    assert [[member.file_path for member in cluster.members] for cluster in clusters] == [
        ["src/a.py", "src/b.py", "src/c.py"],
        ["src/d.py", "src/e.py"],
    ]


def test_cluster_duplicate_pairs_keeps_size_two_clusters_by_default() -> None:
    """A duplicate pair is represented as the smallest duplicate cluster."""
    left = make_chunk("left", "src/left.py", end_line=2)
    right = make_chunk("right", "src/right.py", end_line=2)

    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.9,
        semantic_score=0.9,
        structural_score=0.9,
        token_jaccard=0.9,
    )
    clusters = cluster_duplicate_pairs([pair])

    assert len(clusters) == 1
    assert clusters[0].members == (left, right)


def test_cluster_duplicate_pairs_discards_small_components() -> None:
    """Components smaller than min_cluster_size are omitted."""
    left = make_chunk("left", "src/left.py", end_line=2)
    right = make_chunk("right", "src/right.py", end_line=2)

    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.9,
        semantic_score=0.9,
        structural_score=0.9,
        token_jaccard=0.9,
    )
    assert cluster_duplicate_pairs([pair], min_cluster_size=3) == []


def test_cluster_duplicate_pairs_clamps_min_cluster_size_to_two() -> None:
    """min_cluster_size below two still returns only real duplicate clusters."""
    left = make_chunk("left", "src/left.py", end_line=2)
    right = make_chunk("right", "src/right.py", end_line=2)

    pair = make_duplicate_pair(
        left=left,
        right=right,
        score=0.9,
        semantic_score=0.9,
        structural_score=0.9,
        token_jaccard=0.9,
    )
    clusters = cluster_duplicate_pairs([pair], min_cluster_size=1)

    assert len(clusters) == 1
    assert len(clusters[0].members) == 2


def test_cluster_duplicate_pairs_sorts_members_pairs_and_clusters() -> None:
    """Clusters, members, and pair edges are ordered deterministically."""
    a = make_chunk("a", "src/a.py", end_line=2)
    b = make_chunk("b", "src/b.py", end_line=2)
    c = make_chunk("c", "src/c.py", end_line=2)
    d = make_chunk("d", "src/d.py", end_line=2)
    e = make_chunk("e", "src/e.py", end_line=2)
    ac = make_duplicate_pair(left=a, right=c, score=0.7, semantic_score=0.7, structural_score=0.7, token_jaccard=0.7)
    ab = make_duplicate_pair(left=a, right=b, score=0.9, semantic_score=0.8, structural_score=0.7, token_jaccard=0.7)
    bc = make_duplicate_pair(left=b, right=c, score=0.9, semantic_score=0.95, structural_score=0.6, token_jaccard=0.6)
    de = make_duplicate_pair(
        left=d,
        right=e,
        score=0.95,
        semantic_score=0.95,
        structural_score=0.95,
        token_jaccard=0.95,
    )

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
    a = make_chunk("a", "src/a.py", end_line=2)
    b = make_chunk("b", "src/b.py", end_line=2)
    c = make_chunk("c", "src/c.py", end_line=2)

    clusters = cluster_duplicate_pairs(
        [
            make_duplicate_pair(
                left=a,
                right=b,
                score=0.9,
                semantic_score=0.9,
                structural_score=0.9,
                token_jaccard=0.9,
            ),
            make_duplicate_pair(
                left=b,
                right=c,
                score=0.9,
                semantic_score=0.9,
                structural_score=0.9,
                token_jaccard=0.9,
            ),
        ]
    )

    assert len(clusters) == 1
    assert [member.file_path for member in clusters[0].members] == ["src/a.py", "src/b.py", "src/c.py"]
    assert len(clusters[0].pairs) == 2


def test_same_file_ranges_overlap() -> None:
    """Same-file overlapping ranges are detected with inclusive line ranges."""
    left = make_chunk("left", "src/a.py", end_line=10)
    overlap = make_chunk("overlap", "src/a.py", start_line=10, end_line=20)
    separate = make_chunk("separate", "src/a.py", start_line=11, end_line=20)
    other_file = make_chunk("other", "src/b.py", start_line=5, end_line=8)

    assert _same_file_ranges_overlap(left, overlap)
    assert not _same_file_ranges_overlap(left, separate)
    assert not _same_file_ranges_overlap(left, other_file)


def test_pair_key_is_stable_for_reversed_pairs() -> None:
    """Pair keys are unordered so candidate generation can deduplicate A/B and B/A."""
    left = make_chunk("left", "src/b.py", start_line=5, end_line=8)
    right = make_chunk("right", "src/a.py", end_line=4)

    assert _pair_key(left, right) == _pair_key(right, left)
    assert _pair_key(left, right)[0] == ("src/a.py", 1, 4)
