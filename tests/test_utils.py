from semble.types import DuplicateResult, DuplicateSignals
from semble.utils import _format_duplicate_results
from tests.conftest import make_chunk


def _duplicate_result(*, ast_signals: bool = False) -> DuplicateResult:
    left = make_chunk("def left():\n    return 1", "src/left.py")
    right = make_chunk("def right():\n    return 1", "src/right.py")
    signals = DuplicateSignals(
        semantic_score=0.9,
        structural_score=0.8,
        token_jaccard=0.7,
        ast_type_jaccard=0.6 if ast_signals else None,
        ast_shape_jaccard=0.5 if ast_signals else None,
    )
    return DuplicateResult(left=left, right=right, score=0.84, signals=signals)


def test_format_duplicate_results_empty() -> None:
    """Empty duplicate results render only the header."""
    assert _format_duplicate_results("Duplicate candidates", []) == "Duplicate candidates\n"


def test_format_duplicate_results_token_only_omits_ast_fields() -> None:
    """Token-only duplicate signals do not render unavailable AST fields."""
    out = _format_duplicate_results("Duplicate candidates", [_duplicate_result()])

    assert "semantic=0.900 structural=0.800 tokens=0.700" in out
    assert "ast_type" not in out
    assert "ast_shape" not in out


def test_format_duplicate_results_ast_signals() -> None:
    """AST-backed duplicate signals render AST score fields."""
    out = _format_duplicate_results("Duplicate candidates", [_duplicate_result(ast_signals=True)])

    assert "ast_type=0.600" in out
    assert "ast_shape=0.500" in out


def test_format_duplicate_results_locations_and_code_blocks() -> None:
    """Duplicate formatting includes both locations and both fenced snippets."""
    out = _format_duplicate_results("Duplicate candidates", [_duplicate_result()])

    assert "src/left.py:1-2 <-> src/right.py:1-2" in out
    assert "def left():" in out
    assert "def right():" in out
    assert out.count("```") == 4
    assert "Left:" in out
    assert "Right:" in out
