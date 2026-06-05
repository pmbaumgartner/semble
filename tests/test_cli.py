import json
import sys
import warnings
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import Agent, _agent_path, _cli_main, _maybe_save_index, _run_clear, _run_init, main
from semble.duplicates.search import DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE
from semble.types import ContentType, SearchResult
from tests.conftest import make_chunk, make_duplicate_cluster

_CLAUDE_FILE_PATH = _agent_path(Agent.CLAUDE)


@pytest.mark.parametrize(
    "argv",
    [
        ["semble", "/some/path", "--ref", "main"],
        ["semble"],
    ],
)
def test_main_calls_asyncio_run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """main() delegates to asyncio.run(serve(...)) when no CLI subcommand is given."""
    monkeypatch.setattr(sys, "argv", argv)
    with patch("asyncio.run") as mock_run:
        mock_run.side_effect = lambda coro: coro.close()
        main()
    mock_run.assert_called_once()


@pytest.mark.parametrize(
    "argv, expected_in_output",
    [
        (["semble", "search", "query text", "/some/path"], ["query text", "0.9"]),
        (["semble", "search", "nothing", "/some/path", "--top-k", "3"], ["No results found"]),
    ],
)
def test_cli_search(
    argv: list[str],
    expected_in_output: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main search subcommand calls index.search and prints results."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    has_results = "No results" not in expected_in_output[0]
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)] if has_results else []
    monkeypatch.setattr(sys, "argv", argv)
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out


@pytest.mark.parametrize(
    ("scenario", "expected_stdout", "expected_stderr", "expected_exit_code"),
    [
        ("with_results", ["src/bar.py", "0.8"], None, None),
        ("no_results", ["No related chunks found"], None, None),
        ("unknown_chunk", [], "No chunk found", 1),
    ],
)
def test_cli_find_related(
    scenario: str,
    expected_stdout: list[str],
    expected_stderr: str | None,
    expected_exit_code: int | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related prints results, empty states, and missing-chunk errors."""
    chunk = make_chunk("class Bar: pass", "src/bar.py")
    fake_index = MagicMock()
    fake_index.chunks = [] if scenario == "unknown_chunk" else [chunk]
    fake_index.find_related.return_value = [SearchResult(chunk=chunk, score=0.8)] if scenario == "with_results" else []
    file_path = "unknown.py" if scenario == "unknown_chunk" else "src/bar.py"
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", file_path, "1", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        if expected_exit_code is None:
            _cli_main()
        else:
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == expected_exit_code
    captured = capsys.readouterr()
    for fragment in expected_stdout:
        assert fragment in captured.out
    if expected_stderr:
        assert expected_stderr in captured.err


def test_cli_find_duplicates_maps_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-duplicates maps CLI options to index.find_duplicates."""
    fake_index = MagicMock()
    fake_index.find_duplicates.return_value = [make_duplicate_cluster()]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semble",
            "find-duplicates",
            "/some/path",
            "-k",
            "7",
            "--candidate-k",
            "19",
            "--language",
            "python",
            "--include",
            "src",
            "--include",
            "lib",
            "--exclude",
            "tests",
            "--exclude",
            "src/generated",
            "--include-tests",
            "--include-data",
            "--include-scaffolding",
            "--min-lines",
            "4",
            "--min-score",
            "0.25",
            "--min-structural-score",
            "0.42",
            "--min-cluster-size",
            "3",
        ],
    )
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _cli_main()

    mock_from_path.assert_called_once_with(
        "/some/path",
        content=[ContentType.CODE],
    )
    fake_index.find_duplicates.assert_called_once_with(
        top_k=7,
        candidate_k=19,
        min_lines=4,
        min_score=0.25,
        min_structural_score=0.42,
        min_cluster_size=3,
        filter_languages=["python"],
        include_paths=["src", "lib"],
        exclude_paths=["tests", "src/generated"],
        include_tests=True,
        include_data=True,
        include_scaffolding=True,
    )
    out = json.loads(capsys.readouterr().out)
    assert out["query"] == "Duplicate clusters"
    assert out["clusters"][0]["members"][0]["file_path"] == "src/left.py"
    assert out["clusters"][0]["pairs"][0]["signals"]["semantic_score"] == 0.9


def test_cli_find_duplicates_empty_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-duplicates prints a Semble-style empty state."""
    fake_index = MagicMock()
    fake_index.find_duplicates.return_value = []
    monkeypatch.setattr(sys, "argv", ["semble", "find-duplicates", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _cli_main()

    mock_from_path.assert_called_once_with(
        "/some/path",
        content=[ContentType.CODE],
    )
    fake_index.find_duplicates.assert_called_once_with(
        top_k=5,
        candidate_k=12,
        min_lines=8,
        min_score=0.0,
        min_structural_score=DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
        min_cluster_size=2,
        filter_languages=None,
        include_paths=None,
        exclude_paths=None,
        include_tests=False,
        include_data=False,
        include_scaffolding=False,
    )
    assert json.loads(capsys.readouterr().out) == {"error": "No duplicate clusters found."}


def test_cli_find_duplicates_compact_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """find-duplicates --detail compact prints compact duplicate JSON."""
    fake_index = MagicMock()
    fake_index.find_duplicates.return_value = [make_duplicate_cluster()]
    monkeypatch.setattr(sys, "argv", ["semble", "find-duplicates", "/some/path", "--detail", "compact"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()

    out = json.loads(capsys.readouterr().out)
    assert out["detail"] == "compact"
    assert out["clusters"][0]["members"][0]["location"] == "src/left.py:1-2"
    assert out["clusters"][0]["top_pairs"][0]["signals"]["semantic_score"] == 0.9
    assert out["clusters"][0]["strongest_pair"]["left"]["content"] == "def left():\n    return 1"
    assert "pairs" not in out["clusters"][0]


def test_cli_find_duplicates_uses_git_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """find-duplicates indexes git URLs through SembleIndex.from_git."""
    fake_index = MagicMock()
    fake_index.find_duplicates.return_value = []
    monkeypatch.setattr(sys, "argv", ["semble", "find-duplicates", "https://github.com/org/repo"])
    with patch("semble.cli.SembleIndex.from_git", return_value=fake_index) as mock_from_git:
        _cli_main()

    mock_from_git.assert_called_once_with(
        "https://github.com/org/repo",
        content=[ContentType.CODE],
    )
    assert json.loads(capsys.readouterr().out) == {"error": "No duplicate clusters found."}


def test_cli_find_duplicates_content_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """find-duplicates uses the shared --content parser before loading the index."""
    fake_index = MagicMock()
    fake_index.find_duplicates.return_value = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["semble", "find-duplicates", "/some/path", "--content", "docs", "config"],
    )
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _cli_main()
    mock_from_path.assert_called_once_with(
        "/some/path",
        content=[ContentType.DOCS, ContentType.CONFIG],
    )


@pytest.mark.parametrize("agent", list(Agent))
def test_init_creates_file(
    agent: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_init writes the correct agent file for every supported agent."""
    monkeypatch.chdir(tmp_path)
    _run_init(agent=agent)
    dest = tmp_path / _agent_path(agent)
    expected = files("semble").joinpath(f"agents/{agent.value}.md").read_text(encoding="utf-8")
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == expected
    assert str(_agent_path(agent)) in capsys.readouterr().out


def test_init_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_init exits with code 1 when the file exists and force=False."""
    monkeypatch.chdir(tmp_path)
    _run_init()
    with pytest.raises(SystemExit) as exc_info:
        _run_init()
    assert exc_info.value.code == 1
    assert "already exists" in capsys.readouterr().err


def test_init_overwrites_with_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_init overwrites an existing file when force=True."""
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / _CLAUDE_FILE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("old content", encoding="utf-8")
    _run_init(force=True)
    assert dest.read_text(encoding="utf-8") == files("semble").joinpath("agents/claude.md").read_text(encoding="utf-8")


def test_init_via_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Semble init creates the Claude agent file via _cli_main."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["semble", "init"])
    _cli_main()
    assert (tmp_path / _CLAUDE_FILE_PATH).exists()
    assert str(_CLAUDE_FILE_PATH) in capsys.readouterr().out


def test_main_dispatches_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() routes to _cli_main when first argument is a CLI subcommand."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query text", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        main()
    assert "query text" in capsys.readouterr().out


def test_main_dispatches_find_duplicates_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() routes find-duplicates through the CLI dispatcher."""
    fake_index = MagicMock()
    fake_index.find_duplicates.return_value = []
    monkeypatch.setattr(sys, "argv", ["semble", "find-duplicates", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        main()

    assert json.loads(capsys.readouterr().out) == {"error": "No duplicate clusters found."}


@pytest.mark.parametrize(
    ("argv", "expected_stdout", "expect_system_exit"),
    [
        (["semble", "--help"], "find-duplicates", True),
        (["semble", "search", "query", "/some/path"], "query", False),
    ],
)
def test_cli_entrypoint_works_without_mcp_installed(
    argv: list[str],
    expected_stdout: str,
    expect_system_exit: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI entrypoint paths succeed even when the mcp package is not installed."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    monkeypatch.setitem(sys.modules, "semble.mcp", None)
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        if expect_system_exit:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        else:
            main()
    assert expected_stdout in capsys.readouterr().out


def test_mcp_main_exits_with_message_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_mcp_main prints an actionable message and exits when mcp extras are not installed."""
    monkeypatch.setattr(sys, "argv", ["semble"])
    with patch("semble.cli.find_spec", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1
    assert "pip install 'semble[mcp]'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("search", ["semble", "search", "query", "/no/such/path"]),
        ("find-related", ["semble", "find-related", "src/foo.py", "1", "/no/such/path"]),
    ],
)
def test_cli_path_not_found(
    command: str, argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """index, search, and find-related exit 1 with a friendly message when the path does not exist."""
    monkeypatch.setattr(sys, "argv", argv)
    with patch("semble.cli._build_index", side_effect=FileNotFoundError("Path does not exist: /no/such/path")):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
    assert exc_info.value.code == 1
    assert "Path does not exist" in capsys.readouterr().err


def test_include_text_files_cli_deprecated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--include-text-files on CLI raises DeprecationWarning."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query", "/some/path", "--include-text-files"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _cli_main()
    assert any(
        "include-text-files" in str(w.message).lower() for w in caught if issubclass(w.category, DeprecationWarning)
    )


@pytest.mark.parametrize(
    ("argv_content", "expected"),
    [
        (["--content", "code"], [ContentType.CODE]),
        (["--content", "code", "docs"], [ContentType.CODE, ContentType.DOCS]),
        (["--content", "all"], [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]),
        (["--content", "code", "all"], [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]),
        ([], [ContentType.CODE]),
    ],
)
def test_cli_content_argument(
    argv_content: list[str],
    expected: list[ContentType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--content parses into the right ContentType list (including the 'all' shorthand and default)."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query", "/some/path", *argv_content])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _cli_main()
    assert list(mock_from_path.call_args.kwargs["content"]) == expected


def test_maybe_save_index_logs_error_on_save_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """_maybe_save_index prints to stderr when index.save raises."""
    fake_index = MagicMock()
    fake_index.loaded_from_disk = False
    fake_index.save.side_effect = OSError("disk full")
    with patch("semble.cli.find_index_from_cache_folder", return_value=Path("/cache")):
        _maybe_save_index(fake_index, "/some/path")
    assert "Error saving index" in capsys.readouterr().err


def test_agent_file_tools_are_bash_only() -> None:
    """The agent file must list only Bash and Read — no MCP tools that require schema loading."""
    frontmatter = files("semble").joinpath("agents/claude.md").read_text(encoding="utf-8").split("---")[1]
    tools_line = next(line for line in frontmatter.splitlines() if line.startswith("tools:"))
    tools = [t.strip() for t in tools_line.removeprefix("tools:").split(",")]
    assert set(tools) == {"Bash", "Read"}, f"Unexpected tools in agent file: {tools}"
    assert not any("mcp__" in t for t in tools)


@pytest.mark.parametrize("agent", list(Agent))
def test_agent_files_document_find_duplicates_and_current_content_flag(agent: Agent) -> None:
    """Every bundled agent template documents duplicate discovery and avoids deprecated content flags."""
    content = files("semble").joinpath(f"agents/{agent.value}.md").read_text(encoding="utf-8")
    assert "semble find-duplicates" in content
    assert "--include-text-files" not in content


def _make_valid_index_dir(cache_folder: Path, sha: str = "a" * 64) -> Path:
    """Create a fake valid index directory with the expected structure."""
    index_dir = cache_folder / sha / "index"
    index_dir.mkdir(parents=True)
    # Create the files that PersistencePath.non_existing checks
    (index_dir / "chunks.json").write_text("[]")
    (index_dir / "bm25_index").write_text("")
    (index_dir / "semantic_index").write_text("")
    (index_dir / "metadata.json").write_text("{}")
    return index_dir


@pytest.mark.parametrize(
    ("scenario", "expected_in_output"),
    [
        ("valid", ["Cleared index", "a" * 64, "b" * 64]),
        ("empty", ["No indexes found"]),
        ("non_sha", ["No indexes found"]),
        ("incomplete", ["No indexes found"]),
    ],
)
def test_run_clear_index(
    scenario: str, expected_in_output: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_clear('index') finds valid indexes, and skips non-SHA/incomplete/empty dirs."""
    if scenario == "valid":
        _make_valid_index_dir(tmp_path, "a" * 64)
        _make_valid_index_dir(tmp_path, "b" * 64)
    elif scenario == "non_sha":
        bad_dir = tmp_path / "not-a-sha" / "index"
        bad_dir.mkdir(parents=True)
        (bad_dir / "chunks.json").write_text("[]")
        (bad_dir / "bm25_index").write_text("")
        (bad_dir / "semantic_index").write_text("")
        (bad_dir / "metadata.json").write_text("{}")
    elif scenario == "incomplete":
        index_dir = tmp_path / ("c" * 64) / "index"
        index_dir.mkdir(parents=True)

    with patch("semble.cli.resolve_cache_folder", return_value=tmp_path):
        _run_clear("index")

    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out

    if scenario == "valid":
        assert not (tmp_path / ("a" * 64)).exists()
        assert not (tmp_path / ("b" * 64)).exists()


@pytest.mark.parametrize(
    ("create_file", "expected"),
    [
        (True, "Cleared savings"),
        (False, "No savings file found"),
    ],
)
def test_run_clear_savings(
    create_file: bool, expected: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_clear('savings') deletes the file when present, reports missing otherwise."""
    savings_file = tmp_path / "savings.jsonl"
    if create_file:
        savings_file.write_text('{"tokens": 100}\n')

    with patch("semble.cli.resolve_cache_folder", return_value=tmp_path):
        _run_clear("savings")

    if create_file:
        assert not savings_file.exists()
    out = capsys.readouterr().out
    assert expected in out


@pytest.mark.parametrize(
    ("populate", "expected_fragments"),
    [
        (True, ["Cleared index", "d" * 64, "Cleared savings"]),
        (False, ["No indexes found", "No savings file found"]),
    ],
)
def test_run_clear_all(
    populate: bool, expected_fragments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_clear('all') handles both indexes and savings."""
    if populate:
        _make_valid_index_dir(tmp_path, "d" * 64)
        (tmp_path / "savings.jsonl").write_text('{"tokens": 50}\n')

    with patch("semble.cli.resolve_cache_folder", return_value=tmp_path):
        _run_clear("all")

    out = capsys.readouterr().out
    for fragment in expected_fragments:
        assert fragment in out

    if populate:
        assert not (tmp_path / ("d" * 64)).exists()
        assert not (tmp_path / "savings.jsonl").exists()


@pytest.mark.parametrize(
    ("subcommand", "setup_index", "setup_savings", "expected_fragments"),
    [
        ("index", True, False, ["Cleared index", "e" * 64]),
        ("savings", False, True, ["Cleared savings"]),
        ("all", True, True, ["Cleared index", "Cleared savings"]),
    ],
)
def test_cli_clear_command(
    subcommand: str,
    setup_index: bool,
    setup_savings: bool,
    expected_fragments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `semble clear <subcommand>` CLI dispatches to _run_clear correctly."""
    sha = "e" * 64
    if setup_index:
        _make_valid_index_dir(tmp_path, sha)
    savings_file = tmp_path / "savings.jsonl"
    if setup_savings:
        savings_file.write_text('{"tokens": 200}\n')

    monkeypatch.setattr(sys, "argv", ["semble", "clear", subcommand])
    with patch("semble.cli.resolve_cache_folder", return_value=tmp_path):
        _cli_main()

    out = capsys.readouterr().out
    for fragment in expected_fragments:
        assert fragment in out

    if setup_index:
        assert not (tmp_path / sha).exists()
    if setup_savings:
        assert not savings_file.exists()
