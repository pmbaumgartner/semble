from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semble import DuplicateOptions
from semble.mcp import _CACHE_MAX_SIZE, _IndexCache, create_server, serve
from semble.types import Chunk, DuplicateCluster, DuplicatePair, DuplicateSignals, Encoder, SearchMode, SearchResult
from semble.utils import _format_results, _is_git_url, _resolve_chunk
from tests.conftest import make_chunk


def _tool_text(result: Any) -> str:
    """Extract the text string from a FastMCP call_tool result."""
    return result[0][0].text


async def _call_tool(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    *,
    index_method: str,
    index_return: list[SearchResult | DuplicateCluster],
    index_chunks: list[Chunk] | None = None,
    default_source: str | None = "/some/path",
) -> str:
    """Patch SembleIndex.from_path with a fake index and invoke the tool, returning the text."""
    fake_index = MagicMock()
    getattr(fake_index, index_method).return_value = index_return
    if index_chunks is not None:
        fake_index.chunks = index_chunks
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache, default_source=default_source)
        result = await server.call_tool(tool, args)
    return _tool_text(result)


@pytest.fixture()
def cache() -> _IndexCache:
    """An _IndexCache backed by a stub model."""
    return _IndexCache(model=MagicMock(spec=Encoder))


def _duplicate_result() -> DuplicatePair:
    """Return a representative duplicate result for MCP tests."""
    left = make_chunk("def left():\n    return 1", "src/left.py")
    right = make_chunk("def right():\n    return 1", "src/right.py")
    signals = DuplicateSignals(semantic_score=0.9, structural_score=0.8, token_jaccard=0.7)
    return DuplicatePair(left=left, right=right, score=0.84, signals=signals)


def _duplicate_cluster() -> DuplicateCluster:
    """Return a representative duplicate cluster for MCP tests."""
    result = _duplicate_result()
    return DuplicateCluster(members=(result.left, result.right), pairs=(result,))


def test_resolve_chunk() -> None:
    """_resolve_chunk returns the correct chunk and handles boundary and miss cases."""
    interior = make_chunk("line1\nline2\nline3", "src/a.py")  # start=1, end=3
    boundary = make_chunk("last line", "src/a.py")  # start=1, end=1 (single-line)

    # Line strictly inside a multi-line chunk hits the early-return path.
    assert _resolve_chunk([interior], "src/a.py", 2) is interior

    # Line equal to end_line of a single-line chunk hits the fallback path.
    assert _resolve_chunk([boundary], "src/a.py", 1) is boundary

    # Unknown file returns None.
    assert _resolve_chunk([interior], "src/other.py", 1) is None

    # Line out of range returns None.
    assert _resolve_chunk([interior], "src/a.py", 99) is None


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://github.com/org/repo", True),
        ("http://github.com/org/repo", True),
        ("git://github.com/org/repo", True),
        ("ssh://git@github.com/org/repo", True),
        ("git+ssh://git@github.com/org/repo", True),
        ("file:///tmp/repo", True),
        ("git@github.com:org/repo", True),  # scp-like
        ("/local/path/to/repo", False),
        ("./relative/path", False),
        ("repo_name", False),
    ],
)
def test_is_git_url(path: str, expected: bool) -> None:
    """Remote git URLs are detected; local paths are not."""
    assert _is_git_url(path) is expected


def test_format_results() -> None:
    """_format_results: empty list → header only; with results → numbered fenced blocks with scores."""
    empty_out = _format_results("My header", [])
    assert "My header" in empty_out
    assert "```" not in empty_out

    chunks = [make_chunk(f"def fn_{i}(): pass", f"f{i}.py") for i in range(3)]
    results = [
        SearchResult(chunk=c, score=round(0.1 * (i + 1), 3), source=SearchMode.HYBRID) for i, c in enumerate(chunks)
    ]
    out = _format_results("Results for: 'foo'", results)
    assert "Results for: 'foo'" in out
    assert out.count("```") >= len(results) * 2  # opening + closing fence each
    for i, c in enumerate(chunks, start=1):
        assert f"## {i}." in out
        assert c.content in out
    assert "0.100" in out and "0.200" in out and "0.300" in out


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source", "patch_target"),
    [
        ("local_tmp_path", "from_path"),
        ("https://github.com/org/repo", "from_git"),
    ],
    ids=["local_path", "git_url"],
)
async def test_index_cache_builds_and_caches(
    cache: _IndexCache, tmp_path: Path, source: str, patch_target: str
) -> None:
    """_IndexCache.get() builds via the correct SembleIndex.* entrypoint and caches subsequent calls."""
    resolved_source = str(tmp_path) if source == "local_tmp_path" else source
    fake_index = MagicMock()
    with patch(f"semble.mcp.SembleIndex.{patch_target}", return_value=fake_index) as mock_build:
        first = await cache.get(resolved_source)
        second = await cache.get(resolved_source)
    assert first is fake_index
    assert second is fake_index
    mock_build.assert_called_once()


@pytest.mark.anyio
async def test_index_cache_evicts_on_failure(cache: _IndexCache, tmp_path: Path) -> None:
    """A failed build evicts the entry so the next call can retry."""
    call_count = 0

    def _failing_then_ok(path: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("build failed")
        return MagicMock()

    with patch("semble.mcp.SembleIndex.from_path", side_effect=_failing_then_ok):
        with pytest.raises(RuntimeError, match="build failed"):
            await cache.get(str(tmp_path))
        result = await cache.get(str(tmp_path))
    assert result is not None
    assert call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search", {"query": "foo"}),
        ("find_related", {"file_path": "src/foo.py", "line": 10}),
        ("find_duplicates", {}),
    ],
)
async def test_tool_no_repo_no_default(cache: _IndexCache, tool: str, args: dict[str, object]) -> None:
    """Tools return an error message when no repo and no default source are given."""
    server = create_server(cache, default_source=None)
    result = await server.call_tool(tool, args)
    assert "No repo specified" in _tool_text(result)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search", {"query": "foo", "repo": "https://github.com/x/y"}),
        ("find_related", {"file_path": "src/foo.py", "line": 1, "repo": "https://github.com/x/y"}),
        ("find_duplicates", {"repo": "https://github.com/x/y"}),
    ],
)
async def test_tool_index_failure(cache: _IndexCache, tool: str, args: dict[str, object]) -> None:
    """Tools return a friendly error message when indexing fails."""
    with patch("semble.mcp.SembleIndex.from_git", side_effect=RuntimeError("clone failed")):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    text = _tool_text(result)
    assert "Failed to index" in text
    assert "clone failed" in text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args", "method", "results", "chunks", "expected_substrings"),
    [
        pytest.param(
            "search",
            {"query": "bar"},
            "search",
            [SearchResult(chunk=make_chunk("def bar(): pass", "src/bar.py"), score=0.9, source=SearchMode.HYBRID)],
            None,
            ["bar", "0.900"],
            id="search_with_results",
        ),
        pytest.param(
            "search",
            {"query": "nothing"},
            "search",
            [],
            None,
            ["No results found"],
            id="search_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1},
            "find_related",
            [SearchResult(chunk=make_chunk("class Foo: pass", "src/foo.py"), score=0.8, source=SearchMode.SEMANTIC)],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["src/foo.py:1", "0.800"],
            id="find_related_with_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1},
            "find_related",
            [],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["No related chunks found"],
            id="find_related_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/unknown.py", "line": 1},
            "find_related",
            [],
            [],
            ["No chunk found"],
            id="find_related_unknown_file",
        ),
        pytest.param(
            "find_duplicates",
            {"top_k": 2},
            "find_duplicates",
            [_duplicate_cluster()],
            None,
            ["Duplicate clusters", "src/left.py"],
            id="find_duplicates_with_results",
        ),
        pytest.param(
            "find_duplicates",
            {},
            "find_duplicates",
            [],
            None,
            ["No duplicate clusters found."],
            id="find_duplicates_no_results",
        ),
    ],
)
async def test_tool_output(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    method: str,
    results: list[SearchResult | DuplicateCluster],
    chunks: list[Chunk] | None,
    expected_substrings: list[str],
) -> None:
    """Tools format results (or an empty-state message) through the server."""
    text = await _call_tool(cache, tool, args, index_method=method, index_return=results, index_chunks=chunks)
    for substring in expected_substrings:
        assert substring in text


@pytest.mark.anyio
async def test_find_duplicates_runs_scan_in_thread(cache: _IndexCache) -> None:
    """find_duplicates runs the duplicate scan through asyncio.to_thread with the public MCP options."""
    fake_index = MagicMock()
    with (
        patch.object(cache, "get", new=AsyncMock(return_value=fake_index)) as mock_get,
        patch("semble.mcp.asyncio.to_thread", new=AsyncMock(return_value=[_duplicate_cluster()])) as mock_to_thread,
    ):
        server = create_server(cache, default_source="/some/path")
        result = await server.call_tool(
            "find_duplicates",
            {
                "top_k": 7,
                "candidate_k": 19,
                "language": "python",
                "include_paths": ["src", "lib"],
                "exclude_paths": ["src/generated", "tests"],
                "include_tests": True,
                "include_data": True,
                "include_scaffolding": True,
                "min_lines": 4,
                "min_score": 0.25,
                "min_structural_score": 0.42,
                "min_cluster_size": 3,
            },
        )

    assert "Duplicate clusters" in _tool_text(result)
    mock_get.assert_awaited_once_with("/some/path", ref=None)
    mock_to_thread.assert_awaited_once_with(
        fake_index.find_duplicates,
        options=DuplicateOptions(
            top_k=7,
            candidate_k=19,
            filter_languages=["python"],
            include_paths=["src", "lib"],
            exclude_paths=["src/generated", "tests"],
            include_tests=True,
            include_data=True,
            include_scaffolding=True,
            min_lines=4,
            min_score=0.25,
            min_structural_score=0.42,
            min_cluster_size=3,
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args", "index_method", "index_return", "index_chunks"),
    [
        (
            "search",
            {"query": "foo", "repo": "https://github.com/x/y", "ref": "feature"},
            "search",
            [],
            None,
        ),
        (
            "find_related",
            {
                "file_path": "src/foo.py",
                "line": 1,
                "repo": "https://github.com/x/y",
                "ref": "feature",
            },
            "find_related",
            [],
            [make_chunk("def foo(): pass", "src/foo.py")],
        ),
        (
            "find_duplicates",
            {"repo": "https://github.com/x/y", "ref": "feature"},
            "find_duplicates",
            [],
            None,
        ),
    ],
)
async def test_tools_pass_explicit_ref_to_cache(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    index_method: str,
    index_return: list[SearchResult | DuplicateCluster],
    index_chunks: list[Chunk] | None,
) -> None:
    """MCP tools pass explicit git refs into the cache lookup."""
    fake_index = MagicMock()
    getattr(fake_index, index_method).return_value = index_return
    if index_chunks is not None:
        fake_index.chunks = index_chunks
    with patch.object(cache, "get", new=AsyncMock(return_value=fake_index)) as mock_get:
        server = create_server(cache)
        await server.call_tool(tool, args)

    mock_get.assert_awaited_once_with("https://github.com/x/y", ref="feature")


@pytest.mark.anyio
async def test_default_source_uses_default_ref(cache: _IndexCache) -> None:
    """Default MCP tool calls keep the startup ref attached to the default source."""
    fake_index = MagicMock()
    fake_index.search.return_value = []
    with patch.object(cache, "get", new=AsyncMock(return_value=fake_index)) as mock_get:
        server = create_server(cache, default_source="https://github.com/x/y", default_ref="main")
        await server.call_tool("search", {"query": "foo"})

    mock_get.assert_awaited_once_with("https://github.com/x/y", ref="main")


@pytest.mark.anyio
@pytest.mark.parametrize("with_path", [True, False], ids=["pre_index", "no_path"])
async def test_serve_runs_stdio(tmp_path: Path, with_path: bool) -> None:
    """serve() loads the model, runs stdio, and optionally pre-indexes when a path is given."""
    with (
        patch("semble.mcp.load_model", return_value=MagicMock(spec=Encoder)),
        patch("semble.mcp.SembleIndex.from_path", return_value=MagicMock()),
        patch.object(_IndexCache, "start_watcher", new_callable=AsyncMock),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", new_callable=AsyncMock) as mock_run,
    ):
        await (serve(str(tmp_path)) if with_path else serve())

    mock_run.assert_called_once()


@pytest.mark.anyio
async def test_serve_preserves_default_git_ref() -> None:
    """serve() pre-indexes and exposes the same default git ref to later tool calls."""
    fake_server = MagicMock()
    fake_server.run_stdio_async = AsyncMock()
    with (
        patch("semble.mcp.load_model", return_value=MagicMock(spec=Encoder)),
        patch("semble.mcp.SembleIndex.from_git", return_value=MagicMock()),
        patch("semble.mcp.create_server", return_value=fake_server) as mock_create_server,
    ):
        await serve("https://github.com/x/y", ref="feature")

    mock_create_server.assert_called_once()
    assert mock_create_server.call_args.kwargs["default_source"] == "https://github.com/x/y"
    assert mock_create_server.call_args.kwargs["default_ref"] == "feature"
    fake_server.run_stdio_async.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("repo", "tool", "extra_args"),
    [
        ("file:///home/user/secret", "search", {"query": "foo"}),
        ("ssh://internal-host/repo", "search", {"query": "foo"}),
        ("git@github.com:org/repo", "search", {"query": "foo"}),
        ("file:///home/user/secret", "find_related", {"file_path": "src/foo.py", "line": 1}),
        ("ssh://internal-host/repo", "find_related", {"file_path": "src/foo.py", "line": 1}),
    ],
    ids=["file_search", "ssh_search", "scp_search", "file_find_related", "ssh_find_related"],
)
async def test_tool_rejects_unsafe_repo(
    cache: _IndexCache, repo: str, tool: str, extra_args: dict[str, object]
) -> None:
    """Both tools reject unsafe git transport schemes (ssh://, file://, SCP-form) supplied as repo."""
    server = create_server(cache, default_source=None)
    result = await server.call_tool(tool, {**extra_args, "repo": repo})
    assert "Only https://" in _tool_text(result)


@pytest.mark.anyio
async def test_index_cache_lru_eviction(cache: _IndexCache, tmp_path: Path) -> None:
    """_IndexCache evicts the least-recently-used entry when the cache is full."""
    dirs = [tmp_path / str(i) for i in range(_CACHE_MAX_SIZE + 1)]
    for d in dirs:
        d.mkdir()
    with patch("semble.mcp.SembleIndex.from_path", return_value=MagicMock()):
        for d in dirs[:_CACHE_MAX_SIZE]:
            await cache.get(str(d))
        first_key = str(dirs[0].resolve())
        assert first_key in cache._tasks
        await cache.get(str(dirs[_CACHE_MAX_SIZE]))
    assert first_key not in cache._tasks
    assert len(cache._tasks) == _CACHE_MAX_SIZE


def test_cache_evict(cache: _IndexCache, tmp_path: Path) -> None:
    """evict() removes an existing cache entry by resolved path."""
    key = str(tmp_path.resolve())
    cache._tasks[key] = MagicMock()
    cache.evict(str(tmp_path))
    assert key not in cache._tasks


def test_cache_evict_missing(cache: _IndexCache, tmp_path: Path) -> None:
    """evict() on an unknown path is a no-op."""
    cache.evict(str(tmp_path))  # should not raise


@pytest.mark.anyio
async def test_watch_loop(cache: _IndexCache, tmp_path: Path) -> None:
    """_watch_loop rebuilds on change (inner errors swallowed) and exits cleanly on watcher error."""

    async def fake_awatch(_path: str) -> AsyncGenerator:
        yield set()
        raise RuntimeError("watcher died")

    with patch("semble.mcp.watchfiles.awatch", fake_awatch):
        with patch("semble.mcp.SembleIndex.from_path", side_effect=RuntimeError("build failed")):
            await cache.start_watcher(str(tmp_path))
            assert cache._watcher_task is not None
            await cache._watcher_task
