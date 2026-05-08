from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.duplicates.search import duplicate_options_from_values
from semble.index import DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE, SembleIndex
from semble.index.dense import load_model
from semble.types import Encoder
from semble.utils import _format_duplicate_search_result, _format_results, _is_git_url, _resolve_chunk

_REPO_DESCRIPTION = (
    "Git URL (e.g. https://github.com/org/repo) or local path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)
_REF_DESCRIPTION = "Branch or tag to check out for git URLs. Ignored for local paths."
_NO_REPO_MESSAGE = (
    "No repo specified and no default index. "
    "Pass a git URL (https://github.com/...) or local path as `repo`."
)


async def _get_index_or_error(
    cache: _IndexCache,
    source: str | None,
    ref: str | None = None,
) -> tuple[SembleIndex | None, str | None]:
    """Return an index for source, or a user-facing MCP error message."""
    if not source:
        return None, _NO_REPO_MESSAGE
    try:
        return await cache.get(source, ref=ref), None
    except Exception as exc:
        return None, f"Failed to index {source!r}: {exc}"


def _resolve_source_ref(
    repo: str | None,
    ref: str | None,
    default_source: str | None,
    default_ref: str | None,
) -> tuple[str | None, str | None]:
    """Return the effective source and ref for an MCP tool call."""
    if repo:
        return repo, ref
    return default_source, ref if ref is not None else default_ref


async def _search_codebase(
    cache: _IndexCache,
    source: str | None,
    ref: str | None,
    query: str,
    mode: Literal["hybrid", "semantic", "bm25"],
    top_k: int,
) -> str:
    """Search a codebase and format MCP text output."""
    index, error = await _get_index_or_error(cache, source, ref=ref)
    if error:
        return error
    assert index is not None

    results = index.search(query, top_k=top_k, mode=mode)
    if not results:
        return "No results found."
    return _format_results(f"Search results for: {query!r} (mode={mode})", results)


async def _find_related_code(
    cache: _IndexCache,
    source: str | None,
    ref: str | None,
    file_path: str,
    line: int,
    top_k: int,
) -> str:
    """Find related code for a source location and format MCP text output."""
    index, error = await _get_index_or_error(cache, source, ref=ref)
    if error:
        return error
    assert index is not None

    chunk = _resolve_chunk(index.chunks, file_path, line)
    if chunk is None:
        return (
            f"No chunk found at {file_path}:{line}. "
            "Make sure the file is indexed and the line number is within a known chunk."
        )
    results = index.find_related(chunk, top_k=top_k)
    if not results:
        return f"No related chunks found for {file_path}:{line}."
    return _format_results(f"Chunks related to {file_path}:{line}", results)


async def _find_duplicate_code(
    cache: _IndexCache,
    source: str | None,
    ref: str | None,
    top_k: int,
    candidate_k: int,
    language: str | None,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
    include_tests: bool,
    include_data: bool,
    include_scaffolding: bool,
    min_lines: int,
    min_score: float,
    min_structural_score: float,
    min_cluster_size: int,
) -> str:
    """Find duplicate-code clusters and format MCP text output."""
    index, error = await _get_index_or_error(cache, source, ref=ref)
    if error:
        return error
    assert index is not None

    options = duplicate_options_from_values(
        top_k=top_k,
        candidate_k=candidate_k,
        language=language,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        include_tests=include_tests,
        include_data=include_data,
        include_scaffolding=include_scaffolding,
        min_lines=min_lines,
        min_score=min_score,
        min_structural_score=min_structural_score,
        min_cluster_size=min_cluster_size,
    )
    clusters = await asyncio.to_thread(
        index.find_duplicates,
        options=options,
    )
    return _format_duplicate_search_result(clusters)


def create_server(
    cache: _IndexCache,
    default_source: str | None = None,
    default_ref: str | None = None,
) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or GitHub repository. "
            "Call `search` to find relevant code; call `find_related` on a result to discover similar code elsewhere. "
            "Call `find_duplicates` to identify grouped duplicate implementations and refactoring candidates. "
            "For questions about a library (e.g. a PyPI/npm package), resolve the GitHub URL from your training "
            "knowledge and pass it as `repo`. "
            "Prefer these tools over Grep, Glob, or Read for any question about how code works."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        mode: Annotated[
            Literal["hybrid", "semantic", "bm25"],
            Field(description="Search mode. 'hybrid' is best for most queries."),
        ] = "hybrid",
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
        ref: Annotated[str | None, Field(description=_REF_DESCRIPTION)] = None,
    ) -> str:
        """Search a codebase with a natural-language or code query.

        Pass a git URL or local path as `repo` to index it on demand; pass `ref` for a git branch or tag.
        Indexes are cached for the session.
        Use this to find where something is implemented, understand a library, or locate related code.
        """
        source, source_ref = _resolve_source_ref(repo, ref, default_source, default_ref)
        return await _search_codebase(cache, source, source_ref, query, mode, top_k)

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
        ref: Annotated[str | None, Field(description=_REF_DESCRIPTION)] = None,
    ) -> str:
        """Find code chunks semantically similar to a specific location in a file.

        Use after `search` to explore related implementations or callers.
        Pass file_path and line from a prior search result.
        """
        source, source_ref = _resolve_source_ref(repo, ref, default_source, default_ref)
        return await _find_related_code(cache, source, source_ref, file_path, line, top_k)

    @server.tool()
    async def find_duplicates(
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        ref: Annotated[str | None, Field(description=_REF_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of duplicate clusters to return.", ge=1)] = 5,
        candidate_k: Annotated[
            int,
            Field(description="Semantic neighbors to inspect per chunk before duplicate scoring.", ge=1),
        ] = 12,
        language: Annotated[str | None, Field(description="Only compare chunks in this language.")] = None,
        include_paths: Annotated[
            list[str] | None,
            Field(description="Repo-relative file or directory scopes to include in duplicate discovery."),
        ] = None,
        exclude_paths: Annotated[
            list[str] | None,
            Field(description="Repo-relative file or directory scopes to exclude from duplicate discovery."),
        ] = None,
        include_tests: Annotated[bool, Field(description="Include test files in duplicate discovery.")] = False,
        include_data: Annotated[
            bool,
            Field(description="Include static data/config chunks in duplicate discovery."),
        ] = False,
        include_scaffolding: Annotated[
            bool,
            Field(description="Include import/header/attribute scaffolding chunks in duplicate discovery."),
        ] = False,
        min_lines: Annotated[int, Field(description="Minimum lines per chunk.", ge=1)] = 8,
        min_score: Annotated[float, Field(description="Minimum duplicate score.", ge=0.0)] = 0.0,
        min_structural_score: Annotated[
            float,
            Field(description="Minimum structural similarity score.", ge=0.0),
        ] = DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
        min_cluster_size: Annotated[int, Field(description="Minimum chunks per cluster.", ge=2)] = 2,
    ) -> str:
        """Find duplicate-code clusters in a codebase.

        Use this to identify grouped duplicate implementations, copy-pasted logic, and refactoring candidates.
        Pass a git URL or local path as `repo` to index it on demand; pass `ref` for a git branch or tag.
        Indexes are cached for the session.
        """
        source, source_ref = _resolve_source_ref(repo, ref, default_source, default_ref)
        return await _find_duplicate_code(
            cache,
            source,
            source_ref,
            top_k,
            candidate_k,
            language,
            include_paths,
            exclude_paths,
            include_tests,
            include_data,
            include_scaffolding,
            min_lines,
            min_score,
            min_structural_score,
            min_cluster_size,
        )

    return server


async def serve(path: str | None = None, ref: str | None = None) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source."""
    model = await asyncio.to_thread(load_model)
    cache = _IndexCache(model=model)
    if path:
        await cache.get(path, ref=ref)

    server = create_server(cache, default_source=path, default_ref=ref)
    await server.run_stdio_async()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process."""

    def __init__(self, model: Encoder) -> None:
        """Initialise an empty cache with a shared embedding model."""
        self._model = model
        self._tasks: dict[str, asyncio.Task[SembleIndex]] = {}

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        is_git = _is_git_url(source)
        cache_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

        if cache_key not in self._tasks:
            if is_git:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_git, source, ref=ref, model=self._model)
                )
            else:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_path, cache_key, model=self._model)
                )
        task = self._tasks[cache_key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self._tasks.pop(cache_key, None)
            raise
        except Exception:
            # Build failed: evict so the next caller can retry.
            self._tasks.pop(cache_key, None)
            raise
