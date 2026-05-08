from __future__ import annotations

import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from bm25s import BM25

from semble.duplicates.clustering import cluster_duplicate_pairs
from semble.duplicates.search import (
    DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
    DuplicateOptions,
    duplicate_options_from_values,
    find_duplicate_pairs,
)
from semble.index.create import IndexBuildOptions, build_index_from_path
from semble.index.dense import SelectableBasicBackend, load_model
from semble.path_filters import PathFilter
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.types import Chunk, DuplicateCluster, Encoder, IndexStats, SearchMode, SearchResult


class _DuplicateOptionDefault:
    """Sentinel wrapper that displays like the public default value."""

    def __init__(self, value: object) -> None:
        """Store the public default value for omitted keyword detection."""
        self.value = value

    def __repr__(self) -> str:
        """Return the public default representation for introspection."""
        return repr(self.value)


def _duplicate_option_value(value: object) -> object:
    """Return the public default value for an omitted duplicate option keyword."""
    if isinstance(value, _DuplicateOptionDefault):
        return value.value
    return value


_DEFAULT_DUPLICATE_TOP_K = _DuplicateOptionDefault(5)
_DEFAULT_DUPLICATE_CANDIDATE_K = _DuplicateOptionDefault(12)
_DEFAULT_DUPLICATE_MIN_LINES = _DuplicateOptionDefault(8)
_DEFAULT_DUPLICATE_MIN_SCORE = _DuplicateOptionDefault(0.0)
_DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE = _DuplicateOptionDefault(DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE)
_DEFAULT_DUPLICATE_MIN_CLUSTER_SIZE = _DuplicateOptionDefault(2)
_DEFAULT_DUPLICATE_FILTER_LANGUAGES = _DuplicateOptionDefault(None)
_DEFAULT_DUPLICATE_INCLUDE_PATHS = _DuplicateOptionDefault(None)
_DEFAULT_DUPLICATE_EXCLUDE_PATHS = _DuplicateOptionDefault(None)
_DEFAULT_DUPLICATE_INCLUDE_TESTS = _DuplicateOptionDefault(False)
_DEFAULT_DUPLICATE_INCLUDE_DATA = _DuplicateOptionDefault(False)
_DEFAULT_DUPLICATE_INCLUDE_SCAFFOLDING = _DuplicateOptionDefault(False)

_GIT_CLONE_TIMEOUT = int(os.environ.get("SEMBLE_CLONE_TIMEOUT", 60))


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: Encoder,
        bm25_index: BM25,
        semantic_index: SelectableBasicBackend,
        chunks: list[Chunk],
    ) -> None:
        """Internal constructor — use :meth:`from_path` or :meth:`from_git`.

        :param model: Embedding model to use.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        """
        self.model: Encoder = model
        self.chunks: list[Chunk] = chunks
        self._bm25_index: BM25 = bm25_index
        self._semantic_index: SelectableBasicBackend = semantic_index
        self._file_mapping, self._language_mapping = self._populate_mapping()

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Build (file → chunk indices, language → chunk indices) mappings, in that order."""
        language_to_id = defaultdict(list)
        file_to_id = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            language = chunk.language
            if language:
                language_to_id[language].append(i)
            file_to_id[chunk.file_path].append(i)

        return dict(file_to_id), dict(language_to_id)

    @property
    def stats(self) -> IndexStats:
        """Stats of an index."""
        language_counts: dict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            if chunk.language:
                language_counts[chunk.language] += 1

        return IndexStats(
            indexed_files=len(self._file_mapping),
            total_chunks=len(self.chunks),
            languages=dict(language_counts),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_text_files: bool = False,
        include_paths: Sequence[str] | None = None,
        exclude_paths: Sequence[str] | None = None,
        include_tests: bool = True,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
        :param include_paths: Optional repo-relative file or directory scopes to include.
        :param exclude_paths: Optional repo-relative file or directory scopes to exclude.
        :param include_tests: Whether test-looking paths should be indexed. Defaults to
            True; duplicate discovery still skips test-looking chunks unless
            ``find_duplicates(include_tests=True)`` is passed.
        :return: An indexed SembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        model = model or load_model()
        options = IndexBuildOptions(
            extensions=extensions,
            ignore=ignore,
            include_text_files=include_text_files,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_tests=include_tests,
        )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        return cls._from_resolved_path(path.resolve(), model, options)

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_text_files: bool = False,
        include_paths: Sequence[str] | None = None,
        exclude_paths: Sequence[str] | None = None,
        include_tests: bool = True,
    ) -> SembleIndex:
        """Clone a git repository and index it.

        The repository is cloned into a temporary directory that is removed once
        indexing finishes. Chunk content is preserved in-memory, but
        ``chunk.file_path`` will not point to a readable file after this call
        returns — it is a repo-relative label, not a filesystem path.

        :param url: URL of the git repository to clone (any git provider).
        :param ref: Branch or tag to check out. Defaults to the remote HEAD.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
        :param include_paths: Optional repo-relative file or directory scopes to include.
        :param exclude_paths: Optional repo-relative file or directory scopes to exclude.
        :param include_tests: Whether test-looking paths should be indexed. Defaults to
            True; duplicate discovery still skips test-looking chunks unless
            ``find_duplicates(include_tests=True)`` is passed.
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH, the clone fails, or times out.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # `--` prevents `url` from being interpreted as a git option (e.g. `--upload-pack=...`).
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=_GIT_CLONE_TIMEOUT
                )
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"git clone timed out for {url!r} (limit: {_GIT_CLONE_TIMEOUT} s)") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")
            model = model or load_model()
            resolved_path = Path(tmp_dir).resolve()
            options = IndexBuildOptions(
                extensions=extensions,
                ignore=ignore,
                include_text_files=include_text_files,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                include_tests=include_tests,
            )
            return cls._from_resolved_path(resolved_path, model, options)

    @classmethod
    def _from_resolved_path(
        cls,
        path: Path,
        model: Encoder,
        options: IndexBuildOptions,
    ) -> SembleIndex:
        """Create a SembleIndex from a resolved directory and shared build options."""
        built = build_index_from_path(path, model=model, options=options, display_root=path)
        return cls(model, built.bm25, built.semantic, built.chunks)

    def find_related(self, source: Chunk | SearchResult, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the given chunk or search result.

        :param source: A SearchResult or Chunk to use as the seed.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = source.chunk if isinstance(source, SearchResult) else source
        selector = self._get_selector_vector(filter_languages=[target.language]) if target.language else None
        results = search_semantic(target.content, self.model, self._semantic_index, self.chunks, top_k + 1, selector)
        return [r for r in results if r.chunk != target][:top_k]

    def find_duplicates(
        self,
        *,
        options: DuplicateOptions | None = None,
        top_k: int = cast(int, _DEFAULT_DUPLICATE_TOP_K),
        candidate_k: int = cast(int, _DEFAULT_DUPLICATE_CANDIDATE_K),
        min_lines: int = cast(int, _DEFAULT_DUPLICATE_MIN_LINES),
        min_score: float = cast(float, _DEFAULT_DUPLICATE_MIN_SCORE),
        min_structural_score: float = cast(float, _DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE),
        min_cluster_size: int = cast(int, _DEFAULT_DUPLICATE_MIN_CLUSTER_SIZE),
        filter_languages: Sequence[str] | None = cast(Sequence[str] | None, _DEFAULT_DUPLICATE_FILTER_LANGUAGES),
        include_paths: Sequence[str] | None = cast(Sequence[str] | None, _DEFAULT_DUPLICATE_INCLUDE_PATHS),
        exclude_paths: Sequence[str] | None = cast(Sequence[str] | None, _DEFAULT_DUPLICATE_EXCLUDE_PATHS),
        include_tests: bool = cast(bool, _DEFAULT_DUPLICATE_INCLUDE_TESTS),
        include_data: bool = cast(bool, _DEFAULT_DUPLICATE_INCLUDE_DATA),
        include_scaffolding: bool = cast(bool, _DEFAULT_DUPLICATE_INCLUDE_SCAFFOLDING),
    ) -> list[DuplicateCluster]:
        """Return ranked duplicate-code clusters from indexed chunks.

        :param options: Optional pre-built duplicate discovery options. Do not combine
            this with duplicate option keyword arguments.
        :param top_k: Number of duplicate clusters to return.
        :param candidate_k: Number of semantic neighbors to inspect per eligible chunk.
        :param min_lines: Minimum content line count required for each side.
        :param min_score: Minimum final duplicate score for a pair edge.
        :param min_structural_score: Minimum structural similarity score for a pair edge.
        :param min_cluster_size: Minimum number of chunks in a returned cluster.
        :param filter_languages: Optional exact language filters.
        :param include_paths: Optional repo-relative file or directory scopes to include.
        :param exclude_paths: Optional repo-relative file or directory scopes to exclude.
        :param include_tests: Whether test-looking paths are eligible duplicate candidates.
            This is independent of indexing; tests may be present in the index but are
            skipped here by default.
        :param include_data: Whether static data/config chunks are eligible duplicate candidates.
        :param include_scaffolding: Whether scaffolding-only chunks are eligible duplicate candidates.
        :return: Ranked list of duplicate clusters, best match first.
        :raises ValueError: If `options` is combined with duplicate option keyword arguments.
        """
        duplicate_kwargs = {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "min_lines": min_lines,
            "min_score": min_score,
            "min_structural_score": min_structural_score,
            "min_cluster_size": min_cluster_size,
            "filter_languages": filter_languages,
            "include_paths": include_paths,
            "exclude_paths": exclude_paths,
            "include_tests": include_tests,
            "include_data": include_data,
            "include_scaffolding": include_scaffolding,
        }
        if options is not None:
            mixed = [
                name for name, value in duplicate_kwargs.items() if not isinstance(value, _DuplicateOptionDefault)
            ]
            if mixed:
                raise ValueError("Pass either options or duplicate option keyword arguments, not both.")
            search_options = options
        else:
            search_options = duplicate_options_from_values(
                top_k=cast(int, _duplicate_option_value(top_k)),
                candidate_k=cast(int, _duplicate_option_value(candidate_k)),
                min_lines=cast(int, _duplicate_option_value(min_lines)),
                min_score=cast(float, _duplicate_option_value(min_score)),
                min_structural_score=cast(float, _duplicate_option_value(min_structural_score)),
                min_cluster_size=cast(int, _duplicate_option_value(min_cluster_size)),
                filter_languages=cast(Sequence[str] | None, _duplicate_option_value(filter_languages)),
                include_paths=cast(Sequence[str] | None, _duplicate_option_value(include_paths)),
                exclude_paths=cast(Sequence[str] | None, _duplicate_option_value(exclude_paths)),
                include_tests=cast(bool, _duplicate_option_value(include_tests)),
                include_data=cast(bool, _duplicate_option_value(include_data)),
                include_scaffolding=cast(bool, _duplicate_option_value(include_scaffolding)),
            )
        if search_options.top_k <= 0:
            return []

        pairs = find_duplicate_pairs(
            self.chunks,
            self._semantic_index,
            self._language_mapping,
            search_options,
        )
        return cluster_duplicate_pairs(pairs, min_cluster_size=search_options.min_cluster_size)[: search_options.top_k]

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        """Create a vector of chunk indices to restrict retrieval to."""
        selector = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))

        if selector:
            return np.unique(selector)
        return np.array([], dtype=np.int_) if filter_languages or filter_paths else None

    def _get_search_selector_vector(
        self,
        *,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> npt.NDArray[np.int_] | None:
        """Create a search selector while keeping exact-file and scoped path filters distinct."""
        if filter_paths is not None and (include_paths is not None or exclude_paths is not None):
            raise ValueError(
                "Use either filter_paths for exact indexed file paths or "
                "include_paths/exclude_paths for file-or-directory scopes, not both."
            )

        if include_paths is None and exclude_paths is None:
            return self._get_selector_vector(filter_languages, filter_paths)

        eligible = set(range(len(self.chunks)))
        if filter_languages:
            language_indices: set[int] = set()
            for language in filter_languages:
                language_indices.update(self._language_mapping.get(language, []))
            eligible &= language_indices

        path_filter = PathFilter(include_paths, exclude_paths)
        scoped_indices = {
            index
            for index, chunk in enumerate(self.chunks)
            if path_filter.includes(chunk.file_path)
        }
        eligible &= scoped_indices
        return np.array(sorted(eligible), dtype=np.int_)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode | str = SearchMode.HYBRID,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param mode: Search strategy — "hybrid" (default), "semantic", or "bm25".
        :param alpha: Blend weight for hybrid score combination; 1.0 = full semantic
            weight, 0.0 = full BM25 weight. File-path penalties and diversity reranking
            are applied regardless. ``None`` auto-detects from query type.
        :param filter_languages: Optional list of language codes; if set, only chunks in
            these languages are returned.
        :param filter_paths: Optional list of exact repo-relative indexed file paths; if
            set, only chunks from these files are returned. This legacy exact-file filter
            cannot be combined with include_paths or exclude_paths.
        :param include_paths: Optional repo-relative file or directory scopes to include.
            Uses directory-scope semantics and intersects with filter_languages.
        :param exclude_paths: Optional repo-relative file or directory scopes to exclude.
            Uses directory-scope semantics and intersects with filter_languages.
        :return: Ranked list of :class:`SearchResult` objects, best match first.
        :raises ValueError: If `mode` is not a recognised search strategy, or if exact
            filter_paths are combined with scoped include_paths/exclude_paths.
        """
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self.chunks or not query.strip():
            return []

        selector = self._get_search_selector_vector(
            filter_languages=filter_languages,
            filter_paths=filter_paths,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
        )
        if selector is not None and len(selector) == 0:
            return []

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self.chunks, top_k, selector=selector)
        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, self.model, semantic_index, self.chunks, top_k, selector=selector)
        if mode == SearchMode.HYBRID:
            return search_hybrid(
                query, self.model, semantic_index, bm25_index, self.chunks, top_k, alpha=alpha, selector=selector
            )
        raise ValueError(f"Unknown search mode: {mode!r}")
