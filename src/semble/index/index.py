from __future__ import annotations

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
from bm25s import BM25

from semble.duplicates import _pair_key, _same_file_ranges_overlap, duplicate_score, score_duplicate_pair
from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend, load_model
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.types import Chunk, DuplicateResult, Encoder, IndexStats, SearchMode, SearchResult


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
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
        :return: An indexed SembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        model = model or load_model()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        path = path.resolve()
        bm25, vicinity, chunks = create_index_from_path(
            path,
            model=model,
            extensions=extensions,
            ignore=ignore,
            include_text_files=include_text_files,
            display_root=path,
        )

        index = SembleIndex(model, bm25, vicinity, chunks)

        return index

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_text_files: bool = False,
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
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH or the clone fails.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # `--` prevents `url` from being interpreted as a git option (e.g. `--upload-pack=...`).
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")
            model = model or load_model()
            resolved_path = Path(tmp_dir).resolve()
            bm25, vicinity, chunks = create_index_from_path(
                resolved_path,
                model=model,
                extensions=extensions,
                ignore=ignore,
                include_text_files=include_text_files,
                display_root=resolved_path,
            )

            index = SembleIndex(model, bm25, vicinity, chunks)

            return index

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
        top_k: int = 5,
        candidate_k: int = 12,
        min_lines: int = 8,
        min_score: float = 0.0,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        same_language: bool = True,
    ) -> list[DuplicateResult]:
        """Return ranked duplicate-code candidate pairs from indexed chunks.

        :param top_k: Number of duplicate pairs to return.
        :param candidate_k: Number of semantic neighbors to inspect per eligible chunk.
        :param min_lines: Minimum content line count required for each side.
        :param min_score: Minimum final duplicate score to return.
        :param filter_languages: Optional exact language filters.
        :param filter_paths: Optional repo-relative file or directory scopes to include.
        :param exclude_paths: Optional repo-relative file or directory scopes to exclude.
        :param same_language: If True, only compare chunks in the left chunk's language.
        :return: Ranked list of duplicate candidate pairs, best match first.
        """
        if not self.chunks or top_k <= 0:
            return []

        eligible = [
            index
            for index in self._duplicate_candidate_indices(filter_languages, filter_paths, exclude_paths)
            if self._line_count(self.chunks[index]) >= min_lines
        ]
        if not eligible:
            return []

        candidate_k = max(candidate_k, 1)
        pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicateResult] = {}
        for left_index in eligible:
            left = self.chunks[left_index]
            selector = self._duplicate_neighbor_selector(left, eligible, same_language=same_language)
            if selector.size == 0:
                continue

            related = search_semantic(
                left.content,
                self.model,
                self._semantic_index,
                self.chunks,
                min(candidate_k, len(selector)),
                selector,
            )
            for result in related:
                right = result.chunk
                if right == left or _same_file_ranges_overlap(left, right):
                    continue

                signals = score_duplicate_pair(left, right, semantic_score=result.score)
                score = duplicate_score(signals)
                if score <= 0.0 or score < min_score:
                    continue

                result_left, result_right = self._ordered_duplicate_pair(left, right)
                duplicate = DuplicateResult(left=result_left, right=result_right, score=score, signals=signals)
                pair_key = _pair_key(left, right)
                existing = pairs.get(pair_key)
                if existing is None or self._duplicate_rank_key(duplicate) > self._duplicate_rank_key(existing):
                    pairs[pair_key] = duplicate

        return sorted(pairs.values(), key=self._duplicate_sort_key)[:top_k]

    def _duplicate_candidate_indices(
        self,
        filter_languages: list[str] | None,
        filter_paths: list[str] | None,
        exclude_paths: list[str] | None,
    ) -> list[int]:
        """Return chunk indices eligible for duplicate discovery."""
        eligible = set(range(len(self.chunks)))

        if filter_languages:
            language_indices: set[int] = set()
            for language in filter_languages:
                language_indices.update(self._language_mapping.get(language, []))
            eligible &= language_indices

        if filter_paths:
            path_indices = {
                index
                for index, chunk in enumerate(self.chunks)
                if self._path_in_scope(chunk.file_path, filter_paths)
            }
            eligible &= path_indices

        if exclude_paths:
            excluded = {
                index
                for index, chunk in enumerate(self.chunks)
                if self._path_in_scope(chunk.file_path, exclude_paths)
            }
            eligible -= excluded

        return sorted(eligible)

    def _duplicate_neighbor_selector(
        self,
        left: Chunk,
        eligible: list[int],
        *,
        same_language: bool,
    ) -> npt.NDArray[np.int_]:
        """Create a selector of eligible duplicate neighbors for a left chunk."""
        indices = []
        for index in eligible:
            chunk = self.chunks[index]
            if chunk == left:
                continue
            if same_language and left.language and chunk.language != left.language:
                continue
            indices.append(index)
        return np.array(indices, dtype=np.int_)

    @staticmethod
    def _path_in_scope(file_path: str, scopes: list[str]) -> bool:
        """Return whether a repo-relative path is inside any exact or directory scope."""
        normalized_path = SembleIndex._normalize_scope_path(file_path)
        for scope in scopes:
            normalized_scope = SembleIndex._normalize_scope_path(scope)
            if normalized_scope in {"", "."}:
                return True
            if normalized_path == normalized_scope or normalized_path.startswith(f"{normalized_scope}/"):
                return True
        return False

    @staticmethod
    def _normalize_scope_path(path: str) -> str:
        """Normalize duplicate path scopes without touching the filesystem."""
        normalized = path.replace("\\", "/").strip("/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.rstrip("/")

    @staticmethod
    def _line_count(chunk: Chunk) -> int:
        """Return the number of content lines in a chunk."""
        return len(chunk.content.splitlines())

    @staticmethod
    def _ordered_duplicate_pair(left: Chunk, right: Chunk) -> tuple[Chunk, Chunk]:
        """Return pair chunks in stable location order."""
        left_key = (left.file_path, left.start_line, left.end_line)
        right_key = (right.file_path, right.start_line, right.end_line)
        return (left, right) if left_key <= right_key else (right, left)

    @staticmethod
    def _duplicate_rank_key(result: DuplicateResult) -> tuple[float, float, float]:
        """Return the descending score key used to keep the best version of a pair."""
        return (result.score, result.signals.semantic_score, result.signals.structural_score)

    @staticmethod
    def _duplicate_sort_key(result: DuplicateResult) -> tuple[float, float, float, str, str]:
        """Return the final deterministic duplicate result sort key."""
        return (
            -result.score,
            -result.signals.semantic_score,
            -result.signals.structural_score,
            result.left.location,
            result.right.location,
        )

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        """Create a vector of chunk indices to restrict retrieval to."""
        selector = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))

        return np.unique(selector) if selector else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode | str = SearchMode.HYBRID,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
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
        :param filter_paths: Optional list of repo-relative file paths; if set, only
            chunks from these files are returned.
        :return: Ranked list of :class:`SearchResult` objects, best match first.
        :raises ValueError: If `mode` is not a recognised search strategy.
        """
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self.chunks or not query.strip():
            return []

        selector = self._get_selector_vector(filter_languages, filter_paths)

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self.chunks, top_k, selector=selector)
        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, self.model, semantic_index, self.chunks, top_k, selector=selector)
        if mode == SearchMode.HYBRID:
            return search_hybrid(
                query, self.model, semantic_index, bm25_index, self.chunks, top_k, alpha=alpha, selector=selector
            )
        raise ValueError(f"Unknown search mode: {mode!r}")
