from __future__ import annotations

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
from bm25s import BM25

from semble.duplicates import (
    DuplicateFeatures,
    _pair_key,
    _same_file_ranges_overlap,
    cluster_duplicate_pairs,
    duplicate_features,
    duplicate_features_are_eligible,
    duplicate_score,
    score_duplicate_features,
)
from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend, load_model
from semble.path_filters import normalize_scope_path, path_in_scope, path_is_included
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.types import Chunk, DuplicateCluster, DuplicateResult, Encoder, IndexStats, SearchMode, SearchResult

DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE = 0.40


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
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
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
        :param include_tests: Whether test-looking paths should be indexed.
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
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_tests=include_tests,
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
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
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
        :param include_tests: Whether test-looking paths should be indexed.
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
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                include_tests=include_tests,
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
        min_structural_score: float = DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
        min_cluster_size: int = 2,
        filter_languages: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = False,
        include_data: bool = False,
        include_scaffolding: bool = False,
    ) -> list[DuplicateCluster]:
        """Return ranked duplicate-code clusters from indexed chunks.

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
        :param include_data: Whether static data/config chunks are eligible duplicate candidates.
        :param include_scaffolding: Whether scaffolding-only chunks are eligible duplicate candidates.
        :return: Ranked list of duplicate clusters, best match first.
        """
        if top_k <= 0:
            return []

        pairs = self._find_duplicate_pairs(
            candidate_k=candidate_k,
            min_lines=min_lines,
            min_score=min_score,
            min_structural_score=min_structural_score,
            filter_languages=filter_languages,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_tests=include_tests,
            include_data=include_data,
            include_scaffolding=include_scaffolding,
        )
        return cluster_duplicate_pairs(pairs, min_cluster_size=min_cluster_size)[:top_k]

    def _find_duplicate_pairs(
        self,
        *,
        candidate_k: int,
        min_lines: int,
        min_score: float,
        min_structural_score: float,
        filter_languages: list[str] | None,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
        include_tests: bool,
        include_data: bool,
        include_scaffolding: bool,
    ) -> list[DuplicateResult]:
        """Return all sorted duplicate candidate pairs without top-k slicing."""
        if not self.chunks:
            return []

        features_by_index, eligible = self._eligible_duplicate_features(
            filter_languages=filter_languages,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_tests=include_tests,
            include_data=include_data,
            include_scaffolding=include_scaffolding,
            min_lines=min_lines,
        )
        if not eligible:
            return []

        candidate_k = max(candidate_k, 1)
        pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicateResult] = {}

        vectors = np.asarray(self._semantic_index.vectors)
        for group in self._duplicate_language_groups(eligible).values():
            self._collect_duplicate_pairs_for_group(
                group=group,
                vectors=vectors,
                features_by_index=features_by_index,
                candidate_k=candidate_k,
                min_score=min_score,
                min_structural_score=min_structural_score,
                pairs=pairs,
            )

        return sorted(pairs.values(), key=self._duplicate_sort_key)

    def _eligible_duplicate_features(
        self,
        *,
        filter_languages: list[str] | None,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
        include_tests: bool,
        include_data: bool,
        include_scaffolding: bool,
        min_lines: int,
    ) -> tuple[dict[int, DuplicateFeatures], list[int]]:
        """Return duplicate features and indices for chunks that pass cheap eligibility gates."""
        features_by_index: dict[int, DuplicateFeatures] = {}
        eligible = []
        for index in self._duplicate_candidate_indices(filter_languages, include_paths, exclude_paths, include_tests):
            if self._line_count(self.chunks[index]) < min_lines:
                continue
            features = duplicate_features(self.chunks[index])
            if not duplicate_features_are_eligible(
                features,
                include_data=include_data,
                include_scaffolding=include_scaffolding,
            ):
                continue
            features_by_index[index] = features
            eligible.append(index)
        return features_by_index, eligible

    def _collect_duplicate_pairs_for_group(
        self,
        *,
        group: list[int],
        vectors: npt.NDArray[np.float32],
        features_by_index: dict[int, DuplicateFeatures],
        candidate_k: int,
        min_score: float,
        min_structural_score: float,
        pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicateResult],
    ) -> None:
        """Collect duplicate pairs from one same-language candidate group."""
        if len(group) < 2:
            return

        selector = np.array(group, dtype=np.int_)
        related = self._semantic_index.query(
            vectors[selector],
            k=min(candidate_k + 1, len(selector)),
            selector=selector,
        )

        for left_index, (indices, distances) in zip(group, related):
            for right_index_raw, distance in zip(indices, distances):
                self._maybe_add_duplicate_pair(
                    left_index=left_index,
                    right_index=int(right_index_raw),
                    semantic_score=1.0 - float(distance),
                    features_by_index=features_by_index,
                    min_score=min_score,
                    min_structural_score=min_structural_score,
                    pairs=pairs,
                )

    def _maybe_add_duplicate_pair(
        self,
        *,
        left_index: int,
        right_index: int,
        semantic_score: float,
        features_by_index: dict[int, DuplicateFeatures],
        min_score: float,
        min_structural_score: float,
        pairs: dict[tuple[tuple[str, int, int], tuple[str, int, int]], DuplicateResult],
    ) -> None:
        """Score and record one duplicate pair if it passes pair-level gates."""
        if right_index == left_index:
            return

        left = self.chunks[left_index]
        right = self.chunks[right_index]
        if _same_file_ranges_overlap(left, right):
            return

        signals = score_duplicate_features(
            features_by_index[left_index],
            features_by_index[right_index],
            semantic_score=semantic_score,
        )
        if signals.structural_score < min_structural_score:
            return
        score = duplicate_score(signals)
        if score <= 0.0 or score < min_score:
            return

        result_left, result_right = self._ordered_duplicate_pair(left, right)
        duplicate = DuplicateResult(left=result_left, right=result_right, score=score, signals=signals)
        pair_key = _pair_key(left, right)
        existing = pairs.get(pair_key)
        if existing is None or self._duplicate_rank_key(duplicate) > self._duplicate_rank_key(existing):
            pairs[pair_key] = duplicate

    def _duplicate_language_groups(self, eligible: list[int]) -> dict[str | None, list[int]]:
        """Group eligible duplicate candidates by exact language."""
        groups: dict[str | None, list[int]] = defaultdict(list)
        for index in eligible:
            groups[self.chunks[index].language].append(index)
        return dict(groups)

    def _duplicate_candidate_indices(
        self,
        filter_languages: list[str] | None,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
        include_tests: bool,
    ) -> list[int]:
        """Return chunk indices eligible for duplicate discovery."""
        eligible = set(range(len(self.chunks)))

        if filter_languages:
            language_indices: set[int] = set()
            for language in filter_languages:
                language_indices.update(self._language_mapping.get(language, []))
            eligible &= language_indices

        path_indices = {
            index
            for index, chunk in enumerate(self.chunks)
            if path_is_included(
                chunk.file_path,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                include_tests=include_tests,
            )
        }
        eligible &= path_indices

        return sorted(eligible)

    @staticmethod
    def _path_in_scope(file_path: str, scopes: list[str]) -> bool:
        """Return whether a repo-relative path is inside any exact or directory scope."""
        return path_in_scope(file_path, scopes)

    @staticmethod
    def _normalize_scope_path(path: str) -> str:
        """Normalize duplicate path scopes without touching the filesystem."""
        return normalize_scope_path(path)

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
