import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
from vicinity.backends.basic import BasicArgs

from semble.index.chunker import chunk_source
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import filter_extensions, language_for_path, walk_files
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk, Encoder


@dataclass(frozen=True, slots=True)
class IndexBuildOptions:
    """Options controlling which files are included while building an index."""

    extensions: frozenset[str] | None = None
    ignore: frozenset[str] | None = None
    include_text_files: bool = False
    include_paths: Sequence[str] | None = None
    exclude_paths: Sequence[str] | None = None
    include_tests: bool = True


@dataclass(frozen=True, slots=True)
class BuiltIndex:
    """Indexes and chunks produced by a filesystem scan."""

    bm25: bm25s.BM25
    semantic: SelectableBasicBackend
    chunks: list[Chunk]


def create_index_from_path(
    path: Path,
    model: Encoder,
    extensions: frozenset[str] | None = None,
    ignore: frozenset[str] | None = None,
    include_text_files: bool = False,
    display_root: Path | None = None,
    include_paths: Sequence[str] | None = None,
    exclude_paths: Sequence[str] | None = None,
    include_tests: bool = True,
) -> tuple[bm25s.BM25, SelectableBasicBackend, list[Chunk]]:
    """Create an index from a resolved directory, optionally storing chunk paths relative to display_root.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param extensions: File extensions to include.
    :param ignore: Directory names to skip.
    :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param include_paths: Optional repo-relative file or directory scopes to include.
    :param exclude_paths: Optional repo-relative file or directory scopes to exclude.
    :param include_tests: Whether test-looking paths should be indexed.
    :raises ValueError: if no items were found, no index can be created.
    :return: A bm25 index, vicinity index and list of chunks
    """
    options = IndexBuildOptions(
        extensions=extensions,
        ignore=ignore,
        include_text_files=include_text_files,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        include_tests=include_tests,
    )
    built = build_index_from_path(path, model, options=options, display_root=display_root)
    return built.bm25, built.semantic, built.chunks


def build_index_from_path(
    path: Path,
    model: Encoder,
    *,
    options: IndexBuildOptions,
    display_root: Path | None = None,
) -> BuiltIndex:
    """Create an index bundle from a resolved directory."""
    extensions = filter_extensions(options.extensions, include_text_files=options.include_text_files)
    chunks: list[Chunk] = []

    for file_path in walk_files(
        path,
        extensions,
        options.ignore,
        include_paths=options.include_paths,
        exclude_paths=options.exclude_paths,
        include_tests=options.include_tests,
    ):
        language = language_for_path(file_path)
        with contextlib.suppress(OSError):
            source = file_path.read_text(encoding="utf-8", errors="replace")
            chunk_path = file_path.relative_to(display_root) if display_root else file_path
            chunks.extend(chunk_source(source, str(chunk_path), language))

    if chunks:
        embeddings = embed_chunks(model, chunks)
        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(enrich_for_bm25(chunk)) for chunk in chunks],
            show_progress=False,
        )
        args = BasicArgs()
        semantic_index = SelectableBasicBackend(embeddings, args)
    else:
        raise ValueError(f"No supported files found under {path}.")

    return BuiltIndex(bm25=bm25_index, semantic=semantic_index, chunks=chunks)
