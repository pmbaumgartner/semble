from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class SearchMode(str, Enum):
    """Search mode for SembleIndex.search()."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    BM25 = "bm25"


class Encoder(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: Sequence[str], /) -> EmbeddingMatrix:
        """Encode texts into embeddings as a 2D float32 array."""
        ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single indexable unit of code."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with score and source."""

    chunk: Chunk
    score: float
    source: SearchMode


@dataclass(frozen=True, slots=True)
class DuplicateSignals:
    """Similarity signals used to rank a duplicate candidate pair."""

    semantic_score: float
    structural_score: float
    token_jaccard: float
    ast_type_jaccard: float | None = None
    ast_shape_jaccard: float | None = None


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """A duplicate candidate pair with its final ranking score."""

    left: Chunk
    right: Chunk
    score: float
    signals: DuplicateSignals


@dataclass(frozen=True, slots=True)
class DuplicateCluster:
    """A connected group of duplicate candidate pairs."""

    members: tuple[Chunk, ...]
    pairs: tuple[DuplicatePair, ...]

    @property
    def score(self) -> float:
        """Ranking score for the strongest pair in the cluster."""
        return self.pairs[0].score if self.pairs else 0.0


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Statistics about the current index state."""

    indexed_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
