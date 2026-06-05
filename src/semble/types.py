from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class CallType(str, Enum):
    """Call type for token-savings tracking."""

    SEARCH = "search"
    FIND_RELATED = "find_related"


class ContentType(str, Enum):
    """Content type for indexing and search pipeline selection."""

    CODE = "code"
    DOCS = "docs"
    CONFIG = "config"


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

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dict."""
        d = asdict(self)
        d["location"] = self.location
        return d

    @classmethod
    def from_dict(cls: type[Chunk], data: dict[str, Any]) -> Chunk:
        """Create a Chunk from a dict."""
        data.pop("location", None)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with score and source."""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Dump a search result to a dict."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class DuplicateSignals:
    """Similarity signals used to rank a duplicate candidate pair."""

    semantic_score: float
    structural_score: float
    token_jaccard: float
    ast_type_jaccard: float | None = None
    ast_shape_jaccard: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Dump duplicate similarity signals to a dict."""
        return {
            "semantic_score": self.semantic_score,
            "structural_score": self.structural_score,
            "token_jaccard": self.token_jaccard,
            "ast_type_jaccard": self.ast_type_jaccard,
            "ast_shape_jaccard": self.ast_shape_jaccard,
        }


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """A duplicate candidate pair with its final ranking score."""

    left: Chunk
    right: Chunk
    score: float
    signals: DuplicateSignals
    left_content: str
    right_content: str

    def to_dict(self) -> dict[str, Any]:
        """Dump a duplicate pair to a JSONable dict."""
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "score": self.score,
            "signals": self.signals.to_dict(),
            "left_content": self.left_content,
            "right_content": self.right_content,
        }


@dataclass(frozen=True, slots=True)
class DuplicateCluster:
    """A connected group of duplicate candidate pairs."""

    members: tuple[Chunk, ...]
    pairs: tuple[DuplicatePair, ...]

    @property
    def score(self) -> float:
        """Ranking score for the strongest pair in the cluster."""
        return self.pairs[0].score if self.pairs else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Dump a duplicate cluster to a JSONable dict."""
        return {
            "score": self.score,
            "members": [member.to_dict() for member in self.members],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Statistics about the current index state."""

    indexed_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
