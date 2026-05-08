from semble.duplicates.clustering import cluster_duplicate_pairs
from semble.duplicates.scoring import (
    DuplicateFeatures,
    duplicate_features,
    duplicate_features_are_eligible,
    duplicate_score,
    score_duplicate_features,
    score_duplicate_pair,
)
from semble.duplicates.search import (
    DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
    DuplicateOptions,
    find_duplicate_pairs,
)

__all__ = [
    "DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE",
    "DuplicateFeatures",
    "DuplicateOptions",
    "cluster_duplicate_pairs",
    "duplicate_features",
    "duplicate_features_are_eligible",
    "duplicate_score",
    "find_duplicate_pairs",
    "score_duplicate_features",
    "score_duplicate_pair",
]
