from __future__ import annotations

from semble.types import Chunk, DuplicateCluster, DuplicatePair


def cluster_duplicate_pairs(
    pairs: list[DuplicatePair],
    *,
    min_cluster_size: int = 2,
) -> list[DuplicateCluster]:
    """Group duplicate pairs into connected components of chunks."""
    if not pairs:
        return []

    min_cluster_size = max(min_cluster_size, 2)
    chunks_by_key: dict[tuple[str, int, int], Chunk] = {}
    adjacency: dict[tuple[str, int, int], set[tuple[str, int, int]]] = {}

    for pair in pairs:
        left_key = _chunk_key(pair.left.chunk)
        right_key = _chunk_key(pair.right.chunk)
        chunks_by_key.setdefault(left_key, pair.left.chunk)
        chunks_by_key.setdefault(right_key, pair.right.chunk)
        adjacency.setdefault(left_key, set()).add(right_key)
        adjacency.setdefault(right_key, set()).add(left_key)

    clusters: list[DuplicateCluster] = []
    visited: set[tuple[str, int, int]] = set()
    for start_key in sorted(adjacency):
        if start_key in visited:
            continue

        component: set[tuple[str, int, int]] = set()
        stack = [start_key]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency[current] - visited)

        if len(component) < min_cluster_size:
            continue

        component_pairs = tuple(
            sorted(
                (
                    pair
                    for pair in pairs
                    if _chunk_key(pair.left.chunk) in component and _chunk_key(pair.right.chunk) in component
                ),
                key=_duplicate_pair_sort_key,
            )
        )
        members = tuple(chunks_by_key[key] for key in sorted(component))
        clusters.append(DuplicateCluster(members=members, pairs=component_pairs))

    return sorted(clusters, key=_duplicate_cluster_sort_key)


def _same_file_ranges_overlap(left: Chunk, right: Chunk) -> bool:
    if left.file_path != right.file_path:
        return False
    return max(left.start_line, right.start_line) <= min(left.end_line, right.end_line)


def _chunk_key(chunk: Chunk) -> tuple[str, int, int]:
    return (chunk.file_path, chunk.start_line, chunk.end_line)


def _pair_key(left: Chunk, right: Chunk) -> tuple[tuple[str, int, int], tuple[str, int, int]]:
    left_key = _chunk_key(left)
    right_key = _chunk_key(right)
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)


def _duplicate_pair_sort_key(
    pair: DuplicatePair,
) -> tuple[float, float, float, tuple[tuple[str, int, int], tuple[str, int, int]]]:
    return (
        -pair.score,
        -pair.signals.semantic_score,
        -pair.signals.structural_score,
        _pair_key(pair.left.chunk, pair.right.chunk),
    )


def _duplicate_cluster_sort_key(
    cluster: DuplicateCluster,
) -> tuple[float, int, int, tuple[str, int, int]]:
    return (
        -cluster.score,
        -len(cluster.pairs),
        -len(cluster.members),
        _chunk_key(cluster.members[0]),
    )
