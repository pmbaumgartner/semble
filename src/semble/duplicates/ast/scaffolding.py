from __future__ import annotations

from typing import Any

from semble.duplicates.ast.taxonomy import _is_strippable_scaffolding_node


def _strip_scaffolding_content(content: str, root: Any) -> str:
    source = content.encode("utf-8", errors="ignore")
    ranges: list[tuple[int, int]] = []
    _collect_scaffolding_removal_ranges(source, root, ranges)
    if not ranges:
        return content

    stripped = bytearray()
    cursor = 0
    for start, end in _merge_ranges(ranges):
        stripped.extend(source[cursor:start])
        cursor = end
    stripped.extend(source[cursor:])
    return stripped.decode("utf-8", errors="ignore")


def _collect_scaffolding_removal_ranges(source: bytes, node: Any, ranges: list[tuple[int, int]]) -> None:
    if _is_strippable_scaffolding_node(node.type):
        ranges.append(_expand_removal_range(source, node.start_byte, node.end_byte))
        return
    for child in node.children:
        _collect_scaffolding_removal_ranges(source, child, ranges)


def _expand_removal_range(source: bytes, start: int, end: int) -> tuple[int, int]:
    line_start = source.rfind(b"\n", 0, start) + 1
    line_end = source.find(b"\n", end)
    if line_end == -1:
        line_end = len(source)
    else:
        line_end += 1

    if not source[line_start:start].strip() and not source[end:line_end].strip():
        return line_start, line_end
    return start, end


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged
