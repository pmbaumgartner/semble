from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

_TEST_DIR_NAMES = frozenset({"tests", "test", "testing", "__tests__", "spec", "specs", "e2e"})
_TEST_FILE_PATTERNS = ("test_*.*", "*_test.*", "*_tests.*", "*.test.*", "*.spec.*")


def normalize_scope_path(path: str) -> str:
    """Normalize repo-relative path scopes without touching the filesystem."""
    normalized = path.replace("\\", "/").strip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def path_in_scope(file_path: str, scopes: Sequence[str]) -> bool:
    """Return whether a repo-relative path is inside any exact or directory scope."""
    return _path_in_normalized_scopes(normalize_scope_path(file_path), _normalize_scopes(scopes) or ())


def is_test_path(file_path: str) -> bool:
    """Return whether a repo-relative path looks like a test file or test directory."""
    normalized = normalize_scope_path(file_path)
    if not normalized:
        return False
    parts = normalized.split("/")
    if any(part.lower() in _TEST_DIR_NAMES for part in parts[:-1]):
        return True
    filename = parts[-1]
    normalized_filename = filename.lower()
    return any(fnmatchcase(normalized_filename, pattern) for pattern in _TEST_FILE_PATTERNS) or _is_pascal_test_file(
        filename
    )


def _is_pascal_test_file(filename: str) -> bool:
    stem = filename.rsplit(".", maxsplit=1)[0]
    return stem.endswith(("Test", "Tests"))


def path_is_included(
    file_path: str,
    *,
    include_paths: Sequence[str] | None = None,
    exclude_paths: Sequence[str] | None = None,
    include_tests: bool = True,
) -> bool:
    """Return whether a repo-relative path passes include/exclude/test filters."""
    return PathFilter(include_paths, exclude_paths, include_tests=include_tests).includes(file_path)


@dataclass(frozen=True, slots=True, init=False)
class PathFilter:
    """Compiled repo-relative include/exclude/test path filter."""

    include_paths: tuple[str, ...] | None
    exclude_paths: tuple[str, ...] | None
    include_tests: bool

    def __init__(
        self,
        include_paths: Sequence[str] | None = None,
        exclude_paths: Sequence[str] | None = None,
        *,
        include_tests: bool = True,
    ) -> None:
        """Normalize include and exclude scopes once for repeated path checks."""
        object.__setattr__(self, "include_paths", _normalize_scopes(include_paths))
        object.__setattr__(self, "exclude_paths", _normalize_scopes(exclude_paths))
        object.__setattr__(self, "include_tests", include_tests)

    def includes(self, file_path: str) -> bool:
        """Return whether a repo-relative file path passes this filter."""
        normalized_path = normalize_scope_path(file_path)
        if self.include_paths and not _path_in_normalized_scopes(normalized_path, self.include_paths):
            return False
        if self.exclude_paths and _path_in_normalized_scopes(normalized_path, self.exclude_paths):
            return False
        return self.include_tests or not is_test_path(normalized_path)


def _normalize_scopes(scopes: Sequence[str] | None) -> tuple[str, ...] | None:
    if not scopes:
        return None
    return tuple(normalize_scope_path(scope) for scope in scopes)


def _path_in_normalized_scopes(normalized_path: str, normalized_scopes: Sequence[str]) -> bool:
    for normalized_scope in normalized_scopes:
        if normalized_scope in {"", "."}:
            return True
        if normalized_path == normalized_scope or normalized_path.startswith(f"{normalized_scope}/"):
            return True
    return False
