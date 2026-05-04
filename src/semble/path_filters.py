from __future__ import annotations

from fnmatch import fnmatchcase

_TEST_DIR_NAMES = frozenset({"tests", "test", "testing", "__tests__", "spec", "specs", "e2e"})
_TEST_FILE_PATTERNS = ("test_*.*", "*_test.*", "*_tests.*", "*.test.*", "*.spec.*")


def normalize_scope_path(path: str) -> str:
    """Normalize repo-relative path scopes without touching the filesystem."""
    normalized = path.replace("\\", "/").strip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def path_in_scope(file_path: str, scopes: list[str]) -> bool:
    """Return whether a repo-relative path is inside any exact or directory scope."""
    normalized_path = normalize_scope_path(file_path)
    for scope in scopes:
        normalized_scope = normalize_scope_path(scope)
        if normalized_scope in {"", "."}:
            return True
        if normalized_path == normalized_scope or normalized_path.startswith(f"{normalized_scope}/"):
            return True
    return False


def path_may_contain_scope(dir_path: str, scopes: list[str]) -> bool:
    """Return whether a repo-relative directory might contain an included scope."""
    normalized_dir = normalize_scope_path(dir_path)
    if normalized_dir in {"", "."}:
        return True
    for scope in scopes:
        normalized_scope = normalize_scope_path(scope)
        if normalized_scope in {"", "."}:
            return True
        if normalized_dir == normalized_scope:
            return True
        if normalized_dir.startswith(f"{normalized_scope}/"):
            return True
        if normalized_scope.startswith(f"{normalized_dir}/"):
            return True
    return False


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


def is_test_dir_path(dir_path: str) -> bool:
    """Return whether a repo-relative directory path is inside a test-looking directory."""
    normalized = normalize_scope_path(dir_path)
    if not normalized:
        return False
    return any(part.lower() in _TEST_DIR_NAMES for part in normalized.split("/"))


def _is_pascal_test_file(filename: str) -> bool:
    stem = filename.rsplit(".", maxsplit=1)[0]
    return stem.endswith(("Test", "Tests"))


def path_is_included(
    file_path: str,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    include_tests: bool = True,
) -> bool:
    """Return whether a repo-relative path passes include/exclude/test filters."""
    if include_paths and not path_in_scope(file_path, include_paths):
        return False
    if exclude_paths and path_in_scope(file_path, exclude_paths):
        return False
    return include_tests or not is_test_path(file_path)
