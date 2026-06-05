import argparse
import asyncio
import json
import re
import sys
import warnings
from enum import Enum
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree
from typing import Literal

from model2vec.utils import get_package_extras

from semble.cache import find_index_from_cache_folder, resolve_cache_folder
from semble.duplicates.search import DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE
from semble.index import SembleIndex
from semble.index.types import PersistencePath
from semble.stats import format_savings_report
from semble.types import ContentType
from semble.utils import (
    format_duplicate_clusters,
    format_duplicate_clusters_compact,
    format_results,
    is_git_url,
    resolve_chunk,
)

_CLI_DISPATCH_ARGS = frozenset(
    {
        "search",
        "find-related",
        "find-duplicates",
        "init",
        "install",
        "uninstall",
        "savings",
        "-h",
        "--help",
        "clear",
    }
)
_CLEAR_CHOICE = Literal["all", "index", "savings"]


class Agent(str, Enum):
    CLAUDE = "claude"
    COPILOT = "copilot"
    CURSOR = "cursor"
    GEMINI = "gemini"
    KIRO = "kiro"
    OPENCODE = "opencode"


_DEFAULT_AGENT = Agent.CLAUDE
_SHA_256_REGEX = re.compile(r"^[a-f0-9]{64}$")


def _build_index(path: str, content: list[ContentType]) -> SembleIndex:
    """Build an index from a local path or git URL."""
    return (
        SembleIndex.from_git(path, content=content)
        if is_git_url(path)
        else SembleIndex.from_path(path, content=content)
    )


def _maybe_save_index(index: SembleIndex, path: str) -> None:
    """Save the index to the cache folder if it was not loaded from disk."""
    if not index.loaded_from_disk:
        try:
            cache_folder = find_index_from_cache_folder(path)
            index.save(cache_folder)
        except Exception as e:
            print(f"Error saving index: {e}", file=sys.stderr)


def _agent_path(agent: Agent) -> Path:
    """Return the project-relative path where the semble sub-agent file should be written."""
    base_dir = ".github" if agent is Agent.COPILOT else f".{agent.value}"
    return Path(base_dir) / "agents" / "semble-search.md"


def _add_content_args(p: argparse.ArgumentParser) -> None:
    """Add --content and deprecated --include-text-files to a subparser."""
    p.add_argument(
        "--content",
        nargs="+",
        default=["code"],
        choices=[ct.value for ct in ContentType] + ["all"],
        metavar="TYPE",
        help="Content types to index (space-separated, e.g. --content code docs). Choices: code, docs, config, all. Default: code.",
    )
    p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Deprecated. Use --content all instead.",
    )


def main() -> None:
    """Entry point for the semble command-line tool."""
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_DISPATCH_ARGS:
        _cli_main()
    else:
        _mcp_main()


def _mcp_main() -> None:
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Local directory or git URL to pre-index at startup (optional).",
    )
    parser.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    _add_content_args(parser)
    args = parser.parse_args()
    if any(find_spec(dep) is None for dep in get_package_extras("semble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'semble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    content = _resolve_content(args.content, args.include_text_files)
    asyncio.run(serve(args.path, ref=args.ref, content=content))


def _run_init(*, agent: Agent = _DEFAULT_AGENT, force: bool = False) -> None:
    """Write the semble sub-agent file for the given coding agent into the current project."""
    dest = _agent_path(agent)
    if dest.exists() and not force:
        print(f"{dest} already exists. Run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath(f"agents/{agent.value}.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")


def _resolve_content(content: list[str], include_text_files: bool) -> list[ContentType]:
    """Resolve --content and the deprecated --include-text-files into a list of ContentType values."""
    if include_text_files:
        warnings.warn(
            "--include-text-files is deprecated and will be removed in a future version. Use --content all instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if include_text_files or "all" in content:
        return [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]
    return [ContentType(c) for c in content]


def _load_index(path: str, content: list[ContentType]) -> SembleIndex:
    """Build an index from a local path or git URL, exiting on FileNotFoundError."""
    try:
        return _build_index(path, content)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def _run_search(path: str, query: str, top_k: int, content: list[ContentType]) -> None:
    """Handle the `search` subcommand."""
    index = _load_index(path, content)
    results = index.search(query, top_k=top_k)
    out = format_results(query, results) if results else {"error": "No results found."}
    print(json.dumps(out))
    _maybe_save_index(index, path)


def _run_find_related(path: str, file_path: str, line: int, top_k: int, content: list[ContentType]) -> None:
    """Handle the `find-related` subcommand."""
    index = _load_index(path, content)
    chunk = resolve_chunk(index.chunks, file_path, line)
    if chunk is None:
        print(f"No chunk found at {file_path}:{line}.", file=sys.stderr)
        sys.exit(1)
    results = index.find_related(chunk, top_k=top_k)
    out = (
        format_results(f"Chunks related to {file_path}:{line}", results)
        if results
        else {"error": f"No related chunks found for {file_path}:{line}."}
    )
    print(json.dumps(out))
    _maybe_save_index(index, path)


def _run_find_duplicates(args: argparse.Namespace) -> None:
    """Handle the `find-duplicates` subcommand."""
    content = _resolve_content(args.content, args.include_text_files)
    index = _load_index(args.path, content)
    clusters = index.find_duplicates(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        min_lines=args.min_lines,
        min_score=args.min_score,
        min_structural_score=args.min_structural_score,
        min_cluster_size=args.min_cluster_size,
        filter_languages=[args.language] if args.language else None,
        include_paths=args.include_paths,
        exclude_paths=args.exclude_paths,
        include_tests=args.include_tests,
        include_data=args.include_data,
        include_scaffolding=args.include_scaffolding,
    )
    formatter = format_duplicate_clusters_compact if args.detail == "compact" else format_duplicate_clusters
    out = formatter("Duplicate clusters", clusters) if clusters else {"error": "No duplicate clusters found."}
    print(json.dumps(out))
    _maybe_save_index(index, args.path)


def _run_clear(clear_type: _CLEAR_CHOICE) -> None:
    """Run the `clear` subcommand."""
    cache_folder = resolve_cache_folder()
    if clear_type == "index" or clear_type == "all":
        indexes = []
        for path in cache_folder.glob("*/index"):
            if not _SHA_256_REGEX.match(path.parent.name):
                continue
            if PersistencePath.from_path(path).non_existing():
                continue
            indexes.append(path)

        if not indexes:
            print(f"No indexes found to clear in `{cache_folder}`")
        else:
            for path in indexes:
                index_folder = path.parent
                rmtree(index_folder)
                print(f"Cleared index at `{index_folder}`")

    if clear_type == "savings" or clear_type == "all":
        path = cache_folder / "savings.jsonl"
        if not path.exists():
            print(f"No savings file found at `{path}`")
        else:
            path.unlink()
            print(f"Cleared savings at `{path}`")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    _add_content_args(search_p)

    clear_p = sub.add_parser("clear", help="Clear the index cache.")
    clear_p.add_argument("type", choices=["all", "index", "savings"], help="Type of cache to clear.")

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    _add_content_args(related_p)

    duplicates_p = sub.add_parser("find-duplicates", help="Find duplicate-code clusters.")
    duplicates_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    duplicates_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of duplicate clusters (default: 5).")
    duplicates_p.add_argument(
        "--candidate-k",
        type=int,
        default=12,
        help="Semantic neighbors to inspect per chunk before scoring (default: 12).",
    )
    duplicates_p.add_argument("--language", help="Only compare chunks in this language.")
    duplicates_p.add_argument(
        "--include",
        action="append",
        dest="include_paths",
        help="File or directory scope to include in duplicate discovery.",
    )
    duplicates_p.add_argument(
        "--exclude",
        action="append",
        dest="exclude_paths",
        help="File or directory scope to exclude from duplicate discovery.",
    )
    duplicates_p.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in duplicate discovery.",
    )
    duplicates_p.add_argument(
        "--include-data",
        action="store_true",
        help="Include static data/config chunks in duplicate discovery.",
    )
    duplicates_p.add_argument(
        "--include-scaffolding",
        action="store_true",
        help="Include import/header/attribute scaffolding in duplicate discovery.",
    )
    duplicates_p.add_argument("--min-lines", type=int, default=8, help="Minimum lines per chunk (default: 8).")
    duplicates_p.add_argument("--min-score", type=float, default=0.0, help="Minimum duplicate score (default: 0.0).")
    duplicates_p.add_argument(
        "--min-structural-score",
        type=float,
        default=DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE,
        help=f"Minimum structural similarity score (default: {DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE:.2f}).",
    )
    duplicates_p.add_argument("--min-cluster-size", type=int, default=2, help="Minimum chunks per cluster (default: 2).")
    duplicates_p.add_argument(
        "--detail",
        choices=["compact", "full"],
        default="full",
        help="Duplicate output detail level: compact summary or full result JSON (default: full).",
    )
    _add_content_args(duplicates_p)

    init_p = sub.add_parser("init", help="Write a semble sub-agent file for your coding agent.")
    init_p.add_argument(
        "--agent",
        "-a",
        default=_DEFAULT_AGENT.value,
        choices=[a.value for a in Agent],
        help=f"Coding agent to set up (default: {_DEFAULT_AGENT.value}).",
    )
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    savings_p = sub.add_parser("savings", help="Show token savings and usage stats.")
    savings_p.add_argument("--verbose", action="store_true", help="Also show usage breakdown by call type.")

    sub.add_parser("install", help="Interactively configure semble across coding agents.")
    sub.add_parser("uninstall", help="Interactively remove semble configuration from coding agents.")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(agent=Agent(args.agent), force=args.force)
    elif args.command == "savings":
        print(format_savings_report(verbose=args.verbose))
    elif args.command in ("install", "uninstall"):
        from semble.installer import run

        run(args.command)
    elif args.command == "clear":
        _run_clear(args.type)
    elif args.command == "search":
        _run_search(args.path, args.query, args.top_k, _resolve_content(args.content, args.include_text_files))
    elif args.command == "find-related":
        _run_find_related(
            args.path, args.file_path, args.line, args.top_k, _resolve_content(args.content, args.include_text_files)
        )
    elif args.command == "find-duplicates":
        _run_find_duplicates(args)
