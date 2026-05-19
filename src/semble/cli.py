import argparse
import asyncio
import sys
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from model2vec.utils import get_package_extras

from semble.duplicates.search import DEFAULT_DUPLICATE_MIN_STRUCTURAL_SCORE
from semble.index import SembleIndex
from semble.stats import format_savings_report
from semble.utils import _format_duplicate_clusters, _format_results, _is_git_url, _resolve_chunk

_CLAUDE_FILE_PATH = Path(".claude") / "agents" / "semble-search.md"
_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "find-duplicates", "init", "savings", "-h", "--help"})


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
    parser.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )
    args = parser.parse_args()
    if any(find_spec(dep) is None for dep in get_package_extras("semble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'semble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    asyncio.run(serve(args.path, ref=args.ref, include_text_files=args.include_text_files))


def _run_init(*, force: bool = False) -> None:
    """Write the Claude Code sub-agent file into the current project."""
    dest = _CLAUDE_FILE_PATH
    if dest.exists() and not force:
        print(f"{dest} already exists. Run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath("agents/semble-search.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    search_p.add_argument(
        "-m", "--mode", default="hybrid", choices=["hybrid", "semantic", "bm25"], help="Search mode (default: hybrid)."
    )
    search_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    related_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )

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
    duplicates_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
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

    init_p = sub.add_parser("init", help="Write .claude/agents/semble-search.md for Claude Code sub-agent support.")
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    savings_p = sub.add_parser("savings", help="Show token savings and usage stats.")
    savings_p.add_argument("--verbose", action="store_true", help="Also show usage breakdown by call type.")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(force=args.force)
        return

    if args.command == "savings":
        print(format_savings_report(verbose=args.verbose), end="")
        return

    include_text = args.include_text_files
    index = (
        SembleIndex.from_git(args.path, include_text_files=include_text)
        if _is_git_url(args.path)
        else SembleIndex.from_path(args.path, include_text_files=include_text)
    )

    if args.command == "search":
        results = index.search(args.query, top_k=args.top_k, mode=args.mode)
        if not results:
            print("No results found.")
        else:
            print(_format_results(f"Search results for: {args.query!r} (mode={args.mode})", results))

    elif args.command == "find-related":
        chunk = _resolve_chunk(index.chunks, args.file_path, args.line)
        if chunk is None:
            print(f"No chunk found at {args.file_path}:{args.line}.", file=sys.stderr)
            sys.exit(1)
        results = index.find_related(chunk, top_k=args.top_k)
        if not results:
            print(f"No related chunks found for {args.file_path}:{args.line}.")
        else:
            print(_format_results(f"Chunks related to {args.file_path}:{args.line}", results))

    elif args.command == "find-duplicates":
        print(
            _format_duplicate_clusters(
                "Duplicate clusters",
                index.find_duplicates(
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
                ),
                empty_message="No duplicate clusters found.",
            )
        )
