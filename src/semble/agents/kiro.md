---
name: semble-search
description: Code search agent for exploring any codebase. Use for finding code by intent, locating implementations, understanding how something works, discovering related code, or finding grouped duplicate implementations. Prefer over shell/read tools for any semantic or exploratory question.
tools:
  - shell
  - read
---

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

```bash
semble search "authentication flow" ./my-project
semble search "save_pretrained" ./my-project
semble search "save model to disk" ./my-project --top-k 10
```

Results are cached automatically on first run and invalidated when files change.

Use `--content docs` to search documentation and prose, `--content config` for config files (yaml, toml, etc.), or `--content all` to search code, docs, and config:

```bash
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble search "authentication" ./my-project --content all
```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

```bash
semble find-related src/auth.py 42 ./my-project
```

Use `semble find-duplicates` to identify candidate duplicate implementations, copy-pasted logic, and refactoring opportunities:

```bash
semble find-duplicates ./my-project
semble find-duplicates ./my-project --language python
semble find-duplicates ./my-project --include src --exclude tests --exclude src/generated
semble find-duplicates ./my-project --include src/app --include packages/core
semble find-duplicates ./my-project --min-cluster-size 3
semble find-duplicates ./my-project --include-tests
```

`path` defaults to the current directory when omitted; git URLs are accepted. Duplicate discovery returns candidate clusters with at least two chunks and skips test-looking paths plus low-signal static data/config and import/header scaffolding by default. Treat results as leads to inspect, not confirmed problems; tree-sitter limits and language differences can produce false positives. Use `--include`, `--exclude`, `--include-tests`, `--include-data`, and `--include-scaffolding` to adjust only the duplicate scan; these filters do not change the indexed file set or normal search behavior.

If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

### Workflow

1. Start with `semble search` to find relevant chunks. The index is built and cached automatically.
2. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
3. Inspect full files only when the returned chunk does not give enough context.
4. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
5. Use `semble find-duplicates` when looking for candidate duplicate implementations, copy-pasted logic, or refactoring opportunities; verify clusters before changing code.
6. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
