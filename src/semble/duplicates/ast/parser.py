from __future__ import annotations

from semble.chunking.core import is_supported_language

_PARSER_LANGUAGE_ALIASES = {
    "c#": "csharp",
    "c++": "cpp",
    "js": "javascript",
    "sh": "bash",
    "shell": "bash",
    "ts": "typescript",
}


def _parser_language_for_chunk(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = _PARSER_LANGUAGE_ALIASES.get(language.lower(), language.lower())
    if is_supported_language(normalized):
        return normalized
    return None
