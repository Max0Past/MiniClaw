"""Internet search tool using DuckDuckGo (ddgs package)."""

from __future__ import annotations

import logging
import warnings

from ddgs import DDGS

# Suppress harmless curl_cffi warnings about missing browser impersonation profiles
warnings.filterwarnings("ignore", message=".*Impersonate.*does not exist.*")

logger = logging.getLogger(__name__)


def search_internet(query: str, max_results: int = 3) -> str:
    """Search the web via DuckDuckGo and return formatted results.

    Returns a plain-text summary of titles, URLs, and snippets.
    On failure, returns an error description (never raises).
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        lines: list[str] = []
        for r in results:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"Title: {title}\nURL: {href}\n{body}\n")
        return "\n".join(lines).strip()
    except Exception as exc:
        logger.warning("Search failed: %s", exc)
        return f"Search error: {exc}"
