"""Sliding-window buffer for the active conversation context."""

from __future__ import annotations

import tiktoken


class ShortTermMemory:
    """Keeps the most recent messages within a token budget.

    Uses tiktoken (cl100k_base) for approximate token counting.
    The system message is always preserved; oldest user/assistant pairs
    are trimmed first when the budget is exceeded.
    """

    MIN_KEEP_PAIRS = 2  # always keep at least this many exchanges

    def __init__(self, max_tokens: int = 4096) -> None:
        self._messages: list[dict] = []
        self._max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    # -- public API --------------------------------------------------------

    def set_system(self, content: str) -> None:
        """Set or replace the system message (always index 0)."""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0] = {"role": "system", "content": content}
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def add(self, role: str, content: str) -> None:
        """Append a message and trim if over budget."""
        self._messages.append({"role": role, "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Return a copy of the current message list."""
        return list(self._messages)

    def token_count(self) -> int:
        """Approximate total tokens across all messages."""
        return sum(self._count(m["content"]) for m in self._messages)

    def to_raw(self) -> list[dict]:
        """Return the raw message list (for debug page display)."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages except the system message."""
        system = None
        if self._messages and self._messages[0]["role"] == "system":
            system = self._messages[0]
        self._messages.clear()
        if system:
            self._messages.append(system)

    # -- internal ----------------------------------------------------------

    def _count(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self._enc.encode(text))

    def _trim(self) -> None:
        """Remove oldest non-system messages until within budget."""
        while self.token_count() > self._max_tokens:
            # Find the first non-system message that is safe to remove.
            # We keep at least MIN_KEEP_PAIRS user/assistant exchanges
            # (counted from the end).
            non_system = [
                i for i, m in enumerate(self._messages) if m["role"] != "system"
            ]
            # Number of user messages among non-system messages
            user_msgs = [i for i in non_system if self._messages[i]["role"] == "user"]
            if len(user_msgs) <= self.MIN_KEEP_PAIRS:
                break  # refuse to trim further
            # Remove the oldest non-system message
            self._messages.pop(non_system[0])