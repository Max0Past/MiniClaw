"""Save-memory tool: stores facts and preferences to long-term memory."""

from __future__ import annotations

from memory.manager import MemoryManager

# Module-level manager instance; set by AgentCore at startup.
_manager: MemoryManager | None = None


def set_manager(manager: MemoryManager) -> None:
    """Inject the MemoryManager instance (called once during init)."""
    global _manager
    _manager = manager


def _get_manager() -> MemoryManager:
    if _manager is None:
        raise RuntimeError("MemoryManager not initialised. Call set_manager() first.")
    return _manager


def save_memory(text: str) -> str:
    """Store a fact or preference in long-term memory.

    action_input = the text to remember.
    """
    doc_id = _get_manager().save_to_long_term(text)
    return f"Saved to memory (id={doc_id}): {text}"
