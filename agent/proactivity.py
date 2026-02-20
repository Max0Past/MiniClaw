"""Proactivity engine: checks agent state and suggests actions."""

from __future__ import annotations

from db.todo_store import TodoStore

# Keywords that hint a task is actionable via tools.
_ACTIONABLE_KEYWORDS = {"find", "search", "check", "look up", "get", "fetch"}


class ProactivityEngine:
    """Generates proactive messages based on the agent's current state."""

    def __init__(self, todo_store: TodoStore) -> None:
        self._store = todo_store
        self._startup_checked = False

    def check_on_startup(self) -> str | None:
        """Called once per session. Returns a message if there are pending tasks."""
        if self._startup_checked:
            return None
        self._startup_checked = True

        pending = self._store.get_pending()
        if not pending:
            return None

        count = len(pending)
        if count == 1:
            return (
                f"I see you have an unfinished task: \"{pending[0].text}\". "
                "Want me to work on it?"
            )
        return (
            f"I see you have {count} unfinished tasks. "
            "Want me to help with one of them?"
        )

    def check_after_task_update(self) -> str | None:
        """Called after a to-do mutation. Suggests acting on actionable tasks."""
        pending = self._store.get_pending()
        if not pending:
            return None

        for task in pending:
            text_lower = task.text.lower()
            if any(kw in text_lower for kw in _ACTIONABLE_KEYWORDS):
                return (
                    f"I notice the task \"{task.text}\" looks like something "
                    "I can help with. Shall I do it now?"
                )
        return None
