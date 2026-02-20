"""To-Do list tool functions consumed by the agent.

Tools:
    todo_read    -- show all lists, or tasks in a specific list
    todo_add     -- add one or more tasks to a list (auto-creates list)
    todo_delete  -- delete a task by ID, or an entire list by name
    todo_toggle  -- invert task status (pending <-> done)
"""

from __future__ import annotations

from db.todo_store import TodoStore

# Module-level store instance; set by AgentCore at startup.
_store: TodoStore | None = None


def set_store(store: TodoStore) -> None:
    """Inject the TodoStore instance (called once during init)."""
    global _store
    _store = store


def _get_store() -> TodoStore:
    if _store is None:
        raise RuntimeError("TodoStore not initialised. Call set_store() first.")
    return _store


# ---- Tools ---------------------------------------------------------------


def todo_read(action_input: str) -> str:
    """Read tasks.

    action_input can be:
      - empty / "all"  -> show all lists with their tasks
      - a list name    -> show only that list's tasks
    """
    items = _get_store().get_all()
    if not items:
        return "No lists or tasks exist yet."

    query = action_input.strip().lower()

    # Group by category
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(item.category, []).append(item)

    # If a specific list is requested
    if query and query != "all":
        # Find the matching list (case-insensitive)
        for cat, cat_items in grouped.items():
            if cat.lower() == query:
                lines = [f"== {cat} =="]
                for item in cat_items:
                    mark = "[x]" if item.status == "done" else "[ ]"
                    lines.append(f"  {mark} {item.id} | {item.text}")
                return "\n".join(lines)
        return f"List '{action_input.strip()}' not found. Available lists: {', '.join(grouped.keys())}"

    # Show all lists
    lines: list[str] = []
    for cat, cat_items in grouped.items():
        lines.append(f"== {cat} ==")
        for item in cat_items:
            mark = "[x]" if item.status == "done" else "[ ]"
            lines.append(f"  {mark} {item.id} | {item.text}")
        lines.append("")
    return "\n".join(lines).strip()


def todo_add(action_input: str) -> str:
    """Add one or more tasks to a list.

    Format: ListName | task1 | task2 | task3
    If only one segment (no pipes): adds to General.
    If the list doesn't exist, it is created automatically.
    """
    parts = [p.strip() for p in action_input.split("|")]

    if len(parts) == 1:
        # No pipe: single task in General
        text = parts[0]
        if not text:
            return "Error: empty task."
        item = _get_store().add(text, category="General")
        return f"Added to 'General': [{item.id}] {item.text}"

    # First part is the list name, rest are tasks
    category = parts[0]
    tasks = [t for t in parts[1:] if t]

    if not category:
        return "Error: empty list name."
    if not tasks:
        return "Error: no tasks provided."

    added = []
    for t in tasks:
        item = _get_store().add(t, category=category)
        added.append(f"  [{item.id}] {item.text}")

    return f"Added {len(added)} task(s) to '{category}':\n" + "\n".join(added)


def todo_delete(action_input: str) -> str:
    """Delete a task by ID or an entire list by name.

    action_input can be:
      - a task ID (8 hex chars, e.g. "a1b2c3d4") -> deletes that task
      - a list name (e.g. "Shopping") -> deletes the entire list
    """
    target = action_input.strip()
    if not target:
        return "Error: specify a task ID or list name."

    # Try as task ID first (8 hex characters)
    deleted = _get_store().delete_item(target)
    if deleted:
        return f"Deleted task '{target}'."

    # Try as list name
    count = _get_store().delete_category(target)
    if count > 0:
        return f"Deleted list '{target}' ({count} task(s) removed)."

    return f"Nothing found with ID or list name '{target}'."


def todo_toggle(action_input: str) -> str:
    """Toggle a task's status between pending and done.

    action_input = task ID (e.g. "a1b2c3d4")
    """
    item_id = action_input.strip()
    if not item_id:
        return "Error: specify a task ID."

    item = _get_store().toggle_status(item_id)
    if item is None:
        return f"No task found with ID '{item_id}'."

    new_status = "done" if item.status == "done" else "pending"
    return f"Toggled [{item.id}] {item.text} -> {new_status}"
