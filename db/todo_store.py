"""JSON-file backed persistence for to-do items."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODOS_PATH = DATA_DIR / "todos.json"


@dataclass
class TodoItem:
    """A single to-do entry."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str = ""
    category: str = "General"  # New field for grouping
    status: str = "pending"  # "pending" | "done"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None


class TodoStore:
    """CRUD operations for to-do items, persisted as a JSON file."""

    def __init__(self, path: Path = TODOS_PATH) -> None:
        self._path = path

    # -- read    

    def get_all(self) -> list[TodoItem]:
        """Return every to-do item."""
        data = self._load_data()
        return data["items"]

    def get_pending(self) -> list[TodoItem]:
        """Return only items with status 'pending'."""
        return [t for t in self.get_all() if t.status == "pending"]

    def get_categories(self) -> set[str]:
        """Return all known category names (including empty ones)."""
        data = self._load_data()
        cats = set(data["categories"])
        cats.update(i.category for i in data["items"])
        return cats

    def ensure_category(self, category: str) -> None:
        """Register a category name (even if it has no items yet)."""
        data = self._load_data()
        if category not in data["categories"]:
            data["categories"].append(category)
            self._save_data(data)

    # -- write

    def add(self, text: str, category: str = "General") -> TodoItem:
        """Create and persist a new to-do item. Returns the created item."""
        data = self._load_data()
        item = TodoItem(text=text, category=category)
        data["items"].append(item)
        if category not in data["categories"]:
            data["categories"].append(category)
        self._save_data(data)
        return item

    def mark_done(self, item_id: str) -> TodoItem | None:
        """Mark an item as done by its ID. Returns the item, or None."""
        return self.toggle_status(item_id, "done")

    def mark_pending(self, item_id: str) -> TodoItem | None:
        """Mark an item as pending by its ID. Returns the item, or None."""
        return self.toggle_status(item_id, "pending")

    def toggle_status(self, item_id: str, target: str | None = None) -> TodoItem | None:
        """Toggle or set item status. Returns the item, or None."""
        data = self._load_data()
        for item in data["items"]:
            if item.id == item_id:
                if target:
                    item.status = target
                else:
                    item.status = "pending" if item.status == "done" else "done"
                item.completed_at = (
                    datetime.now(timezone.utc).isoformat()
                    if item.status == "done" else None
                )
                self._save_data(data)
                return item
        return None

    def delete_item(self, item_id: str) -> bool:
        """Remove an item by ID. Returns True if found."""
        data = self._load_data()
        initial_count = len(data["items"])
        data["items"] = [i for i in data["items"] if i.id != item_id]
        if len(data["items"]) < initial_count:
            self._save_data(data)
            return True
        return False

    def delete_category(self, category: str) -> int:
        """Remove all items in a category. Returns count of deleted items."""
        data = self._load_data()
        to_keep = [i for i in data["items"] if i.category != category]
        count = len(data["items"]) - len(to_keep)
        data["items"] = to_keep
        # Also remove from category list
        data["categories"] = [c for c in data["categories"] if c != category]
        self._save_data(data)
        return count

    def bulk_add(self, texts: list[str], category: str = "General") -> list[TodoItem]:
        """Add multiple items at once."""
        data = self._load_data()
        new_items = [TodoItem(text=t, category=category) for t in texts if t.strip()]
        data["items"].extend(new_items)
        if category not in data["categories"]:
            data["categories"].append(category)
        self._save_data(data)
        return new_items

    # -- internal ----------------------------------------------------------

    def _load_data(self) -> dict:
        """Read data from disk. Returns {'items': [...], 'categories': [...]}."""
        if not self._path.exists():
            return {"items": [], "categories": ["General"]}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        # Backward compatibility: old format was a plain list of items
        if isinstance(raw, list):
            items = [TodoItem(**entry) for entry in raw]
            cats = list({i.category for i in items} | {"General"})
            return {"items": items, "categories": cats}
        # New format: dict with items + categories
        items = [TodoItem(**entry) for entry in raw.get("items", [])]
        cats = raw.get("categories", ["General"])
        return {"items": items, "categories": cats}

    def _save_data(self, data: dict) -> None:
        """Write data to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": [asdict(item) for item in data["items"]],
            "categories": data["categories"],
        }
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
