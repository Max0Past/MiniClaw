"""Tool registry: defines tools and dispatches calls from the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolDefinition:
    """Description of a single tool the agent can invoke."""

    name: str
    description: str
    parameter_description: str  # human-readable description of action_input
    execute: Callable[[str], str]


class ToolRegistry:
    """Holds all registered tools and generates prompt descriptions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_prompt_description(self) -> str:
        """Format all tools for injection into the system prompt."""
        lines: list[str] = []
        for tool in self._tools.values():
            lines.append(
                f"- {tool.name}: {tool.description} "
                f"(action_input: {tool.parameter_description})"
            )
        return "\n".join(lines)
