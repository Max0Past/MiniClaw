"""AgentCore: top-level facade that wires everything together."""

from __future__ import annotations

from agent.prompts import build_system_prompt
from agent.proactivity import ProactivityEngine
from agent.reasoning import AgentResponse, ReasoningLoop, ThoughtStep
from config.settings import AppSettings
from db.todo_store import TodoItem, TodoStore
from llm.ollama_client import OllamaClient
from memory.long_term import LongTermMemory, MemoryRecord, MemoryResult
from memory.manager import MemoryManager
from memory.short_term import ShortTermMemory
from tools import memory_tool as memory_tool_mod
from tools import todo as todo_tool_mod
from tools.memory_tool import save_memory
from tools.registry import ToolDefinition, ToolRegistry
from tools.search import search_internet


class AgentCore:
    """Public API consumed by the Streamlit UI layer.

    Constructs and owns all sub-components.  The UI should never
    instantiate OllamaClient, MemoryManager, etc. directly.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

        # -- LLM --
        self._client = OllamaClient(settings.ollama)

        # -- Persistence --
        self._todo_store = TodoStore()

        # -- Memory --
        self._stm = ShortTermMemory(max_tokens=settings.ollama.context_window)
        self._ltm = LongTermMemory()
        self._memory = MemoryManager(stm=self._stm, ltm=self._ltm)

        # -- Tools --
        self._tools = ToolRegistry()
        self._register_tools()

        # -- Agent --
        self._loop = ReasoningLoop(
            client=self._client,
            memory=self._memory,
            tools=self._tools,
        )
        self._proactivity = ProactivityEngine(todo_store=self._todo_store)

        # -- Last trace (for debug page) --
        self._last_trace: list[ThoughtStep] = []

        # -- Build initial system prompt --
        self._refresh_system_prompt()

    # -- message handling --------------------------------------------------

    def handle_message(self, user_input: str) -> AgentResponse:
        """Process a user message through the full pipeline."""
        self._refresh_system_prompt()
        response = self._loop.run(user_input)
        self._last_trace = response.thought_trace
        return response

    # -- proactivity -------------------------------------------------------

    def get_proactive_message(self) -> str | None:
        """Check startup / task-update triggers."""
        msg = self._proactivity.check_on_startup()
        if msg:
            return msg
        return self._proactivity.check_after_task_update()

    # -- debug accessors ---------------------------------------------------

    def get_working_memory(self) -> list[dict]:
        """Return raw STM messages for the debug page."""
        return self._memory.get_working_memory()

    def get_long_term_records(self) -> list[MemoryRecord]:
        """Return all vector-store entries."""
        return self._memory.get_long_term_records()

    def query_long_term(self, query: str, n: int = 5) -> list[MemoryResult]:
        """Search long-term memory (for the debug page query tester)."""
        return self._memory.query_long_term(query, n)

    def get_thought_trace(self) -> list[ThoughtStep]:
        """Return the most recent reasoning trace."""
        return self._last_trace

    def get_todos(self) -> list[TodoItem]:
        """Return all to-do items."""
        return self._todo_store.get_all()

    def delete_memory(self, doc_id: str) -> None:
        """Delete a long-term memory entry by ID."""
        self._memory.ltm.delete(doc_id)

    # -- settings ----------------------------------------------------------

    def reload_settings(self, settings: AppSettings) -> None:
        """Hot-reload persona / model settings without recreating memory."""
        self._settings = settings
        self._client.model = settings.ollama.model
        self._refresh_system_prompt()

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        return self._client.health_check()

    def list_models(self) -> list[str]:
        """Return available Ollama model names."""
        return self._client.list_models()

    # -- internal ----------------------------------------------------------

    def _refresh_system_prompt(self) -> None:
        """Rebuild and set the system prompt from current settings."""
        prompt = build_system_prompt(
            persona=self._settings.persona,
            user=self._settings.user,
            tools_description=self._tools.to_prompt_description(),
        )
        self._memory.set_system(prompt)

    def _register_tools(self) -> None:
        """Wire up all built-in tools."""
        # Inject dependencies into tool modules.
        todo_tool_mod.set_store(self._todo_store)
        memory_tool_mod.set_manager(self._memory)

        self._tools.register(
            ToolDefinition(
                name="search_internet",
                description="Search the web. Returns titles, URLs, and snippets.",
                parameter_description="search query string",
                execute=search_internet,
            )
        )
        self._tools.register(
            ToolDefinition(
                name="todo_read",
                description="Read all lists and tasks, or a specific list. ALWAYS call this before any other todo tool.",
                parameter_description="'all' to see everything, or a list name to see one list",
                execute=todo_tool_mod.todo_read,
            )
        )
        self._tools.register(
            ToolDefinition(
                name="todo_add",
                description="Add tasks to a list. List is created automatically if it does not exist.",
                parameter_description="ListName | task1 | task2 (or just: task text for General)",
                execute=todo_tool_mod.todo_add,
            )
        )
        self._tools.register(
            ToolDefinition(
                name="todo_delete",
                description="Delete a task by its ID, or delete an entire list by its name.",
                parameter_description="task ID (e.g. a1b2c3d4) or list name (e.g. Shopping)",
                execute=todo_tool_mod.todo_delete,
            )
        )
        self._tools.register(
            ToolDefinition(
                name="todo_toggle",
                description="Toggle a task between pending and done. Use the task ID from todo_read.",
                parameter_description="task ID (e.g. a1b2c3d4)",
                execute=todo_tool_mod.todo_toggle,
            )
        )
        self._tools.register(
            ToolDefinition(
                name="save_memory",
                description="Remember a fact or user preference permanently.",
                parameter_description="fact text to store",
                execute=save_memory,
            )
        )
