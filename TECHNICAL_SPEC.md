# OpenClaw Protocol — Technical Specification

> **Version:** 1.0  
> **Date:** 2026-02-19  
> **Project:** MiniClaw (codename for the OpenClaw prototype)

---

## 1. Overview

OpenClaw is a **local-first, autonomous AI agent** delivered as a single-process Streamlit application. It combines:

| Concern | Technology |
|---------|-----------|
| UI / UX | **Streamlit ≥ 1.38** |
| LLM inference | **Ollama** (local, open-source models) |
| Vector memory | **ChromaDB** (persistent, on-disk) |
| Language | **Python 3.10+** |

The agent is **not** a passive chatbot — it reasons through a structured loop, uses tools, persists knowledge across sessions, and can proactively suggest actions.

---

## 2. Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit App                            │
│  ┌──────────────────────┐    ┌────────────────────────────────┐ │
│  │  Page A — Agent Chat │    │  Page B — Under the Hood       │ │
│  │  • Chat window       │    │  • Working memory viewer       │ │
│  │  • To-Do board       │    │  • Long-term storage browser   │ │
│  └──────────┬───────────┘    │  • Internal monologue log      │ │
│             │                └──────────────┬─────────────────┘ │
│             │                               │                   │
│  ┌──────────▼───────────────────────────────▼─────────────────┐ │
│  │                   Agent Core (Logic Layer)                 │ │
│  │  ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐ │ │
│  │  │ Reasoning │ │  Tool      │ │ Memory   │ │ Proactivity│ │ │
│  │  │ Loop      │ │  Registry  │ │ Manager  │ │ Engine     │ │ │
│  │  └─────┬─────┘ └─────┬──────┘ └────┬─────┘ └─────┬──────┘ │ │
│  └────────┼──────────────┼─────────────┼─────────────┼────────┘ │
│           │              │             │             │           │
│  ┌────────▼──────────────▼─────────────▼─────────────▼────────┐ │
│  │                   Data / External Layer                    │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │ │
│  │  │  Ollama  │  │ ChromaDB │  │ ToDo DB  │  │  Search    │ │ │
│  │  │  (LLM)   │  │ (Vector) │  │ (JSON)   │  │  (HTTP)    │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Responsibilities

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| **UI** | `app.py`, `pages/` | Rendering, user input, session state |
| **Logic** | `agent/` | Reasoning loop, tool dispatch, memory orchestration |
| **Data** | `memory/`, `tools/`, `db/` | Ollama client, ChromaDB, JSON persistence, HTTP search |

> [!IMPORTANT]
> All layers communicate through well-defined Python interfaces (protocols / ABCs). The UI must **never** call Ollama or ChromaDB directly.

---

## 3. Project Structure

```
MiniClaw/
├── app.py                     # Streamlit entry-point (multipage setup)
├── requirements.txt
├── pyproject.toml
├── README.md
├── TECHNICAL_SPEC.md          # (this document)
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Pydantic models for all settings
│
├── agent/
│   ├── __init__.py
│   ├── core.py                # AgentCore — public API consumed by UI
│   ├── reasoning.py           # ReAct-style loop implementation
│   ├── prompts.py             # System prompt templates (Jinja2)
│   └── proactivity.py         # Startup / idle check logic
│
├── memory/
│   ├── __init__.py
│   ├── short_term.py          # Conversation-window buffer
│   ├── long_term.py           # ChromaDB wrapper (embed + query)
│   └── manager.py             # MemoryManager — coordinates both stores
│
├── tools/
│   ├── __init__.py
│   ├── registry.py            # Tool registry & dispatch
│   ├── search.py              # Internet search tool
│   ├── todo.py                # To-Do list tool (add / read / done)
│   └── memory_tool.py         # save_memory tool (stores facts to LTM)
│
├── llm/
│   ├── __init__.py
│   └── ollama_client.py       # Thin wrapper around Ollama HTTP API
│
├── db/
│   ├── __init__.py
│   └── todo_store.py          # JSON-file persistence for to-do items
│
├── ui/
│   ├── __init__.py
│   ├── chat_page.py           # Page A — Agent Interface
│   ├── debug_page.py          # Page B — Under the Hood
│   ├── sidebar.py             # Settings sidebar (persona, user profile)
│   └── components.py          # Reusable Streamlit widgets
│
├── data/                      # Runtime data (gitignored)
│   ├── chroma/                # ChromaDB persistent directory
│   └── todos.json             # To-Do list on disk
│
└── tests/
    ├── test_reasoning.py
    ├── test_memory.py
    ├── test_tools.py
    └── test_ollama_client.py
```

---

## 4. Detailed Module Specifications

### 4.1 Configuration & Persona (`config/settings.py`)

#### Data Models (Pydantic v2)

```python
class UserProfile(BaseModel):
    name: str = "User"
    info: str = ""                     # free-form text

class AgentPersona(BaseModel):
    name: str = "Claw"
    role: str = "Personal Assistant"   # e.g. "Grumpy Coder"
    system_instructions: str = ""      # extra behavioural guidelines

class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "phi4-mini"
    temperature: float = 0.7
    context_window: int = 4096         # token budget for short-term memory

class AppSettings(BaseModel):
    user: UserProfile = UserProfile()
    persona: AgentPersona = AgentPersona()
    ollama: OllamaSettings = OllamaSettings()
```

#### Behaviour

- Stored in `st.session_state["settings"]`.
- Editable from the sidebar on every page.
- On change, the system prompt is **regenerated** immediately so subsequent messages reflect the new persona.
- Settings are persisted to `data/settings.json` so they survive app restarts.

---

### 4.2 LLM Client (`llm/ollama_client.py`)

> [!IMPORTANT]
> All interaction with Ollama **must** go through the official `ollama` Python library.
> Raw HTTP calls via `httpx` or `requests` are not allowed.

#### Interface

```python
import ollama

class OllamaClient:
    """Thin wrapper around the official ollama Python library."""

    def __init__(self, settings: OllamaSettings):
        self._client = ollama.Client(host=settings.base_url)
        self._model = settings.model
        self._temperature = settings.temperature

    def chat(
        self,
        messages: list[dict],
        format: str | None = "json",
        temperature: float | None = None,
    ) -> str:
        """Send messages and return the assistant response text."""
        response = self._client.chat(
            model=self._model,
            messages=messages,
            format=format,
            options={"temperature": temperature or self._temperature},
        )
        return response.message.content

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ):
        """Yield response chunks for streaming display."""
        stream = self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            options={"temperature": temperature or self._temperature},
        )
        for chunk in stream:
            yield chunk.message.content

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model exists."""
        try:
            self._client.show(self._model)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of all locally available models."""
        response = self._client.list()
        return [m.model for m in response.models]
```

> [!NOTE]
> Embeddings are handled by ChromaDB's built-in default (see section 4.3.2), **not** by Ollama.
> This keeps the Ollama dependency limited to chat/generation only.

#### Implementation Notes

| Aspect | Detail |
|--------|--------|
| Library | `ollama` Python SDK (official) |
| JSON mode | Pass `format="json"` to `client.chat()` |
| Retry policy | Retry up to 3x with exponential back-off (0.5s, 1s, 2s) on connection errors |
| Streaming | Use `stream=True` parameter for token-by-token display |
| Error handling | Catch `ollama.ResponseError` / `ConnectionError`; raise `OllamaUnavailableError`; UI shows a banner |

---

### 4.3 Memory Architecture (`memory/`)

#### 4.3.1 Short-Term Memory (`memory/short_term.py`)

```python
class ShortTermMemory:
    """Sliding-window buffer of chat messages."""

    def __init__(self, max_tokens: int = 4096): ...

    def add(self, role: str, content: str) -> None: ...
    def get_messages(self) -> list[dict]: ...
    def token_count(self) -> int: ...
    def trim(self) -> None:
        """Remove oldest non-system messages until within budget."""
    def to_raw(self) -> list[dict]:
        """Return the raw message list (for debug page display)."""
```

**Token counting:** Use `tiktoken` with the `cl100k_base` encoding as an approximation (exact tokeniser varies by model, but this is good enough for budget management).

**Trimming strategy:**
1. Never remove the system message.
2. Remove the oldest user/assistant pair first.
3. Always keep at least the last 2 exchanges.

#### 4.3.2 Long-Term Memory (`memory/long_term.py`)

```python
class LongTermMemory:
    """ChromaDB-backed vector store for persistent knowledge."""

    def __init__(self, persist_dir: str = "data/chroma"): ...

    def store(self, text: str, metadata: dict | None = None) -> str:
        """Embed and store a text chunk. Returns the document ID."""

    def query(self, query_text: str, n_results: int = 5) -> list[MemoryResult]:
        """Retrieve the top-n most relevant chunks."""

    def get_all(self) -> list[MemoryRecord]:
        """Return all stored records (for debug page display)."""

    def delete(self, doc_id: str) -> None: ...
```

```python
@dataclass
class MemoryResult:
    id: str
    text: str
    distance: float        # L2 or cosine distance
    metadata: dict

@dataclass
class MemoryRecord:
    id: str
    text: str
    metadata: dict
    embedding: list[float] | None   # optional, for visualisation
```

**Embedding model:** ChromaDB's **built-in default** (`all-MiniLM-L6-v2` via `onnxruntime`).

- Zero configuration — ChromaDB downloads the ONNX model automatically on first use.
- No extra dependencies beyond `chromadb` itself (no `sentence-transformers`, no `torch`).
- Fast on CPU, 384-dimensional vectors.

**ChromaDB configuration:**

| Setting | Value |
|---------|-------|
| Collection name | `"openclaw_memory"` |
| Embedding function | ChromaDB default (no custom function needed) |
| Distance function | `cosine` |
| Persist directory | `data/chroma/` |

#### 4.3.3 Memory Manager (`memory/manager.py`)

Coordinates both stores and exposes a unified API to the agent:

```python
class MemoryManager:
    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory): ...

    def add_message(self, role: str, content: str) -> None:
        """Add to short-term memory."""

    def save_to_long_term(self, text: str, metadata: dict | None = None) -> str:
        """Persist a piece of knowledge."""

    def recall(self, query: str, n: int = 5) -> list[MemoryResult]:
        """Query long-term memory."""

    def build_context(self, query: str | None = None) -> list[dict]:
        """
        Construct the full message list for Ollama:
        [system_prompt, ...recalled_context, ...short_term_messages]
        """
```

**How the agent decides what to save:** The reasoning loop includes a `SAVE_MEMORY` tool (see § 4.5). The LLM is prompted to use this tool when it encounters facts, user preferences, or task results that should persist beyond the current conversation.

---

### 4.4 Tools & Agency (`tools/`)

#### 4.4.1 Tool Registry (`tools/registry.py`)

```python
@dataclass
class ToolDefinition:
    name: str
    description: str                    # injected into the system prompt
    parameter_description: str          # human-readable description of the single string arg
    execute: Callable[[str], str]       # function that runs the tool (receives action_input as str)

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None: ...
    def get(self, name: str) -> ToolDefinition | None: ...
    def list_tools(self) -> list[ToolDefinition]: ...
    def to_prompt_description(self) -> str:
        """Format all tools for injection into the system prompt."""
```

#### 4.4.2 Built-in Tools

| Tool Name | Module | `action_input` | Description |
|-----------|--------|-----------------|-------------|
| `search_internet` | `tools/search.py` | search query string | Performs a web search and returns a text summary of top results |
| `todo_add` | `tools/todo.py` | task description | Adds a new item to the to-do list |
| `todo_read` | `tools/todo.py` | _(ignored)_ | Returns all current to-do items |
| `todo_done` | `tools/todo.py` | item ID | Marks a to-do item as completed |
| `save_memory` | `tools/memory_tool.py` | fact or preference text | Saves a fact/preference to long-term memory |

##### Internet Search (`tools/search.py`)

```python
def search_internet(query: str, max_results: int = 3) -> str:
    """
    Uses DuckDuckGo Search (duckduckgo-search library) to fetch results.
    Returns a formatted string of titles + snippets.
    """
```

- Library: `duckduckgo-search` (no API key required).
- Returns at most `max_results` items, each with title, URL, and snippet.
- On failure, returns a descriptive error string (the agent can then inform the user).

##### To-Do List (`tools/todo.py`)

```python
@dataclass
class TodoItem:
    id: str                 # UUID4
    text: str
    status: str             # "pending" | "done"
    created_at: str         # ISO 8601
    completed_at: str | None

def todo_add(text: str) -> str: ...
def todo_read() -> str: ...
def todo_done(item_id: str) -> str: ...
```

Persistence via `db/todo_store.py`:

```python
class TodoStore:
    """JSON-file backed store at data/todos.json."""

    def load(self) -> list[TodoItem]: ...
    def save(self, items: list[TodoItem]) -> None: ...
    def add(self, text: str) -> TodoItem: ...
    def mark_done(self, item_id: str) -> TodoItem | None: ...
    def get_all(self) -> list[TodoItem]: ...
    def get_pending(self) -> list[TodoItem]: ...
```

---

### 4.5 Reasoning Loop (`agent/reasoning.py`)

The agent uses a **ReAct** (Reasoning + Acting) loop:

```
┌──────────────────────────────────────────────────┐
│                   User Input                     │
└──────────────────────┬───────────────────────────┘
                       ▼
              ┌────────────────┐
              │ Build Context  │  (system prompt + recalled memory
              │                │   + short-term messages)
              └───────┬────────┘
                      ▼
        ┌─────────────────────────────┐
   ┌───►│  LLM Call (JSON mode)       │
   │    │  → returns {thought, action, │
   │    │    action_input, answer}     │
   │    └─────────────┬───────────────┘
   │                  ▼
   │         ┌─────────────────┐
   │    NO   │  action == null  │  YES
   │  ◄──────┤  (final answer?) ├──────►  Return answer to user
   │         └─────────────────┘
   │                  │ NO
   │                  ▼
   │         ┌─────────────────┐
   │         │  Execute Tool   │
   │         │  (from registry)│
   │         └────────┬────────┘
   │                  ▼
   │         ┌─────────────────┐
   │         │  Append          │
   │         │  Observation     │
   └─────────┤  to messages     │
              └─────────────────┘
```

#### JSON Output Schema (requested from LLM)

```json
{
  "thought": "My internal reasoning about the user's request...",
  "action": "tool_name or null",
  "action_input": "a single plain string argument for the tool",
  "answer": "Final response to the user (only when action is null)"
}
```

> [!IMPORTANT]
> `action_input` is a **flat string**, not a nested object. This dramatically improves
> JSON generation reliability on small models (phi4-mini, gemma:2b). Every built-in tool
> accepts a single string argument, so a nested dict is unnecessary.

#### Implementation

```python
class ReasoningLoop:
    MAX_ITERATIONS = 5          # safety cap to prevent infinite loops

    def __init__(
        self,
        client: OllamaClient,
        memory: MemoryManager,
        tools: ToolRegistry,
        persona: AgentPersona,
        user: UserProfile,
    ): ...

    def run(self, user_input: str) -> AgentResponse:
        """
        Execute the full ReAct loop.
        Returns the final answer plus the full thought trace.
        """

@dataclass
class AgentResponse:
    answer: str
    thought_trace: list[ThoughtStep]   # for "Internal Monologue" display

@dataclass
class ThoughtStep:
    iteration: int
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None
```

#### Handling Unreliable JSON from Small Models

Smaller models (e.g. `phi4-mini`, `gemma:2b`) may produce malformed JSON. Mitigation strategy:

1. **Ollama `format: "json"` flag** — most effective first defense.
2. **Regex fallback parser** — if `json.loads()` fails, attempt to extract JSON from the response using a `\{.*\}` regex (with `re.DOTALL`).
3. **Plain-text fallback** — if no JSON is found, treat the entire response as a direct `answer` with no tool call.
4. **Log** all parse failures to the internal monologue for observability.

---

### 4.6 Proactivity Engine (`agent/proactivity.py`)

```python
class ProactivityEngine:
    def __init__(self, todo_store: TodoStore, memory: MemoryManager): ...

    def check_on_startup(self) -> str | None:
        """
        Called once per session start.
        If there are pending to-dos, return a proactive suggestion message.
        Otherwise return None.
        """

    def check_after_task_update(self) -> str | None:
        """
        Called after any to-do mutation.
        Inspects the updated list and returns a follow-up suggestion if appropriate.
        """
```

#### Behaviour

| Trigger | Action |
|---------|--------|
| App launch (first load of session) | Read pending to-dos. If any exist, generate a message: *"I see you have N unfinished tasks. Want me to work on one?"* |
| After `todo_add` or `todo_done` | Check remaining pending tasks. If a task looks actionable (heuristic: contains keywords like "find", "search", "check"), suggest acting on it. |

The proactive message is **injected into the chat** as an assistant message, **not** auto-executed. The user must confirm before the agent acts.

---

### 4.7 Agent Core (`agent/core.py`)

Façade that the UI layer calls:

```python
class AgentCore:
    def __init__(self, settings: AppSettings): ...

    def handle_message(self, user_input: str) -> AgentResponse:
        """Process a user message through the full pipeline."""

    def get_proactive_message(self) -> str | None:
        """Check startup / task-update triggers."""

    def get_working_memory(self) -> list[dict]:
        """Return raw messages for debug display."""

    def get_long_term_records(self) -> list[MemoryRecord]:
        """Return all vector-store entries."""

    def query_long_term(self, query: str, n: int = 5) -> list[MemoryResult]:
        """Search long-term memory (for debug page query tool)."""

    def get_thought_trace(self) -> list[ThoughtStep]:
        """Return the most recent reasoning trace."""

    def get_todos(self) -> list[TodoItem]:
        """Return all to-do items."""

    def reload_settings(self, settings: AppSettings) -> None:
        """Hot-reload persona/model settings."""
```

---

## 5. UI Specification (`ui/`)

### 5.1 App Entry-Point (`app.py`)

```python
import streamlit as st

st.set_page_config(page_title="OpenClaw", page_icon="C", layout="wide")

# Initialize singleton AgentCore in session_state
# Render sidebar (settings)
# Multipage navigation: Page A / Page B
```

### 5.2 Sidebar — Settings (`ui/sidebar.py`)

| Field | Widget | Bound to |
|-------|--------|----------|
| User name | `st.text_input` | `settings.user.name` |
| User info | `st.text_area` | `settings.user.info` |
| Agent name | `st.text_input` | `settings.persona.name` |
| Agent role | `st.selectbox` + custom | `settings.persona.role` |
| System instructions | `st.text_area` | `settings.persona.system_instructions` |
| Ollama model | `st.selectbox` (populated from `ollama list`) | `settings.ollama.model` |
| Temperature | `st.slider(0.0, 2.0)` | `settings.ollama.temperature` |

A **"Save Settings"** button persists changes and calls `agent.reload_settings()`.

### 5.3 Page A — Agent Interface (`ui/chat_page.py`)

**Layout** (Streamlit columns):

```
┌────────────────────────────┬──────────────────┐
│                            │                  │
│        Chat Window         │   To-Do Board    │
│   (st.chat_message loop)   │  (st.container)  │
│                            │                  │
│                            │  [ ] Task 1      │
│                            │  [✓] Task 2      │
│                            │  [ ] Task 3      │
│                            │                  │
├────────────────────────────┤                  │
│  st.chat_input("Message…") │                  │
└────────────────────────────┴──────────────────┘
```

- **Chat messages** rendered with `st.chat_message()` for user / assistant roles.
- **Streaming** — assistant responses rendered token-by-token via `st.write_stream()`.
- **Proactive messages** — on session start, the proactivity engine injects an assistant message if applicable.
- **To-Do Board** — live-rendered from `agent.get_todos()`. Each item shows status (checkbox icon), text, and creation date. No direct user manipulation (the agent manages it through conversation).

### 5.4 Page B — Under the Hood (`ui/debug_page.py`)

Three collapsible sections:

#### Section 1: Working Memory

- Display the raw `list[dict]` from `agent.get_working_memory()`.
- Each message rendered as a card with:
  - **Role** badge (`system` / `user` / `assistant` / `tool`)
  - **Content** (syntax-highlighted JSON or plain text)
  - **Token count** estimate

#### Section 2: Long-Term Storage

- Table of all vector-store records (`agent.get_long_term_records()`).
- Columns: ID, text (truncated), metadata, timestamp.
- **Query tester**: `st.text_input` + button → runs `agent.query_long_term(q)` and shows results with **distance scores**.
- *(Optional)* 2-D embedding visualisation using t-SNE / UMAP → rendered with `st.plotly_chart()`.

#### Section 3: Internal Monologue

- Display the `thought_trace` from the most recent `AgentResponse`.
- Each step rendered as an expandable container:
  - **[THOUGHT]**: the LLM's reasoning
  - **[ACTION]**: tool name + input
  - **[OBSERVATION]**: tool output

---

## 6. System Prompt Template (`agent/prompts.py`)

The system prompt is assembled dynamically from settings + tool descriptions.
Optimised for `phi4-mini`: short sentences, explicit field descriptions,
concrete examples to anchor the JSON schema.

> [!IMPORTANT]
> All interface text, comments, and prompt templates must be in **English only**.
> Emoji are **not allowed** anywhere in the codebase or prompts.

```jinja2
You are {{ persona.name }}, a {{ persona.role }}.
You always respond in English.

{% if persona.system_instructions %}
Special instructions: {{ persona.system_instructions }}
{% endif %}

You are speaking with {{ user.name }}.
{% if user.info %}
About them: {{ user.info }}
{% endif %}

{% if recalled_memories %}
## Recalled Facts
{% for mem in recalled_memories %}
- {{ mem.text }}
{% endfor %}
{% endif %}

## Tools
You have access to these tools:
{{ tools_description }}

## Response Format
Always reply with a single JSON object. No text before or after the JSON.
Use exactly these four keys:

{"thought": "...", "action": "...", "action_input": "...", "answer": "..."}

Rules:
1. "thought" -- your private reasoning. The user will not see this. Always fill it in.
2. If you need a tool, set "action" to the tool name and "action_input" to the string argument. Set "answer" to null.
3. If you can answer directly, set "action" to null, "action_input" to null, and put your reply in "answer".
4. Use the save_memory tool to store important facts and preferences.
5. Stay in character as {{ persona.name }}.

Example -- using a tool:
{"thought": "The user wants the weather. I need to search.", "action": "search_internet", "action_input": "weather Kyiv today", "answer": null}

Example -- direct answer:
{"thought": "The user said hello. No tool needed.", "action": null, "action_input": null, "answer": "Hello! How can I help you today?"}
```

---

## 7. Data Persistence

| Data | Storage | Location | Format |
|------|---------|----------|--------|
| Settings | JSON file | `data/settings.json` | Pydantic `.model_dump_json()` |
| To-Do items | JSON file | `data/todos.json` | List of `TodoItem` dicts |
| Long-term memory | ChromaDB | `data/chroma/` | Embeddings + metadata |
| Chat history (short-term) | In-memory | `st.session_state` | List of message dicts |

The `data/` directory is `.gitignore`-d. All paths are relative to the project root, resolved at runtime with `pathlib.Path`.

---

## 8. Dependencies

### `requirements.txt`

```
streamlit>=1.38,<2.0
ollama>=0.4
chromadb>=1.5.0,<2.0
pydantic>=2.0,<3.0
tiktoken>=0.7
duckduckgo-search>=6.0
jinja2>=3.1
```

### External Services

| Service | Requirement |
|---------|-------------|
| **Ollama** | Must be installed and running on `localhost:11434`. Default model: `ollama pull phi4-mini`. |

> [!NOTE]
> No separate embedding model pull is needed -- ChromaDB downloads its built-in
> ONNX embedding model (`all-MiniLM-L6-v2`) automatically on first use.

---

## 9. Error Handling & Resilience

| Scenario | Handling |
|----------|----------|
| Ollama not running | `health_check()` on startup → display `st.error` banner with instructions |
| Model not found | Catch 404 from Ollama → suggest `ollama pull <model>` |
| Malformed LLM JSON | Regex fallback → plain-text fallback → log to monologue |
| ChromaDB corruption | Catch exceptions → offer "reset memory" button on debug page |
| Search API failure | Tool returns error string → agent informs user gracefully |
| Infinite reasoning loop | Hard cap at `MAX_ITERATIONS = 5` → return partial answer |

---

## 10. Security & Privacy Considerations

- **All data stays local.** No cloud APIs, no telemetry.
- The only external network call is the optional DuckDuckGo search.
- No authentication layer (single-user local app).
- The `data/` directory should not be committed to version control.

---

## 11. Testing Strategy

### Unit Tests

| Module | Key Test Cases |
|--------|---------------|
| `llm/ollama_client.py` | Mock HTTP responses; verify retry logic; verify JSON parse |
| `memory/short_term.py` | Token budget enforcement; trim logic; message ordering |
| `memory/long_term.py` | Store → query round-trip; metadata filtering; empty DB |
| `tools/todo.py` | Add → read → done flow; persistence across reloads |
| `tools/search.py` | Mock DuckDuckGo; verify output formatting |
| `agent/reasoning.py` | Mock LLM; verify loop terminates; verify tool dispatch |

### Integration Tests

| Test | Method |
|------|--------|
| End-to-end chat flow | Start Streamlit in headless mode; send messages via session state; assert response structure |
| Memory round-trip | Store fact → query → verify recall in response context |
| Proactivity | Add pending to-do → restart session → verify proactive message appears |

### Manual Verification

| Item | How |
|------|-----|
| Persona fidelity | Change persona to "Grumpy Coder" → verify tone changes |
| Debug page accuracy | Perform chat → check Working Memory matches sent messages |
| To-Do board sync | Ask agent to add a task → verify board updates |
| Streaming | Verify tokens appear incrementally during generation |

---

## 12. Performance Considerations

| Concern | Mitigation |
|---------|-----------|
| Slow Ollama inference | Use streaming to reduce perceived latency; show spinner / progress |
| Large ChromaDB | Limit to top-5 results; use metadata filters if collection grows |
| Token budget overflow | Aggressive short-term trimming; summarise old context if needed |
| Streamlit rerun overhead | Cache `AgentCore` in `st.session_state`; avoid reconstructing on every rerun |

---

## 13. Future Enhancements (Out of Scope for v1)

- **Multi-user support** with authentication
- **File upload** tool (read PDFs, process CSVs)
- **Calendar / scheduling** integration
- **Voice input** via Whisper
- **Autonomous background execution** (agent runs tasks without waiting for confirmation)
- **Plugin system** for community-built tools
- **Conversation summarisation** for long-term context compression

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **ReAct** | Reasoning + Acting — a prompting framework where the LLM alternates between thinking and executing tools |
| **Short-Term Memory (STM)** | The sliding window of recent chat messages sent as context to the LLM |
| **Long-Term Memory (LTM)** | Facts and preferences persisted in a vector database across sessions |
| **Tool** | A callable function the agent can invoke to interact with the outside world |
| **Proactivity** | The agent's ability to initiate conversation based on its current state |
| **Thought Trace** | The internal log of the agent's reasoning steps during a single interaction |
