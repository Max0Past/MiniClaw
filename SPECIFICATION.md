# OpenClaw -- Technical Specification

Detailed architecture, design decisions, and implementation notes for the OpenClaw agent. I built this prototype to demonstrate how an autonomous AI agent can operate locally with real tools, memory, and agency.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Why Not LangChain?](#why-not-langchain)
3. [Library Choices](#library-choices)
4. [Module-by-Module Breakdown](#module-by-module-breakdown)
5. [The ReAct Reasoning Loop](#the-react-reasoning-loop)
6. [Memory Architecture](#memory-architecture)
7. [Tool System](#tool-system)
8. [Prompt Engineering](#prompt-engineering)
9. [UI Patterns and Streamlit Gotchas](#ui-patterns-and-streamlit-gotchas)
10. [Non-Obvious Implementation Details](#non-obvious-implementation-details)

---

## Architecture Overview

```text
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Chat Page   │  │  Debug Page  │  │     Sidebar        │ │
│  │  + Todo Board│  │  + Memory    │  │  (Settings Form)   │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘ │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                       AgentCore                              │
│  (Facade: wires all components, public API for UI)           │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Reasoning   │  │   Memory     │  │   Tool Registry     │ │
│  │ Loop (ReAct)│◄─┤   Manager    │  │   + Tool Functions  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
│         │                │                      │            │
│         ▼                ▼                      ▼            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Ollama      │  │ STM   │ LTM │  │ search │ todo │ mem  │ │
│  │ Client      │  │(token)│(vec)│  │  (ddgs) (json) (chroma)│
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Data flow for a single user message:**

1. The user types a message in `chat_page.py`.
2. `AgentCore.handle_message()` is invoked to start processing.
3. The system prompt is dynamically rebuilt with the current persona, user info, active tools, and the current UTC date.
4. `MemoryManager.build_context()` assembles the complete context window for the LLM:
   - System prompt (always the very first message)
   - Recalled facts from long-term memory (injected as an artificial `system` message right after the main system prompt)
   - Short-term memory (the recent conversation history, carefully pruned to fit within a predefined token budget)
5. `ReasoningLoop.run()` takes over and enters the ReAct cycle:
   - It sends the message list to Ollama, enforcing `format="json"`.
   - It parses the JSON response (resorting to regex or fallback logic if the model hallucinates formatting).
   - If the model's response specifies a tool `action`: I execute the mapped Python function, append the result as a new `user` message (labeled as an observation), and loop back to query Ollama again.
   - If the model provides a direct `answer`: I break the loop and return the string.
6. The final answer is displayed in the chat interface.
7. `st.rerun()` is triggered to force a full UI refresh, updating the to-do board and memory views to reflect any state changes.

---

## Why Not LangChain?

LangChain is undeniably the most popular framework for building LLM-powered applications. I deliberately chose **not** to use it for this project. Here is my reasoning:

### 1. Excessive Abstraction for Small Models

LangChain's agent abstractions (`AgentExecutor`, `Tool`, `Chain`) are conceptually designed for massive, highly capable models like GPT-4 or Claude 3.5. They assume the model can reliably follow multi-step formatting instructions, parse complex schemas, and inherently understand tool signatures. Small local models running in Ollama (like phi4-mini or Qwen3:4b) absolutely struggle with these assumptions. By writing my own ReAct loop from scratch, I gained total control over:
- Exactly how the prompt is formatted (which is critical for coercing small models into compliance).
- How JSON parsing failures are caught and handled.
- How tool observations are phrased and fed back to the model.

### 2. Dependency Weight

LangChain pulls in a massive dependency tree (dozens of packages including `langchain-core`, `langchain-community`, `langsmith`, `pydantic v1 compat`, etc.). For a local prototype, this is unacceptable overhead. My entire dependency list is just 8 packages. LangChain alone would more than double that footprint and drastically increase installation times.

### 3. Debugging Opacity

When a LangChain agent misbehaves or loops infinitely, debugging requires diving through multiple opaque layers of abstraction, chains, runnables, and callback handlers. With my custom implementation:
- The entire reasoning loop fits into ~130 lines of readable, sequential Python.
- Every single step is explicitly logged and exposed on the "Under the Hood" debug page.
- There are no hidden prompt injections or magic string formatting.

### 4. Ollama SDK is Sufficient

LangChain's `ChatOllama` wrapper adds negligible value over the official `ollama` Python library. The raw SDK already provides everything I need: synchronous `chat()`, `stream=True`, model listing, and health checks.

### 5. Educational Value

A primary goal of developing this prototype was to understand how an autonomous AI agent operates under the hood. Using LangChain would obscure the core mechanisms (prompt construction, loop orchestration, memory management, tool dispatch) behind high-level APIs. By building from first principles, every component in this architecture is radically transparent and fully modifiable.

**In summary:** While LangChain is appropriate for enterprise systems relying on commercial API models, it introduces pure complexity without proportional benefit when building a local prototype with small open-source models.

---

## Library Choices

| Library | Version | Why I Chose This Library |
|---------|---------|--------------------------|
| **streamlit** | >= 1.38 | The fastest way to build interactive Python frontends. I utilized its built-in chat UI components (`st.chat_message`, `st.chat_input`), deeply integrated session state, and reactive reruns. I didn't have to write a single line of React, HTML, or CSS. |
| **ollama** | >= 0.4 | The official Python SDK for interacting with the local Ollama daemon. It provides strongly-typed responses and native JSON mode support. Using it instead of raw HTTP requests via `httpx` ensures the app remains compatible with upstream Ollama API changes. |
| **chromadb** | >= 1.5.0 | A powerful embedded vector database. Crucially, it ships with a built-in default embedding function (`all-MiniLM-L6-v2` running via ONNX runtime). This gave me production-grade vector search without needing to install PyTorch, pull separate embedding models, or rely on external APIs. It writes directly to disk. |
| **pydantic** | >= 2.0 | Used for settings validation and structured serialization. Pydantic's `model_validate_json()` and `model_dump_json()` allowed me to create a robust, type-safe, and round-trippable JSON configuration system with almost zero boilerplate. |
| **tiktoken** | >= 0.7 | OpenAI's fast BPE tokenizer. I use it purely for counting tokens to manage the sliding context window for short-term memory. While it's tuned for OpenAI models, it provides a perfectly acceptable approximation for local Ollama models. |
| **ddgs** | >= 7.0 | DuckDuckGo search client. It requires no API keys and no rate-limit registration, making it ideal for a drop-in local agent. It returns clean, structured results (title, URL, text snippet). |
| **jinja2** | >= 3.1 | Used to assemble the complex system prompt. Jinja2 templates allow conditional logic (like conditionally inserting recalled memories or roleplay instructions) without resulting in unreadable string concatenation. Features like `trim_blocks` ensure the final prompt has no messy whitespace. |

### Libraries I Avoided

| Library | Why I Skipped It |
|---------|------------------|
| **LangChain/LlamaIndex** | Overly abstracted and bloated for this use case. See [Why Not LangChain?](#why-not-langchain). |
| **sentence-transformers** | Importing this pulls in the entire PyTorch ecosystem (~2GB on disk). ChromaDB's built-in ONNX embeddings achieved the exact same result at a fraction of the cost. |
| **FAISS** | Facebook's AI Similarity Search requires complex C++ compilation or specialized `faiss-cpu` wheels. ChromaDB provides the same cosine similarity but offers a much better developer experience. |
| **SQLite/SQLAlchemy** | Absolute overkill. The app's state fits perfectly into two simple JSON files (settings + todos). Adding an ORM would introduce unnecessary schema management. |

---

## Module-by-Module Breakdown

### `config/settings.py`

I implemented settings using Pydantic v2 models arranged in a nested hierarchy: `AppSettings` → (`UserProfile`, `AgentPersona`, `OllamaSettings`). These are eagerly loaded from and saved to `data/settings.json`. I wrote `load_settings()` to return fallback defaults if the JSON file is missing, meaning the application boots instantly on a fresh clone without any manual configuration steps.

### `llm/ollama_client.py`

A focused wrapper around the native `ollama` client. Key design choices include:
- **Exponential Back-off Retry Logic**: Small local models can sometimes take a few seconds to load into VRAM. I implemented retries (0.5s, 1s, 2s) to gracefully handle these cold-start connection timeouts.
- **Strict JSON Mode Enforcement**: I explicitly pass `format="json"` to `client.chat()`. This forces the Ollama backend to algorithmically constrain the model's output to valid JSON tokens.
- **Health Checks**: I use `client.show(model)` prior to chat execution to verify that both the Ollama daemon is running and the specific requested model is pulled locally.

### `db/todo_store.py`

I built a lightweight JSON datastore. The core entity is `TodoItem` (a dataclass with properties like `id`, `text`, `category`, and timestamp fields).
- **ID Generation**: I opted for 8-character hex strings (`uuid4().hex[:8]`). Standard UUIDs are 36 characters long, which consumes unnecessary tokens and confuses small LLMs. 8 chars is plenty for collision resistance in a personal todo app.
- **Format Evolution**: The JSON schema evolved from a flat array to a dictionary containing `{"items": [...], "categories": [...]}`. I wrote backwards-compatibility logic in `_load_data` to auto-migrate old flat arrays.
- **State Toggling**: Instead of complex distinct methods, I built a single `toggle_status()` method that flips between `pending` and `done`.

### `memory/short_term.py`

The agent's conversational sliding window. I implemented token budgeting using `tiktoken`.
- **Trim Policy**: When the conversation exceeds the `max_tokens` threshold, I drop the oldest non-system message. However, I enforced a strict rule (`MIN_KEEP_PAIRS = 2`) that guarantees the system prompt and the latest 2 exchanges are *never* dropped, ensuring the model never abruptly loses the immediate context of the conversation. I chose message-level dropping rather than string truncation to prevent feeding the model broken sentences.

### `memory/long_term.py`

The ChromaDB implementation.
- **Zero-Setup Embeddings**: I relied on ChromaDB's default embedding function (`all-MiniLM-L6-v2` via ONNX runtime). This sacrifices a tiny amount of semantic precision for a massive gain in simplicity—no GPU or model download scripts required.
- **Distance Metric**: I explicitly configured the collection to use cosine similarity via `metadata={"hnsw:space": "cosine"}`.

### `memory/manager.py`

The bridge between Short-Term Memory and Long-Term Memory.
Before every LLM request, my `build_context()` method:
1. Takes the user's latest message and embeds it to query LTM.
2. Filters results based on distance thresholds.
3. Injects the relevant facts dynamically as a dedicated "Recalled Facts" block directly beneath the system prompt, but *before* the conversation history. This effectively acts as a lightweight, zero-infrastructure RAG pipeline.

### `tools/registry.py`

My custom tool dispatch system. I defined a `ToolDefinition` dataclass carrying the `name`, `description`, `parameter_description`, and the underlying `execute` callback. The `ToolRegistry` aggregates these and exposes a `to_prompt_description()` method that automatically generates the instructions injected into the Jinja2 prompt.
- **Deliberate Simplicity**: I completely omitted Pydantic schemas and complex argument parsing. Tools accept exactly one string argument (`action_input`). This dramatically lowers the cognitive load on small LLMs, which reliably fail when asked to generate nested JSON objects.

### `tools/search.py`

A wrapper for `ddgs.DDGS().text()`. I designed it to be defensively robust: all exceptions (network failures, rate limits) are caught and returned as plain text error strings. Tools should never crash the ReAct loop; they should return their failure state so the agent can read the error and apologize or try an alternative approach.

### `tools/todo.py`

I engineered four unified tool functions: `todo_read`, `todo_add`, `todo_delete`, and `todo_toggle`.
- **Pipe Delimiters**: For bulk task additions (`todo_add`), I taught the model to use the pipe character (`|`) to separate the list name from task text (e.g., `Shopping | Milk | Eggs`). Pipes are far safer than commas (which exist inside tasks) or colons.
- **Dual-Mode Deletion**: `todo_delete` accepts either an 8-char hex ID to delete a specific task, or a category name to wipe an entire list. It attempts to match an ID first. Consolidating this to one tool reduces the number of tools the model needs to understand.

### `agent/prompts.py`

The core intelligence of the agent lives in this Jinja2 template. It tightly controls behavior using:
- **Explicit JSON Schema**: I hardcode the exact 4-key JSON structure.
- **Few-Shot Examples**: Every single tool has a concrete JSON example in the prompt. I iteratively discovered that without these exact anchors, models like phi4-mini completely fail to produce valid tool invocation syntax.
- **Rule Enforcement**: Negative constraints ("Do NOT ignore search results") are heavily emphasized.
- **Pre-read Mandate**: I explicitly instructed the model that it *must* execute `todo_read` before attempting to modify tasks. This prevents it from hallucinating task IDs.

### `agent/reasoning.py`

The heart of the application: a custom ~130-line ReAct orchestrator.
1. It loops up to a maximum of 5 iterations per message to prevent infinite runaway loops.
2. It calls Ollama in JSON mode.
3. It routes the output through a robust 3-tier parsing gauntlet (json map -> regex extraction -> plaintext fallback).
4. For tool executions, it wraps the tool's output in an explicitly formatted observation frame telling the LLM to base its next response exactly on that text.

### `agent/proactivity.py`

I built a lightweight proactivity engine that runs exactly once on application startup. It scans pending tasks for actionable verbs ('find', 'search', 'buy', 'check') and, if a match is found, auto-generated an artificial system message prompting the bot to offer assistance (e.g., "I see 'check flight prices' on our list. Should I search for that now?").

### `agent/core.py`

The facade pattern centralizing all submodules. `AgentCore` handles dependency injection in `__init__` and exposes clean public APIs (`handle_message`, `get_todos`) for the Streamlit UI to consume. The UI never touches the database or the reasoning loop directly. It also provides `reload_settings()` allowing hot-swapping the persona and model without losing the conversational context.

---

## The ReAct Reasoning Loop

ReAct (Reason + Act) is a prompting methodology that forces the model to externalize its intermediate reasoning steps before invoking an action. I implemented it natively:

```text
User: "Add 'buy milk' to my shopping list"
  │
  ▼
[Iteration 1] Model JSON output:
{
  "thought": "The user wants to add a task. I must read existing lists first to avoid duplicates and see format.",
  "action": "todo_read",
  "action_input": "all",
  "answer": null
}
  │
  ▼ (I execute todo_read() and append the result as an observation)
  │
[Iteration 2] Model JSON output:
{
  "thought": "I've seen the lists. Now I will add the task to the Shopping category.",
  "action": "todo_add",
  "action_input": "Shopping | Buy milk",
  "answer": null
}
  │
  ▼ (I execute todo_add() and append the result as an observation)
  │
[Iteration 3] Model JSON output:
{
  "thought": "The task was successfully added. I can now inform the user.",
  "action": null,
  "action_input": null,
  "answer": "All set! I've added 'Buy milk' to your Shopping list."
}
```

This strict separation of `thought`, `action`, and `answer` ensures maximum determinism.

---

## Memory Architecture

I designed the memory architecture to solve two distinct problems independently.

### Short-Term Memory (STM)
- **Purpose**: Maintain conversational coherence ("What did we just say?").
- **Implementation**: A token-bounded FIFO queue of `{role, content}` dicts.
- **Why Drop Whole Messages?**: I chose to pop the oldest message entirely rather than truncating strings halfway. Truncating language models usually causes them to lose sentence structure and output gibberish.

### Long-Term Memory (LTM)
- **Purpose**: Persistent knowledge retrieval ("What are my user's preferences from last week?").
- **Implementation**: ChromaDB vector store.
- **Storage Strategy**: The agent only stores information when it actively decides it's important, using the explicitly provided `save_memory` tool. I did not want every mundane "hello" saved to the vector DB.
- **Injection Strategy**: The `build_context` function retrieves matches and injects them as an artificial static block so the agent perceives them as known facts.

Together, these two systems provide the agent with deep context awareness without ever blowing the model's context window.

---

## Tool System

### Design Philosophy

I designed the tools to be as simplistic as functionally possible. Every tool is a standard Python function taking exactly one `str` and returning exactly one `str`. 

1. Small models are notoriously bad at emitting complex JSON objects with multiple typed keys.
2. By forcing a single `action_input` string, the model's chances of syntax failure drop significantly.
3. Complex string parsing (like parsing `Category | Task1 | Task2`) is handled manually in my Python backend, shifting the logic burden away from the fragile LLM.

### Dispatch & Safety

The `ReasoningLoop` checks the model's requested `action` against the `ToolRegistry`. If the tool doesn't exist, I feed an error string right back to the model ("Tool 'X' not found"). Crucially, my tooling architecture treats Python exception throws as normal data: they are caught, stringified, and fed back as observations so the agent can autonomously attempt to recover.

---

## Prompt Engineering

Because I optimized for small (4b-8b parameter) models, I had to employ aggressive prompt engineering techniques:

1. **Explicit JSON Schema**: I hardcode the exact JSON structure it must output. I do not rely on implicit structured output guarantees.
2. **Per-Tool Few-Shot Examples**: I provide a concrete JSON example for *every single tool* available. Without these examples, smaller models inevitably revert to natural language. 
3. **Negative Directives**: "Do NOT ignore search results". Small models react best to strong, capitalized prohibiting constraints.
4. **Time Awareness**: I calculate and inject the exact UTC `datetime` into the prompt so the agent never hallucinates the current day or year.
5. **Observation Enveloping**: When a tool returns data, I don't just dump the raw text back. I wrap it in an aggressive meta-instruction: `Now respond with a JSON object. If the result answers the question, set action to null...` This forces the model back into compliance.

---

## UI Patterns and Streamlit Gotchas

### Input Blocking During Processing

Because Streamlit natively reruns the whole script top-to-bottom on interaction, allowing the user to type messages while the agent is executing loop iterations causes state race conditions. My solution:
1. When a user submits, I set `st.session_state.processing = True` and trigger a rerun.
2. On that rerun, because `processing` is true, the `st.chat_input` widget simply isn't rendered at all (completely hiding it).
3. Once the agent loop resolves and returns text, I set `processing = False` and rerun again to restore the input bar.

### Checkbox State Syncing

When an agent alters a task via a tool (e.g., marks it done), Streamlit's internal widget cache for the corresponding checkbox becomes stale, leading to visual desyncs. 
- **The Fix**: Before calling `st.checkbox()`, I preemptively mutate `st.session_state[key]` with the absolute source-of-truth status from the `TodoStore`. I also omit the `value` parameter to bypass Streamlit's API warnings about state conflicts.

### Delete Buttons and Event Loss

I used `on_click` callbacks for delete operations instead of checking `if st.button():`. I did this because `st.button()` evaluates to `True` only during the exact render loop where it was clicked. If another state change forces a rerun before the delete logic processes, the click event is swallowed. `on_click` callbacks fire reliably before the render pass begins.

---

## Non-Obvious Implementation Details

### 1. JSON Format Mode in Ollama

I deliberately pass `format="json"` in my HTTP payload via the `ollama` SDK. This is a critical token-sampling constraint applied by the Ollama engine, preventing the model from generating non-JSON syntax. Prompting alone is not enough for small models.

### 2. Three-Tier JSON Parsing

Even with server-side JSON constraints, models sometimes screw up. My parser handles three distinct failure states:
- Perfect JSON (parsed natively).
- JSON wrapped in Markdown fences (`` ```json {...} ``` ``) which I strip out.
- Severely malformed text, which triggers a regex `\{.*\}` extraction fallback. If even that fails, the raw text is aggressively treated as the `answer`.

### 3. Embedding Model Bootstrap

First-time users will experience a 30-60 second pause. This is because ChromaDB silently downloads the `all-MiniLM-L6-v2` ONNX weights into local cache on first execution. This is an unavoidable one-time initialization cost.

### 4. Token Counting Approximation

I use `tiktoken` equipped with the `cl100k_base` encoding dictionary. Technically, local models (Llama 3, Qwen) use their own distinct tokenizers. I made the deliberate engineering choice to use `cl100k_base` across all models because retrieving and executing exact local tokenizers implies enormous complexity. `cl100k_base` slightly overestimates token counts for most text, which provides a safe, conservative buffer for my max-token sliding window algorithm.

### 5. Tool Dependency Injection

Because tool functions (`todo_add`, `save_memory`) are pure module-level functions, they lack access to the instantiated `TodoStore` and `MemoryManager` objects. To preserve clean `str -> str` signatures for the LLM without global pollution, I wrote initializer hook functions (`set_store()`, `set_manager()`) that `AgentCore` calls on startup to safely inject the dependencies into the tool modules.

### 6. Single-Shot Proactivity

My proactivity engine operates purely single-shot via a `_startup_checked` boolean flag. This deliberate limitation prevents the agent from nagging the user with identical task suggestions every time the React UI re-renders a frame.

### 7. Instantaneous System Prompt Rebuilds

The system prompt string is constructed fresh via Jinja2 upon *every single message handling pass*. It is never cached. This ensures the injected timestamp is accurate to the second, state changes to the Agent's persona apply instantly mid-conversation, and the RAG memory retrievals perfectly align with the absolute latest conversational context. The processing overhead for a Jinja template render is microsecond-scale, making caching unnecessary.
