# OpenClaw

Local autonomous AI agent with a Streamlit UI and an Ollama backend. Supports tool use (web search, to-do management, long-term memory), a ReAct reasoning loop, and full observability via a built-in debug page.

---

## Features

- **Chat Interface** -- conversational agent with a live to-do board
- **ReAct Reasoning Loop** -- Think, Act, Observe cycle (up to 5 iterations per turn)
- **Tool Use** -- web search (DuckDuckGo), to-do CRUD, persistent memory
- **Interactive To-Do Board** -- multiple lists (categories), checkboxes, inline add/delete
- **Long-Term Memory** -- vector-based recall via ChromaDB (cosine similarity)
- **Short-Term Memory** -- sliding context window with token counting
- **Proactive Suggestions** -- agent checks for pending tasks on startup
- **Under the Hood Page** -- inspect working memory, long-term storage, and the agent's internal monologue
- **Hot-Reload Settings** -- change persona, model, or temperature without restarting

---

## Model Support

OpenClaw works with **any model available through Ollama**. You can switch models at any time from the sidebar.

Currently recommended: **Qwen3:4b-instruct** -- good balance of speed and tool-following ability.

Other models that work:

| Model | Notes |
|-------|-------|
| `qwen3:4b` | Good balance of speed and tool-following |
| `llama3.2:3b` | Strong reasoning, but not great tool user |
| `ministral-3:3b` | Fast, decent tool use |
| `phi4-mini` | Very fast and good at math, weaker at using tool |
| `qwen3:1.7b` | Lightweight, for low-end hardware |

---

## Installation

### Prerequisites

1. **Python 3.10 or higher**
   - Check your version: `python --version`
   - Download from [python.org](https://www.python.org/downloads/) if needed

2. **Ollama** installed and running
   - Install from [ollama.com/download](https://ollama.com/download)
   - On Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Verify: `ollama --version`

3. **Pull a model** (do this before running the app):

   ```bash
   # Recommended model (good quality + reasonable speed):
   ollama pull qwen3:4b

   # Alternative lightweight model (lower hardware requirements):
   ollama pull qwen3:1.7b

   ```

4. **Make sure Ollama is running**:

   ```bash
   ollama serve
   ```

   Ollama listens on `http://localhost:11434` by default.

---

### Option A: Conda (recommended)

Conda handles Python version and native dependencies automatically.

```bash
# 1. Clone the repository
git clone <repo-url>
cd MiniClaw

# 2. Create the environment from the lockfile
conda env create -f environment.yml

# 3. Activate it
conda activate miniclaw

# 4. Install the project in editable mode (registers the package)
pip install -e .
```

To update the environment after pulling new changes:

```bash
conda env update -f environment.yml --prune
```

### Option B: pip + virtual environment

```bash
# 1. Clone the repository
git clone <repo-url>
cd MiniClaw

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows (Command Prompt)
# .venv\Scripts\Activate.ps1    # Windows (PowerShell)

# 4. Install all dependencies (including dev tools)
pip install -e ".[dev]"
```

### Option C: pip (global install)

Not recommended, but works for quick testing:

```bash
pip install -e .
```

### Verifying the installation

```bash
# Check that all core packages are available
python -c "
import streamlit, ollama, chromadb, pydantic, tiktoken, ddgs, jinja2
print('All dependencies OK')
"

# Check Ollama connectivity
python -c "
import ollama
models = ollama.Client().list()
print('Ollama OK, models:', [m.model for m in models.models])
"
```

---

## Running the App

```bash
# Make sure Ollama is running in another terminal:
ollama serve

# Start the Streamlit app:
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### First Launch

On the first launch:
```markdown
1. **IMPORTANT**: ChromaDB will initialise its embedding model (`all-MiniLM-L6-v2` via ONNX). This may take 30-60 seconds the first time.
2. The app will show a health check banner if Ollama is unreachable. Make sure `ollama serve` is running and the selected model is pulled.
3. Default settings are created in `data/settings.json`. You can change them via the sidebar.
```

---

## Project Structure

```
app.py                  Streamlit entry point
config/
  settings.py           Pydantic v2 settings (user, persona, Ollama)
llm/
  ollama_client.py      Ollama SDK wrapper (chat, stream, health check)
db/
  todo_store.py         JSON-backed to-do persistence with categories
memory/
  short_term.py         Sliding window buffer (tiktoken token counting)
  long_term.py          ChromaDB vector store wrapper
  manager.py            Coordinates STM + LTM, builds context
tools/
  registry.py           Tool definitions and dispatch
  search.py             DuckDuckGo web search (ddgs)
  todo.py               Agent-facing to-do tools
  memory_tool.py        Save-to-memory tool
agent/
  prompts.py            Jinja2 system prompt template
  reasoning.py          ReAct loop with JSON fallback parsing
  proactivity.py        Pending task suggestions
  core.py               AgentCore facade (wires everything)
ui/
  sidebar.py            Settings form
  chat_page.py          Page A: chat + interactive to-do board
  debug_page.py         Page B: memory inspector + monologue viewer
  components.py         Reusable Streamlit widgets
data/                   Runtime data (git-ignored)
  settings.json         User/agent/model settings
  todos.json            To-do items
  chroma/               ChromaDB vector store
```

---

## Pages

- **Agent** -- Chat with the bot. Interactive to-do board on the right.
- **Under the Hood** -- Working memory, long-term storage browser (with delete), query tester, internal monologue.

---

## Configuration

Use the sidebar to set:
- User name and info
- Agent persona (name, role, system instructions)
- Ollama model and temperature

Settings are persisted to `data/settings.json` and survive restarts.

---

## License

MIT
