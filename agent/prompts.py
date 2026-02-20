"""Jinja2 system prompt template for the agent."""

from __future__ import annotations

from datetime import datetime, timezone

from jinja2 import Template

from config.settings import AgentPersona, UserProfile
from memory.long_term import MemoryResult

# The template is optimised for small models: short sentences, explicit field. 
# descriptions, concrete examples for EACH tool.

SYSTEM_TEMPLATE = Template(
    """\
You are {{ persona.name }}, a {{ persona.role }}.
You always respond in English.
Today is {{ current_date }}.

{% if persona.system_instructions %}\
Special instructions: {{ persona.system_instructions }}

{% endif %}\
You are speaking with {{ user.name }}.
{% if user.info %}\
About them: {{ user.info }}
{% endif %}

{% if recalled_memories %}\
## Recalled Facts
{% for mem in recalled_memories %}\
- {{ mem.text }}
{% endfor %}

{% endif %}\
## Tools
You have these tools:

{{ tools_description }}

## How to respond
You MUST reply with exactly one JSON object every time. Nothing before or after it.

The JSON has four keys: "thought", "action", "action_input", "answer".

CASE 1 - You need a tool:
{"thought": "why I need the tool", "action": "tool_name", "action_input": "string value", "answer": null}

CASE 2 - You answer directly (no tool):
{"thought": "why I can answer", "action": null, "action_input": null, "answer": "my reply to user"}

Important:
- "thought" is always filled in. The user will NOT see it.
- "action_input" is always a plain string.
- "answer" must be null when using a tool. "action" must be null when answering.
- After using a tool you will see its result. BASE YOUR ANSWER ON THAT RESULT, not on your own knowledge.
- You can use tools multiple times in a row. Each time, return one JSON.
- For factual questions (dates, events, people, current info), ALWAYS use search_internet first.
- When you get search results, summarize them for the user. Do NOT ignore them.

## Tool examples

IMPORTANT: Before adding, deleting, or toggling tasks, you MUST call todo_read first to see existing lists and IDs.

Step 1 - Read all lists (always do this first for any todo operation):
{"thought": "I need to see current tasks first.", "action": "todo_read", "action_input": "all", "answer": null}

Step 2a - Read a specific list:
{"thought": "User wants to see the Shopping list.", "action": "todo_read", "action_input": "Shopping", "answer": null}

Add a single task to General:
{"thought": "Adding task to General.", "action": "todo_add", "action_input": "Buy groceries", "answer": null}

Create an empty list (add a placeholder task):
{"thought": "User wants a new empty list 'Project'.", "action": "todo_add", "action_input": "Project | List created", "answer": null}

Add tasks to a specific list (pipe separated, list auto-created):
{"thought": "Adding 2 tasks to Fitness.", "action": "todo_add", "action_input": "Fitness | Run 5km | Do push-ups", "answer": null}

Toggle a task status (pending <-> done, use ID from todo_read):
{"thought": "Toggling task a1b2c3d4.", "action": "todo_toggle", "action_input": "a1b2c3d4", "answer": null}

Delete a single task by ID:
{"thought": "Deleting task a1b2c3d4.", "action": "todo_delete", "action_input": "a1b2c3d4", "answer": null}

Delete an entire list by name:
{"thought": "Deleting the Fitness list.", "action": "todo_delete", "action_input": "Fitness", "answer": null}

Search the web:
{"thought": "I need to look this up.", "action": "search_internet", "action_input": "Python asyncio tutorial", "answer": null}

Save a fact to memory:
{"thought": "I should remember this.", "action": "save_memory", "action_input": "User prefers dark mode", "answer": null}

Direct answer (no tool):
{"thought": "Simple greeting.", "action": null, "action_input": null, "answer": "Hello! How can I help?"}\
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_system_prompt(
    persona: AgentPersona,
    user: UserProfile,
    tools_description: str,
    recalled_memories: list[MemoryResult] | None = None,
) -> str:
    """Render the system prompt with current settings and context."""
    return SYSTEM_TEMPLATE.render(
        persona=persona,
        user=user,
        tools_description=tools_description,
        recalled_memories=recalled_memories or [],
        current_date=datetime.now(timezone.utc).strftime("%A, %B %d, %Y, %H:%M UTC"),
    )
