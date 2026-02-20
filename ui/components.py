"""Reusable Streamlit widgets for the OpenClaw UI."""

from __future__ import annotations

import streamlit as st

from agent.core import AgentCore
from agent.reasoning import ThoughtStep
from db.todo_store import TodoItem
from memory.long_term import MemoryRecord, MemoryResult


def render_interactive_todo_board(agent: AgentCore) -> None:
    """Render the to-do list as a set of interactive expanders."""
    items = agent.get_todos()
    
    # helper callbacks
    def _on_checkbox_change(item_id: str):
        cb_key = f"task_{item_id}"
        checked = st.session_state.get(cb_key, False)
        if checked:
            agent._todo_store.mark_done(item_id)
        else:
            agent._todo_store.mark_pending(item_id)

    def _on_delete_task(item_id: str):
        agent._todo_store.delete_item(item_id)

    def _on_delete_category(category: str):
        agent._todo_store.delete_category(category)

    def _on_add_task(category: str, key_prefix: str):
        key = f"{key_prefix}_new_task"
        text = st.session_state.get(key, "").strip()
        if text:
            agent._todo_store.add(text, category=category)
            st.session_state[key] = ""  # clear input

    # Group by category
    grouped: dict[str, list[TodoItem]] = {}
    all_categories = sorted(agent._todo_store.get_categories())
    
    for item in items:
        grouped.setdefault(item.category, []).append(item)

    # Render each category
    for category in all_categories:
        cat_items = grouped.get(category, [])
        pending_count = sum(1 for i in cat_items if i.status == "pending")
        label = f"{category} ({pending_count})"
        
        with st.expander(label, expanded=True):
            # Header actions (Delete list) - only if it has items or is custom
            # We allow deleting any list to clear it.
            if st.button("Delete List", key=f"del_cat_{category}"):
                _on_delete_category(category)
                st.rerun()

            # Render existing items
            for item in cat_items:
                is_done = item.status == "done"
                cb_key = f"task_{item.id}"
                
                # Always sync session state BEFORE creating the widget.
                st.session_state[cb_key] = is_done
                
                # Truncate long task text for display
                TRUNCATE_LEN = 60
                display_text = item.text
                is_long = len(display_text) > TRUNCATE_LEN
                
                if is_long:
                    # Collapsible: show truncated label, full text inside expander
                    truncated = display_text[:TRUNCATE_LEN] + "..."
                    with st.expander(truncated, expanded=False):
                        col_check, col_del = st.columns([0.9, 0.1])
                        with col_check:
                            st.checkbox(
                                "Done" if is_done else "Mark done",
                                key=cb_key,
                                on_change=_on_checkbox_change,
                                args=(item.id,),
                            )
                        with col_del:
                            st.button("x", key=f"del_task_{item.id}",
                                      help="Delete task",
                                      on_click=_on_delete_task,
                                      args=(item.id,),
                                      use_container_width=True)
                        st.markdown(display_text)
                else:
                    # Short task: inline checkbox + delete
                    col_check, col_del = st.columns([0.9, 0.1])
                    with col_check:
                        st.checkbox(
                            display_text,
                            key=cb_key,
                            on_change=_on_checkbox_change,
                            args=(item.id,),
                        )
                    with col_del:
                        st.button("x", key=f"del_task_{item.id}",
                                  help="Delete task",
                                  on_click=_on_delete_task,
                                  args=(item.id,),
                                  use_container_width=True)

            # Add new item to this category
            st.text_area(
                "Add task...",
                key=f"add_{category}_new_task",
                on_change=_on_add_task,
                args=(category, f"add_{category}"),
                placeholder=f"New {category} task",
                label_visibility="collapsed",
                height=68,
                help="Press Cmd/Ctrl+Enter to add task",
            )

    def _on_create_list():
        new_cat = st.session_state.get("new_list_name", "").strip()
        if new_cat:
            agent._todo_store.ensure_category(new_cat)
            st.session_state["new_list_name"] = ""

    # Allow creating a completely new list (category)
    with st.expander("➕ New List", expanded=False):
        st.text_input("List Name", key="new_list_name", max_chars=32)
        st.button("Create List", on_click=_on_create_list)


def render_thought_trace(trace: list[ThoughtStep]) -> None:
    """Display the agent's internal reasoning steps."""
    if not trace:
        st.caption("No reasoning trace available.")
        return

    for step in trace:
        with st.expander(f"Step {step.iteration}", expanded=False):
            st.markdown(f"**[THOUGHT]** {step.thought}")
            if step.action:
                st.markdown(f"**[ACTION]** `{step.action}` -- `{step.action_input}`")
            if step.observation:
                st.markdown(f"**[OBSERVATION]** {step.observation}")


def render_memory_records(
    records: list[MemoryRecord],
    on_delete: callable | None = None,
) -> None:
    """Display all long-term memory entries with optional delete."""
    if not records:
        st.caption("Long-term memory is empty.")
        return

    for rec in records:
        with st.expander(f"ID: {rec.id}", expanded=False):
            st.text(rec.text)
            st.json(rec.metadata)
            if on_delete:
                st.button(
                    "Delete",
                    key=f"del_mem_{rec.id}",
                    on_click=on_delete,
                    args=(rec.id,),
                )


def render_memory_results(results: list[MemoryResult]) -> None:
    """Display vector search results with distance scores."""
    if not results:
        st.caption("No results.")
        return

    for r in results:
        st.markdown(f"**{r.id}** (distance: {r.distance:.4f})")
        st.text(r.text)
        st.divider()


def render_working_memory(messages: list[dict]) -> None:
    """Display the raw message list from short-term memory."""
    if not messages:
        st.caption("Working memory is empty.")
        return

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        label = f"[{role.upper()}] message {i}"
        with st.expander(label, expanded=(i == 0)):
            st.code(content, language="text")