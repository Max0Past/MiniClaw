"""Page B: Under the Hood -- debug and memory inspection."""

from __future__ import annotations

import streamlit as st

from agent.core import AgentCore
from ui.components import (
    render_memory_records,
    render_memory_results,
    render_thought_trace,
    render_working_memory,
)


def render_debug_page(agent: AgentCore) -> None:
    """Render the observability / debug page."""

    # -- Section 1: Working Memory -----------------------------------------
    st.subheader("Working Memory")
    st.caption("Raw messages currently in the context window sent to Ollama.")
    render_working_memory(agent.get_working_memory())

    st.divider()

    # -- Section 2: Long-Term Storage --------------------------------------
    st.subheader("Long-Term Storage")
    st.caption("All entries stored in the vector database.")

    render_memory_records(
        agent.get_long_term_records(),
        on_delete=lambda doc_id: agent.delete_memory(doc_id),
    )

    # Query tester
    st.markdown("**Query Tester**")
    query = st.text_input(
        "Search long-term memory",
        key="debug_ltm_query",
        placeholder="Type a query to find relevant memories...",
    )
    if query:
        results = agent.query_long_term(query, n=5)
        render_memory_results(results)

    st.divider()

    # -- Section 3: Internal Monologue -------------------------------------
    st.subheader("Internal Monologue")
    st.caption("The agent's thought process from the most recent interaction.")
    render_thought_trace(agent.get_thought_trace())
