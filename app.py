"""MiniClaw -- Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from agent.core import AgentCore
from config.settings import AppSettings, load_settings
from ui.chat_page import render_chat_page
from ui.debug_page import render_debug_page
from ui.sidebar import render_sidebar

st.set_page_config(page_title="MiniClaw", page_icon="C", layout="wide")


def _get_agent() -> AgentCore:
    """Return the singleton AgentCore, creating it on first run."""
    if "agent" not in st.session_state:
        settings = st.session_state.get("settings", load_settings())
        st.session_state["settings"] = settings
        st.session_state["agent"] = AgentCore(settings)
    return st.session_state["agent"]


def main() -> None:
    agent = _get_agent()
    settings: AppSettings = st.session_state["settings"]

    # -- Health check banner -----------------------------------------------
    if not agent.health_check():
        st.error(
            "Cannot reach Ollama. Make sure it is running "
            f"(model: {settings.ollama.model}). "
            "Try: `ollama serve` and `ollama pull phi4-mini`."
        )

    # -- Sidebar -----------------------------------------------------------
    available_models = agent.list_models()
    new_settings = render_sidebar(settings, available_models)
    if new_settings is not settings:
        agent.reload_settings(new_settings)

    # -- Page navigation ---------------------------------------------------
    page = st.radio(
        "Navigation",
        ["Agent", "Under the Hood"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Agent":
        render_chat_page(agent)
    else:
        render_debug_page(agent)


if __name__ == "__main__":
    main()
