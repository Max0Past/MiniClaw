"""Page A: Agent chat interface with live to-do board."""

from __future__ import annotations

import streamlit as st

from agent.core import AgentCore
from ui.components import render_interactive_todo_board


def render_chat_page(agent: AgentCore) -> None:
    """Render the main chat page with a side-panel to-do board."""

    col_chat, col_todo = st.columns([3, 1])

    # -- To-Do Board (right column) ----------------------------------------
    with col_todo:
        st.subheader("To-Do Board")
        render_interactive_todo_board(agent)

    # -- Chat Window (left column) -----------------------------------------
    with col_chat:
        # Initialise chat history/processing state
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
            
            # Proactive message on first load.
            proactive = agent.get_proactive_message()
            if proactive:
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": proactive}
                )
        
        if "processing" not in st.session_state:
            st.session_state["processing"] = False
        
        if "pending_input" not in st.session_state:
            st.session_state["pending_input"] = None

        # Render existing messages
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # If we are currently processing, show a status instead of input
        if st.session_state["processing"]:
            # Process the pending input
            user_input = st.session_state["pending_input"]
            st.session_state["pending_input"] = None
            
            if user_input:
                # Show user message
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Get agent response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = agent.handle_message(user_input)
                    st.markdown(response.answer)

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": response.answer}
                )

            # Unlock and rerun to show the input box again
            st.session_state["processing"] = False
            st.rerun()
        else:
            # Show the chat input only when NOT processing
            user_input = st.chat_input("Message...")

            if user_input:
                # Store input and set processing flag, then rerun
                # On the next run, processing=True so input box is hidden
                st.session_state["pending_input"] = user_input
                st.session_state["processing"] = True
                st.rerun()
