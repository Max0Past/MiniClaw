"""Settings sidebar rendered on every page."""

from __future__ import annotations

import streamlit as st

from config.settings import AppSettings, save_settings


def render_sidebar(settings: AppSettings, available_models: list[str]) -> AppSettings:
    """Draw the settings sidebar and return updated settings if changed.

    The caller is responsible for persisting changes and reloading the agent.
    """
    with st.sidebar:
        st.header("Settings")

        # -- User Profile --
        st.subheader("User Profile")
        user_name = st.text_input("Your name", value=settings.user.name, key="sb_user_name")
        user_info = st.text_area(
            "About you",
            value=settings.user.info,
            key="sb_user_info",
            height=68,
        )

        # -- Agent Persona --
        st.subheader("Agent Persona")
        agent_name = st.text_input("Agent name", value=settings.persona.name, key="sb_agent_name")

        role_options = ["Personal Assistant", "Grumpy Coder", "Friendly Tutor", "Custom"]
        current_role = settings.persona.role
        if current_role in role_options:
            role_index = role_options.index(current_role)
        else:
            role_index = role_options.index("Custom")

        agent_role = st.selectbox("Role", role_options, index=role_index, key="sb_agent_role")
        if agent_role == "Custom":
            agent_role = st.text_input(
                "Custom role",
                value=current_role if current_role not in role_options else "",
                key="sb_custom_role",
            )

        system_instructions = st.text_area(
            "System instructions",
            value=settings.persona.system_instructions,
            key="sb_sys_instr",
            height=100,
        )

        # -- Ollama --
        st.subheader("Ollama")

        if available_models:
            model_options = available_models
            current_model = settings.ollama.model
            if current_model in model_options:
                model_index = model_options.index(current_model)
            else:
                model_options.insert(0, current_model)
                model_index = 0
            selected_model = st.selectbox(
                "Model",
                model_options,
                index=model_index,
                key="sb_model",
            )
        else:
            selected_model = st.text_input(
                "Model",
                value=settings.ollama.model,
                key="sb_model_text",
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.ollama.temperature,
            step=0.1,
            key="sb_temp",
        )

        # -- Save button --
        if st.button("Save Settings", key="sb_save"):
            new_settings = AppSettings(
                user=settings.user.model_copy(
                    update={"name": user_name, "info": user_info}
                ),
                persona=settings.persona.model_copy(
                    update={
                        "name": agent_name,
                        "role": agent_role,
                        "system_instructions": system_instructions,
                    }
                ),
                ollama=settings.ollama.model_copy(
                    update={
                        "model": selected_model,
                        "temperature": temperature,
                    }
                ),
            )
            save_settings(new_settings)
            st.session_state["settings"] = new_settings
            st.success("Settings saved.")
            return new_settings

    return settings
