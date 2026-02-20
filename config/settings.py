"""Application settings: user profile, agent persona, Ollama config."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SETTINGS_PATH = DATA_DIR / "settings.json"


class UserProfile(BaseModel):
    """Basic information about the human user."""

    name: str = "User"
    info: str = ""


class AgentPersona(BaseModel):
    """Controls the agent's character and behaviour."""

    name: str = "Claw"
    role: str = "Personal Assistant"
    system_instructions: str = ""


class OllamaSettings(BaseModel):
    """Connection and generation parameters for Ollama."""

    base_url: str = "http://localhost:11434"
    model: str = "phi4-mini"
    temperature: float = 0.7
    context_window: int = 4096


class AppSettings(BaseModel):
    """Top-level container for all application settings."""

    user: UserProfile = UserProfile()
    persona: AgentPersona = AgentPersona()
    ollama: OllamaSettings = OllamaSettings()


def load_settings() -> AppSettings:
    """Load settings from disk, returning defaults if the file is missing."""
    if SETTINGS_PATH.exists():
        raw = SETTINGS_PATH.read_text(encoding="utf-8")
        return AppSettings.model_validate_json(raw)
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """Persist settings to disk as JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )
