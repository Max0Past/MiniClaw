"""Thin wrapper around the official ollama Python library."""

from __future__ import annotations

import logging
import time
from typing import Generator

import ollama

from config.settings import OllamaSettings

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Raised when Ollama cannot be reached or the model is missing."""


class OllamaClient:
    """All interaction with Ollama goes through this class."""

    MAX_RETRIES = 3
    BACKOFF_SECONDS = (0.5, 1.0, 2.0)

    def __init__(self, settings: OllamaSettings) -> None:
        self._client = ollama.Client(host=settings.base_url)
        self._model = settings.model
        self._temperature = settings.temperature

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        format: str | None = "json",
        temperature: float | None = None,
    ) -> str:
        """Send messages and return the full assistant response text."""
        return self._call_with_retry(
            messages=messages,
            format=format,
            temperature=temperature or self._temperature,
            stream=False,
        )

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Yield response chunks for streaming display."""
        temp = temperature or self._temperature
        try:
            stream = self._client.chat(
                model=self._model,
                messages=messages,
                stream=True,
                options={"temperature": temp},
            )
            for chunk in stream:
                content = chunk.message.content
                if content:
                    yield content
        except (ollama.ResponseError, ConnectionError) as exc:
            raise OllamaUnavailableError(str(exc)) from exc

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model exists."""
        try:
            self._client.show(self._model)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of all locally available models."""
        try:
            response = self._client.list()
            return [m.model for m in response.models]
        except Exception:
            return []

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    # -- internal ----------------------------------------------------------

    def _call_with_retry(
        self,
        messages: list[dict],
        format: str | None,
        temperature: float,
        stream: bool,
    ) -> str:
        """Call ollama.chat with retry + exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                kwargs: dict = {
                    "model": self._model,
                    "messages": messages,
                    "stream": stream,
                    "options": {"temperature": temperature},
                }
                if format is not None:
                    kwargs["format"] = format
                response = self._client.chat(**kwargs)
                return response.message.content
            except (ollama.ResponseError, ConnectionError) as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "Ollama call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
        raise OllamaUnavailableError(str(last_exc)) from last_exc
