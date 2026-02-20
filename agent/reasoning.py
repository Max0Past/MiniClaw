"""ReAct-style reasoning loop: Think -> Act -> Observe -> repeat."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from llm.ollama_client import OllamaClient
from memory.manager import MemoryManager
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    """One iteration of the reasoning loop."""

    iteration: int
    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class AgentResponse:
    """Final result returned to the caller."""

    answer: str
    thought_trace: list[ThoughtStep] = field(default_factory=list)


class ReasoningLoop:
    """Executes the ReAct cycle with a configurable iteration cap."""

    MAX_ITERATIONS = 5

    def __init__(
        self,
        client: OllamaClient,
        memory: MemoryManager,
        tools: ToolRegistry,
    ) -> None:
        self._client = client
        self._memory = memory
        self._tools = tools

    def run(self, user_input: str) -> AgentResponse:
        """Execute the full loop and return the agent's final answer."""

        # Add the user message to short-term memory.
        self._memory.add_message("user", user_input)

        # Build context (system prompt + recalled LTM + STM).
        messages = self._memory.build_context(query=user_input)

        trace: list[ThoughtStep] = []

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            raw = self._client.chat(
                messages=messages,
                format="json",
            )

            parsed = self._parse_response(raw)
            step = ThoughtStep(
                iteration=iteration,
                thought=parsed.get("thought", ""),
                action=parsed.get("action"),
                action_input=parsed.get("action_input"),
            )

            # Case 1: the LLM wants to give a final answer.
            if step.action is None or step.action == "null":
                answer = parsed.get("answer", raw)
                step.observation = None
                trace.append(step)

                self._memory.add_message("assistant", answer)
                return AgentResponse(answer=answer, thought_trace=trace)

            # Case 2: the LLM wants to use a tool.
            tool = self._tools.get(step.action)
            if tool is None:
                observation = f"Error: unknown tool '{step.action}'."
            else:
                try:
                    observation = tool.execute(step.action_input or "")
                except Exception as exc:
                    observation = f"Tool error: {exc}"

            step.observation = observation
            trace.append(step)

            # Feed the observation back so the LLM can continue.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{step.action}' returned this result:\n"
                    f"---\n{observation}\n---\n"
                    f"Now respond with a JSON object. "
                    f"If the result answers the question, set action to null and put your answer "
                    f"(based on the result above) in the answer field. "
                    f"If you need another tool, call it."
                ),
            })

        # Safety: if we hit the iteration cap, return whatever we have.
        fallback = "I was unable to complete the request within the allowed steps."
        self._memory.add_message("assistant", fallback)
        return AgentResponse(answer=fallback, thought_trace=trace)

    # -- JSON parsing with fallbacks ----------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse the LLM output as JSON with multiple fallback strategies."""

        # Strategy 1: direct parse.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract JSON object via regex.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: plain-text fallback -- treat everything as a direct answer.
        logger.warning("Failed to parse JSON from LLM. Raw: %s", raw[:200])
        return {
            "thought": "(parse failure -- raw text used as answer)",
            "action": None,
            "action_input": None,
            "answer": raw,
        }
