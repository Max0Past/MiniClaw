"""Coordinates short-term and long-term memory for the agent."""

from __future__ import annotations

from memory.long_term import LongTermMemory, MemoryRecord, MemoryResult
from memory.short_term import ShortTermMemory


class MemoryManager:
    """Unified facade over both memory stores.

    The agent interacts with this class instead of touching STM / LTM
    directly. It also builds the final context list sent to Ollama.
    """

    def __init__(
        self,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
    ) -> None:
        self.stm = stm
        self.ltm = ltm

    # -- short-term --------------------------------------------------------

    def set_system(self, content: str) -> None:
        """Set or update the system prompt in STM."""
        self.stm.set_system(content)

    def add_message(self, role: str, content: str) -> None:
        """Append a user / assistant / tool message to STM."""
        self.stm.add(role, content)

    # -- long-term ---------------------------------------------------------

    def save_to_long_term(
        self, text: str, metadata: dict | None = None
    ) -> str:
        """Persist a fact / preference to the vector store."""
        return self.ltm.store(text, metadata)

    def recall(self, query: str, n: int = 5) -> list[MemoryResult]:
        """Search long-term memory for relevant chunks."""
        return self.ltm.query(query, n_results=n)

    # -- context assembly --------------------------------------------------

    def build_context(self, query: str | None = None) -> list[dict]:
        """Build the full message list for an Ollama call.

        Order:
        1. System message (always first, set in STM)
        2. Recalled long-term memories (injected as a system note)
        3. Recent conversation messages from STM
        """
        messages: list[dict] = []

        stm_msgs = self.stm.get_messages()

        # 1. System message
        if stm_msgs and stm_msgs[0]["role"] == "system":
            messages.append(stm_msgs[0])
            stm_msgs = stm_msgs[1:]

        # 2. Recalled context (if a query is provided)
        if query:
            recalled = self.recall(query)
            if recalled:
                facts = "\n".join(f"- {r.text}" for r in recalled)
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Recalled facts from long-term memory:\n" + facts
                        ),
                    }
                )

        # 3. Conversation history
        messages.extend(stm_msgs)
        return messages

    # -- debug helpers -----------------------------------------------------

    def get_working_memory(self) -> list[dict]:
        """Return the raw STM messages for the debug page."""
        return self.stm.to_raw()

    def get_long_term_records(self) -> list[MemoryRecord]:
        """Return all LTM entries for the debug page."""
        return self.ltm.get_all()

    def query_long_term(
        self, query: str, n: int = 5
    ) -> list[MemoryResult]:
        """Expose LTM search for the debug page query tester."""
        return self.ltm.query(query, n_results=n)
