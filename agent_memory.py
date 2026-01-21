from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from history_store import JsonlHistoryStore
from summarizer import LlmSummarizer


@dataclass
class MemoryConfig:
    """
    Guardrails:
    - max_turns: number of active turns before compaction
    - max_tokens: rough token estimate threshold before compaction
    - search_top_k: number of history hits to inject when searching
    - summary_max_chars: safety cap to keep summary short
    """
    max_turns: int = 5
    max_tokens: int = 2500
    search_top_k: int = 5
    summary_max_chars: int = 1400


@dataclass
class PromptLedger:
    """
    Receipt for debugging:
    - what got injected
    - why it got injected
    - rough token estimate
    """
    session_id: str
    token_estimate: int
    included_summary: bool
    included_active_turns: int
    included_history_hits: int
    notes: Dict[str, Any] = field(default_factory=dict)


def _rough_tokens(text: str) -> int:
    # Cheap heuristic: ~4 chars/token (good enough for guardrails)
    return max(1, len(text) // 4)


def _estimate_tokens(turns: List[Dict[str, Any]]) -> int:
    return sum(_rough_tokens(t.get("content", "")) for t in turns)


class AgentMemoryManager:
    """
    Minimal, reusable memory manager.

    Keeps:
    - active_turns: small working set (context now)
    - rolling_summary: compact state (log-style)
    Archives:
    - full transcript via JsonlHistoryStore

    Builds:
    - messages for the next LLM call
    - PromptLedger (debug receipt)
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        store: JsonlHistoryStore,
        summarizer: LlmSummarizer,
        session_id: str,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.summarizer = summarizer
        self.session_id = session_id

        self.active_turns: List[Dict[str, Any]] = []
        self.rolling_summary: str = ""

    # --- recording turns ---
    def add_user(self, content: str) -> None:
        self._add_turn("user", content)

    def add_assistant(self, content: str) -> None:
        self._add_turn("assistant", content)

    def _add_turn(self, role: str, content: str) -> None:
        self.active_turns.append(
            {
                "role": role,
                "content": content,
                "ts": time.time(),
                "id": uuid.uuid4().hex,
            }
        )

    # --- compaction policy ---
    def _should_compact(self) -> bool:
        if len(self.active_turns) >= self.cfg.max_turns:
            return True
        if _estimate_tokens(self.active_turns) >= self.cfg.max_tokens:
            return True
        return False

    def _compact(self) -> None:
        """
        Compaction:
        1) archive current active turns
        2) update rolling_summary via LLM summarizer (log-style)
        3) clear active turns
        """
        if not self.active_turns:
            return

        self.store.append(self.session_id, self.active_turns)

        # Summarize only a recent slice to avoid huge prompts for the summarizer.
        recent = self.active_turns[-10:] if len(self.active_turns) > 10 else self.active_turns
        new_summary = self.summarizer.compact(self.rolling_summary, recent)

        # Safety cap
        self.rolling_summary = (new_summary[: self.cfg.summary_max_chars]).strip()
        self.active_turns = []

    # --- prompt build (the only thing that matters at runtime) ---
    def build_messages(
        self,
        system_prompt: str,
        user_message: str,
        history_query: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], PromptLedger]:
        """
        Returns:
        - messages: list[{"role": "...", "content": "..."}] for your LLM call
        - ledger: debug receipt

        history_query:
        - if set, injects top hits from archived transcript
        - injected as "potentially stale" info (hypothesis)
        """
        if self._should_compact():
            self._compact()

        history_hits: List[Dict[str, Any]] = []
        if history_query:
            history_hits = self.store.keyword_search(
                self.session_id, history_query, top_k=self.cfg.search_top_k
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]

        if self.rolling_summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": "Rolling summary (state):\n" + self.rolling_summary.strip(),
                }
            )

        messages.extend({"role": t["role"], "content": t["content"]} for t in self.active_turns)

        if history_hits:
            blob = "\n\n".join(
                f"[score={h['score']}] {h['turn'].get('role')}: {h['turn'].get('content')}"
                for h in history_hits
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Potentially relevant history (treat as stale unless verified):\n"
                        + blob
                    ),
                }
            )

        messages.append({"role": "user", "content": user_message})

        token_estimate = (
            _rough_tokens(system_prompt)
            + _rough_tokens(self.rolling_summary)
            + _estimate_tokens(self.active_turns)
            + sum(_rough_tokens(h["turn"].get("content", "")) for h in history_hits)
            + _rough_tokens(user_message)
        )

        ledger = PromptLedger(
            session_id=self.session_id,
            token_estimate=token_estimate,
            included_summary=bool(self.rolling_summary.strip()),
            included_active_turns=len(self.active_turns),
            included_history_hits=len(history_hits),
            notes={
                "history_query": history_query,
                "guardrails": f"max_turns={self.cfg.max_turns}, max_tokens={self.cfg.max_tokens}",
            },
        )
        return messages, ledger
