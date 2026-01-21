from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class LlmSummarizer:
    """
    Pluggable LLM summarizer.

    You provide:
      llm_client(system_prompt: str, user_prompt: str) -> str

    The summarizer produces a rolling state summary that behaves like logging:
    - goal / constraints
    - key facts
    - decisions
    - what was tried
    - failures
    - next step
    """
    llm_client: Callable[[str, str], str]

    def compact(self, previous_summary: str, recent_turns: List[Dict[str, str]]) -> str:
        system_prompt = (
            "You summarize agent conversations into a compact state log.\n"
            "Rules:\n"
            "- Keep only stable facts, decisions, failures, and next steps.\n"
            "- Do NOT include fluff.\n"
            "- Do NOT copy the whole transcript.\n"
            "- Write in plain text.\n"
            "- Output format:\n"
            "  Goal:\n"
            "  Key facts:\n"
            "  Decisions:\n"
            "  What we tried:\n"
            "  Failures / risks:\n"
            "  Next step:\n"
        )

        transcript = "\n".join(
            f"{t.get('role')}: {t.get('content', '').strip()}"
            for t in recent_turns
            if t.get("role") in ("user", "assistant") and (t.get("content") or "").strip()
        )

        user_prompt = (
            "Previous state summary (may be empty):\n"
            f"{previous_summary.strip()}\n\n"
            "Recent turns:\n"
            f"{transcript}\n\n"
            "Update the state summary now."
        )

        return self.llm_client(system_prompt, user_prompt).strip()
