from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class JsonlHistoryStore:
    """
    Simple JSONL transcript archive (one file per session).

    - Archive is NOT injected into the LLM context by default.
    - Archive exists for audit/debug and optional tool-style search.
    """

    def __init__(self, root_dir: str = ".agent_history") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def append(self, session_id: str, turns: Iterable[Dict[str, Any]]) -> None:
        p = self._path(session_id)
        with p.open("a", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    def read_all(self, session_id: str) -> List[Dict[str, Any]]:
        p = self._path(session_id)
        if not p.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def keyword_search(self, session_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Minimal keyword search baseline.

        Why keyword search?
        - Zero dependencies
        - Easy to reason about
        - Keeps focus on memory discipline, not retrieval magic
        """
        q = (query or "").strip().lower()
        if not q:
            return []

        hits: List[Dict[str, Any]] = []
        for row in self.read_all(session_id):
            content = (row.get("content") or "")
            score = content.lower().count(q)
            if score > 0:
                hits.append({"score": float(score), "turn": row})

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]
