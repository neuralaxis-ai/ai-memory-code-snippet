# ai-memory-code-snippet
# Agent Memory Manager (Context, Not History)

This is a minimal, reusable "agent memory" module.

Principle:
- History is a transcript (archive/audit), not context for reasoning.
- Context is what matters NOW for the next step.
- Retrieval is a hypothesis (can be stale), not truth.

What it does:
1) Enforces that active chat does not grow forever (turn/token guardrails)
2) When thresholds are hit, compacts active turns into a rolling summary using an LLM (log-style state)
3) Archives full transcript to disk
4) Provides keyword search over archived history (tool-style)
5) Builds model messages + a Prompt Ledger (receipt of what went in and why)

## Files
- `agent_memory.py`    -> main module (AgentMemoryManager)
- `summarizer.py`      -> LLM summarizer (pluggable client)
- `history_store.py`   -> JSONL archive + keyword search

## Installation
Copy these 3 Python files into your project.

## Usage

```python
from agent_memory import AgentMemoryManager, MemoryConfig
from history_store import JsonlHistoryStore
from summarizer import LlmSummarizer

# You provide an LLM client function.
# It must accept (system_prompt: str, user_prompt: str) and return text.
def llm_client(system_prompt: str, user_prompt: str) -> str:
    # Example shape only. Plug in OpenAI / Azure / Anthropic / etc.
    # return openai_call(system_prompt=system_prompt, user_prompt=user_prompt)
    raise NotImplementedError

cfg = MemoryConfig(max_turns=5, max_tokens=2500, search_top_k=5)
store = JsonlHistoryStore(".agent_history")
summarizer = LlmSummarizer(llm_client=llm_client)

mem = AgentMemoryManager(cfg=cfg, store=store, summarizer=summarizer, session_id="demo-123")

# record conversation as your app runs
mem.add_user("We are building an agent. Keep responses short.")
mem.add_assistant("Understood.")

# build next model call context
messages, ledger = mem.build_messages(
    system_prompt="You are a helpful assistant. Be concise.",
    user_message="Give me a plan for memory management."
)

# optional: query old history (tool-style)
messages, ledger = mem.build_messages(
    system_prompt="You are a helpful assistant. Be concise.",
    user_message="What did we decide about token limits?",
    history_query="token limit"
)

print(ledger)

