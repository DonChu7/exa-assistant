from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from lrg_test_tool.brain.router import build_router_prompt
from lrg_test_tool.brain.writer import build_writer_prompt
from lrg_test_tool.brain.llm_client import WorkflowLLMClient
from lrg_test_tool.schemas.schemas import QueryPayload


def sanitize_router_output(raw: str) -> str:
    """Normalize LLM router output into bare JSON text."""
    s = (raw or "").strip()
    if not s:
        return s

    if s.startswith("```") and s.endswith("```"):
        parts = s.strip("`").split("\n", 1)
        if parts:
            if len(parts) == 1:
                s = parts[0]
            else:
                language, rest = parts
                if language.strip().lower() == "json":
                    s = rest.strip()
                else:
                    s = rest.strip()

    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    return s.strip()


@dataclass
class RouterWriterConfig:
    router_url: str
    router_workflow_id: str
    writer_url: str
    writer_workflow_id: str
    router_verify: Optional[str] = None
    writer_verify: Optional[str] = None
    debug_router: bool = False

    @classmethod
    def from_env(cls) -> "RouterWriterConfig":
        router_url = os.getenv(
            "LRG_ROUTER_WORKFLOW_URL",
            "https://idea-test-generation-1.dev3sub4phx.databasede3phx.oraclevcn.com/workflow/processcli",
        )
        router_id = os.getenv("LRG_ROUTER_WORKFLOW_ID", "1814")

        writer_url = os.getenv(
            "LRG_WRITER_WORKFLOW_URL",
            "https://idea-test-generation-1.dev3sub4phx.databasede3phx.oraclevcn.com/workflow/processcli",
        )
        writer_id = os.getenv("LRG_WRITER_WORKFLOW_ID", "1814")

        cfg = cls(
            router_url=router_url,
            router_workflow_id=router_id,
            writer_url=writer_url,
            writer_workflow_id=writer_id,
            router_verify=os.getenv("LRG_ROUTER_VERIFY"),
            writer_verify=os.getenv("LRG_WRITER_VERIFY"),
            debug_router=os.getenv("EXA_ASSIST_DEBUG_ROUTER", "0") == "1",
        )
        return cfg


class RouterWriterPipeline:
    SPIKE_TERMS = {
        "spike",
        "spikes",
        "suddenly slower",
        "suddenly became slower",
        "regression",
        "got slower",
        "why is this slower",
        "increase in runtime",
        "runtime increased",
    }

    HISTORY_TERMS = {
        "last 7 days",
        "last 30 days",
        "last month",
        "last few days",
        "history",
        "trend",
        "recent runtimes",
        "over time",
        "timeline",
    }

    def __init__(self, config: RouterWriterConfig):
        self.config = config
        self.router_client = WorkflowLLMClient(
            config.router_url, config.router_workflow_id, verify=config.router_verify
        )
        self.writer_client = WorkflowLLMClient(
            config.writer_url, config.writer_workflow_id, verify=config.writer_verify
        )
        self.last_router_raw: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RouterWriterPipeline":
        cfg = RouterWriterConfig.from_env()
        return cls(cfg)

    def route(self, user_query: str) -> QueryPayload:
        prompt = build_router_prompt(user_query)
        raw = self.router_client.call(prompt)
        self.last_router_raw = raw
        if self.config.debug_router:
            print("[router raw output]\n" + raw)

        clean = sanitize_router_output(raw)
        data = json.loads(clean)
        qp = QueryPayload(**data)

        text_lower = user_query.lower()
        if any(term in text_lower for term in self.SPIKE_TERMS):
            qp.mode = "runtime_spike"
        elif qp.mode == "runtime_spike" and any(
            term in text_lower for term in self.HISTORY_TERMS
        ):
            qp.mode = "search"

        return qp

    def compose(self, user_query: str, tool_output: dict) -> str:
        prompt = build_writer_prompt(user_query, tool_output)
        return self.writer_client.call(prompt)


__all__ = [
    "RouterWriterConfig",
    "RouterWriterPipeline",
    "sanitize_router_output",
]
