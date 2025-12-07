# app.py — slim Slack bot wired to MCP tools

import os as OS
import re
import json
import time
import base64
import zipfile
from io import BytesIO
import tempfile as TF
from pathlib import Path
import threading
import subprocess
import contextvars
from collections import defaultdict
import requests
from typing import Any, Dict, List, Literal, Optional
import traceback
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import html
import logging

from dotenv import load_dotenv
# from slack_bolt import App
# from slack_bolt.adapter.socket_mode import SocketModeHandler
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from mcp_client import PersistentMCPClient
from skills_loader import load_skill_catalog
from preset_tools import build_preset_tools

import uuid
from metrics_utils import feedback_blocks, record_feedback_click, append_jsonl, utc_iso

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from llm_provider import make_llm
from genai4test_chat_agent import GenAI4TestChatAgent

# ---------------------------------------------------------------------------
# Boot / env
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LANGGRAPH_LOG_PATH = OS.getenv("LANGGRAPH_LOG_PATH", str((BASE_DIR / "metrics/langgraph_debug.log").resolve()))

class LangGraphFileLogger(BaseCallbackHandler):
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        directory = OS.path.dirname(self.path)
        if directory and not OS.path.isdir(directory):
            OS.makedirs(directory, exist_ok=True)

    def _write(self, event: str, payload: dict):
        record = {"ts": utc_iso(), "event": event}
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _truncate(self, value, limit: int = 2000):
        text = str(value)
        return text if len(text) <= limit else text[:limit] + "…"

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name") if isinstance(serialized, dict) else str(serialized)
        self._write("chain_start", {"name": name, "inputs": self._truncate(inputs)})

    def on_chain_end(self, outputs, **kwargs):
        self._write("chain_end", {"outputs": self._truncate(outputs)})

    def on_chain_error(self, error, **kwargs):
        self._write("chain_error", {"error": str(error)})

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name") if isinstance(serialized, dict) else str(serialized)
        self._write("tool_start", {"name": name, "input": self._truncate(input_str)})

    def on_tool_end(self, output, **kwargs):
        self._write("tool_end", {"output": self._truncate(output)})

    def on_tool_error(self, error, **kwargs):
        self._write("tool_error", {"error": str(error)})

    def on_llm_start(self, serialized, prompts, **kwargs):
        name = serialized.get("name") if isinstance(serialized, dict) else str(serialized)
        truncated_prompts = [self._truncate(p) for p in prompts] if isinstance(prompts, list) else self._truncate(prompts)
        self._write("llm_start", {"name": name, "prompts": truncated_prompts})

    def on_llm_end(self, response, **kwargs):
        self._write("llm_end", {"response": self._truncate(response)})

    def on_llm_error(self, error, **kwargs):
        self._write("llm_error", {"error": str(error)})


_LANGGRAPH_LOGGER = LangGraphFileLogger(LANGGRAPH_LOG_PATH)

SLACK_BOT_TOKEN = OS.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = OS.getenv("SLACK_APP_TOKEN", "")
FEEDBACK_PATH = OS.getenv("FEEDBACK_PATH", "./metrics/feedback.jsonl")

# Skills catalog powers MCP configuration
SKILLS      = load_skill_catalog()
RUNINTEG_SK = SKILLS.get("runintegration")
OEDA_SK     = SKILLS.get("oeda")
RAG_SK      = SKILLS.get("exa23ai-rag")
SUM_SK      = SKILLS.get("summarizer")
GENAI4TEST_CHAT_SK = SKILLS.get("genai4test-chat")
LABEL_HEALTH_SK    = SKILLS.get("label-health")
REALHW_SK   = SKILLS.get("realhw")
GENAI4TEST_FUNC_SK = SKILLS.get("genai4test-func")
LRGTEST_SK  = SKILLS.get("lrg-tests")


def _build_skill_tool_summary() -> str:
    lines: List[str] = []
    for entry in SKILLS:
        tool_entries = entry.tools or []
        if not tool_entries:
            continue
        lines.append(f"{entry.name} ({entry.server_id}):")
        for meta in tool_entries:
            name = meta.get("name", "unknown")
            desc = meta.get("description") or ""
            intents = meta.get("intents") or []
            intent_text = f" intents={intents}" if intents else ""
            lines.append(f"  - {name}: {desc}{intent_text}")
    return "\n".join(lines)


SKILL_TOOL_SUMMARY = _build_skill_tool_summary()

def _skill_command(skill_entry, fallback_env_var):
    if skill_entry:
        return skill_entry.command
    value = OS.getenv(fallback_env_var, "")
    return value.split() if value else []

def _skill_env(skill_entry):
    return skill_entry.env if skill_entry else {}

RUNINTEG_CMD = _skill_command(RUNINTEG_SK, "RUNINTEG_CMD") or ["python", "runintegration_server.py"]
OEDA_CMD     = _skill_command(OEDA_SK, "OEDA_CMD") or ["python", "oeda_server.py"]
RAG_CMD      = _skill_command(RAG_SK, "RAG_CMD") or ["python", "exa23ai_rag_server.py"]
SUM_CMD      = _skill_command(SUM_SK, "SUM_CMD") or ["python", "summarizer_server.py"]
GENAI4TEST_CMD = _skill_command(GENAI4TEST_CHAT_SK, "GENAI4TEST_CMD") or ["python", "genai4test_chat_server.py"]
LABELHEALTH_CMD = _skill_command(LABEL_HEALTH_SK, "LABELHEALTH_CMD") or ["python", "label_health_server.py"]
REALHW_CMD    = _skill_command(REALHW_SK, "REALHW_CMD") or ["python", "realhw_mcp_server.py"]
GENAI4TEST_FUNC_CMD = _skill_command(GENAI4TEST_FUNC_SK, "GENAI4TEST_FUNC_CMD") or ["python", "genai4test_server.py"]
LRGTEST_CMD   = _skill_command(LRGTEST_SK, "LRGTEST_CMD") or ["python", "lrg_test_mcp_server.py"]
GENAI4TEST_FOLLOWUP_AGENT = OS.getenv("GENAI4TEST_CHAT_AGENT_NAME", "chat_agent")

# Default genoedaxml path (allowlisted in oeda_server)
DEFAULT_GENXML = OS.getenv("GENOEDAXML_PATH",
    "/net/dbdevfssmnt-shared01.dev3fss1phx.databasede3phx.oraclevcn.com/exadata_dev_image_oeda/genoeda/genoedaxml"
)

# ---------------------------------------------------------------------------
# feedbacks 
# ---------------------------------------------------------------------------
async def post_with_feedback(app, channel_id: str, thread_ts: str | None, text: str, *,
                       context: dict | None = None, user_id: str | None = None, client: AsyncWebClient | None = None,) -> str:
    """Post the answer text and then a separate feedback-control reply."""
    # resolve user label if possible
    user_label = None
    if client and user_id:
        user_label = await _get_user_label(client, user_id)
    if not user_label:
        user_label = "unknown"

    uid = str(uuid.uuid4())
    record = {
        "user": user_label,
        "uuid": uid,
        "ts": utc_iso(),
        "context": context or {},
        "original_text": text,
    }
    # Persist the full payload once 
    append_jsonl(FEEDBACK_PATH, record)

    tiny_payload = json.dumps({"uuid": uid})

    res = await app.client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=text,
    )

    parent_ts = thread_ts or res["ts"]
    feedback_msg = {
        "channel": channel_id,
        "thread_ts": parent_ts,
        "text": "Feedback controls",
        "blocks": feedback_blocks(text, voted=None, payload_json=tiny_payload),
    }
    try:
        await app.client.chat_postMessage(**feedback_msg)
    except SlackApiError as e:
        print("[Slack] failed to post feedback controls:", e)

    return res["ts"]


def _parse_env_params_from_text(text: str):
    t = (text or "").lower()
    if re.search(r"\br1x\b", t): return {"ENV": "r1x"}
    if re.search(r"\br1\b",  t): return {"ENV": "r1"}
    return None

def scp_file_with_key(file_path: str, destination: str, ssh_key_path: str) -> bool:
    try:
        cmd = ["scp", "-i", ssh_key_path, file_path, destination]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] SCP failed:", e.stderr.decode())
        return False

def _pick_genai4test_filename(res: dict) -> str:
    """
    Derive a reasonable filename from genai4test result.
    Prefer URL basename; otherwise infer from script; default to .txt.
    """
    # Prefer URL basename if available
    url = (res.get("absolute_file_url") or res.get("file_url") or "").strip()
    if url:
        name = OS.path.basename(urlparse(url).path)
        if name:
            return name

    # Infer from inline script (sql/sh) if present
    script = (res.get("sql") or "").lstrip()
    if script.startswith("#"):  # looks like shell
        return "genai4test_script.sh"
    # crude SQL heuristic
    if script.startswith("Rem"):
        return "genai4test_script.sql"
    return "genai4test_output.txt"

def _download_bytes(url: str, verify) -> bytes:
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    # If hitting GenAI4Test endpoints, include source header
    try:
        if "/genai4test/" in url:
            s.headers["x-source"] = "Slack"
    except Exception:
        pass
    r = s.get(url, timeout=(10, 600), verify=verify)
    r.raise_for_status()
    return r.content

def _is_text_like(filename: str, mimetype: str) -> bool:
    name = (filename or "").lower()
    mt = (mimetype or "").lower()
    if name.endswith((".sql", ".sh", ".txt", ".tsc", ".py")):
        return True
    if mt.startswith("text/"):
        return True
    return False

def _extract_first_code_block(text: str) -> str | None:
    try:
        m = re.search(r"```[A-Za-z0-9_-]*\n(.*?)```", text or "", re.DOTALL)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None

def _extract_test_steps_text(text: str) -> str | None:
    """Heuristically extract test steps from plain text (no code fences).
    Looks for 'test step' keywords or multiple numbered/bulleted lines.
    Returns a trimmed multi-line string or None.
    """
    if not text:
        return None
    t = (text or "").strip()
    # If keyword present, start from there
    m = re.search(r"(?is)(test\s*steps?\s*:?.*)", t)
    if m:
        candidate = m.group(1).strip()
    else:
        candidate = t
    # Count numbered/bulleted lines (allow no space after the number, e.g., '1.Live')
    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    bullet_re = re.compile(r"^\s*(\d+[\.)]?|[-*])\s*", re.IGNORECASE)
    num_like = sum(1 for ln in lines if bullet_re.match(ln))
    # Heuristics: at least 2 such lines, or keyword present and sufficient length
    if num_like >= 2 or (m and len(candidate) >= 40):
        # Limit length to avoid Slack issues; the backend accepts full text but we store in state
        return candidate[:8000].strip()
    return None

async def _get_first_name(client, user_id: str) -> str:
    """
    Try multiple profile fields (first_name/given_name/display/real_name),
    fall back to the local-part of email, then to 'there'.
    """
    try:
        ui = await client.users_info(user=user_id)   # requires 'users:read' scope
        user = (ui or {}).get("user", {}) or {}
        prof = user.get("profile", {}) or {}

        candidates = [
            prof.get("first_name"),                 # legacy; often empty
            prof.get("given_name"),                 # some workspaces use this
            prof.get("display_name_normalized"),
            prof.get("display_name"),
            prof.get("real_name_normalized"),
            prof.get("real_name"),
            user.get("name"),                       # legacy handle
        ]
        first = next((c for c in candidates if c), None)
        if first and " " in first:
            first = first.split()[0]

        if not first:
            email = prof.get("email")
            if email and "@" in email:
                first = email.split("@", 1)[0]

        return first or "there"
    except Exception as e:
        print("[users_info] error:", type(e).__name__, e)
        return "there"
    

async def _get_user_label(client: AsyncWebClient, user_id: str) -> str:
    """
    Return a readable label for a Slack user:
    'Full Name (@username)' if available,
    else '@username',
    else '<@U12345>' as a last resort.
    """
    if not user_id:
        return "unknown"
    try:
        ui = await client.users_info(user=user_id)  # needs users:read
        user = (ui or {}).get("user", {}) or {}
        prof = user.get("profile", {}) or {}
        real = prof.get("real_name_normalized") or prof.get("real_name") or ""
        handle = user.get("name") or prof.get("display_name") or ""
        if real and handle:
            return f"{real} (@{handle})"
        if real:
            return real
        if handle:
            return f"@{handle}"
        return f"<@{user_id}>"
    except SlackApiError as e:
        # Scope missing or other error — don’t break metrics
        print("[users_info] scope/error:", getattr(e, "response", {}).data if getattr(e, "response", None) else e)
        return f"<@{user_id}>"
    except Exception as e:
        print("[users_info] error:", type(e).__name__, e)
        return f"<@{user_id}>"

def feedback_thanks_blocks(text: str, sentiment: str) -> list[dict]:
    """
    A grey 'thanks' note below the original text.
    sentiment: "up" or "down"
    """
    icon = "👍" if sentiment == "up" else "👎"
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"_Thanks for your feedback {icon}_"}
        ]},
    ]


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    widths = [len(str(h)) for h in headers]
    normalized_rows: list[list[str]] = []
    for row in rows:
        normalized = []
        for idx, header in enumerate(headers):
            value = ""
            if idx < len(row) and row[idx] is not None:
                value = str(row[idx])
            normalized.append(value)
            if len(value) > widths[idx]:
                widths[idx] = len(value)
        normalized_rows.append(normalized)
    widths = [w + 2 for w in widths]
    header_line = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(normalized[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for normalized in normalized_rows
    ]
    return "\n".join([header_line, separator_line] + body_lines)


def _align_table_block(body: str) -> str:
    lines = body.splitlines()
    content = [ln for ln in lines if ln.strip()]
    if not content or any("|" not in ln for ln in content):
        return body

    parsed_rows = []
    separator_idx: set[int] = set()
    max_cols = 0
    for idx, line in enumerate(content):
        raw = line.strip()
        cells = raw.split("|")
        if raw.startswith("|"):
            cells = cells[1:]
        if raw.endswith("|"):
            cells = cells[:-1]
        cells = [c.strip() for c in cells]
        if not cells:
            return body
        if all(set(c) <= set("-: ") for c in cells):
            separator_idx.add(idx)
        parsed_rows.append(cells)
        max_cols = max(max_cols, len(cells))

    if max_cols == 0:
        return body

    widths = [0] * max_cols
    for idx, row in enumerate(parsed_rows):
        if idx in separator_idx:
            continue
        for col in range(max_cols):
            cell = row[col] if col < len(row) else ""
            if len(cell) > widths[col]:
                widths[col] = len(cell)
    widths = [w + 2 for w in widths]

    aligned_lines = []
    for idx, row in enumerate(parsed_rows):
        if idx in separator_idx:
            segs = ["-" * widths[col] for col in range(max_cols)]
        else:
            segs = [
                (row[col] if col < len(row) else "").ljust(widths[col])
                for col in range(max_cols)
            ]
        aligned_lines.append("| " + " | ".join(segs) + " |")

    aligned = "\n".join(aligned_lines)
    if body.endswith("\n"):
        aligned += "\n"
    return aligned


def _align_tables_in_text(text: str) -> str:
    code_pattern = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)

    def repl(match: re.Match) -> str:
        lang = match.group(1) or ""
        body = match.group(2)
        aligned = _align_table_block(body)
        return f"```{lang}\n{aligned}```"

    text = code_pattern.sub(repl, text)

    lines = text.splitlines()
    result: list[str] = []
    buffer: list[str] = []
    inside_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if buffer:
                result.extend(_align_table_block("\n".join(buffer)).splitlines())
                buffer.clear()
            inside_code = not inside_code
            result.append(line)
            continue

        if inside_code:
            result.append(line)
            continue

        if "|" in line:
            buffer.append(line)
        else:
            if buffer:
                result.extend(_align_table_block("\n".join(buffer)).splitlines())
                buffer.clear()
            result.append(line)

    if buffer:
        result.extend(_align_table_block("\n".join(buffer)).splitlines())

    return "\n".join(result)

def _format_lrg_table(lrgs: list[str], max_rows: int = 50) -> tuple[str, str]:
    """Build a monospace table for LRG names."""
    lrgs = [l for l in (lrgs or []) if l]
    shown = lrgs[:max_rows]
    rows = [[str(i), l] for i, l in enumerate(shown, 1)]
    table = _render_markdown_table(["#", "LRG"], rows) if rows else ""
    extra = f"\n(+{len(lrgs)-max_rows} more)" if len(lrgs) > max_rows else ""
    return table, extra

def _format_env_table(rows: list[dict], max_rows: int = 30) -> tuple[str, str]:
    """
    Build a monospace table for RunIntegration envs.
    rows: [{"rack":"...","type":"..."}...]
    Returns (table_text, suffix_message_if_truncated)
    """
    rows = [r for r in (rows or []) if r.get("rack")]
    shown = rows[:max_rows]
    table_rows = []
    for i, r in enumerate(shown, 1):
        table_rows.append([str(i), r.get("rack", ""), r.get("type", "")])
    table = _render_markdown_table(["#", "RACK", "TYPE"], table_rows) if table_rows else ""
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return table, extra

def _unwrap_env_list(res: Any, keys: list[str]) -> list:
    """
    From a response that can be:
      - dict with one of keys pointing to a list
      - list directly
      - anything else
    return a list (or []).
    """
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    return []

def _parse_env_item(it: Any) -> dict | None:
    """
    Accept:
      - dict {"rack_name": "...", "deploy_type": "..."} or {"rack":"...","type":"..."} etc.
      - string "rack : type" (or "rack:type")
    Return: {"rack":"...","type":"..."} or None
    """
    if isinstance(it, dict):
        rack = it.get("rack") or it.get("rack_name") or it.get("full_rack_name")
        typ  = it.get("type") or it.get("deploy_type") or ""
        if rack:
            return {"rack": str(rack).strip(), "type": str(typ).strip()}
        return None
    if isinstance(it, str):
        parts = it.split(":", 1)
        if parts:
            rack = parts[0].strip()
            typ  = parts[1].strip() if len(parts) > 1 else ""
            if rack:
                return {"rack": rack, "type": typ}
        return None
    return None

def _normalize_idle_envs(res: Any) -> list[dict]:
    """
    Handle dict{"idle_envs":[...]} or list[...] (dicts or strings).
    """
    raw_list = _unwrap_env_list(res, keys=["idle_envs", "result", "items", "data"])
    rows: list[dict] = []
    for it in raw_list:
        parsed = _parse_env_item(it)
        if parsed:
            rows.append(parsed)
    return rows

def _normalize_disabled_envs(res: Any) -> list[dict]:
    """
    Handle dict{"disabled_envs":[...]} or list[...] (dicts or strings).
    """
    raw_list = _unwrap_env_list(res, keys=["disabled_envs", "result", "items", "data"])
    rows: list[dict] = []
    for it in raw_list:
        parsed = _parse_env_item(it)
        if parsed:
            rows.append(parsed)
    return rows

def _unwrap_list(res: Any, keys: list[str]) -> list:
    """Return a list from dict containers or pass through a list; else []."""
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    return []

def _normalize_lrg_with_difs(res: Any) -> list[dict]:
    """
    Expect: {"lrgs_with_difs":[{"lrg":...,"sucs":...,"difs":...,"nwdif":...,"intdif":...,"szdif":...,"comments":...}]}
    Return rows as [{"lrg","sucs","difs","nwdif","intdif","szdif","comments"}]
    """
    items = _unwrap_list(res, keys=["lrgs_with_difs", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict): 
            continue
        rows.append({
            "lrg": str(it.get("lrg","")).strip(),
            "sucs": str(it.get("sucs","")).strip(),
            "difs": str(it.get("difs","")).strip(),
            "nwdif": str(it.get("nwdif","")).strip(),
            "intdif": str(it.get("intdif","")).strip(),
            "szdif": str(it.get("szdif","")).strip(),
            "comments": (str(it.get("comments","")).strip() or ""),
        })
    # keep only rows with an LRG
    return [r for r in rows if r["lrg"]]

def _normalize_dif_details(res: Any) -> list[dict]:
    """
    Expect: {"dif_details":[{"lrg","name","rti_number","rti_assigned_to","rti_status","text","comments"}...]}
    """
    items = _unwrap_list(res, keys=["dif_details", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict): 
            continue
        rows.append({
            "lrg": str(it.get("lrg","")).strip(),
            "name": str(it.get("name","")).strip(),
            "rti": str(it.get("rti_number","")).strip(),
            "assignee": str(it.get("rti_assigned_to","")).strip(),
            "status": str(it.get("rti_status","")).strip(),
            "text": str(it.get("text","")).strip(),
            "comments": str(it.get("comments","")).strip(),
        })
    return [r for r in rows if r["lrg"] or r["name"]]

def _format_lrg_with_difs_table(rows: list[dict], max_rows: int = 30) -> tuple[str, str]:
    """Monospace table for lrg+counts."""
    rows = rows or []
    shown = rows[:max_rows]
    table_rows = []
    for i, r in enumerate(shown, 1):
        table_rows.append([
            str(i),
            str(r.get("lrg", "")),
            str(r.get("sucs", "")),
            str(r.get("difs", "")),
            str(r.get("nwdif", "")),
            str(r.get("intdif", "")),
            str(r.get("szdif", "")),
        ])
    table = _render_markdown_table(["#", "LRG", "SUCS", "DIFS", "NWDIF", "INTDIF", "SZDIF"], table_rows) if table_rows else ""
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return table, extra

def _format_dif_details_table(rows: list[dict], max_rows: int = 20) -> tuple[str, str]:
    """Monospace table for dif details (compact)."""
    rows = rows or []
    shown = rows[:max_rows]
    def clip(s, n): return (s[:n-1] + "…") if len(s) > n else s
    table_rows = []
    for i, r in enumerate(shown, 1):
        table_rows.append([
            str(i),
            clip(r.get("lrg", ""), 28),
            clip(r.get("name", ""), 26),
            clip(r.get("rti", ""), 10),
            clip(r.get("status", ""), 7),
            clip(r.get("assignee", ""), 12),
        ])
    headers = ["#", "LRG", "NAME", "RTI", "STATUS", "ASSIGNEE"]
    table = _render_markdown_table(headers, table_rows) if table_rows else ""
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return table, extra

def _unwrap_list_generic(res: Any, keys: list[str]) -> list:
    """Return a list from a dict container (by keys) or pass through list; else []."""
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    return []

def _normalize_dif_occurrences(res: Any) -> list[dict]:
    """
    Server shape (find_dif_occurrence):
      {"dif_occurrences":[{"label","lrg","name","rti_number","rti_assigned_to","comments","text"?}], ...}
    Normalize to rows: [{"label","lrg","name","rti","assignee"}]
    """
    items = _unwrap_list_generic(res, keys=["dif_occurrences", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append({
            "label":      str(it.get("label","")).strip(),
            "lrg":        str(it.get("lrg","")).strip(),
            "name":       str(it.get("name","")).strip(),
            "rti":        str(it.get("rti_number","")).strip(),
            "assignee":   str(it.get("rti_assigned_to","")).strip(),
            # "comments": str(it.get("comments","")).strip(),   # optional column, can be long
        })
    return [r for r in rows if r["label"] or r["lrg"] or r["name"]]

def _normalize_widespread_issues(res: Any) -> list[dict]:
    """
    Server shape (find_widespread_issues):
      {"widespread_issues":[{"name":"dif_name","lrgs":"l1,l2,..."}], ...}
    Normalize to rows: [{"name","lrgs","count"}]
    """
    items = _unwrap_list_generic(res, keys=["widespread_issues", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name","")).strip()
        lrgs = str(it.get("lrgs","")).strip()
        cnt = len([x for x in (lrgs.split(",") if lrgs else []) if x.strip()])
        rows.append({"name": name, "lrgs": lrgs, "count": cnt})
    return [r for r in rows if r["name"]]

def _format_dif_occ_table(rows: list[dict], max_rows: int = 25) -> tuple[str, str]:
    """Monospace table for dif occurrences."""
    rows = rows or []
    shown = rows[:max_rows]
    def clip(s, n): 
        s = s or ""
        return (s[:n-1] + "…") if len(s) > n else s
    table_rows = []
    for i, r in enumerate(shown, 1):
        table_rows.append([
            str(i),
            clip(r.get("label", ""), 27),
            clip(r.get("lrg", ""), 27),
            clip(r.get("name", ""), 27),
            clip(r.get("rti", ""), 10),
            clip(r.get("assignee", ""), 12),
        ])
    headers = ["#", "LABEL", "LRG", "DIF", "RTI", "ASSIGNEE"]
    table = _render_markdown_table(headers, table_rows) if table_rows else ""
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return table, extra

def _format_widespread_table(rows: list[dict], max_rows: int = 20) -> tuple[str, str]:
    """Monospace table for widespread issues (dif name + LRG list + count)."""
    rows = rows or []
    shown = rows[:max_rows]
    def clip(s, n): 
        s = s or ""
        return (s[:n-1] + "…") if len(s) > n else s
    table_rows = []
    for i, r in enumerate(shown, 1):
        table_rows.append([
            str(i),
            clip(r.get("name", ""), 26),
            str(r.get("count", "")),
            clip(r.get("lrgs", ""), 60),
        ])
    headers = ["#", "DIF NAME", "COUNT", "LRGs"]
    table = _render_markdown_table(headers, table_rows) if table_rows else ""
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return table, extra

def _split_for_slack(text: str, max_chars: int = 3500) -> list[str]:
    """
    Split long Slack messages into chunks under the safe limit, keeping code fences balanced.
    - Tracks triple-backtick code fences (```[lang] ... ```)
    - If a chunk ends inside a code block, appends a closing fence and reopens on the next chunk.
    - Skips a leading closing fence at the beginning of the next chunk to avoid duplicate close/open.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0
    inside = False
    lang: str | None = None
    reopen_prefix: str = ""

    def is_fence_line(ln: str) -> bool:
        return ln.strip().startswith("```")

    def is_pure_closing_fence(ln: str) -> bool:
        s = ln.strip()
        if not s.startswith("```"):
            return False
        tag = s[3:].strip()
        return tag == ""

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]

        # If this is the start of a new chunk and we need to reopen a code block, do so
        if not buf and reopen_prefix:
            # If the next line is itself a fence, avoid injecting a duplicate reopen
            if is_fence_line(ln):
                # If it's a pure closing fence, skip it entirely (we already closed the previous chunk)
                if is_pure_closing_fence(ln):
                    i += 1
                    inside = False
                    lang = None
                    # proceed without injecting; keep reopen_prefix for potential next line
                    continue
                else:
                    # It's an opening fence with a language; prefer the original
                    reopen_prefix = ""
            else:
                buf.append(reopen_prefix)
                buf_len += len(reopen_prefix)
                reopen_prefix = ""

        # If adding this line would exceed the limit, flush the current chunk first
        if buf and (buf_len + len(ln) > max_chars):
            if inside:
                # Close code block for this chunk and remember to reopen next
                buf.append("```\n")
                buf_len += 4
                reopen_prefix = f"```{lang or ''}\n"
            else:
                reopen_prefix = ""
            parts.append("".join(buf).rstrip())
            buf, buf_len = [], 0
            # Re-check at top of loop without consuming ln yet
            continue

        # Update fence tracking based on this line
        if is_fence_line(ln):
            s = ln.strip()
            tag = s[3:].strip()
            if not inside:
                inside = True
                lang = tag or None
            else:
                inside = False
                lang = None

        buf.append(ln)
        buf_len += len(ln)
        i += 1

    if buf:
        # If the final chunk ends inside code and doesn't already end with a closing fence, close it
        if inside and not (buf and is_pure_closing_fence(buf[-1])):
            if buf and not buf[-1].endswith("\n"):
                buf.append("\n")
            buf.append("```")
        parts.append("".join(buf).rstrip())

    return parts

def _humanize_html(text: str) -> str:
    """
    Convert any embedded HTML (```html ... ```, or raw <tags>) to human-readable plain text.
    """
    if not text:
        return text

    # Detect ```html blocks
    html_blocks = re.findall(r"```html(.*?)```", text, re.DOTALL)
    for block in html_blocks:
        readable = _strip_html_to_text(block)
        text = text.replace(f"```html{block}```", readable)

    # If there are remaining raw tags outside code fences, sanitize them too
    if "<" in text and ">" in text:
        text = _strip_html_to_text(text)

    return text.strip()

def _strip_html_to_text(fragment: str) -> str:
    """
    Strip HTML tags, decode entities, preserve headings/lists as readable text.
    """
    try:
        soup = BeautifulSoup(fragment, "html.parser")

        # Replace <br> with newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Headings → uppercase section titles
        for h in soup.find_all(["h1","h2","h3","h4","h5","h6","div"]):
            if "heading" in (h.get("class") or []):
                h.insert_before("\n" + h.get_text(strip=True).upper() + "\n")
                h.decompose()

        # List items → table-like numbered lines
        for ul in soup.find_all("ul"):
            items = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
            if items:
                ul.replace_with("\n# | ITEM\n--|-----\n" + "\n".join(f"{i+1} | {itm}" for i, itm in enumerate(items)))

        plain = soup.get_text("\n", strip=True)
        return html.unescape(plain)
    except Exception as e:
        print("[HTML sanitize error]", e)
        return fragment

# ---------------------------------------------------------------------------
# Slack app
# ---------------------------------------------------------------------------
#app = App(token=SLACK_BOT_TOKEN)
app = AsyncApp(token=SLACK_BOT_TOKEN)

# MCP clients (persistent stdio)
RUNINTEG_CLIENT = PersistentMCPClient(RUNINTEG_CMD, env=_skill_env(RUNINTEG_SK))
OEDA_CLIENT     = PersistentMCPClient(OEDA_CMD, env=_skill_env(OEDA_SK))
RAG_CLIENT      = PersistentMCPClient(RAG_CMD, env=_skill_env(RAG_SK))
SUM_CLIENT      = PersistentMCPClient(SUM_CMD, env=_skill_env(SUM_SK))
GENAI4TEST_CLIENT = PersistentMCPClient(GENAI4TEST_CMD, env=_skill_env(GENAI4TEST_CHAT_SK))
LABELHEALTH_CLIENT = PersistentMCPClient(LABELHEALTH_CMD, env=_skill_env(LABEL_HEALTH_SK))
REALHW_CLIENT    = PersistentMCPClient(REALHW_CMD, env=_skill_env(REALHW_SK))
GENAI4TEST_FUNC_CLIENT = PersistentMCPClient(GENAI4TEST_FUNC_CMD, env=_skill_env(GENAI4TEST_FUNC_SK))
LRGTEST_CLIENT  = PersistentMCPClient(LRGTEST_CMD, env=_skill_env(LRGTEST_SK))
try:
    GENAI4TEST_CHAT = GenAI4TestChatAgent()
except Exception as chat_init_err:
    GENAI4TEST_CHAT = None
    print("[genai4test] chat agent init failed:", chat_init_err)

# LangGraph agent setup
CURRENT_THREAD_ID = contextvars.ContextVar("current_thread_id", default=None)
TOOL_RUN_RESULTS: defaultdict[str, List[dict[str, Any]]] = defaultdict(list)

THREAD_HISTORY: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
def _append_history(thread_id: str, role: str, content: str | None):
    if not thread_id or not content:
        return
    history = THREAD_HISTORY[thread_id]
    history.append({"role": role, "content": content})
    if len(history) > 20:
        del history[:-20]

THREAD_REQUESTER_GUID: dict[str, str] = {}

_GUID_PATTERN = re.compile(r"@([A-Za-z0-9._-]+)")


def _remember_thread_requester(thread_id: str, user_label: str) -> None:
    if not thread_id or not user_label:
        return
    match = _GUID_PATTERN.search(user_label)
    guid = match.group(1) if match else None
    if not guid and user_label.startswith("@"):
        guid = user_label[1:]
    if not guid:
        stripped = user_label.strip()
        if stripped and " " not in stripped and "@" not in stripped:
            guid = stripped
    if guid:
        THREAD_REQUESTER_GUID[thread_id] = guid.strip()


def _current_thread_guid() -> str | None:
    thread_id = CURRENT_THREAD_ID.get()
    if not thread_id:
        return None
    return THREAD_REQUESTER_GUID.get(thread_id)

def _format_json(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


def _clean_tool_args(data: dict[str, Any]) -> dict[str, Any]:
    """Strip None values so MCP payloads stay compact."""
    return {k: v for k, v in data.items() if v is not None}

# Maps thread_id -> list of artifacts (most recent last)
THREAD_ARTIFACTS: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
GENAI4TEST_THREAD_STATE: defaultdict[str, dict[str, Any]] = defaultdict(dict)
def _record_artifact(thread_id: str, filename: str, local_path: str, source: str):
    """
    Remember a local file created/obtained in this thread for later reuse.
    We keep paths to temp files; do not unlink them until done uploading.
    """
    if not (thread_id and local_path and OS.path.exists(local_path)):
        return
    THREAD_ARTIFACTS[thread_id].append({
        "filename": filename,
        "local_path": local_path,
        "source": source,
        "ts": time.time(),
    })

def _latest_artifact(thread_id: str) -> dict | None:
    items = THREAD_ARTIFACTS.get(thread_id) or []
    return items[-1] if items else None


async def _handle_genai4test_result(
    result: dict,
    *,
    thread_key: str,
    thread_ts: str,
    channel_id: str,
    client: AsyncWebClient,
    preferred_file_id: str | None = None,
    merge_files: bool = False,
) -> str | None:
    """Upload GenAI4Test artifacts and refresh regenerate controls."""
    if not isinstance(result, dict):
        return None

    summary = (result.get("summary") or "").strip()
    code_text = (result.get("code") or result.get("sql") or "").strip()
    url = ""
    for key in (
        "absolute_file_url",
        "file_url",
        "zip_file_url",
        "zip_url",
        "zipFileUrl",
    ):
        candidate = result.get(key)
        if isinstance(candidate, str) and candidate.strip():
            url = candidate.strip()
            break
        if isinstance(candidate, dict):
            nested_url = candidate.get("url") or candidate.get("href")
            if isinstance(nested_url, str) and nested_url.strip():
                url = nested_url.strip()
                break
    if not url:
        archive = result.get("archive") or result.get("bundle") or {}
        if isinstance(archive, dict):
            nested = archive.get("zip_url") or archive.get("file_url")
            if isinstance(nested, str) and nested.strip():
                url = nested.strip()
    if not url:
        try:
            print("[genai4test] no file URL in result; keys:", sorted(result.keys()))
        except Exception:
            pass
    if url and not url.lower().startswith(("http://", "https://")):
        base = OS.getenv("GENAI4TEST_BASE_URL") or OS.getenv("GENAI4TEST_CHAT_BASE_URL") or ""
        if base:
            url = urljoin(base.rstrip("/") + "/", url.lstrip("/"))

    state = GENAI4TEST_THREAD_STATE[thread_key]
    state["summary"] = summary
    state["agent_name"] = (
        result.get("agent")
        or result.get("agent_name")
        or state.get("agent_name")
        or OS.getenv("GENAI4TEST_AGENT", "bug_agent_dynamic")
    )
    bug_no = result.get("bug_no") or result.get("bug")
    if bug_no:
        state["bug_no"] = bug_no
        state["mode"] = "bug"
    elif result.get("uploaded_input") or result.get("upload_response"):
        state["mode"] = "functional"
    email_val = result.get("email") or state.get("email") or OS.getenv("GENAI4TEST_EMAIL")
    if email_val:
        state["email"] = email_val
    request_id = (
        result.get("request_id")
        or result.get("req_id")
        or bug_no
        or state.get("request_id")
        or THREAD_REQUESTER_GUID.get(thread_key)
    )
    if request_id:
        state["request_id"] = request_id

    state["channel_id"] = channel_id
    state["thread_ts"] = thread_ts
    state["context_id"] = thread_key
    state["last_result"] = result

    final_text = None

    if code_text:
        fname = _pick_genai4test_filename(result)
        suffix = ".sh" if code_text.startswith("#!") or code_text.startswith("#") else ".sql"
        with TF.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(code_text.encode("utf-8", errors="replace"))
            tmp_path = tmp.name

        comment = f"```text\n{summary}\n```" if summary else "Generated test file"
        await client.files_upload_v2(
            channels=[channel_id],
            thread_ts=thread_ts,
            initial_comment=comment,
            file=tmp_path,
            filename=fname,
            title=fname,
        )
        _record_artifact(thread_ts or str(thread_key), fname, tmp_path, source="genai4test")
        final_text = "GenAI4Test: generated test file(s) attached."

    extracted_files: list[dict[str, str]] = []
    existing_files = state.get("files") or {}
    if url:
        verify_env = OS.getenv("GENAI4TEST_CA_BUNDLE") or (
            OS.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() == "true"
        )
        try:
            content = await asyncio.to_thread(_download_bytes, url, verify_env)
        except Exception as zip_dl_err:
            print("[genai4test] ZIP download failed:", zip_dl_err)
            content = None

        if content:
            with TF.NamedTemporaryFile(delete=False, suffix=".zip") as tmpzip:
                tmpzip.write(content)
                zip_path = tmpzip.name
            state["zip_path"] = zip_path

            try:
                lookup: dict[str, str] = {}
                for fid, info in existing_files.items():
                    zname = info.get("zip_path") or info.get("filename")
                    if zname:
                        lookup[zname] = fid
                    fname = info.get("filename")
                    if fname:
                        lookup[fname] = fid

                new_files: dict[str, dict[str, str]] = {}
                with zipfile.ZipFile(BytesIO(content)) as zf:
                    members = [info for info in zf.infolist() if not info.is_dir()]
                    for info in members:
                        raw = zf.read(info.filename)
                        basename = OS.path.basename(info.filename) or info.filename
                        suffix = Path(basename).suffix or ".txt"
                        with TF.NamedTemporaryFile(delete=False, suffix=suffix) as extracted:
                            extracted.write(raw)
                            extracted_path = extracted.name
                        fid = lookup.get(info.filename) or lookup.get(basename)
                        if not fid and preferred_file_id and len(members) == 1:
                            fid = preferred_file_id
                        if not fid:
                            fid = str(uuid.uuid4())
                        new_files[fid] = {
                            "filename": basename,
                            "zip_path": info.filename,
                            "local_path": extracted_path,
                        }
                        extracted_files.append({"file_id": fid, "filename": basename})
                        _record_artifact(thread_ts or str(thread_key), basename, extracted_path, source="genai4test")
                if merge_files and existing_files:
                    merged = dict(existing_files)
                    merged.update(new_files)
                    state["files"] = merged
                else:
                    state["files"] = new_files
            except Exception as unzip_err:
                print("[genai4test] unzip failed:", unzip_err)

            try:
                await client.files_upload_v2(
                    channels=[channel_id],
                    thread_ts=thread_ts,
                    initial_comment="ZIP package from GenAI4Test:",
                    file=zip_path,
                    filename=OS.path.basename(urlparse(url).path),
                    title="Generated ZIP File",
                )
            except Exception as zip_upload_err:
                print("[genai4test] ZIP upload failed:", zip_upload_err)

            final_text = "GenAI4Test: generated test file(s) attached."

    return final_text


async def _post_genai4test_file_buttons(
    client: AsyncWebClient,
    channel_id: str,
    thread_ts: str,
    thread_key: str,
) -> None:
    state = GENAI4TEST_THREAD_STATE.get(thread_key or "") or {}
    files = state.get("files") or {}
    if not files:
        return

    sections = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*GenAI4Test files ready*\nClick any button below to regenerate a specific file again."}
    }]
    for fid, entry in files.items():
        fname = (entry.get("filename") or "").strip()
        lname = fname.lower()
        non_regen = (
            lname == "readme"
            or lname == "readme.txt"
            or lname.endswith("_sum.txt")
            or lname.endswith("_old.java")
            or lname.endswith(".perl")
        )
        section = {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"`{fname}`" if fname else "`(unnamed file)`"},
        }
        if not non_regen:
            payload = json.dumps({"thread": thread_key, "file_id": fid})
            section["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "Regenerate", "emoji": True},
                "value": payload,
                "action_id": "genai4test_request_regen",
            }
        sections.append(section)
    sections.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": "Click a button and leave feedback in this thread to request a fresh version of one file.",
        }],
    })

    message_kwargs = {
        "channel": channel_id,
        "thread_ts": thread_ts,
        "text": "GenAI4Test generated files",
        "blocks": sections,
    }

    buttons_ts = state.get("buttons_ts")
    if buttons_ts:
        try:
            await client.chat_update(
                channel=channel_id,
                ts=buttons_ts,
                thread_ts=thread_ts,
                text=message_kwargs["text"],
                blocks=message_kwargs["blocks"],
            )
        except Exception:
            res = await client.chat_postMessage(**message_kwargs)
            state["buttons_ts"] = res.get("ts")
        else:
            state["buttons_ts"] = buttons_ts
    else:
        res = await client.chat_postMessage(**message_kwargs)
        state["buttons_ts"] = res.get("ts")


class GenerateOedaArgs(BaseModel):
    request: str = Field(..., description="Full natural-language request describing the desired Exadata configuration.")


@tool("generate_oedaxml", args_schema=GenerateOedaArgs)
async def generate_oedaxml_tool(request: str) -> str:
    "Generate Exadata configuration artifacts (minconfig.json and es.xml). Always pass the full user request."
    payload = {
        "request": request,
        "genoedaxml_path": DEFAULT_GENXML,
        "return_xml": True,
        "force_mock": True,
    }
    res = await asyncio.to_thread(OEDA_CLIENT.call_tool, "generate_oedaxml", payload)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "generate_oedaxml", "result": res})
    if res.get("error"):
        return _format_json({"status": "error", "message": res["error"]})
    summary = {
        "status": "ok",
        "live_migration_check": res.get("live_mig_check"),
        "live_migration_reason": res.get("live_mig_reason"),
        "rack_description": res.get("rack_desc"),
        "minconfig_json": res.get("minconfig_json"),
        "es_xml_path": res.get("es_xml_path"),
        "es_xml_available": bool(res.get("es_xml_b64")),
        "notes": res.get("note") or res.get("generator"),
        "post_steps": "cd oss/test/tsage/sosd && run doimageoeda.sh -xml [your/xml/path] -error_report -skip_ahf -remote -skip_qinq_checks_cell",
    }
    return _format_json(summary)


class RunintegrationStatusArgs(BaseModel):
    rack: str = Field(..., description="Rack identifier such as scaXXXadmYY.")


@tool("runintegration_status", args_schema=RunintegrationStatusArgs)
async def runintegration_status_tool(rack: str) -> str:
    "Check RunIntegration status for a specific rack."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.status, rack)
    return _format_json(res)


@tool("runintegration_idle_envs")
async def runintegration_idle_envs_tool() -> str:
    "List idle RunIntegration environments."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.idle_envs)
    print("[DEBUG idle_envs raw]", type(res), (res[:2] if isinstance(res, list) else res))
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "runintegration_idle_envs", "result": res})
    return _format_json(res)

@tool("runintegration_disabled_envs")
async def runintegration_disabled_envs_tool() -> str:
    "List disabled RunIntegration environments."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.disabled_envs)
    print("[DEBUG idle_envs raw]", type(res), (res[:2] if isinstance(res, list) else res))
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "runintegration_disabled_envs", "result": res})
    return _format_json(res)


class RunintegrationTxnArgs(BaseModel):
    txn: str = Field(..., description="RunIntegration transaction name (e.g. 'schavhan_monthly_se_24_1_18_251105_rc2').")


@tool("runintegration_pending_tests", args_schema=RunintegrationTxnArgs)
async def runintegration_pending_tests_tool(txn: str) -> str:
    "List pending RunIntegration tests for a transaction across enabled environments."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.pending_tests, txn)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "runintegration_pending_tests", "result": res})
    return _format_json(res)


@tool("runintegration_disabled_txn_status", args_schema=RunintegrationTxnArgs)
async def runintegration_disabled_txn_status_tool(txn: str) -> str:
    "Show disabled environments and statuses for a RunIntegration transaction."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.disabled_txn_status, txn)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "runintegration_disabled_txn_status", "result": res})
    return _format_json(res)


class BugTestArgs(BaseModel):
    bug_no: str = Field(..., description="Bug number, e.g. 35123456")
    email: str | None = Field(None, description="Email to use (optional)")
    agent: str | None = Field(None, description="genai4test agent (optional)")

@tool("run_bug_test", args_schema=BugTestArgs)
async def run_bug_test_tool(bug_no: str, email: str | None = None, agent: str | None = None) -> str:
    """
    Generate a shell or sql test for a bug via genai4test.
    Returns a summary with script/code and optional file URL.
    """
    # If this thread already contains detected functional test steps, steer away from bug flow.
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        st = GENAI4TEST_THREAD_STATE.get(thread_id) or {}
        if st.get("func_input_text"):
            # Queue a side-effect to (re)post functional agent buttons after this tool finishes
            TOOL_RUN_RESULTS[thread_id].append({"name": "post_func_buttons"})
            return (
                ":information_source: Detected functional test steps in this thread. "
                "Please choose a functional agent (TSC/Java/Exa RH) from the buttons. "
                "If you don't see them, say 'functional test' to show options again."
            )

    # Guard: only treat as bug flow when the bug number is purely numeric (e.g., 35123456)
    bug_str = (bug_no or "").strip()
    if not re.fullmatch(r"\d{5,}", bug_str):
        if thread_id:
            TOOL_RUN_RESULTS[thread_id].append({"name": "post_func_buttons"})
        return (
            ":x: That doesn't look like a numeric bug ID. If you provided test steps, "
            "I'll handle them as a functional test. Please upload/paste the steps and "
            "select an agent (TSC/Java/Exa RH) when prompted."
        )

    # Default email to the requester's GUID so the backend identifies the user by GUID
    eff_email = (email or _current_thread_guid() or OS.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")).strip()
    args = {"bug_no": bug_str, "email": eff_email}
    if agent:
        args["agent"] = agent
    context_id = CURRENT_THREAD_ID.get()
    if context_id:
        args["context_id"] = context_id

    # offload MCP call to a thread (PersistentMCPClient is sync)
    res = await asyncio.to_thread(GENAI4TEST_CLIENT.call_tool, "run_bug_test", args)

    # keep for later slack rendering / debugging
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "run_bug_test", "result": res})

    # If failed, surface the error details directly so Slack shows them
    if not isinstance(res, dict) or not res.get("ok"):
        err = (isinstance(res, dict) and (res.get("error") or res.get("status"))) or "unknown error"
        body = isinstance(res, dict) and res.get("body")
        details = f":x: genai4test failed for bug {bug_no}: {err}"
        if body:
            details += "\n```" + str(body)[:500] + "```"
        # still return text so the agent can show it verbatim
        return details

    # success path: return compact JSON the agent can read
    return _format_json(res) if isinstance(res, dict) else str(res)


@tool("genai4test_health")
async def genai4test_health_tool() -> str:
    "Check health of the GenAI4Test service via the MCP chat server."
    res = await asyncio.to_thread(GENAI4TEST_CLIENT.call_tool, "health", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "genai4test_health", "result": res})
    # Prefer upstream payload shape when available
    if isinstance(res, dict):
        payload = res.get("result") or res
        return _format_json(payload)
    return _format_json({"status": "unknown", "raw": str(res)})

# --- Functional test tools (for test steps input) ---
class FuncTestArgs(BaseModel):
    user_input: str = Field(..., description="Functional test steps or path to a test step file")
    agent: str | None = Field(None, description="Functional agent: func_tsc_agent | func_java_agent | exa_rh_func_test_agent (defaults to func_tsc_agent)")
    email: str | None = Field(None, description="Email or GUID (defaults to requester GUID)")
    context_id: str | None = Field(None, description="Conversation/thread id (required for chat follow-ups)")


@tool("list_func_test_agents")
async def list_func_test_agents_tool() -> str:
    "List available functional-test agents."
    res = await asyncio.to_thread(GENAI4TEST_FUNC_CLIENT.call_tool, "list_func_test_agents", {})
    return _format_json(res)


@tool("run_func_test", args_schema=FuncTestArgs)
async def run_func_test_tool(user_input: str, agent: str | None = None, email: str | None = None, context_id: str | None = None) -> str:
    """
    Capture functional test steps and prompt the user to choose an agent via Slack buttons.
    Actual execution happens only after the requester clicks one of the buttons.
    """
    cleaned = (user_input or "").strip()
    if not cleaned:
        return ":x: I need the functional test steps or file contents in order to proceed."

    thread_id = CURRENT_THREAD_ID.get()
    state = GENAI4TEST_THREAD_STATE.setdefault(thread_id or context_id or cleaned[:12], {})
    state["func_input_text"] = cleaned
    state.setdefault("func_filename", state.get("func_filename") or "inline.txt")
    state["mode"] = "functional"
    state["context_id"] = (context_id or thread_id or state.get("context_id") or str(uuid.uuid4())).strip()
    state["pending_agent_choice"] = True
    if email:
        state["email"] = email.strip()
    if agent:
        state["suggested_agent"] = agent.strip()

    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "post_func_buttons"})

    return (
        ":information_source: Captured the functional test steps. "
        "Please choose either the TSC, Java, or RH agent from the buttons I've posted in this thread."
    )



class FuncMemArgs(BaseModel):
    pdf_path: str = Field(..., description="Path to a PDF file or attachment to send to func_mem_agent")
    email: str | None = Field(None, description="Email or GUID (defaults to requester GUID)")
    context_id: str | None = Field(None, description="Conversation/thread id (optional)")


@tool("run_func_mem_agent", args_schema=FuncMemArgs)
async def run_func_mem_agent_tool(pdf_path: str, email: str | None = None, context_id: str | None = None) -> str:
    """Generate a test plan from a PDF using GenAI4Test func_mem_agent."""
    eff_email = (email or _current_thread_guid() or OS.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")).strip()
    thread_id = CURRENT_THREAD_ID.get()
    ctx = (context_id or thread_id or str(uuid.uuid4())).strip()
    args = {"pdf_path": pdf_path, "email": eff_email, "context_id": ctx}
    res = await asyncio.to_thread(GENAI4TEST_FUNC_CLIENT.call_tool, "run_func_mem_agent", args)
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "run_func_mem_agent", "result": res})
    if not isinstance(res, dict) or not res.get("ok"):
        err = (isinstance(res, dict) and (res.get("error") or res.get("status"))) or "unknown error"
        body = isinstance(res, dict) and res.get("body")
        details = f":x: run_func_mem_agent failed: {err}"
        if body:
            details += "\n```text\n" + str(body)[:500] + "\n```"
        return details
    return _format_json(res)

class LabelSeriesArgs(BaseModel):
    series: str = Field(..., description="Name of the series. The series can be 'OSS_MAIN','OSS_25.2' etc..")
    n: int | None = Field(None, description="Number of labels")

@tool("get_labels_from_series", args_schema=LabelSeriesArgs)
async def get_labels_from_series(series: str, n: int = 10) -> dict:
    """
    Get n recent labels from the given series. The series can be "OSS_MAIN", "OSS_25.2" etc..
    """
    args = {"series": series}
    if n: args["n"] = n

    # offload MCP call to a thread (PersistentMCPClient is sync)
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_labels_from_series", args)
    print("res",res)
    print("args",args)

    # keep for later slack rendering / debugging
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_labels_from_series", "result": res})
        

    # ✅ Only treat as failure if the server returned an 'error'
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        details = f":x: label health app failed for series `{series}`: {err}"
        return details

    # success path: return compact JSON text so the agent can echo nicely
    return _format_json(res)


class MyLrgsStatusArgs(BaseModel):
    identifier: str | None = Field(
        None,
        description="User GUID or email used to fetch LRG assignments (defaults to requester)",
    )
    label: str | None = Field(None, description="Specific label to inspect (defaults to most recent)")
    lrgs: str | None = Field(None, description="Optional comma-separated LRG override")
    series: str | None = Field(None, description="Series name to resolve latest label (defaults to OSS_MAIN)")


@tool("get_my_lrgs_status", args_schema=MyLrgsStatusArgs)
async def get_my_lrgs_status_tool(
    identifier: str | None = None,
    label: str | None = None,
    lrgs: str | None = None,
    series: str | None = None,
) -> str:
    """Fetch the current dif status for a user's LRGs."""

    eff_identifier = (identifier or "").strip()
    if not eff_identifier:
        eff_identifier = _current_thread_guid()
    if not eff_identifier:
        return ":x: I need a GUID or email to look up LRGs. Please provide one."

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        THREAD_REQUESTER_GUID[thread_id] = eff_identifier

    args = {"identifier": eff_identifier}
    if label:
        args["label"] = label.strip()
    if lrgs:
        args["lrgs"] = lrgs.strip()
    if series:
        args["series"] = series.strip()

    logger.info(f"Calling get_my_lrgs_status with args: {args}")

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "get_my_lrgs_status",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_my_lrgs_status", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to resolve LRGs for `{eff_identifier}`: {err}"

    return _format_json(res)


class AddLrgToMyLrgsArgs(BaseModel):
    identifier: str | None = Field(
        None,
        description="User GUID or email (defaults to requester GUID)",
    )
    lrg: str = Field(..., description="LRG name to add to the watchlist")


@tool("add_lrg_to_my_lrgs", args_schema=AddLrgToMyLrgsArgs)
async def add_lrg_to_my_lrgs_tool(lrg: str, identifier: str | None = None) -> str:
    """Add an LRG to a user's "My LRGs" list in Label Health."""

    eff_identifier = (identifier or "").strip()
    if not eff_identifier:
        eff_identifier = _current_thread_guid()
    if not eff_identifier:
        return ":x: I need a GUID or email to add an LRG."

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        THREAD_REQUESTER_GUID[thread_id] = eff_identifier

    args = {"identifier": eff_identifier, "lrg": lrg.strip()}

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "add_lrg_to_my_lrgs",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "add_lrg_to_my_lrgs", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: Failed to add LRG `{lrg}` for `{eff_identifier}`: {err}"

    return _format_json(res)


class MyRtisArgs(BaseModel):
    guid: str | None = Field(
        None,
        description="User GUID whose RTIs should be fetched (defaults to requester GUID)",
    )
    n: int | None = Field(5, description="Number of RTIs to return (default 5)")
    series: str | None = Field("OSS_MAIN", description="Series to filter by (default OSS_MAIN)")


@tool("get_my_rtis", args_schema=MyRtisArgs)
async def get_my_rtis_tool(guid: str | None = None, n: int | None = 5, series: str | None = "OSS_MAIN") -> str:
    """Fetch recent RTIs assigned to the requester (defaults to thread GUID)."""

    eff_guid = (guid or "").strip() or _current_thread_guid()
    if not eff_guid:
        return ":x: I need a GUID to look up your RTIs. Please provide one."

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        THREAD_REQUESTER_GUID[thread_id] = eff_guid

    args = {"guid": eff_guid}
    if n is not None:
        try:
            n_int = int(n)
            if n_int > 0:
                args["n"] = n_int
        except (TypeError, ValueError):
            return ":x: `n` must be a positive integer."
    if series:
        args["series"] = series.strip()

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "get_my_rtis",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_my_rtis", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to fetch RTIs for `{eff_guid}`: {err}"

    return _format_json(res)


class LrgsFromRegressArgs(BaseModel):
    regress: str = Field(..., description="Regress name, e.g. 'SAGE_FC' or 'EXAC_REGRESS'")

@tool("get_lrgs_from_regress", args_schema=LrgsFromRegressArgs)
async def get_lrgs_from_regress_tool(regress: str) -> str:
    """
    Return LRGs associated with the given regress. Example regress: 'SAGE_FC', 'EXAC_REGRESS'.
    """
    regress = (regress or "").strip()
    if not regress:
        return ":x: Please provide a non-empty regress name."

    # offload MCP call to a thread (PersistentMCPClient is sync)
    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "get_lrgs_from_regress",
        {"regress": regress},
    )

    # keep for Slack enrichment / debugging
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_lrgs_from_regress", "result": res})

    # Treat only explicit 'error' as failure
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for regress `{regress}`: {err}"

    # success: return compact JSON so the agent can echo nicely
    return _format_json(res)

class LrgWithDifsArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrgs: str | None = Field(None, description="Optional comma-separated LRGs to filter")
    regress: str | None = Field(None, description="Optional regress filter")

@tool("find_lrg_with_difs", args_schema=LrgWithDifsArgs)
async def find_lrg_with_difs_tool(label: str, lrgs: str | None = None, regress: str | None = None) -> str:
    """
    List LRGs with difs/failures for a label (optionally filtered by specific LRGs).
    """
    args = {"label": label}
    if lrgs: args["lrgs"] = lrgs
    if regress: args["regress"] = regress
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_lrg_with_difs", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_lrg_with_difs", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for label `{label}`: {err}"

    return _format_json(res)

class DifDetailsArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrgs: str | None = Field(None, description="Comma-separated LRGs to filter")
    name: str | None = Field(None, description="Filter: dif name")
    status: str | None = Field(None, description="Filter: status")
    text: str | None = Field(None, description="Filter: text/description")
    rti_number: str | None = Field(None, description="Filter: RTI number")
    rti_assigned_to: str | None = Field(None, description="Filter: RTI assignee")
    rti_status: str | None = Field(None, description="Filter: RTI status")
    comments: str | None = Field(None, description="Filter: comment content")
    regress: str | None = Field(None, description="Filter: regress name")

@tool("find_dif_details", args_schema=DifDetailsArgs)
async def find_dif_details_tool(
    label: str,
    lrgs: str | None = None,
    name: str | None = None,
    status: str | None = None,
    text: str | None = None,
    rti_number: str | None = None,
    rti_assigned_to: str | None = None,
    rti_status: str | None = None,
    comments: str | None = None,
    regress: str | None = None,
) -> str:
    """
    Detailed dif/failure info for a label with optional filters.
    """
    args = {
        "label": label,
        **({} if lrgs is None else {"lrgs": lrgs}),
        **({} if name is None else {"name": name}),
        **({} if status is None else {"status": status}),
        **({} if text is None else {"text": text}),
        **({} if rti_number is None else {"rti_number": rti_number}),
        **({} if rti_assigned_to is None else {"rti_assigned_to": rti_assigned_to}),
        **({} if rti_status is None else {"rti_status": rti_status}),
        **({} if comments is None else {"comments": comments}),
        **({} if regress is None else {"regress": regress}),
    }
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_dif_details", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_dif_details", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for label `{label}`: {err}"

    return _format_json(res)

class DifOccurrenceArgs(BaseModel):
    dif: str = Field(..., description="Dif name to search for")
    series: str = Field(..., description="Series, e.g. 'OSS_MAIN' or 'OSS_25.1'")

@tool("find_dif_occurrence", args_schema=DifOccurrenceArgs)
async def find_dif_occurrence_tool(dif: str, series: str) -> str:
    """
    Find occurrences of a dif in a given series.
    """
    args = {"dif": dif, "series": series}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_dif_occurrence", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_dif_occurrence", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for dif `{dif}` in series `{series}`: {err}"

    return _format_json(res)

class WidespreadArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    n: int = Field(3, description="Minimum occurrences to consider widespread (default: 3)")

@tool("find_widespread_issues", args_schema=WidespreadArgs)
async def find_widespread_issues_tool(label: str, n: int = 3) -> str:
    """
    List widespread issues (dif name + lrgs) for a label.
    """
    args = {"label": label, "n": n}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_widespread_issues", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_widespread_issues", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for label `{label}`: {err}"

    return _format_json(res)

class FindCrashesArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrgs: str | None = Field(None, description="Optional comma-separated LRGs filter (e.g. 'lrg1,lrg2')")
    lrg: str | None = Field(None, description="Optional single LRG filter (alternative to lrgs)")
    regress: str | None = Field(None, description="Optional regress filter")

@tool("find_crashes", args_schema=FindCrashesArgs)
async def find_crashes_tool(label: str, lrgs: str | None = None, lrg: str | None = None, regress: str | None = None) -> str:
    """
    Get crash information for a specific label from the Label Health service.
    """
    args = {"label": label}
    if lrgs:   args["lrgs"] = lrgs
    if lrg:    args["lrg"] = lrg
    if regress: args["regress"] = regress
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_crashes", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_crashes", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get crashes for `{label}`: {err}"

    return _format_json(res)

class LrgHistoryArgs(BaseModel):
    lrg: str = Field(..., description="LRG identifier, e.g. 'lrgrhexaprovcluster'")
    series: str | None = Field(None, description="Optional series filter, e.g. 'OSS_MAIN'")
    n: int | None = Field(10, description="Number of history labels to return (default: 20)")

@tool("get_lrg_history", args_schema=LrgHistoryArgs)
async def get_lrg_history_tool(lrg: str, series: str | None = None, n: int | None = 20) -> str:
    """
    Get LRG history for a given LRG, optionally filtered by series and number of labels.
    """
    args = {"lrg": lrg}
    if series:
        args["series"] = series
    if n is not None:
        args["n"] = n

    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_lrg_history", args)

    # record for debugging / optional enrichments
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_lrg_history", "result": res})

    # error handling
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get LRG history for `{lrg}`: {err}"

    # success: return compact JSON the agent can render (prompt tables)
    return _format_json(res)

class LrgPointOfContactArgs(BaseModel):
    lrg: str = Field(..., description="LRG identifier, e.g. 'lrgrhexaprovcluster_livemig'")


@tool("lrg_point_of_contact", args_schema=LrgPointOfContactArgs)
async def lrg_point_of_contact_tool(lrg: str) -> str:
    """Return owner and backup owner details for an LRG."""
    lrg = (lrg or "").strip()
    if not lrg:
        return ":x: Please provide a valid LRG name."

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "lrg_point_of_contact",
        {"lrg": lrg},
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_point_of_contact", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get contacts for `{lrg}`: {err}"

    return _format_json(res)


class GetIncompleteLrgsArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")


@tool("get_incomplete_lrgs", args_schema=GetIncompleteLrgsArgs)
async def get_incomplete_lrgs_tool(label: str) -> str:
    """List LRGs that are still running for a label."""
    label = (label or "").strip()
    if not label:
        return ":x: Please provide a label."

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "get_incomplete_lrgs",
        {"label": label},
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_incomplete_lrgs", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to fetch incomplete LRGs for `{label}`: {err}"

    return _format_json(res)


class DraftEmailForLrgArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrg: str = Field(..., description="LRG identifier")


@tool("draft_email_for_lrg", args_schema=DraftEmailForLrgArgs)
async def draft_email_for_lrg_tool(label: str, lrg: str) -> str:
    """Draft an email to an LRG owner summarizing diffs in a label."""
    args = {"label": (label or "").strip(), "lrg": (lrg or "").strip()}
    if not args["label"] or not args["lrg"]:
        return ":x: Both `label` and `lrg` are required to draft an email."

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "draft_email_for_lrg",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "draft_email_for_lrg", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to draft an email for `{args['lrg']}`: {err}"

    return _format_json(res)

# --- query_ai_crash_summary -------------------------------------------------
class AiCrashSummaryArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrg: str = Field(..., description="LRG identifier")
    dif_name: str = Field(..., description="Dif name")

@tool("query_ai_crash_summary", args_schema=AiCrashSummaryArgs)
async def query_ai_crash_summary_tool(label: str, lrg: str, dif_name: str) -> str:
    """
    Get AI-generated crash summary for a specific crash (label + LRG + dif).
    """
    args = {"label": label, "lrg": lrg, "dif_name": dif_name}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "query_ai_crash_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "query_ai_crash_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get AI crash summary for `{label}` / `{lrg}` / `{dif_name}`: {err}"

    return _format_json(res)

# --- get_se_rerun_details ---------------------------------------------------
class SeRerunArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    se_job_id: str | None = Field(None, description="Optional SE job id (7–9 digits)")

@tool("get_se_rerun_details", args_schema=SeRerunArgs)
async def get_se_rerun_details_tool(label: str, se_job_id: str | None = None) -> str:
    """
    Get SE rerun details for a label (or a specific SE job id if provided).
    """
    args = {"label": label}
    if se_job_id:
        args["se_job_id"] = se_job_id

    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_se_rerun_details", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_se_rerun_details", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        target = f"`{label}` (SE job `{se_job_id}`)" if se_job_id else f"`{label}`"
        return f":x: label health app failed to get SE rerun details for {target}: {err}"

    return _format_json(res)

# --- get_regress_summary ----------------------------------------------------
class RegressSummaryArgs(BaseModel):
    regress: str = Field(..., description="Regress name, e.g. 'SAGE_FC'")
    series: str = Field(..., description="Series, e.g. 'OSS_MAIN'")

@tool("get_regress_summary", args_schema=RegressSummaryArgs)
async def get_regress_summary_tool(regress: str, series: str) -> str:
    """
    Get regress summary for a regress/series pair.
    """
    args = {"regress": regress, "series": series}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_regress_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_regress_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get regress summary for `{regress}` in `{series}`: {err}"

    return _format_json(res)

# --- get_label_info ---------------------------------------------------------
class LabelInfoArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")

@tool("get_label_info", args_schema=LabelInfoArgs)
async def get_label_info_tool(label: str) -> str:
    """
    Get detailed information about a specific label.
    """
    args = {"label": label}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_label_info", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_label_info", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get label info for `{label}`: {err}"

    return _format_json(res)

# --- get_ai_label_summary ---------------------------------------------------
class AiLabelSummaryArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")

@tool("get_ai_label_summary", args_schema=AiLabelSummaryArgs)
async def get_ai_label_summary_tool(label: str) -> str:
    """
    Get AI-generated summary for a label (if previously created).
    """
    args = {"label": label}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_ai_label_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_ai_label_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get AI label summary for `{label}`: {err}"

    return _format_json(res)

# --- generate_ai_label_summary ----------------------------------------------
class GenerateAiLabelSummaryArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")

@tool("generate_ai_label_summary", args_schema=GenerateAiLabelSummaryArgs)
async def generate_ai_label_summary_tool(label: str) -> str:
    """
    Generate an AI label summary for the given label (long-running call).
    """
    args = {"label": label}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "generate_ai_label_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "generate_ai_label_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to generate AI label summary for `{label}`: {err}"

    return _format_json(res)

# --- get_delta_diffs_between_labels -----------------------------------------
class DeltaDiffsArgs(BaseModel):
    label_1: str = Field(..., description="Source label")
    compare_labels: str = Field(..., description="Comma-separated list of labels to compare against")
    show_common: str = Field(..., description="Whether to show common diffs ('Y' or 'N')")

@tool("get_delta_diffs_between_labels", args_schema=DeltaDiffsArgs)
async def get_delta_diffs_between_labels_tool(label_1: str, compare_labels: str, show_common: str) -> str:
    """
    Get delta diffs between labels (source vs. compare set).
    """
    args = {"label_1": label_1, "compare_labels": compare_labels, "show_common": show_common}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_delta_diffs_between_labels", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_delta_diffs_between_labels", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get delta diffs for `{label_1}`: {err}"

    return _format_json(res)


class AddSeriesToAutoPopulateArgs(BaseModel):
    series: str = Field(..., description="Series name (single or comma-separated list)")
    guid: str | None = Field(None, description="User GUID (optional; defaults to requester)")


@tool("add_series_to_auto_populate", args_schema=AddSeriesToAutoPopulateArgs)
async def add_series_to_auto_populate_tool(series: str, guid: str | None = None) -> str:
    """Add one or more series to the auto-populate list (mutating POST)."""
    series_clean = (series or "").strip()
    eff_guid = (guid or "").strip() or _current_thread_guid() or ""
    if not series_clean:
        return ":x: Please provide a non-empty `series`."
    if not eff_guid:
        return ":x: Unable to determine requester GUID. Please mention me in this thread first or provide a GUID explicitly."
    args = {"series": series_clean, "guid": eff_guid}

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "add_series_to_auto_populate",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "add_series_to_auto_populate", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to add series `{args['series']}`: {err}"

    return _format_json(res)


class AddLabelForSeAnalysisArgs(BaseModel):
    identifier: str = Field(..., description="SE job/input ID (numeric) or label (e.g., 'OSS_25.1.11.0.0_LINUX.X64_251112')")
    guid: str | None = Field(None, description="User GUID (optional; defaults to requester)")


@tool("add_label_for_se_analysis", args_schema=AddLabelForSeAnalysisArgs)
async def add_label_for_se_analysis_tool(identifier: str, guid: str | None = None) -> str:
    """Add a label or ID for SE analysis (mutating POST)."""
    identifier_clean = (identifier or "").strip()
    eff_guid = (guid or "").strip() or _current_thread_guid() or ""
    if not identifier_clean:
        return ":x: Please provide a non-empty `identifier`."
    if not eff_guid:
        return ":x: Unable to determine requester GUID. Please mention me in this thread first or provide a GUID explicitly."
    args = {"identifier": identifier_clean, "guid": eff_guid}

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "add_label_for_se_analysis",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "add_label_for_se_analysis", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to add SE analysis identifier `{args['identifier']}`: {err}"

    view_id = res.get("input_id") or identifier_clean
    note = ""
    if view_id:
        note = (
            "\n\nTo view the analysis on the portal when it's complete, visit "
            f"https://apex.oraclecorp.com/pls/apex/r/lrg_times/oss-label-health/se-label-analysis1?p34_id={view_id}"
        )

    return _format_json(res) + note


class AddLabelForAnalysisArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    guid: str | None = Field(None, description="User GUID (optional; defaults to requester)")


@tool("add_label_for_analysis", args_schema=AddLabelForAnalysisArgs)
async def add_label_for_analysis_tool(label: str, guid: str | None = None) -> str:
    """Add a label to the analysis queue (mutating POST)."""
    label_clean = (label or "").strip()
    eff_guid = (guid or "").strip() or _current_thread_guid() or ""
    if not label_clean:
        return ":x: Please provide a non-empty `label`."
    if not eff_guid:
        return ":x: Unable to determine requester GUID. Please mention me in this thread first or provide a GUID explicitly."
    args = {"label": label_clean, "guid": eff_guid}

    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "add_label_for_analysis",
        args,
    )

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "add_label_for_analysis", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to add label `{args['label']}`: {err}"

    return _format_json(res)



class RagQueryArgs(BaseModel):
    question: str = Field(..., description="Question to answer using the Exadata knowledge base.")
    k: int = Field(3, description="Number of supporting documents to retrieve (default 3).")


@tool("rag_query", args_schema=RagQueryArgs)
async def rag_query_tool(question: str, k: int = 3) -> str:
    "Retrieve grounded answers about Exadata and Oracle topics."
    res = await asyncio.to_thread(RAG_CLIENT.call_tool, "rag_query", {"question": question, "k": k})

    if isinstance(res, dict):
        answer = res.get("answer") or res.get("result") or ""
        sources = []
        for src in res.get("sources") or []:
            sources.append({
                "title": (src.get("title") or "").strip() or "untitled",
                "resource": src.get("source"),
                "score": src.get("score"),
                "chunk_preview": src.get("chunk_preview"),
            })
        res = {"answer": answer, "sources": sources}

    return _format_json(res)


class SummarizeTextArgs(BaseModel):
    text: str = Field(..., description="Text that should be summarized.")


@tool("summarize_text", args_schema=SummarizeTextArgs)
async def summarize_text_tool(text: str) -> str:
    "Summarize a block of text."
    res = await asyncio.to_thread(SUM_CLIENT.call_tool, "lc_summarize_text", {"text": text})
    return _format_json(res)

# ---------------------------------------------------------------------------
# LRG Test Tools (via lrg_test_mcp_server)
# ---------------------------------------------------------------------------

class LrgPlanArgs(BaseModel):
    plan: Dict[str, Any] | None = Field(None, description="Optional QueryPayload override.")
    k: int = Field(10, description="Max rows to request.")


def _format_lrg_result(tool_name: str, res: Dict[str, Any] | str) -> str:
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": tool_name, "result": res})
    if isinstance(res, dict):
        answer = res.get("answer")
        formatted = json.dumps(res, indent=2, ensure_ascii=False)
        if answer:
            return f"{answer}\n\n```json\n{formatted}\n```"
        return f"```json\n{formatted}\n```"
    return _format_json(res)


class LrgArg(BaseModel):
    lrg: str = Field(..., description="LRG identifier")


async def _smart_search_call(question: str, plan: Dict[str, Any], k: int, tool_name: str) -> str:
    payload = {"question": question, "plan": plan, "k": k}
    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "smart_search", payload)
    return _format_lrg_result(tool_name, res)


@tool("list_tests_for_lrg", args_schema=LrgArg)
async def list_tests_for_lrg_tool(lrg: str) -> str:
    """Return the tests mapped to a specific LRG."""
    plan = {
        "mode": "search",
        "text": "",
        "filters": {
            "suite": [],
            "lrg": [lrg],
            "setup": [],
            "flag": [],
            "doc_type": ["TEST"],
        },
        "ops": {
            "limit": 20,
            "offset": 0,
            "group_by": "none",
            "sort": [{"by": "score", "dir": "desc"}],
        },
    }
    return await _smart_search_call(f"tests for {lrg}", plan, 20, "list_tests_for_lrg")


@tool("get_suites_for_lrg", args_schema=LrgArg)
async def get_suites_for_lrg_tool(lrg: str) -> str:
    """Show suite metadata for an LRG."""
    plan = {
        "mode": "search",
        "text": "",
        "filters": {
            "suite": [],
            "lrg": [lrg],
            "setup": [],
            "flag": [],
            "doc_type": ["LRG"],
        },
        "ops": {
            "limit": 5,
            "offset": 0,
            "group_by": "none",
            "sort": [{"by": "score", "dir": "desc"}],
        },
    }
    return await _smart_search_call(f"suites for {lrg}", plan, 5, "get_suites_for_lrg")


@tool("get_lrg_runtime_stats", args_schema=LrgArg)
async def get_lrg_runtime_stats_tool(lrg: str) -> str:
    """Summarize recent runtime stats for an LRG."""
    plan = {
        "mode": "runtime_spike",
        "text": f"runtime stats {lrg}",
        "filters": {
            "suite": [],
            "lrg": [lrg],
            "setup": [],
            "flag": [],
            "doc_type": ["LRG"],
        },
        "ops": {
            "limit": 10,
            "offset": 0,
            "group_by": "none",
            "sort": [{"by": "runtime", "dir": "asc"}],
        },
    }
    return await _smart_search_call(f"runtime stats for {lrg}", plan, 10, "get_lrg_runtime_stats")


class TestArg(BaseModel):
    test: str = Field(..., description="Test name or id")


@tool("get_test_details", args_schema=TestArg)
async def get_test_details_tool(test: str) -> str:
    """Fetch test metadata, flags, and LRG context."""
    plan = {
        "mode": "search",
        "text": test,
        "filters": {
            "suite": [],
            "lrg": [],
            "setup": [],
            "flag": [],
            "doc_type": ["TEST"],
        },
        "ops": {
            "limit": 5,
            "offset": 0,
            "group_by": "none",
            "sort": [{"by": "score", "dir": "desc"}],
        },
    }
    return await _smart_search_call(f"details for {test}", plan, 5, "get_test_details")


class SearchTestsArg(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(20, description="Max results")


@tool("search_tests", args_schema=SearchTestsArg)
async def search_tests_tool(query: str, k: int = 20) -> str:
    """Free-form test search via the router pipeline."""
    plan = {
        "mode": "search",
        "text": query,
        "filters": {
            "suite": [],
            "lrg": [],
            "setup": [],
            "flag": [],
            "doc_type": ["TEST"],
        },
        "ops": {
            "limit": k,
            "offset": 0,
            "group_by": "none",
            "sort": [{"by": "score", "dir": "desc"}],
        },
    }
    return await _smart_search_call(query, plan, k, "search_tests")


@tool("lrg_test_health")
async def lrg_test_health_tool() -> str:
    """Report availability/staleness of LRG mapping and index files."""
    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "lrg_test_health", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_test_health", "result": res})
    return _format_json(res)


class LrgRagQueryArgs(BaseModel):
    q: str = Field(..., description="Natural-language description of the desired tests.")
    k: int = Field(10, description="Number of tests to return.")
    index: str | None = Field(None, description="Override FAISS index path.")
    meta: str | None = Field(None, description="Override metadata jsonl path.")
    t2l: str | None = Field(None, description="Optional test→LRG mapping json path.")
    lrgs_json: str | None = Field(None, description="Optional LRG metadata json path.")
    model: str | None = Field(None, description="Embedding model id or snapshot path.")
    pool_mult: int = Field(8, description="Pool multiplier before RRF rerank.")
    rrf_k: float = Field(60.0, description="Reciprocal Rank Fusion smoothing constant.")
    timeout_sec: int = Field(900, description="Subprocess timeout in seconds.")


@tool("lrg_test_rag_query", args_schema=LrgRagQueryArgs)
async def lrg_test_rag_query_tool(
    q: str,
    k: int = 10,
    index: str | None = None,
    meta: str | None = None,
    t2l: str | None = None,
    lrgs_json: str | None = None,
    model: str | None = None,
    pool_mult: int = 8,
    rrf_k: float = 60.0,
    timeout_sec: int = 900,
) -> str:
    """Semantic search across test descriptions (FAISS + MiniLM + RRF)."""
    args = _clean_tool_args({
        "q": q,
        "k": k,
        "index": index,
        "meta": meta,
        "t2l": t2l,
        "lrgs_json": lrgs_json,
        "model": model,
        "pool_mult": pool_mult,
        "rrf_k": rrf_k,
        "timeout_sec": timeout_sec,
    })
    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "rag_query_tests", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_test_rag_query", "result": res})
    return _format_json(res)


class LrgFtsQueryArgs(BaseModel):
    q: str = Field(..., description="Natural-language or structured query text.")
    k: int = Field(10, description="Number of documents to return.")
    db: str | None = Field(None, description="Override path to the FTS SQLite DB.")
    t2l: str | None = Field(None, description="Optional test→LRG mapping json path.")
    l2t: str | None = Field(None, description="Optional LRG→test mapping json path.")
    lrgs_json: str | None = Field(None, description="Optional LRG metadata json path.")
    doc_type: Literal["TEST", "LRG"] | None = Field(None, description="Restrict hits to TEST or LRG.")
    require_setup: str | None = Field(None, description="Exact setup filter.")
    require_flag: List[str] | None = Field(None, description="List of required flag expressions key=value.")
    prefer_flag: List[str] | None = Field(None, description="Flag expressions to boost.")
    require_suite: List[str] | None = Field(None, description="Suites to require (e.g., perf).")
    prefer_suite: List[str] | None = Field(None, description="Suites to boost.")
    prefer_setup: str | None = Field(None, description="Setup to soft-prefer.")
    pool: int = Field(0, description="Candidate expansion pool before rerank.")
    structured_only: bool = Field(False, description="Skip FTS scoring; filter via metadata only.")
    no_expand: bool = Field(False, description="Avoid TEST↔LRG expansion.")
    no_flag_values: bool = Field(False, description="Hide flag values in results.")
    show_desc: bool = Field(False, description="Include test descriptions in stdout.")
    desc_chars: int = Field(800, description="Description character limit if enabled.")
    only_direct: bool = Field(False, description="Show only direct TEST↔LRG matches.")
    k_lrgs: int = Field(50, description="Max LRG rows when doc_type includes LRGs.")
    lrg_order: Literal["count", "runtime", "name"] = Field("count", description="LRG sort order.")
    timeout_sec: int = Field(900, description="Subprocess timeout in seconds.")


@tool("lrg_test_fts_query", args_schema=LrgFtsQueryArgs)
async def lrg_test_fts_query_tool(
    q: str,
    k: int = 10,
    db: str | None = None,
    t2l: str | None = None,
    l2t: str | None = None,
    lrgs_json: str | None = None,
    doc_type: Literal["TEST", "LRG"] | None = None,
    require_setup: str | None = None,
    require_flag: List[str] | None = None,
    prefer_flag: List[str] | None = None,
    require_suite: List[str] | None = None,
    prefer_suite: List[str] | None = None,
    prefer_setup: str | None = None,
    pool: int = 0,
    structured_only: bool = False,
    no_expand: bool = False,
    no_flag_values: bool = False,
    show_desc: bool = False,
    desc_chars: int = 800,
    only_direct: bool = False,
    k_lrgs: int = 50,
    lrg_order: Literal["count", "runtime", "name"] = "count",
    timeout_sec: int = 900,
) -> str:
    """Constraint-heavy search over the SQLite FTS metadata (setups, flags, suites)."""
    args = _clean_tool_args({
        "q": q,
        "k": k,
        "db": db,
        "t2l": t2l,
        "l2t": l2t,
        "lrgs_json": lrgs_json,
        "doc_type": doc_type,
        "require_setup": require_setup,
        "prefer_setup": prefer_setup,
        "pool": pool if pool else None,
        "structured_only": structured_only or None,
        "no_expand": no_expand or None,
        "no_flag_values": no_flag_values or None,
        "show_desc": show_desc or None,
        "desc_chars": desc_chars if show_desc else None,
        "only_direct": only_direct or None,
        "k_lrgs": k_lrgs,
        "lrg_order": lrg_order,
        "timeout_sec": timeout_sec,
    })
    if require_flag:
        args["require_flag"] = require_flag
    if prefer_flag:
        args["prefer_flag"] = prefer_flag
    if require_suite:
        args["require_suite"] = require_suite
    if prefer_suite:
        args["prefer_suite"] = prefer_suite
    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "fts_query", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_test_fts_query", "result": res})
    return _format_json(res)


class LrgSmartSearchArgs(BaseModel):
    question: str = Field(..., description="Raw user request to route and answer.")
    k: int = Field(10, description="Number of rows to request from the backend.")
    plan: Optional[Dict[str, Any]] = Field(None, description="Optional precomputed QueryPayload dict to skip routing.")


@tool("lrg_test_smart_search", args_schema=LrgSmartSearchArgs)
async def lrg_test_smart_search_tool(
    question: str,
    k: int = 10,
    plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Run the router→backend→writer pipeline for LRG/test queries."""
    req: Dict[str, Any] = {"question": question, "k": k}
    if plan is not None:
        req["plan"] = plan

    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "smart_search", req)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_test_smart_search", "result": res})

    if isinstance(res, dict):
        answer = res.get("answer")
        formatted = json.dumps(res, indent=2, ensure_ascii=False)
        if answer:
            return f"{answer}\n\n```json\n{formatted}\n```"
        return f"```json\n{formatted}\n```"

    return _format_json(res)


@tool("lrg_test_health_check")
async def lrg_test_health_check_tool() -> str:
    """Quick health report for indexes, FTS db, and mapping files backing test search."""
    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "health_check", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_test_health_check", "result": res})
    return _format_json(res)


@tool("lrg_test_rag_warmup")
async def lrg_test_rag_warmup_tool() -> str:
    """Prime the semantic search model so first user call is faster."""
    res = await asyncio.to_thread(LRGTEST_CLIENT.call_tool, "rag_warmup", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "lrg_test_rag_warmup", "result": res})
    return _format_json(res)

# ---------------------------------------------------------------------------
# RealHW Tools (via realhw_mcp_server)
# ---------------------------------------------------------------------------

async def _scheduler_from_lrg(lrg: str) -> tuple[str | None, str | None]:
    """Resolve scheduler from an LRG via the RealHW mapper tool."""
    lrg_clean = (lrg or "").strip()
    if not lrg_clean:
        return None, ":x: LRG is required to resolve a scheduler."
    res = await asyncio.to_thread(
        REALHW_CLIENT.call_tool,
        "map_lrg_to_scheduler",
        {"lrg": lrg_clean},
    )
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return None, f":x: Unable to resolve scheduler for LRG `{lrg_clean}`: {err}"
    sched = res.get("scheduler")
    if not sched:
        return None, f":x: Scheduler not found for LRG `{lrg_clean}`."
    return sched, None


class RealHWSchedMapArgs(BaseModel):
    lrg: str = Field(..., description="LRG name to map to its scheduler")

@tool("map_lrg_to_scheduler", args_schema=RealHWSchedMapArgs)
async def map_lrg_to_scheduler_tool(lrg: str) -> str:
    """Return the scheduler that hosts the hardware for a given LRG."""
    args = {"lrg": (lrg or "").strip()}
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "map_lrg_to_scheduler", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "map_lrg_to_scheduler", "result": res})
    return _format_json(res)


class RealHWViewSchedArgs(BaseModel):
    sched: str | None = Field(None, description="Scheduler name, e.g. 'nshqap04', 'x9_se_sched'.")
    lrg: str | None = Field(None, description="LRG to map to a scheduler (optional)")

@tool("view_status_of_sched", args_schema=RealHWViewSchedArgs)
async def view_status_of_sched_tool(sched: str | None = None, lrg: str | None = None) -> str:
    """View the status of a scheduler: pool, in-use, and waitlist."""
    eff_sched = (sched or "").strip()
    if not eff_sched and lrg:
        eff_sched, err = await _scheduler_from_lrg(lrg)
        if err:
            return err
    if not eff_sched:
        return ":x: Provide either `sched` or `lrg`."

    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "view_status_of_sched", {"sched": eff_sched})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "view_status_of_sched", "result": res})
    return _format_json(res)


class RealHWReserveArgs(BaseModel):
    sched: str | None = Field(None, description="Scheduler name (optional; auto-detect from hardware if omitted)")
    lrg: str | None = Field(None, description="LRG to map to a scheduler (optional)")
    guid: str | None = Field(None, description="User GUID (optional; defaults to requester)")
    hardware: str | None = Field(None, description="Comma-separated nodes, e.g. 'hw1,hw2' (optional when using LRG). Leave blank to auto-pick when using LRG.")
    comment: str = Field(..., description="Reservation comment")

@tool("reserve_hardware", args_schema=RealHWReserveArgs)
async def reserve_hardware_tool(sched: str | None = None, lrg: str | None = None, guid: str | None = None, hardware: str | None = None, comment: str = "") -> str:
    """Reserve hardware nodes on a scheduler (or via LRG) for the requester GUID with a comment."""
    eff_guid = _current_thread_guid() or (guid or "").strip()
    if not eff_guid:
        return ":x: Unable to determine requester GUID. Please mention me in this thread first or provide a GUID explicitly."
    eff_comment = (comment or "").strip()
    if not eff_comment:
        return ":x: Please provide a reservation comment."
    eff_sched = (sched or "").strip()
    if not eff_sched and lrg:
        eff_sched, err = await _scheduler_from_lrg(lrg)
        if err:
            return err
    args = _clean_tool_args({
        "guid": eff_guid,
        "comment": eff_comment,
        "lrg": lrg,
        "hardware": hardware,
        "sched": (eff_sched.lower() if eff_sched else None),
    })
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "reserve_hardware", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "reserve_hardware", "result": res})
    return _format_json(res)


class RealHWUnreserveArgs(BaseModel):
    sched: str | None = Field(None, description="Scheduler name (optional; auto-detect from hardware if omitted)")
    lrg: str | None = Field(None, description="LRG to map to a scheduler (optional)")
    guid: str | None = Field(None, description="User GUID (optional; defaults to requester)")
    hardware: str | None = Field(None, description="Comma-separated nodes, e.g. 'hw1,hw2' (optional when using LRG)")

@tool("unreserve_hardware", args_schema=RealHWUnreserveArgs)
async def unreserve_hardware_tool(sched: str | None = None, lrg: str | None = None, guid: str | None = None, hardware: str | None = None) -> str:
    """Release hardware nodes previously reserved for the requester GUID on a scheduler (or via LRG)."""
    eff_guid = _current_thread_guid() or (guid or "").strip()
    if not eff_guid:
        return ":x: Unable to determine requester GUID. Please mention me in this thread first or provide a GUID explicitly."
    eff_sched = (sched or "").strip()
    if not eff_sched and lrg:
        eff_sched, err = await _scheduler_from_lrg(lrg)
        if err:
            return err
    args = _clean_tool_args({
        "guid": eff_guid,
        "lrg": lrg,
        "hardware": hardware,
        "sched": (eff_sched.lower() if eff_sched else None),
    })
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "unreserve_hardware", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "unreserve_hardware", "result": res})
    return _format_json(res)


class FarmJobStatusArgs(BaseModel):
    job_id: str = Field(..., description="Farm job ID to query across schedulers")
    lrg: str | None = Field(None, description="LRG to filter results by mapped scheduler (optional)")


@tool("get_farm_job_status", args_schema=FarmJobStatusArgs)
async def get_farm_job_status_tool(job_id: str, lrg: str | None = None) -> str:
    """Get farm job status for a specific farm job ID across all schedulers as well as other details about farm job in the farm_cli field
    """
    args = {"job_id": (job_id or "").strip()}
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "get_farm_job_status", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_farm_job_status", "result": res})
    if not isinstance(res, dict) or res.get("error"):
        return _format_json(res)

    if lrg:
        sched, err = await _scheduler_from_lrg(lrg)
        if err:
            return err
        statuses = res.get("job_status") or []
        filtered = []
        for entry in statuses:
            sched_val = (entry.get("scheduler") or entry.get("sched") or "").strip()
            if sched_val == sched:
                filtered.append(entry)
        res = dict(res)
        res["job_status"] = filtered
        res["count"] = len(filtered)
        res["filtered_by_scheduler"] = sched
    logger.info(f"Dict sent for formatting: {res}")
    return _format_json(res)


class MoveJobToTopArgs(BaseModel):
    lrg: str = Field(..., description="LRG name (e.g., 'lrgrhexaprovcluster')")
    job_id: str = Field(..., description="Farm job ID (e.g., '39814499')")


@tool("move_job_to_top", args_schema=MoveJobToTopArgs)
async def move_job_to_top_tool(lrg: str, job_id: str) -> str:
    """Prioritize and move a farm job to the top of the waitlist for an LRG's scheduler."""
    args = {"lrg": (lrg or "").strip(), "job_id": (job_id or "").strip()}
    if not args["lrg"] or not args["job_id"]:
        return ":x: Provide both `lrg` and `job_id`."
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "move_job_to_top", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "move_job_to_top", "result": res})
    return _format_json(res)


class RealHWSimulateArgs(BaseModel):
    sched: str = Field(..., description="Scheduler name to estimate completion times for")


@tool("simulate_sched_end_time", args_schema=RealHWSimulateArgs)
async def simulate_sched_end_time_tool(sched: str) -> str:
    """Estimate job completion/queue drain timelines for a scheduler."""
    args = {"sched": (sched or "").strip()}
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "simulate_sched_end_time", args)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "simulate_sched_end_time", "result": res})
    return _format_json(res)


@tool("flag_usage_issues")
async def flag_usage_issues_tool() -> str:
    """List suspicious or problematic usage patterns across schedulers."""
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "flag_usage_issues", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "flag_usage_issues", "result": res})
    return _format_json(res)


@tool("flag_large_waitlist")
async def flag_large_waitlist_tool() -> str:
    """Highlight schedulers with large waitlists that may need attention."""
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "flag_large_waitlist", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "flag_large_waitlist", "result": res})
    return _format_json(res)


class RealHWQuarantinedArgs(BaseModel):
    lrg: str | None = Field(None, description="LRG to filter quarantined hardware by scheduler (optional)")


@tool("get_quarantined_hardware", args_schema=RealHWQuarantinedArgs)
async def get_quarantined_hardware_tool(lrg: str | None = None) -> str:
    """List quarantined hardware nodes across schedulers (optionally filtered by LRG's scheduler)."""
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "get_quarantined_hardware", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_quarantined_hardware", "result": res})
    if not isinstance(res, dict) or res.get("error") or not lrg:
        return _format_json(res)

    sched, err = await _scheduler_from_lrg(lrg)
    if err:
        return err
    items = res.get("quarantined_hardware") or []
    filtered = [item for item in items if (item.get("scheduler") or "").strip() == sched]
    res = dict(res)
    res["quarantined_hardware"] = filtered
    res["count"] = len(filtered)
    res["filtered_by_scheduler"] = sched
    return _format_json(res)


@tool("get_functional_hardware_mapping")
async def get_functional_hardware_mapping_tool() -> str:
    """List available functional hardware schedulers and their properties."""
    res = await asyncio.to_thread(REALHW_CLIENT.call_tool, "get_functional_hardware_mapping", {})
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_functional_hardware_mapping", "result": res})
    return _format_json(res)


def slack_recurring_prompt(state: AgentState) -> List[AnyMessage]:
    prompt = f"""You are Exadata Slack Assistant. You help users operate Oracle Exadata environments.\nAVAILABLE MCP TOOLS\n{SKILL_TOOL_SUMMARY}\n
OUTPUT & FORMATTING RULES (STRICT)
- Lists → table: When ANY tool returns a LIST, reply with ONE short sentence of context and ALWAYS render the list as a compact monospace table in a fenced code block (```), with a header row. Never use bullets for tool outputs.
- Summaries → table-like block: When ANY tool returns a SUMMARY (single object or narrative), ALWAYS render the summary in a fenced code block (```), with a simple header row. Keep the lead-in sentence short.
- Attachments note: Mention when attachments (such as es.xml) will appear if relevant. Only call tools when needed. Keep prose Slack-short and precise. If no tool is appropriate, answer directly.

TOOL SELECTION
- Lean on the tools listed above. They are available via MCP manifests.
- Prefer higher-level wrappers (e.g., smart_search) before calling low-level nodes.
TABLE FORMATTING GUIDELINES (TOOL-SPECIFIC)

RunIntegration:
- runintegration_idle_envs: Input may be list of dicts with “rack_name” and “deploy_type”. Table columns: “# | RACK | TYPE”.
- runintegration_disabled_envs: Input may be list of strings “rack : type” or dicts. Normalize to columns “# | RACK | TYPE”.
- runintegration_pending_tests: Show pending tests as “# | TEST | HOST” (use pending_details if present; otherwise list `pending_tests`).
- runintegration_disabled_txn_status: Table columns “# | HOST | TEST | STATUS”.

LRG Test Search:
- lrg_test_rag_query / lrg_test_fts_query / lrg_test_smart_search: Lead with one sentence, then present the returned stdout (and stderr if relevant) inside a fenced code block. Call out non-zero rc as an error row.
- get_lrg_runtime_stats: After the one-line lead-in, render scheduler data vertically inside a fenced code block.

Label Health – Series/Regress:
- get_labels_from_series: Table columns “# | LABEL”.
- get_lrgs_from_regress: Table columns “# | LRG”.
- find_lrg_with_difs: Each row has “lrg, sucs, difs, nwdif, intdif, szdif”. Table columns “# | LRG | SUCS | DIFS | NWDIF | INTDIF | SZDIF”.
- get_lrg_history: Table with columns like “# | LABEL | LRG | (other fields if present)”.

Label Health – Difs:
- find_dif_details: Rows with “lrg, name, rti_number, rti_assigned_to, rti_status, text/comments”. Table columns “# | LRG | NAME | RTI | STATUS | ASSIGNEE”. Truncate NAME to ~26 chars.
- find_dif_occurrence: Rows with “label, lrg, name, rti_number, rti_assigned_to”. Table columns “# | LABEL | LRG | DIF | RTI | ASSIGNEE”.
- find_widespread_issues: Rows with “name, lrgs (comma-separated), count=number of LRGs”. Table columns “# | DIF NAME | COUNT | LRGs”.

Label Health – Crashes:
- find_crashes: Rows with “lrg, name, status, rti_number, rti_assigned_to”. Table columns “# | LRG | NAME | STATUS | RTI | ASSIGNEE”.

Also print out the actual error message if you encounter.

Important Notes:
* Whenever you need to pass scheduler name, pass it as 'sched' and guid as 'guid'. Always pass sched in lowercase
* In pool status responses, lines starting with '#' indicate hardware that has been reserved by a person or quarantined due to issues. The rest of the hardwares are free/idle
* While reserving/unreserving, pass the node names as 'hardware'. If there are multiple nodes, pass them in 'hardware' only as a comma-separated string. If the user does not give a sched name, dont pass the sched argument. The tool will auto detect it
* Always tell user about the whole simulation if you use the simulate_sched_end_time tool
"""
    system_message = SystemMessage(content=prompt)
    existing_messages: List[AnyMessage] = list(state["messages"])
    return [system_message, *existing_messages]


LLM = make_llm()
AGENT_TOOLS = [
    generate_oedaxml_tool,
    runintegration_status_tool,
    runintegration_idle_envs_tool,
    runintegration_disabled_envs_tool,
    runintegration_pending_tests_tool,
    runintegration_disabled_txn_status_tool,
    map_lrg_to_scheduler_tool,
    view_status_of_sched_tool,
    reserve_hardware_tool,
    unreserve_hardware_tool,
    flag_usage_issues_tool,
    flag_large_waitlist_tool,
    get_quarantined_hardware_tool,
    get_farm_job_status_tool,
    move_job_to_top_tool,
    simulate_sched_end_time_tool,
    get_functional_hardware_mapping_tool,
    list_tests_for_lrg_tool,
    get_suites_for_lrg_tool,
    get_lrg_runtime_stats_tool,
    get_test_details_tool,
    search_tests_tool,
    lrg_test_health_tool,
    lrg_test_rag_query_tool,
    lrg_test_fts_query_tool,
    lrg_test_smart_search_tool,
    lrg_test_health_check_tool,
    lrg_test_rag_warmup_tool,
    summarize_text_tool,
    rag_query_tool,
    run_bug_test_tool,
    genai4test_health_tool,
    list_func_test_agents_tool,
    run_func_test_tool,
    run_func_mem_agent_tool,
    get_labels_from_series,
    get_lrgs_from_regress_tool,
    find_lrg_with_difs_tool,
    find_dif_details_tool, 
    find_dif_occurrence_tool,
    find_widespread_issues_tool,
    find_crashes_tool,
    get_my_lrgs_status_tool,
    add_lrg_to_my_lrgs_tool,
    get_my_rtis_tool,
    get_lrg_history_tool,  
    lrg_point_of_contact_tool,
    get_incomplete_lrgs_tool,
    draft_email_for_lrg_tool,
    query_ai_crash_summary_tool,
    get_se_rerun_details_tool,
    get_regress_summary_tool, 
    get_label_info_tool,
    get_ai_label_summary_tool,
    generate_ai_label_summary_tool,
    get_delta_diffs_between_labels_tool,
    add_series_to_auto_populate_tool,
    add_label_for_se_analysis_tool,
    add_label_for_analysis_tool,
]
AGENT = create_react_agent(
    model=LLM,
    tools=AGENT_TOOLS,
    prompt=slack_recurring_prompt,
    debug=True,
)


def _extract_message_text(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _collect_tool_names(messages: List[AnyMessage]) -> List[str]:
    names: List[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            if name:
                names.append(name)
    return names


async def _handle_tool_side_effects(thread_id: str, channel_id: str, thread_ts: str | None, slack_client) -> None:
    runs = TOOL_RUN_RESULTS.pop(thread_id, []) if thread_id else []
    for entry in runs:
        name = entry.get("name")
        result = entry.get("result") or {}
        if name == "post_func_buttons":
            try:
                await _post_func_buttons(slack_client, channel_id, thread_ts or thread_id, thread_id)
            except Exception as e:
                try:
                    await slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=f"⚠️ Unable to post functional test buttons: {e}")
                except Exception:
                    pass
            continue
        if name == "generate_oedaxml":
            es_b64 = result.get("es_xml_b64")
            tmp_path = None
            if es_b64:
                try:
                    xml_bytes = base64.b64decode(es_b64)
                    with TF.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
                        tmp.write(xml_bytes)
                        tmp_path = tmp.name
                    await slack_client.files_upload_v2(
                        channels=[channel_id],
                        thread_ts=thread_ts,
                        initial_comment="Attached is the generated `es.xml` file from `generate_oedaxml`.",
                        file=tmp_path,
                        filename="es.xml",
                        title="es.xml",
                    )
                    _record_artifact(thread_ts or str(thread_id), "es.xml", tmp_path, source="oeda")
                except Exception as upload_err:
                    await slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f":warning: Failed to upload es.xml: {upload_err}",
                    )
            live_check = result.get("live_mig_check")
            if live_check == "fail":
                reason = result.get("live_mig_reason") or "Live migration validation failed."
                rack_desc = result.get("rack_desc") or "unknown"
                await slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f":no_entry: Live migration check failed: {reason}\nRack description: `{rack_desc}`",
                )


# ---------------------------------------------------------------------------
# Slack event handler
# ---------------------------------------------------------------------------
@app.action("genai4test_request_regen")
async def handle_genai4test_request_regen(ack, body, client):
    await ack()
    action = (body.get("actions") or [{}])[0]
    payload = action.get("value")
    try:
        selection = json.loads(payload) if payload else {}
    except Exception:
        selection = {"raw": payload}
    channel_id = (body.get("channel") or {}).get("id") or (body.get("container") or {}).get("channel_id")
    user_id = (body.get("user") or {}).get("id")
    thread_key = selection.get("thread") or (body.get("container") or {}).get("thread_ts")
    file_id = selection.get("file_id")

    state = GENAI4TEST_THREAD_STATE.get(thread_key or "")
    file_info = (state.get("files") or {}).get(file_id) if isinstance(state, dict) else None

    state = GENAI4TEST_THREAD_STATE.get(thread_key or "")
    owner_id = isinstance(state, dict) and state.get("owner_id")
    if owner_id and user_id and owner_id != user_id:
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="⚠️ Only the original requester can regenerate these GenAI4Test files.",
        )
        return

    if not (channel_id and user_id and thread_key and file_id and file_info):
        msg = "⚠️ I couldn't locate that GenAI4Test artifact — please regenerate from the latest message."
        if channel_id and user_id:
            await client.chat_postEphemeral(channel=channel_id, user=user_id, text=msg)
        print("[genai4test] regenerate request missing context:", selection)
        return

    thread_ts = (body.get("message") or {}).get("thread_ts") or (body.get("container") or {}).get("thread_ts") or thread_key
    trigger_id = body.get("trigger_id")
    if not trigger_id:
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="⚠️ Unable to open regenerate dialog right now. Please try again.",
        )
        return

    if isinstance(state, dict):
        state["pending_selection"] = {"file_id": file_id, "thread": thread_key}

    private_meta = json.dumps({
        "thread": thread_key,
        "file_id": file_id,
        "channel": channel_id,
        "thread_ts": thread_ts,
        "filename": file_info.get("filename"),
    })

    try:
        await client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": "genai4test_regen_modal",
                "private_metadata": private_meta,
                "title": {"type": "plain_text", "text": "GenAI4Test regenerate"},
                "submit": {"type": "plain_text", "text": "Regenerate"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*File:* `{file_info.get('filename')}`\nProvide extra guidance so GenAI4Test knows what to change.",
                        },
                    },
                    {
                        "type": "input",
                        "block_id": "genai4test_regen_prompt",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "regen_prompt",
                            "multiline": True,
                            "placeholder": {"type": "plain_text", "text": "What should change in this file?"},
                        },
                        "label": {"type": "plain_text", "text": "Feedback / instructions"},
                    },
                ],
            },
        )
    except Exception as modal_err:
        print("[genai4test] modal open failed:", modal_err)
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="⚠️ Couldn't open the regenerate dialog. Please try again.",
        )
        return

@app.action("fb_up")
async def handle_fb_up(ack, body, client, say):
    await ack()

    # pull tiny payload with the uuid we stored in the buttons' value
    raw = (body.get("actions") or [{}])[0].get("value") or "{}"
    try:
        meta = json.loads(raw)
    except Exception:
        meta = {}
    uid = meta.get("uuid")

    # who / where
    user_id = (body.get("user") or {}).get("id")
    user_label = await _get_user_label(client, user_id)
    channel_id = (body.get("channel") or {}).get("id") or (body.get("container") or {}).get("channel_id")
    msg_ts = (body.get("message") or {}).get("ts") or (body.get("container") or {}).get("message_ts")
    original_text = (body.get("message") or {}).get("text") or "Thanks for the feedback."

    # metrics: append a click line with user
    try:
        append_jsonl(FEEDBACK_PATH, {
            "user": user_label,
            "uuid": uid,
            "ts": utc_iso(),
            "event": "thumb_up",
        })
    except Exception as e:
        print("[feedback up] append_jsonl error:", e)

    # UI: replace buttons with 'thanks' (prefer your feedback_blocks if it supports voted="up")
    try:
        if 'feedback_blocks' in globals():
            blocks = feedback_blocks(original_text, voted="up", payload_json=raw)
        else:
            blocks = feedback_thanks_blocks(original_text, "up")
        await client.chat_update(channel=channel_id, ts=msg_ts, text=original_text, blocks=blocks)
    except Exception as e:
        print("[feedback up] chat_update error:", e)

@app.action("fb_down")
async def handle_fb_down(ack, body, client, say):
    await ack()

    raw = (body.get("actions") or [{}])[0].get("value") or "{}"
    try:
        meta = json.loads(raw)
    except Exception:
        meta = {}
    uid = meta.get("uuid")

    # who / where
    user_id = (body.get("user") or {}).get("id")
    user_label = await _get_user_label(client, user_id)
    channel_id = (body.get("channel") or {}).get("id") or (body.get("container") or {}).get("channel_id")
    msg_ts = (body.get("message") or {}).get("ts") or (body.get("container") or {}).get("message_ts")
    original_text = (body.get("message") or {}).get("text") or "Thanks for the feedback."
    thread_ts = (body.get("message") or {}).get("thread_ts") or msg_ts

    # metrics: click with user
    try:
        append_jsonl(FEEDBACK_PATH, {
            "user": user_label,
            "uuid": uid,
            "ts": utc_iso(),
            "event": "thumb_down",
        })
    except Exception as e:
        print("[feedback down] append_jsonl error:", e)

    # UI: remove buttons now (same visual as before)
    try:
        if 'feedback_blocks' in globals():
            blocks = feedback_blocks(original_text, voted="down", payload_json=raw)
        else:
            blocks = feedback_thanks_blocks(original_text, "down")
        await client.chat_update(channel=channel_id, ts=msg_ts, text=original_text, blocks=blocks)
    except Exception as e:
        print("[feedback down] chat_update error:", e)

    # open modal to collect comment (keep the uuid and context)
    try:
        private_meta = json.dumps({
            "uuid": uid,
            "channel": channel_id,
            "thread_ts": thread_ts,
            "user": user_id,
        })
        await client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "fb_comment_modal",
                "private_metadata": private_meta,
                "title": {"type": "plain_text", "text": "Help us improve"},
                "submit": {"type": "plain_text", "text": "Send"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "fb_comment_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "fb_comment",
                            "multiline": True,
                            "placeholder": {"type": "plain_text", "text": "What didn’t work in this answer?"}
                        },
                        "label": {"type": "plain_text", "text": "Your feedback"}
                    }
                ],
            },
        )
    except Exception as e:
        print("[feedback modal] open error:", e)

@app.view("fb_comment_modal")
async def handle_fb_comment_submission(ack, body, client, view):
    await ack()
    # Extract comment
    try:
        comment = (view["state"]["values"]["fb_comment_block"]["fb_comment"]["value"] or "").strip()
    except Exception:
        comment = ""

    # Restore metadata
    meta_raw = view.get("private_metadata") or "{}"
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {}

    uid = meta.get("uuid")
    channel_id = meta.get("channel")
    thread_ts = meta.get("thread_ts")
    user_id = (body.get("user") or {}).get("id")

    # Record to feedback.jsonl
    user_label = await _get_user_label(client, user_id)
    append_jsonl(FEEDBACK_PATH, {
        "user": user_label,
        "uuid": uid,
        "ts": utc_iso(),
        "event": "feedback_comment",
        "comment": comment,
        "channel": channel_id,
        "thread_ts": thread_ts,
    })

    # Ephemeral thank-you in the same thread (no extra scopes needed)
    try:
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="Thanks for the feedback! 🙏 We’ll use it to improve.",
            thread_ts=thread_ts,
        )
    except Exception as e:
        print("[feedback modal] ephemeral thank-you error:", e)

@app.view("genai4test_regen_modal")
async def handle_genai4test_regen_modal(ack, body, client, view):
    await ack()

    try:
        metadata = json.loads(view.get("private_metadata") or "{}")
    except Exception as meta_err:
        print("[genai4test] regen metadata parse error:", meta_err)
        metadata = {}

    thread_key = metadata.get("thread")
    file_id = metadata.get("file_id")
    channel_id = metadata.get("channel")
    thread_ts = metadata.get("thread_ts") or (body.get("view") or {}).get("root_view_id") or thread_key
    filename = metadata.get("filename") or "selected file"
    user_id = (body.get("user") or {}).get("id")

    values = (view.get("state") or {}).get("values") or {}
    prompt_block = values.get("genai4test_regen_prompt") or {}
    user_prompt = ""
    for item in prompt_block.values():
        user_prompt = (item.get("value") or "").strip()
        break

    state = GENAI4TEST_THREAD_STATE.get(thread_key or "")
    file_info = (state.get("files") or {}).get(file_id) if state else None
    if not (state and file_info and channel_id and thread_ts):
        if channel_id and user_id:
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="⚠️ Regeneration context expired. Please run GenAI4Test again.",
            )
        print("[genai4test] regen modal missing context:", metadata)
        return

    if GENAI4TEST_CHAT is None:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="⚠️ GenAI4Test chat agent is not configured on this workspace.",
        )
        return

    bug_no = state.get("bug_no")
    base_request_id = state.get("request_id") or THREAD_REQUESTER_GUID.get(thread_key or "") or user_id or thread_key
    request_id = bug_no or base_request_id
    agent_name = GENAI4TEST_FOLLOWUP_AGENT
    file_name = file_info.get("zip_path") or file_info.get("filename")
    # Prefer the requester's GUID as the email/identifier for GenAI4Test chat
    email_val = (
        THREAD_REQUESTER_GUID.get(thread_key or "")
        or state.get("email")
        or OS.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")
    )
    mode = state.get("mode")
    if mode == "functional":
        user_input = state.get("request_id") or base_request_id
    else:
        user_input = request_id
    context_id = thread_key or thread_ts

    try:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f":hourglass_flowing_sand: Regenerating `{file_info.get('filename')}` with GenAI4Test…",
        )
    except Exception as notify_err:
        print("[genai4test] regen notify failed:", notify_err)

    response = await asyncio.to_thread(
        GENAI4TEST_CHAT.submit,
        email=email_val,
        user_input=user_input,
        agent_name=agent_name,
        user_prompt=user_prompt,
        context_id=context_id,
        file_name=file_name,
    )

    if not isinstance(response, dict) or not response.get("ok"):
        err = (isinstance(response, dict) and response.get("error")) or "unknown error"
        body_preview = isinstance(response, dict) and response.get("body")
        text = f":x: GenAI4Test regeneration failed: {err}"
        if body_preview:
            text += f"\n```text\n{str(body_preview)[:800]}\n```"
        await client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=text)
        return

    result_payload = response.get("result")
    if not isinstance(result_payload, dict):
        result_payload = response
    text_response = (result_payload.get("text_response") or "").strip()
    file_url = (result_payload.get("file_url") or "").strip()
    absolute_file_url = (result_payload.get("absolute_file_url") or "").strip()

    state.pop("pending_selection", None)
    state["last_prompt"] = user_prompt

    new_text = await _handle_genai4test_result(
        result_payload,
        thread_key=thread_key,
        thread_ts=thread_ts,
        channel_id=channel_id,
        client=client,
        preferred_file_id=file_id,
        merge_files=True,
    )

    confirmation_title = f":white_check_mark: GenAI4Test regenerated `{file_info.get('filename')}`."

    # Inline code preview for regenerated file when available (short, fenced)
    try:
        code_candidate = (result_payload.get("script") or result_payload.get("sql") or "").strip()
    except Exception:
        code_candidate = ""
    code_block = ""
    if code_candidate:
        fname = (file_info.get("filename") or "").lower()
        lang = "sql" if fname.endswith(".sql") else ("bash" if fname.endswith(".sh") else "text")
        # Keep preview short for Slack rendering; avoid snippet conversion issues
        max_lines = 120
        max_chars = 2500
        lines = code_candidate.splitlines()
        snippet = "\n".join(lines[:max_lines])
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars]
        if len(lines) > max_lines or len(code_candidate) > len(snippet):
            snippet = snippet.rstrip() + "\n... (truncated preview)"
        # Avoid double-fencing if upstream already added fences in the snippet
        if "```" not in snippet:
            code_block = f"```{lang}\n{snippet}\n```"
        else:
            code_block = snippet

    # If no code in payload or we still need a preview, try to fetch from the ZIP URL
    if not code_block:
        try:
            # Build absolute ZIP URL
            abs_zip = (result_payload.get("absolute_file_url") or "").strip()
            rel_zip = (result_payload.get("file_url") or "").strip()
            if not abs_zip and rel_zip:
                base = OS.getenv("GENAI4TEST_BASE_URL", "https://phoenix228912.dev3sub3phx.databasede3phx.oraclevcn.com:8000").rstrip("/") + "/"
                prefix = "/genai4test/"
                rel = rel_zip
                if not rel.startswith(prefix):
                    rel = f"genai4test/{rel.lstrip('/')}"
                from urllib.parse import urljoin
                abs_zip = urljoin(base, rel)

            if abs_zip:
                # Download ZIP and extract the selected file's content for preview
                verify_arg = OS.getenv("GENAI4TEST_CA_BUNDLE") or (OS.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() == "true")
                try:
                    content = _download_bytes(abs_zip, verify=verify_arg)
                except Exception as zip_dl_err:
                    print("[genai4test] ZIP download failed:", zip_dl_err)
                    content = None
                if content:
                    try:
                        with zipfile.ZipFile(BytesIO(content)) as zf:
                            names = zf.namelist()
                            target = None
                            base_target = (file_name or "").strip()
                            base_name = OS.path.basename(base_target) if base_target else ""
                            # Prefer exact filename match; otherwise pick first code-like file
                            if base_name:
                                for n in names:
                                    if OS.path.basename(n) == base_name:
                                        target = n
                                        break
                            if not target:
                                candidates = [n for n in names if n.lower().endswith((".sql", ".sh", ".py", ".tsc", ".txt"))]
                                target = candidates[0] if candidates else (names[0] if names else None)
                            if target:
                                raw = zf.read(target)
                                try:
                                    text = raw.decode("utf-8", errors="replace")
                                except Exception:
                                    text = raw.decode("latin-1", errors="replace")
                                # Truncate for Slack block limits
                                max_lines = 120
                                max_chars = 2500
                                lines = text.splitlines()
                                snippet = "\n".join(lines[:max_lines])
                                if len(snippet) > max_chars:
                                    snippet = snippet[:max_chars]
                                if len(lines) > max_lines or len(text) > len(snippet):
                                    snippet = snippet.rstrip() + "\n... (truncated preview)"
                                ext = OS.path.splitext(target)[1].lower()
                                lang = "sql" if ext == ".sql" else ("bash" if ext == ".sh" else ("python" if ext == ".py" else "text"))
                                if "```" not in snippet:
                                    code_block = f"```{lang}\n{snippet}\n```"
                                else:
                                    code_block = snippet
                    except Exception as unzip_err:
                        print("[genai4test] unzip failed:", unzip_err)
        except Exception as preview_err:
            print("[genai4test] preview-from-zip error:", preview_err)

    # Build Slack blocks to ensure fenced code renders as monospace inline
    blocks = []
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": confirmation_title}
    })
    # Prefer code-fenced preview over plain text to avoid duplicate content
    if code_block:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": code_block}
        })
    elif text_response:
        # Truncate long text response to fit Slack block limits
        if len(text_response) > 2800:
            text_response = text_response[:2770].rstrip() + "…"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": text_response}
        })
    # Download link section
    link_url = ""
    if absolute_file_url:
        link_url = absolute_file_url
    elif file_url:
        base = OS.getenv("GENAI4TEST_BASE_URL", "https://phoenix228912.dev3sub3phx.databasede3phx.oraclevcn.com:8000").rstrip("/") + "/"
        prefix = "/genai4test/"
        rel = file_url
        if not rel.startswith(prefix):
            rel = f"genai4test/{rel.lstrip('/')}"
        from urllib.parse import urljoin
        link_url = urljoin(base, rel)
    if link_url:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Download here:"},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Download ZIP", "emoji": True},
                "url": link_url,
                "action_id": "genai4test_download_link",
            }
        })

    await client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=confirmation_title,
        blocks=blocks,
    )
    try:
        await _post_genai4test_file_buttons(client, channel_id, thread_ts, thread_key)
    except Exception as e:
        print("[genai4test] follow-up buttons render error:", e)
@app.event("app_mention")
async def handle_app_mention(event, say, client):
    user_question = event.get("text", "")
    cleaned = " ".join(tok for tok in user_question.split() if not tok.startswith("<@"))
    lower = cleaned.lower()
    channel_id = event["channel"]
    base_ts = event.get("ts") or str(uuid.uuid4())
    thread_ts = event.get("thread_ts") or base_ts
    thread_key = thread_ts or base_ts
    pending_uploads: list[dict[str, str]] = []   # [{"path": "...", "filename": "...", "comment": "..."}]
    files = event.get("files") or []
    
    # Immediate feedback message
    user_id = event.get("user")  # Slack user ID 
    first_name = await _get_first_name(client, user_id)
    user_label = await _get_user_label(client, user_id)
    _remember_thread_requester(thread_key, user_label)
    st = GENAI4TEST_THREAD_STATE[thread_key]
    st["owner_id"] = user_id

    await say(f"Hi {first_name}, I received your request.\nPlease wait while I generate a response… :hourglass_flowing_sand:",
          thread_ts=thread_ts)


    # Functional test: detect uploaded test steps or inline code, then present agent choices
    try:
        selected = None
        for f in files:
            name = f.get("name", "")
            mt = f.get("mimetype", "")
            if _is_text_like(name, mt):
                selected = f
                break
        func_input_text = None
        if selected:
            name = selected.get("name", "test.txt")
            headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
            def _dl(u,h):
                r = requests.get(u, headers=h, timeout=60)
                r.raise_for_status()
                return r.content
            try:
                content = await asyncio.to_thread(_dl, selected["url_private_download"], headers)
                try:
                    func_input_text = content.decode("utf-8", errors="replace")
                except Exception:
                    func_input_text = content.decode("latin-1", errors="replace")
            except Exception as e:
                await say(f":x: Download error for `{name}`: {e}", thread_ts=thread_ts)
        if not func_input_text:
            # Try fenced code first
            code = _extract_first_code_block(cleaned)
            if code and len(code) > 8:
                func_input_text = code
        if not func_input_text:
            # Then try to extract numbered/bulleted test steps
            steps = _extract_test_steps_text(cleaned)
            if steps and len(steps) > 20:
                func_input_text = steps
        if func_input_text:
            st = GENAI4TEST_THREAD_STATE[thread_key]
            st["func_input_text"] = func_input_text
            st["func_filename"] = selected.get("name") if selected else "inline.txt"
            await _post_func_buttons(client, channel_id, thread_ts, thread_key)
            return
    except Exception as func_detect_err:
        print("[genai4test] func detect error:", func_detect_err)

    try:
        pdf_candidate = None
        for f in files:
            name = (f.get("name") or "").lower()
            mt = (f.get("mimetype") or "").lower()
            if name.endswith(".pdf") or "pdf" in mt:
                pdf_candidate = f
                break
        mem_keywords = ("func_mem", "func mem", "mem agent", "test plan")
        wants_mem_agent = bool(pdf_candidate and any(k in lower for k in mem_keywords))
        if wants_mem_agent:
            await _handle_func_mem_agent_request(
                client, channel_id, thread_ts, thread_key, pdf_candidate
            )
            return
    except Exception as mem_detect_err:
        print("[genai4test] func_mem detect error:", mem_detect_err)

    if "functional test" in lower:
        await _post_func_buttons(client, channel_id, thread_ts, thread_key)
        return

    if "summarize" in lower:
        try:
            files = event.get("files") or []
            had_pdf = False
            for f in files:
                name = f.get("name", "document.pdf")
                mt = f.get("mimetype", "")
                if name.lower().endswith(".pdf") or mt in ("application/pdf", "application/octet-stream"):
                    had_pdf = True
                    await say(f":page_facing_up: Got `{name}` — summarizing…", thread_ts=thread_ts)

                    # Download securely (use Slack token)
                    headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
                    def _dl(u,h):
                        r = requests.get(u, headers=h, timeout=60)
                        r.raise_for_status()
                        return r.headers.get("content-type",""), r.content
                    try:
                        ctype, content = await asyncio.to_thread(_dl, f["url_private_download"], headers)
                    except Exception as e:
                        await say(f":x: Download error for `{name}`: {e}", thread_ts=thread_ts)
                        continue

                    # Basic sanity: ensure it's a PDF
                    if not (content.startswith(b"%PDF") or "pdf" in ctype.lower()):
                        await say(
                            f":x: `{name}` does not look like a valid PDF (content-type `{ctype}` / magic {content[:4]!r}).",
                            thread_ts=thread_ts,
                        )
                        continue

                    # Write temp file
                    with TF.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name

                    # Call summarizer (wrap in thread; PersistentMCPClient is sync)
                    try:
                        res = await asyncio.to_thread(
                            SUM_CLIENT.call_tool, "lc_summarize_pdf_file", {"path": tmp_path}
                        )
                    except Exception as e:
                        await say(f":x: Summarizer call failed: {e}", thread_ts=thread_ts)
                        # Clean up file
                        try: OS.unlink(tmp_path)
                        except Exception: pass
                        continue

                    # Clean up temp file (we don't need it after the tool returns)
                    try:
                        OS.unlink(tmp_path)
                    except Exception:
                        pass

                    # Log raw for diagnosis (remove after stabilizing)
                    print("[SUM] raw:", res)

                    # Surface errors & missing summary with reasons
                    if not isinstance(res, dict):
                        await say(":x: Summarizer returned a non-JSON response.", thread_ts=thread_ts)
                        continu

                    if res.get("error"):
                        err = res["error"]
                        trace = res.get("trace") or res.get("details") or ""
                        msg = f":x: Summarizer error for `{name}`: {err}"
                        if trace:
                            msg += f"\n```text\n{str(trace)[:1200]}\n```"
                        await say(msg, thread_ts=thread_ts)
                        continue

                    summary = res.get("summary")
                    notes = res.get("notes") or res.get("reason") or res.get("chain_type") or res.get("message")
                    pages = res.get("pages") or res.get("num_pages")

                    if not summary:
                        # No summary but we have a reason—show it
                        msg = f":warning: Summarizer returned no summary for `{name}`."
                        if notes:
                            msg += f"\n*Reason:* {notes}"
                        await say(msg, thread_ts=thread_ts)
                    else:
                        # Success — show summary (code-fenced) + optional notes
                        out = f"*Summary for* `{name}` ({pages if pages is not None else '?'} pages):\n```\n{summary.strip()}\n```"
                        if notes:
                            out += f"\n_Note_: {notes}"
                        await say(out, thread_ts=thread_ts)

            if had_pdf:
                return
        except Exception as e:
            await say(f":x: MCP error (summarizer): {e}", thread_ts=thread_ts)
            return

    if ("jenkins" in lower and ("upgrade loop run" in lower or "upgrade_loop_run" in lower) and
        any(x in lower for x in ["submit", "build", "kick", "start"])):
        try:
            params = _parse_env_params_from_text(lower)
            await say(text="Got it ✅ submitting Jenkins build: UPGRADE_LOOP_RUN / 01_PRE_SETUP_FOR_SM", thread_ts=thread_ts)
            result = await trigger_upgrade_loop_run(params=params)
            msg = "\n".join(filter(None, [
                f"Params: {params}" if params else "",
                f"Queue: {result.get('queue_url')}",
                f"Job: {result.get('job_url')}",
            ]))
            await say(text=msg, thread_ts=thread_ts)
            threading.Thread(
                target=_monitor_and_notify,
                args=(result.get("queue_url"), result.get("job_url"), channel_id, thread_ts),
                daemon=True
            ).start()
        except Exception as e:
            await say(text=f"Trigger failed: `{e}`", thread_ts=thread_ts)
        return

    if any(w in lower for w in ["send", "transfer", "upload"]) and any(w in lower for w in ["file", "attachment", "generated", "test"]):
        # parse destination
        match = re.search(r'\b[\w.-]+@[\d.]+:[\w/\-_.]+\b', user_question)
        if not match:
            await say("❌ Please include a destination like `user@host:/path`.", thread_ts=thread_ts)
            return
        dest = match.group()

        # 1) Prefer explicit attachments (if user provided)
        files = event.get("files") or []
        if files:
            try:
                for f in files:
                    name = f["name"]
                    url = f["url_private_download"]
                    headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
                    # download in a thread
                    def _dl(u,h):
                        r = requests.get(u, headers=h, timeout=60)
                        r.raise_for_status()
                        return r.content
                    content = await asyncio.to_thread(_dl, url, headers)
                    with TF.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    ok = scp_file_with_key(tmp_path, dest, ssh_key_path="/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect")
                    if ok:
                        await say(f"✅ Sent `{name}` to `{dest}`", thread_ts=thread_ts)
                    else:
                        await say(f"❌ Failed to send `{name}` to `{dest}`", thread_ts=thread_ts)
                    # keep? usually safe to unlink explicit uploads
                    try: OS.unlink(tmp_path)
                    except Exception: pass
            except Exception as e:
                await say(f"⚠️ Error sending file: {e}", thread_ts=thread_ts)
            return

        # 2) No attachments — reuse the latest artifact generated in this thread
        art = _latest_artifact(thread_ts or str(thread_key))
        if not art or not OS.path.exists(art["local_path"]):
            await say("⚠️ I couldn’t find a recent generated file in this thread. Please attach the file or re-run generation.", thread_ts=thread_ts)
            return

        try:
            ok = scp_file_with_key(art["local_path"], dest, ssh_key_path="/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect")
            if ok:
                await say(f"✅ Sent `{art['filename']}` to `{dest}`", thread_ts=thread_ts)
            else:
                await say(f"❌ Failed to send `{art['filename']}` to `{dest}`", thread_ts=thread_ts)
        except Exception as e:
            await say(f"⚠️ Error sending file: {e}", thread_ts=thread_ts)
        return

    config = {"configurable": {"thread_id": thread_key}, "callbacks": [_LANGGRAPH_LOGGER]}
    prior_messages = list(THREAD_HISTORY[thread_key])
    agent_messages = prior_messages + [{"role": "user", "content": cleaned}]
    token = CURRENT_THREAD_ID.set(thread_key)
    try:
        result = await AGENT.ainvoke({"messages": agent_messages}, config=config)
    except Exception as agent_err:
        print(f"[AGENT ERROR] {agent_err}")
        traceback.print_exc()
        await say(":x: I hit an error while routing that request—trying a direct search fallback.", thread_ts=thread_ts)
        try:
            fallback = await asyncio.to_thread(RAG_CLIENT.call_tool, "rag_query", {"question": cleaned, "k": 3})
            _append_history(thread_key, "user", cleaned)
            if fallback.get("error"):
                err_msg = f":warning: RAG fallback errored: {fallback['error']}"
                await say(err_msg, thread_ts=thread_ts)
                _append_history(thread_key, "assistant", err_msg)
            else:
                ans = fallback.get("answer", "[no answer]")
                srcs = fallback.get("sources", []) or []
                src_lines = "\n".join(
                    f"• {s.get('title', 'untitled')} ({s.get('resource') or s.get('source') or 'n/a'})"
                    for s in srcs
                )
                fallback_text = f"{ans}\n\n*Sources:*\n{src_lines or '—'}"
                await say(fallback_text, thread_ts=thread_ts)
                _append_history(thread_key, "assistant", fallback_text)
        except Exception as fallback_err:
            err_msg = f":x: MCP error (RAG fallback): {fallback_err}"
            await say(err_msg, thread_ts=thread_ts)
            _append_history(thread_key, "assistant", err_msg)
        return
    finally:
        CURRENT_THREAD_ID.reset(token)

    messages = result.get("messages") or []
    final_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_message = msg
            break
    if final_message is None and messages:
        final_message = messages[-1]
    final_text = _extract_message_text(final_message) if final_message else ""
    final_text = _align_tables_in_text(final_text)
    tool_names = _collect_tool_names(messages)

    _append_history(thread_key, "user", cleaned)
    if final_text.strip():
        _append_history(thread_key, "assistant", final_text.strip())

    # --- genai4test enrichment: try to attach the generated test file, or at least show a link ---
    # --- genai4test enrichment: summary + inline test + zip, no script in final_text ---
    if "run_bug_test" in tool_names:
        try:
            # get the most recent run_bug_test result for this thread
            recent = None
            for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
                if e.get("name") == "run_bug_test":
                    recent = e.get("result") or {}
                    break

            if isinstance(recent, dict):
                new_text = await _handle_genai4test_result(
                    recent,
                    thread_key=thread_key,
                    thread_ts=thread_ts,
                    channel_id=channel_id,
                    client=client,
                )
                if new_text:
                    final_text = new_text
                try:
                    await _post_genai4test_file_buttons(client, channel_id, thread_ts or thread_key, thread_key)
                except Exception as buttons_err:
                    print("[genai4test] button render error:", buttons_err)

        except Exception as e:
            print("[genai4test] enrichment error:", e)
    # --- end enrichment ---

    feedback_context = {
        "feature": "langgraph_agent",
        "tools": tool_names,
        "question": cleaned[:300],
    }
    
    if final_text.strip():
        final_text = _humanize_html(final_text)
        chunks = _split_for_slack(final_text.strip(), max_chars=3500)
        total = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.rstrip()
            # If a chunk starts with a fence, add a tiny non-code prefix so Slack doesn't
            # convert it into a code snippet attachment and drop the opening fence in the next chunk.
            if total > 1 and chunk_text.lstrip().startswith("```"):
                prefix = f"Part {i}/{total}:\n\u200B"
                text_out = prefix + chunk_text
            else:
                text_out = chunk_text
            try:
                if i == total:
                    # ✅ Only the LAST chunk gets the feedback buttons
                    await post_with_feedback(
                        app, channel_id, thread_ts, text_out,
                        context=feedback_context,
                        user_id=event.get("user"),
                        client=client,
                    )
                else:
                    # Earlier parts: plain message in the same thread
                    await say(text_out, thread_ts=thread_ts)
            except Exception:
                await say(text_out, thread_ts=thread_ts)

            # No state to carry; splitter already balances fences across chunks.
    else:
        await say(":grey_question: I couldn't produce a response for that.", thread_ts=thread_ts)

    # --- DO THE UPLOADS *AFTER* POSTING THE SUMMARY ---
    if pending_uploads:
        for item in pending_uploads:
            try:
                await client.files_upload_v2(
                    channels=[channel_id],
                    thread_ts=thread_ts,                  # keep in the same thread
                    initial_comment=item.get("comment") or "Attachment:",
                    file=item["path"],
                    filename=item["filename"],
                    title=item["filename"],
                )
            finally:
                # clean up temp file
                try:
                    OS.unlink(item["path"])
                except Exception:
                    pass
    await _handle_tool_side_effects(thread_key, channel_id, thread_ts, client)




async def _post_func_buttons(client, channel_id: str, thread_ts: str, thread_key: str):
    st = GENAI4TEST_THREAD_STATE.get(thread_key or "") or {}
    func_input_text = (st.get("func_input_text") or "").strip()
    if not func_input_text:
        return
    buttons = {"type": "actions", "elements": []}
    button_defs = [
        ("Tsc Test Agent", "func_tsc_agent"),
        ("Java Test Agent", "func_java_agent"),
        ("RH Shell Test Agent", "exa_rh_func_test_agent"),
    ]
    for agent_label, agent_name in button_defs:
        payload = json.dumps({"thread": thread_key, "agent": agent_name})
        buttons["elements"].append({
            "type": "button",
            "text": {"type": "plain_text", "text": agent_label, "emoji": True},
            "value": payload,
            "style": "primary",
            "action_id": f"genai4test_func_agent:{agent_name}",
        })
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Detected test steps.* Choose a functional test agent:"}},
        buttons,
    ]
    await client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text="Choose functional test agent", blocks=blocks)





async def _handle_func_mem_agent_request(client, channel_id: str, thread_ts: str, thread_key: str, file_info: dict):
    name = file_info.get("name") or "document.pdf"
    url = file_info.get("url_private_download")
    if not url:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=":x: I couldn't access that PDF attachment for `func_mem_agent`.",
        )
        return

    try:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f":hourglass_flowing_sand: Generating a test plan from `{name}` via `func_mem_agent`…",
        )
    except Exception:
        pass

    headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}

    def _download_file():
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        return r.content

    try:
        content = await asyncio.to_thread(_download_file)
    except Exception as dl_err:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f":x: Failed to download `{name}`: {dl_err}",
        )
        return

    suffix = Path(name).suffix or ".pdf"
    with TF.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        pdf_path = tmp.name

    email_val = THREAD_REQUESTER_GUID.get(thread_key or "") or OS.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")
    args = {"pdf_path": pdf_path, "email": email_val, "context_id": thread_key}

    try:
        response = await asyncio.to_thread(
            GENAI4TEST_FUNC_CLIENT.call_tool,
            "run_func_mem_agent",
            args,
        )
    finally:
        try:
            OS.unlink(pdf_path)
        except Exception:
            pass

    if not isinstance(response, dict) or not response.get("ok"):
        err = (isinstance(response, dict) and (response.get("error") or response.get("status"))) or "unknown error"
        body = isinstance(response, dict) and response.get("body")
        msg = f":x: GenAI4Test `func_mem_agent` failed: {err}"
        if body:
            msg += f"\n```text\n{str(body)[:800]}\n```"
        await client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=msg)
        return

    state = GENAI4TEST_THREAD_STATE[thread_key]
    state["last_func_mem_agent"] = response

    summary = (response.get("summary") or "").strip()
    plan_text = (response.get("plan_text") or response.get("text_response") or response.get("plan") or "").strip()
    preview = plan_text or summary
    snippet = None
    if preview:
        lines = preview.splitlines()
        snippet = "\n".join(lines[:120])
        if len(snippet) > 2500:
            snippet = snippet[:2500]
        if len(lines) > 120 or len(preview) > len(snippet):
            snippet = snippet.rstrip() + "\n... (truncated preview)"
        if "```" not in snippet:
            snippet = f"```text\n{snippet}\n```"

    link_url = response.get("absolute_file_url") or response.get("file_url") or ""
    if link_url and not link_url.startswith(("http://", "https://")):
        base = OS.getenv("GENAI4TEST_BASE_URL", "https://phoenix228912.dev3sub3phx.databasede3phx.oraclevcn.com:8000").rstrip("/") + "/"
        prefix = "/genai4test/"
        rel = link_url
        if not rel.startswith(prefix):
            rel = f"genai4test/{rel.lstrip('/')}"
        from urllib.parse import urljoin
        link_url = urljoin(base, rel)

    blocks = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": f":white_check_mark: Generated a test plan via `func_mem_agent` from `{name}`."}
    }]
    if snippet:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": snippet}})
    elif summary:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"```text\n{summary}\n```"}})
    if link_url:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Download here:"},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Download ZIP", "emoji": True},
                "url": link_url,
                "action_id": "genai4test_download_link",
            }
        })

    await client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text="GenAI4Test test plan generated",
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# Functional test buttons handler + repost helper
# ---------------------------------------------------------------------------

@app.action(re.compile(r"^genai4test_func_agent"))
async def handle_genai4test_func_agent(ack, body, client):
    await ack()
    try:
        raw = (body.get("actions") or [{}])[0].get("value")
        data = json.loads(raw or "{}")
    except Exception:
        data = {}
    thread_key = (data.get("thread") or (body.get("container") or {}).get("thread_ts") or (body.get("message") or {}).get("ts"))
    channel_id = (body.get("container") or {}).get("channel_id") or (body.get("channel") or {}).get("id")
    agent_name = data.get("agent") or "func_tsc_agent"
    state = GENAI4TEST_THREAD_STATE.setdefault(thread_key or "", {})
    user_input = (state.get("func_input_text") or "").strip()
    email_val = THREAD_REQUESTER_GUID.get(thread_key or "") or OS.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")
    owner_id = state.get("owner_id")
    actor_id = (body.get("user") or {}).get("id")
    if owner_id and actor_id and owner_id != actor_id:
        await client.chat_postEphemeral(
            channel=channel_id,
            user=actor_id,
            text="⚠️ Only the requester who started this thread can trigger GenAI4Test actions here.",
        )
        return

    if not (thread_key and channel_id and user_input):
        await client.chat_postMessage(channel=channel_id, thread_ts=thread_key, text=":x: Functional test input expired. Please upload or paste again.")
        return

    state["mode"] = "functional"
    state["func_thread"] = thread_key
    state["channel_id"] = channel_id
    state["thread_ts"] = thread_key
    state["context_id"] = thread_key
    state["request_id"] = thread_key

    try:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_key,
            text=f":hourglass_flowing_sand: Running `{agent_name}` on the captured steps…",
        )
    except Exception:
        pass

    response = await asyncio.to_thread(
        GENAI4TEST_FUNC_CLIENT.call_tool,
        "run_func_test",
        {"user_input": user_input, "agent": agent_name, "email": email_val, "context_id": thread_key},
    )

    if not isinstance(response, dict) or not response.get("ok"):
        err = (isinstance(response, dict) and response.get("error")) or "unknown error"
        body_preview = isinstance(response, dict) and response.get("body")
        text = f":x: Functional test generation failed: {err}"
        if body_preview:
            text += f"\n```text\n{str(body_preview)[:800]}\n```"
        await client.chat_postMessage(channel=channel_id, thread_ts=thread_key, text=text)
        return

    result = response
    state["agent_name"] = agent_name
    state["email"] = email_val
    upload_note = await _handle_genai4test_result(
        result,
        thread_key=thread_key,
        thread_ts=thread_key,
        channel_id=channel_id,
        client=client,
    )
    text_response = (result.get("summary") or "").strip()
    file_url = (result.get("file_url") or "").strip()
    abs_url = (result.get("absolute_file_url") or "").strip()

    blocks = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": f":white_check_mark: Generated functional test via `{agent_name}`."},
    }]
    code_candidate = (result.get("script") or result.get("sql") or "").strip()
    if code_candidate:
        lines = code_candidate.splitlines()
        snippet = "\n".join(lines[:120])
        if len(snippet) > 2500:
            snippet = snippet[:2500]
        if len(lines) > 120 or len(code_candidate) > len(snippet):
            snippet = snippet.rstrip() + "\n... (truncated preview)"
        if "```" not in snippet:
            snippet = f"```text\n{snippet}\n```"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": snippet}})
    elif text_response:
        if len(text_response) > 2800:
            text_response = text_response[:2770].rstrip() + "…"
        summary_block = f"```text\n{text_response}\n```"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": summary_block}})

    link_url = abs_url or ""
    if not link_url and file_url:
        base = OS.getenv("GENAI4TEST_BASE_URL", "https://phoenix228912.dev3sub3phx.databasede3phx.oraclevcn.com:8000").rstrip("/") + "/"
        prefix = "/genai4test/"
        rel = file_url
        if not rel.startswith(prefix):
            rel = f"genai4test/{rel.lstrip('/')}"
        from urllib.parse import urljoin
        link_url = urljoin(base, rel)
    if link_url:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Download here:"},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Download ZIP", "emoji": True},
                "url": link_url,
                "action_id": "genai4test_download_link",
            }
        })
    if upload_note and not code_candidate and not text_response:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": upload_note}})

    await client.chat_postMessage(channel=channel_id, thread_ts=thread_key, text="Functional test generated", blocks=blocks)

    # After previewing the regenerated file, re-post the buttons list for further per-file regeneration
    # using the latest extracted_files state if available.
    try:
        await _post_genai4test_file_buttons(client, channel_id, thread_key, thread_key)
    except Exception as e:
        print("[genai4test] follow-up buttons render error:", e)



# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def _run():
    print("[BOOT] Starting Slack bot...")
    print("  LLM_PROVIDER:", OS.getenv("LLM_PROVIDER"))
    print("  RAG_CMD:", OS.getenv("RAG_CMD"))
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(_run())
    finally:
        # graceful MCP shutdown
        for cli in (RUNINTEG_CLIENT, OEDA_CLIENT, RAG_CLIENT, SUM_CLIENT):
            try: cli.close()
            except Exception: pass
