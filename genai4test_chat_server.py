#!/usr/bin/env python3
import os
import requests
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from urllib.parse import quote, urljoin
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

BASE_URL = os.getenv(
    "GENAI4TEST_BASE_URL",
    "https://testassist.oraclecorp.com/vm3/",
)

VERIFY_SSL = os.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() == "true" \
             or os.getenv("GENAI4TEST_CA_BUNDLE")
CA_BUNDLE = os.getenv("GENAI4TEST_CA_BUNDLE")
TIMEOUT_S = float(os.getenv("GENAI4TEST_TIMEOUT_S", "600"))
DEFAULT_EMAIL = os.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")
DEFAULT_AGENT = os.getenv("GENAI4TEST_AGENT", "bug_agent_dynamic")

app = FastMCP("genai4test-chat-mcp")


def _make_session() -> requests.Session:
    sess = requests.Session()
    sess.trust_env = False
    sess.proxies = {"http": None, "https": None}
    # Identify Slack as the source for GenAI4Test service
    try:
        sess.headers["x-source"] = "Slack"
    except Exception:
        pass
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10, pool_block=True)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _verify_arg():
    if CA_BUNDLE:
        return CA_BUNDLE
    return bool(VERIFY_SSL)


@app.tool()
def health() -> Dict[str, Any]:
    """
    Proxy health check for the upstream GenAI4Test service.
    Returns the remote JSON when available, otherwise a minimal status payload.
    """
    base = BASE_URL.rstrip("/") + "/"
    url = urljoin(base, "genai4test/health")
    sess = _make_session()
    try:
        resp = sess.get(url, timeout=30, verify=_verify_arg())
        ct = (resp.headers.get("content-type") or "").lower()
        payload: Dict[str, Any] = {}
        if "application/json" in ct:
            try:
                payload = resp.json()
            except Exception:
                payload = {}
        if not payload:
            payload = {
                "status": "healthy" if resp.ok else "unhealthy",
                "checks": {},
                "timestamp": None,
            }
        return {
            "ok": resp.ok,
            "status_code": resp.status_code,
            "request_url": getattr(resp, "url", url),
            "result": payload,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "request_url": url}


@app.tool()
def run_bug_test(
    bug_no: str,
    email: Optional[str] = None,
    agent: Optional[str] = None,
    prompt: Optional[str] = None,
    context_id: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call GenAI4Test chat endpoint to generate or regenerate artefacts.
    bug_no maps to the chat API's user_input parameter.
    """
    bug_no = (bug_no or "").strip()
    if not bug_no:
        return {"ok": False, "error": "bug_no is required"}

    email = (email or DEFAULT_EMAIL).strip()
    agent = (agent or DEFAULT_AGENT).strip()
    prompt = (prompt or "").strip()
    context_id = (context_id or "").strip()
    file_name = (file_name or "").strip()

    email_q = quote(email, safe="")
    bug_q = quote(bug_no, safe="")
    agent_q = quote(agent, safe="")

    base = BASE_URL.rstrip("/") + "/"
    url = urljoin(base, f"genai4test/run-chat/{email_q}/{bug_q}/{agent_q}")

    params: Dict[str, Any] = {}
    if prompt:
        params["prompt"] = prompt
    if context_id:
        params["context_id"] = context_id
    if file_name:
        params["file_name"] = file_name

    sess = _make_session()
    try:
        resp = sess.get(
            url,
            params=params or None,
            timeout=(10, TIMEOUT_S),
            verify=_verify_arg(),
        )
        if resp.status_code != 200:
            body = resp.text[:500]
            return {
                "ok": False,
                "status": resp.status_code,
                "error": f"HTTP {resp.status_code}",
                "body": body,
                "request_url": resp.url,
            }

        data = resp.json()
    except requests.exceptions.ReadTimeout as exc:
        return {"ok": False, "error": f"ReadTimeout: {exc}", "request_url": url}
    except requests.exceptions.SSLError as exc:
        return {"ok": False, "error": f"SSLError: {exc}", "request_url": url}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "request_url": url}

    script = data.get("sql") or data.get("code") or data.get("script")
    file_url = data.get("file_url")
    abs_url = None
    if isinstance(file_url, str) and file_url:
        prefix = "/genai4test/"
        if not file_url.startswith(prefix):
            file_url = f"genai4test/{file_url.lstrip('/')}"
        abs_url = urljoin(base, file_url)

    return {
        "ok": True,
        "request_url": url,
        "request_url_with_params": resp.url if 'resp' in locals() else url,
        "bug_no": bug_no,
        "request_id": bug_no,
        "agent": agent,
        "email": email,
        "summary": data.get("summary"),
        "script": script,
        "sql": script,
        "file_url": file_url,
        "absolute_file_url": abs_url,
    }


@app.tool()
def tool_manifest() -> Dict[str, Any]:
    return {
        "service": "genai4test-chat-mcp",
        "tools": [
            {
                "name": "health",
                "description": "Check connectivity to the GenAI4Test chat service.",
                "intents": ["metadata"],
            },
            {
                "name": "run_bug_test",
                "description": "Generate or regenerate chat artefacts for a bug using GenAI4Test.",
                "intents": ["action", "generate"],
            },
        ],
    }


if __name__ == "__main__":
    app.run()
