#!/usr/bin/env python3
import os, json, re, traceback
import requests
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from urllib.parse import quote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import quote, urlparse, urljoin
import base64

# --- Config via env ---
BASE_URL = os.getenv("GENAI4TEST_BASE_URL",
    "https://testassist.oraclecorp.com/vm3/")
#BASE_URL = os.getenv("GENAI4TEST_BASE_URL",
#    "https://phoenix518455.dev3sub2phx.databasede3phx.oraclevcn.com:9000")
VERIFY_SSL = os.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() == "true" \
             or os.getenv("GENAI4TEST_CA_BUNDLE")  # path to CA file if provided
CA_BUNDLE = os.getenv("GENAI4TEST_CA_BUNDLE")  # e.g. /etc/ssl/certs/your-ca.pem
TIMEOUT_S = float(os.getenv("GENAI4TEST_TIMEOUT_S", "600"))
DEFAULT_EMAIL = os.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")
DEFAULT_AGENT = os.getenv("GENAI4TEST_AGENT", "bug_agent_dynamic")
FUNC_TEST_AGENTS = [
    "func_tsc_agent",
    "func_java_agent",
    "exa_rh_func_test_agent",
]
FOLLOWUP_AGENT = os.getenv("GENAI4TEST_CHAT_AGENT_NAME", "chat_agent")

app = FastMCP("genai4test-mcp")


def _make_session() -> requests.Session:
    sess = requests.Session()
    sess.trust_env = False
    sess.proxies = {"http": None, "https": None}
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
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10, pool_block=True)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _call_run_chat(
    *,
    email: str,
    user_input: str,
    agent: str,
    prompt: str | None = None,
    extra_params: Dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> Dict[str, Any]:
    email_q = quote(email, safe="")
    input_q = quote(user_input, safe="")
    agent_q = quote(agent, safe="")
    url = f"{BASE_URL}/genai4test/run-chat/{email_q}/{input_q}/{agent_q}"

    sess = session or _make_session()
    verify_arg = CA_BUNDLE if CA_BUNDLE else bool(VERIFY_SSL)
    timeout_arg = (10, TIMEOUT_S)

    params: Dict[str, Any] = {}
    if prompt:
        params["prompt"] = prompt
    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v})

    resp = sess.get(url, params=params or None, timeout=timeout_arg, verify=verify_arg)
    body_preview = resp.text[:500]
    if resp.status_code != 200:
        return {
            "ok": False,
            "status": resp.status_code,
            "error": f"HTTP {resp.status_code}",
            "body": body_preview,
            "request_url": url,
        }
    try:
        data = resp.json()
    except ValueError:
        return {
            "ok": False,
            "status": resp.status_code,
            "error": "invalid-json",
            "body": body_preview,
            "request_url": url,
        }
    data.setdefault("ok", True)
    data.setdefault("request_url", url)
    return data


@app.tool()
def health() -> Dict[str, Any]:
    """Quick reachability probe for the genai4test service."""
    try:
        verify_arg = CA_BUNDLE if CA_BUNDLE else bool(VERIFY_SSL)
        r = requests.get(f"{BASE_URL}/docs", timeout=30, verify=verify_arg)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.tool()
def run_bug_test(bug_no: str, email: str | None = None, agent: str | None = None, context_id: str | None = None) -> dict:
    """
    Call genai4test to generate a test from a bug.
    Returns: {ok, summary, sql, file_url, request_url, status, error?}
    """
    try:
        bug_no = (bug_no or "").strip()
        if not bug_no:
            return {"ok": False, "error": "bug_no is required"}

        email = (email or DEFAULT_EMAIL).strip()
        agent = (agent or DEFAULT_AGENT).strip()

        sess = _make_session()
        extra = {"context_id": context_id} if context_id else None
        data = _call_run_chat(email=email, user_input=bug_no, agent=agent, extra_params=extra, session=sess)
        if not data.get("ok"):
            return data

        script = data.get("sql") or data.get("code") or data.get("script")
        file_url = data.get("file_url")
        abs_url = None
        if isinstance(file_url, str) and file_url:
            # Ensure file_url is correctly rooted under /genai4test/
            prefix = "/genai4test/"
            if not file_url.startswith(prefix):
                # Strip leading slashes and enforce /genai4test/ prefix
                file_url = f"genai4test/{file_url.lstrip('/')}"
            abs_url = urljoin(BASE_URL.rstrip("/") + "/", file_url)

        return {
            "ok": True,
            "request_url": data.get("request_url"),
            "bug_no": bug_no,
            "request_id": bug_no,
            "agent": agent,
            "summary": data.get("summary"),
            "script": script,
            "sql": script,                    # <-- key your Slack code already expects
            "file_url": file_url,
            "absolute_file_url": abs_url,     # <-- optional convenience
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.tool()
def list_func_test_agents() -> dict:
    """
    List available function-test agents for GenAI4Test.
    Returns: {ok, agents: [names]}
    """
    return {"ok": True, "agents": FUNC_TEST_AGENTS}


@app.tool()
def run_func_test(
    user_input: str,
    agent: str,
    email: str | None = None,
    context_id: str | None = None,
) -> dict:
    """
    Call GenAI4Test functional-test endpoint to generate tests from a test step file or text.
    Returns: {ok, request_url, agent, email, file_url?, absolute_file_url?, summary?, sql?/script?}
    """
    try:
        user_input = (user_input or "").strip()
        if not user_input:
            return {"ok": False, "error": "user_input is required"}
        agent = (agent or "").strip()
        if not agent:
            return {"ok": False, "error": "agent is required (choose one of: " + ", ".join(FUNC_TEST_AGENTS) + ")"}
        email = (email or DEFAULT_EMAIL).strip()
        context_id = (context_id or "").strip()
        if not context_id:
            return {"ok": False, "error": "context_id is required"}

        sess = _make_session()

        verify_arg = CA_BUNDLE if CA_BUNDLE else bool(VERIFY_SSL)
        timeout_arg = (10, TIMEOUT_S)

        # Step 1: upload user input (text or file) so run-test can reference it
        upload_url = urljoin(BASE_URL.rstrip('/') + '/', 'genai4test/upload-file/')
        upload_kwargs: Dict[str, Any] = {"timeout": timeout_arg, "verify": verify_arg}
        upload_debug: Dict[str, Any] = {"input_kind": "text", "text_len": len(user_input)}
        file_handle = None
        if os.path.exists(user_input) and os.path.isfile(user_input):
            upload_payload = {"input_type": "File", "email": email, "text": ""}
            file_handle = open(user_input, "rb")
            upload_kwargs["data"] = upload_payload
            upload_kwargs["files"] = {"file": (os.path.basename(user_input) or "input.txt", file_handle)}
            upload_debug = {
                "input_kind": "file",
                "file_path": user_input,
                "file_size": os.path.getsize(user_input),
            }
        else:
            upload_payload = {
                "input_type": "Text",
                "text": user_input,
                "email": email,
            }
            upload_kwargs["data"] = upload_payload
            upload_debug["text_preview"] = user_input[:120]
        try:
            upload_resp = sess.post(upload_url, **upload_kwargs)
        finally:
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass
        if upload_resp.status_code != 200:
            body = upload_resp.text[:500]
            print("[genai4test] upload-file error", {
                "status": upload_resp.status_code,
                "body": body,
                "debug": upload_debug,
            })
            return {
                "ok": False,
                "status": upload_resp.status_code,
                "error": f"upload-file HTTP {upload_resp.status_code}",
                "body": body,
                "upload_url": upload_url,
                "upload_debug": upload_debug,
            }
        upload_data = upload_resp.json()
        print("[genai4test] upload-file response", {
            "debug": upload_debug,
            "response_keys": list(upload_data.keys()),
        })
        uploaded_path = (
            upload_data.get("file_path")
            or upload_data.get("path")
            or upload_data.get("file")
            or upload_data.get("user_input")
            or upload_data.get("input_file")
            or upload_data.get("saved_to")
        )
        if not uploaded_path:
            print("[genai4test] upload-file missing path", {
                "response": upload_data,
                "debug": upload_debug,
            })
            return {
                "ok": False,
                "error": "upload-file response missing file path",
                "upload_url": upload_url,
                "upload_response": upload_data,
                "upload_debug": upload_debug,
            }

        data = _call_run_chat(
            email=email,
            user_input=str(uploaded_path),
            agent=agent,
            extra_params={"context_id": context_id},
            session=sess,
        )
        if not data.get("ok"):
            data["uploaded_input"] = uploaded_path
            data.setdefault("upload_response", upload_data)
            return data

        script = data.get("sql") or data.get("code") or data.get("script")
        file_url = data.get("file_url")
        abs_url = None
        if isinstance(file_url, str) and file_url:
            prefix = "/genai4test/"
            if not file_url.startswith(prefix):
                file_url = f"genai4test/{file_url.lstrip('/')}"
            abs_url = urljoin(BASE_URL.rstrip("/") + "/", file_url)

        request_id = data.get("request_id") or data.get("req_id")

        return {
            "ok": True,
            "request_url": data.get("request_url"),
            "agent": agent,
            "email": email,
            "uploaded_input": uploaded_path,
            "upload_response": upload_data,
            "context_id": context_id,
             "request_id": request_id,
            "summary": data.get("summary"),
            "script": script,
            "sql": script,
            "file_url": file_url,
            "absolute_file_url": abs_url,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}



@app.tool()
def run_func_mem_agent(pdf_path: str, email: str | None = None, context_id: str | None = None) -> dict:
    """
    Generate a test plan using func_mem_agent from a PDF document.
    """
    try:
        pdf_path = (pdf_path or "").strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            return {"ok": False, "error": f"pdf_path not found: {pdf_path!r}"}
        email = (email or DEFAULT_EMAIL).strip()

        sess = _make_session()
        verify_arg = CA_BUNDLE if CA_BUNDLE else bool(VERIFY_SSL)
        timeout_arg = (10, TIMEOUT_S)

        upload_url = f"{BASE_URL}/genai4test/upload-file/"
        upload_payload = {"input_type": "File", "email": email, "text": ""}
        with open(pdf_path, "rb") as handle:
            files = {"file": (os.path.basename(pdf_path) or "input.pdf", handle)}
            upload_resp = sess.post(
                upload_url,
                data=upload_payload,
                files=files,
                timeout=timeout_arg,
                verify=verify_arg,
            )

        if upload_resp.status_code != 200:
            body = upload_resp.text[:500]
            return {
                "ok": False,
                "status": upload_resp.status_code,
                "error": f"upload-file HTTP {upload_resp.status_code}",
                "body": body,
                "upload_url": upload_url,
            }
        upload_data = upload_resp.json()
        uploaded_path = (
            upload_data.get("file_path")
            or upload_data.get("path")
            or upload_data.get("file")
            or upload_data.get("user_input")
            or upload_data.get("input_file")
            or upload_data.get("saved_to")
        )
        if not uploaded_path:
            return {
                "ok": False,
                "error": "upload-file response missing file path",
                "upload_response": upload_data,
            }

        extra = {"context_id": context_id} if context_id else None
        data = _call_run_chat(
            email=email,
            user_input=str(uploaded_path),
            agent="func_mem_agent",
            extra_params=extra,
            session=sess,
        )
        if not data.get("ok"):
            data["uploaded_input"] = uploaded_path
            data.setdefault("upload_response", upload_data)
            return data

        file_url = data.get("file_url")
        abs_url = None
        if isinstance(file_url, str) and file_url:
            prefix = "/genai4test/"
            if not file_url.startswith(prefix):
                file_url = f"genai4test/{file_url.lstrip('/')}"
            abs_url = urljoin(BASE_URL.rstrip("/") + "/", file_url)

        result = dict(data)
        if abs_url:
            result["absolute_file_url"] = abs_url
        result.setdefault("agent", "func_mem_agent")
        result["uploaded_input"] = uploaded_path
        result["upload_response"] = upload_data
        result["context_id"] = context_id
        return result
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.tool()
def tool_manifest() -> Dict[str, Any]:
    return {
        "service": "genai4test-mcp",
        "tools": [
            {
                "name": "health",
                "description": "Check connectivity to the GenAI4Test service.",
                "intents": ["metadata"],
            },
            {
                "name": "run_bug_test",
                "description": "Generate regression test artifacts from a bug ID.",
                "intents": ["action", "generate"],
            },
            {
                "name": "list_func_test_agents",
                "description": "List available functional test agents.",
                "intents": ["query"],
            },
            {
                "name": "run_func_test",
                "description": "Generate functional tests using a specified agent and inputs.",
                "intents": ["action", "generate"],
            },
            {
                "name": "run_func_mem_agent",
                "description": "Produce a test plan via func_mem_agent for an uploaded PDF.",
                "intents": ["generate", "analyze"],
            },
        ],
    }

if __name__ == "__main__":
    app.run()
