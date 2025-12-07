import os
from typing import Any, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin, quote


class GenAI4TestChatAgent:
    """
    Lightweight HTTP client for the GenAI4Test chat workflow.
    Handles proxy bypass, retries, and JSON payload marshalling.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        default_base = "https://testassist.oraclecorp.com/vm3/"
        raw_base = base_url or os.getenv("GENAI4TEST_CHAT_BASE_URL") or os.getenv("GENAI4TEST_BASE_URL") or default_base
        self.base_url = raw_base.rstrip("/")
        if not self.base_url:
            raise ValueError("GENAI4TEST_CHAT_BASE_URL or GENAI4TEST_BASE_URL must be configured.")

        self.endpoint = endpoint or os.getenv("GENAI4TEST_CHAT_ENDPOINT", "/genai4test/run-chat")
        self.timeout_s = float(timeout_s or os.getenv("GENAI4TEST_CHAT_TIMEOUT_S", os.getenv("GENAI4TEST_TIMEOUT_S", "600")))

        ca_bundle = os.getenv("GENAI4TEST_CA_BUNDLE")
        if ca_bundle:
            self.verify: Union[str, bool] = ca_bundle
        else:
            self.verify = os.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() in ("1", "true", "yes")

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
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10, pool_block=True)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        self.session = sess

    def _build_url(self, email: str, user_input: str, agent_name: str) -> str:
        base = self.base_url.rstrip("/") + "/"
        endpoint = self.endpoint.strip("/")
        return urljoin(
            base,
            f"{endpoint}/{quote(email, safe='')}/{quote(user_input, safe='')}/{quote(agent_name, safe='')}",
        )

    def submit(
        self,
        *,
        email: str,
        user_input: str,
        agent_name: str,
        user_prompt: str,
        context_id: Optional[str] = None,
        file_name: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a chat/regeneration request.

        Returns the parsed JSON response, augmented with an `ok` flag.
        """
        url = self._build_url(str(email), str(user_input), str(agent_name))
        params: Dict[str, Any] = {}
        prompt_val = (user_prompt or "").strip()
        if prompt_val:
            params["prompt"] = prompt_val
        if context_id:
            params["context_id"] = context_id
        if file_name:
            params["file_name"] = file_name
        if extra_payload:
            params.update({k: v for k, v in extra_payload.items() if v})

        try:
            resp = self.session.get(
                url,
                params=params or None,
                timeout=(10, self.timeout_s),
                verify=self.verify,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "request_url": url,
            }

        body_preview = resp.text[:1200] if hasattr(resp, "text") else ""
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
        data.setdefault("status", resp.status_code)
        return data


__all__ = ["GenAI4TestChatAgent"]
