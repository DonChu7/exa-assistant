# lrg_test_tool/brain/llm_client.py

import requests
from typing import Optional


class WorkflowLLMClient:
    """
    Minimal client for calling your LLM workflows via HTTP.

    Assumes your workflow expects a JSON body like:
        {
          "workflow_id": WORKFLOW_ID,
          "variables": {
            "prompt": "<prompt text>"
          }
        }

    And returns JSON that looks like:
        {
          "response": [
            { "text": "<model output here>" }
          ],
          "session_id": "..."
        }

    Adjust parsing in `call()` if your actual schema differs.
    """

    def __init__(self, url: str, workflow_id: str, verify: Optional[str] = None):
        """
        :param url: Workflow HTTP endpoint
        :param workflow_id: ID of the workflow to run
        :param verify: Optional path to CA bundle for TLS verification.
                       - None  => use default requests/OS behavior
                       - False => disable verification (NOT recommended)
                       - str   => path to a CA bundle file
        """
        self.url = url
        self.workflow_id = workflow_id
        self.verify = verify  # e.g. "/etc/pki/tls/certs/ca-bundle.crt"

    def call(self, prompt: str) -> str:
        """
        Send the prompt to the workflow and return the model's text output.

        This is what assistant_cli.py uses.
        """
        payload = {
            "workflow_id": self.workflow_id,
            "variables": {
                "prompt": prompt
                # add more variables here if your workflow needs them
            },
        }

        resp = requests.post(
            self.url,
            json=payload,
            timeout=60,
            verify=self.verify if self.verify is not None else True,
        )
        resp.raise_for_status()
        data = resp.json()

        # You showed something like:
        # {'response': [{'text': '{ ...json... }'}], 'session_id': '...'}
        text = None

        responses = data.get("response") or data.get("responses")
        if isinstance(responses, list) and responses:
            first = responses[0]
            if isinstance(first, dict):
                # Most likely field names for the model text:
                text = first.get("text") or first.get("content") or first.get("result")

        # Fallbacks if your workflow uses a different top-level key
        if text is None:
            text = data.get("result") or data.get("text")

        # As a last resort, just dump the whole JSON for debugging
        if text is None:
            text = str(data)

        return text