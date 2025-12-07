# test_llm_config.py

import os
import json
import time
from typing import Any, Dict

from llm_provider import make_llm
from exaboard_client import ExaboardClient, ExaboardAuth

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
from llm_provider import make_llm


SYSTEM_PROMPT_PATH = "system_prompt.txt"


def call_llm_build_payload(request: str) -> Dict[str, Any]:
    """Call OCI LLM with tightened system prompt and return parsed JSON payload."""
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    llm = make_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]

    print("=== Calling OCI LLM ===")
    print("User request:", request)

    result = llm.invoke(messages)
    output = result.content.strip()

    print("\n=== Raw LLM output ===")
    print(output)

    print("\n=== Parsing JSON from LLM ===")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON: {e}\nRaw:\n{output}") from e

    # enforce id="new"
    payload["id"] = "new"
    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end test: LLM → ExaBoard OEDA config → generate → deploy readiness → stage → steps list."
    )
    parser.add_argument(
        "--request",
        required=True,
        help="Natural language description of the desired env",
    )
    parser.add_argument(
        "--system-id",
        type=int,
        required=True,
        help="ExaBoard system ID (e.g. 14194)",
    )
    parser.add_argument(
        "--base-url",
        default="https://exaboard.oraclecorp.com/oeda/api/",
        help="Base URL for ExaBoard OEDA API (default: %(default)s)",
    )
    args = parser.parse_args()

    exa_user = os.environ.get("EXABOARD_USER")
    exa_pass = os.environ.get("EXABOARD_PASS")
    if not exa_user or not exa_pass:
        raise SystemExit(
            "Please set EXABOARD_USER and EXABOARD_PASS environment variables with your ExaBoard credentials."
        )

    auth = ExaboardAuth(username=exa_user, password=exa_pass)
    client = ExaboardClient(base_url=args.base_url, auth=auth)

    # 1) Call LLM to get JSON payload
    payload = call_llm_build_payload(args.request)

    print("\n=== LLM-generated payload ===")
    print(json.dumps(payload, indent=2))

    # 2) Create config
    print("\n=== Step 1: Create config ===")
    cfg = client.create_config(payload)
    print("Create config response:")
    print(json.dumps(cfg, indent=2))

    config_id = cfg.get("id") or cfg.get("config_id")
    if not config_id:
        raise RuntimeError("Could not find config id in create_config response")

    print(f"\nConfig ID: {config_id}")
    print(f"Config URL: {args.base_url.rstrip('/')}/config/{config_id}/")

    # 3) Validate generate readiness
    print("\n=== Step 2: Validate generate readiness ===")
    gen_ready = client.generate_readiness(config_id)
    print(json.dumps(gen_ready, indent=2))

    no_create_reasons = gen_ready.get("no_create_config_reasons") or []
    if no_create_reasons:
        print("\nConfig is NOT ready to generate. Reasons:")
        for entry in no_create_reasons:
            print("-", entry.get("reason"), "=>", entry.get("advice"))
        return
    print("Generate readiness OK (no_create_config_reasons is empty).")

    # 4) Generate config
    print("\n=== Step 3: Generate config ===")
    gen_resp = client.generate_config(config_id)
    if gen_resp:
        print("Generate response:")
        print(json.dumps(gen_resp, indent=2))
    else:
        print("Generate response was empty (normal for this API).")

    print(
        f"You can monitor generation via GET {args.base_url.rstrip('/')}/config/{config_id}/"
    )

    # Small pause so generation can start/update status
    time.sleep(5)

    # 5) Validate deploy readiness
    print("\n=== Step 4: Validate deploy readiness ===")
    dep_ready = client.deploy_readiness(config_id)
    print(json.dumps(dep_ready, indent=2))

    validation_errors = dep_ready.get("validation_errors") or []
    if validation_errors:
        print("\nConfig is NOT ready to deploy. Errors:")
        for entry in validation_errors:
            print("-", entry.get("error"))
        # Might still continue to stage in a dry-run scenario, but we'll stop here.
        return

    print("Deploy readiness OK (no validation_errors).")

    # 6) Stage (init_deploy)
    print("\n=== Step 5: Stage (init_deploy) ===")
    stage_resp = client.init_deploy(config_id, reset=True)
    if stage_resp:
        print("init_deploy response:")
        print(json.dumps(stage_resp, indent=2))
    else:
        print("init_deploy response was empty (normal).")

    print(
        f"You can check staging via POST {args.base_url.rstrip('/')}/config/{config_id}/init_deploy_status/"
    )

    # 7) Check staging status once (monitor)
    print("\n=== Step 6: Check staging status (init_deploy_status) ===")
    status_resp = client.init_deploy_status(config_id)
    print(json.dumps(status_resp, indent=2))

    # 8) List deployment steps
    print("\n=== Step 7: List deployment steps (no execution) ===")
    steps = client.list_deploy_steps(args.system_id)
    print(json.dumps(steps, indent=2))

    print("\nDone. Config created, validated, generated, staging kicked, and steps listed.")


if __name__ == "__main__":
    main()