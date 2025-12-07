# oeda_exaboard_agent.py
import os
import json
from typing import Any, Dict
from pathlib import Path
from dotenv import load_dotenv

from exaboard_client import ExaboardClient, ExaboardAuth
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
from llm_provider import make_llm

import time

def wait_for_generation(client: ExaboardClient, config_id: int, poll_interval: int = 10):
    """
    Poll GET /oeda/api/config/{config_id}/ until generator_task.state is SUCCESS or FAILURE.
    """
    print(f"Waiting for config {config_id} generation to complete...")

    while True:
        cfg = client.get_config(config_id)
        task = cfg.get("generator_task") or {}

        state = task.get("state", "UNKNOWN")
        msg = task.get("message", "")

        print(f"  generation state={state}, message={msg}")

        if state == "SUCCESS":
            print("✔ Config generation completed.")
            return

        if state in ("FAILURE", "ERROR"):
            raise RuntimeError(f"✖ Config generation failed: {msg}")

        time.sleep(poll_interval)


def wait_for_staging(client: ExaboardClient, config_id: int, poll_interval: int = 10):
    """
    Poll POST /oeda/api/config/{config_id}/init_deploy_status/ until staging finishes.
    """
    print(f"Waiting for staging to complete for config {config_id}...")

    while True:
        status = client.init_deploy_status(config_id)

        state = status.get("state", "UNKNOWN")
        progress = status.get("progress", 0)

        print(f"  staging state={state}, progress={progress}%")

        if state == "SUCCESS":
            print("✔ Staging completed.")
            return

        if state in ("FAILED", "ERROR"):
            raise RuntimeError(f"✖ Staging failed: {status}")

        time.sleep(poll_interval)


def clean_llm_json(text: str) -> str:
    """
    Strip markdown ```json fences, extra text, and attempt to extract the JSON object.
    """
    t = text.strip()
    t = t.replace("```json", "").replace("```", "").strip()

    # If the model wrapped text with explanation, extract the first {...} block.
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1:
            t = t[start : end + 1]

    return t

def resolve_system(client: ExaboardClient, user_request: str) -> Dict[str, Any]:
    """
    Search Exaboard systems based on user request.
    Return a dict with {system_id, system_string}.
    """

    systems = client.search_systems(user_request)
    if not systems:
        raise RuntimeError("No matching ExaBoard systems found from user request.")

    # If exactly one system matches → use it
    if len(systems) == 1:
        s = systems[0]
        return {
            "system_id": s["id"],
            "system_string": s.get("system") or s.get("system_string") or s["name"],
        }

    # If many, try to pick one with highest textual match on compressed hostnames
    # (This placeholder logic can be improved later)
    user_lower = user_request.lower()
    best = None
    best_score = -1
    for s in systems:
        sys_str = (s.get("system") or s.get("system_string") or s["name"]).lower()
        score = sum(w in sys_str for w in user_lower.split())
        if score > best_score:
            best_score = score
            best = s

    if best:
        return {
            "system_id": best["id"],
            "system_string": best.get("system") or best.get("system_string") or best["name"],
        }

    raise RuntimeError("Unable to resolve a specific system from ExaBoard search.")


def build_config_from_request(llm, user_request: str, exaboard_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the OCI LLM to turn a natural language request + Exaboard metadata
    into a config payload for POST /oeda/api/config/.
    """
    system_string = exaboard_context.get("system_string", "")
    system_id = exaboard_context.get("system_id", "")
    # You can also pre-fetch OEDA types, shapes, etc. via ExaboardClient and include in the prompt.

    system_prompt = """
You are an expert Exadata deployment assistant specialized in generating
valid ExaBoard OEDA configuration bodies for:

    POST /oeda/api/config/

Your job is to translate a natural language request into a STRICTLY valid
JSON dictionary that matches the ExaBoard OEDA config schema. You must obey
the following rules exactly:

============================================================
STRUCTURE RULES
============================================================

You must output a JSON object with these top-level fields:

{
  "id": "new",
  "tag": "short_snake_case_label",
  "description": "human readable description",
  "system": "<system string>",
  "oeda_version": "<version or default>",
  "use_deploy_oeda_for_generate": false,
  "active": false,
  "options": { ... }
}

- "id" must ALWAYS be "new".
- "tag" must be short, snake_case, and never contain spaces.
- "system" must be the correct system host string (e.g. "scaqan07adm07-08,celadm10-12").
- ALWAYS include "options" (never omit it).

============================================================
OPTIONS RULES (ALLOWED FIELDS)
============================================================

The "options" object may contain the following fields ONLY — do not invent new ones:

General:
- "oeda_type": { "id": <int>, "name": "<string>" }
- "customer_name"
- "application_name"
- "virtual" (true for virtual clusters)
- "secure_fabric" (true or false)
- "ip_version" (4 or 6)
- "use_backup_network" (boolean)
- "writeback_flash_cache" ("Automatic" or "Disabled")
- "asm_scoped_security" (boolean)
- "users_and_groups" ("Default")
- "home_dir_oracle"
- "home_dir_grid"
- UID/GID fields:
   - "uid_oracle"
   - "uid_grid"
   - "gid_oinstall"
   - "gid_dba"
   - "gid_racoper"
   - "gid_asmdba"
   - "gid_asmoper"
   - "gid_asmadmin"

Cluster-specific:
- "cluster_count": integer (1–32)
- "clusterGuestStorage": array of strings (length == cluster_count)
  Allowed values:
    - "localdisk"
    - "celldisk"
    - "edv"
- If user says "exascale", set EVERY cluster to EDV unless explicitly overridden.

If the user gives fewer values than cluster_count, auto-fill with the LAST provided value.

============================================================
INPUT MAPPING RULES
============================================================

1. **Host Strings**
   If user provides compressed hosts like:
     "scaqan07adm0708,scaqan07celadm10-12"
   You must preserve EXACTLY that string in the "system" field.

2. **Rack Prefix Extraction**
   If the system string begins with e.g. "scaqan07", you may put that in "description".

3. **Cluster Count**
   If user says "x clusters", map to:
     "cluster_count": x

4. **Cluster Guest Storage Logic**
   Examples:
   - "first 3 clusters on celldisk, last on localdisk" ->
       "clusterGuestStorage": ["celldisk","celldisk","celldisk","localdisk"]

   - If user says "exascale":
       default = "edv" unless overridden.

5. **Exascale**
   If user mentions:
     "exascale", "X9M EDV", "elastic storage"
   Then:
     - set "virtual" = true
     - default storage = "edv"

6. **Network Mode**
   - If user mentions IPv4 → "ip_version": 4
   - If user mentions IPv6 → "ip_version": 6
   - If unspecified → default: 4

7. **Secure Fabric**
   - If user says “secure fabric”, “SF”, “enable secure fabric”
     → secure_fabric = true
   - If user says “no secure fabric” → secure_fabric = false

8. **No mock values**
   Never hallucinate OEDA types. If no type is given:
   Use:
     "oeda_type": { "id": 0, "name": "Unknown" }

============================================================
OUTPUT RULES
============================================================

- Output MUST be STRICT JSON.
- No comments.
- No trailing commas.
- No explanations.
- No natural language.
- Do NOT wrap in Markdown fences unless the user explicitly asks for them.

Your entire response must be the JSON object only.
"""

    user_content = {
        "request": user_request,
        "system_string": system_string,
        "system_id": system_id,
        "exaboard_context": exaboard_context,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_content)},
    ]

    resp = llm.invoke(messages)
    raw_text = resp.content if hasattr(resp, "content") else str(resp)
    cleaned = clean_llm_json(raw_text)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}\nRAW={raw_text}") from e

    # Minimal safety checks:
    payload["id"] = "new"
    if "system" not in payload:
        payload["system"] = system_string

    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI-driven ExaBoard OEDA config agent")
    parser.add_argument(
        "--request",
        required=True,
        help='Natural language description, e.g. '
             '"I want 4 exascale clusters on scaqan07adm0708,scaqan07celadm10-12 '
             'with first 3 on celldisk and last on localdisk on IPv6 with secure fabric."',
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Perform staging and show deployment steps after generating configuration.",
    )
    args = parser.parse_args()

    # 1) Make LLM (OCI only, via your llm_provider)
    llm = make_llm()

    # 2) Set up Exaboard client
    exaboard_user = os.environ.get("EXABOARD_USER")
    exaboard_pass = os.environ.get("EXABOARD_PASS")
    if not exaboard_user or not exaboard_pass:
        raise SystemExit("Please set EXABOARD_USER and EXABOARD_PASS environment variables")

    auth = ExaboardAuth(username=exaboard_user, password=exaboard_pass)
    client = ExaboardClient(auth=auth)

    # 3) Automatic system resolution
    sysinfo = resolve_system(client, args.request)
    system_id = sysinfo["system_id"]
    system_string = sysinfo["system_string"]

    print(f"Resolved system_id = {system_id}")
    print(f"Resolved system_string = {system_string}")

    # 4) Build context
    exaboard_context = {
        "system_id": system_id,
        "system_string": system_string,
    }

    # 5) Build LLM config
    config_payload = build_config_from_request(llm, args.request, exaboard_context)
    print("=== Generated OEDA Config Payload ===")
    print(json.dumps(config_payload, indent=2))

    # 6) Create config
    cfg = client.create_config(config_payload)
    config_id = cfg.get("id") or cfg.get("config_id")
    print(f"Created config id={config_id}")

    # 7) Validate readiness
    readiness = client.generate_readiness(config_id)
    reasons = readiness.get("no_create_config_reasons") or []
    if reasons:
        print("Cannot generate config because:")
        for r in reasons:
            print(f"- {r['reason']} => {r['advice']}")
        return

    # 8) Generate
    print("Generating config...")
    client.generate_config(config_id)
    wait_for_generation(client, config_id)

    # Stop here unless deploy requested
    if not args.deploy:
        return

    # 9) Deploy readiness
    deploy_ready = client.deploy_readiness(config_id)
    reasons = deploy_ready.get("no_deploy_reasons") or []
    if reasons:
        print("Cannot deploy:")
        for r in reasons:
            print(f"- {r['reason']} => {r['advice']}")
        return

    # 10) Stage
    print("Staging...")
    client.init_deploy(config_id)
    wait_for_staging(client, config_id)

    # 11) Steps
    steps = client.list_deploy_steps(system_id)
    print("Deployment steps:")
    print(json.dumps(steps, indent=2))


if __name__ == "__main__":
    main()