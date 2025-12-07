import json
import sys
import os
import time

from llm_provider import make_llm
from exaboard_client import ExaboardClient, ExaboardAuth

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import re

def find_best_matching_system_name(user_prompt: str, client: ExaboardClient) -> str:
    """
    Extract rack ranges (ADM/CEL) and match them to an Exaboard system name.
    Supports inputs with or without dashes, e.g. 'adm0506' or 'adm05-06'.
    """
    # Keep delimiters to preserve boundaries
    text = re.sub(r"\s+", " ", user_prompt.lower())

    # 1. Extract prefix that appears immediately before 'adm' or 'celadm'
    prefix_match = re.search(r"([a-z][a-z0-9]+?)(?=(?:celadm|adm))", text)
    if not prefix_match:
        raise ValueError(f"No system prefix found in: {user_prompt}")
    prefix = prefix_match.group(1)

    # 2. Normalize rack ranges by inserting dashes if missing on a nospace variant
    text_no_space = text.replace(" ", "")
    text_no_space = re.sub(r"(adm)(\d{2})(\d{2})", r"\1\2-\3", text_no_space)
    text_no_space = re.sub(r"(celadm)(\d{2})(\d{2})", r"\1\2-\3", text_no_space)

    # 3. Extract rack patterns for this prefix
    pat = rf"{re.escape(prefix)}adm\d{{2}}-\d{{2}}|{re.escape(prefix)}celadm\d{{2}}-\d{{2}}"
    parts = re.findall(pat, text_no_space)
    parts = [p.strip() for p in parts if p]

    if not parts:
        # Fallback: pick ranges without prefix, then prepend the detected prefix
        ranges = re.findall(r"(?:celadm|adm)\d{2}-\d{2}", text_no_space)
        if ranges:
            parts = [prefix + r for r in ranges]

    if not parts:
        raise ValueError(f"No ADM/CEL rack patterns found for prefix {prefix}")

    # Derive rack tokens without the prefix (e.g., 'adm05-06', 'celadm07-09')
    tokens = [re.sub(rf"^{re.escape(prefix)}", "", p) for p in parts]

    def norm(s: str) -> str:
        return s.lower().replace(" ", "")

    # Restrict candidate systems to those matching the prefix
    try:
        candidates = client.search_systems(prefix)
    except Exception:
        candidates = client.list_systems()

    # Pass 1: strict substring match for tokens
    for s in candidates:
        name_norm = norm(s["system_name"])
        if prefix in name_norm and all(t in name_norm for t in tokens):
            return s["system_name"]

    # Pass 2: relaxed numeric matching (allow optional leading zeros)
    def token_to_regex(t: str) -> str:
        # Replace two-digit numbers with optional leading zero (e.g. 05 -> 0?5)
        return re.sub(r"(\d{2})", lambda m: f"0?{int(m.group(1))}", t)

    token_patterns = [re.compile(token_to_regex(t)) for t in tokens]

    for s in candidates:
        name_norm = norm(s["system_name"])
        if prefix in name_norm and all(p.search(name_norm) for p in token_patterns):
            return s["system_name"]

    # Final fallback: scan all systems strictly
    systems = client.list_systems()
    for s in systems:
        name_norm = norm(s["system_name"])
        if prefix in name_norm and all(t in name_norm for t in tokens):
            return s["system_name"]

    raise ValueError(f"Could not match system from parts={parts}")

# ---------------------------
# JSON CLEANING UTILITIES
# ---------------------------

def clean_llm_json(text: str) -> str:
    """Strip fences/explanations and extract the JSON object."""
    t = text.strip()

    # Remove markdown fences ```json ... ```
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:].strip()

    # Extract the first JSON object between { ... }
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output:\n" + t)

    return t[start : end + 1]


def extract_json(response):
    """Universal extractor for OCI/LC responses."""
    if isinstance(response, dict):
        return response

    # Extract raw text
    if hasattr(response, "content"):
        if isinstance(response.content, str):
            raw = response.content
        elif isinstance(response.content, list):
            raw = "".join(
                getattr(c, "text", str(c))
                for c in response.content
            )
        else:
            raw = str(response.content)
    else:
        raw = str(response)

    cleaned = clean_llm_json(raw)
    return json.loads(cleaned)


# ---------------------------
# MAIN TEST FLOW
# ---------------------------

def run():
    if len(sys.argv) < 2:
        print("Usage: python test_oeda_full_flow.py \"<user request>\"")
        sys.exit(1)

    request = sys.argv[1]

    print("=== INIT LLM ===")
    llm = make_llm()

    print("=== LOAD STRICT SYSTEM PROMPT ===")
    prompt_path = BASE_DIR / "system_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print("=== INIT EXABOARD CLIENT ===")
    auth = ExaboardAuth(
        username=os.getenv("EXABOARD_USER"),
        password=os.getenv("EXABOARD_PASS"),
    )
    client = ExaboardClient(auth=auth)

    print("=== USING STATIC SYSTEM ID ===")
    system_id = 43814
    print("System ID:", system_id)

    print("=== CALLING LLM TO GENERATE STRICT OEDA CONFIG JSON ===")
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ])

    config_body = extract_json(response)
    config_body["id"] = "new"
    # Extract dynamic system name from the user input
    try:
        system_name = find_best_matching_system_name(request, client)
        print(f"=== Matched system name from Exaboard: {system_name} ===")
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    config_body["system"] = system_name

    print("\n=== GENERATED CONFIG BODY ===")
    print(json.dumps(config_body, indent=2))

    print("\n=== CREATING CONFIG IN EXABOARD ===")
    cfg = client.create_config(config_body)
    print(json.dumps(cfg, indent=2))

    config_id = cfg["id"]
    print(f"Created config ID: {config_id}")

    print("\n=== GENERATE READINESS ===")
    ready = client.generate_readiness(config_id)
    print(json.dumps(ready, indent=2))

    print("\n=== GENERATE CONFIG ===")
    gen = client.generate_config(config_id)
    print(json.dumps(gen, indent=2))

    print("\n=== DEPLOY READINESS ===")
    deploy_ready = client.deploy_readiness(config_id)
    print(json.dumps(deploy_ready, indent=2))

    print("\n=== INIT DEPLOY (STAGING) ===")
    stage = client.init_deploy(config_id)
    print(json.dumps(stage, indent=2))

    print("\n=== POLLING STAGING STATUS ===")
    while True:
        status = client.init_deploy_status(config_id)
        print(json.dumps(status, indent=2))

        state = status.get("state")
        if state == "SUCCESS":
            print("Staging completed.")
            break
        if state in ("FAILED", "ERROR"):
            raise RuntimeError("Staging failed.")

        time.sleep(10)

    print("\n=== LIST DEPLOY STEPS ===")
    steps = client.list_deploy_steps(system_id)
    print(json.dumps(steps, indent=2))

    print("\n=== DONE ===")


if __name__ == "__main__":
    run()
