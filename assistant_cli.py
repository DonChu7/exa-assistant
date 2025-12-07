#!/usr/bin/env python3
"""Interactive CLI for the LRG test assistant."""
import os
from pathlib import Path

from mcp_client import PersistentMCPClient

from lrg_test_tool.brain.pipeline import RouterWriterPipeline


# Adjust this to how you start your MCP server
SERVER = ["python", str(Path(__file__).resolve().parent / "lrg_test_mcp_server.py")]


def main():
    pipeline = RouterWriterPipeline.from_env()
    client = PersistentMCPClient(SERVER)
    print("Exa test assistant CLI. Type 'quit' to exit.\n")

    try:
        while True:
            user_query = input("you> ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("quit", "exit"):
                break

            try:
                qp = pipeline.route(user_query)
            except Exception as e:
                print(f"\nassistant> Router failure: {e}")
                if getattr(pipeline, "last_router_raw", None):
                    print("[router raw output]\n" + pipeline.last_router_raw)
                continue

            tool_resp = client.call_tool("smart_search", {
                "question": user_query,
                "plan": qp.model_dump(),
            })

            if isinstance(tool_resp, dict) and not tool_resp.get("ok", True):
                error = tool_resp.get("error") or "backend error"
                print(f"\nassistant> smart_search failed: {error}")
                continue

            try:
                answer = pipeline.compose(user_query, tool_resp)
            except Exception as e:
                print(f"\nassistant> Writer failure: {e}")
                continue

            print("\nassistant>")
            print(answer)
            print()

    finally:
        client.close()


if __name__ == "__main__":
    main()
