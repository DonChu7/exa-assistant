import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

# Adjust if your flags_kb.json is elsewhere
KB_PATH = Path(__file__).resolve().parent / "flags_kb.json"


@lru_cache(maxsize=1)
def load_flags_kb() -> List[Dict[str, Any]]:
    """
    Load the flags/setup knowledge base from JSON once and cache it.
    """
    data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{KB_PATH} must contain a JSON array")
    return data


def search_flags_kb(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Very simple keyword-based search over the KB.

    - Splits the user query into tokens
    - Scores each KB entry by how many tokens appear in id/title/summary/keywords/aliases
    - Returns top N entries
    """
    q = (query or "").lower().strip()
    if not q:
        return []

    tokens = [t for t in q.replace("?", " ").split() if t]
    if not tokens:
        return []

    entries = load_flags_kb()
    scored: List[tuple[int, Dict[str, Any]]] = []

    for e in entries:
        haystack = " ".join(
            [
                str(e.get("id") or ""),
                str(e.get("title") or ""),
                str(e.get("summary") or ""),
                " ".join(e.get("keywords") or []),
                " ".join(e.get("aliases") or []),
            ]
        ).lower()

        score = 0
        for t in tokens:
            if t in haystack:
                score += 1

        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for score, e in scored[:limit]]