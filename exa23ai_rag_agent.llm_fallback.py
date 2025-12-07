from __future__ import annotations
import os, oracledb
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import SystemMessage, HumanMessage

_CONN: Optional[oracledb.Connection] = None
_EMB: Optional[OracleEmbeddings] = None
_VS: Optional[OracleVS] = None
_RETRIEVER = None
_QA = None
_LLM: Optional[ChatOCIGenAI] = None
_CONFIG: Dict[str, Any] = {}

_DEFAULT_TABLE = "SLACKBOT_VECTORS"
_DEFAULT_MODEL = "ALL_MINILM_L12_V2"


def _make_rag_llm() -> ChatOCIGenAI:
    explicit_provider = (
        os.getenv("RAG_OCI_PROVIDER")
        or os.getenv("OCI_RAG_PROVIDER")
        or os.getenv("OCI_GENAI_PROVIDER")
        or ""
    )
    model_id = (
        os.getenv("RAG_OCI_MODEL_ID")
        or os.getenv("OCI_RAG_MODEL_ID")
        or os.getenv("OCI_GENAI_MODEL_ID")
    )
    if not model_id:
        raise RuntimeError("Missing RAG_OCI_MODEL_ID / OCI_GENAI_MODEL_ID for RAG LLM.")

    provider = explicit_provider.strip().lower()
    if not provider:
        mid = model_id.lower()
        if mid.startswith("cohere"):
            provider = "cohere"
        elif mid.startswith("xai"):
            provider = "meta"  # mimic ExaCopilot fallback
        elif mid.startswith("openai"):
            provider = "meta"  # OCI currently maps OpenAI models via meta bridge
        else:
            provider = "meta"

    endpoint = (
        os.getenv("RAG_OCI_ENDPOINT")
        or os.getenv("OCI_RAG_ENDPOINT")
        or os.getenv("OCI_GENAI_ENDPOINT")
    )
    if not endpoint:
        raise RuntimeError("Missing RAG_OCI_ENDPOINT / OCI_GENAI_ENDPOINT for RAG LLM.")

    compartment = (
        os.getenv("RAG_OCI_COMPARTMENT_ID")
        or os.getenv("OCI_RAG_COMPARTMENT_ID")
        or os.getenv("OCI_COMPARTMENT_ID")
    )
    auth_type = os.getenv("RAG_OCI_AUTH_TYPE") or os.getenv("OCI_AUTH_TYPE", "API_KEY")
    auth_profile = (
        os.getenv("RAG_OCI_CONFIG_PROFILE")
        or os.getenv("OCI_RAG_CONFIG_PROFILE")
        or os.getenv("OCI_CONFIG_PROFILE")
        or "DEFAULT"
    )

    temperature = float(
        os.getenv("RAG_OCI_TEMPERATURE", os.getenv("OCI_TEMPERATURE", "1"))
    )
    max_tokens = int(
        os.getenv("RAG_OCI_MAX_TOKENS", os.getenv("OCI_MAX_TOKENS", "2048"))
    )

    model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    print(
        "[RAG] Using OCI chat model:",
        f"provider={provider}",
        f"model_id={model_id}",
        f"endpoint={endpoint}",
        f"compartment={compartment}",
    )

    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment,
        provider=provider,
        auth_type=auth_type,
        auth_profile=auth_profile,
        model_kwargs=model_kwargs,
    )

def init_once() -> Dict[str, Any]:
    global _CONN, _EMB, _VS, _RETRIEVER, _QA, _LLM, _CONFIG
    if _QA is not None:
        return {"ok": True, "note": "already initialized"}

    user = os.getenv("ORA_USER")
    password = os.getenv("ORA_PASSWORD")
    dsn = os.getenv("ORA_DSN")
    table = os.getenv("ORA_TABLE", _DEFAULT_TABLE)
    model_name = os.getenv("ORA_MODEL_NAME", _DEFAULT_MODEL)
    rag_model_id = (
        os.getenv("RAG_OCI_MODEL_ID")
        or os.getenv("OCI_RAG_MODEL_ID")
        or os.getenv("OCI_GENAI_MODEL_ID")
    )
    rag_provider = (
        os.getenv("RAG_OCI_PROVIDER")
        or os.getenv("OCI_RAG_PROVIDER")
        or os.getenv("OCI_GENAI_PROVIDER")
        or "cohere"
    )

    _CONFIG = {
        "user": user,
        "dsn": dsn,
        "table": table,
        "model_name": model_name,
        "rag_model": rag_model_id,
        "rag_provider": rag_provider,
    }

    # Debug: print key RAG configuration (safe fields only)
    try:
        print("[RAG] Config:", {"table": table, "embed_model": model_name, "rag_provider": rag_provider, "rag_model": rag_model_id})
    except Exception:
        pass

    if not user or not password or not dsn:
        raise RuntimeError("Missing ORA_USER/ORA_PASSWORD/ORA_DSN.")

    try:
        _CONN = oracledb.connect(user=user, password=password, dsn=dsn)
        try:
            print("[RAG] Oracle connected:", bool(_CONN), "table=", table, "embed_model=", model_name)
        except Exception:
            pass
    except Exception as exc:
        # Do NOT raise — allow LLM-only mode when DB is unavailable
        print(f"[RAG] Oracle connection failed; continuing with LLM-only mode: {exc}")
        _CONN = None

    if _CONN:
        _EMB = OracleEmbeddings(conn=_CONN, params={"provider": "database", "model": model_name})
        _VS = OracleVS(client=_CONN, table_name=table, embedding_function=_EMB,
                       distance_strategy=DistanceStrategy.COSINE)
        _RETRIEVER = _VS.as_retriever(search_kwargs={"k": 3})
    else:
        _EMB = None
        _VS = None
        _RETRIEVER = None

    _LLM = _make_rag_llm()
    print("[RAG] LLM created:", type(_LLM).__name__, "provider=", _CONFIG.get("rag_provider"))
    if _RETRIEVER:
        _QA = RetrievalQA.from_chain_type(llm=_LLM, chain_type="stuff",
                                          retriever=_RETRIEVER, return_source_documents=True)
    else:
        _QA = None

    try:
        print("[RAG] Mode:", "qa" if _QA else ("llm" if _LLM else "off"))
    except Exception:
        pass

    try:
        if _QA:
            _ = _QA.invoke({"query": "health check"})
        elif _LLM:
            _ = _LLM.invoke("health check")
    except Exception as e:
        print(f"[RAG] Warning: health probe failed: {e}")
        return {"ok": True, "warning": f"health probe failed: {e}"}
    return {"ok": True}

def health() -> Dict[str, Any]:
    return {
        "oracle_connected": bool(_CONN),
        "table": _CONFIG.get("table", _DEFAULT_TABLE),
        "embed_model": _CONFIG.get("model_name", _DEFAULT_MODEL),
        "initialized": bool(_QA),
        "llm_ready": bool(_LLM),
        "mode": ("qa" if _QA else ("llm" if _LLM else "off")),
        "llm_provider": _CONFIG.get("rag_provider"),
        "oci_model_id": _CONFIG.get("rag_model"),
    }

def query(question: str, k: int = 3) -> Dict[str, Any]:
    if not question or not question.strip():
        return {"error": "empty question"}
    if _QA is None:
        init_once()
    if _RETRIEVER and k:
        _RETRIEVER.search_kwargs["k"] = int(k)

    try:
        print("[RAG] Query run:", {"k": _RETRIEVER.search_kwargs.get("k") if _RETRIEVER else k, "rag_model": _CONFIG.get("rag_model"), "rag_provider": _CONFIG.get("rag_provider")})
    except Exception:
        pass

    if _QA:
        result = _QA.invoke({"query": question})
        answer = result.get("result", "") or ""
        sources = result.get("source_documents", []) or []
        src: List[Dict[str, Any]] = []
        for d in sources:
            meta = d.metadata or {}
            src.append({
                "title": meta.get("title") or os.path.basename(meta.get("source","") or "") or "untitled",
                "source": meta.get("source"),
                "score": meta.get("score"),
                "chunk_preview": (d.page_content or "")[:300]
            })
        try:
            print("[RAG] query: mode=qa answer_len=", len(answer))
        except Exception:
            pass
        return {"answer": answer, "sources": src}
    elif _LLM:
        try:
            sys_prompt = (
                "You are an Exadata/Exascale assistant. The vector database is currently unavailable, "
                "so you must answer from general knowledge. Provide a concise resolution with concrete steps, "
                "probable causes, and commands to verify or remediate. If the issue is environment-specific, "
                "state assumptions and give best-practice guidance. Do not return an empty answer."
            )
            messages = [SystemMessage(content=sys_prompt), HumanMessage(content=question)]
            resp = _LLM.invoke(messages)
            text = getattr(resp, "content", None)

            # Flatten LC structured content if needed
            if isinstance(text, list):
                try:
                    parts = []
                    for seg in text:
                        if isinstance(seg, dict):
                            parts.append(str(seg.get("text", "")))
                        else:
                            parts.append(str(seg))
                    text = "\n".join([p for p in parts if p])
                except Exception:
                    text = "".join(str(x) for x in text)

            if text is None:
                try:
                    text = resp.get("content")  # type: ignore[attr-defined]
                except Exception:
                    text = str(resp)

            # Debug: print response metadata (model id, request id)
            try:
                meta = getattr(resp, "response_metadata", {}) or {}
                print("[RAG] LLM-only response meta:", {
                    "model_id": meta.get("model_id"),
                    "request_id": meta.get("request_id"),
                    "finish_reason": meta.get("finish_reason") or meta.get("finishDetails")
                })
            except Exception:
                pass

            answer_text = (text or "").strip()
            if not answer_text:
                answer_text = (
                    "Here are probable causes and next steps:\n"
                    "- Validate ExaCTRL service reachability and credentials\n"
                    "- Restart ESCLI to re-establish ExaCTRL session\n"
                    "- Check network/DNS and service health on the control plane\n"
                    "- Review ExaCTRL logs for connection errors\n"
                    "Without the knowledge base online, this is a best-effort guidance."
                )
        except Exception as e:
            return {"error": f"LLM-only mode failed: {e}"}
        try:
            print("[RAG] query: mode=llm answer_len=", len(answer_text))
        except Exception:
            pass
        return {"answer": answer_text, "sources": []}
    else:
        return {"error": "RAG not initialized: no retriever and no LLM"}
