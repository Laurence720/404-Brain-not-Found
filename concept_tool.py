
"""
concept_tool.py — Concept Card generator (Definition + Intuition + Why it matters)

Features
--------
- Generates a structured "Concept Card" for a finance/investing term.
- Can use a local FAISS vector store (teach_db/) for retrieval-augmented generation (RAG).
- Works with IBM watsonx (ChatWatsonx) if configured; otherwise lets you pass any LangChain-compatible ChatModel.
- Exposes both a plain Python API and a LangChain StructuredTool for agent use.

Quick start
-----------
from concept_tool import generate_concept_card, get_concept_card_tool

# 1) Direct function call
card = generate_concept_card(term="Beta", language="zh")
print(card["card"])

# 2) As a LangChain tool (for an agent)
tool = get_concept_card_tool()
# Then add `tool` into your agent/toolset

Environment (optional)
----------------------
# For IBM watsonx (if you don't pass an llm yourself):
PROJ_ID / PROJECT_ID / WATSONX_PROJECT_ID
WATSONX_CHAT_MODEL=ibm/granite-3-2-8b-instruct

Dependencies (optional)
-----------------------
langchain-ibm, langchain-core, langchain-community, sentence-transformers, faiss-cpu
If FAISS/embeddings are missing, the tool will still return a card using only your prompt (no RAG).

Return schema
-------------
{
  "card": {
    "term": str,
    "definition": str,
    "intuition": str,
    "why_it_matters": str,
    "keywords": [str, ...],
    "citations": [{"title": str, "url": str, "source": str}],
    "lang": "zh" | "en"
  },
  "meta": {
    "used_rag": bool,
    "retrieved": int
  }
}
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional, Callable

# -------- Optional dependencies (all gracefully optional) --------
try:
    from langchain_ibm import ChatWatsonx  # type: ignore
    _WATSONX_AVAILABLE = True
except Exception:
    ChatWatsonx = None  # type: ignore
    _WATSONX_AVAILABLE = False

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    _LC_CORE_AVAILABLE = True
except Exception:
    SystemMessage = None  # type: ignore
    HumanMessage = None   # type: ignore
    StructuredTool = None # type: ignore
    _LC_CORE_AVAILABLE = False

try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb  # type: ignore
except Exception:
    try:
        # alt import path
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb  # type: ignore
        from langchain_community.vectorstores import FAISS  # type: ignore
    except Exception:
        _HFEmb = None  # type: ignore
        FAISS = None   # type: ignore

HuggingFaceEmbeddings = _HFEmb


# ----------------- LLM builder (optional watsonx) -----------------
def _get_project_id() -> Optional[str]:
    for key in ("PROJ_ID", "PROJECT_ID", "WATSONX_PROJECT_ID"):
        v = os.getenv(key)
        if v:
            return v
    return None

def build_default_llm():
    """
    Build a default ChatModel if available (IBM watsonx). You can pass your own llm to `generate_concept_card`.
    """
    if not _WATSONX_AVAILABLE:
        return None
    model_id = os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-2-8b-instruct")
    project_id = _get_project_id()
    if not project_id:
        return None
    return ChatWatsonx(
        model_id=model_id,
        project_id=project_id,
        params={"decoding_method": "greedy", "max_new_tokens": 400, "temperature": 0.0},
    )


# ----------------- Retrieval helpers (optional) -----------------
def _load_teach_vs(teach_db_dir: Optional[str] = None):
    """
    Try to load FAISS vector store teach_db (defaults to ./teach_db).
    Returns (retriever, used) where retriever has .get_relevant_documents(query)
    """
    base = teach_db_dir or os.path.join(os.path.dirname(__file__), "teach_db")
    if (FAISS is None) or (HuggingFaceEmbeddings is None):
        return None, False
    if not os.path.isdir(base):
        return None, False
    try:
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = FAISS.load_local(base, embeddings=embed, allow_dangerous_deserialization=True)
        retriever = vs.as_retriever(search_kwargs={"k": 8})
        return retriever, True
    except Exception:
        return None, False

def _format_docs_for_prompt(docs: List[Any], max_chars: int = 3000) -> str:
    """
    Turn retrieved docs into a compact string context.
    """
    chunks: List[str] = []
    used = 0
    for i, d in enumerate(docs or [], 1):
        text = str(getattr(d, "page_content", "") or "").strip()
        if not text:
            continue
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("url") or meta.get("title") or ""
        head = f"[{i}] {src}".strip() if src else f"[{i}]"
        snippet = (text[:800] + "...") if len(text) > 800 else text
        block = f"{head}\n{snippet}"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks)

def _collect_citations(docs: List[Any], limit: int = 5) -> List[Dict[str, str]]:
    cites = []
    for d in (docs or [])[:limit]:
        meta = getattr(d, "metadata", {}) or {}
        title = str(meta.get("title") or meta.get("source") or "")[:160]
        url = str(meta.get("url") or "")
        source = str(meta.get("source") or meta.get("site") or "")
        cites.append({"title": title, "url": url, "source": source})
    return cites


# ----------------- JSON extraction helper -----------------
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    import re, json as _json
    if not text:
        return None
    fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            return _json.loads(fence.group(1))
        except Exception:
            pass
    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        try:
            return _json.loads(brace.group(1))
        except Exception:
            pass
    return None


# ----------------- Core API -----------------
def generate_concept_card(
    term: str,
    *,
    user_question: Optional[str] = None,
    language: str = "zh",
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate a Concept Card (Definition + Intuition + Why it matters) as a dict.
    If FAISS teach_db is available, uses RAG; otherwise, falls back to zero-shot.

    Args:
      term: the term to explain, e.g., "Beta".
      user_question: optional additional question/context.
      language: "zh" or "en".
      teach_db_dir: optional path to a FAISS teach_db.
      llm: optional LangChain ChatModel; if None, tries watsonx via env.

    Returns:
      Dict with keys "card" and "meta" (see module docstring).
    """
    term = (term or "").strip()
    if not term:
        return {
            "card": {
                "term": "",
                "definition": "",
                "intuition": "",
                "why_it_matters": "",
                "keywords": [],
                "citations": [],
                "lang": language,
            },
            "meta": {"used_rag": False, "retrieved": 0, "note": "empty_term"}
        }

    retriever, rag_ok = _load_teach_vs(teach_db_dir)
    docs = retriever.get_relevant_documents(term) if (retriever and rag_ok) else []
    context = _format_docs_for_prompt(docs, max_chars=3000) if docs else ""
    citations = _collect_citations(docs, limit=5)

    # If no llm provided, try default watsonx
    chat = llm or build_default_llm()

    # Build prompt
    lang = (language or "zh").lower()
    sys_text = (
        "你是一名投资教学助教。基于提供的上下文，用指定语言生成严格JSON的“概念卡（Concept Card）”。"
        if lang.startswith("zh") else
        "You are a finance tutor. Using the given context, produce a strict-JSON Concept Card."
    )
    schema = (
        "```json\n{\n  \"card\": {\n    \"term\": \"...\"," 
        "\n    \"definition\": \"<=80字精确定义，不可编造\",\n    \"intuition\": \"<=80字直觉理解/类比\",\n"
        "    \"why_it_matters\": \"<=80字说明该概念的用途/影响\",\n"
        "    \"keywords\": [\"...\"],\n"
        "    \"citations\": [{\"title\":\"\",\"url\":\"\",\"source\":\"\"}],\n    \"lang\": \"zh\"\n  }\n}\n```"
        if lang.startswith("zh") else
        "```json\n{\n  \"card\": {\n    \"term\": \"...\"," 
        "\n    \"definition\": \"<=40 words precise definition\",\n    \"intuition\": \"<=40 words intuition/metaphor\",\n"
        "    \"why_it_matters\": \"<=50 words on relevance\",\n"
        "    \"keywords\": [\"...\"],\n"
        "    \"citations\": [{\"title\":\"\",\"url\":\"\",\"source\":\"\"}],\n    \"lang\": \"en\"\n  }\n}\n```"
    )

    human_text = (
        f"【术语】{term}\n"
        f"【追问】{user_question or ''}\n"
        f"【上下文】\n{context}\n"
        "必须只输出一个 ```json fenced block```，严禁编造事实和URL；如无URL留空。"
        if lang.startswith("zh") else
        f"[TERM] {term}\n[QUESTION] {user_question or ''}\n[CONTEXT]\n{context}\n"
        "Return only one ```json fenced block```. No fabrication; leave URL empty if unknown."
    )

    if (chat is not None) and _LC_CORE_AVAILABLE:
        # Run a single-shot LLM call
        msgs = [SystemMessage(content=sys_text + "\n" + schema), HumanMessage(content=human_text)]
        ai = chat.invoke(msgs)
        raw = getattr(ai, "content", "") or ""
        parsed = _extract_json(raw) or {}
        card = parsed.get("card", {})
        # Attach citations from retrieval if LLM leaves them blank
        if isinstance(card, dict):
            if not card.get("citations"):
                card["citations"] = citations
            if not card.get("lang"):
                card["lang"] = "zh" if lang.startswith("zh") else "en"
            if not card.get("term"):
                card["term"] = term
        return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs)}}

    # Fallback: no LLM available — return a skeleton using retrieved context hints
    return {
        "card": {
            "term": term,
            "definition": f"{'依据上下文的' if context else '通用的'}简要定义（需要LLM生成）",
            "intuition": "直觉/类比（需要LLM生成）" if lang.startswith("zh") else "Intuition/metaphor (needs LLM)",
            "why_it_matters": "为什么重要（需要LLM生成）" if lang.startswith("zh") else "Why it matters (needs LLM)",
            "keywords": [],
            "citations": citations,
            "lang": "zh" if lang.startswith("zh") else "en",
        },
        "meta": {"used_rag": bool(docs), "retrieved": len(docs), "note": "no_llm_fallback"}
    }


# ----------------- LangChain Tool factory -----------------
def get_concept_card_tool(
    *,
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    name: str = "concept_card",
    description: str = "生成概念卡（定义+直觉+为何重要）。输入: term:str, user_question?:str, language?:'zh'|'en'. 返回严格JSON。",
):
    """
    Returns a LangChain StructuredTool wrapping generate_concept_card for agent use.
    If LangChain is not available, returns a simple callable with the same signature.
    """
    def _tool(term: str, user_question: Optional[str] = None, language: str = "zh") -> str:
        res = generate_concept_card(
            term=term, user_question=user_question, language=language,
            teach_db_dir=teach_db_dir, llm=llm
        )
        # Return as JSON string (agents expect text)
        return json.dumps(res, ensure_ascii=False)

    if StructuredTool is None:
        # Fallback simple object
        class _CallableTool:
            __name__ = name
            def __call__(self, term: str, user_question: Optional[str] = None, language: str = "zh"):
                return _tool(term, user_question, language)
        return _CallableTool()

    # With LangChain: define as StructuredTool (simple arg signature)
    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )
