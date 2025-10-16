
"""
quiz_tool.py — Quiz Card generator
测验卡：2–3 道单选/判断 + 简短解析（便于记忆巩固）

功能
----
- 围绕某个术语/主题（term）自动生成 2–3 道测验题（单选/判断），并附简短解析。
- 可选 RAG：若提供 teach_db/（FAISS + sentence-transformers），将检索上下文，帮助更贴近事实。
- 可选 LLM（如 IBM watsonx）；无 LLM 时返回可复现的“通用测验”骨架卡。
- 同时提供 Python API 与 LangChain StructuredTool，方便大模型“调用工具”。

返回结构
--------
{
  "card": {
    "term": "…",
    "questions": [
      {
        "type": "single" | "bool",
        "question": "…",
        "options": ["A","B","C","D"],     # 对于 type=single
        "answer": 0 | true | false,       # 单选为正确选项索引；判断为布尔
        "explanation": "简短解析（≤40字/≤30 words）"
      }
    ],
    "citations": [{"title":"", "url":"", "source":""}],
    "lang": "zh" | "en"
  },
  "meta": {"used_rag": bool, "retrieved": int, "note"?: str}
}
"""

from __future__ import annotations
import os, json, random
from typing import Any, Dict, List, Optional

# ---------- 可选依赖（优雅降级） ----------
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
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb  # type: ignore
        from langchain_community.vectorstores import FAISS  # type: ignore
    except Exception:
        _HFEmb = None  # type: ignore
        FAISS = None   # type: ignore

HuggingFaceEmbeddings = _HFEmb


# ---------- LLM 构造（可选 watsonx） ----------
def _get_project_id() -> Optional[str]:
    for key in ("PROJ_ID", "PROJECT_ID", "WATSONX_PROJECT_ID"):
        v = os.getenv(key)
        if v:
            return v
    return None

def build_default_llm():
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


# ---------- teach_db 检索（可选 RAG） ----------
def _load_teach_vs(teach_db_dir: Optional[str] = None):
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

def _format_docs_for_prompt(docs: List[Any], max_chars: int = 2800) -> str:
    chunks, used = [], 0
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
        chunks.append(block); used += len(block)
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


# ---------- JSON 抽取 ----------
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    import re
    try:
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            return json.loads(m.group(1))
    except Exception:
        return None
    return None


# ---------- 核心 API ----------
def generate_quiz_card(
    term: str,
    *,
    language: str = "zh",
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    max_questions: int = 3,
    prefer_types: Optional[List[str]] = None,    # 例如 ["single","bool"]
) -> Dict[str, Any]:
    """
    生成“测验卡（Quiz Card）”：2–3 道单选/判断，附简短解析。
    - 若提供 teach_db/，会检索上下文辅助出题；
    - 若无 LLM，会返回“通用测验”骨架卡（围绕学习方法和基本陷阱，避免杜撰事实）。
    """
    term = (term or "").strip()
    if not term:
        return {
            "card": {"term":"", "questions":[], "citations":[], "lang": language},
            "meta": {"used_rag": False, "retrieved": 0, "note": "empty_term"}
        }

    prefer_types = prefer_types or ["single", "bool"]
    max_questions = max(2, min(3, int(max_questions or 3)))

    retriever, rag_ok = _load_teach_vs(teach_db_dir)
    docs = retriever.get_relevant_documents(term) if (retriever and rag_ok) else []
    context = _format_docs_for_prompt(docs, max_chars=2800) if docs else ""
    citations = _collect_citations(docs, limit=5)

    chat = llm or build_default_llm()
    lang = (language or "zh").lower()

    sys_text = (
        "你是投资教学助教。基于上下文为指定主题生成严格JSON的“测验卡”："
        "题目 2–3 道；类型仅限 single(单选，4项且唯一正确) 或 bool(判断)。"
        "答案：单选用正确选项索引（0-3），判断用 true/false。"
        "解析须 ≤40字，强调记忆点。禁止编造事实/URL。"
        "JSON 结构示例："
        "```json\n{\n  \"card\": {\n    \"term\":\"...\",\n    \"questions\":[{\"type\":\"single\",\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":2,\"explanation\":\"...\"}],\n    \"citations\":[{\"title\":\"\",\"url\":\"\",\"source\":\"\"}],\n    \"lang\":\"zh\"\n  }\n}\n```"
        if lang.startswith("zh") else
        "You are a finance tutor. Produce a strict-JSON quiz card with 2–3 questions (single-choice or true/false). "
        "Single-choice has 4 options and a single correct index (0-3). Bool uses true/false. "
        "Explanations ≤30 words, focus on memory hooks. No fabrication/URLs."
    )
    human_text = (
        f"【主题】{term}\n【上下文】\n{context}\n【题量】≤{max_questions}\n【类型优先】{','.join(prefer_types)}\n"
        "仅返回一个 ```json fenced block```。"
        if lang.startswith("zh") else
        f"[TERM] {term}\n[CONTEXT]\n{context}\n[MAX_QUESTIONS] {max_questions}\n[PREFERRED_TYPES] {','.join(prefer_types)}\nReturn one ```json fenced block```."
    )

    if (chat is not None) and _LC_CORE_AVAILABLE:
        msgs = [SystemMessage(content=sys_text), HumanMessage(content=human_text)]
        try:
            ai = chat.invoke(msgs)
            raw = getattr(ai, "content", "") or ""
            parsed = _extract_json(raw) or {}
            card = parsed.get("card", {})
            if not isinstance(card, dict):
                card = {}
            # 兜底字段
            card.setdefault("term", term)
            qs = card.get("questions") or []
            # 只留 2–3 道
            card["questions"] = qs[:max_questions]
            card.setdefault("citations", citations)
            card.setdefault("lang", "zh" if lang.startswith("zh") else "en")
            return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs)}}
        except Exception:
            pass

    # ---- 无 LLM 骨架：通用但不涉具体事实 ----
    zh = lang.startswith("zh")
    if zh:
        skeleton = [
            {
                "type": "bool",
                "question": f"判断：精确定义与直觉类比都有助于掌握“{term}”。",
                "answer": True,
                "explanation": "双轨记忆：先准确定义，再用类比巩固。",
            },
            {
                "type": "single",
                "question": f"关于“{term}”，哪项做法最能避免误解？",
                "options": ["只记公式", "结合反例", "只看单一区间", "忽略单位"],
                "answer": 1,
                "explanation": "反例能暴露边界条件，防止过度泛化。",
            },
            {
                "type": "single",
                "question": "学习效果最可能提升的是哪一项？",
                "options": ["定义", "直觉", "为何重要", "以上皆是"],
                "answer": 3,
                "explanation": "三件套能构建完整语义网络。",
            },
        ]
    else:
        skeleton = [
            {
                "type": "bool",
                "question": f"True/False: Both precise definition and intuition help learn '{term}'.",
                "answer": True,
                "explanation": "Dual coding: definition + analogy boosts recall.",
            },
            {
                "type": "single",
                "question": f"For '{term}', which helps avoid misconceptions most?",
                "options": ["Memorize formula only", "Study counterexamples", "Use a single time window", "Ignore units"],
                "answer": 1,
                "explanation": "Counterexamples reveal edge cases.",
            },
            {
                "type": "single",
                "question": "Which improves retention the most?",
                "options": ["Definition", "Intuition", "Why it matters", "All of the above"],
                "answer": 3,
                "explanation": "The trio forms a complete schema.",
            },
        ]

    card = {
        "term": term,
        "questions": skeleton[:max_questions],
        "citations": citations,
        "lang": "zh" if zh else "en",
    }
    return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs), "note": "no_llm_fallback"}}


# ---------- LangChain 工具工厂 ----------
def get_quiz_tool(
    *,
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    name: str = "quiz_card",
    description: str = "生成测验卡（2–3道单选/判断，附简短解析）。输入 term:str, language?:'zh'|'en', options_json?:str。",
):
    """
    返回 LangChain StructuredTool；options_json 可传：
      {"max_questions": 3, "prefer_types": ["single","bool"]}
    """
    def _tool(term: str, language: str = "zh", options_json: Optional[str] = None) -> str:
        opts = {}
        try:
            opts = json.loads(options_json) if options_json else {}
        except Exception:
            opts = {}
        res = generate_quiz_card(
            term=term,
            language=language,
            teach_db_dir=teach_db_dir,
            llm=llm,
            max_questions=int(opts.get("max_questions", 3)),
            prefer_types=opts.get("prefer_types") or ["single","bool"],
        )
        return json.dumps(res, ensure_ascii=False)

    if StructuredTool is None:
        class _CallableTool:
            __name__ = name
            def __call__(self, term: str, language: str = "zh", options_json: Optional[str] = None):
                return _tool(term, language, options_json)
        return _CallableTool()

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )
