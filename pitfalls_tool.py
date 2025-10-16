
"""
pitfalls_tool.py — Pitfalls Card generator
误区卡：常见误解 + 反例 + 自检清单 + 红旗 + 修正建议

功能
----
- 为某个术语/主题（如 “高收益=低风险” 或 “用CAGR评估极端波动资产”）生成结构化“误区卡”。
- 可选 RAG：若提供 teach_db/（FAISS + sentence-transformers），会检索上下文，辅助更贴近事实的示例/表述。
- 可选 LLM（如 IBM watsonx）；无 LLM 时返回可复现的骨架卡。
- 既提供 Python API，也提供 LangChain StructuredTool 以供代理调用。

返回结构
--------
{
  "card": {
    "term": "…",
    "common_misconceptions": [
      {"myth":"…","truth":"…","why_wrong":"…","quick_check":"…","source":""}
    ],
    "counterexamples": [
      {"title":"…","setup":"…","explain":"…","numbers":{"k":"v"}}
    ],
    "diagnostic_checks": ["…","…"],
    "red_flags": ["…","…"],
    "how_to_fix": ["…","…"],
    "citations": [{"title":"", "url":"", "source":""}],
    "lang": "zh" | "en"
  },
  "meta": {"used_rag": bool, "retrieved": int, "note"?: str}
}
"""

from __future__ import annotations
import os, json
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
        params={"decoding_method": "greedy", "max_new_tokens": 500, "temperature": 0.0},
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

def _format_docs_for_prompt(docs: List[Any], max_chars: int = 3000) -> str:
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
def generate_pitfalls_card(
    term: str,
    *,
    language: str = "zh",
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    max_misconceptions: int = 4,
    max_counterexamples: int = 3,
) -> Dict[str, Any]:
    """
    生成“误区卡（Pitfalls Card）”：常见误解与反例 + 自检清单 + 红旗 + 修正建议。
    支持 RAG；若无 LLM，则返回骨架卡（含通用投教项）。
    """
    term = (term or "").strip()
    if not term:
        return {
            "card": {
                "term": "",
                "common_misconceptions": [],
                "counterexamples": [],
                "diagnostic_checks": [],
                "red_flags": [],
                "how_to_fix": [],
                "citations": [],
                "lang": language,
            },
            "meta": {"used_rag": False, "retrieved": 0, "note": "empty_term"}
        }

    retriever, rag_ok = _load_teach_vs(teach_db_dir)
    docs = retriever.get_relevant_documents(term) if (retriever and rag_ok) else []
    context = _format_docs_for_prompt(docs, max_chars=3000) if docs else ""
    citations = _collect_citations(docs, limit=5)

    chat = llm or build_default_llm()
    lang = (language or "zh").lower()

    sys_text = (
        "你是投资教学助教。基于上下文为指定主题生成“误区卡（Pitfalls Card）”，严格输出 JSON："
        "```json\n{\n  \"card\": {\n    \"term\":\"...\",\n"
        "    \"common_misconceptions\":[{\"myth\":\"\",\"truth\":\"\",\"why_wrong\":\"\",\"quick_check\":\"\",\"source\":\"\"}],\n"
        "    \"counterexamples\":[{\"title\":\"\",\"setup\":\"\",\"explain\":\"\",\"numbers\":{}}],\n"
        "    \"diagnostic_checks\":[\"...\"],\n"
        "    \"red_flags\":[\"...\"],\n"
        "    \"how_to_fix\":[\"...\"],\n"
        "    \"citations\":[{\"title\":\"\",\"url\":\"\",\"source\":\"\"}],\n"
        "    \"lang\":\"zh\"\n  }\n}\n```"
        "不得编造事实或 URL；如无 URL 留空；每个字段尽量简洁。"
        if lang.startswith("zh") else
        "You are a finance tutor. Produce a strict-JSON 'Pitfalls Card' with myths, counterexamples, checks, red flags, fixes."
    )

    human_text = (
        f"【主题】{term}\n【上下文】\n{context}\n"
        f"【数量】myths≤{max_misconceptions}, counterexamples≤{max_counterexamples}\n"
        "仅返回一个 ```json fenced block```。"
        if lang.startswith("zh") else
        f"[TERM] {term}\n[CONTEXT]\n{context}\n[COUNTS] myths≤{max_misconceptions}, counterexamples≤{max_counterexamples}\nReturn one ```json fenced block```."
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
            # 附加默认与引用
            card.setdefault("term", term)
            card.setdefault("common_misconceptions", [])
            card.setdefault("counterexamples", [])
            card.setdefault("diagnostic_checks", [])
            card.setdefault("red_flags", [])
            card.setdefault("how_to_fix", [])
            card.setdefault("citations", citations)
            card.setdefault("lang", "zh" if lang.startswith("zh") else "en")
            if not card.get("citations"):
                card["citations"] = citations
            # 截断数量
            card["common_misconceptions"] = (card.get("common_misconceptions") or [])[:max_misconceptions]
            card["counterexamples"] = (card.get("counterexamples") or [])[:max_counterexamples]
            return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs)}}
        except Exception:
            pass

    # ---- 无 LLM 骨架（通用投教） ----
    skeleton_myths = [
        {"myth": "高收益=低风险", "truth": "更高收益通常伴随更高波动/回撤概率。", "why_wrong": "忽视风险溢价与尾部损失。", "quick_check": "对比年化波动/最大回撤；回看熊市表现。", "source": ""},
        {"myth": "历史年化回报可直接外推到未来", "truth": "分布可能非平稳；估值与宏观环境会变。", "why_wrong": "幸存者偏差与周期错配。", "quick_check": "看不同区间滚动回报；加入估值/利率背景。", "source": ""},
        {"myth": "分散持有越多越好", "truth": "超出边际后收益递减并增加跟踪复杂度。", "why_wrong": "相关性结构与交易成本被忽略。", "quick_check": "评估新增资产的相关性与边际贡献。", "source": ""},
        {"myth": "波动小=风险小", "truth": "低波动也可能伴随左尾风险（跳跃/流动性）。", "why_wrong": "仅关注σ而忽视偏度与尾部。", "quick_check": "查看回撤、偏度、极端日损与成交量枯竭。", "source": ""},
    ][:max_misconceptions]

    skeleton_counterexamples = [
        {"title": "高收益但高回撤的资产", "setup": "长期高回报但在单年出现深度下跌。", "explain": "说明收益与风险并非同向降低。", "numbers": {}},
        {"title": "低波动却有跳跃风险的产品", "setup": "日常波动小，但极端事件损失集中。", "explain": "仅看σ会低估尾部风险。", "numbers": {}},
        {"title": "分散化的边际收益递减", "setup": "新增资产与组合高度相关。", "explain": "相关性导致风险未显著下降。", "numbers": {}},
    ][:max_counterexamples]

    skeleton = {
        "term": term,
        "common_misconceptions": skeleton_myths,
        "counterexamples": skeleton_counterexamples,
        "diagnostic_checks": [
            "这是否依赖单一区间的历史回溯？",
            "是否只看均值/年化回报而忽略回撤与尾部？",
            "新增信息是否显著改变结论（敏感性分析）？",
            "是否存在数据挖掘或幸存者偏差？",
        ],
        "red_flags": [
            "过度承诺确定性或保证收益",
            "样本量过小或区间选择性",
            "忽略费用/滑点/税务影响",
            "把相关性当因果阐述",
        ],
        "how_to_fix": [
            "加入波动、最大回撤、偏度/峰度等多维风险指标。",
            "使用滚动窗口与多区间检验稳健性。",
            "纳入相关性与情景/压力测试。",
            "明确假设与局限，给出容错区间。",
        ],
        "citations": citations,
        "lang": "zh" if lang.startswith("zh") else "en",
    }

    return {"card": skeleton, "meta": {"used_rag": bool(docs), "retrieved": len(docs), "note": "no_llm_fallback"}}


# ---------- 工具工厂 ----------
def get_pitfalls_tool(
    *,
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    name: str = "pitfalls_card",
    description: str = "生成误区卡（常见误解、反例、自检、红旗、修正）。输入 term:str, language?:'zh'|'en', options_json?:str。",
):
    """
    返回 LangChain StructuredTool；options_json 可传：
      {"max_misconceptions": 4, "max_counterexamples": 3}
    """
    def _tool(term: str, language: str = "zh", options_json: Optional[str] = None) -> str:
        opts = {}
        try:
            opts = json.loads(options_json) if options_json else {}
        except Exception:
            opts = {}
        res = generate_pitfalls_card(
            term=term,
            language=language,
            teach_db_dir=teach_db_dir,
            llm=llm,
            max_misconceptions=int(opts.get("max_misconceptions", 4)),
            max_counterexamples=int(opts.get("max_counterexamples", 3)),
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
