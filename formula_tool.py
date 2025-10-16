
"""
formula_tool.py — Formula Card generator (公式 + 符号解释 + 常见单位 + 计算示例)

Features
--------
- Generates a structured "Formula Card" for a finance/investing term or metric.
- Optional RAG with a local FAISS vector store (teach_db/).
- Works with IBM watsonx (ChatWatsonx) if configured; otherwise accept any LangChain ChatModel.
- Exposes both a plain Python API and a LangChain StructuredTool for agent use.
- Optional built-in numeric example computation (safe-eval on a Pythonic expression).

Quick start
-----------
from formula_tool import generate_formula_card, get_formula_card_tool

# Direct call
card = generate_formula_card(
    term="CAGR",
    language="zh",
    example_inputs={"end": 150, "start": 100, "years": 3},
    eval_expr="(end/start)**(1/years)-1",   # Pythonic RHS; optional
)
print(card["card"])

# As a tool
tool = get_formula_card_tool()
# Add `tool` to your agent tools

Return schema
-------------
{
  "card": {
    "term": str,
    "formulae": [
      {
        "name": str,
        "expr": str,                  # LaTeX或纯文本表达式
        "format": "latex"|"plain",
        "symbols": [{"symbol": str, "meaning": str, "units": [str]}]
      }
    ],
    "units_common": [str],
    "worked_example": {
      "inputs": {"var": number},
      "steps": [str, ...],
      "final_formula": str,
      "result_explain": str,
      "computed_result": number|null
    },
    "citations": [{"title": str, "url": str, "source": str}],
    "lang": "zh"|"en"
  },
  "meta": {"used_rag": bool, "retrieved": int, "note"?: str}
}
"""

from __future__ import annotations
import os
import json
import math
import ast
from typing import Any, Dict, List, Optional

# -------- Optional dependencies (gracefully optional) --------
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


# ----------------- LLM builder (optional watsonx) -----------------
def _get_project_id() -> Optional[str]:
    for key in ("PROJ_ID", "PROJECT_ID", "WATSONX_PROJECT_ID"):
        v = os.getenv(key)
        if v:
            return v
    return None

def build_default_llm():
    """Build a default ChatModel if available (IBM watsonx)."""
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


# ----------------- Retrieval helpers (optional) -----------------
def _load_teach_vs(teach_db_dir: Optional[str] = None):
    """Try to load FAISS vector store (./teach_db)."""
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


# ----------------- Safe numeric evaluator for examples -----------------
_ALLOWED_FUNCS = {
    'sqrt': math.sqrt,
    'log': math.log,
    'ln': math.log,
    'exp': math.exp,
    'pow': math.pow,
    'abs': abs,
    'min': min,
    'max': max,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Call, ast.Constant, ast.FloorDiv
)

def _safe_eval(expr: str, variables: Dict[str, float]) -> Optional[float]:
    """
    Evaluate a simple Pythonic expression with a whitelist of nodes and functions.
    Only variable names present in `variables` and allowed functions can be used.
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except Exception:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_FUNCS:
                    return None
            else:
                return None
        if isinstance(node, ast.Name):
            if node.id not in variables and node.id not in _ALLOWED_FUNCS:
                return None

    env = dict(_ALLOWED_FUNCS)
    env.update(variables)
    try:
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env))
    except Exception:
        return None

def _rhs(expr: str) -> str:
    """Extract right-hand side if expression contains '=', else return expr."""
    if "=" in expr:
        return expr.split("=", 1)[1].strip()
    return expr.strip()


# ----------------- Core API -----------------
def generate_formula_card(
    term: str,
    *,
    language: str = "zh",
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    preferred_format: str = "latex",          # or "plain"
    example_inputs: Optional[Dict[str, float]] = None,
    eval_expr: Optional[str] = None,          # Pythonic RHS for computation, e.g., "(end/start)**(1/years)-1"
) -> Dict[str, Any]:
    """
    Generate a Formula Card (公式 + 符号解释 + 常见单位 + 计算示例).
    If FAISS teach_db is available, uses RAG; otherwise zero-shot.

    Args:
      term: formula/metric name, e.g., "CAGR", "Sharpe Ratio", "Beta".
      language: "zh" or "en".
      preferred_format: desired formula display format: "latex" or "plain".
      example_inputs: optional dict of numbers for a worked example.
      eval_expr: optional Pythonic expression to compute numeric result (RHS).
    """
    term = (term or "").strip()
    if not term:
        return {
            "card": {
                "term": "",
                "formulae": [],
                "units_common": [],
                "worked_example": {"inputs": {}, "steps": [], "final_formula": "", "result_explain": "", "computed_result": None},
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
    fmt = preferred_format if preferred_format in {"latex", "plain"} else "latex"

    sys_text = (
        "你是一名投资教学助教。基于上下文，用指定语言生成严格JSON的“公式卡（Formula Card）”。"
        if lang.startswith("zh") else
        "You are a finance tutor. Using the context, produce a strict-JSON Formula Card."
    )
    if lang.startswith("zh"):
        schema = (
            "```json\n{\n  \"card\": {\n    \"term\": \"...\","
            f"\n    \"formulae\": [{{\"name\":\"主公式\",\"expr\":\"...\",\"format\":\"{fmt}\","
            "\n      \"symbols\":[{\"symbol\":\"x\",\"meaning\":\"...\",\"units\":[\"...\"]}]}],"
            "\n    \"units_common\": [\"...\"],\n"
            "    \"worked_example\": {\"inputs\": {\"var\": 0}, \"steps\": [\"...\"],"
            " \"final_formula\": \"...\", \"result_explain\": \"<=60字\", \"computed_result\": null},\n"
            "    \"citations\": [{\"title\":\"\",\"url\":\"\",\"source\":\"\"}],\n"
            "    \"lang\": \"zh\"\n  }\n}\n```"
        )
    else:
        schema = (
            "```json\n{\n  \"card\": {\n    \"term\": \"...\","
            f"\n    \"formulae\": [{{\"name\":\"Main Formula\",\"expr\":\"...\",\"format\":\"{fmt}\","
            "\n      \"symbols\":[{\"symbol\":\"x\",\"meaning\":\"...\",\"units\":[\"...\"]}]}],"
            "\n    \"units_common\": [\"...\"],\n"
            "    \"worked_example\": {\"inputs\": {\"var\": 0}, \"steps\": [\"...\"],"
            " \"final_formula\": \"...\", \"result_explain\": \"<=40 words\", \"computed_result\": null},\n"
            "    \"citations\": [{\"title\":\"\",\"url\":\"\",\"source\":\"\"}],\n"
            "    \"lang\": \"en\"\n  }\n}\n```"
        )

    human_text = (
        f"【术语】{term}\n【上下文】\n{context}\n"
        f"【格式倾向】{fmt}\n"
        f"【示例输入】{json.dumps(example_inputs or {}, ensure_ascii=False)}\n"
        "必须只输出一个 ```json fenced block```，严禁编造事实/URL；如无URL留空。"
        if lang.startswith("zh") else
        f"[TERM] {term}\n[CONTEXT]\n{context}\n"
        f"[FORMAT] {fmt}\n"
        f"[EXAMPLE_INPUTS] {json.dumps(example_inputs or {})}\n"
        "Return only one ```json fenced block```. Do not fabricate; leave URL empty."
    )

    # ---- LLM path ----
    if (chat is not None) and _LC_CORE_AVAILABLE:
        msgs = [SystemMessage(content=sys_text + "\n" + schema), HumanMessage(content=human_text)]
        ai = chat.invoke(msgs)
        raw = getattr(ai, "content", "") or ""
        parsed = _extract_json(raw) or {}
        card = parsed.get("card", {})
        if not isinstance(card, dict):
            card = {}

        # Attach defaults & citations
        card.setdefault("term", term)
        card.setdefault("formulae", [])
        card.setdefault("units_common", [])
        card.setdefault("worked_example", {"inputs": example_inputs or {}, "steps": [], "final_formula": "", "result_explain": "", "computed_result": None})
        card.setdefault("citations", citations)
        card.setdefault("lang", "zh" if lang.startswith("zh") else "en")
        if not card.get("citations"):
            card["citations"] = citations

        # Optional numeric computation
        if example_inputs and (eval_expr or (card.get("worked_example") or {}).get("final_formula")):
            expr_src = eval_expr or (card.get("worked_example") or {}).get("final_formula") or ""
            rhs = _rhs(expr_src)
            num = _safe_eval(rhs, {k: float(v) for k, v in (example_inputs or {}).items()})
            try:
                card["worked_example"]["computed_result"] = float(num) if num is not None else None
            except Exception:
                pass

        return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs)}}

    # ---- Fallback path (no LLM) ----
    # Build a skeleton and compute numeric example if possible.
    skeleton = {
        "term": term,
        "formulae": [{
            "name": "主公式" if lang.startswith("zh") else "Main Formula",
            "expr": "示例：CAGR = (end/start)**(1/years)-1" if lang.startswith("zh") else "Example: CAGR = (end/start)**(1/years)-1",
            "format": fmt,
            "symbols": [
                {"symbol": "end", "meaning": "期末值" if lang.startswith("zh") else "Ending value", "units": []},
                {"symbol": "start", "meaning": "期初值" if lang.startswith("zh") else "Starting value", "units": []},
                {"symbol": "years", "meaning": "年数" if lang.startswith("zh") else "Years", "units": []},
            ],
        }],
        "units_common": ["%"] if lang.startswith("zh") else ["percent"],
        "worked_example": {
            "inputs": example_inputs or {},
            "steps": [],
            "final_formula": eval_expr or "(end/start)**(1/years)-1",
            "result_explain": "数值为示例，需以LLM/文档校对。" if lang.startswith("zh") else "Illustrative only; validate with docs/LLM.",
            "computed_result": None,
        },
        "citations": citations,
        "lang": "zh" if lang.startswith("zh") else "en",
    }

    if example_inputs and (eval_expr or skeleton["worked_example"]["final_formula"]):
        rhs = _rhs(eval_expr or skeleton["worked_example"]["final_formula"])
        num = _safe_eval(rhs, {k: float(v) for k, v in (example_inputs or {}).items()})
        skeleton["worked_example"]["computed_result"] = float(num) if num is not None else None

    return {"card": skeleton, "meta": {"used_rag": bool(docs), "retrieved": len(docs), "note": "no_llm_fallback"}}


# ----------------- LangChain Tool factory -----------------
def get_formula_card_tool(
    *,
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    name: str = "formula_card",
    description: str = "生成公式卡（公式+符号解释+常见单位+计算示例）。输入: term:str, language?:'zh'|'en', example_json?:str。",
):
    """
    Returns a LangChain StructuredTool wrapping generate_formula_card.
    To pass example inputs via agents, supply example_json like:
      {"example_inputs": {"end":150, "start":100, "years":3}, "eval_expr": "(end/start)**(1/years)-1", "preferred_format":"plain"}
    """
    def _tool(term: str, language: str = "zh", example_json: Optional[str] = None) -> str:
        ex = {}
        try:
            ex = json.loads(example_json) if example_json else {}
        except Exception:
            ex = {}
        res = generate_formula_card(
            term=term,
            language=language,
            teach_db_dir=teach_db_dir,
            llm=llm,
            preferred_format=ex.get("preferred_format", "latex"),
            example_inputs=ex.get("example_inputs"),
            eval_expr=ex.get("eval_expr"),
        )
        return json.dumps(res, ensure_ascii=False)

    if StructuredTool is None:
        class _CallableTool:
            __name__ = name
            def __call__(self, term: str, language: str = "zh", example_json: Optional[str] = None):
                return _tool(term, language, example_json)
        return _CallableTool()

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )
