
"""
worked_example_tool.py — Worked Example generator
用“真实标的/指标”走一遍：下载价格（可选）、计算指标、生成结构化步骤

功能亮点
--------
- 针对常见财金指标提供内置计算（beta / 年化波动 ann_vol / 最大回撤 mdd / CAGR / Sharpe）。
- 可选 RAG：若本地存在 teach_db/（FAISS + sentence-transformers），会注入上下文并供 LLM写更好的讲解。
- LLM 可选：可接入任意 LangChain ChatModel；若未提供且已配置 IBM watsonx，会自动使用；否则走可复现的“无模型”输出。
- 既提供 Python API，也提供 LangChain StructuredTool，方便大模型“调用工具”。

快速上手
--------
from worked_example_tool import generate_worked_example, get_worked_example_tool

# 1) 直接函数调用（示例：AAPL 对 SPY 计算 6 个月 β）
res = generate_worked_example(
    metric="beta", ticker="AAPL", benchmark="SPY",
    period="6mo", interval="1d", language="zh"
)
print(res["card"])

# 2) 作为 LangChain 工具
tool = get_worked_example_tool()
# 然后把 tool 加入你的 agent 工具列表即可

返回结构（JSON）
---------------
{
  "card": {
    "title": "Worked Example: Beta (AAPL vs SPY, 6mo, 1d)",
    "metric": "beta",
    "ticker": "AAPL",
    "benchmark": "SPY",
    "period": "6mo",
    "interval": "1d",
    "inputs": {"rows": 126, "rf": 0.0},
    "formula": "β = Cov(R_i, R_m) / Var(R_m)",
    "steps": ["下载复权价...", "对齐日收益...", "..."],
    "result": 1.23,
    "units": "",
    "notes": ["..."],
    "citations": [{"title":"...", "url":"...", "source":"..."}],
    "lang": "zh"
  },
  "meta": {"used_rag": true, "retrieved": 6, "used_prices": true, "rows": 126}
}
"""

from __future__ import annotations
import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple

# ---------- 可选依赖（均为优雅降级） ----------
try:
    import pandas as _pd  # type: ignore
    import numpy as _np   # type: ignore
    _PANDAS_OK = True
except Exception:
    _pd = None
    _np = None
    _PANDAS_OK = False

try:
    import yfinance as _yf  # type: ignore
    _YF_OK = True
except Exception:
    _yf = None
    _YF_OK = False

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


# ---------- teach_db 辅助（可选 RAG） ----------
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


# ---------- 价格抓取与指标计算 ----------
def _safe_download(ticker: str, period: str, interval: str):
    if not (_YF_OK and _PANDAS_OK):
        return None
    try:
        df = _yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, _pd.MultiIndex):
            # pick Close
            if ("Close" in df.columns.levels[0]) and (ticker in df["Close"].columns):
                s = df["Close"][ticker].dropna()
            else:
                s = df["Adj Close"][ticker].dropna() if ("Adj Close" in df.columns.levels[0]) else df.xs("Close", level=0, axis=1).iloc[:,0].dropna()
        else:
            s = df["Close"].dropna() if "Close" in df.columns else df.iloc[:,0].dropna()
        return s
    except Exception:
        return None

def _to_returns(price: "Optional[_pd.Series]"):
    if price is None or not _PANDAS_OK:
        return None
    r = price.pct_change().dropna()
    return r if len(r) > 3 else None

def _align(a: "Optional[_pd.Series]", b: "Optional[_pd.Series]"):
    if (a is None) or (b is None) or (not _PANDAS_OK):
        return None, None
    df = _pd.concat([a, b], axis=1, join="inner").dropna()
    if df.empty:
        return None, None
    return df.iloc[:,0], df.iloc[:,1]

def _ann_factor(interval: str) -> float:
    if interval.endswith("d"):
        return 252.0
    if interval.endswith("wk"):
        return 52.0
    if interval.endswith("mo"):
        return 12.0
    # fallback：按日
    return 252.0

def _calc_beta(r_i: "_pd.Series", r_m: "_pd.Series") -> Optional[float]:
    if not _PANDAS_OK:
        return None
    cov = _np.cov(r_i, r_m, ddof=1)
    var_m = _np.var(r_m, ddof=1)
    if var_m == 0:
        return None
    return float(cov[0,1] / var_m)

def _calc_ann_vol(r: "_pd.Series", interval: str) -> Optional[float]:
    if not _PANDAS_OK:
        return None
    af = _ann_factor(interval)
    return float(_np.std(r, ddof=1) * math.sqrt(af))

def _calc_mdd(price: "_pd.Series") -> Optional[float]:
    if not _PANDAS_OK:
        return None
    cummax = price.cummax()
    dd = (cummax - price) / cummax
    return float(dd.max())

def _calc_cagr(price: "_pd.Series") -> Optional[float]:
    if not _PANDAS_OK or price is None or price.empty:
        return None
    start = float(price.iloc[0]); end = float(price.iloc[-1])
    days = (price.index[-1] - price.index[0]).days
    years = max(days / 365.25, 1e-9)
    try:
        return float((end/start)**(1/years) - 1)
    except Exception:
        return None

def _calc_sharpe(r: "_pd.Series", interval: str, rf: float = 0.0) -> Optional[float]:
    if not _PANDAS_OK:
        return None
    af = _ann_factor(interval)
    mu_a = float(r.mean() * af)
    sigma_a = float(_np.std(r, ddof=1) * math.sqrt(af))
    if sigma_a == 0:
        return None
    return float((mu_a - rf) / sigma_a)


# ---------- 主函数 ----------
def generate_worked_example(
    *,
    metric: str,
    ticker: Optional[str] = None,
    benchmark: str = "SPY",
    period: str = "6mo",
    interval: str = "1d",
    language: str = "zh",
    rf: float = 0.0,
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    extra_inputs: Optional[Dict[str, float]] = None,  # 允许外部直接传指标输入（无行情时）
) -> Dict[str, Any]:
    """
    生成“案例卡 Worked Example”：用真实标的/指标走一遍。
    支持 metric: beta | ann_vol | mdd | cagr | sharpe

    - 若可用，会用 yfinance 拉取行情并计算指标；
    - 若无法拉取，可通过 extra_inputs 直接提供所需数值（例如 sharpe 需要 'mean_ann' 和 'vol_ann' 等）。

    返回见文件头部的 JSON 结构。
    """
    metric_key = (metric or "").lower().strip()
    lang = (language or "zh").lower()
    used_prices = False
    rows = 0
    notes: List[str] = []
    formula = ""
    units = ""
    result: Optional[float] = None
    inputs: Dict[str, float] = dict(extra_inputs or {})

    # 1) 拉上下文（可选）
    retriever, rag_ok = _load_teach_vs(teach_db_dir)
    docs = retriever.get_relevant_documents(metric_key) if (retriever and rag_ok) else []
    context = _format_docs_for_prompt(docs, max_chars=2800) if docs else ""
    citations = _collect_citations(docs, limit=5)

    # 2) 价格路径计算
    if _YF_OK and _PANDAS_OK and ticker:
        px = _safe_download(ticker, period, interval)
        ret = _to_returns(px)
        if px is not None:
            rows = len(px)
        if metric_key in {"beta"}:
            mkt_px = _safe_download(benchmark, period, interval)
            mkt_ret = _to_returns(mkt_px)
            if ret is not None and mkt_ret is not None:
                r_i, r_m = _align(ret, mkt_ret)
                if r_i is not None and r_m is not None and len(r_i) > 3:
                    result = _calc_beta(r_i, r_m)
                    used_prices = True
                    inputs.update({"rows": float(len(r_i))})
                    formula = "β = Cov(R_i, R_m) / Var(R_m)"
                    units = ""
                else:
                    notes.append("无法对齐收益序列，转为使用 extra_inputs（若提供）。")
        elif metric_key in {"ann_vol", "vol", "volatility"}:
            if ret is not None:
                result = _calc_ann_vol(ret, interval)
                used_prices = True
                inputs.update({"rows": float(len(ret))})
                formula = "σ_annual = σ_periodic × √AF"
                units = ""
        elif metric_key in {"mdd", "max_drawdown"}:
            if px is not None and len(px) > 3:
                result = _calc_mdd(px)
                used_prices = True
                inputs.update({"rows": float(len(px))})
                formula = "MDD = max( (Peak - Price) / Peak )"
                units = ""
        elif metric_key in {"cagr"}:
            if px is not None and len(px) > 3:
                result = _calc_cagr(px)
                used_prices = True
                inputs.update({"rows": float(len(px))})
                formula = "CAGR = (End/Start)^(1/Years) - 1"
                units = ""
        elif metric_key in {"sharpe"}:
            if ret is not None:
                result = _calc_sharpe(ret, interval, rf=rf)
                used_prices = True
                inputs.update({"rows": float(len(ret)), "rf": float(rf)})
                formula = "Sharpe = (μ_ann - r_f) / σ_ann"
                units = ""
        else:
            notes.append("未知指标；请使用支持的 metric 或通过 extra_inputs 直接提供。")
    else:
        if not _YF_OK:
            notes.append("yfinance 未安装或不可用。")
        if not _PANDAS_OK:
            notes.append("pandas/numpy 未安装或不可用。")

    # 3) 若没算出来，尝试用 extra_inputs 兜底
    if result is None and extra_inputs:
        # 仅做轻度兜底展示；不进行复杂计算
        notes.append("使用外部传入的 extra_inputs 构建示例（未进行行情计算）。")
        used_prices = False
        if metric_key == "sharpe" and {"mu_ann","sigma_ann"}.issubset(extra_inputs.keys()):
            mu = float(extra_inputs["mu_ann"]); sigma = float(extra_inputs["sigma_ann"])
            result = (mu - rf) / sigma if sigma != 0 else None
            formula = "Sharpe = (μ_ann - r_f) / σ_ann"
        elif metric_key == "cagr" and {"end","start","years"}.issubset(extra_inputs.keys()):
            end = float(extra_inputs["end"]); start = float(extra_inputs["start"]); years = float(extra_inputs["years"])
            result = (end/start)**(1/years) - 1 if (start>0 and years>0) else None
            formula = "CAGR = (End/Start)^(1/Years) - 1"
        elif metric_key == "ann_vol" and {"sigma_periodic","AF"}.issubset(extra_inputs.keys()):
            sigma_p = float(extra_inputs["sigma_periodic"]); AF = float(extra_inputs["AF"])
            result = sigma_p * math.sqrt(AF)
            formula = "σ_annual = σ_periodic × √AF"
        elif metric_key == "beta" and {"cov_im","var_m"}.issubset(extra_inputs.keys()):
            result = float(extra_inputs["cov_im"]) / float(extra_inputs["var_m"]) if float(extra_inputs["var_m"])!=0 else None
            formula = "β = Cov(R_i, R_m) / Var(R_m)"
        elif metric_key == "mdd" and {"peak","trough"}.issubset(extra_inputs.keys()):
            peak = float(extra_inputs["peak"]); trough = float(extra_inputs["trough"])
            result = (peak - trough) / peak if peak>0 else None
            formula = "MDD = (Peak - Trough) / Peak"

    # 4) 构造步骤（LLM 或 无模型版）
    title = (
        f"Worked Example: {metric_key.upper()} ({ticker or 'N/A'}"
        + (f" vs {benchmark}" if metric_key=='beta' else "")
        + f", {period}, {interval})"
    )
    steps: List[str] = []
    if used_prices:
        if metric_key == "beta":
            steps = [
                "下载标的与基准的复权收盘价（相同区间/频率）。",
                "计算日收益率，并按日期对齐两序列。",
                "计算协方差 Cov(R_i, R_m) 与方差 Var(R_m)，求 β = Cov/Var。",
                "解读：β>1 表示相对大盘波动被放大，β<1 表示更稳。",
            ]
            units = ""
        elif metric_key in {"ann_vol","vol","volatility"}:
            steps = [
                "下载标的复权收盘价并计算日收益率。",
                "计算收益率标准差 σ_periodic。",
                "按年化因子 AF（例如日频≈252）进行年化：σ_annual = σ_periodic × √AF。",
                "解读：年化波动越大，价格越“跳”，风险越高。",
            ]
            units = ""
        elif metric_key in {"mdd","max_drawdown"}:
            steps = [
                "下载标的复权收盘价。",
                "计算历史滚动峰值 Peak(t) 与当日价格 Price(t)。",
                "最大回撤 MDD = max_t (Peak(t) - Price(t)) / Peak(t)。",
                "解读：MDD 越大，历史最糟糕下跌越深。",
            ]
            units = ""
        elif metric_key == "cagr":
            steps = [
                "下载标的复权收盘价，取区间首末价 Start/End。",
                "计算年数 Years（按日期差/365.25）。",
                "CAGR = (End/Start)^(1/Years) - 1。",
                "解读：反映复合增长速度，平滑掉中间波动。",
            ]
            units = ""
        elif metric_key == "sharpe":
            steps = [
                "下载标的复权收盘价并计算日收益率。",
                "年化平均收益 μ_ann 与年化波动 σ_ann（按 AF 年化）。",
                "夏普比 Sharpe = (μ_ann - r_f) / σ_ann（此处 r_f 默认为 0）。",
                "解读：同等波动下，回报越高，夏普越高。",
            ]
            units = ""
    else:
        steps = [
            "未使用行情数据；改用外部输入/示例参数。",
            "代入公式并给出计算过程与结果。",
        ]

    # 5) LLM 编写自然语言解释（可选）
    chat = llm or build_default_llm()
    if (chat is not None) and _LC_CORE_AVAILABLE:
        sys_text = (
            "你是一名投资教学助教。根据上下文与已计算结果，生成严格JSON的“案例卡”："
            "```json\n{\n  \"card\": {\n    \"title\":\"...\", \"metric\":\"...\", \"ticker\":\"...\", \"benchmark\":\"...\", \"period\":\"...\", \"interval\":\"...\",\n"
            "    \"inputs\": {\"rows\":0}, \"formula\":\"...\", \"steps\":[\"...\"], \"result\": 0.0, \"units\":\"\", \"notes\":[\"...\"],\n"
            "    \"citations\":[{\"title\":\"\",\"url\":\"\",\"source\":\"\"}], \"lang\":\"zh\"\n  }\n}\n```"
            "只能输出一个 ```json fenced block```；不要杜撰事实/URL。"
            if lang.startswith("zh") else
            "You are a finance tutor. Using context and computed result, output a strict-JSON worked example card."
        )
        human_payload = {
            "context": context,
            "title": title,
            "metric": metric_key,
            "ticker": ticker,
            "benchmark": benchmark,
            "period": period,
            "interval": interval,
            "inputs": inputs,
            "formula": formula,
            "steps": steps,
            "result": result,
            "units": units,
            "notes": notes,
            "citations": citations,
            "lang": "zh" if lang.startswith("zh") else "en",
        }
        msgs = [SystemMessage(content=sys_text), HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))]
        try:
            ai = chat.invoke(msgs)
            raw = getattr(ai, "content", "") or ""
            parsed = _extract_json(raw)
            if isinstance(parsed, dict) and "card" in parsed:
                card = parsed["card"]
                # 补全缺失字段
                card.setdefault("citations", citations)
                card.setdefault("lang", "zh" if lang.startswith("zh") else "en")
                return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs), "used_prices": used_prices, "rows": rows}}
        except Exception:
            notes.append("LLM 解释失败，返回无模型版。")

    # 6) 无模型版卡片（可复现）
    card = {
        "title": title,
        "metric": metric_key,
        "ticker": ticker,
        "benchmark": benchmark if metric_key == "beta" else "",
        "period": period,
        "interval": interval,
        "inputs": inputs,
        "formula": formula,
        "steps": steps,
        "result": result,
        "units": units,
        "notes": notes,
        "citations": citations,
        "lang": "zh" if lang.startswith("zh") else "en",
    }
    return {"card": card, "meta": {"used_rag": bool(docs), "retrieved": len(docs), "used_prices": used_prices, "rows": rows}}


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


# ---------- LangChain 工厂 ----------
def get_worked_example_tool(
    *,
    teach_db_dir: Optional[str] = None,
    llm: Optional[Any] = None,
    name: str = "worked_example",
    description: str = "生成案例卡（用真实标的/指标走一遍）。输入 metric, ticker?, benchmark?, period?, interval?, rf?, extra_json?.",
):
    """
    返回一个 StructuredTool，方便大模型调用。
    传参示例：
      tool(metric="beta", ticker="AAPL", extra_json='{"benchmark":"SPY","period":"6mo","interval":"1d"}')
    """
    def _tool(metric: str, ticker: Optional[str] = None, extra_json: Optional[str] = None, language: str = "zh") -> str:
        ex = {}
        try:
            ex = json.loads(extra_json) if extra_json else {}
        except Exception:
            ex = {}
        res = generate_worked_example(
            metric=metric,
            ticker=ticker,
            benchmark=ex.get("benchmark", "SPY"),
            period=ex.get("period", "6mo"),
            interval=ex.get("interval", "1d"),
            language=language,
            rf=float(ex.get("rf", 0.0) or 0.0),
            teach_db_dir=teach_db_dir,
            llm=llm,
            extra_inputs=ex.get("extra_inputs"),
        )
        return json.dumps(res, ensure_ascii=False)

    if StructuredTool is None:
        class _CallableTool:
            __name__ = name
            def __call__(self, metric: str, ticker: Optional[str] = None, extra_json: Optional[str] = None, language: str = "zh"):
                return _tool(metric, ticker, extra_json, language)
        return _CallableTool()

    return StructuredTool.from_function(
        func=_tool,
        name=name,
        description=description,
    )
