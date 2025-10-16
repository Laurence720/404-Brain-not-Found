from __future__ import annotations

import os
import sys
import re
import json
import time
import inspect
import threading
import builtins as _builtins
from typing import Any, Dict, List, Optional

# LangChain / LangGraph 相关
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# ✅ 向量库 & 向量模型（一定要分两行）
try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb  # type: ignore
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb  # type: ignore
    except Exception:
        _HFEmb = None  # type: ignore

try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:
    FAISS = None  # type: ignore

HuggingFaceEmbeddings = _HFEmb  # re-export for downstream usage
os.environ["TERMINAL_SIMPLE_DEFAULT"] = "1"  # Keep minimalist mode enabled by default

if "--terminal-simple" in sys.argv or "--simple" in sys.argv:
    os.environ["TERMINAL_SIMPLE"] = "1"
    sys.argv = [arg for arg in sys.argv if arg not in {"--terminal-simple", "--simple"}]

_TERMINAL_SIMPLE_DEFAULT = os.getenv("TERMINAL_SIMPLE_DEFAULT", "").lower()
if _TERMINAL_SIMPLE_DEFAULT in {"1", "true", "yes", "on"}:
    os.environ["TERMINAL_SIMPLE"] = "1"
elif _TERMINAL_SIMPLE_DEFAULT in {"0", "false", "no", "off"}:
    os.environ["TERMINAL_SIMPLE"] = "0"

_TERMINAL_SIMPLE = os.getenv("TERMINAL_SIMPLE", "0").lower() in {"1", "true", "yes", "on"}


def _console_print(*args, **kwargs):
    if _TERMINAL_SIMPLE:
        return
    _builtins.print(*args, **kwargs)


print = _console_print  # type: ignore


class _SimpleSpinner:
    def __init__(self, message: str = "AI reasoning in progress"):
        self._message = message
        self._frames = ["|", "/", "-", "\\"]
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True

        def _run():
            idx = 0
            try:
                while self._running:
                    frame = self._frames[idx % len(self._frames)]
                    _builtins.print(f"\r{self._message} {frame}", end="", flush=True)
                    time.sleep(0.15)
                    idx += 1
            finally:
                _builtins.print("\r" + " " * (len(self._message) + 2) + "\r", end="", flush=True)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.6)
            self._thread = None

# =============================================
# Logging config (compact by default)
#   LOG_LEVEL=verbose  -> print full prompts/outputs
#   LOG_LEVEL=compact  -> trimmed to LOG_TRUNC chars
# =============================================
import os
_LOG_LEVEL = os.getenv("LOG_LEVEL", "compact").lower()
_LOG_TRUNC  = int(os.getenv("LOG_TRUNC", "600"))

def _trim_for_log(text: str) -> str:
    if _LOG_LEVEL == "verbose":
        return text
    t = text or ""
    if len(t) <= _LOG_TRUNC:
        return t
    return t[:_LOG_TRUNC] + "..."

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from news_fetcher import NewsFetcher
news_fetcher = NewsFetcher(provider_priority=["finnhub","polygon","fmp","newsapi","marketaux"])
trend_fetcher = NewsFetcher(provider_priority=["newsapi", "finnhub"])
# Load environment variables from a local .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# watsonx LLM
try:
    from langchain_ibm import ChatWatsonx
    _WATSONX_AVAILABLE = True
except Exception:
    _WATSONX_AVAILABLE = False

# Optional dependencies
try:
    import yfinance as yf  # type: ignore
    _YFINANCE_AVAILABLE = True
except Exception:
    _YFINANCE_AVAILABLE = False

try:
    import requests  # for optional NewsAPI / Finnhub
    _REQUESTS_AVAILABLE = True
except Exception:
    _REQUESTS_AVAILABLE = False


# =============================================
# Defaults / Persona
# =============================================
_DEFAULT_PERSONA = {
    "user_id": "demo_user",
    "risk_level": "medium",  # low/medium/high
    "preferences": {
        "esg": True,
        "regions": ["US"],
        "sectors": [""],#technology
        "exclude": ["Tobacco", "Weapons"],
        "max_single_weight": 0.15,  # 15% cap per single name for incremental allocation
    },
    "constraints": {"target_extra_allocation": 0.30},  # 30% incremental allocation
}


# =============================================
# TOOLS (plain funcs)
# =============================================
def get_user_profile(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Load a user persona from ./data/user_profiles/{user_id}.json, or fallback to default."""
    uid = user_id or "demo_user"
    base = os.path.join(os.path.dirname(__file__), "data", "user_profiles")
    path = os.path.join(base, f"{uid}.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        return {"error": f"failed_to_load_profile: {e}", **_DEFAULT_PERSONA}
    return _DEFAULT_PERSONA


def get_prices(tickers: List[str]) -> Dict[str, Any]:
    """Return {"prices": {ticker: {"price": float, "currency": "USD", "timestamp": int}}}."""
    out: Dict[str, Any] = {"prices": {}}
    ts = int(time.time())
    if not tickers:
        return out

    if _YFINANCE_AVAILABLE:
        try:
            data = yf.download(
                tickers=tickers,
                period="1d",
                interval="1m",
                progress=False,
                threads=False,
            )
            for t in tickers:
                try:
                    if ("Adj Close" in data) and (t in getattr(data["Adj Close"], "columns", [])):
                        price = float(data["Adj Close"][t].dropna().iloc[-1])
                    elif "Adj Close" in data and not isinstance(data["Adj Close"], dict):
                        price = float(data["Adj Close"].dropna().iloc[-1])
                    else:
                        if ("Close" in data) and (t in getattr(data["Close"], "columns", [])):
                            price = float(data["Close"][t].dropna().iloc[-1])
                        elif "Close" in data:
                            price = float(data["Close"].dropna().iloc[-1])
                        else:
                            price = float("nan")
                except Exception:
                    price = float("nan")
                out["prices"][t] = {"price": price, "currency": "USD", "timestamp": ts}
            return out
        except Exception as e:
            out["warning"] = f"yfinance_failed: {e}"

    # placeholder when yfinance unavailable/fails
    for t in tickers:
        out["prices"][t] = {"price": 100.0, "currency": "USD", "timestamp": ts}
    out.setdefault("note", "using_placeholder_prices")
    return out


def get_news(query: str, limit: int = 1) -> Dict[str, Any]:
    """If NEWSAPI_KEY exists and requests available, fetch news; else stub headline."""
    if not query:
        return {"query": query, "articles": [], "note": "empty_query"}

    key = os.getenv("NEWSAPI_KEY")
    if _REQUESTS_AVAILABLE and key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "pageSize": max(1, int(limit)),
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": key,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            arts = []
            for a in data.get("articles", [])[: max(1, int(limit))]:
                arts.append({
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "content": a.get("content"),
                    "url": a.get("url"),
                    "source": (a.get("source") or {}).get("name"),
                    "published": a.get("publishedAt"),
                })
            return {"query": query, "articles": arts}
        except Exception as e:
            return {
                "query": query,
                "articles": [{
                    "title": f"(fallback) Latest update on {query}",
                    "description": f"(fallback) Latest update on {query}",
                    "url": "https://example.com/news",
                    "source": f"newsapi_error:{e}",
                    "published": "1970-01-01T00:00:00Z",
                }],
                "note": "newsapi_failed_fallback_stub",
            }

    # Finnhub fallback (if NEWSAPI_KEY is missing but FINNHUB_API_KEY is available)
    fk = os.getenv("FINNHUB_API_KEY")
    if _REQUESTS_AVAILABLE and fk:
        try:
            to_ts = int(time.time())
            from_ts = to_ts - 7 * 24 * 3600  # last 7 days
            to_str = time.strftime("%Y-%m-%d", time.gmtime(to_ts))
            from_str = time.strftime("%Y-%m-%d", time.gmtime(from_ts))
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": query, "from": from_str, "to": to_str, "token": fk},
                timeout=8,
            )
            r.raise_for_status()
            data = r.json() if isinstance(r.json(), list) else []
            arts = []
            for a in data[: max(1, int(limit))]:
                published = a.get("datetime")
                if isinstance(published, (int, float)):
                    published = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(published)))
                arts.append({
                    "title": a.get("headline") or a.get("title"),
                    "description": a.get("summary") or a.get("headline"),
                    "url": a.get("url"),
                    "source": a.get("source"),
                    "published": published,
                })
            if arts:
                return {"query": query, "articles": arts, "note": "finnhub_company_news"}
        except Exception:
            pass

    # stub
    return {
        "query": query,
        "articles": [{
            "title": f"(stub) Latest update on {query}",
            "description": f"(stub) Latest update on {query}",
            "url": "https://example.com/news",
            "source": "demo",
            "published": "1970-01-01T00:00:00Z",
        }][: max(1, int(limit))]
    }


def get_analyst_reports(ticker: str, limit: int = 1) -> Dict[str, Any]:
    """If FMP_API_KEY exists and requests available, fetch consensus & PT via FinancialModelingPrep; else stub."""
    if not ticker:
        return {"ticker": ticker, "reports": []}

    # -- FMP implementation --------------------------------
    key = os.getenv("FMP_API_KEY")  # <-- set this in your .env file
    if _REQUESTS_AVAILABLE and key:
        rating = None  # Buy / Sell / Hold
        pt = None      # average price target
        notes: List[str] = []
        try:
            # Endpoint 1: analyst consensus (Buy/Sell/Hold counts)
            rec_url = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}?limit={max(1, int(limit))}&apikey={key}"
            rec_data = requests.get(rec_url, timeout=8).json() or []
            if rec_data:
                # FMP returns list of objects with fields: date, rating, ratingScore, ratingDetails...
                r0 = rec_data[0]
                rating_raw = (r0.get("rating") or "").title()
                rating = rating_raw if rating_raw in {"Buy", "Sell", "Hold"} else None
                notes.append(f"Last rating {rating_raw} on {r0.get('date')}")
        except Exception as e:
            notes.append(f"rec_err:{e}")

        try:
            # Endpoint 2: price target
            pt_url = f"https://financialmodelingprep.com/api/v3/price-target/{ticker}?apikey={key}"
            pt_data = requests.get(pt_url, timeout=8).json() or {}
            pt = pt_data.get("priceTargetAverage") or pt_data.get("priceTarget")
            if pt is None:
                # sometimes FMP returns list
                if isinstance(pt_data, list) and pt_data:
                    pt = pt_data[0].get("priceTargetAverage")
        except Exception as e:
            notes.append(f"pt_err:{e}")

        snippet_parts = []
        if rating:
            snippet_parts.append(f"Consensus {rating}")
        if pt is not None:
            snippet_parts.append(f"mean PT {pt}")
        if notes:
            snippet_parts.append("; ".join(notes))
        snippet = ", ".join(snippet_parts) if snippet_parts else "No analyst data."

        return {
            "ticker": ticker,
            "reports": [{
                "rating": rating or "N/A",
                "pt": pt,
                "snippet": snippet,
            }][: max(1, int(limit))]
        }
    # -- end FMP implementation ----------------------------
    # stub
    return {
        "ticker": ticker,
        "reports": [{
            "rating": "Buy",
            "pt": 250.0,
            "snippet": f"(stub) Positive outlook on {ticker} driven by AI tailwinds.",
        }][: max(1, int(limit))]
    }


# =============================================
# LLM builder
# =============================================
def _get_project_id() -> Optional[str]:
    for key in ("PROJ_ID", "PROJECT_ID", "WATSONX_PROJECT_ID"):
        v = os.getenv(key)
        if v:
            return v
    return None


def build_llm():
    if not _WATSONX_AVAILABLE:
        raise RuntimeError("ChatWatsonx not available. Install langchain-ibm and configure env.")
    try:
        from dotenv import load_dotenv as _ld
        _ld()
    except Exception:
        pass
    model_id = os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-2-8b-instruct")
    project_id = _get_project_id()
    if not project_id:
        raise ValueError(
            "Missing project id. Set PROJ_ID / PROJECT_ID / WATSONX_PROJECT_ID (e.g., in .env)."
        )
    return ChatWatsonx(
        model_id=model_id,
        project_id=project_id,
        params={"decoding_method": "greedy", "max_new_tokens": 512, "temperature": 0.0},
    )


# ---------------------------------------------
# Helper: wrap llm.invoke with logging (compact/verbose)
# ---------------------------------------------
def _llm_invoke_with_log(llm: "ChatWatsonx", messages, tag: str):
    """Invoke LLM and print compact/verbose logs depending on LOG_LEVEL."""
    sep = "=" * 60
    print(f"\n{sep}\n[LLM CALL -> {tag.upper()}]\n{sep}")
    print(f"  msgs={len(messages)}  mode={_LOG_LEVEL}")
    for i, m in enumerate(messages, 1):
        role = getattr(m, "role", type(m).__name__)
        content = str(getattr(m, "content", ""))
        content = _trim_for_log(content).replace("\n", " \n   ")
        print(f"  {i:02d}. {role:<7}: {content}")
    print("-" * 60)
    ai = llm.invoke(messages)
    out = _trim_for_log(ai.content or "")
    print(f"[LLM RESP <- {tag.upper()}] {out}\n{sep}\n")
    return ai


def insights_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()

    facts = _get_json(state, "FACTS_JSON")
    tickers: List[str] = facts.get("tickers", [])
    news = facts.get("news", {})
    analyst = facts.get("analyst", {})
    prices = (facts.get("prices", {}) or {}).get("prices", {})

    if not tickers:
        return {"messages": [HumanMessage(content="INSIGHTS_JSON: {\"insights\": {}}")]}

    # Assemble contextual bundle for each ticker
    features = {}
    for t in tickers:
        ticker_news = []
        for art in (news.get(t, {}) or {}).get("articles", [])[:3]:
            ticker_news.append({
                "title": art.get("title"),
                "preview": art.get("preview") or _news_preview(
                    art.get("description") or art.get("content") or art.get("title") or ""
                ),
                "source": art.get("source"),
                "published": art.get("published"),
                "url": art.get("url"),
            })
        reports = []
        for rep in (analyst.get(t, {}) or {}).get("reports", [])[:3]:
            reports.append({
                "rating": rep.get("rating"),
                "pt": rep.get("pt"),
                "snippet": rep.get("snippet"),
            })
        features[t] = {
            "price": (prices.get(t) or {}).get("price"),
            "news": ticker_news,
            "news_summary": (news.get(t, {}) or {}).get("summary"),
            "analyst": reports,
        }

    sys = SystemMessage(content=(
        "You are an equity research co-pilot.\n"
        "For each ticker, reflect on the provided news previews, analyst snippets, and recent price.\n"
        "Produce concise structured insights capturing the key bullish and bearish angles before scoring.\n"
        "Return ONLY one fenced JSON with exactly this schema:\n"
        "```json\n{\n  \"insights\": {\n    \"TICKER\": {\n      \"summary\": \"15-25 words overall view\",\n      \"bullish\": [\"<=15 words point\", \"...\"],\n      \"bearish\": [\"<=15 words point\", \"...\"]\n    }\n  }\n}\n```"
    ))
    human = HumanMessage(content=f"FACTS_FOR_INSIGHTS_JSON: {json.dumps(features, ensure_ascii=False)}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "insights")

    parsed = _extract_json_from_text(ai.content) or {}
    insights = parsed.get("insights")
    if not isinstance(insights, dict):
        insights = {}

    # Fallbacks and normalization
    normalized = {}
    for t in tickers:
        item = insights.get(t) if isinstance(insights.get(t), dict) else {}
        summary = str(item.get("summary") or "").strip()
        if not summary:
            summary = str((news.get(t, {}) or {}).get("summary") or "").strip()
        if not summary:
            first_article = ((news.get(t, {}) or {}).get("articles") or [{}])[0]
            preview = first_article.get("preview") or _news_preview(
                first_article.get("description") or first_article.get("content") or first_article.get("title") or ""
            )
            summary = preview or "No key highlight available; keep monitoring."
        bullish = [str(p).strip() for p in item.get("bullish", []) if str(p).strip()]
        bearish = [str(p).strip() for p in item.get("bearish", []) if str(p).strip()]
        normalized[t] = {
            "summary": summary,
            "bullish": bullish[:3],
            "bearish": bearish[:3],
        }
        print(f"[insight-summary] {t}: {summary}")
        if bullish:
            print(f"[insight-bull] {t}: {' | '.join(bullish[:3])}")
        if bearish:
            print(f"[insight-bear] {t}: {' | '.join(bearish[:3])}")

    payload = {"insights": normalized}
    return {"messages": [HumanMessage(content=f"INSIGHTS_JSON: {json.dumps(payload, ensure_ascii=False)}")]}

def analyze_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()

    facts = _get_json(state, "FACTS_JSON")
    insights = _get_json(state, "INSIGHTS_JSON")
    tickers: List[str] = facts.get("tickers", [])
    news = facts.get("news", {})
    analyst = facts.get("analyst", {})
    prices = (facts.get("prices", {}) or {}).get("prices", {})
    insight_map = (insights.get("insights") if isinstance(insights.get("insights"), dict) else insights) or {}

    # Build compact features per ticker for LLM reasoning
    features = {}
    for t in tickers:
        arts = (news.get(t, {}) or {}).get("articles", [])[:3]
        titles = [a.get("title") or "" for a in arts]
        previews = [
            a.get("preview") or _news_preview(a.get("description") or a.get("content") or a.get("title") or "")
            for a in arts
        ]
        reps = (analyst.get(t, {}) or {}).get("reports", [])[:3]
        rep_snips = [r.get("snippet") or "" for r in reps]
        pt = (reps[0].get("pt") if reps else None)
        rating = (reps[0].get("rating") if reps else None)
        insight = insight_map.get(t, {}) if isinstance(insight_map.get(t), dict) else {}
        features[t] = {
            "price": (prices.get(t) or {}).get("price"),
            "news_titles": titles,
            "news_previews": previews,
            "news_summary": (news.get(t, {}) or {}).get("summary"),
            "analyst_snippets": rep_snips,
            "analyst_first_pt": pt,
            "analyst_first_rating": rating,
            "insight_summary": insight.get("summary"),
            "insight_bullish": insight.get("bullish"),
            "insight_bearish": insight.get("bearish"),
        }

    sys = SystemMessage(content=(
        "You are an equity analyst. Using ONLY the provided features, score each candidate 0-100 "
        "for incremental allocation suitability under ESG preference. Consider the news summaries and previews, analyst view/targets, "
        "explicit bullish/bearish insights, and potential risks implied. Penalize clear controversies in headlines. "
        "Return ONLY one fenced JSON:\n"
        "```json\n{\n  \"scores\": {\"TICKER\": int, ...},\n  \"notes\": {\"TICKER\": \"<=20 words reason\"}\n}\n```"
    ))
    human = HumanMessage(content=f"CANDIDATE_FEATURES_JSON: {json.dumps(features, ensure_ascii=False)}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "analyze")

    parsed = _extract_json_from_text(ai.content) or {}
    scores = parsed.get("scores") or {}
    notes = parsed.get("notes") or {}

    # Fallback: equal mid scores if LLM fails
    if not isinstance(scores, dict) or not scores:
        scores = {t: 50 for t in tickers}
    # Clip & clean
    clean_scores: Dict[str, int] = {}
    for t, v in scores.items():
        try:
            iv = int(float(v))
        except Exception:
            iv = 50
        clean_scores[t] = max(0, min(100, iv))

    note_map = {t: str(notes.get(t, "")) for t in tickers}
    payload = {"scores": clean_scores, "notes": note_map}

    for t in tickers:
        score = clean_scores.get(t, 0)
        note_text = note_map.get(t, "")
        print(f"[analysis-score] {t}: score={score} note={note_text}")

    return {"messages": [HumanMessage(content=f"ANALYSIS_JSON: {json.dumps(payload, ensure_ascii=False)}")]}


def allocate_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()

    facts = _get_json(state, "FACTS_JSON")
    analysis = _get_json(state, "ANALYSIS_JSON")
    profile = _get_json(state, "USER_PROFILE_JSON")
    selection = _get_json(state, "SELECTION_JSON")

    tickers: List[str] = facts.get("tickers", [])
    scores: Dict[str, int] = analysis.get("scores", {})
    target = float(facts.get("target_extra_allocation") or 0.0)
    cap = float(facts.get("max_single_weight", 0.15))
    chosen = selection.get("chosen") if isinstance(selection.get("chosen"), list) else []
    rationale = selection.get("rationale") if isinstance(selection.get("rationale"), dict) else {}

    if not chosen:
        chosen = tickers
    chosen = [t for t in chosen if t in tickers]
    if not chosen:
        chosen = tickers

    candidate_bundle = []
    for t in chosen:
        candidate_bundle.append({
            "ticker": t,
            "score": int(scores.get(t, 50)),
            "selection_reason": rationale.get(t, ""),
        })

    # Build compact input for allocator
    alloc_input = {
        "target": target,
        "cap": cap,
        "candidates": candidate_bundle,
        "profile": profile,
    }

    sys = SystemMessage(content=(
        "You are a portfolio allocator.\n"
        "Use the pre-selected tickers, their scores, and rationale to assign weights that sum to the target.\n"
        "Rules: total equals target (float), per-name <= cap, prefer higher scores, diversify across tickers, "
        "and reflect any qualitative rationale (e.g., momentum vs. valuation) in the final explanation.\n"
        "If target==0, return zeros.\n"
        "Return ONLY one fenced JSON:\n"
        "```json\n{\n  \"allocation\": {\"TICKER\": float, ...},\n  \"rationale\": {\"TICKER\": \"<=20 words weight reason\"}\n}\n```"
    ))
    human = HumanMessage(content=f"ALLOCATE_INPUT_JSON: {json.dumps(alloc_input, ensure_ascii=False)}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "allocate")
    picked = _extract_json_from_text(ai.content) or {"allocation": {}}
    alloc: Dict[str, float] = {}
    alloc_reason: Dict[str, str] = {}
    s = 0.0
    for t, w in (picked.get("allocation") or {}).items():
        try:
            w = float(w)
        except Exception:
            continue
        if t in chosen:
            v = max(0.0, min(cap, w))
            alloc[t] = v
            s += v
    for t, reason in (picked.get("rationale") or {}).items():
        if t in chosen:
            alloc_reason[t] = str(reason)[:200]

    # Safety/fallback: if empty or sum!=target, build from top scores deterministically
    if not alloc or s <= 1e-12:
        # choose top-k by score (k between 3 and 6)
        top = sorted([(t, scores.get(t, 50)) for t in chosen], key=lambda x: x[1], reverse=True)
        k = min(6, max(3, len(top)))
        chosen = [t for t, _ in top[:k]]
        alloc = _build_suggested_weights(chosen, target, cap)
        s = sum(alloc.values())
        alloc_reason = {t: rationale.get(t, "Backtest weight generated from scores") for t in alloc}

    if s > 0 and abs(s - target) > 1e-6:
        scale = target / s
        for t in list(alloc.keys()):
            alloc[t] = min(cap, alloc[t] * scale)

    for t in alloc:
        print(f"[allocation-weight] {t}: weight={alloc[t]:.4f} score={scores.get(t, 0)} reason={alloc_reason.get(t, '')}")

    payload = {"allocation": alloc, "rationale": alloc_reason}
    return {"messages": [HumanMessage(content=f"PICKS_JSON: {json.dumps(payload, ensure_ascii=False)}")]}


# =============================================
# HELPERS
# =============================================
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object, preferring ```json fenced blocks."""
    if not text:
        return None
    fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        try:
            return json.loads(brace.group(1))
        except Exception:
            pass
    return None


def _first_sentence(text: Optional[str]) -> str:
    """Return the first sentence from text, falling back to ~120 characters."""
    if not text:
        return ""
    s = str(text).strip()
    # Check sentence terminators in priority order
    for sep in [".", "!", "?"]:
        idx = s.find(sep)
        if idx != -1:
            return s[: idx + 1].strip()
    # Fallback: truncate
    return s[:120].strip()


def _news_preview(text: Optional[str], max_sentences: int = 2, max_chars: int = 320) -> str:
    """Return a longer preview (first 2 sentences or up to max_chars)."""
    if not text:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", s)
    preview = " ".join(sentences[:max_sentences]).strip()
    if not preview:
        preview = s[:max_chars].strip()
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    return preview

def _format_docs_for_context(docs, max_chars: int = 3600) -> str:
    buf = []
    used = 0
    for i, d in enumerate(docs, 1):
        piece = (d.page_content or "").strip()
        if not piece:
            continue
        piece = piece[:1200]  # 单段限长，避免 LLM 太啰嗦
        s = f"[{i}] {piece}"
        if used + len(s) > max_chars:
            break
        buf.append(s)
        used += len(s)
    return "\n\n".join(buf)


_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}\b")
_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "WILL", "WONT",
    "USA", "US", "NYSE", "NASDAQ", "DJIA", "SP", "SPX", "DOW", "ETF",
    "TECH", "AI", "IPO", "CEO", "EV", "FDA", "EPS", "GDP", "CPI", "PCE",
    "DATA", "NEWS", "MARKET", "INDEX", "STOCK", "STOCKS",
    "STUB", "SMALL", "CAP", "HONG", "KONG", "RED", "CHIP", "CHIPS",
}


def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _is_valid_ticker(token: Optional[str]) -> bool:
    if not token:
        return False
    t = str(token).upper()
    if t in _TICKER_STOPWORDS:
        return False
    if not re.fullmatch(r"[A-Z]{1,5}", t):
        return False
    return True


def _extract_tickers_from_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    tokens = _TICKER_TOKEN_RE.findall(str(text).upper())
    tickers: List[str] = []
    for tok in tokens:
        if tok in _TICKER_STOPWORDS:
            continue
        if not _is_valid_ticker(tok):
            continue
        tickers.append(tok)
    return tickers


def _find_hot_tickers_for_sector(sector: Optional[str], max_articles: int = 6, max_tickers: int = 12) -> Dict[str, Any]:
    """Use news headlines to extract currently mentioned tickers for a sector/theme."""
    queries: List[str] = []
    sector_clean = (sector or "").strip()
    if sector_clean:
        queries.extend([
            f"{sector_clean} stocks",
            f"{sector_clean} companies",
            f"{sector_clean} market",
        ])
    queries.append("US stock market")
    aggregated_tickers: List[str] = []
    articles_info: List[Dict[str, Any]] = []

    for q in queries:
        try:
            articles = trend_fetcher.fetch(q, limit=max_articles)
        except Exception as exc:
            try:
                print(f"[hotlist] fetch error for query='{q}': {exc}")
            except Exception:
                pass
            continue
        for art in articles or []:
            if not art or (art.get("source") or "").lower() == "stub" or (art.get("title") or "").lower().startswith("(stub)"):
                continue
            title = art.get("title") or ""
            desc = art.get("description") or art.get("content") or ""
            snippet = f"{title} {desc}"
            tickers = _unique_preserve(_extract_tickers_from_text(snippet))
            if not tickers:
                continue
            aggregated_tickers.extend(tickers)
            articles_info.append({
                "query": q,
                "title": title,
                "source": art.get("source"),
                "published": art.get("published"),
                "tickers": tickers,
                "url": art.get("url"),
            })
        if len(_unique_preserve(aggregated_tickers)) >= max_tickers:
            break

    unique_tickers = _unique_preserve(aggregated_tickers)[:max_tickers]
    return {
        "sector": sector_clean or "general",
        "tickers": unique_tickers,
        "articles": articles_info,
    }
def get_index_constituents(index_symbol: str = "^NDX") -> List[str]:
    """Fetch index constituents from Finnhub. Returns a list of tickers, or []."""
    key = os.getenv("FINNHUB_API_KEY")
    if not (_REQUESTS_AVAILABLE and key):
        return []
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/index/constituents",
            params={"symbol": index_symbol, "token": key},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json() or {}
        tickers = data.get("constituents") or []
        if isinstance(tickers, list):
            return [t for t in tickers if isinstance(t, str)]
    except Exception:
        pass
    return []


def get_company_profile(ticker: str) -> Dict[str, Any]:
    """Fetch minimal company profile (country, sector) from Finnhub. Returns {} on failure."""
    key = os.getenv("FINNHUB_API_KEY")
    if not (_REQUESTS_AVAILABLE and key and ticker):
        return {}
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": ticker, "token": key},
            timeout=6,
        )
        r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}


def discover_candidates(preferences: Dict[str, Any], max_candidates: int = 12) -> List[str]:
    """Discover tickers dynamically via APIs (no hardcoded list).
    Strategy:
      1) Choose index by theme (Technology -> ^NDX, otherwise -> ^GSPC)
      2) Pull constituents once
      3) Filter by country==US and sector keyword (if provided)
      4) Return up to max_candidates tickers
    """
    sectors = set([s.lower() for s in (preferences or {}).get("sectors", []) if isinstance(s, str)])
    wants_tech = any(str(s).lower() in {"technology", "tech"} for s in sectors)
    index_symbol = "^NDX" if wants_tech else "^GSPC"

    pool = get_index_constituents(index_symbol) or []
    if not pool:
        # last resort: try the other index
        pool = get_index_constituents("^GSPC") or []

    results: List[str] = []
    for t in pool:
        if len(results) >= max_candidates:
            break
        prof = get_company_profile(t)
        if not prof:
            continue
        country_ok = (str(prof.get("country") or prof.get("countryOfIncorporation") or "US").upper() == "US")
        sector_name = (prof.get("finnhubIndustry") or prof.get("sector") or "").lower()
        sector_ok = True
        if sectors:
            sector_ok = any(k in sector_name for k in sectors if k)
        if country_ok and sector_ok:
            results.append(t)
    try:
        print(f"[discover] index={index_symbol} -> {len(results)} candidates (limit {max_candidates})")
    except Exception:
        pass
    return results[:max_candidates]


# Generic getter for JSON payloads by tag
def _get_json(state: MessagesState, tag: str) -> Dict[str, Any]:
    """Return last JSON payload whose message starts with '{tag}:'."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.startswith(f"{tag}:"):
            try:
                return json.loads(m.content.split(f"{tag}:", 1)[1].strip())
            except Exception:
                pass
    return {}


# Build suggested weights (deterministic allocator)
def _build_suggested_weights(
    tickers: List[str], target: float, cap: float
) -> Dict[str, float]:
    """
    Deterministic allocator to hit target sum and obey per-name cap.
    - Start equal weight = min(cap, target / n)
    - Distribute remaining evenly across names not at cap
    - Final proportional rescale if tiny drift remains
    """
    n = max(1, len(tickers))
    base = min(cap, target / n)
    w = {t: base for t in tickers}
    used = base * n
    remain = max(0.0, target - used)
    if remain > 1e-9:
        # one pass spread
        flex = [t for t in tickers if w[t] < cap]
        if flex:
            add = min(cap - base, remain / len(flex))
            for t in flex:
                w[t] = min(cap, w[t] + add)
            used = sum(w.values())
            remain = max(0.0, target - used)

    if remain > 1e-9:
        # small final proportional push on those still below cap
        flex = [t for t in tickers if w[t] < cap]
        if flex:
            total_flex = sum((cap - w[t]) for t in flex)
            if total_flex > 0:
                for t in flex:
                    delta = remain * (cap - w[t]) / total_flex
                    w[t] = min(cap, w[t] + delta)
                used = sum(w.values())

    # final exact rescale if off by tiny eps
    if abs(sum(w.values()) - target) > 1e-6 and sum(w.values()) > 0:
        scale = target / sum(w.values())
        for t in w:
            w[t] = min(cap, w[t] * scale)

    # clip numerical noise
    s = sum(w.values())
    if s > 0 and abs(s - target) > 1e-6:
        # normalize tiny drift keeping caps
        factor = target / s
        for t in w:
            w[t] = min(cap, w[t] * factor)
    return w


# =============================================
# NODES (LangGraph)
# =============================================
def profile_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    profile = get_user_profile(user_id="demo_user")
    return {"messages": [HumanMessage(content=f"USER_PROFILE_JSON: {json.dumps(profile, ensure_ascii=False)}")]}


def plan_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()
    sys = SystemMessage(content=(
        "You are a planner for an investment assistant.\n"
        "Use USER_PROFILE_JSON and the user's most recent message. Identify target sectors/themes from the user's message; if absent, fall back to USER_PROFILE_JSON.preferences.sectors.\n"
        "Constrain the universe to U.S.-listed equities.\n"
        "Always propose 4-8 candidate tickers that best match the intent (valid US-listed tickers only). Prefer real symbols over ETFs; avoid duplicates.\n"
        "Return ONLY ONE fenced JSON block with the EXACT schema and NOTHING ELSE:\n"
        "```json\n"
        "{\n"
        "  \"target_sectors\": [\"Technology\"],\n"
        "  \"target_themes\": [\"AI\"],\n"
        "  \"universe\": \"US-listed equities\",\n"
        "  \"candidate_tickers\": [\"AAPL\", \"MSFT\"],\n"
        "  \"notes\": \"<=30 words summary\"\n"
        "}\n"
        "```"
    ))
    msgs = state["messages"] + [sys]
    ai = _llm_invoke_with_log(llm, msgs, "plan")
    # --- enhanced harmonization and patching of plan_json ---
    plan_json = _extract_json_from_text(ai.content)
    if plan_json is not None:
        # ---- harmonize schema ----
        # 1) Promote symbols list to `tickers` if LLM returned recommended_stocks
        if "tickers" not in plan_json:
            rec = plan_json.get("recommended_stocks")
            if isinstance(rec, list):
                plan_json["tickers"] = [
                    item.get("symbol") for item in rec if isinstance(item, dict) and item.get("symbol")
                ]
        # 2) Inject max_single_weight from user profile when absent
        if "max_single_weight" not in plan_json:
            prof_json = _get_json(state, "USER_PROFILE_JSON")
            mx = (prof_json.get("preferences", {}) or {}).get("max_single_weight", 0.15)
            plan_json["max_single_weight"] = mx
        # 3) If missing target_extra_allocation, try regex 50% etc.
        if "target_extra_allocation" not in plan_json:
            latest_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
            pct = None
            if latest_human and isinstance(latest_human.content, str):
                m = re.search(r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%", latest_human.content)
                if m:
                    pct = float(m.group(1)) / 100.0
            # fallback to profile default 0.3
            if pct is None:
                prof_json = _get_json(state, "USER_PROFILE_JSON")
                pct = (prof_json.get("constraints", {}) or {}).get("target_extra_allocation", 0.3)
            plan_json["target_extra_allocation"] = round(float(pct), 4)
        # 4) Normalize candidate tickers
        candidates = []
        raw_candidates = plan_json.get("candidate_tickers") or plan_json.get("tickers") or []
        if isinstance(raw_candidates, list):
            candidates = [
                t.upper() for t in raw_candidates if isinstance(t, str) and _is_valid_ticker(t)
            ]
        plan_json["tickers"] = candidates[:8]
        # Re-embed patched JSON back into ai.content
        patched = json.dumps(plan_json, ensure_ascii=False, indent=2)
        ai.content = f"```json\n{patched}\n```"
    return {"messages": [AIMessage(content=ai.content)]}


def intent_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()
    latest_user_msg = next(
        (
            m for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
            and not str(m.content).startswith(("USER_PROFILE_JSON", "TOOL_PLAN_JSON", "FACTS_JSON", "HOTLIST_JSON", "INTENT_JSON"))
        ),
        None,
    )
    latest_human = latest_user_msg or next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    profile = _get_json(state, "USER_PROFILE_JSON")
    profile_pretty = json.dumps(profile, ensure_ascii=False)
    user_text = latest_user_msg.content if isinstance(latest_user_msg, HumanMessage) else ""

    detected_regions: List[str] = []
    detected_keywords: List[str] = []
    text_lower = str(user_text).lower() if isinstance(user_text, str) else ""
    if "red chip" in text_lower:
        detected_keywords.append("red chip")
        detected_regions.extend(["China", "Hong Kong"])
    if any(term in text_lower for term in ["us stock", "u.s. stock", "united states", "america"]):
        detected_regions.append("US")
    if not detected_regions:
        detected_regions.extend((profile or {}).get("preferences", {}).get("regions", []) or [])
    detected_regions = _unique_preserve([r for r in detected_regions if r])
    detected_keywords = _unique_preserve(detected_keywords)

    sys = SystemMessage(content=(
        "You are an intent analyst for a wealth advisory agent.\n"
        "Summarize the user's request, identify the key investment objectives, ESG/sector hints, risk tolerance cues, "
        "and any tool calls implied (e.g., need prices, news, allocation). "
        "If information is missing, state assumptions you would adopt. "
        "Respond in concise English using <=5 bullet points."
    ))
    human_payload = {
        "user_message": user_text,
        "user_profile": profile,
        "recent_messages_count": len(state["messages"]),
        "detected_keywords": detected_keywords,
        "detected_regions": detected_regions,
    }
    human = HumanMessage(content=f"INTENT_INPUT_JSON: {json.dumps(human_payload, ensure_ascii=False)}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "intent")
    summary_text = ai.content.strip()
    try:
        print(f"[intent-summary] {summary_text}")
    except Exception:
        pass

    payload = {
        "summary": summary_text,
        "timestamp": int(time.time()),
        "detected_keywords": detected_keywords,
        "detected_regions": detected_regions,
    }
    return {"messages": [HumanMessage(content=f"INTENT_JSON: {json.dumps(payload, ensure_ascii=False)}")]}


def hotlist_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    profile = _get_json(state, "USER_PROFILE_JSON")
    target_sectors: List[str] = []

    plan_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and "target_sectors" in str(m.content)),
        None,
    )
    if plan_ai:
        plan_json = _extract_json_from_text(plan_ai.content) or {}
        sectors = plan_json.get("target_sectors") or []
        if isinstance(sectors, list):
            target_sectors = [str(s) for s in sectors if s]

    if not target_sectors:
        prefs = (profile or {}).get("preferences", {}) or {}
        sectors_pref = prefs.get("sectors") or []
        if isinstance(sectors_pref, list):
            target_sectors = [str(s) for s in sectors_pref if s]

    if not target_sectors:
        target_sectors = ["US"]

    per_sector: Dict[str, Any] = {}
    aggregated: List[str] = []
    for sector in target_sectors:
        hot = _find_hot_tickers_for_sector(sector, max_articles=6, max_tickers=12)
        per_sector[hot["sector"]] = hot
        aggregated.extend(hot.get("tickers") or [])
        try:
            print(f"[hotlist] sector={hot['sector']} tickers={','.join(hot.get('tickers') or [])}")
        except Exception:
            pass

    unique_all = _unique_preserve([t for t in aggregated if t])

    payload = {
        "requested_sectors": target_sectors,
        "timestamp": int(time.time()),
        "per_sector": per_sector,
        "all_tickers": unique_all,
    }
    return {"messages": [HumanMessage(content=f"HOTLIST_JSON: {json.dumps(payload, ensure_ascii=False)}")]}


def tool_plan_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    """Let the LLM decide whether to call prices/news/analyst, and with what limits."""
    llm = build_llm()
    sys = SystemMessage(content=(
        "You are a tool-use planner for an investment assistant.\n"
        "Given USER_PROFILE_JSON, the user's latest request, and the prior ticker plan, decide which tools are necessary.\n"
        "Available tools to consider: get_prices(list[str]); get_news(query, limit); get_analyst_reports(ticker, limit).\n"
        "Guidelines:\n"
        "- Fetch prices when making any allocation.\n"
        "- Fetch news \n"
        "- Fetch analyst reports when user mentions target price/valuation/analyst view, or if conviction is low.\n"
        "- You may override the ticker list if the previous plan conflicts with user intent or ESG constraints.\n"
        "- Prefer minimal calls and set small limits (e.g., 1).\n"
        "Return ONLY one fenced JSON with EXACT schema and nothing else:\n"
        "```json\n{\n  \"use_prices\": true,\n  \"use_news\": {\"enabled\": true, \"limit\": 1},\n  \"use_analyst\": {\"enabled\": false, \"limit\": 1},\n  \"tickers_override\": null\n}\n```"
    ))
    msgs = state["messages"] + [sys]
    ai = _llm_invoke_with_log(llm, msgs, "toolplan")
    plan = _extract_json_from_text(ai.content) or {
        "use_prices": True,
        "use_news": {"enabled": True, "limit": 3},
        "use_analyst": {"enabled": True, "limit": 3},
        "tickers_override": None,
    }
    # Normalize types and limits
    def _lim(v: Any, d: int = 3) -> int:
        try:
            n = int(v)
            return max(1, n)
        except Exception:
            return d
    un = plan.get("use_news") or {}
    ua = plan.get("use_analyst") or {}
    plan = {
        "use_prices": bool(plan.get("use_prices", True)),
        "use_news": {"enabled": bool(un.get("enabled", True)), "limit": max(3, _lim(un.get("limit", 3)))},
        "use_analyst": {"enabled": bool(ua.get("enabled", True)), "limit": max(3, _lim(ua.get("limit", 3)))},
        "tickers_override": [
            t for t in (plan.get("tickers_override") or [])
            if isinstance(t, str) and _is_valid_ticker(t)
        ] or None,
    }
    return {"messages": [HumanMessage(content=f"TOOL_PLAN_JSON: {json.dumps(plan, ensure_ascii=False)}")]}
def select_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()
    facts = _get_json(state, "FACTS_JSON")
    profile = _get_json(state, "USER_PROFILE_JSON")
    analysis = _get_json(state, "ANALYSIS_JSON")
    insights = _get_json(state, "INSIGHTS_JSON")
    tickers: List[str] = facts.get("tickers", [])
    target = float(facts.get("target_extra_allocation") or 0.0)
    cap = float(facts.get("max_single_weight", 0.15))
    news = facts.get("news", {})
    prices = (facts.get("prices", {}) or {}).get("prices", {})

    scores = analysis.get("scores", {}) if isinstance(analysis, dict) else {}
    notes = analysis.get("notes", {}) if isinstance(analysis, dict) else {}
    insight_map = insights.get("insights", {}) if isinstance(insights, dict) else {}

    selection_features = {}
    for t in tickers:
        top_article = ((news.get(t, {}) or {}).get("articles") or [{}])[0]
        selection_features[t] = {
            "score": int(scores.get(t, 50)),
            "analysis_note": notes.get(t, ""),
            "insight_summary": (insight_map.get(t) or {}).get("summary") if isinstance(insight_map.get(t), dict) else "",
            "insight_bullish": (insight_map.get(t) or {}).get("bullish") if isinstance(insight_map.get(t), dict) else [],
            "insight_bearish": (insight_map.get(t) or {}).get("bearish") if isinstance(insight_map.get(t), dict) else [],
            "price": (prices.get(t) or {}).get("price"),
            "top_news_headline": top_article.get("title"),
            "top_news_preview": top_article.get("preview"),
            "news_summary": (news.get(t, {}) or {}).get("summary"),
        }

    sys = SystemMessage(content=(
        "You are an investment selector.\n"
        "Based on candidate scores, insights, and the user profile (which may contain irregular instructions), "
        "decide which tickers should proceed to allocation.\n"
        "Respect ESG preferences and diversification, and justify inclusions/exclusions.\n"
        "Pick between 3 and 6 tickers (or all if fewer) prioritising higher scores but allowing qualitative overrides when justified.\n"
        "Return ONLY one fenced JSON with EXACT schema:\n"
        "```json\n{\n  \"chosen\": [\"TICKER\", ...],\n  \"rationale\": {\"TICKER\": \"<=25 words reason\"},\n  \"excluded\": {\"TICKER\": \"<=15 words reason\"}\n}\n```"
    ))
    human_payload = {
        "target_allocation": target,
        "cap": cap,
        "profile": profile,
        "candidates": selection_features,
    }
    human = HumanMessage(content=f"SELECTION_INPUT_JSON: {json.dumps(human_payload, ensure_ascii=False)}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "select")

    parsed = _extract_json_from_text(ai.content) or {}
    chosen = parsed.get("chosen") if isinstance(parsed.get("chosen"), list) else []
    rationale = parsed.get("rationale") if isinstance(parsed.get("rationale"), dict) else {}
    excluded = parsed.get("excluded") if isinstance(parsed.get("excluded"), dict) else {}

    chosen = [t for t in chosen if isinstance(t, str) and t in tickers]
    if not chosen:
        # fallback: use top scores
        sorted_scores = sorted(((t, scores.get(t, 50)) for t in tickers), key=lambda x: x[1], reverse=True)
        limit = min(len(sorted_scores), max(3, min(6, len(sorted_scores))))
        chosen = [t for t, _ in sorted_scores[:limit]]

    # Ensure chosen count between 3-6 when possible
    if len(chosen) > 6:
        chosen = chosen[:6]
    if len(chosen) < 3 and len(tickers) >= 3:
        extras = [
            t for t, _ in sorted(
                ((tt, scores.get(tt, 50)) for tt in tickers if tt not in chosen),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        chosen.extend(extras[: (3 - len(chosen))])

    # Populate rationale fallbacks
    rationale_out = {}
    for t in chosen:
        reason = rationale.get(t)
        if not reason:
            base = notes.get(t) or (insight_map.get(t, {}) or {}).get("summary")
            rationale_out[t] = (base or "Top score and aligns with preferences")[:200]
        else:
            rationale_out[t] = str(reason)[:200]
        print(f"[selection-chosen] {t}: {rationale_out[t]}")
    excluded_out: Dict[str, str] = {}
    for t in tickers:
        if t not in chosen:
            reason = excluded.get(t)
            if not reason:
                reason = "Score too low or conflicts with preferences"
            reason_str = str(reason)[:160]
            excluded_out[t] = reason_str
            print(f"[selection-excluded] {t}: {reason_str}")

    payload = {
        "chosen": chosen,
        "rationale": rationale_out,
        "excluded": excluded_out,
    }
    return {"messages": [HumanMessage(content=f"SELECTION_JSON: {json.dumps(payload, ensure_ascii=False)}")]}

def gather_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    # Prefer the most recent AIMessage that actually contains target_extra_allocation,
    # otherwise fall back to the last AIMessage.
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and "target_extra_allocation" in str(m.content)),
        None,
    )
    if last_ai is None:
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    tickers: List[str] = []
    target_alloc: Optional[float] = None
    max_single = 0.15
    hotlist = _get_json(state, "HOTLIST_JSON")
    hot_tickers = [
        str(t) for t in (hotlist.get("all_tickers") or []) if isinstance(t, str)
    ]
    if last_ai:
        data = _extract_json_from_text(last_ai.content)
        if isinstance(data, dict):
            tickers = [t for t in data.get("tickers", []) if isinstance(t, str)]
            tmp_alloc = data.get("target_extra_allocation", target_alloc)
            try:
                target_alloc = float(tmp_alloc) if tmp_alloc is not None else None
            except Exception:
                target_alloc = None                             
            max_single = float(data.get("max_single_weight", max_single))
    if not tickers and hot_tickers:
        tickers = hot_tickers[:12]
    if not tickers:
        # Discover dynamically from APIs (no hardcoded lists)
        profile = _get_json(state, "USER_PROFILE_JSON")
        prefs = (profile or {}).get("preferences", {})
        tickers = discover_candidates(prefs, max_candidates=12)
    if not tickers and hot_tickers:
        tickers = hot_tickers[:12]
    if not tickers:
        # last resort: still try to get some dynamic universe from index
        tickers = get_index_constituents("^NDX")[:10] or get_index_constituents("^GSPC")[:10] or []

    # Read tool plan and optionally override tickers / limits
    tool_plan = _get_json(state, "TOOL_PLAN_JSON")
    use_prices = True
    use_news = {"enabled": True, "limit": 3}
    use_analyst = {"enabled": True, "limit": 2}
    if tool_plan:
        try:
            ov = tool_plan.get("tickers_override")
            if ov and isinstance(ov, list) and all(isinstance(x, str) for x in ov):
                filtered = [t for t in ov if _is_valid_ticker(t)]
                if filtered:
                    tickers = filtered
        except Exception:
            pass
        try:
            use_prices = bool(tool_plan.get("use_prices", True))
            t_un = tool_plan.get("use_news") or {}
            t_ua = tool_plan.get("use_analyst") or {}
            use_news = {
                "enabled": bool(t_un.get("enabled", True)),
                "limit": max(3, int(t_un.get("limit", 3) or 3)),
            }
            use_analyst = {
                "enabled": bool(t_ua.get("enabled", True)),
                "limit": max(2, int(t_ua.get("limit", 2) or 2)),
            }
        except Exception:
            pass

    # One-line query logs
    if use_prices:
        try:
            print(f"[query] get_prices: {','.join(tickers)}")
        except Exception:
            pass
        prices = get_prices(tickers)
    else:
        print("[skip] get_prices disabled by tool plan")
        prices = {"prices": {}}

    news_dict: Dict[str, Any] = {}
    reports_dict: Dict[str, Any] = {}
    for t in tickers:
        # NEWS (conditional)
        if use_news.get("enabled", True):
            try:
                print(f"[query] get_news: {t} limit={use_news['limit']}")
            except Exception:
                pass
            # news_dict[t] = get_news(t, limit=use_news.get("limit", 1))
            try:
                articles_raw = news_fetcher.fetch(t, limit=use_news.get("limit", 3))
                enriched_articles = []
                preview_pool: List[str] = []
                for art in articles_raw or []:
                    art_dict = dict(art) if isinstance(art, dict) else {"title": str(art)}
                    preview_src = (
                        art_dict.get("description")
                        or art_dict.get("content")
                        or art_dict.get("title")
                        or ""
                    )
                    preview = _news_preview(preview_src)
                    if preview:
                        preview_pool.append(preview)
                    art_dict["preview"] = preview
                    enriched_articles.append(art_dict)
                summary_text = _news_preview(" ".join(preview_pool) if preview_pool else " ".join(
                    (art.get("title") or "") for art in enriched_articles
                ), max_sentences=3, max_chars=480)
                news_dict[t] = {"query": t, "articles": enriched_articles}
                if summary_text:
                    news_dict[t]["summary"] = summary_text
            except Exception as e:
                news_dict[t] = {"query": t, "articles": [], "note": f"fetch_error:{e}"}
            # Source indicator (one line)
            try:
                meta_note = news_dict[t].get("note") or ""
                art0 = (news_dict[t].get("articles") or [{}])[0]
                src_name = (art0.get("source") or "")
                if meta_note == "newsapi_failed_fallback_stub":
                    mode = "fallback"
                elif meta_note == "finnhub_company_news":
                    mode = "finnhub"
                elif src_name == "demo":
                    mode = "stub"
                else:
                    mode = "newsapi"
                print(f"[news-src] {t}: {mode}")
            except Exception:
                pass
            try:
                for idx, art in enumerate((news_dict[t].get("articles") or [])[: use_news.get("limit", 1)]):
                    headline = art.get("title") or ""
                    preview = art.get("preview") or _news_preview(
                        art.get("description") or art.get("content") or art.get("title") or ""
                    )
                    if headline:
                        print(f"[news-headline] {t}#{idx+1}: {headline}")
                    if preview:
                        print(f"[news-preview] {t}#{idx+1}: {preview}")
                summary_log = news_dict[t].get("summary")
                if summary_log:
                    print(f"[news-summary] {t}: {summary_log}")
            except Exception:
                pass
        else:
            print(f"[skip] get_news disabled by tool plan for {t}")
            news_dict[t] = {"query": t, "articles": []}

        # ANALYST (conditional)
        if use_analyst.get("enabled", True):
            try:
                print(f"[query] get_analyst_reports: {t} limit={use_analyst['limit']}")
            except Exception:
                pass
            reports_dict[t] = get_analyst_reports(t, limit=use_analyst.get("limit", 1))
            # --- full first analyst snippet ---
            try:
                rep_full = (reports_dict[t].get("reports") or [{}])[0]
                rating = rep_full.get("rating")
                pt     = rep_full.get("pt")
                snip   = rep_full.get("snippet")
                print(f"[analyst-full] {t}: rating={rating}, pt={pt}, snippet={snip}")
            except Exception:
                pass
        else:
            print(f"[skip] get_analyst_reports disabled by tool plan for {t}")
            reports_dict[t] = {"ticker": t, "reports": []}

    if target_alloc is None:
        # fallback to profile constraint 0.3
        prof = _get_json(state, "USER_PROFILE_JSON")
        target_alloc = (prof.get("constraints", {}) or {}).get("target_extra_allocation", 0.3)
    ta = float(target_alloc)
    suggested = _build_suggested_weights(tickers, ta, max_single)
    facts = {
        "tickers": tickers,
        "target_extra_allocation": target_alloc,
        "max_single_weight": max_single,
        "suggested_weights": suggested,
        "prices": prices,
        "news": news_dict,
        "analyst": reports_dict,
    }
    return {"messages": [HumanMessage(content=f"FACTS_JSON: {json.dumps(facts, ensure_ascii=False)}")]}


def synthesize_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    """
    Deterministic finalization:
    - Use suggested_weights from FACTS_JSON to GUARANTEE sum == target and cap obeyed.
    - Ask LLM only to write one-line rationale & one key risk per ticker, given news/analyst snippets.
    - Then compose the final human-readable plan + JSON using our fixed weights.
    """
    llm = build_llm()

    facts = _get_json(state, "FACTS_JSON")
    picks = _get_json(state, "PICKS_JSON")
    analysis = _get_json(state, "ANALYSIS_JSON")
    selection = _get_json(state, "SELECTION_JSON")
    scores = analysis.get("scores", {}) if isinstance(analysis, dict) else {}
    selection_rationale = selection.get("rationale", {}) if isinstance(selection, dict) else {}
    alloc_comment = picks.get("rationale", {}) if isinstance(picks, dict) else {}
    suggested: Dict[str, float] = (picks.get("allocation") or facts.get("suggested_weights") or {})
    if not suggested:
        tickers_fallback: List[str] = facts.get("tickers", [])
        suggested = _build_suggested_weights(tickers_fallback, float(facts.get("target_extra_allocation") or 0.0), float(facts.get("max_single_weight", 0.15)))
    tickers: List[str] = [t for t, w in suggested.items() if w > 0] or list(suggested.keys())
    target = float(facts.get("target_extra_allocation") or 0.0)
    cap = float(facts.get("max_single_weight", 0.15))
    news = facts.get("news", {})
    analyst = facts.get("analyst", {})
    prices = facts.get("prices", {}).get("prices", {})

    # Fallback if suggested missing: compute here
    if (not suggested) and tickers:
        suggested = _build_suggested_weights(tickers, target, cap)

    # Log a one-liner for the final chosen weights
    try:
        total = sum(float(suggested.get(t, 0.0)) for t in tickers)
        print(f"[debug] final_weights_sum={total:.6f} target={target:.6f} tickers={','.join(tickers)}")
    except Exception:
        pass

    if not tickers:
        fallback_text = (
            "No suitable ticker candidates were found, likely because the user description is too vague or the current filters are too strict. "
            "Please provide additional investment preferences (sector/region/risk) or relax the ESG/weight limits; alternatively, we can deliver a market briefing."
        )
        return {"messages": [AIMessage(content=fallback_text)]}

    # Ask LLM *only* for rationale & risk text (no weights)
    rationale_sys = SystemMessage(content=(
        "You are a concise investment writer.\n"
        "For each ticker you are given, write ONE short rationale (<=25 words) using the provided 3-news titles and 3-analyst snippets,\n"
        "and ONE key risk (<=12 words). Output ONLY a fenced JSON with this schema:\n"
        "```json\n{\n  \"rationales\": {\"AAPL\": \"...\", ...},\n  \"risks\": {\"AAPL\": \"...\", ...}\n}\n```"
    ))
    mini_facts = {
        "tickers": tickers,
        "snippets": {
            t: {
                "news_titles": " | ".join([a.get("title") or "" for a in (news.get(t, {}) or {}).get("articles", [])][:3]),
                "news_summary": (news.get(t, {}) or {}).get("summary", ""),
                "analyst_snippets": " | ".join([r.get("snippet") or "" for r in (analyst.get(t, {}) or {}).get("reports", [])][:3]),
                "selection_rationale": selection_rationale.get(t, ""),
            } for t in tickers
        }
    }
    msgs = state["messages"] + [rationale_sys, HumanMessage(content=f"FACTS_MINI_JSON: {json.dumps(mini_facts, ensure_ascii=False)}")]
    ai = _llm_invoke_with_log(llm, msgs, "synthesize-rationale")
    parsed = _extract_json_from_text(ai.content) or {}
    rationales: Dict[str, str] = parsed.get("rationales", {}) if isinstance(parsed.get("rationales"), dict) else {}
    risks: Dict[str, str] = parsed.get("risks", {}) if isinstance(parsed.get("risks"), dict) else {}

    # Compose human-readable plan using our fixed weights
    lines = []
    for t in tickers:
        w = float(suggested.get(t, 0.0))
        titles = [a.get("title") for a in (news.get(t, {}) or {}).get("articles", [])][:3]
        pt = ((analyst.get(t, {}) or {}).get("reports") or [{}])[0].get("pt")
        px = (prices.get(t) or {}).get("price")
        previews = [
            a.get("preview") or _news_preview(a.get("description") or a.get("content") or a.get("title") or "")
            for a in (news.get(t, {}) or {}).get("articles", [])[:2]
        ]
        news_digest = " ".join([p for p in previews if p])
        if not news_digest:
            news_digest = _news_preview(" / ".join(titles)) or "No recent news summary available"
        news_summary_text = (news.get(t, {}) or {}).get("summary") or news_digest

        selection_reason = selection_rationale.get(t, "").strip()
        analysis_reason = (rationales.get(t) or "").strip()
        if not analysis_reason:
            analysis_reason = selection_reason or (" / ".join(titles) or "News flow is positive; analyst commentary is supportive")
        risk_text = (risks.get(t) or "Sector volatility and regulatory oversight").strip()
        alloc_reason = (alloc_comment.get(t) or "Allocation driven by scores and diversification constraints").strip()
        score_val = scores.get(t, "N/A")

        detail_lines: List[str] = []
        if selection_reason:
            detail_lines.append(f"Stock selection: {selection_reason}")
        if analysis_reason and analysis_reason not in {selection_reason}:
            detail_lines.append(f"Model analysis: {analysis_reason}")
        if alloc_reason and alloc_reason not in {selection_reason, analysis_reason}:
            detail_lines.append(f"Weight rationale: {alloc_reason}")
        elif alloc_reason and not detail_lines:
            detail_lines.append(f"Weight rationale: {alloc_reason}")
        if news_summary_text:
            detail_lines.append(f"News summary: {news_summary_text}")
        if risk_text:
            detail_lines.append(f"Key risks: {risk_text}")

        bullet_body = "\n".join(f"  - {dl}" for dl in detail_lines if dl)
        lines.append(
            f"- **{t}**: {w*100:.2f}% (~price {px if px is not None else 'N/A'}; PT {pt if pt is not None else 'N/A'}; score {score_val})\n"
            f"{bullet_body}"
        )

    # Build narrative summary
    bullets = "\n".join(lines)
    human_readable = (
        "Here is your incremental allocation proposal in natural language. The weights sum exactly to the target increase, and no single position exceeds the cap:\n"
        + bullets
    )
    return {"messages": [AIMessage(content=human_readable)]}
def teach_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")

    # 1) 准备查询：用“最后一条用户问题 + 本轮候选ticker/主题”做教学检索提示
    latest_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_q = (latest_user.content if isinstance(latest_user, HumanMessage) else "").strip()

    facts = _get_json(state, "FACTS_JSON")
    tickers = facts.get("tickers", []) if isinstance(facts, dict) else []
    intent = _get_json(state, "INTENT_JSON")
    intent_summary = intent.get("summary", "") if isinstance(intent, dict) else ""

    teach_query = " | ".join([user_q] + tickers + [intent_summary])
    teach_query = (teach_query or "tech stocks basics ESG risk diversification valuation").strip()

    # 2) 载入教学向量库（如果没找到，给出友好提示）
    try:
        embed_teach = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_dir = os.path.join(os.path.dirname(__file__), "teach_db")
        teach_vs = FAISS.load_local(db_dir, embeddings=embed_teach, allow_dangerous_deserialization=True)
        teach_retriever = teach_vs.as_retriever(search_kwargs={"k": 6})
        docs = teach_retriever.get_relevant_documents(teach_query)
        context = _format_docs_for_context(docs, max_chars=3600)
    except Exception as e:
        warn = (
            "⚠️ 未检测到教学向量库 teach_db/。请先运行：\n"
            "    python teach_ingest.py\n"
            "然后重试。也可以先参考以下公共资源：\n"
            "1) Investopedia Financial Education Library\n"
            "2) CFA Institute Investor Education\n"
            "3) Morningstar Investment Glossary"
        )
        return {"messages": [AIMessage(content=warn)]}

    # 3) 让 LLM 产出“教学卡片”
    llm = build_llm()
    sys = SystemMessage(content=(
        "你是投资教学助教。基于提供的资料与用户问题，输出一个【教学卡片】（中文）：\n"
        "结构必须包含：\n"
        "1) 关键词术语速查（<=6条，每条<=20字）\n"
        "2) 本题投资思路（3-5点，短句分条）\n"
        "3) 延伸阅读（3条，保留来源名 + URL）\n"
        "注意：只用提供的上下文信息，不可胡编。若上下文没有URL，可从文段中提取原始来源标题/站点名并附此前述三大站点的URL。"
    ))
    human = HumanMessage(content=(
        f"【用户问题】\n{user_q}\n\n"
        f"【相关上下文（截断）】\n{context}\n\n"
        "请严格按要求输出，不要添加其他前后缀。"
    ))
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "teach")

    teaching_text = "📚 教学卡片\n" + (ai.content or "").strip()
    return {"messages": [AIMessage(content=teaching_text)]}


# =============================================
# GRAPH (build)
# =============================================
def build_agent():
    g = StateGraph(MessagesState)
    # 你已有的节点...
    g.add_node("analyze", analyze_node)
    g.add_node("allocate", allocate_node)
    g.add_node("select", select_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("insights", insights_node)
    g.add_node("profile", profile_node)
    g.add_node("intent", intent_node)
    g.add_node("plan", plan_node)
    g.add_node("hotlist", hotlist_node)
    g.add_node("toolplan", tool_plan_node)
    g.add_node("gather", gather_node)

    # ✅ 新增教学节点
    g.add_node("teach", teach_node)

    # 你已有的边...
    g.add_edge(START, "profile")
    g.add_edge("profile", "intent")
    g.add_edge("intent", "plan")
    g.add_edge("plan", "hotlist")
    g.add_edge("hotlist", "toolplan")
    g.add_edge("toolplan", "gather")
    g.add_edge("gather", "insights")
    g.add_edge("insights", "analyze")
    g.add_edge("analyze", "select")
    g.add_edge("select", "allocate")
    g.add_edge("allocate", "synthesize")

    # ✅ 让教学在“最终自然语言方案”之后输出
    g.add_edge("synthesize", "teach")
    g.add_edge("teach", END)

    return g.compile()


# =============================================
# DEMO (main)
# =============================================
if __name__ == "__main__":
    proj_hint = os.getenv("PROJ_ID") or os.getenv("PROJECT_ID") or os.getenv("WATSONX_PROJECT_ID")
    if not proj_hint:
        print("[Env] Project ID not found. Set PROJ_ID / PROJECT_ID / WATSONX_PROJECT_ID (e.g., in .env).")
    else:
        print(f"[Env] Project ID detected: {proj_hint[:6]}... (length {len(proj_hint)})")

    agent = build_agent()

    user_query = (
        # "I want to add some U.S. consumer stocks with an ESG tilt; please suggest 5 tickers for a 50% allocation"
        "I would like to buy some technology stocks"
        # "I would like to increase exposure to U.S. energy stocks and rare earth names"

        
    )

    # Start the conversation with the user's request
    messages = [HumanMessage(content=user_query)]

    final_output: Optional[str] = None
    spinner: Optional[_SimpleSpinner] = None
    if _TERMINAL_SIMPLE:
        _builtins.print(user_query)
        spinner = _SimpleSpinner()
        spinner.start()
    else:
        print("---- Running agent (stream) ----")
    for update in agent.stream({"messages": messages}, stream_mode="updates"):
        for node, ev in update.items():
            if not ev.get("messages"):
                continue
            if _TERMINAL_SIMPLE:
                if node in {"synthesize", "teach"}:   # 原来只有 "synthesize"
                    for msg in ev["messages"]:
                        if isinstance(msg, AIMessage):
                            if final_output:
                                final_output += "\n\n" + msg.content  # 先方案，后教学
                            else:
                                final_output = msg.content
                            if spinner:
                                spinner.stop()
                continue
            print(f"\n[# {node.upper()} NODE OUTPUT] ->")
            for msg in ev["messages"]:
                if isinstance(msg, AIMessage):
                    print(f"  AI : {msg.content}")
                elif isinstance(msg, HumanMessage):
                    print(f"  USER: {msg.content}")
                elif isinstance(msg, SystemMessage):
                    print(f"  SYS : {msg.content}")
                else:
                    print(f"  {type(msg).__name__}: {getattr(msg,'content',str(msg))}")
    if spinner:
        spinner.stop()
    if _TERMINAL_SIMPLE:
        if final_output is not None:
            _builtins.print(final_output)
    else:
        print("---- Done ----")

    
