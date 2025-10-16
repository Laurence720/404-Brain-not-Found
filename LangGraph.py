from __future__ import annotations
import os
import sys
import re
import json
import time
import inspect
import threading
import math
from datetime import datetime, timezone
import builtins as _builtins
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

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
_TEACH_VECTOR_AVAILABLE = HuggingFaceEmbeddings is not None and FAISS is not None

from langchain_core.messages import AIMessage, HumanMessage
os.environ["TERMINAL_SIMPLE_DEFAULT"] = "0"  # Keep minimalist mode enabled by default

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


def _log_tool_call(name: str, **kwargs):
    """Emit a structured tool-call log when not in simple terminal mode."""
    if _TERMINAL_SIMPLE:
        return
    try:
        if kwargs:
            parts = []
            for key, val in kwargs.items():
                text = str(val)
                if len(text) > 120:
                    text = text[:117] + "..."
                parts.append(f"{key}={text}")
            args_str = ", ".join(parts)
        else:
            args_str = ""
        if args_str:
            _builtins.print(f"[tool] {name}({args_str})")
        else:
            _builtins.print(f"[tool] {name}()")
    except Exception:
        _builtins.print(f"[tool] {name}(...)")

_AGENT_LOCK = threading.Lock()
_AGENT_INSTANCE: Optional[Any] = None


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
from smart_pii_protection import SmartPIIProtector
from risk_layer import get_risk_assessment
from concept_tool import generate_concept_card
from formula_tool import generate_formula_card
from worked_example_tool import generate_worked_example
from pitfalls_tool import generate_pitfalls_card
from quiz_tool import generate_quiz_card
news_fetcher = NewsFetcher(provider_priority=["finnhub","polygon","fmp","newsapi","marketaux"])
trend_fetcher = NewsFetcher(provider_priority=["newsapi", "finnhub"])
pii_protector = SmartPIIProtector()
_PII_PROTECTION_LEVEL = os.getenv("PII_PROTECTION_LEVEL", "balanced").strip().lower() or "balanced"
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
    """Load a user persona, preferring encrypted profiles when available."""
    uid = user_id or _DEFAULT_PERSONA.get("user_id", "demo_user")
    base = os.path.join(os.path.dirname(__file__), "data", "user_profiles")
    encrypted_path = os.path.join(base, f"{uid}.json.enc")
    plain_path = os.path.join(base, f"{uid}.json")
    _log_tool_call("get_user_profile", user_id=uid, encrypted=os.path.exists(encrypted_path))

    try:
        if os.path.exists(encrypted_path):
            try:
                from crypto_utils import UserDataEncryption
                crypto = UserDataEncryption()
                return crypto.load_encrypted_profile(uid, base)
            except Exception as exc:
                print(f"[pii-warning] Failed to decrypt profile {encrypted_path}: {exc}")
                if os.path.exists(plain_path):
                    with open(plain_path, "r", encoding="utf-8") as fh:
                        return json.load(fh)
        elif os.path.exists(plain_path):
            with open(plain_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        fallback = dict(_DEFAULT_PERSONA)
        fallback["error"] = f"failed_to_load_profile: {exc}"
        return fallback
    return _DEFAULT_PERSONA


def save_user_profile(user_id: str, profile: Dict[str, Any], encrypt: bool = True) -> str:
    """Persist a user profile, optionally encrypting the payload."""
    uid = user_id or _DEFAULT_PERSONA.get("user_id", "demo_user")
    base = os.path.join(os.path.dirname(__file__), "data", "user_profiles")
    os.makedirs(base, exist_ok=True)
    _log_tool_call("save_user_profile", user_id=uid, encrypt=encrypt)

    if encrypt:
        try:
            from crypto_utils import UserDataEncryption
            crypto = UserDataEncryption()
            return crypto.save_encrypted_profile(uid, profile, base)
        except Exception as exc:
            print(f"[pii-warning] Encrypting profile failed; falling back to plaintext: {exc}")
            encrypt = False

    path = os.path.join(base, f"{uid}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)
    return path


def protect_user_message(user_input: str, user_id: Optional[str] = None, protection_level: Optional[str] = None) -> Dict[str, Any]:
    """Apply intelligent PII protection to user text before sending to the LLM."""
    if not user_input:
        return {
            "session_id": None,
            "original_text": "",
            "anonymized_text": "",
            "pii_count": 0,
            "pii_types": [],
        }
    level = (protection_level or _PII_PROTECTION_LEVEL) or "balanced"
    uid = user_id or _DEFAULT_PERSONA.get("user_id", "demo_user")
    _log_tool_call("protect_user_message", level=level, user_id=uid)
    try:
        return pii_protector.smart_protect_user_input(user_input, uid, level)
    except Exception as exc:
        print(f"[pii-warning] smart_protect_user_input failed: {exc}")
        return {
            "session_id": None,
            "original_text": user_input,
            "anonymized_text": user_input,
            "pii_count": 0,
            "pii_types": [],
        }


def restore_llm_response(llm_response: str, session_id: Optional[str]) -> str:
    """Restore protected PII tokens in an LLM response."""
    if not llm_response or not session_id:
        return llm_response
    _log_tool_call("restore_llm_response", session_id=session_id)
    try:
        return pii_protector.restore_smart_pii(llm_response, session_id)
    except Exception as exc:
        print(f"[pii-warning] restore_smart_pii failed: {exc}")
        return llm_response


def get_prices(tickers: List[str]) -> Dict[str, Any]:
    """Return {"prices": {ticker: {"price": float, "currency": "USD", "timestamp": int}}}."""
    _log_tool_call("get_prices", tickers=",".join(tickers or []))
    out: Dict[str, Any] = {"prices": {}}
    ts = int(time.time())
    if not tickers:
        return out

    if _YFINANCE_AVAILABLE:
        try:
            # yfinance's default multithreaded downloader can trigger numpy/pandas type errors
            # on Apple Silicon; disable threading to keep the fetch stable.
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
    _log_tool_call("get_news", query=query, limit=limit)
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
    _log_tool_call("get_analyst_reports", ticker=ticker, limit=limit)
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
    risk_data = facts.get("risk", {}) if isinstance(facts, dict) else {}
    risk_metrics = risk_data.get("metrics") or {}
    risk_kept = set(risk_data.get("kept") or [])
    risk_dropped = risk_data.get("dropped") or {}

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
            "risk_metrics": risk_metrics.get(t) or {},
            "risk_screen": {
                "kept": t in risk_kept,
                "dropped_reason": risk_dropped.get(t),
            },
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
    risk_data = facts.get("risk", {}) if isinstance(facts, dict) else {}
    risk_metrics = risk_data.get("metrics") or {}
    risk_kept = set(risk_data.get("kept") or [])
    risk_dropped = risk_data.get("dropped") or {}

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
            "risk_metrics": risk_metrics.get(t) or {},
            "risk_screen": {
                "kept": t in risk_kept,
                "dropped_reason": risk_dropped.get(t),
            },
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
    raw_notes = parsed.get("notes") or {}

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

    normalized_notes: Dict[str, str] = {}
    if isinstance(raw_notes, dict):
        normalized_notes = {t: str(raw_notes.get(t, "")) for t in tickers}
    elif isinstance(raw_notes, list):
        normalized_notes = {t: str(note) for t, note in zip(tickers, raw_notes)}
    elif isinstance(raw_notes, str):
        stripped = raw_notes.strip()
        if stripped:
            try:
                parsed_notes = json.loads(stripped)
            except Exception:
                parsed_notes = None
            if isinstance(parsed_notes, dict):
                normalized_notes = {t: str(parsed_notes.get(t, "")) for t in tickers}
            elif isinstance(parsed_notes, list):
                normalized_notes = {t: str(note) for t, note in zip(tickers, parsed_notes)}
            else:
                normalized_notes = {t: stripped for t in tickers}

    note_map = {t: normalized_notes.get(t, "") for t in tickers}
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
    risk_excluded_map = selection.get("risk_excluded") if isinstance(selection.get("risk_excluded"), dict) else {}

    tickers: List[str] = facts.get("tickers", [])
    scores: Dict[str, int] = analysis.get("scores", {})
    target = float(facts.get("target_extra_allocation") or 0.0)
    cap = float(facts.get("max_single_weight", 0.15))
    chosen = selection.get("chosen") if isinstance(selection.get("chosen"), list) else []
    rationale = selection.get("rationale") if isinstance(selection.get("rationale"), dict) else {}
    risk_data = facts.get("risk", {}) if isinstance(facts, dict) else {}
    risk_metrics = risk_data.get("metrics") or {}
    risk_dropped = risk_data.get("dropped") or {}

    if not chosen:
        chosen = tickers
    chosen = [t for t in chosen if t in tickers]
    if not chosen:
        chosen = tickers

    risk_level = str((profile or {}).get("risk_level", "medium") or "medium").lower()
    candidate_bundle = []
    for t in chosen:
        risk_info = risk_excluded_map.get(t) if isinstance(risk_excluded_map.get(t), dict) else None
        if risk_info and risk_level == "low" and risk_info.get("code") == "beta_too_high":
            print(f"[allocate-skip] {t} removed due to beta_too_high in low-risk profile")
            continue
        candidate_bundle.append({
            "ticker": t,
            "score": int(scores.get(t, 50)),
            "selection_reason": rationale.get(t, ""),
            "risk_metrics": risk_metrics.get(t) or {},
            "risk_screen": {"dropped_reason": risk_dropped.get(t)},
        })

    if not candidate_bundle:
        for t in chosen:
            candidate_bundle.append({
                "ticker": t,
                "score": int(scores.get(t, 50)),
                "selection_reason": rationale.get(t, ""),
                "risk_metrics": risk_metrics.get(t) or {},
                "risk_screen": {"dropped_reason": risk_dropped.get(t)},
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
            risk_info = risk_excluded_map.get(t) if isinstance(risk_excluded_map.get(t), dict) else None
            if risk_info and risk_level == "low" and risk_info.get("code") == "beta_too_high":
                print(f"[allocate-skip] {t} allocation blocked due to beta requirement")
                continue
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
        piece = piece[:1200]  # keep slices compact for LLM input
        snippet = f"[{i}] {piece}"
        if used + len(snippet) > max_chars:
            break
        buf.append(snippet)
        used += len(snippet)
    return "\n\n".join(buf)


_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}\b")
_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "WILL", "WONT",
    "USA", "US", "NYSE", "NASDAQ", "DJIA", "SP", "SPX", "DOW", "ETF",
    "TECH", "AI", "IPO", "CEO", "EV", "FDA", "EPS", "GDP", "CPI", "PCE",
    "DATA", "NEWS", "MARKET", "INDEX", "STOCK", "STOCKS",
    "STUB", "SMALL", "CAP", "HONG", "KONG", "RED", "CHIP", "CHIPS",
}

_RECOMMEND_KEYWORDS = {
    "recommend", "recommendation", "recommendations", "should i buy", "should i sell",
    "what stock should i buy", "which stock should i buy", "best stock", "top stock",
    "buy list", "sell list", "give me stocks", "pick stocks", "stock pick", "stock picks",
    "investment advice", "precise pick", "exact stock", "target stock", "allocation suggestion",
}
_ALLOC_KEYWORDS = {
    "exact weight", "percentage", "allocation", "weighting", "how much should i invest",
}
_ANALYZE_KEYWORDS = {
    "analyze", "analyse", "analysis", "analyser", "analyst",
    "analysis request", "stock insight", "stock review", "investment review",
    "review", "insight on", "take on", "view on",
}

_SECTOR_FALLBACKS = {
    "technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSM", "ADBE"],
    "consumer": ["KO", "PG", "PEP", "MCD", "WMT", "COST", "DIS", "NKE"],
    "consumer staples": ["KO", "PG", "PEP", "COST", "WMT", "CL", "MDLZ", "K"],
    "consumer discretionary": ["AMZN", "HD", "MCD", "SBUX", "NKE", "TSLA", "CMG", "LOW"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "HAL", "DVN"],
    "financials": ["JPM", "BAC", "MS", "GS", "V", "MA", "BLK", "MSCI"],
    "health care": ["UNH", "JNJ", "PFE", "MRK", "LLY", "ABBV", "MDT", "TMO"],
    "industrials": ["CAT", "HON", "UNP", "GE", "DE", "RTX", "BA", "UPS"],
    "utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "SRE"],
    "materials": ["LIN", "APD", "SHW", "DD", "NEM", "FCX", "MLM", "VMC"],
    "communication services": ["GOOGL", "META", "NFLX", "TMUS", "VZ", "DIS", "CMCSA", "TTD"],
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


def _fallback_sector_tickers(sectors: List[str], limit: int = 10) -> List[str]:
    picked: List[str] = []
    for sector in sectors:
        key = str(sector or "").lower()
        if not key:
            continue
        for candidate, tickers in _SECTOR_FALLBACKS.items():
            if candidate == key or candidate in key or key in candidate:
                for t in tickers:
                    if t not in picked:
                        picked.append(t)
                        if len(picked) >= limit:
                            return picked
    return picked[:limit]


def _sanitize_history_for_agent(history: Optional[List[Dict[str, str]]], protection_level: Optional[str] = None) -> List[Dict[str, str]]:
    """Return a sanitized copy of history suitable for LLM consumption."""
    if not history:
        return []
    sanitized: List[Dict[str, str]] = []
    level = (protection_level or _PII_PROTECTION_LEVEL) or "balanced"
    uid = _DEFAULT_PERSONA.get("user_id", "demo_user")
    for item in history:
        if not isinstance(item, dict):
            continue
        raw_role = str(item.get("role", "")).lower()
        role = item.get("role", raw_role)
        content = item.get("content")
        if content is None:
            continue
        text = str(content)
        if not text.strip():
            continue
        if raw_role in {"user", "human"}:
            protected = protect_user_message(text, uid, level)
            sanitized.append({"role": role, "content": protected.get("anonymized_text", text)})
        else:
            sanitized.append({"role": role, "content": text})
    return sanitized


def _gateway_router(state: MessagesState) -> str:
    gateway = _get_json(state, "GATEWAY_JSON")
    if isinstance(gateway, dict) and not gateway.get("is_finance", True):
        return "general"
    return "finance"


def _format_news_timestamp(value: Optional[str]) -> str:
    """Convert ISO-like strings to 'YYYY-MM-DD HH:MM UTC' for display."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def _is_precise_recommendation_request(text: Optional[str]) -> bool:
    if not text:
        return False
    raw = str(text)
    normalized = raw.lower()
    if any(k in normalized for k in _RECOMMEND_KEYWORDS):
        return True
    if any(k in normalized for k in _ALLOC_KEYWORDS):
        if _extract_tickers_from_text(raw):
            return True
    return False


def _screen_user_prompt(user_input: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Inspect the incoming user input (and optional history) to ensure there are no requests
    for precise stock recommendations before invoking the agent graph.
    Returns guardrail metadata when disallowed intent is detected.
    """
    # Guardrails intentionally disabled; always allow the agent to respond freely.
    return None


def _history_tail(history: Optional[List[Dict[str, Any]]], limit: int = 4) -> List[Dict[str, str]]:
    """Return a simplified slice of recent conversation history."""
    if not history:
        return []
    tail: List[Dict[str, str]] = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        if role not in {"user", "assistant", "human", "ai"}:
            continue
        content = item.get("content")
        if content is None:
            continue
        tail.append({"role": role, "content": str(content)})
    return tail


def _collect_market_snapshot(tickers: List[str], news_limit: int = 2) -> Dict[str, Any]:
    """Fetch price/news/analyst/profile data for the given tickers."""
    unique = _unique_preserve([t for t in tickers if _is_valid_ticker(t)])
    price_info = get_prices(unique).get("prices", {})
    news_info = {t: get_news(t, limit=news_limit) for t in unique}
    analyst_info = {t: get_analyst_reports(t, limit=1) for t in unique}
    profile_info = {t: get_company_profile(t) for t in unique}
    snapshot: Dict[str, Any] = {}
    for t in unique:
        px = price_info.get(t, {})
        price_val = px.get("price")
        snapshot[t] = {
            "price": price_val if isinstance(price_val, (int, float)) and math.isfinite(price_val) else None,
            "currency": px.get("currency"),
            "price_ts": px.get("timestamp"),
            "news": news_info.get(t, {}),
            "analyst": analyst_info.get(t, {}),
            "profile": profile_info.get(t, {}),
        }
    return snapshot


def _messages_to_simple_history(messages: List[Any], limit: int = 6) -> List[Dict[str, str]]:
    simple: List[Dict[str, str]] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            simple.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            simple.append({"role": "assistant", "content": str(msg.content)})
    return simple


def _detect_tickers_with_llm(user_input: Optional[str], history_tail: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Ask the LLM to identify tickers mentioned in the user input.
    Returns {"tickers": [...], "raw": {...}}.
    """
    if not user_input or not str(user_input).strip():
        return {"tickers": [], "raw": {}}
    payload = {
        "user_input": str(user_input),
        "history_tail": history_tail or [],
        "instructions": "Extract only liquid, tradeable stock tickers (1-5 uppercase letters).",
    }
    sys_text = (
        "You are a professional markets analyst. Find any explicit stock tickers mentioned in the user message. "
        "Only output valid U.S.-listed tickers. If none are present, return an empty list. "
        "Return strictly one JSON object with fields: \"tickers\" (list of strings) and \"reason\" (<=25 words). "
        "Never guess tickers; if uncertain, leave the list empty."
    )
    try:
        llm = build_llm()
    except Exception:
        return {"tickers": [], "raw": {}}
    sys_msg = SystemMessage(content=sys_text)
    human_msg = HumanMessage(content=f"TICKER_DETECT_INPUT_JSON: {json.dumps(payload, ensure_ascii=False)}")
    try:
        ai = _llm_invoke_with_log(llm, [sys_msg, human_msg], "ticker-detect")
        data = _extract_json_from_text(ai.content) or {}
    except Exception as exc:
        try:
            print(f"[ticker-detect] llm error: {exc}")
        except Exception:
            pass
        data = {}
    raw_tickers = data.get("tickers") if isinstance(data, dict) else []
    tickers: List[str] = []
    if isinstance(raw_tickers, list):
        for item in raw_tickers:
            if isinstance(item, str):
                sym = item.strip().upper()
                if _is_valid_ticker(sym):
                    tickers.append(sym)
    tickers = _unique_preserve(tickers)
    return {"tickers": tickers, "raw": data}


def _generate_guardrail_response(guard: Dict[str, Any]) -> str:
    """Ask the LLM to craft an actionable response when additional guidance is needed."""
    tickers = [t for t in guard.get("tickers") or [] if _is_valid_ticker(t)]
    history_tail = guard.get("history_tail") or []
    snapshot = _collect_market_snapshot(tickers, news_limit=1) if tickers else {}

    sys_text = (
        "You are an investment companion. The user asked for precise buy/sell or allocation guidance. "
        "Leverage the structured market context (prices, news, analyst views, profile data) to deliver a concise, actionable response "
        "with clear recommendations, suggested price levels or allocations, and the reasoning behind them."
    )
    payload = {
        "user_input": guard.get("user_input"),
        "history_tail": history_tail,
        "tickers": tickers,
        "snapshot": snapshot,
        "base_message": guard.get("message"),
    }
    try:
        llm = build_llm()
    except Exception:
        base_msg = guard.get("message") or "Unable to generate a strategy right now; please try again later."
        return base_msg

    sys = SystemMessage(content=sys_text)
    human = HumanMessage(content=f"GUARDRAIL_CONTEXT_JSON: {json.dumps(payload, ensure_ascii=False)}")
    try:
        ai = _llm_invoke_with_log(llm, [sys, human], "guardrail")
        text = (ai.content or "").strip()
        if text:
            return text
    except Exception as exc:
        try:
            print(f"[guardrail] llm error: {exc}")
        except Exception:
            pass
    base_msg = guard.get("message") or "Unable to generate a strategy right now; please try again later."
    return base_msg


def _generate_analysis_response(user_input: str, history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """For simple analysis-only requests, let the LLM craft a qualitative overview using live data.
    Returns None if no shortcut applies."""
    normalized = (user_input or "").lower()
    if not user_input or not any(k in normalized for k in _ANALYZE_KEYWORDS):
        return None

    history_tail = _history_tail(history)
    detect_main = _detect_tickers_with_llm(user_input, history_tail)
    tickers = detect_main.get("tickers", [])
    if not tickers and history:
        for item in reversed(history or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() not in {"user", "human"}:
                continue
            prev_text = str(item.get("content") or "")
            detect_prev = _detect_tickers_with_llm(prev_text, None)
            prev_tickers = detect_prev.get("tickers", [])
            if prev_tickers:
                tickers = prev_tickers
                break

    tickers = [t for t in tickers if _is_valid_ticker(t)]
    if not tickers or len(tickers) > 4:
        return None

    snapshot = _collect_market_snapshot(tickers, news_limit=2)
    history_tail = _history_tail(history)
    sys_text = (
        "You are an investment research assistant. The user wants actionable analysis on the supplied tickers. "
        "Use the structured market snapshot to summarize recent performance, key news themes, analyst viewpoints, "
        "and potential risks. Provide clear buy/sell/hold views, price targets, or allocation suggestions when helpful, "
        "and spell out the reasoning, trade-offs, and key uncertainties."
    )
    payload = {
        "user_input": user_input,
        "history_tail": history_tail,
        "tickers": tickers,
        "snapshot": snapshot,
    }
    try:
        llm = build_llm()
    except Exception:
        return None

    sys = SystemMessage(content=sys_text)
    human = HumanMessage(content=f"ANALYSIS_CONTEXT_JSON: {json.dumps(payload, ensure_ascii=False)}")
    try:
        ai = _llm_invoke_with_log(llm, [sys, human], "analysis-direct")
        text = (ai.content or "").strip()
        if text:
            return text
    except Exception as exc:
        try:
            print(f"[analysis-shortcut] llm error: {exc}")
        except Exception:
            pass
    return None


def _find_hot_tickers_for_sector(sector: Optional[str], max_articles: int = 6, max_tickers: int = 10) -> Dict[str, Any]:
    """Use news headlines to extract currently mentioned tickers for a sector/theme."""
    _log_tool_call("find_hot_tickers_for_sector", sector=sector, max_articles=max_articles, max_tickers=max_tickers)
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
    _log_tool_call("get_index_constituents", index=index_symbol)
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
    _log_tool_call("get_company_profile", ticker=ticker)
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


def discover_candidates(preferences: Dict[str, Any], max_candidates: int = 10) -> List[str]:
    """Discover tickers dynamically via APIs (no hardcoded list).
    Strategy:
      1) Choose index by theme (Technology -> ^NDX, otherwise -> ^GSPC)
      2) Pull constituents once
      3) Filter by country==US and sector keyword (if provided)
      4) Return up to max_candidates tickers
    """
    _log_tool_call("discover_candidates", max_candidates=max_candidates)
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


def _get_allocation_ssot(state: MessagesState) -> tuple[Mapping[str, float], Mapping[str, str]]:
    """
    Return read-only views over PICKS_JSON allocation data.
    The allocation mapping acts as the single source of truth (SSOT) during rendering.
    """
    picks = _get_json(state, "PICKS_JSON")
    raw_alloc = picks.get("allocation") or {}
    raw_rationale = picks.get("rationale") or {}
    if not isinstance(raw_alloc, dict) or not raw_alloc:
        return MappingProxyType({}), MappingProxyType({})
    normalized_alloc: Dict[str, float] = {}
    for ticker, weight in raw_alloc.items():
        try:
            normalized_alloc[str(ticker)] = float(weight)
        except Exception:
            continue
    normalized_rationale: Dict[str, str] = {}
    for ticker in normalized_alloc:
        normalized_rationale[ticker] = str(raw_rationale.get(ticker, ""))[:200]
    return MappingProxyType(normalized_alloc), MappingProxyType(normalized_rationale)


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
TEACH_KWS = [
    "what is", "explain", "difference between", "how to", "why",
    "RAG", "embedding", "retrieval", "greedy", "sampling", "temperature",
    "prompt", "few-shot", "chain of thought", "vector", "retrieval basics", "knowledge base", "explain", "definition", "principle"
]

def _looks_like_teaching(msg: str) -> bool:
    s = (msg or "").lower()
    return any(kw in s for kw in TEACH_KWS)

def teach_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    latest_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_q = (latest_user.content if isinstance(latest_user, HumanMessage) else "").strip()
    latest_plan = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    plan_text = str(latest_plan.content).strip() if isinstance(latest_plan, AIMessage) else ""

    jargon_data = _get_json(state, "JARGON_JSON")
    jargon_terms: List[str] = []
    jargon_notes: Dict[str, str] = {}
    if isinstance(jargon_data, dict):
        for term in jargon_data.get("terms") or []:
            text = str(term).strip()
            if text and text not in jargon_terms:
                jargon_terms.append(text[:80])
        notes_data = jargon_data.get("notes")
        if isinstance(notes_data, dict):
            for key, val in notes_data.items():
                jargon_notes[str(key)] = str(val)[:160]

    topic_source = user_q.strip() or plan_text.strip() or "Investing fundamentals"
    topic_lines = [line.strip() for line in topic_source.splitlines() if line.strip()]
    fallback_topic = next(
        (
            line for line in topic_lines
            if not re.match(r"^[A-Z_]+_JSON\s*:", line)
        ),
        topic_lines[0] if topic_lines else "Investing fundamentals",
    )[:80]
    teaching_topics = [t for t in jargon_terms[:3] if t]
    if not teaching_topics:
        teaching_topics = [fallback_topic]
    combined_context = (f"User question:\n{user_q}\n\nModel response:\n{plan_text}").strip()

    sections: List[str] = []
    tool_errors: List[str] = []

    if plan_text:
        sections.append("### Investment Plan\n" + plan_text)
    if jargon_terms:
        vocab_lines = ["### Vocabulary Highlights"]
        for term in teaching_topics:
            note = jargon_notes.get(term)
            if note:
                vocab_lines.append(f"- {term}: {note}")
            else:
                vocab_lines.append(f"- {term}")
        sections.append("\n".join(vocab_lines))

    def _safe_call(label: str, func):
        try:
            return func()
        except Exception as exc:
            tool_errors.append(f"{label}: {exc}")
            return None

    try:
        teaching_llm = build_llm()
    except Exception:
        teaching_llm = None

    for idx, topic in enumerate(teaching_topics):
        concept_res = _safe_call(
            f"concept[{topic}]",
            lambda topic=topic: generate_concept_card(
                term=topic,
                user_question=combined_context,
                language="en",
                teach_db_dir=None,
                llm=teaching_llm,
            ),
        )
        if isinstance(concept_res, dict) and concept_res.get("card"):
            card = concept_res["card"]
            keywords = card.get("keywords") or []
            cites = card.get("citations") or []
            title = "### Concept Card" if idx == 0 else f"### Concept Card: {topic}"
            section_lines = [
                title,
                f"**Term:** {card.get('term') or topic}",
                f"**Definition:** {card.get('definition') or 'N/A'}",
                f"**Intuition:** {card.get('intuition') or 'N/A'}",
                f"**Why it matters:** {card.get('why_it_matters') or 'N/A'}",
            ]
            if keywords:
                section_lines.append("**Keywords:** " + ", ".join(keywords[:6]))
            if cites:
                cite_lines = [f"[{c.get('title') or c.get('source') or ''}] {c.get('url') or ''}" for c in cites[:3]]
                section_lines.append("**Citations:** " + " | ".join(cite_lines))
            sections.append("\n".join(section_lines))

    quiz_topic = teaching_topics[0]
    quiz_res = _safe_call(
        "quiz",
        lambda: generate_quiz_card(
            term=quiz_topic,
            language="en",
            teach_db_dir=None,
            llm=teaching_llm,
        ),
    )
    if isinstance(quiz_res, dict) and quiz_res.get("card"):
        card = quiz_res["card"]
        questions = card.get("questions") or []
        quiz_lines = [
            "### Quiz",
            f"**Term:** {card.get('term') or quiz_topic}",
        ]
        for idx, q in enumerate(questions[:3], 1):
            q_text = q.get("question") or "Question"
            quiz_lines.append(f"{idx}. {q_text}")
            if q.get("type") == "single":
                options = q.get("options") or []
                quiz_lines.append(
                    "   Options: " + "; ".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
                )
                ans_idx = q.get("answer")
                if isinstance(ans_idx, int) and 0 <= ans_idx < len(options):
                    quiz_lines.append(f"   Answer: {chr(65+ans_idx)}")
            elif q.get("type") == "bool":
                quiz_lines.append(f"   Answer: {q.get('answer')}")
            explanation = q.get("explanation")
            if explanation:
                quiz_lines.append(f"   Why: {explanation}")
        sections.append("\n".join(quiz_lines))

    if sections:
        teaching_text = "📚 InvestMate Learning Pack\n\n" + "\n\n".join(sections)
        if tool_errors:
            teaching_text += "\n\n⚠️ Tool notes:\n" + "\n".join(f"- {msg}" for msg in tool_errors)
        return {"messages": [AIMessage(content=teaching_text)]}

    fallback_text = (
        "InvestMate could not assemble an education pack from the latest exchange. "
        "Please provide more details about what you would like to learn."
    )
    return {"messages": [AIMessage(content=fallback_text)]}

def intent_gateway_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()
    latest_user = next(
        (
            m for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
            and not str(m.content).startswith(("USER_PROFILE_JSON", "GATEWAY_JSON"))
        ),
        None,
    )
    user_text = str(latest_user.content) if isinstance(latest_user, HumanMessage) else ""

    sys = SystemMessage(content=(
        "You are an intent gateway for an investment assistant.\n"
        "Analyze the latest user message and decide whether it requests or implies financial/investment/market related assistance.\n"
        "Respond ONLY with a JSON object using this schema:\n"
        "{\n  \"is_finance\": true|false,\n  \"confidence\": \"high|medium|low\",\n  \"reason\": \"<=40 words explanation\"\n}\n"
        "Classify as finance when the user asks about investments, markets, portfolios, companies, macro data, or financial planning."
    ))
    human = HumanMessage(content=f"LATEST_USER_MESSAGE: {user_text}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "gateway")

    parsed = _extract_json_from_text(ai.content) or {}
    is_finance = bool(parsed.get("is_finance", True))
    confidence = str(parsed.get("confidence", "unknown"))
    reason = str(parsed.get("reason", ai.content)).strip()

    payload = {
        "is_finance": bool(is_finance),
        "confidence": confidence,
        "reason": reason,
    }
    if not _TERMINAL_SIMPLE:
        _builtins.print(f"[gateway-decision] is_finance={bool(is_finance)} confidence={confidence} reason={reason}")
    return {"messages": [HumanMessage(content=f"GATEWAY_JSON: {json.dumps(payload, ensure_ascii=False)}")]} 


def profile_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    profile = get_user_profile(user_id="demo_user")
    return {"messages": [HumanMessage(content=f"USER_PROFILE_JSON: {json.dumps(profile, ensure_ascii=False)}")]}


def general_chat_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()
    latest_user = next(
        (
            m for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
            and not str(m.content).startswith(("USER_PROFILE_JSON", "GATEWAY_JSON"))
        ),
        None,
    )
    user_text = str(latest_user.content) if isinstance(latest_user, HumanMessage) else ""
    gateway = _get_json(state, "GATEWAY_JSON")
    reason = gateway.get("reason") if isinstance(gateway, dict) else ""

    sys = SystemMessage(content=(
        "You are InvestMate, a friendly conversational assistant whose primary mission is to provide investment guidance and financial literacy.\n"
        "The current prompt has been classified as non-financial, so respond in a warm, conversational tone while gently reminding the user that you specialize in finance, markets, and personal investing.\n"
        "Invite them to ask about financial goals, market trends, portfolio ideas, or any money-related questions. Avoid giving unsolicited advice unrelated to finance, but keep the conversation engaging and helpful."
    ))
    human = HumanMessage(content=f"USER_MESSAGE: {user_text}\nCLASSIFICATION_REASON: {reason}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "general-chat")

    return {"messages": [AIMessage(content=ai.content)]}


def plan_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    llm = build_llm()
    sys = SystemMessage(content=(
        "You are a planner for an investment assistant.\n"
        "Use USER_PROFILE_JSON and the user's most recent message. Identify target sectors/themes from the user's message; if absent, fall back to USER_PROFILE_JSON.preferences.sectors.\n"
        "Constrain the universe to U.S.-listed equities.\n"
        "Always propose 8-10 candidate tickers that best match the intent (valid US-listed tickers only). Prefer real symbols over ETFs; avoid duplicates.\n"
        "Return ONLY ONE fenced JSON block with the EXACT schema and NOTHING ELSE:\n"
        "```json\n"
        
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
        plan_json["tickers"] = candidates[:10]
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
        hot = _find_hot_tickers_for_sector(sector, max_articles=6, max_tickers=10)
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
    risk_data = facts.get("risk", {}) if isinstance(facts, dict) else {}
    risk_metrics = risk_data.get("metrics") or {}
    risk_kept = set(risk_data.get("kept") or [])
    risk_dropped = risk_data.get("dropped") or {}

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
            "risk_metrics": risk_metrics.get(t) or {},
            "risk_screen": {
                "kept": t in risk_kept,
                "dropped_reason": risk_dropped.get(t),
            },
        }

    risk_level = str((profile or {}).get("risk_level", "medium") or "medium").lower()
    human_payload = {
        "target_allocation": target,
        "cap": cap,
        "profile": profile,
        "candidates": selection_features,
    }
    human = HumanMessage(content=f"SELECTION_INPUT_JSON: {json.dumps(human_payload, ensure_ascii=False)}")
    # Risk-screen the LLM output afterwards instead of adjusting prompt
    thresholds = {
        "low": {"beta": 1.05, "ann_vol": 0.28, "mdd_6m": 0.12},
        "medium": {"beta": 1.30, "ann_vol": 0.40, "mdd_6m": 0.20},
        "high": {"beta": None, "ann_vol": None, "mdd_6m": None},
    }
    risk_limit = thresholds.get(risk_level, thresholds["medium"])
    sys = SystemMessage(content=(
        "You are an investment selector.\n"
        "Review the candidate tickers, their scores, qualitative insights, prices, and risk metrics.\n"
        "Respect the user's profile (risk, ESG, sector focus) and keep diversification in mind.\n"
        "Choose between 3 and 6 tickers (or fewer if the universe is smaller) prioritizing higher scores, "
        "but you may include lower-score names when qualitative factors justify it.\n"
        "For every inclusion/exclusion, provide succinct rationale text that fits within the specified schema.\n"
        "Return ONLY one fenced JSON:\n"
        "```json\n{\n  \"chosen\": [\"TICKER\", ...],\n  \"rationale\": {\"TICKER\": \"<=25 words inclusion reason\"},\n"
        "  \"excluded\": {\"TICKER\": \"<=18 words exclusion reason\"}\n}\n```"
    ))
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "select")

    parsed = _extract_json_from_text(ai.content) or {}
    raw_chosen = parsed.get("chosen") if isinstance(parsed.get("chosen"), list) else []
    rationale = parsed.get("rationale") if isinstance(parsed.get("rationale"), dict) else {}
    excluded = parsed.get("excluded") if isinstance(parsed.get("excluded"), dict) else {}

    chosen = [t for t in raw_chosen if isinstance(t, str) and t in tickers]
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
    risk_excluded: Dict[str, Dict[str, str]] = {}
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

    def _fails_screen(metrics: Dict[str, Any]) -> Optional[str]:
        if not metrics:
            return None
        beta = metrics.get("beta")
        ann_vol = metrics.get("ann_vol")
        mdd = metrics.get("mdd_6m")
        if risk_limit.get("beta") is not None and isinstance(beta, (int, float)) and beta > risk_limit["beta"]:
            return "beta_too_high"
        if risk_limit.get("ann_vol") is not None and isinstance(ann_vol, (int, float)) and ann_vol > risk_limit["ann_vol"]:
            return "vol_too_high"
        if risk_limit.get("mdd_6m") is not None and isinstance(mdd, (int, float)) and mdd > risk_limit["mdd_6m"]:
            return "drawdown_too_high"
        return None

    filtered_chosen = []
    for t in chosen:
        fail_reason = _fails_screen(risk_metrics.get(t) or {})
        if fail_reason:
            message = {
                "beta_too_high": "Beta above threshold",
                "vol_too_high": "Volatility above threshold",
                "drawdown_too_high": "Drawdown above threshold",
            }.get(fail_reason, "Risk screen failed")
            risk_excluded[t] = {"code": fail_reason, "message": message}
            excluded_out[t] = message
            print(f"[risk-screen] exclude {t}: {message}")
        else:
            filtered_chosen.append(t)

    chosen = filtered_chosen

    if not chosen:
        eligible = [
            t for t in tickers
            if _fails_screen(risk_metrics.get(t) or {}) is None
        ]
        eligible_scores = sorted(
            ((t, scores.get(t, 50)) for t in eligible),
            key=lambda x: x[1],
            reverse=True,
        )
        limit = min(len(eligible_scores), max(3, min(6, len(eligible_scores))))
        chosen = [t for t, _ in eligible_scores[:limit]]

    # Top up with additional low-risk names if we cannot hit the target under caps
    eligible_sorted = [
        t for t, _ in sorted(
            ((tt, scores.get(tt, 50)) for tt in tickers if _fails_screen(risk_metrics.get(tt) or {}) is None),
            key=lambda x: x[1],
            reverse=True,
        )
    ]
    if cap > 1e-9:
        required_by_cap = int(math.ceil(max(target, 0.0) / cap))
    else:
        required_by_cap = len(eligible_sorted)
    base_required = 3 if len(eligible_sorted) >= 3 else len(eligible_sorted)
    min_required = min(len(eligible_sorted), max(base_required, required_by_cap))
    for candidate in eligible_sorted:
        if len(chosen) >= min_required or len(chosen) >= 6:
            break
        if candidate not in chosen:
            chosen.append(candidate)

    for t in chosen:
        if t not in rationale_out:
            base = notes.get(t) or (insight_map.get(t, {}) or {}).get("summary")
            rationale_out[t] = (base or "Added to reach target allocation within risk limits")[:200]
        excluded_out.pop(t, None)


    payload = {
        "chosen": chosen,
        "rationale": rationale_out,
        "excluded": excluded_out,
        "risk_excluded": risk_excluded,
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
    profile = _get_json(state, "USER_PROFILE_JSON")
    detected = _get_json(state, "TICKER_DETECT_JSON")
    detected_tickers = [str(t).upper() for t in (detected.get("tickers") or []) if isinstance(t, str)]
    hotlist = _get_json(state, "HOTLIST_JSON")
    hot_tickers = [
        str(t) for t in (hotlist.get("all_tickers") or []) if isinstance(t, str)
    ]
    plan_sectors: List[str] = []
    hot_sectors = [str(s) for s in (hotlist.get("requested_sectors") or []) if isinstance(s, str)]
    if last_ai:
        data = _extract_json_from_text(last_ai.content)
        if isinstance(data, dict):
            tickers = [t for t in data.get("tickers", []) if isinstance(t, str)]
            sectors = data.get("target_sectors") or []
            if isinstance(sectors, list):
                plan_sectors = [str(s) for s in sectors if isinstance(s, str) and s]
            tmp_alloc = data.get("target_extra_allocation", target_alloc)
            try:
                target_alloc = float(tmp_alloc) if tmp_alloc is not None else None
            except Exception:
                target_alloc = None                             
            max_single = float(data.get("max_single_weight", max_single))
    if not tickers and detected_tickers:
        tickers = detected_tickers[:10]
    if not tickers and hot_tickers:
        tickers = hot_tickers[:10]
    if not tickers:
        # Discover dynamically from APIs (no hardcoded lists)
        profile = _get_json(state, "USER_PROFILE_JSON")
        prefs = (profile or {}).get("preferences", {})
        tickers = discover_candidates(prefs, max_candidates=10)
    if not tickers and plan_sectors:
        tickers = _fallback_sector_tickers(plan_sectors, limit=10)
    if not tickers and hot_sectors:
        tickers = _fallback_sector_tickers(hot_sectors, limit=10)
    if not tickers and hot_tickers:
        tickers = hot_tickers[:10]
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

    risk_result: Dict[str, Any] = {}
    if tickers:
        risk_level_pref = str((profile or {}).get("risk_level", "medium") or "medium").lower()
        try:
            _log_tool_call("get_risk_assessment", tickers=",".join(tickers), risk_level=risk_level_pref)
            risk_result = get_risk_assessment(tickers, risk_level=risk_level_pref)
        except Exception as exc:
            try:
                print(f"[risk] risk assessment failed: {exc}")
            except Exception:
                pass
            risk_result = {"error": str(exc), "tickers": tickers, "risk_level": risk_level_pref}

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
                    published_display = _format_news_timestamp(art_dict.get("published"))
                    if published_display:
                        art_dict["published_display"] = published_display
                    preview_with_time = preview
                    if published_display:
                        if preview_with_time:
                            preview_with_time = f"[{published_display}] {preview_with_time}"
                        else:
                            preview_with_time = f"[{published_display}] {art_dict.get('title') or ''}"
                    if preview_with_time:
                        preview_pool.append(preview_with_time)
                    art_dict["preview"] = preview_with_time
                    enriched_articles.append(art_dict)
                if preview_pool:
                    summary_text = " ".join(preview_pool[:3])
                else:
                    summary_text = _news_preview(" ".join(
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
        prof = profile or {}
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
        "risk": risk_result,
    }
    return {"messages": [HumanMessage(content=f"FACTS_JSON: {json.dumps(facts, ensure_ascii=False)}")]}


def synthesize_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    """
    Deterministic finalization:
    - Use the PICKS_JSON allocation as the SSOT and keep it immutable during rendering.
    - Ask LLM only to write one-line rationale & one key risk per ticker, given news/analyst snippets.
    - Then compose the final human-readable plan + JSON using our fixed weights.
    """
    llm = build_llm()

    facts = _get_json(state, "FACTS_JSON")
    allocation_ssot, rationale_ssot = _get_allocation_ssot(state)
    analysis = _get_json(state, "ANALYSIS_JSON")
    selection = _get_json(state, "SELECTION_JSON")
    profile = _get_json(state, "USER_PROFILE_JSON")
    scores = analysis.get("scores", {}) if isinstance(analysis, dict) else {}
    selection_rationale = selection.get("rationale", {}) if isinstance(selection, dict) else {}
    allocation_rationale = rationale_ssot
    if not allocation_ssot:
        return {"messages": [AIMessage(content="Unable to render summary: PICKS_JSON allocation missing.")]}

    tickers: List[str] = list(allocation_ssot.keys())
    target = float(facts.get("target_extra_allocation") or 0.0)
    cap = float(facts.get("max_single_weight", 0.15))
    news = facts.get("news", {})
    analyst = facts.get("analyst", {})
    prices = facts.get("prices", {}).get("prices", {})
    risk_data = facts.get("risk", {}) if isinstance(facts, dict) else {}
    risk_metrics = risk_data.get("metrics") or {}
    risk_dropped = risk_data.get("dropped") or {}
    selection_risk_excluded = selection.get("risk_excluded") if isinstance(selection.get("risk_excluded"), dict) else {}
    risk_level = str((profile or {}).get("risk_level", "medium") or "medium").lower()

    # Log a one-liner for the final chosen weights
    try:
        total = sum(float(allocation_ssot.get(t, 0.0)) for t in tickers)
        print(f"[debug] final_weights_sum={total:.6f} target={target:.6f} tickers={','.join(tickers)}")
    except Exception:
        pass

    if not tickers:
        return {"messages": [AIMessage(content="No tickers present in PICKS_JSON allocation.")]}

    source_tickers = set(allocation_ssot.keys())
    if set(tickers) != source_tickers:
        raise ValueError("Rendered tickers diverge from PICKS_JSON allocation keys.")

    total_weight = sum(float(allocation_ssot.get(t, 0.0)) for t in tickers)
    if total_weight - target > 1e-6:
        raise ValueError(f"Allocation total {total_weight:.8f} exceeds target {target:.8f}")
    shortfall = max(0.0, target - total_weight)
    for t, w in allocation_ssot.items():
        if w < -1e-12 or w - cap > 1e-12:
            raise ValueError(f"Allocation for {t} = {w} outside [0, {cap}]")

    risk_compromise = False
    risk_compromise_notes: Dict[str, str] = {}
    if risk_level == "low":
        thresholds = {"beta": 1.05, "ann_vol": 0.28, "mdd_6m": 0.12}
        for t in tickers:
            metrics = risk_metrics.get(t) or {}
            beta = metrics.get("beta")
            ann_vol = metrics.get("ann_vol")
            mdd = metrics.get("mdd_6m")
            breaches: List[str] = []
            if isinstance(beta, (int, float)) and beta > thresholds["beta"] + 1e-9:
                breaches.append(f"beta {beta:.2f} > {thresholds['beta']:.2f}")
            if isinstance(ann_vol, (int, float)) and ann_vol > thresholds["ann_vol"] + 1e-9:
                breaches.append(f"ann vol {ann_vol:.2f} > {thresholds['ann_vol']:.2f}")
            if isinstance(mdd, (int, float)) and mdd > thresholds["mdd_6m"] + 1e-9:
                breaches.append(f"mdd6m {mdd:.2f} > {thresholds['mdd_6m']:.2f}")
            if breaches:
                risk_compromise_notes[t] = "; ".join(breaches)
        if risk_compromise_notes:
            if len(risk_compromise_notes) < len(tickers):
                offenders = ", ".join(f"{t}: {txt}" for t, txt in risk_compromise_notes.items())
                raise ValueError(f"Low-risk screen failed for subset of tickers: {offenders}")
            risk_compromise = True

    # Ask LLM *only* for rationale & risk text (no weights)
    rationale_sys = SystemMessage(content=(
        "You are a concise investment writer.\n"
        "For each ticker you are given, write AT LEAST THREE bullet-point rationales (each <=25 words) using the provided 3-news titles, analyst snippets, and selection notes.\n"
        "Also provide exactly one key risk (<=12 words).\n"
        "Return ONLY a fenced JSON with this schema:\n"
        "```json\n{\n  \"rationales\": {\"AAPL\": [\"reason1\", \"reason2\", \"reason3\"], ...},\n  \"risks\": {\"AAPL\": \"...\", ...}\n}\n```\n"
        "Ensure each ticker has at least three distinct rationale entries."
    ))
    mini_facts = {
        "tickers": tickers,
        "snippets": {
            t: {
                "news_titles": " | ".join([a.get("title") or "" for a in (news.get(t, {}) or {}).get("articles", [])][:3]),
                "news_summary": (news.get(t, {}) or {}).get("summary", ""),
                "analyst_snippets": " | ".join([r.get("snippet") or "" for r in (analyst.get(t, {}) or {}).get("reports", [])][:3]),
                "selection_rationale": selection_rationale.get(t, ""),
                "risk_metrics": risk_metrics.get(t) or {},
                "risk_screen": selection_risk_excluded.get(t) if isinstance(selection_risk_excluded.get(t), dict) else risk_dropped.get(t),
            } for t in tickers
        }
    }
    msgs = state["messages"] + [rationale_sys, HumanMessage(content=f"FACTS_MINI_JSON: {json.dumps(mini_facts, ensure_ascii=False)}")]
    ai = _llm_invoke_with_log(llm, msgs, "synthesize-rationale")
    parsed = _extract_json_from_text(ai.content) or {}
    raw_ratios = parsed.get("rationales", {}) if isinstance(parsed.get("rationales"), dict) else {}
    rationales: Dict[str, List[str]] = {}
    for t in tickers:
        entry = raw_ratios.get(t)
        if isinstance(entry, list):
            clean = [str(r).strip()[:80] for r in entry if str(r).strip()]
            if len(clean) >= 3:
                rationales[t] = clean[:6]
        elif isinstance(entry, str):
            clean = [part.strip() for part in entry.split(";") if part.strip()]
            if len(clean) >= 3:
                rationales[t] = clean[:6]
    risks: Dict[str, str] = parsed.get("risks", {}) if isinstance(parsed.get("risks"), dict) else {}

    # Compose human-readable plan using our fixed weights
    lines = []
    rendered_tickers: set[str] = set()
    for t in tickers:
        w = float(allocation_ssot.get(t, 0.0))
        rendered_tickers.add(t)
        titles = [a.get("title") for a in (news.get(t, {}) or {}).get("articles", [])][:3]
        pt = ((analyst.get(t, {}) or {}).get("reports") or [{}])[0].get("pt")
        px = (prices.get(t) or {}).get("price")
        article_slice = (news.get(t, {}) or {}).get("articles", [])[:2]
        previews = [
            (a.get("preview") or _news_preview(a.get("description") or a.get("content") or a.get("title") or ""))
            for a in article_slice
        ]
        formatted_previews: List[str] = []
        for a, p in zip(article_slice, previews):
            text = p
            if isinstance(a, dict):
                ts = _format_news_timestamp(a.get("published"))
                if ts:
                    if text and not str(text).startswith("["):
                        text = f"[{ts}] {text}"
                    elif not text:
                        text = f"[{ts}]"
            formatted_previews.append(text)
        previews = formatted_previews
        news_digest = " ".join([p for p in previews if p])
        if not news_digest:
            news_digest = _news_preview(" / ".join(titles)) or "No recent news summary available"
        news_summary_text = (news.get(t, {}) or {}).get("summary") or news_digest

        selection_reason = selection_rationale.get(t, "").strip()
        analysis_points = rationales.get(t) or []
        if not analysis_points:
            fallback_point = selection_reason or (" / ".join(titles) or "News flow is positive; analyst commentary is supportive")
            analysis_points = [fallback_point]
        risk_text = (risks.get(t) or "Sector volatility and regulatory oversight").strip()
        alloc_reason = (allocation_rationale.get(t) or "Allocation driven by scores and diversification constraints").strip()
        score_val = scores.get(t, "N/A")

        detail_lines: List[str] = []
        if selection_reason:
            detail_lines.append(f"Stock selection: {selection_reason}")
        for idx, point in enumerate(analysis_points, 1):
            label = "Model analysis" if idx == 1 else f"Analysis {idx}"
            detail_lines.append(f"{label}: {point}")
        if alloc_reason and alloc_reason not in analysis_points:
            detail_lines.append(f"Weight rationale: {alloc_reason}")
        if news_summary_text:
            detail_lines.append(f"News summary: {news_summary_text}")
        risk_metrics_t = risk_metrics.get(t) or {}
        risk_desc_parts: List[str] = []
        ann_vol = risk_metrics_t.get("ann_vol")
        beta = risk_metrics_t.get("beta")
        mdd = risk_metrics_t.get("mdd_6m")
        ret6 = risk_metrics_t.get("ret_6m")
        if isinstance(ann_vol, (int, float)):
            risk_desc_parts.append(f"ann vol {ann_vol:.2f}")
        if isinstance(beta, (int, float)):
            risk_desc_parts.append(f"beta {beta:.2f}")
        if isinstance(mdd, (int, float)):
            risk_desc_parts.append(f"mdd6m {mdd:.2f}")
        if isinstance(ret6, (int, float)):
            risk_desc_parts.append(f"ret6m {ret6:.2f}")
        drop_reason = risk_dropped.get(t)
        if drop_reason:
            drop_msg = {
                "beta_too_high": "Beta exceeds profile limit",
                "vol_too_high": "Volatility exceeds profile limit",
                "drawdown_too_high": "Drawdown exceeds profile limit",
            }.get(drop_reason, str(drop_reason))
            risk_desc_parts.append(f"risk screen: {drop_msg}")
        sel_risk = selection_risk_excluded.get(t)
        if isinstance(sel_risk, dict) and sel_risk.get("message"):
            risk_desc_parts.append(f"selection screen: {sel_risk['message']}")
        if risk_desc_parts:
            detail_lines.append(f"Risk view: {'; '.join(risk_desc_parts)}")
        if risk_text:
            detail_lines.append(f"Key risks: {risk_text}")
        if risk_compromise_notes.get(t):
            detail_lines.append(
                f"⚠️ Low-risk policy breached: {risk_compromise_notes[t]} (profile risk={risk_level})."
            )

        bullet_body = "\n".join(f"  - {dl}" for dl in detail_lines if dl)
        lines.append(
            f"- **{t}**: {w*100:.2f}% (~price {px if px is not None else 'N/A'}; PT {pt if pt is not None else 'N/A'}; score {score_val})\n"
            f"{bullet_body}"
        )

    if rendered_tickers != source_tickers:
        raise ValueError("Rendered tickers diverge from PICKS_JSON allocation keys.")

    # Build narrative summary
    bullets = "\n".join(lines)
    intro = "Here is your incremental allocation proposal in natural language."
    if shortfall > 1e-6:
        intro += (
            f" Risk filters and the {cap*100:.2f}% per-name cap limited deployable capital to "
            f"{total_weight*100:.2f}%, leaving {shortfall*100:.2f}% unallocated."
        )
    if risk_compromise:
        intro += (
            " ⚠️ All available tickers breach the low-risk thresholds; proceeding with disclosure as it conflicts with your "
            f"{risk_level} risk preference."
        )
    intro += " The weights obey the per-name cap and reflect the surviving low-risk candidates:\n"
    human_readable = intro + bullets
    return {"messages": [AIMessage(content=human_readable)]}


def jargon_node(state: MessagesState):
    print(f"[call] {inspect.currentframe().f_code.co_name}")
    latest_plan = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    plan_text = str(latest_plan.content or "").strip() if isinstance(latest_plan, AIMessage) else ""

    if not plan_text:
        payload = {"terms": [], "notes": {}, "source": "synthesize", "reason": "empty_plan_text"}
        return {"messages": [HumanMessage(content=f"JARGON_JSON: {json.dumps(payload, ensure_ascii=False)}")]}

    llm = build_llm()
    sys = SystemMessage(content=(
        "You are a financial writing editor.\n"
        "Review the provided investment summary and identify the single most jargon-heavy or potentially obscure finance term or short phrase.\n"
        "The chosen item must appear verbatim in the text and be relevant to finance. If no such item exists, return an empty list.\n"
        "Return ONLY one fenced JSON with this schema:\n"
        "```json\n{\n  \"terms\": [\"term\"],\n  \"notes\": {\"term\": \"<=25 words why it's tricky\"}\n}\n```"
    ))
    human = HumanMessage(content=f"INVESTMENT_SUMMARY:\n{plan_text}")
    ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "jargon")
    parsed = _extract_json_from_text(ai.content) or {}

    raw_terms = parsed.get("terms") if isinstance(parsed.get("terms"), list) else []
    terms: List[str] = []
    for term in raw_terms:
        t = str(term).strip()
        if t:
            terms.append(t[:80])
            break

    notes_raw = parsed.get("notes") if isinstance(parsed.get("notes"), dict) else {}
    notes: Dict[str, str] = {}
    for term in terms:
        note = notes_raw.get(term)
        if note:
            notes[term] = str(note).strip()[:160]

    if terms:
        candidate = terms[0]
        check_sys = SystemMessage(content=(
            "You are a domain classifier. Reply with either 'yes' or 'no'.\n"
            "'yes' means the term is related to finance, investing, or capital markets.\n"
            "'no' means it is not primarily financial jargon."
        ))
        check_human = HumanMessage(content=f"Term: {candidate}\nContext:\n{plan_text}")
        try:
            verdict = _llm_invoke_with_log(llm, [check_sys, check_human], "jargon-verify")
            answer = (verdict.content or "").strip().lower()
            if not answer.startswith("y"):
                terms = []
                notes = {}
        except Exception:
            pass

    payload = {
        "terms": terms,
        "notes": notes,
        "source": "synthesize",
    }
    return {"messages": [HumanMessage(content=f"JARGON_JSON: {json.dumps(payload, ensure_ascii=False)}")]}

def _start_router(state: MessagesState) -> str:
    latest = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    txt = latest.content if (latest and isinstance(latest.content, str)) else ""
    return "teach" if _looks_like_teaching(txt) else "profile"

# =============================================
# GRAPH (build)
# =============================================
def build_agent():
    g = StateGraph(MessagesState)

    # Node registration (following the original flow)
    g.add_node("teach", teach_node)
    g.add_node("profile", profile_node)
    g.add_node("gateway", intent_gateway_node)
    g.add_node("general_chat", general_chat_node)
    g.add_node("intent", intent_node)
    g.add_node("plan", plan_node)
    g.add_node("hotlist", hotlist_node)
    g.add_node("toolplan", tool_plan_node)
    g.add_node("gather", gather_node)
    g.add_node("insights", insights_node)
    g.add_node("analyze", analyze_node)
    g.add_node("select", select_node)
    g.add_node("allocate", allocate_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("jargon", jargon_node)

    # Route START to either the teaching or primary investment branch
    g.add_conditional_edges(
        START,
        _start_router,
        {
            "teach": "teach",
            "profile": "profile",
        },
    )

    # Main pipeline
    g.add_edge("profile", "gateway")
    g.add_conditional_edges(
        "gateway",
        _gateway_router,
        {
            "finance": "intent",
            "general": "general_chat",
        },
    )
    g.add_edge("general_chat", END)
    g.add_edge("intent", "plan")
    g.add_edge("plan", "hotlist")
    g.add_edge("hotlist", "toolplan")
    g.add_edge("toolplan", "gather")
    g.add_edge("gather", "insights")
    g.add_edge("insights", "analyze")
    g.add_edge("analyze", "select")
    g.add_edge("select", "allocate")
    g.add_edge("allocate", "synthesize")
    g.add_edge("synthesize", "jargon")
    g.add_edge("jargon", "teach")
    g.add_edge("teach", END)

    return g.compile()


# =============================================
# Public helpers for integration
# =============================================
def get_agent():
    """Return a cached LangGraph agent instance."""
    global _AGENT_INSTANCE
    if _AGENT_INSTANCE is None:
        with _AGENT_LOCK:
            if _AGENT_INSTANCE is None:
                _AGENT_INSTANCE = build_agent()
    return _AGENT_INSTANCE


def _prepare_messages(user_query: str, history: Optional[List[Dict[str, str]]] = None) -> List[Any]:
    """Convert optional chat history into LangChain message objects and append the new user input."""
    prepared: List[Any] = []
    for item in history or []:
        if isinstance(item, (HumanMessage, AIMessage, SystemMessage)):
            prepared.append(item)
            continue
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            continue
        content_str = str(content)
        if not content_str.strip():
            continue
        role = str(item.get("role", "")).lower()
        if role in {"user", "human"}:
            prepared.append(HumanMessage(content=content_str))
        elif role in {"assistant", "ai", "model"}:
            prepared.append(AIMessage(content=content_str))
        elif role == "system":
            prepared.append(SystemMessage(content=content_str))
    prepared.append(HumanMessage(content=user_query))
    return prepared


def run_agent_sync(user_query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Execute the agent synchronously and return the final synthesized response along with debug events.
    """
    guard = _screen_user_prompt(user_query, history)
    if guard:
        analysis = _generate_guardrail_response(guard)
        return {
            "final": analysis,
            "events": [
                {"node": "guardrail", "role": "ai", "content": analysis},
            ],
            "guardrail": guard,
        }

    analysis_short = _generate_analysis_response(user_query, history)
    if analysis_short:
        return {
            "final": analysis_short,
            "events": [
                {"node": "analysis_short", "role": "ai", "content": analysis_short},
            ],
            "shortcut": {"message": analysis_short},
        }

    user_id = _DEFAULT_PERSONA.get("user_id", "demo_user")
    sanitized_history = _sanitize_history_for_agent(history, _PII_PROTECTION_LEVEL)
    protection = protect_user_message(user_query, user_id, _PII_PROTECTION_LEVEL)
    session_id = protection.get("session_id")
    anonymized_query = protection.get("anonymized_text", user_query)

    agent = get_agent()
    messages = _prepare_messages(anonymized_query, sanitized_history)
    events: List[Dict[str, Any]] = []
    plan_text: Optional[str] = None
    teach_text: Optional[str] = None
    general_text: Optional[str] = None
    try:
        for update in agent.stream({"messages": messages}, stream_mode="updates"):
            for node, ev in update.items():
                payload = ev.get("messages") or []
                for msg in payload:
                    content_raw = getattr(msg, "content", "")
                    if session_id and isinstance(msg, AIMessage):
                        content = restore_llm_response(content_raw, session_id)
                    else:
                        content = content_raw
                    role = getattr(msg, "type", msg.__class__.__name__)
                    events.append({"node": node, "role": role, "content": content})
                    if node == "synthesize" and isinstance(msg, AIMessage):
                        plan_text = content
                    elif node == "teach" and isinstance(msg, AIMessage):
                        teach_text = content
                    elif node == "general_chat" and isinstance(msg, AIMessage):
                        general_text = content
        if plan_text is None and teach_text is None and general_text is None:
            state = agent.invoke({"messages": messages})
            gateway_state = _get_json(state, "GATEWAY_JSON")
            is_finance_branch = not (isinstance(gateway_state, dict) and not gateway_state.get("is_finance", True))
            plan_candidate: Optional[str] = None
            teach_candidate: Optional[str] = None
            general_candidate: Optional[str] = None
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, AIMessage):
                    restored_content = restore_llm_response(msg.content, session_id) if session_id else msg.content
                    if not is_finance_branch and general_candidate is None:
                        general_candidate = restored_content
                        events.append({
                            "node": "general_chat",
                            "role": getattr(msg, "type", "ai"),
                            "content": general_candidate,
                            "source": "invoke",
                        })
                        break
                    if is_finance_branch and teach_candidate is None:
                        teach_candidate = restored_content
                        events.append({
                            "node": "teach",
                            "role": getattr(msg, "type", "ai"),
                            "content": teach_candidate,
                            "source": "invoke",
                        })
                        continue
                    if is_finance_branch:
                        plan_candidate = restored_content
                        events.append({
                            "node": "synthesize",
                            "role": getattr(msg, "type", "ai"),
                            "content": plan_candidate,
                            "source": "invoke",
                        })
                        break
            swapped = False
            if plan_candidate is None and teach_candidate is not None:
                plan_candidate, teach_candidate = teach_candidate, None
                swapped = True
            plan_text = plan_text or plan_candidate
            teach_text = teach_text or teach_candidate
            general_text = general_text or general_candidate
            if swapped:
                for ev in reversed(events):
                    if ev.get("source") == "invoke" and ev.get("node") == "teach":
                        ev["node"] = "synthesize"
                        break
    except Exception as exc:
        return {"final": None, "events": events, "error": str(exc)}
    final_text: Optional[str]
    if plan_text and teach_text:
        final_text = f"{plan_text}\n\n{teach_text}"
    elif plan_text or teach_text:
        final_text = plan_text or teach_text
    else:
        final_text = general_text
    return {"final": final_text, "events": events}


def run_agent_stream(user_query: str, history: Optional[List[Dict[str, str]]] = None):
    """
    Stream responses as the downstream nodes produce them (synthesize plan, followed by teach card).
    """
    guard = _screen_user_prompt(user_query, history)
    if guard:
        yield _generate_guardrail_response(guard)
        return

    analysis_short = _generate_analysis_response(user_query, history)
    if analysis_short:
        yield analysis_short
        return

    user_id = _DEFAULT_PERSONA.get("user_id", "demo_user")
    sanitized_history = _sanitize_history_for_agent(history, _PII_PROTECTION_LEVEL)
    protection = protect_user_message(user_query, user_id, _PII_PROTECTION_LEVEL)
    session_id = protection.get("session_id")
    anonymized_query = protection.get("anonymized_text", user_query)

    agent = get_agent()
    messages = _prepare_messages(anonymized_query, sanitized_history)
    for update in agent.stream({"messages": messages}, stream_mode="updates"):
        for node, ev in update.items():
            if node not in {"synthesize", "teach", "general_chat"}:
                continue
            for msg in ev.get("messages") or []:
                if isinstance(msg, AIMessage):
                    text = msg.content
                    restored = restore_llm_response(text, session_id) if session_id else text
                    yield restored

# =============================================
# DEMO (main)
# =============================================
if __name__ == "__main__":
    proj_hint = os.getenv("PROJ_ID") or os.getenv("PROJECT_ID") or os.getenv("WATSONX_PROJECT_ID")
    if not proj_hint:
        print("[Env] Project ID not found. Set PROJ_ID / PROJECT_ID / WATSONX_PROJECT_ID (e.g., in .env).")
    else:
        print(f"[Env] Project ID detected: {proj_hint[:6]}... (length {len(proj_hint)})")

    agent = get_agent()

    if len(sys.argv) <= 1:
        print("No CLI prompt provided. Run this script with a query argument or use the web frontend.")
        sys.exit(0)
    user_query = " ".join(sys.argv[1:])

    guard_cli = _screen_user_prompt(user_query, history=None)
    if guard_cli:
        print(_generate_guardrail_response(guard_cli))
        sys.exit(0)

    analysis_cli = _generate_analysis_response(user_query, history=None)
    if analysis_cli:
        print(analysis_cli)
        sys.exit(0)

    # Start the conversation with the user's request (PII-protected)
    cli_protection = protect_user_message(user_query, _DEFAULT_PERSONA.get("user_id", "demo_user"), _PII_PROTECTION_LEVEL)
    cli_session = cli_protection.get("session_id")
    sanitized_cli_query = cli_protection.get("anonymized_text", user_query)
    messages = [HumanMessage(content=sanitized_cli_query)]

    final_outputs: List[str] = []
    spinner: Optional[_SimpleSpinner] = None
    if _TERMINAL_SIMPLE:
        _builtins.print(sanitized_cli_query)
        spinner = _SimpleSpinner()
        spinner.start()
    else:
        print("---- Running agent (stream) ----")
    for update in agent.stream({"messages": messages}, stream_mode="updates"):
        for node, ev in update.items():
            if not ev.get("messages"):
                continue
            if _TERMINAL_SIMPLE:
                if node in {"synthesize", "teach", "general_chat"}:
                    for msg in ev["messages"]:
                        if isinstance(msg, AIMessage):
                            restored_content = restore_llm_response(msg.content, cli_session) if cli_session else msg.content
                            if node == "synthesize":
                                final_outputs = [restored_content]
                            elif node == "teach":
                                if final_outputs:
                                    final_outputs.append(restored_content)
                                else:
                                    final_outputs = [restored_content]
                            else:  # general_chat
                                final_outputs = [restored_content]
                            if spinner:
                                spinner.stop()
                continue
            print(f"\n[# {node.upper()} NODE OUTPUT] ->")
            for msg in ev["messages"]:
                if isinstance(msg, AIMessage):
                    restored_content = restore_llm_response(msg.content, cli_session) if cli_session else msg.content
                    print(f"  AI : {restored_content}")
                elif isinstance(msg, HumanMessage):
                    print(f"  USER: {msg.content}")
                elif isinstance(msg, SystemMessage):
                    print(f"  SYS : {msg.content}")
                else:
                    print(f"  {type(msg).__name__}: {getattr(msg,'content',str(msg))}")
    if spinner:
        spinner.stop()
    if _TERMINAL_SIMPLE:
        if final_outputs:
            _builtins.print("\n\n".join(final_outputs))
    else:
        print("---- Done ----")

    
