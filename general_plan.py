# general_plan.py
# -----------------------------------------------------------
# A general "orchestration" node for your LangGraph pipeline.
# It can be invoked after ANY node to:
#   1) Decide where to go next (jump/return),
#   2) Optionally execute tools (prices/news/analyst/risk),
#   3) Patch/extend FACTS_JSON,
#   4) Emit a single ROUTE_PLAN_JSON as SSOT for routing.
# -----------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState

# These helper references are injected from the host application via configure_general_plan().
_get_json = None
_log_tool_call = None
build_llm = None
_llm_invoke_with_log = None
get_prices = None
news_fetcher = None
get_analyst_reports = None
get_risk_assessment = None
discover_candidates = None
_unique_preserve = None
_is_valid_ticker = None
_news_preview = None
_format_news_timestamp = None
_MIN_CANDIDATE_COUNT = 5

REQUIRED_HELPERS = [
    "_get_json",
    "_log_tool_call",
    "build_llm",
    "_llm_invoke_with_log",
    "get_prices",
    "news_fetcher",
    "get_analyst_reports",
    "get_risk_assessment",
    "discover_candidates",
    "_unique_preserve",
    "_is_valid_ticker",
    "_news_preview",
    "_format_news_timestamp",
    "_MIN_CANDIDATE_COUNT",
]


def configure_general_plan(**helpers: Any) -> None:
    """Inject dependencies from the host application."""
    globals().update({name: helpers.get(name, globals().get(name)) for name in REQUIRED_HELPERS})


def _ensure_helpers() -> None:
    missing = [name for name in REQUIRED_HELPERS if globals().get(name) is None]
    if missing:
        raise RuntimeError(f"general_plan helpers not configured: {', '.join(missing)}")


# Public: nodes that can be targeted by the router
POSSIBLE_NODES = {
    "profile", "gateway", "general_chat", "intent", "plan", "hotlist", "toolplan",
    "gather", "candidate_enforcer", "insights", "analyze", "select", "allocate", "synthesize", "jargon",
    "teach", "END"
}

__all__ = [
    "general_plan_node",
    "route_from_general_plan",
    "POSSIBLE_NODES",
    "configure_general_plan",
]


# -----------------------
# Internal small helpers
# -----------------------
def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-recursive merge for small dicts: values from b override a."""
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_next_from_state(state: MessagesState) -> str:
    """Deterministic fallback routing if the LLM plan is missing."""
    facts = _get_json(state, "FACTS_JSON")
    tickers = (facts or {}).get("tickers") or []
    target = (facts or {}).get("candidate_target") or len(tickers)
    if isinstance(target, (int, float)) and len(tickers) < int(target):
        return "candidate_enforcer"
    if _get_json(state, "PICKS_JSON"):
        return "synthesize"
    if _get_json(state, "SELECTION_JSON"):
        return "allocate"
    if _get_json(state, "ANALYSIS_JSON"):
        return "select"
    if _get_json(state, "INSIGHTS_JSON"):
        return "analyze"
    if _get_json(state, "FACTS_JSON"):
        return "insights"
    gw = _get_json(state, "GATEWAY_JSON")
    if isinstance(gw, dict) and not gw.get("is_finance", True):
        return "general_chat"
    return "plan"


def _gpn_pick_tickers(state: MessagesState, limit: int = 10) -> List[str]:
    """Pick a reasonable ticker set for tool execution when the plan didn't specify."""
    facts = _get_json(state, "FACTS_JSON")
    sel   = _get_json(state, "SELECTION_JSON")
    hot   = _get_json(state, "HOTLIST_JSON")

    # 1) chosen → 2) facts.tickers → 3) hotlist → 4) discover_candidates
    if isinstance(sel, dict) and isinstance(sel.get("chosen"), list) and sel["chosen"]:
        ts = [t for t in sel["chosen"] if _is_valid_ticker(t)]
        if ts:
            return ts[:limit]
    if isinstance(facts, dict) and isinstance(facts.get("tickers"), list) and facts["tickers"]:
        ts = [t for t in facts["tickers"] if _is_valid_ticker(t)]
        if ts:
            return ts[:limit]
    if isinstance(hot, dict) and isinstance(hot.get("all_tickers"), list) and hot["all_tickers"]:
        ts = [t for t in hot["all_tickers"] if _is_valid_ticker(t)]
        if ts:
            return ts[:limit]

    profile = _get_json(state, "USER_PROFILE_JSON")
    prefs   = (profile or {}).get("preferences", {})
    ts = discover_candidates(prefs, max_candidates=limit) or []
    return ts[:limit]


# -----------------------
# The Orchestration Node
# -----------------------
def general_plan_node(state: MessagesState):
    """
    A universal planner node that can be called after any node.
    It decides the next hop and (optionally) executes tools before jumping.

    Output messages:
      - FACTS_JSON: (optional) patched facts if tools executed
      - ROUTE_PLAN_JSON: router SSOT, e.g.:
        {
          "decision": {
            "next_node": "insights",
            "return_node": null,
            "reason": "We have FACTS but no INSIGHTS."
          },
          "tools": {
            "execute": true,
            "prices":  {"enabled": true,  "tickers": ["AAPL","MSFT"]},
            "news":    {"enabled": false, "tickers": null, "limit": 2},
            "analyst": {"enabled": false, "tickers": null, "limit": 1},
            "risk":    {"enabled": false, "tickers": null}
          }
        }
    """
    _ensure_helpers()
    print("[call] general_plan_node")
    # Snapshot current status
    facts     = _get_json(state, "FACTS_JSON")
    insights  = _get_json(state, "INSIGHTS_JSON")
    analysis  = _get_json(state, "ANALYSIS_JSON")
    selection = _get_json(state, "SELECTION_JSON")
    picks     = _get_json(state, "PICKS_JSON")
    profile   = _get_json(state, "USER_PROFILE_JSON")
    last_plan = _get_json(state, "ROUTE_PLAN_JSON")  # may contain a pending return_node

    status = {
        "has_facts": bool(facts),
        "has_insights": bool(insights),
        "has_analysis": bool(analysis),
        "has_selection": bool(selection),
        "has_picks": bool(picks),
        "risk_level": (profile or {}).get("risk_level", "medium"),
        "return_to": ((last_plan.get("decision") or {}).get("return_node") if isinstance(last_plan, dict) else None),
        "candidate_count": len((facts or {}).get("tickers") or []),
        "candidate_target": (facts or {}).get("candidate_target"),
    }

    # LLM-plan (optional; falls back to deterministic rule)
    plan: Optional[Dict[str, Any]] = None
    try:
        llm = build_llm()
        sys = SystemMessage(content=(
            "You are an orchestration planner for a LangGraph investing agent.\n"
            "Given which JSON tags are present (FACTS/INSIGHTS/ANALYSIS/SELECTION/PICKS), "
            "decide the next node to run, and whether to fetch more data with tools.\n"
            "Rules:\n"
            "- If PICKS exists → next=synthesize.\n"
            "- Else if SELECTION exists → next=allocate.\n"
            "- Else if ANALYSIS exists → next=select.\n"
            "- Else if INSIGHTS exists → next=analyze.\n"
            "- Else if FACTS exists → next=insights.\n"
            "- Else → next=plan.\n"
            "You MAY request tools (prices/news/analyst/risk) to fill gaps before jumping.\n"
            "Return only one fenced JSON:\n"
            "```json\n"
            "{\n"
            "  \"decision\": {\n"
            "    \"next_node\": \"...\",\n"
            "    \"return_node\": null,\n"
            "    \"reason\": \"<=40 words\"\n"
            "  },\n"
            "  \"tools\": {\n"
            "    \"execute\": false,\n"
            "    \"prices\":  {\"enabled\": true,  \"tickers\": null},\n"
            "    \"news\":    {\"enabled\": false, \"tickers\": null, \"limit\": 2},\n"
            "    \"analyst\": {\"enabled\": false, \"tickers\": null, \"limit\": 1},\n"
            "    \"risk\":    {\"enabled\": false, \"tickers\": null}\n"
            "  }\n"
            "}\n"
        ))
        human = HumanMessage(content="ORCH_STATUS_JSON: " + __import__("json").dumps(status, ensure_ascii=False))
        ai = _llm_invoke_with_log(llm, state["messages"] + [sys, human], "general-plan")
        plan = _extract_json(ai.content)
    except Exception:
        plan = None

    if not isinstance(plan, dict):
        plan = {
            "decision": {
                "next_node": _default_next_from_state(state),
                "return_node": None,
                "reason": "fallback_rule"
            },
            "tools": {
                "execute": False,
                "prices":  {"enabled": True,  "tickers": None},
                "news":    {"enabled": False, "tickers": None, "limit": 2},
                "analyst": {"enabled": False, "tickers": None, "limit": 1},
                "risk":    {"enabled": False, "tickers": None},
            }
        }

    # Parse tool plan
    tools = plan.get("tools") or {}
    do_execute = bool(tools.get("execute", False))

    # Safety override: ensure we gather/enforce when the candidate universe is missing or undersized.
    tickers = (facts or {}).get("tickers") or []
    candidate_target = (facts or {}).get("candidate_target")
    if not isinstance(candidate_target, int):
        candidate_target = max(len(tickers), _MIN_CANDIDATE_COUNT)
    decision = plan.setdefault("decision", {})
    next_node = str(decision.get("next_node") or "").strip()
    if not tickers:
        decision["next_node"] = "gather"
        tools.setdefault("execute", True)
        do_execute = True
    elif len(tickers) < candidate_target and next_node not in {"candidate_enforcer", "gather"}:
        decision["next_node"] = "candidate_enforcer"
    plan["tools"] = tools

    # Enforce required downstream nodes regardless of LLM suggestion.
    has_picks = bool(_get_json(state, "PICKS_JSON"))
    has_selection = bool(_get_json(state, "SELECTION_JSON"))
    if not has_picks and has_selection and decision.get("next_node") != "allocate":
        decision["next_node"] = "allocate"
    elif not has_selection and decision.get("next_node") == "allocate":
        decision["next_node"] = _default_next_from_state(state)

    # Decide working ticker set
    tickers = _collect_tickers_from_tools(tools)
    if not tickers:
        tickers = _gpn_pick_tickers(state, limit=10)

    # Execute tools (optional) and patch FACTS
    facts_patch: Dict[str, Any] = {}
    if do_execute and tickers:
        _log_tool_call("GPN_execute_tools", tickers=",".join(tickers))

        # Prices
        if (tools.get("prices") or {}).get("enabled", True):
            prices = get_prices(tickers)
            facts_patch = _deep_merge(facts_patch, {"prices": prices})

        # News
        news_cfg = tools.get("news") or {}
        if news_cfg.get("enabled", False):
            lim = int(news_cfg.get("limit") or 2)
            news_dict: Dict[str, Any] = {}
            for t in tickers:
                try:
                    arts = news_fetcher.fetch(t, limit=lim)
                except Exception:
                    arts = []
                enriched, previews = [], []
                for a in arts or []:
                    d = dict(a) if isinstance(a, dict) else {"title": str(a)}
                    pv = _news_preview(d.get("description") or d.get("content") or d.get("title") or "")
                    ts = _format_news_timestamp(d.get("published"))
                    if ts:
                        d["published_display"] = ts
                        pv = f"[{ts}] {pv}" if pv else f"[{ts}]"
                    d["preview"] = pv
                    enriched.append(d)
                    if pv:
                        previews.append(pv)
                summary = " ".join(previews[:3]) if previews else ""
                news_dict[t] = {"query": t, "articles": enriched, **({"summary": summary} if summary else {})}
            facts_patch = _deep_merge(facts_patch, {"news": news_dict})

        # Analyst
        ana_cfg = tools.get("analyst") or {}
        if ana_cfg.get("enabled", False):
            lim = int(ana_cfg.get("limit") or 1)
            reports = {t: get_analyst_reports(t, limit=lim) for t in tickers}
            facts_patch = _deep_merge(facts_patch, {"analyst": reports})

        # Risk
        risk_cfg = tools.get("risk") or {}
        if risk_cfg.get("enabled", False):
            try:
                risk_level_pref = str((profile or {}).get("risk_level", "medium") or "medium").lower()
                risk_info = get_risk_assessment(tickers, risk_level=risk_level_pref)
            except Exception as exc:
                risk_info = {"error": str(exc), "tickers": tickers}
            facts_patch = _deep_merge(facts_patch, {"risk": risk_info})

    new_messages: List[Any] = []
    if facts_patch:
        base = dict(facts or {})
        if not base.get("tickers"):
            base["tickers"] = tickers
        merged = _deep_merge(base, facts_patch)
        new_messages.append(HumanMessage(content="FACTS_JSON: " + __import__("json").dumps(merged, ensure_ascii=False)))

    # Normalize decision & emit ROUTE_PLAN_JSON
    decision = plan.get("decision") or {}
    nxt = str(decision.get("next_node") or "").strip()
    if nxt not in POSSIBLE_NODES:
        nxt = _default_next_from_state(state)
    decision["next_node"] = nxt

    route_payload = {
        "decision": {
            "next_node": decision.get("next_node"),
            "return_node": decision.get("return_node"),
            "reason": decision.get("reason") or "n/a"
        },
        "tools": {
            "execute": bool(tools.get("execute", False)),
            "prices": tools.get("prices") or {"enabled": True, "tickers": None},
            "news":   tools.get("news")   or {"enabled": False, "tickers": None, "limit": 2},
            "analyst":tools.get("analyst")or {"enabled": False, "tickers": None, "limit": 1},
            "risk":   tools.get("risk")   or {"enabled": False, "tickers": None},
        }
    }
    new_messages.append(HumanMessage(content="ROUTE_PLAN_JSON: " + __import__("json").dumps(route_payload, ensure_ascii=False)))
    return {"messages": new_messages}


def route_from_general_plan(state: MessagesState) -> str:
    """
    Read ROUTE_PLAN_JSON and decide next node.
    Also supports 'return_node' semantics: if a previous plan requested to return
    to a node and its prerequisite TAG is ready now, prefer returning.
    """
    _ensure_helpers()
    plan = _get_json(state, "ROUTE_PLAN_JSON")
    nxt = None

    # Simple 'return' semantics
    ret = (plan.get("decision") or {}).get("return_node") if isinstance(plan, dict) else None
    if ret in POSSIBLE_NODES:
        ready_need_tag = {
            "synthesize": "PICKS_JSON",
            "allocate":   "SELECTION_JSON",
            "select":     "ANALYSIS_JSON",
            "analyze":    "INSIGHTS_JSON",
            "insights":   "FACTS_JSON",
        }
        need = ready_need_tag.get(ret)
        if not need or _get_json(state, f"{need}"):
            nxt = ret

    if not nxt:
        nxt = (plan.get("decision") or {}).get("next_node") if isinstance(plan, dict) else None
    if nxt not in POSSIBLE_NODES:
        nxt = _default_next_from_state(state)
    return nxt


# -----------------------
# Tiny local utilities
# -----------------------
def _extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object, preferring ```json fenced blocks."""
    if not text:
        return None
    import re, json
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


def _collect_tickers_from_tools(tools: Dict[str, Any]) -> List[str]:
    """Gather tickers explicitly provided in the tool plan."""
    tickers = []
    for k in ("prices", "news", "analyst", "risk"):
        cfg = tools.get(k) or {}
        arr = cfg.get("tickers") or []
        if isinstance(arr, list):
            tickers.extend([t for t in arr if isinstance(t, str) and _is_valid_ticker(t)])
    return _unique_preserve(tickers)
