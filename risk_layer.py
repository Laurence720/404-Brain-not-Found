
"""
risk_layer.py — Reusable risk assessment utilities (3-tier risk policy, metrics, screening, blending)

Quick start
-----------
from risk_layer import get_risk_assessment

result = get_risk_assessment(
    tickers=["AAPL", "MSFT", "NVDA"],
    risk_level="low",           # "low" | "medium" | "high"
    period="6mo",
    interval="1d",
    benchmark="SPY",
)

print(result["metrics"]["AAPL"])   # {'ann_vol': ..., 'beta': ..., 'mdd_6m': ..., 'ret_6m': ...}
print(result["kept"])              # tickers kept by the risk screen
print(result["dropped"])           # {ticker: reason}
print(result["policy"])            # policy used

Optionally blend your own LLM scores:
-------------------------------------
blended = blend_llm_with_risk(
    llm_scores={"AAPL": 78, "MSFT": 74, "NVDA": 82},
    risk_metrics=result["metrics"],
    risk_level="low",
)

Notes
-----
- yfinance is optional; if unavailable or no network, metrics fall back to None gracefully.
- This module does not make any trading recommendations; it  risk metrics and a simple screen.
"""

# get_risk_assessment 会自动：
# 	1.	用 yfinance（若可用）拉近 6 个月日线收盘价；
# 	2.	计算每只标的的年化波动率 ann_vol、相对基准 beta、近 6 个月最大回撤 mdd_6m、区间收益 ret_6m；


from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import math

# Optional deps
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
    _YF_AVAILABLE = True
except Exception:  # pragma: no cover
    yf = None  # type: ignore
    _YF_AVAILABLE = False


# -----------------------------
# 1) Three-tier Risk Policies
# -----------------------------
RISK_POLICIES: Dict[str, Dict[str, Any]] = {
    "low": {
        "instrument_set": ["ETF","STOCK"],
        "core_etfs": ["VTI","VOO","SCHX","XLV","XLP","AGG","IEF"],
        "max_single_weight": 0.10,
        "sector_cap": 0.25,
        "min_market_cap_usd": 20e9,     # optional if you provide market_caps
        "min_adv_usd": 80e6,            # optional if you provide adv_usd
        "beta_max": 1.0,
        "ann_vol_max": 0.22,
        "max_drawdown_6m": 0.25,
        "allow_leverage": False,
        "hedge_etfs": ["SHY","IEF"],
    },
    "medium": {
        "instrument_set": ["STOCK","ETF"],
        "core_etfs": ["VTI","QQQ","IWM"],
        "max_single_weight": 0.15,
        "sector_cap": 0.35,
        "min_market_cap_usd": 5e9,
        "min_adv_usd": 30e6,
        "beta_max": 1.3,
        "ann_vol_max": 0.35,
        "max_drawdown_6m": 0.35,
        "allow_leverage": False,
        "hedge_etfs": [],
    },
    "high": {
        "instrument_set": ["STOCK","ETF"],
        "core_etfs": ["QQQ","IWM"],
        "max_single_weight": 0.20,
        "sector_cap": 0.50,
        "min_market_cap_usd": 1e9,
        "min_adv_usd": 10e6,
        "beta_min": 1.0,
        "ann_vol_min": 0.22,
        "allow_leverage": False,
        "hedge_etfs": [],
    },
}


# --------------------------------------------
# 2) Price download & metric computations
# --------------------------------------------
def _safe_download_prices(tickers: List[str], period: str = "6mo", interval: str = "1d", auto_adjust: bool = True) -> Dict[str, "pd.Series"]:
    """
    Download OHLCV via yfinance, return dict of Adjusted Close series per symbol.
    If yfinance/pandas/numpy are unavailable or an error occurs, returns {}.
    """
    if not (_YF_AVAILABLE and pd is not None):
        return {}

    try:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            progress=False,
            threads=False,
            auto_adjust=auto_adjust,
        )
    except Exception:
        return {}

    if df is None or len(df) == 0:
        return {}

    # Normalize to a dict of Series (adj close preferred)
    out: Dict[str, "pd.Series"] = {}

    def _to_series(candidate) -> Optional["pd.Series"]:
        if candidate is None or len(candidate) == 0:
            return None
        if isinstance(candidate, pd.Series):
            return candidate.dropna()
        if isinstance(candidate, pd.DataFrame):
            # Multi-asset wide frame -> we will split by columns later
            return None
        return None

    # yfinance returns:
    # - Single ticker: DataFrame with columns; or Series for fields
    # - Multi tickers: Column level MultiIndex
    if isinstance(df, pd.DataFrame):
        # Prefer Adj Close, fallback Close
        if "Adj Close" in df:
            sub = df["Adj Close"]
        elif "Close" in df:
            sub = df["Close"]
        else:
            sub = None

        if sub is None:
            return {}

        if isinstance(sub, pd.Series):
            # Single ticker case
            # Try infer ticker name
            t = tickers[0]
            out[t] = sub.dropna()
        elif isinstance(sub, pd.DataFrame) and hasattr(sub, "columns"):
            for c in sub.columns:
                # yfinance may provide non-string columns in some edge cases
                if isinstance(c, (str,)):
                    ser = sub[c].dropna()
                    if len(ser) >= 5:
                        out[c] = ser
    elif isinstance(df, pd.Series):
        t = tickers[0]
        out[t] = df.dropna()

    return out


def _series_to_returns(px: "pd.Series") -> "pd.Series":
    """Daily simple returns from price series."""
    return px.pct_change().dropna()


def _max_drawdown(px: "pd.Series") -> Optional[float]:
    """Max drawdown as positive fraction, e.g., 0.25 for -25% from peak."""
    if px is None or len(px) == 0 or pd is None:
        return None
    cummax = px.cummax()
    dd = (px / cummax) - 1.0
    # return absolute magnitude
    if len(dd) == 0:
        return None
    return float(abs(dd.min()))


def _beta_vs_benchmark(ret_asset: "pd.Series", ret_bench: "pd.Series") -> Optional[float]:
    """Beta = Cov(asset, bench) / Var(bench), using aligned daily returns."""
    if np is None or pd is None:
        return None
    if ret_asset is None or ret_bench is None:
        return None
    a, b = ret_asset.align(ret_bench, join="inner")
    if a is None or b is None or len(a) < 5 or len(b) < 5:
        return None
    av = a.values
    bv = b.values
    # sample covariance / variance (ddof=1)
    try:
        cov = float(np.cov(av, bv, ddof=1)[0, 1])
        var_b = float(np.var(bv, ddof=1))
        if var_b <= 0:
            return None
        return cov / var_b
    except Exception:
        return None


def compute_metrics_from_prices(
    prices: Dict[str, "pd.Series"],
    benchmark: str = "SPY",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute {ticker: {'ann_vol', 'beta', 'mdd_6m', 'ret_6m'}} given price series (dict of Series).
    Missing fields will be None if not computable.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if pd is None or np is None:
        # Without pandas/numpy: cannot compute; return skeleton
        for t in prices.keys():
            out[t] = {"ann_vol": None, "beta": None, "mdd_6m": None, "ret_6m": None}
        return out

    bench_px = prices.get(benchmark)
    bench_ret = _series_to_returns(bench_px) if bench_px is not None else None

    for t, px in prices.items():
        r = _series_to_returns(px) if px is not None else None
        # Annualized volatility (daily -> annual using sqrt(252))
        ann_vol = float(np.sqrt(252.0) * r.std()) if r is not None and len(r) > 1 else None
        # Max drawdown on price series
        mdd = _max_drawdown(px)
        # Beta vs benchmark
        beta = _beta_vs_benchmark(r, bench_ret) if r is not None and bench_ret is not None else None
        # 6m simple return (cumulative)
        ret_6m = None
        if px is not None and len(px) >= 2:
            ret_6m = float(px.iloc[-1] / px.iloc[0] - 1.0)
        out[t] = {"ann_vol": ann_vol, "beta": beta, "mdd_6m": mdd, "ret_6m": ret_6m}
    return out


def compute_risk_metrics(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = "SPY",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    High-level helper: download prices for tickers + benchmark (if missing), then compute metrics.
    Returns {ticker: {'ann_vol','beta','mdd_6m','ret_6m'}}. Unavailable metrics are None.
    """
    symbols = list(dict.fromkeys(tickers + [benchmark]))  # preserve order, unique
    px = _safe_download_prices(symbols, period=period, interval=interval, auto_adjust=True)
    # Ensure benchmark exists in dict for beta; if not, beta will be None
    return compute_metrics_from_prices(px, benchmark=benchmark)


# ------------------------------------------------------
# 3) Policy application (screen keep/drop + simple backfills)
# ------------------------------------------------------
def apply_risk_policy(
    tickers: List[str],
    risk_level: str,
    risk_metrics: Dict[str, Dict[str, Optional[float]]],
    market_caps: Optional[Dict[str, float]] = None,
    adv_usd: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Screen tickers with the selected policy. Returns:
    {
      "kept": [...],
      "dropped": {ticker: reason},
      "policy": <policy-dict>
    }
    - If metrics are missing, this function will be lenient (keeps the ticker).
    - If low-risk keeps <3, we backfill with core ETFs (if provided).
    """
    policy = RISK_POLICIES.get((risk_level or "medium").lower(), RISK_POLICIES["medium"])
    kept: List[str] = []
    dropped: Dict[str, str] = {}

    for t in tickers:
        m = risk_metrics.get(t, {}) if risk_metrics else {}
        ann_vol = m.get("ann_vol")
        beta = m.get("beta")
        mdd = m.get("mdd_6m")

        # Optional size/liquidity constraints
        if market_caps and policy.get("min_market_cap_usd") and market_caps.get(t, float("inf")) < policy["min_market_cap_usd"]:
            dropped[t] = "market_cap_too_small"
            continue
        if adv_usd and policy.get("min_adv_usd") and adv_usd.get(t, float("inf")) < policy["min_adv_usd"]:
            dropped[t] = "liquidity_too_low"
            continue

        # Risk thresholds
        # Low/Medium have max constraints; High may have min constraints
        if policy.get("beta_max") is not None and beta is not None and beta > policy["beta_max"]:
            dropped[t] = "beta_too_high"; continue
        if policy.get("beta_min") is not None and beta is not None and beta < policy["beta_min"]:
            dropped[t] = "beta_too_low"; continue

        if policy.get("ann_vol_max") is not None and ann_vol is not None and ann_vol > policy["ann_vol_max"]:
            dropped[t] = "vol_too_high"; continue
        if policy.get("ann_vol_min") is not None and ann_vol is not None and ann_vol < policy["ann_vol_min"]:
            dropped[t] = "vol_too_low"; continue

        if policy.get("max_drawdown_6m") is not None and mdd is not None and mdd > policy["max_drawdown_6m"]:
            dropped[t] = "drawdown_too_large"; continue

        kept.append(t)

    # Backfill for conservative portfolios
    if (risk_level or "").lower() == "low" and len(kept) < 3:
        for etf in policy.get("core_etfs", []):
            if etf not in kept:
                kept.append(etf)
        # keep length reasonable
        seen = set()
        kept = [x for x in kept if not (x in seen or seen.add(x))][:8]

    return {"kept": kept, "dropped": dropped, "policy": policy}


# ------------------------------------------------------
# 4) Optional: Blend LLM scores with risk-fit
# ------------------------------------------------------
def _risk_fit_score(
    m: Optional[Dict[str, Optional[float]]],
    risk_level: str,
) -> int:
    """
    Convert metrics into a 0-100 "risk fit" score for the specified risk level.
    The exact mapping is heuristic and can be tuned.
    """
    if m is None:
        return 50
    vol = m.get("ann_vol")
    beta = m.get("beta")
    mdd = m.get("mdd_6m")

    # If metrics are missing, return a neutral score
    if vol is None and beta is None and mdd is None:
        return 50

    # Heuristic scoring
    if (risk_level or "").lower() == "low":
        s = 100.0
        if vol is not None: s -= max(0.0, (vol - 0.22)) * 300.0
        if beta is not None: s -= max(0.0, (beta - 1.0)) * 80.0
        if mdd is not None: s -= max(0.0, (mdd - 0.25)) * 200.0
        return int(max(0, min(100, s)))
    if (risk_level or "").lower() == "high":
        s = 40.0
        if vol is not None: s += max(0.0, (vol - 0.22)) * 200.0
        if beta is not None: s += max(0.0, (beta - 1.0)) * 60.0
        return int(max(0, min(100, s)))

    # medium
    s = 70.0
    if vol is not None: s -= max(0.0, abs(vol - 0.28)) * 200.0
    if beta is not None: s -= max(0.0, abs((beta or 1.1) - 1.1)) * 80.0
    return int(max(0, min(100, s)))


def blend_llm_with_risk(
    llm_scores: Dict[str, int],
    risk_metrics: Dict[str, Dict[str, Optional[float]]],
    risk_level: str,
) -> Dict[str, int]:
    """
    Blend your LLM analysis scores (0-100) with risk-fit scores into a single score per ticker.
    Weights depend on risk_level:
      low    -> (w_risk, w_llm) = (0.7, 0.3)
      medium -> (0.5, 0.5)
      high   -> (0.3, 0.7)
    """
    w_risk, w_llm = {
        "low": (0.7, 0.3),
        "medium": (0.5, 0.5),
        "high": (0.3, 0.7),
    }.get((risk_level or "medium").lower(), (0.5, 0.5))

    out: Dict[str, int] = {}
    for t, s in llm_scores.items():
        try:
            s_llm = int(float(s))
        except Exception:
            s_llm = 50
        s_fit = _risk_fit_score(risk_metrics.get(t), risk_level=risk_level)
        blended = int(round(w_llm * s_llm + w_risk * s_fit))
        out[t] = max(0, min(100, blended))
    return out


# ------------------------------------------------------
# 5) One-stop API for main program
# ------------------------------------------------------
def get_risk_assessment(
    tickers: List[str],
    risk_level: str = "medium",
    period: str = "6mo",
    interval: str = "1d",
    benchmark: str = "SPY",
    market_caps: Optional[Dict[str, float]] = None,
    adv_usd: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    One-call API:
    - Computes risk metrics for tickers
    - Applies selected risk policy to keep/drop
    - Returns a dict with metrics/policy/kept/dropped

    Returns:
    {
      "metrics": {ticker: {"ann_vol": float|None, "beta": float|None, "mdd_6m": float|None, "ret_6m": float|None}},
      "kept": [tickers],
      "dropped": {ticker: reason},
      "policy": <policy dict>,
      "params": {"risk_level": ..., "period": ..., "interval": ..., "benchmark": ...}
    }
    """
    # Compute metrics (gracefully handles missing yfinance)
    metrics = compute_risk_metrics(tickers=tickers, period=period, interval=interval, benchmark=benchmark)

    # Apply policy screen
    screened = apply_risk_policy(
        tickers=tickers,
        risk_level=risk_level,
        risk_metrics=metrics,
        market_caps=market_caps,
        adv_usd=adv_usd,
    )

    return {
        "metrics": metrics,
        "kept": screened["kept"],
        "dropped": screened["dropped"],
        "policy": screened["policy"],
        "params": {
            "risk_level": risk_level,
            "period": period,
            "interval": interval,
            "benchmark": benchmark,
        },
    }


# ------------------------------------------------------
# Demo (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    # Simple smoke test (will attempt yfinance download; if offline, you'll just see None metrics)
    syms = ["AAPL", "MSFT", "NVDA"]
    res = get_risk_assessment(syms, risk_level="low", period="6mo", interval="1d", benchmark="SPY")
    print("Policy:", res["policy"])
    print("Kept:", res["kept"])
    print("Dropped:", res["dropped"])
    for t, m in res["metrics"].items():
        print(t, m)
