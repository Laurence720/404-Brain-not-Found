# news_fetcher.py
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
os.environ["FINNHUB_API_KEY"] = "d3jsq49r01qtciv0e5egd3jsq49r01qtciv0e5f0"
os.environ["NEWSAPI_KEY"] = ""
def _iso(ts: Optional[int] = None) -> str:
    if ts is None:
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _to_iso_utc(value: Optional[Any]) -> str:
    """Normalize assorted timestamp formats to ISO8601 UTC."""
    if value is None:
        return _iso()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return _iso()
        try:
            dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            try:
                # try to parse as integer seconds
                as_int = int(float(cleaned))
                return datetime.fromtimestamp(as_int, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                return cleaned
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return _iso()

class NewsFetcher:
    def __init__(self, provider_priority: Optional[List[str]] = None, session_timeout: int = 8):
        # supported providers: "finnhub", "newsapi"
        self.provider_priority = provider_priority or ["finnhub", "newsapi"]
        self.session_timeout = session_timeout
        # session will be created lazily to avoid import-time failures
        self._session = None
        self._requests_missing = False

    def _build_session(self):
        if self._session is not None:
            return self._session
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except Exception:
            # requests / urllib3 missing -> mark and continue (we will error at call time)
            self._requests_missing = True
            self._session = None
            return None

        s = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"])
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.headers.update({"User-Agent": "news-fetcher/1.0 (+https://example)"})
        self._session = s
        return s

    def fetch(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch list of article dicts. Each dict: {title, description, url, source, published}
        """
        if not query:
            return []
        session = self._build_session()
        if self._requests_missing:
            raise RuntimeError("Python package 'requests' (and/or urllib3) is required. Install with `pip install requests`.")

        # try providers in order
        for p in self.provider_priority:
            try:
                if p.lower() == "finnhub":
                    out = self._fetch_finnhub(query, limit)
                    if out:
                        return out
                elif p.lower() == "newsapi":
                    out = self._fetch_newsapi(query, limit)
                    if out:
                        return out
            except Exception:
                # provider-specific failure -> try next
                continue
        # fallback stub article
        ts = _iso()
        return [{
            "title": f"(stub) Latest update on {query}",
            "description": f"(stub) Latest update on {query}",
            "url": "https://example.com/news",
            "source": "stub",
            "published": ts,
        }][: max(1, int(limit))]

    def _fetch_finnhub(self, symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
        key = os.getenv("FINNHUB_API_KEY")
        if not key:
            return []
        # finnhub company-news requires from/to dates (YYYY-MM-DD)
        to_dt = datetime.utcnow().date()
        from_dt = to_dt - timedelta(days=7)
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": symbol, "from": from_dt.isoformat(), "to": to_dt.isoformat(), "token": key}
        import requests
        r = requests.get(url, params=params, timeout=self.session_timeout)
        r.raise_for_status()
        data = r.json() if isinstance(r.json(), list) else (r.json() or [])
        out = []
        for a in (data or [])[: max(1, int(limit))]:
            published = _to_iso_utc(a.get("datetime"))
            out.append({
                "title": a.get("headline") or a.get("title"),
                "description": a.get("summary") or a.get("headline"),
                "url": a.get("url"),
                "source": a.get("source"),
                "published": published,
            })
        return out

    def _fetch_newsapi(self, q: str, limit: int = 1) -> List[Dict[str, Any]]:
        key = os.getenv("NEWSAPI_KEY")
        if not key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": q,
            "pageSize": max(1, int(limit)),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": key,
        }
        import requests
        r = requests.get(url, params=params, timeout=self.session_timeout)
        r.raise_for_status()
        data = r.json() or {}
        arts = []
        for a in (data.get("articles") or [])[: max(1, int(limit))]:
            published_raw = a.get("publishedAt") or a.get("published") or a.get("date")
            arts.append({
                "title": a.get("title"),
                "description": a.get("description") or a.get("content"),
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "published": _to_iso_utc(published_raw),
            })
        return arts
    
if __name__ == "__main__":
    """
    Simple CLI test harness for NewsFetcher.

    Usage examples (from project root):
      python news_fetcher.py AAPL
      python news_fetcher.py bitcoin "Microsoft" -n 3
      python news_fetcher.py AAPL TSLA -n 2 -p finnhub,newsapi

    Notes:
      - Set FINNHUB_API_KEY and/or NEWSAPI_KEY as env vars to get real responses.
      - If run from inside a package and you still see ModuleNotFoundError when importing,
        run it from the repository root (see README below).
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Test harness for NewsFetcher. Provide one or more queries/tickers (e.g. AAPL TSLA) to fetch."
    )
    parser.add_argument("queries", nargs="+", help="One or more queries or tickers to fetch news for")
    parser.add_argument("--limit", "-n", type=int, default=1, help="Number of articles per query (default: 1)")
    parser.add_argument(
        "--providers", "-p",
        type=str,
        default=None,
        help="Comma-separated provider list (e.g. finnhub,newsapi). If omitted, uses class defaults."
    )
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",")] if args.providers else None
    nf = NewsFetcher(provider_priority=providers)

    results = {}
    for q in args.queries:
        try:
            articles = nf.fetch(q, limit=args.limit)
        except Exception as e:
            # keep the error for visibility in the output
            articles = [{"error": str(e)}]
        results[q] = articles

    # pretty-print JSON (UTF-8 friendly)
    print(json.dumps(results, indent=2, ensure_ascii=False))
