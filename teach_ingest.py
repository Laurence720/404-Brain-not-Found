import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

TEACH_DB = Path("teach_db")
TEACH_DB.mkdir(exist_ok=True)

INVESTO = "https://www.investopedia.com/financial-term-dictionary-4769738"
CFA     = "https://www.cfainstitute.org/en/research/foundation"
MSTAR   = "https://www.morningstar.com/lp/investment-glossary"

def _bs4_extractor(html: str) -> str:
    """简洁提取正文，去掉导航/脚注。"""
    soup = BeautifulSoup(html, "lxml")
    # 删除常见无关块
    for sel in ["nav", "header", "footer", "aside", "script", "style"]:
        for tag in soup.find_all(sel):
            tag.decompose()
    # 取主要文本
    text = soup.get_text(separator="\n", strip=True)
    # 轻量清洗
    lines = [ln for ln in (text or "").splitlines() if ln and len(ln) > 2]
    return "\n".join(lines)

def load_quick() -> List:
    """仅抓取3个入口页（最快）"""
    loader = WebBaseLoader([INVESTO, CFA, MSTAR])
    return loader.load()

def load_crawl() -> List:
    """受限递归抓取（更全）。每站点 max_depth=2、每站点最多 ~60 页（按需调小）。"""
    docs = []
    # Investopedia：只抓 glossary/terms 相关路径
    inv = RecursiveUrlLoader(
        url=INVESTO, max_depth=2, extractor=_bs4_extractor,
        use_async=True, timeout=10, check_response_status=True
    )
    inv_docs = inv.load()
    inv_docs = [d for d in inv_docs if "investopedia.com" in (d.metadata.get("source",""))]
    docs.extend(inv_docs[:60])

    # CFA：foundation 研究页面
    cfa = RecursiveUrlLoader(
        url=CFA, max_depth=2, extractor=_bs4_extractor,
        use_async=True, timeout=10, check_response_status=True
    )
    cfa_docs = cfa.load()
    cfa_docs = [d for d in cfa_docs if "cfainstitute.org" in (d.metadata.get("source",""))]
    docs.extend(cfa_docs[:60])

    # Morningstar：glossary 官方页（有时较重，抓少量）
    ms = RecursiveUrlLoader(
        url=MSTAR, max_depth=1, extractor=_bs4_extractor,
        use_async=True, timeout=10, check_response_status=True
    )
    ms_docs = ms.load()
    ms_docs = [d for d in ms_docs if "morningstar.com" in (d.metadata.get("source",""))]
    docs.extend(ms_docs[:30])
    return docs

def main(mode: str = "quick"):
    print(f"[teach-ingest] mode={mode}")
    if mode == "crawl":
        docs = load_crawl()
    else:
        docs = load_quick()
    print(f"[teach-ingest] loaded: {len(docs)} docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=160)
    chunks = splitter.split_documents(docs)
    print(f"[teach-ingest] chunks: {len(chunks)}")

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(str(TEACH_DB))

    # Write provenance metadata for downstream inspection.
    sources: Set[str] = set()
    for doc in docs:
        src = (doc.metadata or {}).get("source")
        if isinstance(src, str) and src:
            sources.add(src.strip())
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "sources": sorted(sources),
    }
    meta_path = TEACH_DB / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ teaching vector DB saved to {TEACH_DB}/")

if __name__ == "__main__":
    import sys
    mode = (sys.argv[1] if len(sys.argv) > 1 else "quick").lower()
    main(mode)
    
