
"""
chat_ephemeral_rag.py — Ephemeral RAG for Chat History
把“用户-LLM 对话历史”做成**临时向量库**（teach_db 目录），LLM 仅检索相关片段而非查看全量历史。

设计目标
--------
- 输入：一段对话历史（list[dict] 或 LangChain messages）
- 处理：按“若干轮窗口”切分 → 生成带角色/时间标注的可检索文本块（可选摘要）
- 输出：保存为本地临时 teach_db（FAISS）；在你的工具里作为 teach_db_dir 传入即可
- 检索：提供便捷检索函数，便于在本地先做 smoke test
- 依赖：FAISS + sentence-transformers 可选；缺失则降级写 JSONL（仍可审计调试）

快速上手
--------
from chat_ephemeral_rag import EphemeralChatDB, build_teach_db_from_chat, preview_retrieve

history = [
  {"role":"user","content":"我想看下AAPL近6个月波动和Beta"},
  {"role":"assistant","content":"可以，我们将对AAPL相对SPY计算β与年化波动..."}
]

with EphemeralChatDB(ttl_seconds=3600) as db:
    teach_dir = build_teach_db_from_chat(history, db=db, window_turns=8, stride=4)
    # 现在把 teach_dir 传给你的 RAG 工具：
    # res = generate_worked_example(metric="beta", ticker="AAPL", teach_db_dir=teach_dir)

    # 本地先检索看看：
    hits = preview_retrieve(teach_dir, "AAPL beta 近6个月", k=4)
    for h in hits: print(h["score"], h["preview"])

"""

from __future__ import annotations
import os, json, time, uuid, shutil, tempfile
from typing import Any, Dict, List, Optional, Tuple

# ---------- 可选依赖（优雅降级） ----------
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception:
    FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb  # type: ignore
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb  # type: ignore
    except Exception:
        _HFEmb = None
HuggingFaceEmbeddings = _HFEmb

try:
    from langchain_core.documents import Document  # type: ignore
    _LC_DOC = True
except Exception:
    Document = None  # type: ignore
    _LC_DOC = False

# ---------- 工具函数 ----------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _now_ts() -> int:
    return int(time.time())

def _norm_role(r: Optional[str]) -> str:
    r = (r or "").lower()
    if r in {"user","human"}: return "user"
    if r in {"assistant","ai","model"}: return "assistant"
    if r == "system": return "system"
    return r or "unknown"

def _fmt_ts(ts: Optional[float]) -> str:
    try:
        import datetime as _dt
        if ts is None:
            return ""
        if ts > 1e11:  # ms
            ts = ts/1000.0
        return _dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""

def _stringify_message(m: Any) -> Dict[str, Any]:
    """Normalize a message-like object (dict or LangChain message) to {role, content, ts}."""
    if isinstance(m, dict):
        role = _norm_role(m.get("role"))
        content = str(m.get("content") or "")
        ts = m.get("ts") or m.get("timestamp")
        ts = float(ts) if ts is not None else None
        return {"role": role, "content": content, "ts": ts}
    # LangChain message
    role = getattr(m, "type", getattr(m, "role", "")).lower()
    role = _norm_role(role)
    content = str(getattr(m, "content", "") or "")
    # 尝试从 .additional_kwargs/metadata 取时间
    ts = None
    for k in ("additional_kwargs","metadata","__dict__"):
        try:
            meta = getattr(m, k, None) or {}
            if isinstance(meta, dict):
                raw = meta.get("ts") or meta.get("timestamp")
                if raw is not None:
                    ts = float(raw)
                    break
        except Exception:
            pass
    return {"role": role, "content": content, "ts": ts}

def _chunk_chat(messages: List[Any], window_turns: int = 8, stride: int = 4, max_chars: int = 2200) -> List[Dict[str, Any]]:
    """
    将对话按“轮数窗口”切块。每块包含：窗口索引、起止时间/下标、拼接文本。
    - window_turns: 每块包含的消息条数（不是 Q/A 对）
    - stride: 滑动步长，默认为一半重叠
    - max_chars: 超过则截断（避免段太长影响检索质量）
    """
    norm = [_stringify_message(m) for m in messages if m]
    n = len(norm)
    chunks: List[Dict[str, Any]] = []
    if n == 0:
        return chunks
    i = 0; idx = 0
    while i < n:
        j = min(n, i + window_turns)
        subset = norm[i:j]
        start_ts = next((m["ts"] for m in subset if m.get("ts") is not None), None)
        end_ts   = next((m["ts"] for m in reversed(subset) if m.get("ts") is not None), None)
        lines: List[str] = []
        for m in subset:
            stamp = _fmt_ts(m.get("ts"))
            head = f"[{stamp}] " if stamp else ""
            lines.append(f"{head}{m['role']}: {m['content']}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        chunk = {
            "index": idx,
            "i": i, "j": j,
            "start_ts": start_ts, "end_ts": end_ts,
            "text": text,
            "turns": len(subset),
        }
        chunks.append(chunk)
        idx += 1
        if j == n:
            break
        i += max(1, stride)
    return chunks

def _make_docs(chunks: List[Dict[str, Any]], topic_hint: Optional[str] = None) -> List[Any]:
    docs = []
    for ch in chunks:
        meta = {
            "chunk_index": ch["index"],
            "i": ch["i"], "j": ch["j"],
            "start_ts": ch.get("start_ts"),
            "end_ts": ch.get("end_ts"),
            "turns": ch.get("turns"),
            "topic": topic_hint or "",
            "source": "chat_history",
        }
        if _LC_DOC and Document is not None:
            docs.append(Document(page_content=ch["text"], metadata=meta))
        else:
            docs.append({"page_content": ch["text"], "metadata": meta})
    return docs

# ---------- 临时目录管理 ----------
class EphemeralChatDB:
    """
    Context-managed temporary directory for a local FAISS index of chat history.
    使用 with 语句可在会话末尾按需清理目录；也可设置 ttl_seconds 自动回收。
    """
    def __init__(self, base_dir: Optional[str] = None, ttl_seconds: Optional[int] = None):
        self._parent = base_dir or tempfile.gettempdir()
        self._dir = os.path.join(self._parent, f"teach_db_chat_{uuid.uuid4().hex[:8]}")
        self._created = time.time()
        self._ttl = ttl_seconds
        _ensure_dir(self._dir)

    @property
    def path(self) -> str:
        return self._dir

    def expired(self) -> bool:
        return self._ttl is not None and (time.time() - self._created) > self._ttl

    def cleanup(self):
        try:
            if os.path.isdir(self._dir):
                shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.expired() or (self._ttl is None):
            self.cleanup()

# ---------- 构建 teach_db ----------
def build_teach_db_from_chat(
    messages: List[Any],
    *,
    db: Optional[EphemeralChatDB] = None,
    window_turns: int = 8,
    stride: int = 4,
    max_chars: int = 2200,
    topic_hint: Optional[str] = None,
) -> str:
    """
    把对话切块并写入本地 teach_db（FAISS）；返回 teach_db 目录路径。
    - 若缺少 FAISS/Embeddings，则降级写 docs.jsonl（仍可审核）。
    """
    owns = False
    if db is None:
        db = EphemeralChatDB(ttl_seconds=None)
        owns = True

    chunks = _chunk_chat(messages, window_turns=window_turns, stride=stride, max_chars=max_chars)
    docs = _make_docs(chunks, topic_hint=topic_hint)

    if FAISS is not None and HuggingFaceEmbeddings is not None:
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if _LC_DOC and Document is not None:
            vs = FAISS.from_documents(docs, embed)
        else:
            texts = [d["page_content"] for d in docs]
            metas = [d["metadata"] for d in docs]
            vs = FAISS.from_texts(texts=texts, embedding=embed, metadatas=metas)
        vs.save_local(db.path)
        with open(os.path.join(db.path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"count": len(docs), "created": _now_ts(), "topic": topic_hint or ""}, f, ensure_ascii=False, indent=2)
    else:
        # 无向量库依赖时，写 JSONL 方便你审阅
        with open(os.path.join(db.path, "docs.jsonl"), "w", encoding="utf-8") as f:
            for d in docs:
                if _LC_DOC and Document is not None:
                    rec = {"text": d.page_content, "metadata": d.metadata}
                else:
                    rec = {"text": d["page_content"], "metadata": d["metadata"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    path = db.path
    if owns:
        # 如果函数内部创建的 db，保留目录供后续使用；交给调用方在会话结束时清理
        pass
    return path

# ---------- 本地检索（调试用） ----------
def preview_retrieve(teach_db_dir: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    从本地 teach_db 中检索 top-k 文本块（仅本地调试用）。
    若无 FAISS/Embeddings，将读取 docs.jsonl 进行“关键词包含度”粗糙排序。
    """
    if FAISS is not None and HuggingFaceEmbeddings is not None and os.path.isdir(teach_db_dir):
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = FAISS.load_local(teach_db_dir, embeddings=embed, allow_dangerous_deserialization=True)
        retriever = vs.as_retriever(search_kwargs={"k": int(k)})
        docs = retriever.get_relevant_documents(query)
        out = []
        for d in docs:
            meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            text = getattr(d, "page_content", "") if hasattr(d, "page_content") else ""
            out.append({
                "preview": (text[:240] + "...") if len(text) > 240 else text,
                "metadata": meta,
                "score": getattr(d, "score", None),
            })
        return out

    # 降级：关键词匹配
    path = os.path.join(teach_db_dir, "docs.jsonl")
    if not os.path.exists(path):
        return []
    qs = query.lower().split()
    hits = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = str(rec.get("text") or "")
            meta = rec.get("metadata") or {}
            score = sum(1 for q in qs if q in text.lower())
            hits.append({"preview": (text[:240]+"...") if len(text)>240 else text, "metadata": meta, "score": score})
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k]

# ---------- JSONL->messages 的便捷导入 ----------
def load_chat_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    读取形如 {role, content, ts?} 的 JSON Lines 文件为消息列表。
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            role = _norm_role(rec.get("role"))
            content = str(rec.get("content") or "")
            ts = rec.get("ts") or rec.get("timestamp")
            ts = float(ts) if ts is not None else None
            out.append({"role": role, "content": content, "ts": ts})
    return out

if __name__ == "__main__":
    # smoke test with a tiny chat
    hist = [
        {"role":"user","content":"我想根据风险承受力匹配股票。风险越高=波动越大，对吧？","ts": 1700000000},
        {"role":"assistant","content":"一般相关，但要综合回撤与尾部风险。","ts": 1700001000},
        {"role":"user","content":"那 beta 和年化波动、夏普怎么一起看？","ts": 1700002000},
        {"role":"assistant","content":"可以用 β 衡量相对系统性风险；年化波动刻画幅度；夏普衡量单位风险收益。","ts": 1700003000},
    ]
    with EphemeralChatDB(ttl_seconds=None) as db:
        p = build_teach_db_from_chat(hist, db=db, window_turns=3, stride=2, topic_hint="risk matching")
        print("teach_db at:", p)
        demo = preview_retrieve(p, "beta 夏普", k=2)
        print(demo)
