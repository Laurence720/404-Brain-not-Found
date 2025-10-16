from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import threading
from typing import AsyncGenerator, Dict, List, Any, Optional
from pathlib import Path


def _lazy_import_langgraph():
    from LangGraph import run_agent_sync, run_agent_stream
    return run_agent_sync, run_agent_stream


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if STATIC_DIR.exists() and index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"detail": "index.html not found"}, status_code=404)


def _extract_final_from_events(events: List[Dict[str, Any]]) -> Optional[str]:
    """Return the most recent AI-like message from the event stream."""
    for event in reversed(events):
        role = str(event.get("role", "")).lower()
        if role in {"ai", "aimessage", "assistant"}:
            content = event.get("content")
            if content:
                return str(content)
    return None


async def _run_graph_sync(user_input: str, history: List[Dict[str, str]]) -> str:
    """Invoke LangGraph synchronously via a worker thread."""
    def _worker() -> Dict[str, Any]:
        run_agent_sync, _ = _lazy_import_langgraph()
        return run_agent_sync(user_input, history)

    result = await asyncio.to_thread(_worker)
    error = result.get("error")
    if error:
        return f"[agent_error] {error}"

    final = result.get("final")
    if final:
        return final

    fallback = _extract_final_from_events(result.get("events", []))
    if fallback:
        return fallback
    return "The agent did not produce a response. Please refine your request and try again."


async def _run_graph_stream(user_input: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Bridge LangGraph's streaming generator into an async generator for FastAPI."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def _enqueue(chunk: Optional[str]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()

    def _worker() -> None:
        try:
            _, run_agent_stream = _lazy_import_langgraph()
            for chunk in run_agent_stream(user_input, history):
                if not chunk:
                    continue
                _enqueue(chunk)
        except Exception as exc:
            _enqueue(f"[agent_error] {exc}")
        finally:
            _enqueue(None)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    user_input: str = body.get("input", "")
    history: List[Dict[str, str]] = body.get("history", [])
    want_stream: bool = bool(body.get("stream", False))

    if not user_input:
        return JSONResponse({"output": "Please provide input"}, status_code=400)

    if want_stream:
        async def streamer():
            async for chunk in _run_graph_stream(user_input, history):
                yield f"data: {chunk}\n\n"
        return StreamingResponse(streamer(), media_type="text/event-stream")

    output = await _run_graph_sync(user_input, history)
    return JSONResponse({"output": output})
