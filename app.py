import os
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from LangGraph import run_agent_stream, run_agent_sync

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


def _extract_final_from_events(events: List[Dict[str, Any]]) -> Optional[str]:
    """Return the last assistant style chunk from a LangGraph event list."""
    for event in reversed(events or []):
        role = str(event.get("role", "")).lower()
        if role in {"ai", "aimessage", "assistant"}:
            content = event.get("content")
            if content:
                return str(content)
    return None


def _invoke_agent_sync(user_input: str, history: List[Dict[str, Any]]) -> Dict[str, str]:
    """Guarded synchronous LangGraph invocation."""
    try:
        result = run_agent_sync(user_input, history)
    except Exception as exc:
        return {"output": f"[agent_error] {exc}"}

    if not isinstance(result, dict):
        return {"output": str(result)}

    error = result.get("error")
    if error:
        return {"output": f"[agent_error] {error}"}

    final = result.get("final")
    if final:
        return {"output": str(final)}

    fallback = _extract_final_from_events(result.get("events", []))
    if fallback:
        return {"output": fallback}

    return {"output": "The agent did not produce a response. Please refine your request and try again."}


@app.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = str(payload.get("input", "")).strip()
    history = payload.get("history") or []
    want_stream = bool(payload.get("stream", False))

    if not user_input:
        return jsonify({"output": "Please provide input"}), 400

    if want_stream:
        def generate():
            try:
                for chunk in run_agent_stream(user_input, history):
                    if not chunk:
                        continue
                    yield f"data: {chunk}\n\n"
            except Exception as exc:
                yield f"data: [agent_error] {exc}\n\n"

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        return response

    return jsonify(_invoke_agent_sync(user_input, history))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
