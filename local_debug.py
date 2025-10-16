"""
Utilities for local debugging without touching the deployment Flask app.

Run `python local_debug.py` (or `uvicorn local_debug:asgi_app`) to launch the
same application using an ASGI wrapper that works with uvicorn reload/debug
tooling.
"""

import os

from asgiref.wsgi import WsgiToAsgi

from app import app as flask_app

# Wrap the existing Flask WSGI app so uvicorn can serve it during local testing.
asgi_app = WsgiToAsgi(flask_app)


def _main() -> None:
    """Start an auto-reload uvicorn server for developer use."""
    import uvicorn

    port = int(os.getenv("DEBUG_PORT", os.getenv("PORT", "8000")))
    uvicorn.run(
        "local_debug:asgi_app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    _main()
