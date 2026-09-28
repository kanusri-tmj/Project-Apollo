"""WSGI entry point for production servers (Render / gunicorn / uWSGI).

Render runs the start command declared in ``render.yaml``::

    gunicorn --bind 0.0.0.0:$PORT wsgi:app

Keeping the entry point at the project root means ``src`` and ``app`` are
importable without any packaging step, exactly as they are when running
``python app/flask_app.py`` locally, and the server never depends on the process
working directory.

Local smoke test (no WSGI server needed)::

    PORT=5000 python wsgi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.flask_app import app  # noqa: E402

__all__ = ["app"]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
