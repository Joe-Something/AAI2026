"""Answer "can I run this?" and "is it running?" in one go.

    python scripts/check_setup.py      (or double-click check_setup.bat)

Checks the Python environment and packages, which model provider will be used, runs one real
request end to end, and reports whether the web app is currently running on port 7860.
"""

from __future__ import annotations

import importlib
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD, INFO = "[OK]", "[X] ", "[i] "


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    problems = 0
    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    print(f"{OK if in_venv else BAD} Python {sys.version.split()[0]} at {sys.executable}")
    if not in_venv:
        problems += 1
        print("     This is not the project's .venv Python. Use run_app.bat / check_setup.bat, or .venv\\Scripts\\python.exe")

    for pkg in ("langgraph", "google.genai", "chromadb", "gradio", "mcp", "pydantic", "numpy"):
        try:
            importlib.import_module(pkg)
            print(f"{OK} package {pkg}")
        except ImportError:
            problems += 1
            print(f"{BAD} package {pkg} missing: run  .venv\\Scripts\\pip install -r requirements.txt")
    if problems:
        print("\nFix the items marked [X] above, then run this check again.")
        return 1

    from helpdesk.config import settings

    cfg = settings()
    print(f"{OK} model provider: {cfg.mode_label}")
    if cfg.provider == "offline":
        print(f"{INFO}No AI model found, so answers use keyword rules. For the AI version, start Ollama (local, no key)")
        print("     or set GEMINI_API_KEY.")
    elif cfg.provider == "local":
        print(f"{OK} Ollama is running at {cfg.ollama_url} with {cfg.local_model} and {cfg.local_embedding_model}")

    print(f"{INFO}Running one real request end to end (the first one loads the model, up to a minute)...")
    from helpdesk.graph import handle_request
    from helpdesk.store import reset_runtime

    reset_runtime()
    t0 = time.perf_counter()
    try:
        state = handle_request("How do I reset my password?", "jdoe")
        print(f"{OK} answered in {time.perf_counter() - t0:.1f} s via {' -> '.join(state['path'])}")
        print("     " + state["answer"].replace("\n", " ")[:160] + "...")
    except Exception as e:
        print(f"{BAD} the request failed: {e!r}")
        return 1
    finally:
        reset_runtime()

    print()
    print("You CAN run the program: double-click run_app.bat")
    if port_open(7860):
        print("The web app IS RUNNING right now: open http://127.0.0.1:7860")
    else:
        print("The web app is NOT running right now (nothing on port 7860).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
