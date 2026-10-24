"""Command-line demo: python -m helpdesk.cli --user asmith "I'm locked out again"

Without a message it runs a short scripted demo covering every agent path.
"""

from __future__ import annotations

import argparse
import sys

from .config import settings
from .graph import handle_request
from .knowledge_base import get_kb
from .store import reset_runtime

DEMO = [
    ("jdoe", "How do I set up the authenticator app on my new phone?"),
    ("asmith", "I'm locked out of my account again, can you unlock it?"),
    ("pchen", "Is the VPN down? It keeps saying gateway not responding"),
    ("ceo", "I got an MFA prompt I didn't start and my password is Harbor2026!"),
    ("rpatel", "How do I request a standing desk for my office?"),
]


def print_result(user: str, message: str, state: dict) -> None:
    bar = "─" * 78
    print(f"\n{bar}\n👤 {user}: {message}\n{bar}")
    for t in state["trace"]:
        print(f"  ▸ {t['agent']:<10} {t['ms']:>5} ms  {t['detail']}")
    if state.get("redactions"):
        print(f"  🔒 redacted before LLM/ticket: {', '.join(state['redactions'])}")
    print(f"\n🤖 {state['answer']}\n")
    if state.get("sources"):
        print("  📚 Sources: " + " | ".join(state["sources"]))
    print(f"  ⏱  {state['latency_ms']} ms total · path: {' → '.join(state['path'])} · mode: {state['mode']}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("message", nargs="?", help="Message to send (omit to run the scripted demo)")
    p.add_argument("--user", default="jdoe", help="Requester username (see data/directory.json)")
    p.add_argument("--reset", action="store_true", help="Restore seed data and clear tickets first")
    args = p.parse_args(argv)

    if args.reset or not args.message:
        reset_runtime()
    kb = get_kb()
    kb.llm.warm_up()
    print(f"Bayview IT Assistant · mode: {settings().mode_label} · KB: {kb.size['articles']} articles / "
          f"{kb.size['chunks']} chunks · embeddings: {kb.llm.embedding_model}")
    for user, msg in ([(args.user, args.message)] if args.message else DEMO):
        print_result(user, msg, handle_request(msg, user))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
