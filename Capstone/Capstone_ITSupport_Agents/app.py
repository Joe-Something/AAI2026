"""Gradio web UI for the Bayview IT Assistant.

    python app.py            # http://127.0.0.1:7860
    python app.py --share    # public link (used in Colab)

Layout: chat on the left; on the right, an "under the hood" panel showing which agents ran,
retrieval scores, the automation result, and the live ticket queue. Showing the agent path
builds user trust and makes the demo explain itself.
"""

from __future__ import annotations

import argparse

import gradio as gr

from helpdesk import store
from helpdesk.config import settings
from helpdesk.graph import handle_request
from helpdesk.knowledge_base import get_kb

AGENT_ICON = {"Intake": "🧭", "Knowledge": "📚", "Workflow": "⚙️", "Escalation": "🙋"}
EXAMPLES = [
    ("jdoe", "How do I set up the authenticator app on my new phone?"),
    ("asmith", "I'm locked out of my account again, can you unlock it?"),
    ("pchen", "Is the VPN down? It keeps saying gateway not responding"),
    ("asmith", "Why does my account keep locking? Can you check the logs?"),
    ("ceo", "I got an MFA prompt I didn't start"),
    ("jdoe", "Please reset the password for asmith"),
]


def _user_choices() -> list[tuple[str, str]]:
    return [(f"{u['name']} ({u['username']}) · {u['department']}", u["username"]) for u in store.list_users()]


def _trace_markdown(state: dict) -> str:
    lines = [f"**Path:** {' → '.join(AGENT_ICON[a] + ' ' + a for a in state['path'])}  ",
             f"**Total:** {state['latency_ms']} ms · **Mode:** `{state['mode']}`", ""]
    for t in state["trace"]:
        lines.append(f"- {AGENT_ICON[t['agent']]} **{t['agent']}** ({t['ms']} ms): {t['detail']}")
    if state.get("redactions"):
        lines.append(f"- 🔒 Redacted before LLM / ticket: {', '.join(state['redactions'])}")
    if state.get("retrieved"):
        lines += ["", "**Retrieved chunks** (cosine similarity)", "", "| Chunk | Score |", "|---|---|"]
        lines += [f"| {r['citation']} | {r['score']:.2f} |" for r in state["retrieved"]]
    if state.get("tool_result"):
        tr = state["tool_result"]
        lines += ["", f"**Automation:** `{tr['tool']}` → {'✅ ok' if tr['ok'] else '⛔ refused'}"]
    if state.get("ticket"):
        t = state["ticket"]
        lines += ["", f"**Ticket:** {t['number']} · {t['priority']} · {t['queue']}", f"> {t['summary']}"]
    return "\n".join(lines)


def _tickets_table() -> list[list]:
    return [[t["number"], t["status"], t["priority"], t["queue"], t["requester"], t["title"][:60]]
            for t in store.list_tickets(limit=15)]


def respond(message: str, history: list[dict], username: str):
    if not message.strip():
        return history, gr.update(), gr.update(), ""
    prior = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])}
             for m in history]
    state = handle_request(message, username, prior)
    reply = state["answer"]
    if state.get("sources"):
        reply += "\n\n<sub>📚 " + " · ".join(state["sources"]) + "</sub>"
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return history, _trace_markdown(state), _tickets_table(), ""


def reset_demo():
    store.reset_runtime()
    return [], "_Demo data reset._", _tickets_table()


def build_ui() -> gr.Blocks:
    kb = get_kb()
    with gr.Blocks(title="Bayview IT Assistant") as demo:
        gr.Markdown(
            f"## 🛟 Bayview IT Assistant\n"
            f"Multi-agent IT support · Intake → Knowledge / Workflow / Escalation · "
            f"mode `{settings().mode_label}` · KB {kb.size['articles']} articles / {kb.size['chunks']} chunks")
        with gr.Row():
            with gr.Column(scale=3):
                user = gr.Dropdown(_user_choices(), value="jdoe", label="Signed in as (simulated SSO)")
                chat = gr.Chatbot(height=460, label="Conversation", elem_id="chat",
                                  placeholder="Ask about passwords, VPN, email, printers, software…")
                with gr.Row():
                    box = gr.Textbox(placeholder="Describe your IT issue…", show_label=False, scale=5, autofocus=True, elem_id="msgbox")
                    send = gr.Button("Send", variant="primary", scale=1, elem_id="send")
                gr.Markdown("**Try:**")
                with gr.Row():
                    for who, text in EXAMPLES:
                        btn = gr.Button(f"{who}: {text[:38]}…", size="sm")
                        btn.click(lambda w=who, t=text: (w, t), outputs=[user, box])
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Under the hood")
                trace = gr.Markdown("_Agent activity appears here after each message._", elem_id="trace")
                tickets = gr.Dataframe(_tickets_table(), headers=["Ticket", "Status", "Pri", "Queue", "User", "Title"],
                                       label="Ticket queue (live)", interactive=False, wrap=True, elem_id="tickets")
                gr.Button("Reset demo data", size="sm").click(reset_demo, outputs=[chat, trace, tickets])
        for trigger in (send.click, box.submit):
            trigger(respond, inputs=[box, chat, user], outputs=[chat, trace, tickets, box])
        user.change(lambda: [], outputs=chat)  # new identity = new conversation
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_ui().launch(share=a.share, server_port=a.port)
