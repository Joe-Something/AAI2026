"""The four agents. Each is a LangGraph node: it reads the shared state and returns a partial update.

    Intake     - redacts PII, classifies the request, extracts entities (Gemini structured output)
    Knowledge  - RAG over the KB in Chroma; answers only from retrieved context, with citations
    Workflow   - runs an automation tool (unlock, reset, status check, log analysis)
    Escalation - writes a technician handoff and opens a routed ticket (optionally a GitHub issue)

Knowledge and Workflow hand off to Escalation by setting ``escalation_reason``.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from . import offline, tools
from .config import settings
from .knowledge_base import get_kb
from .llm import get_llm
from .privacy import redact
from .schemas import HandoffSummary, IntakeResult, KnowledgeAnswer

SLA_FIRST_RESPONSE = {"P1": "15 minutes", "P2": "1 hour", "P3": "4 business hours", "P4": "1 business day"}


class HelpdeskState(TypedDict, total=False):
    requester: str
    message: str
    history: list[dict]
    redactions: list[str]
    intake: dict
    retrieved: list[dict]
    tool_result: dict
    answer: str
    sources: list[str]
    escalation_reason: str
    ticket: dict
    resolved_by: str
    trace: Annotated[list[dict], operator.add]  # every agent appends; LangGraph concatenates


def _trace(agent: str, started: float, detail: str, **extra: Any) -> list[dict]:
    return [{"agent": agent, "ms": round((time.perf_counter() - started) * 1000), "detail": detail, **extra}]


# ---------------------------------------------------------------------------- Intake
INTAKE_SYSTEM = """You are the Intake Agent for the Bayview Logistics IT Service Desk.
Classify the employee's message and extract entities. Rules:
- request_type "action" + an action when the user wants their account unlocked, their password reset,
  a system's status checked ("is VPN down?", "is anyone else affected?"), or their sign-in logs investigated
  ("why does my account keep locking?").
- request_type "question" with action "none" for how-to questions, even about passwords ("how do I reset my password?").
- category "security" and request_type "incident" for phishing, clicked links, compromised accounts, unexpected MFA prompts,
  lost/stolen devices, malware. Never pick an automation for security incidents.
- target_username: only when the user asks for an action on a different person's account.
- urgency: critical for security incidents or many users affected; high when the user cannot work; low for how-to.
- wants_human: only when they explicitly ask for a person."""


def intake_agent(state: HelpdeskState) -> dict:
    t0 = time.perf_counter()
    message, redactions = redact(state["message"])
    llm = get_llm()
    if llm.offline:
        result = offline.classify(message, state["requester"])
    else:
        history = "\n".join(f"{h['role']}: {redact(h['content'])[0]}" for h in state.get("history", [])[-6:])
        prompt = (f"Requester username: {state['requester']}\n"
                  + (f"Earlier conversation:\n{history}\n" if history else "")
                  + f"Latest message: {message}")
        result = llm.structured(prompt, IntakeResult, INTAKE_SYSTEM)
    intake = result.model_dump()
    detail = f"{intake['category']} / {intake['request_type']} / action={intake['action']} / urgency={intake['urgency']}"
    return {"message": message, "redactions": redactions, "intake": intake,
            "trace": _trace("Intake", t0, detail, redactions=redactions, confidence=intake["confidence"])}


def route_from_intake(state: HelpdeskState) -> str:
    """Deterministic routing policy on top of the LLM classification (auditable, testable)."""
    i = state["intake"]
    if i["category"] == "security" or i["request_type"] == "incident" or i["wants_human"]:
        return "escalation"
    if i["action"] != "none":
        return "workflow"
    return "knowledge"


# ---------------------------------------------------------------------------- Knowledge
KNOWLEDGE_SYSTEM = """You are the Knowledge Agent for the Bayview Logistics IT Service Desk.
Answer ONLY from the numbered context chunks. Do not invent URLs, settings or policies.
If the context does not answer the question, set grounded=false and say so briefly.
Write for a non-technical employee: a one-line answer, then short numbered steps. Max 150 words.
Cite the chunk ids you used in cited_ids."""


def knowledge_agent(state: HelpdeskState) -> dict:
    t0 = time.perf_counter()
    cfg, llm, kb = settings(), get_llm(), get_kb()
    intake = state["intake"]
    hits = kb.search(state["message"], k=cfg.retrieval_top_k)
    retrieved = [{"id": h.chunk.id, "citation": h.citation, "score": h.score} for h in hits]

    if not hits or hits[0].score < cfg.min_relevance:
        top = hits[0].score if hits else 0
        return {"retrieved": retrieved,
                "escalation_reason": f"No knowledge-base article is relevant enough (best match {top:.2f} < {cfg.min_relevance})",
                "trace": _trace("Knowledge", t0, f"retrieval below threshold (top={top:.2f})", retrieved=retrieved)}

    # Collaboration with the Workflow tools: surface live outages related to the question.
    banner = ""
    if intake.get("service", "none") != "none":
        status = tools.check_service_status(intake["service"])
        if status.ok and status.data["impacted"]:
            banner = f"⚠️ Known issue right now - {status.message}\n\n"

    if llm.offline:
        answer, grounded, cited = offline.extractive_answer(state["message"], hits), True, [hits[0].chunk.id]
    else:
        context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
        out = llm.structured(f"Context:\n{context}\n\nEmployee question: {state['message']}", KnowledgeAnswer, KNOWLEDGE_SYSTEM)
        answer, grounded, cited = out.answer, out.grounded, out.cited_ids

    if not grounded:
        return {"retrieved": retrieved, "escalation_reason": "Knowledge base does not cover this question",
                "trace": _trace("Knowledge", t0, "model reported answer not grounded in context", retrieved=retrieved)}

    by_id = {h.chunk.id: h for h in hits}
    sources = list(dict.fromkeys(by_id[c].citation for c in cited if c in by_id)) or [hits[0].citation]
    tools.log_resolved(state["requester"], intake["category"], intake["summary"][:80], answer[:500], "Knowledge Agent")
    return {"retrieved": retrieved, "answer": banner + answer, "sources": sources, "resolved_by": "Knowledge Agent",
            "trace": _trace("Knowledge", t0, f"answered from {len(sources)} source(s), top score {hits[0].score:.2f}", retrieved=retrieved)}


# ---------------------------------------------------------------------------- Workflow
WORKFLOW_SYSTEM = """You are the Workflow Agent for the Bayview Logistics IT Service Desk.
An automation just ran. Explain the outcome to the employee in plain language (max 120 words):
what was done, what they should do next, and any tip from the log findings or knowledge snippet.
Never reveal raw log lines, IP addresses, or other users' names."""

_LOG_FOR_SERVICE = {"vpn": "vpn"}


def workflow_agent(state: HelpdeskState) -> dict:
    t0 = time.perf_counter()
    intake, requester = state["intake"], state["requester"]
    action = intake["action"]
    target = intake.get("target_username") or None

    if action in ("reset_password", "unlock_account"):
        result = tools.WORKFLOW_ACTIONS[action](requester, target)
    elif action == "check_service_status":
        svc = intake.get("service")
        result = tools.check_service_status(None if svc in (None, "none") else svc)
    else:  # analyze_logs: scoped to the requester; investigating someone else's logs is a technician task
        result = tools.analyze_logs(_LOG_FOR_SERVICE.get(intake.get("service", ""), "auth"), target or requester)

    # A user-reported outage the status page doesn't know about yet is new information for Network Ops.
    if (result.tool == "check_service_status" and result.ok and not result.data["impacted"]
            and intake.get("service", "none") != "none" and intake["urgency"] in ("high", "critical")):
        result.needs_escalation = True
        result.escalation_reason = f"User reports a {intake['service']} outage that is not on the status page yet"

    detail = f"{result.tool} -> {'ok' if result.ok else 'refused'}"
    if not result.ok or result.needs_escalation:
        return {"tool_result": result.to_dict(),
                "escalation_reason": result.escalation_reason or result.message,
                "trace": _trace("Workflow", t0, detail + " (escalating)", tool=result.to_dict())}

    # Pull one KB tip so the user gets prevention advice, not just the action result.
    hits = get_kb().search(f"{intake['category']} {state['message']}", k=1)
    tip = hits[0] if hits and hits[0].score >= settings().min_relevance else None
    llm = get_llm()
    if llm.offline:
        answer = f"✅ {result.message}"
        findings = [f for f in result.data.get("log_findings", result.data.get("findings", [])) if requester in f]
        if findings:
            answer += "\n\nWhat I found in the sign-in logs: " + "; ".join(f.split(": ", 1)[-1] for f in findings)
        if tip:
            answer += f"\n\nTip from {tip.chunk.doc_id}: " + tip.chunk.text.split("\n", 1)[-1].strip().split("\n")[0]
    else:
        prompt = (f"Employee message: {state['message']}\nAutomation result: {result.to_dict()}\n"
                  + (f"Knowledge snippet [{tip.chunk.id}]: {tip.chunk.text}" if tip else ""))
        answer = llm.text(prompt, WORKFLOW_SYSTEM)

    tools.log_resolved(requester, intake["category"], intake["summary"][:80], result.message, f"Workflow Agent ({result.tool})")
    return {"tool_result": result.to_dict(), "answer": answer, "resolved_by": f"Workflow Agent ({result.tool})",
            "sources": [tip.citation] if tip else [],
            "trace": _trace("Workflow", t0, detail, tool=result.to_dict())}


# ---------------------------------------------------------------------------- Escalation
ESCALATION_SYSTEM = """You are the Escalation Agent for the Bayview Logistics IT Service Desk.
Write a handoff for the human technician so the employee never has to repeat themselves.
Be factual; include what automation or knowledge lookups already happened."""


def escalation_agent(state: HelpdeskState) -> dict:
    t0 = time.perf_counter()
    intake, requester = state["intake"], state["requester"]
    reason = state.get("escalation_reason") or (
        "Security incident - always handled by Security Ops" if intake["category"] == "security"
        else "User asked for a person" if intake["wants_human"] else "Routed to a technician")
    steps = []
    if state.get("retrieved"):
        steps.append("Searched knowledge base: " + ", ".join(r["id"] for r in state["retrieved"][:3]))
    if state.get("tool_result"):
        tr = state["tool_result"]
        steps.append(f"Automation {tr['tool']}: {tr['message']}")
    transcript = "\n".join(f"{h['role']}: {redact(h['content'])[0]}" for h in state.get("history", [])) + f"\nuser: {state['message']}"

    llm = get_llm()
    if llm.offline:
        handoff = HandoffSummary(title=intake["summary"][:80], summary=f"{intake['summary']} Escalation reason: {reason}.",
                                 steps_tried=steps or ["None - routed directly to a technician"])
    else:
        prompt = (f"Requester: {requester}\nClassification: {intake}\nEscalation reason: {reason}\n"
                  f"Steps already taken: {steps}\nTranscript:\n{transcript}")
        handoff = llm.structured(prompt, HandoffSummary, ESCALATION_SYSTEM)

    result = tools.create_ticket(requester, intake["category"], intake["urgency"], handoff.title,
                                 handoff.summary, handoff.steps_tried, transcript)
    t = result.data
    answer = (f"I've passed this to a person. Ticket **{t['number']}** is open with **{t['queue']}** "
              f"(priority {t['priority']}; first response within {SLA_FIRST_RESPONSE[t['priority']]}).\n\n"
              f"Why: {reason}.\n\nThey'll see everything we've covered, so you won't need to repeat yourself.")
    if intake["category"] == "security":
        answer += ("\n\nWhile you wait: don't click any more links, and if you entered your password anywhere, "
                   "change it now from a trusted device (KB-011).")
    if t.get("external_url"):
        answer += f"\n\nTracked on GitHub: {t['external_url']}"
    return {"ticket": t, "answer": answer, "escalation_reason": reason, "resolved_by": "Escalated to human",
            "sources": ["KB-012 · Service Desk priorities, SLAs and escalation"],
            "trace": _trace("Escalation", t0, f"{t['number']} -> {t['queue']} ({t['priority']})", reason=reason)}


def route_to_escalation_if_needed(state: HelpdeskState) -> str:
    return "escalation" if state.get("escalation_reason") and not state.get("answer") else "end"
