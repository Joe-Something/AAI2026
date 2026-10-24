"""Workflow tools: the automations the Workflow Agent (and the MCP server) can run.

Every tool returns a ``ToolResult`` so callers can tell success from a refusal that must
be escalated. Tools enforce their own guardrails (self-service only, MFA required,
security holds) instead of trusting the LLM to have checked.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from . import store
from .config import settings


@dataclass
class ToolResult:
    tool: str
    ok: bool
    message: str
    data: dict = field(default_factory=dict)
    needs_escalation: bool = False
    escalation_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- log analysis
_LOG_LINE = re.compile(r"^(?P<ts>\S+)\s+(?P<level>[A-Z]+)\s+(?P<source>\S+)\s+(?P<event>\S+)\s*(?P<kv>.*)$")


def _parse_log(raw: str) -> list[dict]:
    events = []
    for line in raw.splitlines():
        m = _LOG_LINE.match(line.strip())
        if not m:
            continue
        ev = m.groupdict()
        ev.update(dict(re.findall(r"(\w+)=(\S+)", ev.pop("kv"))))
        events.append(ev)
    return events


def analyze_logs(log: str, username: str | None = None) -> ToolResult:
    """Parse a system log and surface findings (lockout causes, gateway health, security signals)."""
    try:
        events = _parse_log(store.read_log(log))
    except FileNotFoundError:
        return ToolResult("analyze_logs", False, f"No log named '{log}'. Available: vpn, auth.")
    if username:
        events = [e for e in events if e.get("user") in (username, None)]

    levels = Counter(e["level"] for e in events)
    reasons = Counter(e.get("reason") for e in events if e["level"] == "ERROR" and e.get("reason"))
    per_user: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        if e.get("user"):
            per_user[e["user"]][e["event"]] += 1

    findings: list[str] = []
    security_alert = False
    for e in events:
        if e["event"] in ("impossible_travel", "mfa_denied"):
            security_alert = True
            findings.append(f"SECURITY: {e['event']} for user {e.get('user')} from {e.get('new_src') or e.get('src')} at {e['ts']}")
    for user, counts in per_user.items():
        mobile_fails = sum(1 for e in events if e.get("user") == user and e.get("reason") == "bad_password" and "mobile" in e.get("app", ""))
        if counts.get("account_locked") and mobile_fails >= 3:
            findings.append(f"{user}: lockout caused by {mobile_fails} bad-password attempts from a mobile mail app (stale saved password, see KB-002)")
        elif counts.get("account_locked"):
            findings.append(f"{user}: account locked after repeated failed sign-ins")
    unhealthy = [e for e in events if e["event"] == "health_check" and int(e.get("latency_ms", 0)) > int(e.get("threshold_ms", 1e9))]
    if unhealthy:
        gw = Counter(e["source"] for e in unhealthy).most_common(1)[0][0]
        tls = sum(1 for e in events if e["source"] == gw and e["event"] == "tls_handshake_failed")
        findings.append(f"{gw}: {len(unhealthy)} failed health checks (latency up to {max(int(e['latency_ms']) for e in unhealthy)} ms) and {tls} TLS handshake timeouts - gateway degraded")
    for e in events:
        if e.get("reason") == "device_cert_expired":
            findings.append(f"{e.get('user')}: VPN rejected because the device certificate expired (see KB-004)")

    data = {"log": log, "filter_user": username, "events": len(events), "levels": dict(levels),
            "top_error_reasons": reasons.most_common(5), "findings": findings, "security_alert": security_alert}
    msg = f"Analyzed {len(events)} events in {log}.log" + (f" for {username}" if username else "") + ". "
    msg += ("Findings: " + "; ".join(findings)) if findings else "No anomalies found."
    return ToolResult("analyze_logs", True, msg, data,
                      needs_escalation=security_alert,
                      escalation_reason="Security signal detected in logs" if security_alert else "")


def _security_hold(username: str) -> str | None:
    result = analyze_logs("auth", username)
    if result.data.get("security_alert"):
        return next(f for f in result.data["findings"] if f.startswith("SECURITY"))
    return None


# ---------------------------------------------------------------- account automations
def _check_self_service(tool: str, requester: str, target: str | None) -> tuple[dict | None, ToolResult | None]:
    target = (target or requester).lower()
    if target != requester.lower():
        return None, ToolResult(tool, False, f"You can only run {tool.replace('_', ' ')} for your own account. "
                                "Requests for other people need identity verification by a technician.",
                                needs_escalation=True, escalation_reason="Action requested for another user's account")
    user = store.get_user(target)
    if not user:
        return None, ToolResult(tool, False, f"No account named '{target}' in the directory.",
                                needs_escalation=True, escalation_reason="Unknown account")
    if not user["mfa_enrolled"]:
        return None, ToolResult(tool, False, "Your account has no MFA method enrolled, so I can't verify your identity automatically.",
                                needs_escalation=True, escalation_reason="Identity verification requires a technician (no MFA enrolled)")
    hold = _security_hold(target)
    if hold:
        return None, ToolResult(tool, False, "I've paused this request because of unusual sign-in activity on your account. "
                                "A security analyst will contact you.", {"signal": hold},
                                needs_escalation=True, escalation_reason=f"Security hold: {hold}")
    return user, None


def unlock_account(requester: str, username: str | None = None) -> ToolResult:
    """Unlock the requester's own account after MFA verification."""
    user, refusal = _check_self_service("unlock_account", requester, username)
    if refusal:
        return refusal
    cause = analyze_logs("auth", user["username"]).data["findings"]
    if not user["locked"]:
        return ToolResult("unlock_account", True, f"Account {user['username']} is not locked; no change needed.",
                          {"username": user["username"], "was_locked": False, "log_findings": cause})
    store.update_user(user["username"], locked=False)
    return ToolResult("unlock_account", True,
                      f"MFA push approved. Account {user['username']} is unlocked.",
                      {"username": user["username"], "was_locked": True, "mfa_verified": True, "log_findings": cause})


def reset_password(requester: str, username: str | None = None) -> ToolResult:
    """Send a one-time password reset link to the requester's own account after MFA verification."""
    user, refusal = _check_self_service("reset_password", requester, username)
    if refusal:
        return refusal
    was_locked = user["locked"]
    store.update_user(user["username"], locked=False)
    return ToolResult("reset_password", True,
                      f"MFA push approved. A one-time reset link (valid 15 minutes) was sent to {user['email']}."
                      + (" Your account was also unlocked." if was_locked else ""),
                      {"username": user["username"], "mfa_verified": True, "unlocked": was_locked, "link_ttl_min": 15})


# ---------------------------------------------------------------- service status
def check_service_status(service: str | None = None) -> ToolResult:
    """Return live status for one service (vpn, email, sso, wifi, printing, hr_portal, company_portal) or all."""
    services = store.load_services()
    if service:
        match = [s for s in services if s["name"] == service.lower()]
        if not match:
            return ToolResult("check_service_status", False, f"Unknown service '{service}'.",
                              {"known": [s["name"] for s in services]})
        services = match
    lines = [f"{s['display']}: {s['status'].upper()}" + (f" - {s['note']}" if s["note"] else "") for s in services]
    impacted = [s["name"] for s in services if s["status"] != "operational"]
    return ToolResult("check_service_status", True, "\n".join(lines), {"services": services, "impacted": impacted})


# ---------------------------------------------------------------- tickets
PRIORITY_QUEUE = {
    "security": ("P1", "Security Ops"),
    "account_lockout": ("P2", "Service Desk L1"),
    "password_reset": ("P2", "Service Desk L1"),
    "account_access": ("P3", "Identity & Access"),
    "access_request": ("P4", "Identity & Access"),
    "vpn": ("P3", "Network Ops"),
    "network": ("P3", "Network Ops"),
    "email": ("P3", "Service Desk L1"),
    "printer": ("P4", "Desktop Support L2"),
    "hardware": ("P3", "Desktop Support L2"),
    "software": ("P4", "Desktop Support L2"),
    "other": ("P4", "Service Desk L1"),
}


def triage(category: str, urgency: str, requester: str) -> tuple[str, str]:
    """Map category + urgency + VIP status to (priority, queue) per KB-012."""
    priority, queue = PRIORITY_QUEUE.get(category, PRIORITY_QUEUE["other"])
    rank = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}[priority]
    if urgency == "critical":
        rank = 1
    elif urgency == "high":
        rank = min(rank, 2)
    user = store.get_user(requester)
    if user and user.get("vip"):
        rank = min(rank, 2)
    return f"P{rank}", queue


def create_ticket(requester: str, category: str, urgency: str, title: str, summary: str,
                  steps_tried: list[str] | None = None, transcript: str = "") -> ToolResult:
    """Open a ticket routed by the SLA policy; mirrors it to GitHub Issues when configured."""
    priority, queue = triage(category, urgency, requester)
    t = store.create_ticket(requester=requester, category=category, priority=priority, queue=queue,
                            title=title, summary=summary, steps_tried=steps_tried, transcript=transcript)
    url = _mirror_to_github(t)
    if url:
        store.set_external_url(t["id"], url)
        t["external_url"] = url
    return ToolResult("create_ticket", True, f"Ticket {t['number']} opened ({priority}, queue: {queue}).", t)


def log_resolved(requester: str, category: str, title: str, summary: str, resolved_by: str) -> dict:
    """Record a request the assistant resolved itself, for reporting (deflection rate)."""
    return store.create_ticket(requester=requester, category=category, priority="P4", queue="Self-service",
                               title=title, summary=summary, status="resolved", resolved_by=resolved_by)


def _mirror_to_github(ticket: dict) -> str | None:
    cfg = settings()
    if not (cfg.github_token and cfg.github_repo):
        return None
    body = (f"**{ticket['number']}** · {ticket['priority']} · queue: {ticket['queue']}\n\n"
            f"**Requester:** {ticket['requester']}\n\n**Summary:** {ticket['summary']}\n\n"
            f"**Steps already tried:**\n" + "\n".join(f"- {s}" for s in ticket["steps_tried"]) +
            "\n\n_Opened automatically by the Bayview IT Assistant Escalation Agent._")
    payload = {"title": f"[{ticket['priority']}] {ticket['title']}", "body": body,
               "labels": ["helpdesk", ticket["priority"], ticket["category"]]}
    req = urllib.request.Request(
        f"https://api.github.com/repos/{cfg.github_repo}/issues", data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {cfg.github_token}", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["html_url"]
    except Exception:
        return None  # GitHub is a mirror; the local ticket is the system of record.


# Registry used by the Workflow Agent and the MCP server.
WORKFLOW_ACTIONS = {
    "reset_password": reset_password,
    "unlock_account": unlock_account,
    "check_service_status": check_service_status,
    "analyze_logs": analyze_logs,
}
