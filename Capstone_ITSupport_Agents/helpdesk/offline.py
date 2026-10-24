"""Rule-based stand-ins for the LLM calls, used only when no GEMINI_API_KEY is available.

These make the system runnable end to end (CI, demos without quota). They are also a useful
baseline: the evaluation report compares Gemini against these rules.
"""

from __future__ import annotations

import re

from . import store
from .schemas import IntakeResult

_CATEGORY_RULES: list[tuple[str, str]] = [
    ("security", r"phish|suspicious (e-?mail|link)|clicked (a|the) link|hacked|compromis|stolen|ransom|virus|malware"
                 r"|(lost|left) my (company )?(laptop|phone)|(didn'?t|never|did not) (start|request|initiate)"),
    ("account_lockout", r"locked|lockout|lock(ed)? out|unlock"),
    ("password_reset", r"password|forgot|reset my|expired"),
    ("account_access", r"\bmfa\b|authenticator|2fa|two.factor|new phone"),
    ("vpn", r"\bvpn\b|globalconnect|gateway|remote access"),
    ("email", r"outlook|e-?mail|inbox|mailbox|calendar"),
    ("printer", r"print|scan|toner|paper jam"),
    ("software", r"install|software|\bapp\b|application|license|company portal|zoom|slack|tableau|python"),
    ("network", r"wi-?fi|wireless|internet|network|ethernet"),
    ("hardware", r"slow|freez|laptop|battery|blue screen|bsod|overheat|fan|monitor|keyboard"),
    ("access_request", r"access to|permission|shared (drive|folder|mailbox)|need access"),
]


def classify(message: str, requester: str) -> IntakeResult:
    text = message.lower()
    category = next((c for c, pat in _CATEGORY_RULES if re.search(pat, text)), "other")
    wants_human = bool(re.search(r"(talk|speak) to (a |an )?(human|person|agent|someone|technician)|real person", text))

    action, request_type = "none", "question"
    asks_to_do = re.search(r"\b(please|can you|could you|i need you to|reset|unlock|check|is the|are the|is there)\b", text)
    how_to = re.search(r"\bhow (do|can|to)\b|\bwhat (is|are)\b|\bwhere\b", text)
    if re.search(r"\blogs?\b|why (do|does|did|am|is) .*(lock|fail|keep)", text) and asks_to_do:
        action, request_type = "analyze_logs", "action"
    elif category == "account_lockout" and not how_to:
        action, request_type = "unlock_account", "action"
    elif category == "password_reset" and re.search(r"reset|forgot|expired", text) and not how_to:
        action, request_type = "reset_password", "action"
    elif re.search(r"\b(down|outage|status|anyone else|everyone|whole (floor|team|office))\b", text):
        action, request_type = "check_service_status", "action"
    if category == "security":
        action, request_type = "none", "incident"

    service = "none"
    for svc, pat in [("vpn", r"vpn|globalconnect"), ("email", r"outlook|e-?mail"), ("wifi", r"wi-?fi|wireless"),
                     ("printing", r"print"), ("hr_portal", r"hr portal|payroll portal|benefits"), ("sso", r"\bsso\b|single sign"),
                     ("company_portal", r"company portal")]:
        if re.search(pat, text):
            service = svc
            break

    # Only treat a word as a target username if it exists in the directory ("unlock jdoe", "for asmith").
    target = ""
    for word in re.findall(r"\b(?:for|unlock|reset|user(?:name)?)\s+([a-z]{2,10})\b", text):
        if store.get_user(word) and word != requester:
            target = word
            break

    urgency = "medium"
    if category == "security" or re.search(r"urgent|asap|emergency|(can'?t|cannot|nobody can|no one can) work|critical|client meeting", text):
        urgency = "high" if category != "security" else "critical"
    elif request_type == "question":
        urgency = "low"

    return IntakeResult(category=category, request_type=request_type, action=action, target_username=target,
                        service=service, urgency=urgency, wants_human=wants_human,
                        summary=message.strip()[:160], confidence=0.6 if category != "other" else 0.3)


def extractive_answer(question: str, hits) -> str:
    """Return the most relevant KB section, lightly trimmed, as the answer."""
    best = hits[0].chunk
    body = best.text.split("\n", 1)[1].strip() if "\n" in best.text else best.text
    return f"Here's what our knowledge base says ({best.title} › {best.section}):\n\n{body}"
