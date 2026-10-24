"""PII redaction applied before text is sent to the LLM, stored in tickets, or logged.

Usernames and work emails are kept (agents need them to act); secrets and personal
identifiers are masked. This is a guardrail, not a DLP product: the scaling document
describes replacing it with a managed service such as Google Cloud DLP.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "my password is hunter2" / "pwd: hunter2" -> never forward a password anywhere.
    ("PASSWORD", re.compile(r"(?i)\b(pass(?:word|code|wd)?|pwd)\b(\s*(?:is|was|=|:)\s*)(\S+)")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@(?!bayview\.example)[\w-]+\.[\w.]+\b")),  # personal emails only
]


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, list_of_redaction_types_applied)."""
    found: list[str] = []
    for label, pattern in _PATTERNS:
        if label == "PASSWORD":
            text, n = pattern.subn(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED-PASSWORD]", text)
        else:
            text, n = pattern.subn(f"[REDACTED-{label}]", text)
        if n:
            found.append(label)
    return text, found
