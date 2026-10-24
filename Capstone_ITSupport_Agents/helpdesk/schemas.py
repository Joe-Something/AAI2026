"""Typed contracts between agents. Gemini is asked to return JSON matching these models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "password_reset", "account_lockout", "account_access", "vpn", "email", "printer",
    "software", "network", "hardware", "access_request", "security", "other",
]
Action = Literal["none", "reset_password", "unlock_account", "check_service_status", "analyze_logs"]
Service = Literal["none", "vpn", "email", "sso", "wifi", "printing", "hr_portal", "company_portal"]


class IntakeResult(BaseModel):
    category: Category = Field(description="Best-fitting IT support category.")
    request_type: Literal["question", "action", "incident"] = Field(
        description="question = user wants information/how-to; action = user wants something done to their account or a system checked; incident = security event.")
    action: Action = Field(description="Automation that would resolve the request, or 'none'.")
    target_username: str = Field(description="Username the action is for, if the user named someone else. Empty string when it is the requester.")
    service: Service = Field(description="System the request is about, or 'none'.")
    urgency: Literal["low", "medium", "high", "critical"]
    wants_human: bool = Field(description="True if the user explicitly asks for a person.")
    summary: str = Field(description="One-sentence neutral summary of the request for a technician.")
    # No defaults or numeric bounds: the Gemini API rejects them in response schemas.
    confidence: float = Field(description="Confidence in the category, 0-1.")


class KnowledgeAnswer(BaseModel):
    answer: str = Field(description="Answer for the user, in short steps, using only the provided context.")
    cited_ids: list[str] = Field(description="Chunk ids (e.g. KB-004#2) the answer relies on.")
    grounded: bool = Field(description="False if the context does not contain the answer.")


class HandoffSummary(BaseModel):
    title: str = Field(description="Ticket title, under 80 characters.")
    summary: str = Field(description="2-3 sentence summary for the technician: problem, impact, relevant facts.")
    steps_tried: list[str] = Field(description="What the user or assistant already tried.")
