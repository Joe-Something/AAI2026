"""Offline unit and integration tests: privacy, tools/guardrails, retrieval, and graph routing."""

from helpdesk import store, tools
from helpdesk.graph import handle_request
from helpdesk.knowledge_base import get_kb, load_chunks
from helpdesk.privacy import redact


# ---------------------------------------------------------------- privacy
def test_redacts_password_and_ssn_but_keeps_work_email():
    text, found = redact("my password is Hunter2! ssn 123-45-6789, email jdoe@bayview.example or me@gmail.com")
    assert "Hunter2" not in text and "123-45-6789" not in text and "me@gmail.com" not in text
    assert "jdoe@bayview.example" in text
    assert set(found) == {"PASSWORD", "SSN", "EMAIL"}


# ---------------------------------------------------------------- tools and guardrails
def test_unlock_own_locked_account_succeeds_and_explains_cause():
    r = tools.unlock_account("asmith")
    assert r.ok and r.data["was_locked"]
    assert store.get_user("asmith")["locked"] is False
    assert any("mobile mail app" in f for f in r.data["log_findings"])


def test_cannot_reset_someone_elses_password():
    r = tools.reset_password("jdoe", "asmith")
    assert not r.ok and r.needs_escalation


def test_no_mfa_means_no_automation():
    r = tools.reset_password("rpatel")
    assert not r.ok and "MFA" in r.message


def test_security_hold_blocks_automation_on_suspicious_account():
    r = tools.reset_password("ceo")
    assert not r.ok and r.escalation_reason.startswith("Security hold")


def test_log_analysis_flags_degraded_gateway():
    r = tools.analyze_logs("vpn")
    assert any("vpn-gw-02" in f and "degraded" in f for f in r.data["findings"])


def test_log_reader_rejects_path_traversal():
    assert not tools.analyze_logs("../directory").ok


def test_triage_vip_bumps_priority():
    assert tools.triage("printer", "low", "jdoe") == ("P4", "Desktop Support L2")
    assert tools.triage("printer", "low", "mrivera")[0] == "P2"


# ---------------------------------------------------------------- knowledge base
def test_chunking_by_section():
    chunks = load_chunks()
    assert len({c.doc_id for c in chunks}) == 12
    assert any(c.doc_id == "KB-004" and c.section == "Error messages" for c in chunks)


def test_retrieval_finds_vpn_article():
    hits = get_kb().search("GlobalConnect portal address", k=3)
    assert hits[0].chunk.doc_id == "KB-004"


# ---------------------------------------------------------------- graph routing (end to end)
def test_how_to_goes_to_knowledge_with_citation():
    s = handle_request("How do I set up the authenticator app on my new phone?", "jdoe")
    assert s["path"] == ["Intake", "Knowledge"] and s["sources"][0].startswith("KB-003")


def test_unlock_goes_to_workflow_and_logs_resolution():
    s = handle_request("I'm locked out, please unlock my account", "asmith")
    assert s["path"] == ["Intake", "Workflow"]
    assert store.list_tickets(status="resolved")[0]["resolved_by"].startswith("Workflow Agent")


def test_security_incident_escalates_p1_and_redacts():
    s = handle_request("I clicked a phishing link and typed my password is Summer2026", "jdoe")
    assert s["path"] == ["Intake", "Escalation"]
    assert s["ticket"]["priority"] == "P1" and s["ticket"]["queue"] == "Security Ops"
    assert "Summer2026" not in s["ticket"]["transcript"]


def test_failed_workflow_hands_off_to_escalation():
    s = handle_request("Please reset the password for asmith", "jdoe")
    assert s["path"] == ["Intake", "Workflow", "Escalation"]


def test_out_of_scope_question_escalates_instead_of_guessing():
    s = handle_request("What's on the cafeteria menu today?", "tkim")
    assert s["path"][-1] == "Escalation"
