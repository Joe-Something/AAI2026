# Evaluation report - offline (no GEMINI_API_KEY)

37 scenarios · embeddings `offline-hashed-bow` · relevance threshold 0.15 · 0 LLM calls · 0 errors

| Metric | Value |
|---|---|
| Intake category accuracy | 89.2% |
| Action selection accuracy | 100.0% |
| Routing accuracy (final agent) | 97.3% |
| End-to-end pass rate | 86.5% |
| Retrieval hit@1 | 81.2% |
| Retrieval hit@4 | 100.0% |
| Escalation precision | 92.3% |
| Escalation recall | 100.0% |
| Containment (no human needed) | 64.9% |
| Latency p50 / p95 / max | 25.0 / 34.0 / 42 ms |

## Failures

| ID | Message | Expected | Got |
|---|---|---|---|
| K05 | How do I stop Outlook from asking for my password over and over? | email / none / knowledge | password_reset / none / knowledge |
| K12 | How long does it take for an access request to be approved? | access_request / none / knowledge | other / none / knowledge |
| K15 | How do I scan a document and send it to my email? | printer / none / knowledge | email / none / knowledge |
| K16 | What is the response time for a P2 ticket? | * / none / knowledge | other / none / escalation |
| W08 | Why does my account keep locking? Can you check the logs? | account_lockout / analyze_logs / workflow | other / analyze_logs / workflow |
