"""Evaluation harness: runs every scenario in eval/scenarios.json through the agent graph.

    python eval/run_eval.py                 # all scenarios
    python eval/run_eval.py --judge         # + LLM-as-judge satisfaction scores (needs an LLM: gemini or local)
    python eval/run_eval.py --pace 4        # seconds between Gemini calls (free-tier rate limits)

Metrics
  - Intake category accuracy and action accuracy
  - Routing accuracy: did the right agent produce the final answer?
  - Escalation precision / recall: escalate when (and only when) a human is needed
  - Retrieval hit@1 / hit@k for questions with a known source article
  - Containment rate: share of requests resolved without a human
  - Latency p50 / p95 / max
  - Satisfaction (1-5): LLM judge rating helpfulness, accuracy and tone of each answer

Outputs go to eval/results/<mode>/: results.csv, summary.json, report.md, charts.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from helpdesk.config import reset_settings, settings  # noqa: E402
from helpdesk.graph import handle_request  # noqa: E402
from helpdesk.knowledge_base import get_kb  # noqa: E402
from helpdesk.llm import get_llm  # noqa: E402
from helpdesk.store import reset_runtime  # noqa: E402

FINAL = {"Knowledge": "knowledge", "Workflow": "workflow", "Escalation": "escalation"}


class JudgeScore(BaseModel):
    helpfulness: int = Field(description="1-5: does the reply move the user toward resolution?")
    accuracy: int = Field(description="1-5: is it correct and consistent with the expected behaviour?")
    tone: int = Field(description="1-5: clear, friendly, non-technical where appropriate?")
    rationale: str = Field(description="One sentence.")


JUDGE_SYSTEM = """You evaluate an IT helpdesk assistant's reply as if you were the employee who asked.
Score 1-5 for helpfulness, accuracy and tone. An appropriate escalation to a human (e.g. security
incidents, actions on someone else's account, out-of-scope requests) deserves high scores if it is
clearly explained. Judge accuracy against the company knowledge provided: this company's documented
procedures are correct for this company even when they differ from general practice."""


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def run(args) -> dict:
    scenarios = json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"].startswith(tuple(args.only.split(",")))]
    cfg, llm = settings(), get_llm()
    print(f"Mode: {cfg.mode_label}")
    try:
        llm.warm_up()  # local models load into GPU memory on first use; keep that out of the latency figures
        kb = get_kb()  # embeds the whole KB on first run: the first model call, so the first to fail
    except Exception as e:
        print(f"\n❌ Could not build the knowledge-base index: {e!r}\n{_hint(repr(e))}")
        sys.exit(1)
    mode = cfg.provider  # offline | local | gemini: each provider gets its own results folder
    out_dir = ROOT / "eval" / "results" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(scenarios)} scenarios · embeddings={llm.embedding_model}")

    rows: list[dict] = []
    stopped_early = ""
    try:
        for s in scenarios:
            row = run_scenario(s, args, cfg, kb, llm)
            mark = "✅" if row["route_ok"] and row["category_ok"] else "❌"
            print(f"  {mark} {s['id']:<4} {row['path']:<32} {row['latency_ms']:>6} ms  {s['message'][:58]}")
            if row["error"]:
                print(f"       ⚠️ {row['error'][:160]}")
            rows.append(row)
            # Daily free-tier quota: every later call would fail too, so stop instead of burning retries.
            recent = [r["error"] for r in rows[-3:]]
            if len(recent) == 3 and all(_is_quota(e) for e in recent):
                stopped_early = f"stopped after {len(rows)} scenarios: API quota exhausted"
                print(f"\n⛔ {stopped_early}.\n{_hint(recent[-1])}")
                break
    except KeyboardInterrupt:
        stopped_early = f"interrupted after {len(rows)} scenarios"
    if not rows:
        print("No scenarios completed; nothing to report.")
        sys.exit(1)

    summary = summarize(rows, mode, cfg, llm)
    summary["stopped_early"] = stopped_early
    write_outputs(rows, summary, out_dir)
    return summary


def _is_quota(error: str) -> bool:
    return "429" in error or "RESOURCE_EXHAUSTED" in error


def _hint(error: str) -> str:
    if _is_quota(error):
        return ("Hint: the Gemini API quota is used up (free tiers have per-minute and per-day limits). Wait and re-run, "
                "raise GEMINI_MIN_INTERVAL, run a subset with --only K,S, or drop --judge to halve the calls.")
    if "404" in error or "NOT_FOUND" in error:
        return "Hint: a model name isn't available to this key. Set GEMINI_MODEL / GEMINI_EMBEDDING_MODEL to one listed in AI Studio."
    if "403" in error or "PERMISSION_DENIED" in error or "API_KEY_INVALID" in error or "400" in error:
        return "Hint: check that the GEMINI_API_KEY secret is correct and that the Generative Language API is enabled for it."
    return "Hint: see the error above; re-run with --only K01 to reproduce it quickly."


def run_scenario(s: dict, args, cfg, kb, llm) -> dict:
    reset_runtime()  # every scenario starts from the same directory/ticket state
    t0 = time.perf_counter()
    error = ""
    try:
        state = handle_request(s["message"], s["requester"])
    except Exception as e:  # keep going; a crash counts as a failure for that scenario
        state, error = {"intake": {}, "path": [], "answer": "", "latency_ms": round((time.perf_counter() - t0) * 1000)}, repr(e)[:300]
    intake = state.get("intake") or {}
    final = FINAL.get(state["path"][-1], "") if state["path"] else ""
    hits = []
    if s.get("expected_doc"):
        try:
            hits = kb.search(s["message"], k=cfg.retrieval_top_k)
        except Exception as e:
            error = error or repr(e)[:300]
    ranked_docs = [h.chunk.doc_id for h in hits]
    row = {
        "id": s["id"], "requester": s["requester"], "message": s["message"],
        "expected_category": s["expected_category"] or "*", "category": intake.get("category", ""),
        "category_ok": s["expected_category"] is None or intake.get("category") == s["expected_category"],
        "expected_action": s["expected_action"], "action": intake.get("action", ""),
        "action_ok": intake.get("action") == s["expected_action"],
        "expected_final": s["expected_final"], "final": final, "route_ok": final == s["expected_final"],
        "path": " > ".join(state["path"]),
        "expected_doc": s.get("expected_doc") or "",
        "hit_at_1": bool(ranked_docs) and ranked_docs[0] == s.get("expected_doc"),
        "hit_at_k": s.get("expected_doc") in ranked_docs if s.get("expected_doc") else "",
        "top_score": hits[0].score if hits else "",
        "latency_ms": state["latency_ms"], "answer": state.get("answer", ""), "error": error,
    }
    if args.judge and not llm.offline and not error:
        try:
            expected = f"expected final agent: {s['expected_final']}, expected action: {s['expected_action']}"
            # Give the judge the KB sections the assistant saw, so it grades against company policy.
            context = "\n\n".join(kb.chunk(r["id"]).text for r in (state.get("retrieved") or [])[:2])
            j = llm.structured(f"Employee ({s['requester']}): {s['message']}\nExpected behaviour: {expected}\n"
                               + (f"Company knowledge available to the assistant:\n{context}\n" if context else "")
                               + f"Assistant reply:\n{row['answer']}", JudgeScore, JUDGE_SYSTEM)
            row.update(sat_helpfulness=j.helpfulness, sat_accuracy=j.accuracy, sat_tone=j.tone,
                       satisfaction=round((j.helpfulness + j.accuracy + j.tone) / 3, 2), judge_note=j.rationale)
        except Exception as e:
            row["judge_note"] = f"judge failed: {e!r}"[:120]
    return row


def summarize(rows: list[dict], mode: str, cfg, llm) -> dict:
    n = len(rows)
    lat = [r["latency_ms"] for r in rows]
    retrieval = [r for r in rows if r["expected_doc"]]
    should = [r for r in rows if r["expected_final"] == "escalation"]
    did = [r for r in rows if r["final"] == "escalation"]
    tp = sum(1 for r in did if r["expected_final"] == "escalation")
    sat = [r["satisfaction"] for r in rows if "satisfaction" in r]
    pct = lambda k: round(100 * sum(1 for r in rows if r[k]) / n, 1)  # noqa: E731
    return {
        "mode": cfg.mode_label, "embedding_model": llm.embedding_model, "scenarios": n,
        "category_accuracy_pct": pct("category_ok"), "action_accuracy_pct": pct("action_ok"),
        "routing_accuracy_pct": pct("route_ok"),
        "end_to_end_pass_pct": round(100 * sum(1 for r in rows if r["route_ok"] and r["category_ok"] and r["action_ok"]) / n, 1),
        "retrieval_hit_at_1_pct": round(100 * sum(1 for r in retrieval if r["hit_at_1"]) / len(retrieval), 1) if retrieval else None,
        "retrieval_hit_at_k_pct": round(100 * sum(1 for r in retrieval if r["hit_at_k"]) / len(retrieval), 1) if retrieval else None,
        "retrieval_k": cfg.retrieval_top_k,
        "escalation_precision_pct": round(100 * tp / len(did), 1) if did else None,
        "escalation_recall_pct": round(100 * tp / len(should), 1) if should else None,
        "containment_rate_pct": round(100 * sum(1 for r in rows if r["final"] in ("knowledge", "workflow")) / n, 1),
        "latency_p50_ms": percentile(lat, 0.5), "latency_p95_ms": percentile(lat, 0.95), "latency_max_ms": max(lat),
        "satisfaction_avg_1to5": round(statistics.mean(sat), 2) if sat else None,
        "llm_calls": llm.stats.llm_calls, "llm_retries": llm.stats.retries, "errors": sum(1 for r in rows if r["error"]),
        "min_relevance": cfg.min_relevance,
    }


def write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = {
        "category_accuracy_pct": "Intake category accuracy", "action_accuracy_pct": "Action selection accuracy",
        "routing_accuracy_pct": "Routing accuracy (final agent)", "end_to_end_pass_pct": "End-to-end pass rate",
        "retrieval_hit_at_1_pct": "Retrieval hit@1", "retrieval_hit_at_k_pct": f"Retrieval hit@{summary['retrieval_k']}",
        "escalation_precision_pct": "Escalation precision", "escalation_recall_pct": "Escalation recall",
        "containment_rate_pct": "Containment (no human needed)",
    }
    md = [f"# Evaluation report - {summary['mode']}", "",
          f"{summary['scenarios']} scenarios · embeddings `{summary['embedding_model']}` · "
          f"relevance threshold {summary['min_relevance']} · {summary['llm_calls']} LLM calls · {summary['errors']} errors", ""]
    if summary.get("stopped_early"):
        md += [f"> ⚠️ Partial run: {summary['stopped_early']}. Metrics cover completed scenarios only.", ""]
    md += ["| Metric | Value |", "|---|---|"]
    md += [f"| {label} | {summary[k]}% |" for k, label in labels.items() if summary[k] is not None]
    md += [f"| Latency p50 / p95 / max | {summary['latency_p50_ms']} / {summary['latency_p95_ms']} / {summary['latency_max_ms']} ms |"]
    if summary["satisfaction_avg_1to5"] is not None:
        md.append(f"| Satisfaction (LLM judge, 1-5) | {summary['satisfaction_avg_1to5']} |")
    md += ["", "## Failures", ""]
    fails = [r for r in rows if not (r["route_ok"] and r["category_ok"] and r["action_ok"])]
    if not fails:
        md.append("None.")
    else:
        md += ["| ID | Message | Expected | Got |", "|---|---|---|---|"]
        md += [f"| {r['id']} | {r['message']} | {r['expected_category']} / {r['expected_action']} / {r['expected_final']} | "
               f"{r['category']} / {r['action']} / {r['final']} |" for r in fails]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _charts(rows, summary, labels, out_dir / "charts.png")
    print(f"\nWrote {out_dir.relative_to(ROOT)}/ (results.csv, summary.json, report.md, charts.png)")


def _charts(rows, summary, labels, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    keys = [k for k in labels if summary[k] is not None]
    vals = [summary[k] for k in keys]
    ax = axes[0]
    bars = ax.barh([labels[k] for k in keys][::-1], vals[::-1], color="#2a6fdb")
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_xlim(0, 110)
    ax.set_title("Quality metrics")

    ax = axes[1]
    ax.hist([r["latency_ms"] for r in rows], bins=12, color="#1f9d74")
    ax.axvline(summary["latency_p95_ms"], color="#c0392b", ls="--", label=f"p95 {summary['latency_p95_ms']:.0f} ms")
    ax.set_title("End-to-end latency")
    ax.set_xlabel("ms")
    ax.legend()

    ax = axes[2]
    order = ["knowledge", "workflow", "escalation"]
    grid = [[sum(1 for r in rows if r["expected_final"] == e and r["final"] == g) for g in order] for e in order]
    ax.imshow(grid, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, grid[i][j], ha="center", va="center", color="white" if grid[i][j] > 6 else "black", fontsize=13)
    ax.set_xticks(range(3), order)
    ax.set_yticks(range(3), order)
    ax.set_xlabel("actual final agent")
    ax.set_ylabel("expected")
    ax.set_title("Routing confusion matrix")
    fig.suptitle(f"Bayview IT Assistant evaluation · {summary['mode']} · {summary['scenarios']} scenarios")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge satisfaction scoring (gemini or local provider)")
    ap.add_argument("--pace", type=float, default=None, help="Seconds between Gemini calls")
    ap.add_argument("--only", default="", help="Comma-separated scenario id prefixes, e.g. K,S0")
    a = ap.parse_args()
    if a.pace is not None:
        os.environ["GEMINI_MIN_INTERVAL"] = str(a.pace)
        reset_settings()
    s = run(a)
    print(json.dumps(s, indent=2))
