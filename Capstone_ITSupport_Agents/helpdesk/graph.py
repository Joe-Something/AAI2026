"""LangGraph orchestration.

    START -> intake --(route_from_intake)--> knowledge  --(answered?)--> END
                                       \\--> workflow   --(ok?)-------> END
                                        \\-> escalation <-(otherwise)--/
                                                 |
                                                END
"""

from __future__ import annotations

import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .agents import (HelpdeskState, escalation_agent, intake_agent, knowledge_agent, route_from_intake,
                     route_to_escalation_if_needed, workflow_agent)
from .config import settings


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(HelpdeskState)
    g.add_node("intake", intake_agent)
    g.add_node("knowledge", knowledge_agent)
    g.add_node("workflow", workflow_agent)
    g.add_node("escalation", escalation_agent)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_from_intake, ["knowledge", "workflow", "escalation"])
    for node in ("knowledge", "workflow"):
        g.add_conditional_edges(node, route_to_escalation_if_needed, {"escalation": "escalation", "end": END})
    g.add_edge("escalation", END)
    return g.compile()


def handle_request(message: str, requester: str, history: list[dict] | None = None) -> dict:
    """Run one user message through the agent graph. Returns the final state plus timing."""
    t0 = time.perf_counter()
    state = build_graph().invoke({"message": message, "requester": requester, "history": history or [], "trace": []})
    state["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    state["path"] = [t["agent"] for t in state["trace"]]
    state["mode"] = settings().mode_label
    return state


def mermaid() -> str:
    """Mermaid source of the compiled graph (used in docs and the notebook)."""
    return build_graph().get_graph().draw_mermaid()
