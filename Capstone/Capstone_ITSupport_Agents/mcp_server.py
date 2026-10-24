"""MCP server exposing the helpdesk to any MCP client (VS Code Copilot agent mode, Claude, etc.).

    python mcp_server.py          # stdio transport (what .vscode/mcp.json launches)

Why MCP instead of a REST API: the same tool definitions (name, typed parameters, docstring)
are discovered automatically by every MCP client - no per-client SDK, OpenAPI glue or auth
wrapper to write. A technician in VS Code can ask Copilot "show open P1 tickets and analyze
the auth log for ceo" and the client calls these tools directly; combined with the GitHub MCP
server in the same workspace, escalated issues can be triaged next to the code that fixes them.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from helpdesk import store, tools
from helpdesk.graph import handle_request
from helpdesk.knowledge_base import get_kb

server = MCPServer(
    name="bayview-helpdesk",
    instructions="Bayview Logistics IT helpdesk: search the knowledge base, run account automations, "
                 "check service status, analyze logs and manage tickets.",
)


@server.tool()
def ask_it_assistant(message: str, requester: str) -> dict:
    """Send a message through the full multi-agent pipeline as `requester` (a directory username)."""
    s = handle_request(message, requester)
    return {"answer": s["answer"], "path": s["path"], "sources": s.get("sources", []),
            "ticket": s.get("ticket", {}).get("number"), "latency_ms": s["latency_ms"]}


@server.tool()
def search_knowledge_base(query: str, k: int = 4) -> list[dict]:
    """Semantic search over IT knowledge-base articles. Returns chunks with similarity scores."""
    return [{"citation": h.citation, "score": h.score, "text": h.chunk.text} for h in get_kb().search(query, k)]


@server.tool()
def check_service_status(service: str = "") -> dict:
    """Live status for a service (vpn, email, sso, wifi, printing, hr_portal, company_portal); empty = all."""
    return tools.check_service_status(service or None).to_dict()


@server.tool()
def analyze_logs(log: str, username: str = "") -> dict:
    """Analyze the 'vpn' or 'auth' log, optionally filtered to one username; returns findings."""
    return tools.analyze_logs(log, username or None).to_dict()


@server.tool()
def list_tickets(status: str = "", queue: str = "") -> list[dict]:
    """List recent tickets, optionally filtered by status (open/resolved) or queue name."""
    return [{k: t[k] for k in ("number", "status", "priority", "queue", "requester", "title", "created_at")}
            for t in store.list_tickets(status or None, queue or None)]


@server.tool()
def get_ticket(number: str) -> dict:
    """Full ticket detail including summary, steps tried and transcript. number like INC000004."""
    t = store.get_ticket(int(number.upper().removeprefix("INC")))
    return t or {"error": f"{number} not found"}


@server.tool()
def resolve_ticket(number: str, technician: str) -> dict:
    """Mark a ticket resolved by a technician."""
    t = store.update_ticket_status(int(number.upper().removeprefix("INC")), "resolved", technician)
    return t or {"error": f"{number} not found"}


if __name__ == "__main__":
    server.run("stdio")
