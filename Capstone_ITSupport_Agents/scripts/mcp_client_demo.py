"""Connect to the helpdesk MCP server over stdio, exactly as VS Code does, and call its tools.

    python scripts/mcp_client_demo.py

Demonstrates the MCP value proposition: the client knows nothing about the helpdesk in advance.
It discovers tool names, descriptions and JSON schemas at runtime, then calls them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


def _show(result) -> str:
    if getattr(result, "structured_content", None) is not None:
        payload = result.structured_content
    else:
        payload = "\n".join(getattr(c, "text", str(c)) for c in result.content)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return text if len(text) < 1400 else text[:1400] + "\n…"


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")], cwd=str(ROOT))
    async with Client(params) as client:
        info = client.server_info
        print(f"Connected to MCP server: {info.name if info else 'bayview-helpdesk'}\n")
        tools = (await client.list_tools()).tools
        print(f"Discovered {len(tools)} tools:")
        for t in tools:
            params_ = ", ".join(t.input_schema.get("properties", {}).keys())
            print(f"  • {t.name}({params_}) - {(t.description or '').splitlines()[0]}")

        calls = [
            ("check_service_status", {"service": "vpn"}),
            ("analyze_logs", {"log": "auth", "username": "asmith"}),
            ("ask_it_assistant", {"message": "I clicked a link in a weird invoice email", "requester": "jdoe"}),
            ("list_tickets", {"status": "open"}),
        ]
        for name, args in calls:
            print(f"\n▶ call_tool {name} {json.dumps(args)}")
            print(_show(await client.call_tool(name, args)))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
