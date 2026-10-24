"""Capture screenshots of successful runs into docs/screenshots/ (for the report and slides).

    python scripts/capture_screenshots.py            # everything (runs the evaluation too)
    python scripts/capture_screenshots.py --ui-only  # just the Gradio screenshots

1. Runs the CLI demo, unit tests, evaluation and MCP client demo, and renders each console
   output as a terminal-style image.
2. Starts the Gradio app and drives it with Playwright (using the installed Microsoft Edge),
   capturing the UI after each scenario.
Each image's footer records the mode (Gemini or offline), so screenshots are never ambiguous.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

TERMINAL_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#0d1117;font-family:'Cascadia Mono',Consolas,monospace}}
.win{{margin:18px;border-radius:10px;overflow:hidden;box-shadow:0 8px 30px #0008;border:1px solid #30363d}}
.bar{{background:#161b22;color:#8b949e;padding:9px 14px;font:13px 'Segoe UI',sans-serif;display:flex;gap:8px;align-items:center}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
pre{{margin:0;padding:16px 18px;color:#e6edf3;font-size:13.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word;background:#0d1117}}
.cmd{{color:#7ee787}} .ok{{color:#3fb950}} .bad{{color:#f85149}} .dim{{color:#8b949e}}
</style></head><body><div class="win"><div class="bar"><span class="dot" style="background:#ff5f56"></span>
<span class="dot" style="background:#ffbd2e"></span><span class="dot" style="background:#27c93f"></span>
<span style="margin-left:8px">{title}</span></div><pre><span class="cmd">PS F:\\Capstone_ITSupport_Agents&gt; {cmd}</span>
{body}</pre></div></body></html>"""


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", timeout=1800)
    return (res.stdout + ("\n" + res.stderr if res.returncode else "")).rstrip()


def colorize(text: str) -> str:
    out = []
    for line in html.escape(text).splitlines():
        if "✅" in line or " passed" in line or "PASSED" in line:
            line = f'<span class="ok">{line}</span>'
        elif "❌" in line or "FAILED" in line or "Error" in line:
            line = f'<span class="bad">{line}</span>'
        elif line.startswith("──"):
            line = f'<span class="dim">{line}</span>'
        out.append(line)
    return "\n".join(out)


def terminal_shot(page, name: str, title: str, cmd_display: str, cmd: list[str], width: int = 1180) -> str:
    text = run(cmd)
    page.set_viewport_size({"width": width, "height": 400})
    page.set_content(TERMINAL_HTML.format(title=html.escape(title), cmd=html.escape(cmd_display), body=colorize(text)))
    page.screenshot(path=str(OUT / name), full_page=True)
    print(f"  📸 {name}")
    return text


def wait_for_server(url: str, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(url)


def gradio_shots(browser) -> None:
    subprocess.run([PY, "-c", "from helpdesk.store import reset_runtime; reset_runtime()"], cwd=ROOT, env=ENV, check=True)
    server = subprocess.Popen([PY, "app.py", "--port", "7861"], cwd=ROOT, env=ENV,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server("http://127.0.0.1:7861")
        page = browser.new_page(viewport={"width": 1480, "height": 1000}, device_scale_factor=1)
        page.goto("http://127.0.0.1:7861")
        page.wait_for_selector("#msgbox textarea")
        page.screenshot(path=str(OUT / "10_ui_home.png"), full_page=True)
        print("  📸 10_ui_home.png")
        scenarios = [
            ("11_ui_knowledge.png", "jdoe: How do I set up the authenticator"),
            ("12_ui_workflow_unlock.png", "asmith: I'm locked out of my account"),
            ("13_ui_vpn_status.png", "pchen: Is the VPN down?"),
            ("14_ui_log_analysis.png", "asmith: Why does my account keep lockin"),
            ("15_ui_security_escalation.png", "ceo: I got an MFA prompt I didn't start"),
            ("16_ui_refused_escalation.png", "jdoe: Please reset the password for as"),
        ]
        for name, button_prefix in scenarios:
            before = page.inner_text("#trace")
            # The example button fills the box via a server round trip: wait for the text before sending.
            page.fill("#msgbox textarea", "")
            page.get_by_role("button", name=button_prefix).first.click()
            page.wait_for_function("() => document.querySelector('#msgbox textarea').value.length > 0", timeout=30_000)
            page.wait_for_timeout(600)  # a user switch also clears the chat; let that land first
            sent = page.input_value("#msgbox textarea")[:30]
            page.click("#send")
            # Done = the chat shows this message and no "processing" overlay remains anywhere on the page
            # (Gradio draws the overlay over the previous result, so the old trace text is still there).
            try:
                page.wait_for_function("m => !document.body.innerText.includes('processing |')"
                                       " && document.querySelector('#chat').innerText.includes(m)"
                                       " && document.querySelector('#trace').innerText.includes('Path:')",
                                       arg=sent, timeout=180_000)
            except Exception:
                page.screenshot(path=str(OUT / f"_debug_{name}"), full_page=True)
                print("  trace text:", repr(page.inner_text("#trace"))[:300])
                raise
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / name), full_page=True)
            print(f"  📸 {name}")
    finally:
        server.terminate()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    ui_only = "--ui-only" in sys.argv
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        if ui_only:
            print("Capturing Gradio UI…")
            gradio_shots(browser)
            browser.close()
            return
        page = browser.new_page(device_scale_factor=1)
        print("Capturing console runs…")
        terminal_shot(page, "01_cli_demo.png", "Scripted demo - every agent path", "python -m helpdesk.cli", [PY, "-m", "helpdesk.cli"])
        terminal_shot(page, "02_unit_tests.png", "Unit & integration tests", "python -m pytest -v tests", [PY, "-m", "pytest", "-v", "tests"])
        terminal_shot(page, "03_evaluation.png", "Evaluation harness - 37 scenarios", "python eval/run_eval.py --judge",
                      [PY, "eval/run_eval.py", "--judge"])
        terminal_shot(page, "04_mcp_client.png", "MCP client discovering and calling helpdesk tools", "python scripts/mcp_client_demo.py",
                      [PY, "scripts/mcp_client_demo.py"])
        sys.path.insert(0, str(ROOT))
        from helpdesk.config import settings

        shutil.copy(ROOT / f"eval/results/{settings().provider}/charts.png", OUT / "05_eval_charts.png")
        print("  📸 05_eval_charts.png")
        print("Capturing Gradio UI…")
        gradio_shots(browser)
        browser.close()
    print(f"Done → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
