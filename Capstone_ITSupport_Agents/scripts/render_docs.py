"""Render docs assets with the installed Microsoft Edge (via Playwright).

    python scripts/render_docs.py

- docs/architecture.svg  -> docs/architecture.png
- docs/DESIGN.md         -> docs/Design_Implementation_Testing_Scaling.pdf
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

PDF_CSS = """
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#1e293b;font-size:10.5pt;line-height:1.5;margin:0}
h1{font-size:22pt;color:#0f172a;margin:0 0 4pt;border-bottom:3px solid #2563eb;padding-bottom:6pt}
h2{font-size:15pt;color:#1d4ed8;margin-top:20pt;border-bottom:1px solid #cbd5e1;padding-bottom:3pt;page-break-after:avoid}
h3{font-size:12pt;color:#0f172a;margin-top:14pt;page-break-after:avoid}
table{border-collapse:collapse;width:100%;margin:8pt 0;font-size:9.5pt;page-break-inside:avoid}
th,td{border:1px solid #cbd5e1;padding:4pt 6pt;text-align:left;vertical-align:top}
th{background:#eff6ff}
code{background:#f1f5f9;padding:1px 4px;border-radius:3px;font-size:9pt}
pre{background:#f8fafc;border:1px solid #e2e8f0;padding:8pt;border-radius:6px;font-size:8.5pt;white-space:pre-wrap;page-break-inside:avoid}
pre code{background:none;padding:0}
img{max-width:100%;page-break-inside:avoid}
blockquote{border-left:4px solid #93c5fd;margin:8pt 0;padding:2pt 10pt;color:#334155;background:#f8fafc}
"""


def render_architecture(page) -> None:
    svg = (DOCS / "architecture.svg").read_text(encoding="utf-8")
    page.set_viewport_size({"width": 1400, "height": 790})
    page.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
    page.locator("svg").screenshot(path=str(DOCS / "architecture.png"))
    print("wrote docs/architecture.png")


def render_pdf(page) -> None:
    text = (DOCS / "DESIGN.md").read_text(encoding="utf-8")
    # Mermaid does not render in a PDF: swap each diagram for its pre-rendered PNG when one exists.
    def mermaid_to_img(m: re.Match) -> str:
        name = m.group(1)
        png = DOCS / "diagrams" / f"{name}.png"
        return f"![{name}](diagrams/{name}.png)" if png.exists() else m.group(0)
    text = re.sub(r"<!-- mermaid:(\S+) -->\n```mermaid\n.*?```", mermaid_to_img, text, flags=re.S)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{PDF_CSS}</style></head><body>{body}</body></html>"
    # Load from a file next to the images: pages built with set_content() can't read file:// images.
    tmp = DOCS / "_design_render.html"
    tmp.write_text(html, encoding="utf-8")
    page.goto(tmp.as_uri(), wait_until="networkidle")
    tmp.unlink()
    out = DOCS / "Design_Implementation_Testing_Scaling.pdf"
    page.pdf(path=str(out), format="Letter", margin={"top": "0.7in", "bottom": "0.7in", "left": "0.75in", "right": "0.75in"},
             print_background=True, display_header_footer=True, header_template="<span></span>",
             footer_template="<div style='font-size:8pt;color:#64748b;width:100%;text-align:center'>"
                             "Bayview IT Assistant · Design, Implementation, Testing &amp; Scaling · "
                             "<span class='pageNumber'></span>/<span class='totalPages'></span></div>")
    print(f"wrote {out.relative_to(ROOT)}")


def render_mermaid(page) -> None:
    """Render ```mermaid blocks tagged <!-- mermaid:name --> to docs/diagrams/name.png using mermaid.js from a CDN."""
    text = (DOCS / "DESIGN.md").read_text(encoding="utf-8")
    blocks = re.findall(r"<!-- mermaid:(\S+) -->\n```mermaid\n(.*?)```", text, flags=re.S)
    if not blocks:
        return
    (DOCS / "diagrams").mkdir(exist_ok=True)
    page.set_viewport_size({"width": 1100, "height": 800})
    for name, src in blocks:
        page.set_content("<html><body style='margin:0;background:white'><pre class='mermaid'>" + src.replace("<", "&lt;") + "</pre>"
                         "<script src='https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js'></script>"
                         "<script>mermaid.initialize({startOnLoad:true,theme:'default',sequence:{useMaxWidth:false}});</script></body></html>",
                         wait_until="networkidle")
        page.wait_for_selector("pre.mermaid svg", timeout=20000)
        page.locator("pre.mermaid svg").screenshot(path=str(DOCS / "diagrams" / f"{name}.png"))
        print(f"wrote docs/diagrams/{name}.png")


if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(device_scale_factor=2)
        render_architecture(page)
        if (DOCS / "DESIGN.md").exists():
            render_mermaid(page)
            render_pdf(page)
        browser.close()
