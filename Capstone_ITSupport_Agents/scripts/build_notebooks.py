"""Generate the Google Colab notebooks in notebooks/ from the project source.

    python scripts/build_notebooks.py

Each notebook is self-contained: it writes the project files with %%writefile (code stays
readable in the notebook) so it runs in Colab without cloning a repository. Re-run this script
after changing any source file so the notebooks stay in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
COLAB_DIR = "/content/Capstone_ITSupport_Agents"

CODE_FILES = [
    "helpdesk/__init__.py", "helpdesk/config.py", "helpdesk/privacy.py", "helpdesk/schemas.py",
    "helpdesk/llm.py", "helpdesk/offline.py", "helpdesk/knowledge_base.py", "helpdesk/store.py",
    "helpdesk/tools.py", "helpdesk/agents.py", "helpdesk/graph.py", "helpdesk/cli.py",
    "app.py", "mcp_server.py", "scripts/mcp_client_demo.py", "eval/run_eval.py",
]
DATA_GLOBS = ["data/kb/*.md", "data/*.json", "data/logs/*.log", "eval/scenarios.json"]

PIP = "!pip install -q langgraph google-genai chromadb gradio \"mcp[cli]\" pydantic python-dotenv numpy matplotlib"

KEY_MD = """## 🔑 Gemini API key
This notebook reads your key from a **Colab secret named `GEMINI_API_KEY`**:
1. Click the 🔑 **Secrets** icon in the left sidebar.
2. Add a secret named `GEMINI_API_KEY` with your key from https://aistudio.google.com/apikey
3. Turn on **Notebook access** for it.

The key is never printed. Without it, the system runs in clearly-labelled offline mode (rule-based baseline)."""

KEY_CODE = """import os
from google.colab import userdata
try:
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
    print("✅ GEMINI_API_KEY loaded from Colab secrets")
except Exception as e:
    print("⚠️ GEMINI_API_KEY secret not available - running in offline mode.", type(e).__name__)

# Optional tuning (see helpdesk/config.py):
os.environ.setdefault("GEMINI_MODEL", "gemini-flash-latest")      # current stable Flash
os.environ.setdefault("GEMINI_MIN_INTERVAL", "4")                  # seconds between calls; keeps free tier under RPM limits
# os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")        # optional: mirror escalations to GitHub Issues
# os.environ["GITHUB_REPO"] = "your-user/your-repo\""""


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def setup_cells() -> list:
    cells = [
        md("## 1 · Install dependencies"), code(PIP),
        md(KEY_MD), code(KEY_CODE),
        md("## 2 · Project files\nThe cells below write the project source into `" + COLAB_DIR + "`. "
           "Each module has a docstring explaining its role."),
        code(f"%mkdir -p {COLAB_DIR}/helpdesk {COLAB_DIR}/scripts {COLAB_DIR}/eval {COLAB_DIR}/data/kb {COLAB_DIR}/data/logs\n%cd {COLAB_DIR}"),
    ]
    data = {}
    for pattern in DATA_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            data[p.relative_to(ROOT).as_posix()] = p.read_text(encoding="utf-8")
    cells.append(md(f"### Data: {sum(1 for k in data if k.startswith('data/kb'))} knowledge-base articles, "
                    "mock user directory, service status, system logs and evaluation scenarios"))
    blob = json.dumps(data, indent=1, ensure_ascii=False)
    assert "'''" not in blob, "seed data must not contain ''' (it is embedded in a raw triple-quoted string)"
    cells.append(code("# Seed data (knowledge base, directory, services, logs, scenarios)\nimport json, pathlib\nDATA = json.loads(r'''"
                      + blob + "''')\n"
                      "for path, text in DATA.items():\n    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding='utf-8')\n"
                      "print(f'wrote {len(DATA)} data files')"))
    for f in CODE_FILES:
        cells.append(code(f"%%writefile {f}\n" + (ROOT / f).read_text(encoding="utf-8")))
    cells.append(code("import sys, importlib\nsys.path.insert(0, '.')\n"
                      "for m in [m for m in list(sys.modules) if m.startswith('helpdesk')]:\n    del sys.modules[m]  # pick up rewritten files on re-run\n"
                      "from helpdesk.config import settings\nprint('Mode:', settings().mode_label)"))
    cells.append(md("### Gemini connection check\nOne chat call and one embedding call, so key, quota or model problems show up here "
                    "instead of halfway through a run."))
    cells.append(code("from helpdesk.llm import get_llm\nllm = get_llm()\nif llm.offline:\n"
                      "    print('Offline mode: the GEMINI_API_KEY secret was not loaded (see the key cell above).')\nelse:\n"
                      "    try:\n        print('✅ chat:', llm.cfg.mode_label, '→', llm.text('Reply with the word OK.', 'Be brief.')[:20])\n"
                      "        llm.embed(['connection test'], task='query')\n        print('✅ embeddings:', llm.embedding_model)\n"
                      "    except Exception as e:\n        print('❌', repr(e)[:500])\n"
                      "        print('429 = quota used up (wait or slow down); 404 = model not available to this key; 400/403 = key problem.')"))
    return cells


def demo_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 🛟 Bayview IT Assistant - Multi-Agent IT Support (LangGraph + Gemini)\n"
           "**BUS 118s-01 Capstone.** Four cooperating agents resolve IT requests:\n\n"
           "| Agent | Role |\n|---|---|\n"
           "| 🧭 Intake | Redacts PII, classifies the request, extracts entities (Gemini structured output) |\n"
           "| 📚 Knowledge | RAG: Gemini embeddings → Chroma vector DB → grounded answer with citations |\n"
           "| ⚙️ Workflow | Automations: unlock account, password reset, service status, log analysis |\n"
           "| 🙋 Escalation | Writes a technician handoff and opens a routed ticket (optional GitHub issue) |\n\n"
           "Run all cells top to bottom (Runtime → Run all)."),
        *setup_cells(),
        md("## 3 · Knowledge base and RAG\nArticles are split by section, embedded with Gemini embeddings and stored in Chroma (cosine)."),
        code("from helpdesk.knowledge_base import get_kb\nkb = get_kb()\nprint(kb.size, '· embedding model:', kb.llm.embedding_model)\n"
             "for h in kb.search('VPN says certificate error', k=3):\n    print(f'{h.score:.3f}  {h.citation}')"),
        md("## 4 · The agent graph (LangGraph)"),
        code("from helpdesk.graph import build_graph, mermaid\nfrom IPython.display import Image, Markdown, display\n"
             "try:\n    display(Image(build_graph().get_graph().draw_mermaid_png()))\n"
             "except Exception:\n    display(Markdown('```mermaid\\n' + mermaid() + '\\n```'))"),
        md("## 5 · Scripted demo - every agent path\nKnowledge answer · workflow automation · status check · security escalation (with PII redaction) · out-of-scope escalation."),
        code("!python -m helpdesk.cli"),
        md("### Try your own message"),
        code("from helpdesk.graph import handle_request\nfrom helpdesk.cli import print_result\n"
             "msg, user = 'Why does my account keep locking? Can you check the logs?', 'asmith'\n"
             "print_result(user, msg, handle_request(msg, user))"),
        md("## 6 · MCP integration\nThe helpdesk is also an **MCP server** (`mcp_server.py`). This client discovers its tools at runtime over stdio - "
           "exactly what VS Code does via `.vscode/mcp.json`, alongside the GitHub MCP server."),
        code("!python scripts/mcp_client_demo.py"),
        md("## 7 · Web UI (Gradio)\nOpens a public `*.gradio.live` link. The right panel shows which agents ran, retrieval scores and the live ticket queue."),
        code("from app import build_ui\nbuild_ui().launch(share=True, debug=False)"),
    ]
    return nb


def eval_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# 📊 Bayview IT Assistant - Testing & Validation\n"
           "Runs 37 realistic IT support scenarios (`eval/scenarios.json`) through the agent graph and measures:\n\n"
           "- **Accuracy**: intake category, action selection, routing (which agent answered), retrieval hit@1 / hit@k\n"
           "- **Safety**: escalation precision / recall (security incidents and refused automations must reach a human)\n"
           "- **Speed**: end-to-end latency p50 / p95\n"
           "- **Satisfaction**: LLM-as-judge 1-5 rating of each reply (helpfulness, accuracy, tone)\n\n"
           "⏱ With the free tier and `GEMINI_MIN_INTERVAL=4` this takes about 10 minutes."),
        *setup_cells(),
        md("## 3 · Run the evaluation (Gemini + LLM judge)"),
        code("!python eval/run_eval.py --judge"),
        md("## 4 · Results"),
        code("import os, pandas as pd\nfrom IPython.display import Image, Markdown, display\nmode = settings().provider\n"
             "out = f'eval/results/{mode}'\n"
             "if not os.path.exists(f'{out}/report.md'):\n"
             "    print(f'No {mode} results yet: the evaluation in step 3 stopped before writing them.\\n'\n"
             "          'Scroll up to its output for the error and hint (most often a 429 free-tier quota error).')\n"
             "else:\n"
             "    display(Markdown(open(f'{out}/report.md', encoding='utf-8').read()))\n    display(Image(f'{out}/charts.png'))\n"
             "    df = pd.read_csv(f'{out}/results.csv')\n"
             "    cols = [c for c in ['id','message','category','action','path','route_ok','top_score','latency_ms','satisfaction','error'] if c in df]\n"
             "    display(df[cols])"),
        md("## 5 · Baseline comparison: rule-based offline mode\nThe same scenarios with no LLM (keyword classifier + hashed embeddings) show what Gemini adds."),
        code("!HELPDESK_OFFLINE=1 python eval/run_eval.py\n"
             "import json\nfor m in ['offline', 'local', 'gemini']:\n    try:\n        s = json.load(open(f'eval/results/{m}/summary.json'))\n"
             "        print(f\"{m:8} routing {s['routing_accuracy_pct']}%  category {s['category_accuracy_pct']}%  e2e {s['end_to_end_pass_pct']}%  \"\n"
             "              f\"esc-recall {s['escalation_recall_pct']}%  p95 {s['latency_p95_ms']} ms\")\n    except FileNotFoundError:\n        pass"),
        md("## 6 · Download results\nZip the results folder to include in the report / slides."),
        code("!cd eval && zip -qr results.zip results && ls -la results.zip\nfrom google.colab import files\nfiles.download('eval/results.zip')"),
    ]
    return nb


def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = {"colab": {"provenance": [], "toc_visible": True}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}
    for name, nb in [("01_IT_Assistant_Demo.ipynb", demo_notebook()), ("02_Evaluation.ipynb", eval_notebook())]:
        nb.metadata.update(meta)
        nbf.validate(nb)
        nbf.write(nb, OUT / name)
        print(f"wrote notebooks/{name} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
