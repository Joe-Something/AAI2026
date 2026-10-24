import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HELPDESK_OFFLINE"] = "1"  # unit tests never call the Gemini API


@pytest.fixture(autouse=True)
def clean_state():
    from helpdesk.config import reset_settings
    from helpdesk.store import reset_runtime

    reset_settings()
    reset_runtime()
    yield
