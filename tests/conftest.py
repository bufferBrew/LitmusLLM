"""Put the project root on sys.path so tests import the app modules directly.

LitmusLLM is a flat module layout (no package), which is what `main.py` and
every module already assume when they `import scoring` / `import database`.
Tests join that same namespace rather than restructuring the app around them.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
