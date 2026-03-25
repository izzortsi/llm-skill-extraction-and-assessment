"""pytest configuration: register dotted directory imports and load .env."""

import sys
from pathlib import Path

_project = Path(__file__).resolve().parent
_repo = _project.parent

if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import _bootstrap

_bootstrap.setup_project(_project)

# load .env file if present (for API keys like ZHIPU_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(_project / ".env")
except ImportError:
    pass
