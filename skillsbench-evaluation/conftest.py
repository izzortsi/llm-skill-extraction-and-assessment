"""pytest configuration: register dotted directory imports."""

import sys
from pathlib import Path

_project = Path(__file__).resolve().parent
_repo = _project.parent

if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import _bootstrap

_bootstrap.setup_project(_project)
_bootstrap.setup_providers()
_bootstrap.setup_shared_data()
