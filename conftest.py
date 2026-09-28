"""Root pytest configuration.

Guarantees the project root is on ``sys.path`` so ``import src`` works whether
the suite is launched via ``pytest`` or ``python -m pytest``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
