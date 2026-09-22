#!/usr/bin/env python3
"""Build the shared geometry cache used by the support filter.

The implementation lives in eval/support_filter.py, which the main aggregation
step uses as well; this wrapper keeps the documented figure-building command
working. Equivalent to: python -B eval/support_filter.py --build
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from support_filter import build_cache  # noqa: E402

if __name__ == "__main__":
    build_cache()
    sys.exit(0)
