#!/usr/bin/env python
"""Check that this machine can run the reproduction: GPU memory, package versions, Hugging Face
access and disk space. Exits with status 1 and lists the problems if anything is missing.

    python reproduce/local/check_env.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hastika.reproduce import check_environment  # noqa: E402

if __name__ == "__main__":
    sys.exit(1 if check_environment() else 0)
