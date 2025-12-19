#!/usr/bin/env python3
"""Simple launcher to run the Return Request deploy from the project root.

Usage:
  python deploy_return_request.py --agent-name return_request

This script adds `latoagentcore/src` to sys.path so you do not need to set
PYTHONPATH or install the package. It calls the existing CLI entrypoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package source dir is importable when running from repo root
project_root = Path(__file__).resolve().parent
src_path = project_root / "latoagentcore" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import and run the module's CLI
from deploy.return_request import main

if __name__ == "__main__":
    main()
