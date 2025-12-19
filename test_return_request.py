#!/usr/bin/env python3
"""Simple script to test the deployed Return Request agent.

Usage:
  python test_return_request.py
  python test_return_request.py --prompt "I want to return order 12345"
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package source dir is importable
project_root = Path(__file__).resolve().parent
src_path = project_root / "latoagentcore" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import argparse
import logging

from bedrock_agentcore_starter_toolkit import Runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent(prompt: str) -> str:
    """Invoke the deployed agent with the given prompt."""
    logger.info("Invoking agent with prompt: %s", prompt)
    
    runtime = Runtime()
    response = runtime.invoke(prompt)
    
    logger.info("Agent response: %s", response)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the deployed Return Request agent")
    parser.add_argument(
        "--prompt",
        default="frederich.torresi@optimumsystems.com, My Percision Gear Set with Serial No: 8dc04f7a-ec70-4c79-9eba-759396546948 has jammed, I think some of the cables have frayed..",
        help="Prompt to send to the agent",
    )
    args = parser.parse_args()

    response = test_agent(args.prompt)
    print("\n" + "=" * 60)
    print("AGENT RESPONSE:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
