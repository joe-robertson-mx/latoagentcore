"""Simple deploy script for the Return Request agent.

This script is intentionally minimal: it forwards a small set of options to
`deploy_and_launch()` and is designed for use in quick CI or local workflows.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from .deploy_agent import deploy_agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def deploy_return_request(
    agent_name: str = "return_request",
    entrypoint: str = "src/return_request_agent.py",
    region: Optional[str] = None,
):
    """Minimal wrapper that delegates to `deploy_agent()` (configure-only)."""
    return deploy_agent(
        agent_name=agent_name,
        entrypoint=entrypoint,
        region=region,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure the Return Request agent (minimal)")
    parser.add_argument("--agent-name", default="return_request", help="Agent name to register")
    parser.add_argument("--entrypoint", default="src/return_request_agent.py", help="Agent entrypoint file (relative to latoagentcore)")
    parser.add_argument("--region", default=None, help="AWS region to use")

    args = parser.parse_args()

    deploy_return_request(
        agent_name=args.agent_name,
        entrypoint=args.entrypoint,
        region=args.region,
    )


if __name__ == "__main__":
    main()
