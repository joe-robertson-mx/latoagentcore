"""Deploy helper for agents using bedrock_agentcore_starter_toolkit.Runtime.

This is a minimal, configure-only helper: it calls `Runtime.configure()` and
returns the response. No launch, wait, retries, or dry-run behavior is
implemented here by design.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def deploy_agent(
    agent_name: str,
    entrypoint: str = "langgraph_bedrock.py",
    requirements_file: Optional[str] = None,
    region: Optional[str] = None,
    auto_create_execution_role: bool = True,
    auto_create_ecr: bool = True,
    runtime: Optional[Runtime] = None,
):
    """Configure an agent using the Runtime from bedrock_agentcore_starter_toolkit.

    This function is intentionally minimal and only calls `Runtime.configure()`.
    If requirements_file is None, the toolkit will auto-detect pyproject.toml or requirements.txt.
    """
    boto_session = Session()
    resolved_region = region or boto_session.region_name or "us-west-2"
    logger.info("Using region: %s", resolved_region)

    runtime = runtime or Runtime()

    resp = runtime.configure(
        entrypoint=entrypoint,
        auto_create_execution_role=auto_create_execution_role,
        auto_create_ecr=auto_create_ecr,
        requirements_file=requirements_file,
        region=resolved_region,
        agent_name=agent_name,
    )

    logger.info("Configure response: %s", resp)
    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure an agent in AgentCore Runtime.")
    parser.add_argument("--agent-name", required=True, help="Name to register for the agent")
    parser.add_argument("--entrypoint", default="langgraph_bedrock.py")
    parser.add_argument("--requirements-file", default="requirements.txt")
    parser.add_argument("--region", default=None)
    parser.add_argument("--no-role", dest="auto_create_execution_role", action="store_false")
    parser.add_argument("--no-ecr", dest="auto_create_ecr", action="store_false")

    args = parser.parse_args()

    resp = deploy_agent(
        agent_name=args.agent_name,
        entrypoint=args.entrypoint,
        requirements_file=args.requirements_file,
        region=args.region,
        auto_create_execution_role=args.auto_create_execution_role,
        auto_create_ecr=args.auto_create_ecr,
    )

    # Print the configure response for simple CLI usage
    print(resp)


if __name__ == "__main__":
    main()
