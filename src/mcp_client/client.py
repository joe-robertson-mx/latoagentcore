import os
from langchain_mcp_adapters.client import MultiServerMCPClient
import requests

COGNITO_TOKEN_URL = os.getenv("COGNITO_TOKEN_URL")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET")
COGNITO_SCOPE = os.getenv("COGNITO_SCOPE")

PI_PROD_MCP_SSE_URL = os.getenv("PI_PROD_MCP_SSE_URL", "https://lato-product-inventory.apps.eu-1c.mendixcloud.com/mendix-mcp/sse/")
PI_PROD_MCP_USERNAME = os.getenv("PI_PROD_MCP_USERNAME", "User")
PI_PROD_MCP_PASSWORD = os.getenv("PI_PROD_MCP_PASSWORD", "Pass01")

def _get_access_token():
    """
    Make a POST request to the Cognito OAuth token URL using client credentials.
    """
    response = requests.post(
        COGNITO_TOKEN_URL,
        auth=(COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET),
        data={
            "grant_type": "client_credentials",
            "scope": COGNITO_SCOPE,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return response.json()["access_token"]


def get_streamable_http_mcp_client() -> MultiServerMCPClient:
    """
    Returns an MCP Client for AgentCore Gateway compatible with LangGraph
    """
    gateway_url = os.getenv("GATEWAY_URL")
    if not gateway_url:
        raise RuntimeError("Missing required environment variable: GATEWAY_URL")
    access_token = _get_access_token()
    return MultiServerMCPClient(
        {
            "agentcore_gateway": {
                "transport": "streamable_http",
                "url": gateway_url,
                "headers": {
                    "Authorization": f"Bearer {access_token}"
                }
            }
        }
    )


def get_pi_prod_sse_mcp_client() -> MultiServerMCPClient:
    """Connect to the Product Inventory (PI) Mendix MCP endpoint over SSE.

    Expects these environment variables:
    - PI_PROD_MCP_SSE_URL
    - PI_PROD_MCP_USERNAME
    - PI_PROD_MCP_PASSWORD

    Notes:
    - This uses custom headers ("Username"/"Password") because some Mendix deployments
      are configured that way.
    - Avoid hardcoding secrets in source; prefer env vars or a secret manager.
    """

    if not PI_PROD_MCP_SSE_URL:
        raise RuntimeError("Missing required environment variable: PI_PROD_MCP_SSE_URL")
    if not PI_PROD_MCP_USERNAME:
        raise RuntimeError("Missing required environment variable: PI_PROD_MCP_USERNAME")
    if not PI_PROD_MCP_PASSWORD:
        raise RuntimeError("Missing required environment variable: PI_PROD_MCP_PASSWORD")

    return MultiServerMCPClient(
        {
            "pi_prod": {
                "transport": "sse",
                "url": PI_PROD_MCP_SSE_URL,
                "headers": {
                    "Accept": "text/event-stream",
                    "Username": PI_PROD_MCP_USERNAME,
                    "Password": PI_PROD_MCP_PASSWORD,
                },
                # Fail fast on initial connection; keep-alive/read timeout is separate.
                "timeout": 15.0,
                "sse_read_timeout": 60.0,
            }
        }
    )