# Lato Agent Core

A demo showcasing **Amazon Bedrock AgentCore** and **LangGraph** for the Lato Enquiry Management use case. This project demonstrates a multi-agent system that helps customer service representatives (CSRs) handle e-bike customer enquiries—including returns, orders, warranty claims, and general queries.

## Overview

This solution uses:

- **[Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/)** — Enterprise-grade runtime for deploying and managing AI agents with zero infrastructure management
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Framework for building stateful, multi-agent applications as graphs
- **[LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)** — Integration with Model Context Protocol (MCP) servers for external tool access
- **Claude (via Amazon Bedrock)** — Foundation model powering agent reasoning

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Bedrock AgentCore Runtime                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                            │
│  │   Router    │  Classifies enquiry type                   │
│  │   Agent     │  (ORDER, RETURN REQUEST, WARRANTY, OTHER)  │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Specialist Worker Agents                │   │
│  ├──────────────┬─────────────────┬────────────────────┤   │
│  │ Return/      │    Order        │     General        │   │
│  │ Warranty     │    Agent        │     Agent          │   │
│  └──────────────┴─────────────────┴────────────────────┘   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Tools                              │   │
│  │  • CustomerData_Get (Mendix Order API)               │   │
│  │  • MCP Tools (Product Inventory via SSE)             │   │
│  │  • Classification Tools                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — Fast Python package manager
- AWS CLI configured with appropriate credentials
- Docker (for local development)

### 1. Set Up Environment

```bash
cd latoagentcore

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/Scripts/activate  # Windows Git Bash
# or: source .venv/bin/activate  # Linux/Mac
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in required values:

```bash
# AWS Configuration
AWS_REGION=eu-west-1

# MCP Server Credentials
PI_PROD_MCP_SSE_URL=https://...
PI_PROD_MCP_USERNAME=...
PI_PROD_MCP_PASSWORD=...

```

See the Lato Product Inventory Mendix app for these credentials and MCP server details.

### 3. Run Locally

```bash
# Launch local development server (requires Docker)
agentcore launch --local

# In another terminal, test with a sample request
agentcore invoke '{"prompt": "joe@example.com, my gear set is jammed"}' --local
```

### 4. Deploy to AWS

```bash
# Deploy the agent to Bedrock AgentCore Runtime
agentcore launch

# Invoke the deployed agent
agentcore invoke --agent=enquiry-mgmt '{"prompt":"frederich.torresi@optimumsystems.com, My Precision Gear Set with Serial No: 8dc04f7a-ec70-4c79-9eba-759396546948 has jammed, I think some of the cables have frayed.."}'

```

### 5. Monitor & Observe

Check agent status and access observability:

```bash
agentcore status
```

View traces and logs in the [AWS CloudWatch GenAI Observability dashboard](https://console.aws.amazon.com/cloudwatch/home#gen-ai-observability/agent-core):

- **Agents View** — All deployed agents and their runtime metrics
- **Sessions View** — Conversation sessions across agents
- **Traces View** — Detailed execution traces and span information

Tail logs directly via AWS CLI:

```bash
aws logs tail /aws/bedrock-agentcore/runtimes/<AGENT_ID>-DEFAULT --follow
```

## Agents

| Agent | Description |
|-------|-------------|
| `enquiry-mgmt` (default) | Multi-agent router that classifies enquiries and delegates to specialist workers |
| `return_request` | Standalone agent for handling product return/repair requests |

## Project Structure

```
latoagentcore/
├── .bedrock_agentcore.yaml   # AgentCore CLI configuration
├── pyproject.toml            # Python dependencies
├── src/
│   ├── enquiry_mgmt.py       # Multi-agent LangGraph workflow (router + workers)
│   ├── return_request_agent.py  # Standalone return request agent
│   └── mcp_client/
│       └── client.py         # MCP client helpers (Gateway + PI Prod SSE)
├── test/                     # Unit tests
└── terraform/                # Infrastructure as code (optional)
```

## Key Features

- **Router-Worker Pattern** — Intelligent routing based on enquiry classification
- **MCP Integration** — Connect to external systems via Model Context Protocol
- **Structured Output** — Classification tools capture actionable recommendations for CSRs
- **Session Management** — Multi-turn conversation support with memory
- **Observability** — Built-in tracing and monitoring via CloudWatch

## CLI Reference

| Command | Description |
|---------|-------------|
| `agentcore configure --entrypoint <file>` | Configure agent entrypoint |
| `agentcore launch --local` | Run locally (requires Docker) |
| `agentcore launch` | Deploy to AWS Bedrock AgentCore |
| `agentcore invoke '<json>'` | Invoke the agent |
| `agentcore invoke '<json>' --local` | Invoke local agent |
| `agentcore invoke '<json>' --session-id <id>` | Invoke with session for multi-turn |
| `agentcore status` | Check deployment status and get log paths |

## Further Reading

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [Bedrock AgentCore Starter Toolkit](https://aws.github.io/bedrock-agentcore-starter-toolkit/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
