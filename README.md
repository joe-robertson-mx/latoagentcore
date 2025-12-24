# Lato Agent Core

Multi-agent system for handling customer requests (returns, orders, and general queries).

## Quick Start

### Local Testing

1. **Set up environment**
   ```bash
   cd latoagentcore
   source .venv/Scripts/activate  # Windows Git Bash
   # or: source .venv/bin/activate  # Linux/Mac
   ```

2. **Configure environment variables**
   Copy `.env.example` to `.env` and fill in required values (AWS region, MCP credentials, etc.)

3. **Run local dev server**
   ```bash
   agentcore dev
   ```

4. **Test with a request**
   ```bash
   agentcore invoke --dev '{"prompt": "joe@example.com, my gear set is jammed"}'
   ```

### Deploy to AWS

1. **Deploy the agent**
   ```bash
   cd latoagentcore
   agentcore deploy --agent enquiry-mgmt
   ```

2. **Invoke the deployed runtime**
   ```bash
   agentcore invoke '{"prompt": "joe@example.com, my gear set is jammed"}'
   ```

3. **View observability traces**
   ```bash
   agentcore obs list --agent enquiry-mgmt
   agentcore obs show --session-id <session-id> --verbose
   ```

## Agents

- **enquiry-mgmt** (default): Routes requests to specialized workers (return requests, orders, general queries)
- **return_request**: Handles product return/repair requests

## Structure

- `src/enquiry_mgmt.py` - Multi-agent router and workers
- `src/return_request_agent.py` - Return request handler
- `src/mcp_client/client.py` - MCP client helpers (gateway + PI Prod SSE)
- `.bedrock_agentcore.yaml` - Agent configuration for CLI
