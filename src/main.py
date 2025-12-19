import os
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from bedrock_agentcore import BedrockAgentCoreApp
from .mcp_client.client import get_streamable_http_mcp_client as deployed_get_tools
from model.load import load_model

if os.getenv("LOCAL_DEV") == "1":
    # In local dev, instantiate dummy MCP client so the code runs without deploying
    async def get_tools():
        class DummyClient:
            async def get_tools(self):
                return []

        return DummyClient()
else:
    get_tools = deployed_get_tools

# Instantiate model
llm = load_model()

# Define a simple function tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers"""
    return a+b

# Note: `get_tools` is a coroutine factory that returns a client object.
# We must call it inside the async handler and await it to get the client.

# Integrate with Bedrock AgentCore
app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload):
    # assume payload input is structured as { "prompt": "<user input>" }

    # Load MCP Tools
    # instantiate the mcp client (factory may be async)
    mcp_client = await get_tools()
    tools = await mcp_client.get_tools()

    # If this invocation is intended for the return-request agent, route to the
    # streaming implementation which yields an async generator (SSE-capable).
    if payload.get("agent") == "return_request":
        from .return_request_agent import stream_return_request_agent

        from_field = payload.get("from", "unknown")
        prompt = payload.get("prompt", payload.get("input", ""))
        # Return the async generator directly; the runtime converts it to SSE
        return stream_return_request_agent(from_field, prompt, llm, external_tools=tools)

    # Define the default agent
    graph = create_agent(llm, tools=tools + [add_numbers])

    # Process the user prompt
    prompt = payload.get("prompt", "What is Agentic AI?")

    # Run the agent
    result = await graph.ainvoke({"messages": [HumanMessage(content=prompt)]})

    # Return result
    return {
        "result": result["messages"][-1].content
    }

if __name__ == "__main__":
    app.run()