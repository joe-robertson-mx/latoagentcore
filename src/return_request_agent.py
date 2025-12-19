from typing import Any
from langchain_core.tools import tool
from langchain.agents import create_agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_aws import ChatBedrock
import requests

app = BedrockAgentCoreApp()

SYSTEM_PROMPT = (
    "Role: You are an AI assistant for a customer service representative (CSR) at an e-bike manufacturer. "
    "Your job is to organize customer repair issues into a clear, actionable plan for the CSR.\n\n"
    "Core Task: Your task is to receive a customer's problem description, identify the key details needed for a repair, "
    "and research useful information from various Lato systems and provide that to the CSR. You should consider the customers "
    "order history, value of the part and cost of repair when proposing your plan.\n\n"
    "Final Output: Your response should be easy to follow and contain a short summary and next steps."
)

USER_PROMPT_TEMPLATE = "From: {{From}}\nContent: {{InputContent}}"

CUSTOMER_ORDERS_URL = "https://midaas-accp.mendixcloud.com/rest/ordermanagement/v1/order"
# Use a Bedrock chat model that supports on-demand invocation without an inference profile.
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

@tool
def get_customer_order_data_tool(email: str) -> Any:
    """Fetch order details for a customer email from the Mendix order service."""
    url = CUSTOMER_ORDERS_URL
    params = {"email": email}
    headers = {"Accept": "application/json"}
    timeout = (3.05, 15)  # (connect_timeout, read_timeout)

    with requests.Session() as session:
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # Try to parse JSON; if the response body is empty or not JSON, return useful debug info
        try:
            return resp.json()
        except ValueError:
            # JSON decoding failed — include status code and a snippet of the body for debugging
            text = resp.text or ""
            snippet = text[:1000]
            return {
                "_error": "invalid_json_response",
                "status_code": resp.status_code,
                "content_type": resp.headers.get("Content-Type"),
                "text_snippet": snippet,
            }
        
def create_return_request_agent(): 
    llm = ChatBedrock(model_id=MODEL_ID, model_kwargs={"temperature": 0.1})
    tools = [get_customer_order_data_tool]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

#Initialise the agent
agent = create_return_request_agent()

@app.entrypoint
def return_request_agent(payload):
    input = payload.get("prompt")
    response = agent.invoke({"messages": [{"role": "user", "content": input}]})
    # Extract the final message content
    return response["messages"][-1].content

if __name__ == "__main__":
    app.run()