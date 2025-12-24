"""Router-worker multi-agent system for Lato enquiries.

This module defines a LangGraph workflow composed of a router agent plus
specialised worker agents. It is designed to be used as the Bedrock
AgentCore entrypoint when deploying the multi-agent experience.
"""
from __future__ import annotations

from typing import Any, Callable, Awaitable, Dict, List, Literal, Optional, TypedDict, Union

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_aws import ChatBedrock
from langgraph.graph import END, StateGraph
import asyncio
import requests
from langchain_mcp_adapters.client import MultiServerMCPClient


# =============================================================================
# WORKAROUND: Bedrock Converse API compatibility for MCP tool results
# =============================================================================
# Issue: langchain-mcp-adapters v0.2.x adds an `id` field to each content block
# in tool results. AWS Bedrock's Converse API uses strict schema validation and
# rejects any extra fields, causing:
#   ValidationException: messages.2.content.1.tool_result.content.0.text.id:
#   Extra inputs are not permitted
#
# Root cause: The `_convert_call_tool_result` function in langchain-mcp-adapters
# (tools.py lines 137-194) adds a LangChain-specific `id` to content blocks:
#   {"type": "text", "text": "result", "id": "some-uuid"}
#
# When langchain-aws converts this to Bedrock format, the `id` field passes
# through to the API call and triggers strict validation rejection.
#
# Solution: This wrapper class extends ChatBedrock to sanitize messages before
# they're sent to the model, stripping the `id` field from tool result content.
#
# TODO: Remove this workaround once one of these is fixed:
# - langchain-mcp-adapters stops adding `id` to content blocks, OR
# - langchain-aws strips unknown fields when converting to Bedrock format
# Track: https://github.com/langchain-ai/langchain-mcp-adapters/issues
# =============================================================================


def strip_content_block_ids(content: Any) -> Any:
    """Recursively strip 'id' fields from content blocks.
    
    This sanitizes tool result content for Bedrock Converse API compatibility.
    
    Args:
        content: Tool result content - can be a string, dict, or list.
        
    Returns:
        Content with 'id' fields removed from any content block dicts.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        # Remove 'id' if present, keep everything else
        return {k: strip_content_block_ids(v) for k, v in content.items() if k != "id"}
    elif isinstance(content, list):
        return [strip_content_block_ids(item) for item in content]
    else:
        return content


def sanitize_messages_for_bedrock(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Sanitize a list of messages for Bedrock Converse API compatibility.
    
    This processes ToolMessage content to remove the 'id' field that
    langchain-mcp-adapters adds to content blocks.
    
    Args:
        messages: List of LangChain messages to sanitize.
        
    Returns:
        List of messages with tool result content blocks sanitized.
    """
    sanitized = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Sanitize the content to remove 'id' fields
            sanitized_content = strip_content_block_ids(msg.content)
            sanitized.append(ToolMessage(
                content=sanitized_content,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            ))
        else:
            sanitized.append(msg)
    return sanitized


class BedrockCompatChatModel(ChatBedrock):
    """ChatBedrock wrapper that sanitizes messages for Bedrock Converse API.
    
    This wrapper intercepts messages before they're sent to Bedrock and
    removes the 'id' field from tool result content blocks, which Bedrock's
    strict validation rejects.
    
    Usage:
        llm = BedrockCompatChatModel(model_id="...", model_kwargs={...})
        # Use llm as normal - it will automatically sanitize tool messages
    """
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        sanitized = sanitize_messages_for_bedrock(messages)
        return super()._generate(sanitized, stop=stop, run_manager=run_manager, **kwargs)
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        sanitized = sanitize_messages_for_bedrock(messages)
        return await super()._agenerate(sanitized, stop=stop, run_manager=run_manager, **kwargs)

MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
SUPPORTED_CATEGORIES = {"ORDER", "WARRANTY CLAIM", "RETURN REQUEST", "OTHER"}


class WorkflowState(TypedDict, total=False):
    """State that flows through the LangGraph workflow."""


    enquiry: str
    from_field: str
    category: Literal["ORDER", "WARRANTY CLAIM", "RETURN REQUEST", "OTHER"]
    router_reasoning: str
    notes: List[str]
    response: str
    return_request_classification: Dict[str, Any]
    order_classification: Dict[str, Any]


@tool("EnquiryClassification")
def enquiry_classification(category: str, reasoning: str) -> Dict[str, str]:
    """Record how the enquiry should be routed. Allowed categories: ORDER, WARRANTY CLAIM, RETURN REQUEST, OTHER."""

    category_upper = category.upper()
    if category_upper not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category '{category}'.")
    return {"category": category_upper, "reasoning": reasoning}


@tool("CustomerData_Get")
def customer_data_get(email: str) -> Any:
    """Retrieve customer order data for a given email.

    Returns the parsed JSON response from the order management REST endpoint.
    """
    url = "https://midaas-accp.mendixcloud.com/rest/ordermanagement/v1/order"
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


@tool("ReturnRequestClassification")
def return_request_classification(
    summary: str,
    recommendation: str = "",
    priority: str = "normal",
    severity: Optional[str] = None,
    known_part: bool = False,
    clarification_required: bool = False,
    book_in_for_repair: bool = False,
) -> Dict[str, Any]:
    """Capture structured guidance for a return / repair scenario.

    Fields:
    - recommendation: concise next-step recommendation for CSR.
    - priority: low | normal | high | urgent (free-text allowed, but prefer these).
    - known_part: True if part/sku is clearly identified or found in order history or part research.
    - clarification_required: True if missing key info to proceed.
    - book_in_for_repair: True if the customer should be booked in for repair.
    """

    resolved_priority = priority
    if severity and (not priority or priority == "normal"):
        # Backward-compat: older callers used 'severity'. Treat it as priority.
        resolved_priority = severity

    return {
        "summary": summary,
        "recommendation": recommendation,
        "priority": resolved_priority,
        "known_part": known_part,
        "clarification_required": clarification_required,
        "book_in_for_repair": book_in_for_repair,
    }


@tool("OrderClassification")
def order_classification(
    summary: str,
    recommendation: str = "",
    priority: str = "normal",
    known_part: bool = False,
    clarification_required: bool = False,
    raise_po: bool = False,
) -> Dict[str, Any]:
    """Capture structured guidance for CSR order follow-ups.

    Fields:
    - recommendation: concise next-step recommendation for CSR.
    - priority: low | normal | high | urgent (free-text allowed, but prefer these).
    - known_part: True if part/sku is clearly identified or found in order history or part research.
    - clarification_required: True if missing key info to proceed.
    - raise_po: True if a PO should be raised.
    """

    return {
        "summary": summary,
        "recommendation": recommendation,
        "priority": priority,
        "known_part": known_part,
        "clarification_required": clarification_required,
        "raise_po": raise_po,
    }


@tool("SetEmailResponse")
def set_email_response(response: str) -> str:
    """Record the outbound email draft that should be sent to the customer."""

    return response


@tool("AddNote")
def add_note(note: str) -> str:
    """Attach an internal CSR note."""

    return note


# Use BedrockCompatChatModel instead of ChatBedrock to work around the
# langchain-mcp-adapters content block ID issue (see workaround notes above)
llm = BedrockCompatChatModel(model_id=MODEL_ID, model_kwargs={"temperature": 0.1})


# PI Product Inventory MCP client configuration (connection happens in async entrypoint)
mcp_client = MultiServerMCPClient(
    {
        "pi_prod": {
            "transport": "sse",
            "url": "https://lato-product-inventory.apps.eu-1c.mendixcloud.com/mendix-mcp/sse",
            "headers": {
                "Username": "User",
                "Authorization": "Pass01",
            },
            "timeout": 15.0,
            "sse_read_timeout": 60.0,
        }
    },
)


ROUTER_PROMPT = (
    "You are an email triage assistant. Read the enquiry and classify it as "
    "ORDER, WARRANTY CLAIM, RETURN REQUEST, or OTHER. Provide concise reasoning "
    "and call the EnquiryClassification tool exactly once. Never ask follow-up questions."
)

RETURN_REQUEST_PROMPT = (
    "You are a Return Request specialist. Review the enquiry, evaluate the order "
    "history and part value, and determine if a repair booking is required. "
    "Use CustomerData_Get when it helps.\n\n"
    "For product/part research, always call GetProductDetails or GETPartDetails.\n"
    "When a part, SKU, or product is mentioned, call the MCP tools to look it up.\n\n"
    "You MUST call ReturnRequestClassification exactly once with these fields:\n"
    "- summary (1-2 sentences)\n"
    "- recommendation (what CSR should do next)\n"
    "- priority: low|normal|high|urgent\n"
    "- known_part (true/false)\n"
    "- clarification_required (true/false)\n"
    "- book_in_for_repair (true/false)\n\n"
    "If clarification_required=true, ensure the email asks ONLY the minimum questions needed. "
    "Draft the customer-facing email via SetEmailResponse. Use AddNote for internal tasks."
)

ORDER_PROMPT = (
    "You are an Order specialist. Organize the customer's issues into an "
    "actionable CSR plan. Leverage order history and AddNote for follow-up tasks.\n\n"
    "For product/part research, always call GetProductDetails or GETPartDetails.\n"
    "When a part, SKU, or product is mentioned, call the MCP tools to look it up.\n\n"
    "You MUST call OrderClassification exactly once with these fields:\n"
    "- summary (1-2 sentences)\n"
    "- recommendation (what CSR should do next)\n"
    "- priority: low|normal|high|urgent\n"
    "- known_part (true/false)\n"
    "- clarification_required (true/false)\n"
    "- raise_po (true/false)"
)

GENERAL_PROMPT = (
    "You are a general researcher helping CSRs with uncategorised enquiries. Use "
    "CustomerData_Get, AddNote, and available MCP tools to assemble a concise context pack."
)

# Router agent (no MCP tools needed - can be created at module level)
router_agent = create_agent(
    model=llm,
    tools=[enquiry_classification],
    system_prompt=ROUTER_PROMPT,
)


def _extract_tool_args_from_messages(messages: List[Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Scan a list of messages and return the first matching tool call args."""

    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None) or []
        for call in tool_calls:
            if call.get("name") == tool_name:
                return call.get("args", {})
    return None


async def route_enquiry(state: WorkflowState) -> WorkflowState:
    """Ask the router agent to classify the enquiry."""

    result = await router_agent.ainvoke({"messages": [HumanMessage(content=state["enquiry"])]})
    messages = result.get("messages", [])
    args = _extract_tool_args_from_messages(messages, "EnquiryClassification")
    if not args:
        raise ValueError("Router did not emit an EnquiryClassification tool call")
    category = args.get("category", "OTHER").upper()
    reasoning = args.get("reasoning", "")
    return {
        "category": category,
        "router_reasoning": reasoning,
    }


async def _run_worker(agent, state: WorkflowState, worker_label: str) -> WorkflowState:
    """Invoke a worker agent and capture its narrative output."""

    result = await agent.ainvoke({"messages": [HumanMessage(content=state["enquiry"])]})
    messages = result.get("messages", [])
    message = messages[-1] if messages else AIMessage(content="")
    content = message.content if isinstance(message.content, str) else str(message.content)
    notes = list(state.get("notes", []))
    notes.append(f"[{worker_label}] {content}")

    updates: WorkflowState = {"notes": notes}

    rr_args = _extract_tool_args_from_messages(messages, "ReturnRequestClassification")
    if rr_args:
        updates["return_request_classification"] = rr_args

    order_args = _extract_tool_args_from_messages(messages, "OrderClassification")
    if order_args:
        updates["order_classification"] = order_args

    response = state.get("response") or ""

    if worker_label == "ReturnRequest":
        # Prefer tool-call args over scraping narrative content.
        email_args = _extract_tool_args_from_messages(messages, "SetEmailResponse")
        if email_args and isinstance(email_args.get("response"), str):
            response = email_args["response"]
        elif not response:
            # Backstop: store the last assistant content.
            response = content

    updates["response"] = response
    return updates


def finalize_state(state: WorkflowState) -> WorkflowState:
    """Ensure a response string exists before returning to the runtime."""

    response = state.get("response") or "No outbound response was generated; CSR follow-up required."
    return {"response": response}


def _decide_route(state: WorkflowState) -> str:
    mapping = {
        "RETURN REQUEST": "return_request",
        "WARRANTY CLAIM": "return_request",
        "ORDER": "order",
        "OTHER": "general",
    }
    return mapping.get(state.get("category", "OTHER"), "general")


app = BedrockAgentCoreApp()


@app.entrypoint
async def multi_agent_entrypoint(payload: Dict[str, str]) -> Dict[str, object]:
    """Handle runtime invocations by running the LangGraph workflow."""

    enquiry = payload.get("prompt") or payload.get("content")
    if not enquiry:
        raise ValueError("Payload must include 'prompt' or 'content'")
    from_field = payload.get("from") or payload.get("from_field", "unknown")
    
    mcp_tools = await mcp_client.get_tools()
    print(f"✓ Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")
    
    # Create worker agents with MCP tools
    return_request_agent = create_agent(
        model=llm,
        tools=[customer_data_get, return_request_classification, set_email_response, add_note, *mcp_tools],
        system_prompt=RETURN_REQUEST_PROMPT,
    )
    order_agent = create_agent(
        model=llm,
        tools=[customer_data_get, order_classification, add_note, *mcp_tools],
        system_prompt=ORDER_PROMPT,
    )
    general_agent = create_agent(
        model=llm,
        tools=[customer_data_get, add_note, *mcp_tools],
        system_prompt=GENERAL_PROMPT,
    )
    
    # Define async node wrappers
    async def run_return_request(state: WorkflowState) -> WorkflowState:
        return await _run_worker(return_request_agent, state, "ReturnRequest")
    
    async def run_order(state: WorkflowState) -> WorkflowState:
        return await _run_worker(order_agent, state, "Order")
    
    async def run_general(state: WorkflowState) -> WorkflowState:
        return await _run_worker(general_agent, state, "General")
    
    # Build workflow
    workflow = StateGraph(WorkflowState)
    workflow.add_node("router", route_enquiry)
    workflow.add_node("return_request", run_return_request)
    workflow.add_node("order", run_order)
    workflow.add_node("general", run_general)
    workflow.add_node("finalize", finalize_state)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        _decide_route,
        {
            "return_request": "return_request",
            "order": "order",
            "general": "general",
        },
    )
    workflow.add_edge("return_request", "finalize")
    workflow.add_edge("order", "finalize")
    workflow.add_edge("general", "finalize")
    workflow.add_edge("finalize", END)
    
    enquiry_graph = workflow.compile()
    
    # Run workflow
    state: WorkflowState = {
        "enquiry": f"From: {from_field}\nContent: {enquiry}",
        "from_field": from_field,
        "notes": [],
    }
    result_state = await enquiry_graph.ainvoke(state)

    classification: Dict[str, Any] = {}
    if result_state.get("category") in {"RETURN REQUEST", "WARRANTY CLAIM"}:
        classification = result_state.get("return_request_classification", {})
    elif result_state.get("category") == "ORDER":
        classification = result_state.get("order_classification", {})

    return {
        "category": result_state.get("category", "UNKNOWN"),
        "reasoning": result_state.get("router_reasoning", ""),
        "classification": classification,
        "notes": result_state.get("notes", []),
        "response": result_state.get("response"),
    }


if __name__ == "__main__":
    app.run()
