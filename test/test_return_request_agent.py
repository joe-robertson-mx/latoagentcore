import pytest
import asyncio
import types
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock

# Ensure src/ is on sys.path so tests can import package modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from return_request_agent import stream_return_request_agent


@pytest.mark.asyncio
async def test_stream_return_request_agent_chunks_and_done(monkeypatch):
    # Patch create_agent to return an object whose ainvoke returns a fake final message
    fake_result = {"messages": [{"content": "This is a long response from the agent."}]}

    fake_agent = Mock()
    fake_agent.ainvoke = AsyncMock(return_value=fake_result)

    with patch("return_request_agent.create_agent", return_value=fake_agent) as _:
        events = []
        async for ev in stream_return_request_agent("joe@example.com", "My bike is broken", llm=Mock(), chunk_size=10):
            events.append(ev)

    assert events[0]["type"] == "start"
    # At least one delta and a final done
    assert any(e.get("type") == "delta" for e in events)
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1
    assert done_events[0]["raw"] == fake_result


@pytest.mark.asyncio
async def test_invoke_routes_to_streaming_handler(monkeypatch):
    # Import here to avoid circular import at module import time
    import main as main_module

    # Create a fake async generator for the streaming handler
    async def fake_stream(from_field, prompt, llm, external_tools=None):
        yield {"type": "start"}
        yield {"type": "delta", "text": "hello"}
        yield {"type": "done", "summary": "ok", "raw": {}}

    # Patch get_tools to avoid gateway calls
    fake_mcp_client = Mock()
    fake_mcp_client.get_tools = AsyncMock(return_value=[])
    with patch("main.get_tools", AsyncMock(return_value=fake_mcp_client)):
        with patch("main.stream_return_request_agent", fake_stream):
            payload = {"agent": "return_request", "from": "joe@example.com", "prompt": "hi"}
            result = await main_module.invoke(payload)
            # Because our invoke returns the async generator directly, it should be an async generator object
            assert isinstance(result, types.AsyncGeneratorType)
            # Collect items to ensure it yields
            collected = []
            async for item in result:
                collected.append(item)
            assert collected and collected[-1]["type"] == "done"
