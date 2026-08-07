import sys
import os
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mcp_servers/code_exec/server.py"
)


def get_mcp_client():
    return MultiServerMCPClient({
        "code_exec": {
            "command": sys.executable,
            "args": ["-m", "mcp_servers.code_exec.server"],
            "transport": "stdio",
        }
    })


# ---------------------------------------------------------------------
# ORIGINAL (stateless): a new subprocess per tool call.
# ---------------------------------------------------------------------
async def load_tools():
    client = get_mcp_client()
    try:
        tools = await client.get_tools()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tools from 'code_exec' MCP server "
            f"(check that {SERVER_PATH} exists and starts without errors): {e}"
        ) from e
    return tools






# ---------------------------------------------------------------------
# FIXED (persistent): one subprocess/session shared across all calls,
# kept alive as long as `exit_stack` stays open.
# ---------------------------------------------------------------------
async def load_tools_persistent(exit_stack: AsyncExitStack):
    client = get_mcp_client()
    try:
        session = await exit_stack.enter_async_context(client.session("code_exec"))
        tools = await load_mcp_tools(session)
    except Exception as e:
        raise RuntimeError(
            f"Failed to start persistent 'code_exec' session "
            f"(check that {SERVER_PATH} exists and starts without errors): {e}"
        ) from e
    return tools