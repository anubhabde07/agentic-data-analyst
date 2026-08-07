from contextlib import AsyncExitStack
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from orchestrator.mcp_tools import load_tools_persistent
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a data analyst agent. You have tools to load and
explore a dataset before answering questions. Always check the schema
before assuming column names exist. Explore as many times as needed, then
give a clear final answer summarizing what you found."""


@dataclass
class Agent:
    """Wraps the compiled graph together with the resource that owns the
    persistent MCP subprocess, instead of monkey-patching an attribute onto
    CompiledStateGraph (which isn't declared on that class and upsets type
    checkers)."""
    graph: CompiledStateGraph
    _exit_stack: AsyncExitStack

    async def ainvoke(self, *args, **kwargs):
        return await self.graph.ainvoke(*args, **kwargs)

    async def aclose(self):
        await self._exit_stack.aclose()


async def build_agent() -> Agent:
    # Owns the lifetime of the stdio subprocess. Kept open for as long as
    # `agent` is in use, so every tool call across the whole conversation
    # (all thread turns) reuses the SAME subprocess instead of spawning a
    # new one per call.
    exit_stack = AsyncExitStack()
    tools = await load_tools_persistent(exit_stack)

    model = init_chat_model("anthropic:claude-haiku-4-5-20251001")
    checkpointer = InMemorySaver()
    graph = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer
    )

    return Agent(graph=graph, _exit_stack=exit_stack)


async def close_agent(agent: Agent):
    """Call this on shutdown to terminate the underlying MCP subprocess(es)."""
    await agent.aclose()