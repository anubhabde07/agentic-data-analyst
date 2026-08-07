import asyncio
import uuid
from orchestrator.langgraph_agent import build_agent, close_agent

async def main():
    agent = await build_agent()
    try:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content":
                "Load data/Walmart_Sales.csv, then tell me what columns are in it."}]},
            config,
        )

        for msg in result["messages"]:
            print(f"[{msg.type}] {msg.content}")
    finally:
        await close_agent(agent)  # terminates the persistent MCP subprocess

asyncio.run(main())