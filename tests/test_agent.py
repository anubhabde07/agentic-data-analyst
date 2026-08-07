import asyncio
import uuid
from orchestrator.langgraph_agent import build_agent, close_agent

async def main():
    agent = await build_agent()
    try:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content":
                "Clean this data/laptopData.csv dataset, remove duplicates, handle missing values and outliers, then tell me which features have the strongest influence on laptop price. Visualize the top relationship and export the cleaned dataset."}]},
            config,
        )

        for msg in result["messages"]:
            print(f"[{msg.type}] {msg.content}")
    finally:
        await close_agent(agent)  # terminates the persistent MCP subprocess

asyncio.run(main())