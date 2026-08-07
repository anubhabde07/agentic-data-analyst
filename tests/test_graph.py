import asyncio
from orchestrator.graph import build_graph

async def main():
    graph = await build_graph()

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Load data/Walmart_Sales.csv, then tell me the columns."}]},
        {"configurable": {"thread_id": "test-1"}},
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())