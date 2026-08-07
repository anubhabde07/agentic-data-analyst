import asyncio
from orchestrator.graph import build_graph

async def main():
    graph = await build_graph()
    config = {"configurable": {"thread_id": "self-correct-test"}}

    result = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content":
                "Load data/Walmart_Sales.csv as df, then compute the average of the "
                "'Unemp' column (note: this column name is likely wrong on purpose — "
                "figure out the correct name and use it)."}],
            "retry_count": 0,
        },
        config,
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {str(msg.content)[:200]}")
    print("Final retry_count:", result["retry_count"])

asyncio.run(main())