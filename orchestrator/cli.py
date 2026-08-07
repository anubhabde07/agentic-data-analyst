import asyncio
import uuid
from orchestrator.langgraph_agent import build_agent, close_agent


async def main():
    agent = await build_agent()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Session: {thread_id} — type a question ('quit' to exit)")

    try:
        while True:
            q = input("> ")
            if q.strip().lower() == "quit":
                break
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": q}]}, config
            )
            print(result["messages"][-1].content)
    finally:
        # Terminates the persistent MCP subprocess. Without this, closing
        # the REPL with 'quit' (or Ctrl+C) leaves the code_exec subprocess
        # running in the background.
        await close_agent(agent)


if __name__ == "__main__":
    asyncio.run(main())