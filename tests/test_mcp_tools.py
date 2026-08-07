import asyncio
from orchestrator.mcp_tools import load_tools

async def main():
    tools = await load_tools()
    for t in tools:
        print(t.name, "-", t.description)
        
asyncio.run(main())