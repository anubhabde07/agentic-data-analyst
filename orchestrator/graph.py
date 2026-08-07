from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from orchestrator.state import AgentState
from orchestrator.mcp_tools import load_tools
from langchain.messages import SystemMessage, ToolMessage
from dotenv import load_dotenv

SYSTEM_PROMPT = """You are a data analyst agent. You have tools to load and
explore a dataset before answering questions. Always check the schema
before assuming column names exist."""

MAX_MESSAGES = 10


def get_text(message) -> str:
    """Extract text from a message's content, whether it's a plain string
    or a list of content blocks (e.g. from MCP tool results)."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""



async def build_graph():
    load_dotenv()
    
    tools = await load_tools()
    model = init_chat_model("groq:qwen/qwen3.6-27b").bind_tools(tools)
    
    # def agent_node(state: AgentState):
    #     """Call the model with the current message history, injecting the system
    #     prompt once if it isn't already present."""
    #     messages = state["messages"]
    #     if not any(isinstance(m, SystemMessage) for m in messages):
    #         messages = [SystemMessage(SYSTEM_PROMPT)] + messages
    #     response = model.invoke(messages)
    #     return {"messages": [response]}
    
    def agent_node(state: AgentState):
        """
        Call the model using a compact message history.

        - Keep only recent messages.
        - Compress old ToolMessages.
        - Preserve error messages for retries.
        """

        recent_messages = state["messages"][-MAX_MESSAGES:]

        compact_messages = []

        for msg in recent_messages:

            if isinstance(msg, ToolMessage):
                text = get_text(msg)

                if text.startswith("ERROR:"):
                    # Preserve enough of the error so the model can retry.
                    summary = text[:300]
                else:
                    # Replace large successful tool outputs.
                    summary = "Tool completed successfully."

                compact_messages.append(
                    ToolMessage(
                        content=summary,
                        tool_call_id=msg.tool_call_id,
                    )
                )

            else:
                compact_messages.append(msg)

        if not any(isinstance(m, SystemMessage) for m in compact_messages):
            compact_messages.insert(0, SystemMessage(SYSTEM_PROMPT))

        response = model.invoke(compact_messages)

        return {"messages": [response]}


    def should_continue(state: AgentState) -> str:
        """Route to tools if the model's last response requested a tool call,
        otherwise end the run with that response as the final answer."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        
        return END


    def decide_next_step(state: AgentState) -> str:
        """After a tool runs: retry on error (if under the cap), give up if the cap
        is hit, or hand back to the agent on success."""
        last = state["messages"][-1]
        is_error = get_text(last).startswith("ERROR:")

        if is_error and state["retry_count"] < 3:
            return "agent"
        if is_error:
            return "give_up"
        
        return "agent"


    def bump_retry_count(state: AgentState):
        """Increment retry_count only when the last tool result was an error."""
        last = state["messages"][-1]
        is_error = get_text(last).startswith("ERROR:")
        
        return {"retry_count": state["retry_count"] + 1} if is_error else {}


    def give_up(state: AgentState):
        """Final message shown when the retry cap is hit."""
        last = state["messages"][-1]
        return {"messages": [{"role": "assistant", "content":
            "I wasn't able to get this working after several attempts. "
            "Here's what went wrong on the last try:\n" + get_text(last)}]}
 
 
    tool_node = ToolNode(tools)
 
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("bump_retry_count", bump_retry_count)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("agent")
    # workflow.add_conditional_edges(source_node, routing_function, path_map)
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "bump_retry_count")
    workflow.add_conditional_edges("bump_retry_count", decide_next_step, {"agent": "agent", "give_up": "give_up"})
    workflow.add_edge("give_up", END)

    return workflow.compile(checkpointer=InMemorySaver())