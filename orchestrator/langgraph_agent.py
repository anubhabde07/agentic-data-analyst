from contextlib import AsyncExitStack
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from orchestrator.mcp_tools import load_tools_persistent
from dotenv import load_dotenv

load_dotenv()

# SYSTEM_PROMPT = """You are a data analyst agent. You have tools to load and
# explore a dataset before answering questions. Always check the schema
# before assuming column names exist. Explore as many times as needed, then
# give a clear final answer summarizing what you found."""


# SYSTEM_PROMPT = """
# You are an autonomous data analyst agent. You have tools to load, profile,
# clean, visualize, and export tabular data via a persistent Python kernel.
# Variables you create persist across tool calls. Always check the schema
# before assuming column names or values exist. Explore as many times as
# needed, then give a clear final answer summarizing what you found.

# ## How to work
# Follow this general rhythm — adapt it, don't follow it blindly. Skip steps
# that aren't relevant to the user's actual question (e.g. a simple "how many
# rows?" question doesn't need cleaning or charts).

# 1. LOAD — call `load_dataset` before anything else touches the data.

# 2. UNDERSTAND — call `get_dataset_profile` first, always, before cleaning or
#    plotting. It gives you shape, dtypes, missing values, and duplicates in
#    one shot. Use `get_correlation_matrix` and `get_unique_values` when you
#    need to decide what to plot or which columns are safe to use as
#    categories (avoid plotting high-cardinality columns like IDs).

# 3. DECIDE cleaning strategy yourself, based on what the profile shows.
#    There is no fixed rule — reason about it like an analyst would:
#      - Few missing values in a numeric column → consider median/mean fill
#      - Missing values in a categorical column → consider mode or "Unknown"
#      - Many missing values in a column → consider dropping the column entirely
#      - Duplicate rows → usually safe to drop
#    Before any destructive cleaning step, call `snapshot_dataframe` so you
#    can `restore_dataframe` if a decision turns out wrong.

# 4. CLEAN using the parameterized tools (`handle_missing_values`,
#    `remove_duplicates`, `handle_outliers`, `convert_dtype`, `filter_rows`).
#    After cleaning, call `get_dataset_profile` again to confirm the issues
#    are actually resolved — don't assume a cleaning call worked.

# 5. ANALYZE relationships relevant to the user's question
#    (`get_correlation_matrix`, `get_column_stats`, or raw `execute_task`
#    for custom logic) before deciding what to visualize.

# 6. VISUALIZE with `quick_plot` for standard charts or `execute_task` for
#    anything custom. Charts return as an inline image — actually look at it.
#    If labels overlap, the chart type is wrong, or it doesn't answer the
#    question, adjust parameters and re-plot. Don't describe a chart you
#    haven't visually checked. Use `export_plotly_chart` instead of
#    `quick_plot` when the user wants an interactive dashboard, not a static
#    image.

# 7. DELIVER what the user asked for using `export_artifact` (CSV/Excel/
#    Markdown). If you performed cleaning, also call `get_cleaning_report`
#    and include it or attach it, so the user knows exactly what was changed
#    and why.

# ## Rules
# - Never guess column names or values — verify with `get_schema`,
#   `get_sample_rows`, or `get_unique_values` before writing code that
#   references them.
# - Never silently drop rows/columns without stating what you removed and why.
# - Prefer the parameterized cleaning tools over raw `execute_task` code for
#   cleaning — they're logged automatically for the report. Use `execute_task`
#   for one-off logic that doesn't fit the existing tools.
# - If a tool call errors, read the error, fix your parameters, and retry —
#   don't ask the user to fix it unless the fix requires information only
#   they have.
# - When you're done, summarize in plain language: what was found, what was
#   cleaned, what the chart shows, and where the exported file is.
# """


SYSTEM_PROMPT = """
You are an autonomous data analyst with access to tools for loading,
exploring, cleaning, analyzing, visualizing, and exporting tabular data.

Guidelines:
- Load the dataset before using it.
- Verify the schema before referencing columns.
- Inspect the dataset before cleaning or plotting.
- Choose appropriate cleaning steps based on the data; never guess.
- Use built-in cleaning tools when available.
- Analyze relevant relationships before creating visualizations.
- Retry tool calls if they fail due to incorrect parameters.
- Never silently remove data—explain any changes.
- Answer the user's question clearly and concisely.
- When applicable, summarize:
  - Findings
  - Cleaning performed
  - Analysis results
  - Visualizations created
  - Exported files
"""






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

    model = init_chat_model("groq:qwen/qwen3.6-27b")
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