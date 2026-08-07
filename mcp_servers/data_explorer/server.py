from fastmcp import FastMCP
from orchestrator.kernel_session import KernelSession

mcp = FastMCP("CodeExecution")
kernel = KernelSession()
print("Kernel started successfully")

def execute_code(code : str) -> str:
    output = kernel.execute(code)
    if output["error"]:
        return f"ERROR:\n{output['error']}"
    parts = []
    if output["stdout"]:
        parts.append(output["stdout"])
    if output["result"]:
        parts.append(output["result"])
    if output["images"]:
        parts.append(f"[{len(output['images'])} image(s) generated]")
        
    return "\n".join(parts) if parts else "(no output)"




@mcp.tool
def execute_task(code : str) -> str:
    """Execute Python/pandas code in a persistent sandboxed kernel.
    Variables and imports persist across calls. Returns stdout and the
    result, or 'ERROR:' followed by a traceback if the code raised."""
    
    print("=" * 50)
    print("repr(code):")
    print(repr(code))
    print("=" * 50)
        
    return execute_code(code)


@mcp.tool()
def load_dataset(path: str, var_name: str = "df") -> str:
    """Load a CSV into the kernel as a pandas DataFrame under var_name.
    Call this once before using get_schema/get_sample_rows/get_column_stats."""
    # code = f"{var_name} = pd.read_csv('{path}')\nprint('Loaded', len({var_name}), 'rows')"
    
    path = path.strip('"').strip("'")

    
    code = f"""
import pandas as pd

{var_name} = pd.read_csv(r"{path}")
print("Loaded", len({var_name}), "rows")
"""
            
    return execute_code(code)


@mcp.tool()
def get_schema(var_name: str = "df") -> str:
    """Return column names and dtypes for the loaded dataframe."""
    return execute_code(f"print({var_name}.dtypes.to_string())")

@mcp.tool()
def get_sample_rows(var_name: str = "df", n: int = 5) -> str:
    """Return the first n rows of the loaded dataframe."""
    return execute_code(f"print({var_name}.head({n}).to_string())")

@mcp.tool()
def get_column_stats(var_name: str = "df", column: str = "") -> str:
    """Return summary statistics for one column (numeric describe(),
    or top value counts if categorical)."""
    code = f"""
import pandas as pd
col = {var_name}['{column}']
if pd.api.types.is_numeric_dtype(col):
    print(col.describe().to_string())
else:
    print(col.value_counts().head(10).to_string())
"""
    return execute_code(code)





if __name__ == "__main__":
    mcp.run(transport="stdio")
