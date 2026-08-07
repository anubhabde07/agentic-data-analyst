from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from orchestrator.kernel_session import KernelSession
import hashlib
import os
import re

mcp = FastMCP("CodeExecution")
kernel = KernelSession()

EXPORT_DIR = "/outputs"

_seen_image_hashes = set()


def execute_code(code : str):
    output = kernel.execute(code)
    if output["error"]:
        return f"ERROR:\n{output['error']}"
    
    parts = []
    if output["stdout"]:
        parts.append(output["stdout"])
    if output["result"]:
        parts.append(output["result"])
        
    content = [parts and "\n".join(parts) or "(no outut)"]
    
    for img_bytes in output.get("images", []):
        content.append(Image(data=img_bytes, format="png"))
        
    return content



def _diff_images(content_list, seen_hashes: set):
    """Filter Image items out of an execute_code-style content list and
    split them into new vs. duplicate based on a hash of their bytes."""
    images = [item for item in content_list if not isinstance(item, str)]

    new_images, duplicate_images, all_hashes = [], [], set()
    for img in images:
        raw = img.data.encode("utf-8") if isinstance(img.data, str) else img.data
        img_hash = hashlib.sha256(raw).hexdigest()
        all_hashes.add(img_hash)
        (duplicate_images if img_hash in seen_hashes else new_images).append(img)

    return new_images, duplicate_images, all_hashes


@mcp.tool
def execute_task(code : str):
    """Execute Python/pandas code in a persistent sandboxed kernel.
    Variables and imports persist across calls. Returns stdout and the
    result, or 'ERROR:' followed by a traceback if the code raised."""
        
    return execute_code(code)



@mcp.tool()
def load_dataset(path: str, var_name: str = "df"):
    """Load a CSV into the kernel as a pandas DataFrame under var_name.
    Call this once before using get_schema/get_sample_rows/get_column_stats."""
    # code = f"{var_name} = pd.read_csv('{path}')\nprint('Loaded', len({var_name}), 'rows')"
    
    path = path.strip('"').strip("'")

    
    code = f"""
    import pandas as pd

    {var_name} = pd.read_csv(r"{path!r}")
    print("Loaded", len({var_name}), "rows")
    """
            
    return execute_code(code)



@mcp.tool()
def get_dataset_profile(var_name: str = "df"):
    """One-shot health report: shape, dtypes, missing values,
    duplicates, sample rows, and per-column summaries."""
    code = f"""
import pandas as pd

df_ = {var_name}
print("Shape:", df_.shape)
print("Memory usage:", round(df_.memory_usage(deep=True).sum() / 1e6, 2), "MB")
print("Duplicate rows:", df_.duplicated().sum())
print()
print("--- Missing values ---")
print(df_.isna().sum()[df_.isna().sum() > 0].to_string())
print()
print("--- Preview ---")
print(df_.head(3).to_string())
print()
print("--- Numeric summary ---")
print(df_.describe(include='number').to_string())
print()
print("--- Categorical summary ---")
for col in df_.select_dtypes(include='object').columns:
    print(f"{{col}}:")
    print(df_[col].value_counts().head(5).to_string())
    print()
"""
    return execute_code(code)



@mcp.tool()
def handle_missing_values(column: str, strategy: str = "median", var_name: str = "df"):
    """Fill or drop missing values in a column.
    strategy: 'mean', 'median', 'mode', 'drop_rows', or a literal value like 'Unknown'."""
    if strategy == "drop_rows":
        code = f"""
before = len({var_name})
{var_name} = {var_name}.dropna(subset=[{column!r}])
print(f"Dropped {{before - len({var_name})}} rows with missing {column!r}")
"""
    elif strategy in ("mean", "median", "mode"):
        code = f"""
fill_val = {var_name}[{column!r}].{strategy}()
if hasattr(fill_val, 'iloc'):  # mode returns a Series
    fill_val = fill_val.iloc[0]
n_missing = {var_name}[{column!r}].isna().sum()
{var_name}[{column!r}] = {var_name}[{column!r}].fillna(fill_val)
print(f"Filled {{n_missing}} missing values in {column!r} with {{fill_val}}")
"""
    else:
        code = f"""
n_missing = {var_name}[{column!r}].isna().sum()
{var_name}[{column!r}] = {var_name}[{column!r}].fillna({strategy!r})
print(f"Filled {{n_missing}} missing values in {column!r} with {strategy!r}")
"""
    return execute_code(code)



@mcp.tool()
def remove_duplicates(subset: list[str] = None, var_name: str = "df"):
    """Remove duplicate rows. subset=None checks all columns; or pass specific columns."""
    subset_arg = f"subset={subset!r}" if subset else ""
    code = f"""
before = len({var_name})
{var_name} = {var_name}.drop_duplicates({subset_arg})
print(f"Removed {{before - len({var_name})}} duplicate rows")
"""
    return execute_code(code)


@mcp.tool()
def handle_outliers(column: str, method: str = "iqr", action: str = "clip", var_name: str = "df"):
    """Detect/handle outliers in a numeric column.
    method: 'iqr' or 'zscore'. action: 'clip' (cap values) or 'remove' (drop rows)."""
    code = f"""
col = {var_name}[{column!r}]
if {method!r} == "iqr":
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
else:
    mean, std = col.mean(), col.std()
    lower, upper = mean - 3*std, mean + 3*std

n_outliers = ((col < lower) | (col > upper)).sum()
if {action!r} == "clip":
    {var_name}[{column!r}] = col.clip(lower, upper)
    print(f"Clipped {{n_outliers}} outliers in {column!r} to range [{{lower:.2f}}, {{upper:.2f}}]")
else:
    {var_name} = {var_name}[(col >= lower) & (col <= upper)]
    print(f"Removed {{n_outliers}} outlier rows from {column!r}")
"""
    return execute_code(code)




@mcp.tool()
def convert_dtype(column: str, target_type: str, var_name: str = "df"):
    """Convert a column's type: 'int', 'float', 'str', 'datetime', 'category'."""
    code = f"""
try:
    if {target_type!r} == "datetime":
        {var_name}[{column!r}] = pd.to_datetime({var_name}[{column!r}])
    else:
        {var_name}[{column!r}] = {var_name}[{column!r}].astype({target_type!r})
    print(f"Converted {column!r} to {{ {target_type!r} }}")
except Exception as e:
    print(f"Conversion failed: {{e}}")
"""
    return execute_code(code)


@mcp.tool()
def filter_rows(condition: str, var_name: str = "df"):
    """Filter dataframe using a pandas query condition string,
    e.g. condition="age > 18 and income < 100000" """
    code = f"""
before = len({var_name})
{var_name} = {var_name}.query({condition!r})
print(f"Filtered from {{before}} to {{len({var_name})}} rows")
"""
    return execute_code(code)


@mcp.tool()
def snapshot_dataframe(var_name: str = "df", snapshot_name: str = "checkpoint"):
    """Save a copy of the current dataframe state so it can be restored later."""
    return execute_code(f"_{snapshot_name} = {var_name}.copy()\nprint('Snapshot saved: {snapshot_name}')")


@mcp.tool()
def restore_dataframe(var_name: str = "df", snapshot_name: str = "checkpoint"):
    """Restore a previously saved dataframe snapshot."""
    return execute_code(f"{var_name} = _{snapshot_name}.copy()\nprint('Restored from snapshot: {snapshot_name}')")



@mcp.tool()
def get_correlation_matrix(var_name: str = "df", top_n: int = 10):
    """Return the top absolute correlations between numeric columns."""

    code = f"""
import pandas as pd
import numpy as np

corr = {var_name}.corr(numeric_only=True)

mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)

pairs = (
    corr.where(mask)
        .stack()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head({top_n})
)

for (a, b), value in pairs.items():
    print(f"{{a}} <-> {{b}}: {{value:.3f}}")
"""
    return execute_code(code)




@mcp.tool
def quick_plot(
    chart_type: str,
    x: str,
    y: str = None,
    hue: str = None,
    agg: str = None,
    bins: int = 30,
    title: str = "",
    x_label: str = None,
    y_label: str = None,
    figsize: tuple[float, float] = (8, 5),
):
    # """Generate a standard chart (bar, line, scatter, histogram, box, pie,
    # heatmap) from the currently loaded dataframe `df` and return it as an
    # inline image. Runs in the same persistent kernel as execute_task, so it
    # sees whatever df/columns/imports already exist. Use get_dataset_profile
    # or get_unique_values first to confirm column names/types are appropriate
    # for the chosen chart_type. Returns the rendered image, or 'ERROR:'
    # followed by a traceback if the code raised.
    # """
    
    """Generate a chart from the current dataframe `df` and return it as an
inline image. Supports bar, line, scatter, histogram, box, pie, and
heatmap. Uses the existing Python session. Returns the image or
`ERROR:` with a traceback on failure.
"""

    code = f"""
import matplotlib.pyplot as plt

_chart_type = {chart_type!r}
_x = {x!r}
_y = {y!r}
_hue = {hue!r}
_agg = {agg!r}
_bins = {bins!r}
_title = {title!r}
_x_label = {x_label!r}
_y_label = {y_label!r}
_figsize = {figsize!r}

_plot_df = df.copy()

# Optional aggregation (e.g. mean revenue per category)
if _agg and _y:
    group_cols = [_x] + ([_hue] if _hue else [])
    _plot_df = _plot_df.groupby(group_cols, as_index=False)[_y].agg(_agg)

fig, ax = plt.subplots(figsize=_figsize)

if _chart_type == "bar":
    if _hue:
        _plot_df.pivot(index=_x, columns=_hue, values=_y).plot(kind="bar", ax=ax)
    else:
        ax.bar(_plot_df[_x], _plot_df[_y])

elif _chart_type == "line":
    if _hue:
        for key, grp in _plot_df.groupby(_hue):
            ax.plot(grp[_x], grp[_y], label=str(key))
        ax.legend(title=_hue)
    else:
        ax.plot(_plot_df[_x], _plot_df[_y])

elif _chart_type == "scatter":
    if _hue:
        for key, grp in _plot_df.groupby(_hue):
            ax.scatter(grp[_x], grp[_y], label=str(key), alpha=0.7)
        ax.legend(title=_hue)
    else:
        ax.scatter(_plot_df[_x], _plot_df[_y], alpha=0.7)

elif _chart_type == "histogram":
    ax.hist(_plot_df[_x].dropna(), bins=_bins)

elif _chart_type == "box":
    if _hue:
        _plot_df.boxplot(column=_y, by=_hue, ax=ax)
    else:
        ax.boxplot(_plot_df[_x].dropna())

elif _chart_type == "pie":
    _counts = _plot_df.groupby(_x)[_y].sum() if _y else _plot_df[_x].value_counts()
    ax.pie(_counts, labels=_counts.index, autopct="%1.1f%%")

elif _chart_type == "heatmap":
    _corr = _plot_df.corr(numeric_only=True)
    im = ax.imshow(_corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(_corr.columns)))
    ax.set_yticks(range(len(_corr.columns)))
    ax.set_xticklabels(_corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(_corr.columns)
    fig.colorbar(im, ax=ax)

else:
    raise ValueError(f"Unsupported chart_type: {{_chart_type}}")

ax.set_title(_title)
ax.set_xlabel(_x_label or _x)
if _y_label or _y:
    ax.set_ylabel(_y_label or _y)

plt.tight_layout()
plt.show()
plt.close(fig)
"""

    result = execute_code(code)

    # If execute_code errored, pass the error straight through.
    if isinstance(result, str) and result.startswith("ERROR:"):
        return result

    global _seen_image_hashes
    new_images, duplicate_images, all_hashes = _diff_images(result, _seen_image_hashes)
    _seen_image_hashes |= all_hashes

    if duplicate_images and not new_images:
        # Nothing new came back — likely a leaked/stale figure, not this call's plot.
        return "ERROR:\nquick_plot produced no new image (possible stale/leaked figure from kernel state)."

    # Reassemble: keep text output, drop confirmed duplicate images, keep new ones.
    text_parts = [item for item in result if isinstance(item, str)]
    return text_parts + new_images




@mcp.tool
def export_artifact(
    code_to_generate_df: str,
    file_format: str,
    output_filename: str,
):
    # """Export a derived dataframe (or subset of the current df) to a file
    # on disk and return its path. Runs in the same persistent kernel as
    # execute_task/quick_plot, so it can reference df and any other variables
    # already in scope. Use this for the final DELIVER step once cleaning
    # and analysis are done.

    # Args:
    #     code_to_generate_df: a single Python expression that evaluates to
    #         the dataframe to export, e.g. "df[df.revenue > 1000]" or just
    #         "df" to export the whole current dataframe. Must be an
    #         expression, not a multi-line script.
    #     file_format: one of "csv", "excel", "markdown", "json".
    #     output_filename: desired filename, e.g. "high_value_sales.csv".
    #         Extension is enforced to match file_format regardless of what
    #         is passed in.
    # """
    
    
    """Export a dataframe to CSV, Excel, Markdown, or JSON and return the
saved file path.

Args:
    code_to_generate_df: Python expression evaluating to the dataframe
        to export (e.g. "df" or "df[df.sales > 1000]").
    file_format: "csv", "excel", "markdown", or "json".
    output_filename: Output filename. Extension is set automatically.
"""


    valid_formats = {"csv", "excel", "markdown", "json"}
    if file_format not in valid_formats:
        return f"ERROR:\nUnsupported file_format '{file_format}'. Must be one of {valid_formats}."

    ext_map = {"csv": ".csv", "excel": ".xlsx", "markdown": ".md", "json": ".json"}
    save_fn_map = {
        "csv": "to_csv({path!r}, index=False)",
        "excel": "to_excel({path!r}, index=False)",
        "markdown": "to_markdown({path!r}, index=False)",
        "json": "to_json({path!r}, orient='records', indent=2)",
    }

    # Enforce correct extension regardless of what filename was passed in
    base_name = re.sub(r"\.[^.]+$", "", output_filename)
    safe_name = base_name + ext_map[file_format]
    full_path = os.path.join(EXPORT_DIR, safe_name)

    save_call = save_fn_map[file_format].format(path=full_path)

    code = f"""
import os

_export_df = ({code_to_generate_df})

if not hasattr(_export_df, "to_csv"):
    raise TypeError(
        "code_to_generate_df must evaluate to a pandas DataFrame, "
        f"got {{type(_export_df)}} instead."
    )

os.makedirs({EXPORT_DIR!r}, exist_ok=True)
_export_df.{save_call}
_export_row_count = len(_export_df)
_export_path = {full_path!r}
print(f"Exported {{_export_row_count}} rows to {{_export_path}}")
"""

    result = execute_code(code)

    if isinstance(result, str) and result.startswith("ERROR:"):
        return result

    # Confirm the file actually landed on disk before claiming success
    if not os.path.exists(full_path):
        return f"ERROR:\nExport code ran but no file was found at {full_path}."

    return {
        "path": full_path,
        "filename": safe_name,
        "format": file_format,
        "message": f"Saved to {full_path}",
    }






if __name__ == "__main__":
    mcp.run(transport="stdio")