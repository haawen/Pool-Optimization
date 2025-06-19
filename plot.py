import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

print(
    "Make sure to have run flops and profiling before plotting. Flops needs only be run when flop count changes."
)

# dont put dark colors
color_styles = {
    "Default": "o",
    "Basic Implementation": "blue",
    "Code Motion": "green",
    "Less SQRT": "purple",
    "Less SQRT 2": "orchid",
    "Branch Pred": "grey",
    "Removed Unused Branches": "#aec6cf",
    "SIMD": "orange",
    "Precompute": "gold",
    "Scalar Improvements": "red",
    "Scalar Less SQRT": "firebrick",
    "scalar Less SQRT + Approx": "Cyan",
    "Approx + Symmetry": "DarkOrange",
    "Reciprocal Sqrt": "DarkOrange",
    "Reciprocal Sqrt IF": "black",
    "Reciprocal Sqrt Less IF": "chocolate",
    "Full SIMD": "DarkOliveGreen",
    "SIMD scalar loop": "DarkGoldenRod",
    "SIMD Optimized Impulse": "DarkViolet",
    "Improved Symmetry": "DarkSeaGreen",
    "Register Relieve": "DarkSlateGray",
    "Reciprocal Sqrt Hoist": "Yellow",
    "SIMD SSD": "DarkTurquoise",
}


# Make sure the output directory exists
os.makedirs("plots", exist_ok=True)

# Read and preprocess
df = pd.read_csv("build/benchmark.csv")
df["Test Case"] = "TC " + df["Test Case"].astype(str)

# Compute average cycles per (Function, Test Case)
grouped = df.groupby(["Function", "Test Case"])[["Cycles"]].mean().reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 2) READ & PREPROCESS THE BENCHMARK.CSV
# ─────────────────────────────────────────────────────────────────────────────
#    Columns of build/benchmark.csv:
#      Function, Test Case, Iteration, Cycles
#
bench_df = pd.read_csv("build/benchmark.csv")
bench_df["Test Case"] = "TC " + bench_df["Test Case"].astype(str)

# Compute mean cycles per (Function, Test Case) for benchmark data:
bench_grouped = (
    bench_df.groupby(["Function", "Test Case"])[["Cycles"]].mean().reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 3) READ & PREPROCESS THE ORIGINAL.CSV
# ─────────────────────────────────────────────────────────────────────────────
#    original.csv has no header, but each line is:
#      Original, <Test Case>, <Iteration>, <Cycles>
#    Example:
#      Original,0,0,246679
#      Original,0,1,181633
#      Original,0,2,176453
#
orig_df = pd.read_csv(
    "build/original.csv",
    header=None,
    names=["Function", "Test Case", "Iteration", "Cycles"],
)
# All rows in this file have Function == "Original"
orig_df["Test Case"] = "TC " + orig_df["Test Case"].astype(str)

# Compute mean cycles per (Function="Original", Test Case) for original data:
orig_grouped = (
    orig_df.groupby(["Function", "Test Case"])[["Cycles"]].mean().reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) CONCATENATE BENCHMARK + ORIGINAL GROUPS
# ─────────────────────────────────────────────────────────────────────────────
#    Now we have two DataFrames with columns ["Function","Test Case","Cycles"].
#    We'll stack them so every plot includes "Original" alongside the other functions.
#
grouped_all = pd.concat([bench_grouped, orig_grouped], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5) BUILD A CONSISTENT COLOR MAP FOR ALL FUNCTIONS (INCLUDING "Original")
# ─────────────────────────────────────────────────────────────────────────────
all_funcs = sorted(grouped_all["Function"].unique())
num_funcs = len(all_funcs)

# Choose a categorical colormap. Here we pick "tab10" (10 distinct colors).
# If you have >10 functions, it will cycle through.
cmap = plt.get_cmap("tab10")

func_to_color = {func: cmap(i % cmap.N) for i, func in enumerate(all_funcs)}

# (Optional) Print to verify:
# for func, color in func_to_color.items():
#     print(func, "→", color)


# ─────────────────────────────────────────────────────────────────────────────
# 6) DEFINE AN X‐AXIS FORMATTER TO SHOW THOUSANDS AS “k”
# ─────────────────────────────────────────────────────────────────────────────
def thousands_formatter(x, pos):
    """
    Convert a raw number (e.g. 50000) into a string like "50 k" or "75.4 k".
    This will be used by FuncFormatter.
    """
    val = x / 1000.0
    if val.is_integer():
        return f"{int(val)} k"
    else:
        return f"{val:.1f} k"


# ─────────────────────────────────────────────────────────────────────────────
# 7) PLOT ONE HORIZONTAL‐BAR CHART PER TEST CASE
# ─────────────────────────────────────────────────────────────────────────────
#    This loop generates:
#      plots/benchmark_TC_0_horizontal.png
#      plots/benchmark_TC_1_horizontal.png
#      ...
#      plots/benchmark_TC_4_horizontal.png
#
#  a) Find all unique test‐cases, sorted by their numeric suffix (0,1,2,3,4).
all_tcs = sorted(grouped_all["Test Case"].unique(), key=lambda s: int(s.split()[-1]))

plt.rcParams.update({"figure.autolayout": True})  # auto‐tight layout

for tc in all_tcs:
    # b) Filter to exactly this test case
    subset = grouped_all[grouped_all["Test Case"] == tc].copy()
    # c) Sort by descending cycles (slowest first)
    subset_sorted = subset.sort_values("Cycles", ascending=False).reset_index(drop=True)

    functions = subset_sorted["Function"].tolist()
    cycles_vals = subset_sorted["Cycles"].tolist()
    y_pos = np.arange(len(functions))

    # d) Make a wider figure so labels never clip
    fig_width = 8
    fig_height = len(functions) * 0.4 + 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # e) Draw one horizontal bar per function, using its assigned color
    for i, func in enumerate(functions):
        ax.barh(y_pos[i], cycles_vals[i], color=func_to_color[func], edgecolor="black")

    # f) Expand the x‐limit to leave room for the right‐hand annotations
    max_cycles = max(cycles_vals)
    ax.set_xlim(0, max_cycles * 1.15)

    # g) Annotate each bar with the integer cycle count
    for i, val in enumerate(cycles_vals):
        ax.text(
            val * 1.005,  # slightly to the right of the bar
            i,
            f"{int(round(val, 0))}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # h) Flip y‐axis so the slowest bar is at the top
    ax.set_yticks(y_pos)
    ax.set_yticklabels(functions, fontsize=9)
    ax.invert_yaxis()

    # i) Format the x‐axis in “k” with only ~5 ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    # j) Titles, labels, grid
    ax.set_xlabel("Cycles", fontsize=10)
    ax.set_title(f"Average Cycles — {tc}", fontsize=12)
    ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    # k) Save and close
    outfile = f"plots/benchmark_{tc.replace(' ', '_')}_horizontal.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 8) PLOT “AVERAGE ACROSS ALL TEST CASES” (INCLUDING “Original”)
# ─────────────────────────────────────────────────────────────────────────────
#    This generates: plots/benchmark_avg_all_cases_horizontal.png
#
avg_all = grouped_all.groupby("Function")[["Cycles"]].mean().reset_index()
avg_all_sorted = avg_all.sort_values("Cycles", ascending=False).reset_index(drop=True)

functions = avg_all_sorted["Function"].tolist()
cycles_vals = avg_all_sorted["Cycles"].tolist()
y_pos = np.arange(len(functions))

fig_width = 9.5
fig_height = len(functions) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(functions):
    ax.barh(y_pos[i], cycles_vals[i], color=func_to_color[func], edgecolor="black")

max_cycles = max(cycles_vals)
ax.set_xlim(0, max_cycles * 1.15)

for i, val in enumerate(cycles_vals):
    ax.text(
        val * 1.005 + 500,
        i,
        f"{int(round(val, 0))}",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )


ax.set_yticks(y_pos)
ax.set_yticklabels(
    functions,
    fontsize=14,
    fontweight="bold",
)
ax.invert_yaxis()

ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

ax.set_title("Mean Cycles Across 5 Initial Conditions", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/benchmark_avg_all_cases_horizontal.png", dpi=150)
plt.close(fig)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)


# ─────────────────────────────────────────────────────────────────────────────
#  X) COST & FLOPS: HORIZONTAL BAR IMPROVEMENTS
# ─────────────────────────────────────────────────────────────────────────────
flops = pd.read_csv("build/flops.csv")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# (If you haven’t already read flops earlier, uncomment the next line:)
# flops = pd.read_csv("build/flops.csv")

# You already have `color_styles` defined. We’ll use `section_colors` and
# `op_colors` from your existing code for the new horizontal plots.

op_colors = {
    "ADDS": "green",
    "MULS": "blue",
    "DIVS": "orange",
    "SQRT": "red",  # changed to lowercase “red” for consistency
}

section_colors = {
    "Initialization": "green",
    "Impulse": "blue",
    "Delta": "orange",
    "Velocity": "purple",
    "Transform to World Frame": "red",
    "collide_balls": "black",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: format x-axis ticks in “k” (thousands)
# ─────────────────────────────────────────────────────────────────────────────
def thousands_formatter(x, pos):
    """
    Turn a raw number like 50000 → “50 k”, 75432 → “75.4 k”, etc.
    Used by FuncFormatter.
    """
    val = x / 1000.0
    if val.is_integer():
        return f"{int(val)} k"
    else:
        return f"{val:.1f} k"


# ─────────────────────────────────────────────────────────────────────────────
#  (1) COST PER OPERATION — Horizontal Bars, Thicker
# ─────────────────────────────────────────────────────────────────────────────
section_colors = {
    "Initialization": "green",
    "Impulse": "blue",
    "Delta": "orange",
    "Velocity": "purple",
    "Transform to World Frame": "red",
    "collide_balls": "black",
}


def thousands_formatter(x, pos):
    """
    Convert 50000 → '50 k', 75432 → '75.4 k'.
    Used by FuncFormatter.
    """
    val = x / 1000.0
    if val.is_integer():
        return f"{int(val)} k"
    else:
        return f"{val:.1f} k"


for op in ["ADDS", "MULS", "DIVS", "SQRT"]:
    # 1) Compute mean operation count per (Function, Section)
    cost_grouped = flops.groupby(["Function", "Section"])[[op]].mean().reset_index()
    pivot = cost_grouped.pivot(index="Function", columns="Section", values=op).fillna(0)

    functions = pivot.index.tolist()  # e.g. ["Basic Implementation", "Code Motion", …]
    sections = pivot.columns.tolist()  # e.g. ["Initialization", "Impulse", "Delta", …]

    N_funcs = len(functions)
    N_secs = len(sections)

    # 2) Decide how tall each cluster is, and how big a gap to leave BETWEEN clusters:
    cluster_height = 1.0  # inches of “height” per function‐cluster
    bar_height = cluster_height / N_secs
    gap_between = 1.0  # inches of blank whitespace BETWEEN clusters

    # 3) We’ll treat each “cluster center” as at y = i * (cluster_height + gap_between)
    #    Then drawing uses these y-coordinates directly (in data units).
    cluster_spacing = cluster_height + gap_between
    y_centers = np.arange(N_funcs) * cluster_spacing

    # 4) Compute figure height: we need to allow up to the last cluster center + half cluster_height,
    #    plus a little extra margin at top.  (We work in inches here, since figsize is in inches.)
    fig_width = 10
    fig_height = N_funcs * cluster_spacing + 0.5  # extra 0.5" top margin
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 5) Draw bars for each section at the appropriate offset within each cluster:
    for sec_idx, sec in enumerate(sections):
        counts = pivot[sec].values  # length = N_funcs
        # Each bar’s y-position = cluster_center + offset
        y_positions = (
            y_centers + sec_idx * bar_height - (cluster_height / 2) + (bar_height / 2)
        )

        ax.barh(
            y_positions,
            counts,
            height=bar_height * 0.9,  # leave a 10% vertical gap between sub-bars
            color=section_colors.get(sec, "gray"),
            edgecolor="black",
            label=sec,
        )

    # 6) Draw faint horizontal divider lines (in that gap_between) at halfway between clusters:
    for i in range(N_funcs - 1):
        # Divider sits midway between y_centers[i] + (cluster_height/2) and y_centers[i+1] - (cluster_height/2).
        # That midpoint is y_centers[i] + (cluster_height/2) + (gap_between/2).
        divider_y = y_centers[i] + (cluster_height / 2) + (gap_between / 2)
        ax.axhline(divider_y, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # 7) Expand x‐limit so annotations never run off
    max_count = pivot.values.max()
    ax.set_xlim(0, max_count * 1.15)

    # 8) Annotate every bar—if count > 0, print near its right; if count = 0, print “0” near left edge:
    for func_idx, func in enumerate(functions):
        for sec_idx, sec in enumerate(sections):
            val = pivot.loc[func, sec]

            if val > 0:
                x_pos = val * 1.005
            else:
                # Place “0” just inside the left margin (5% of max_count)
                x_pos = max_count * 0.005

            # y‐position for this particular sub‐bar
            y_pos = (
                y_centers[func_idx]
                + sec_idx * bar_height
                - (cluster_height / 2)
                + (bar_height / 2)
            )

            ax.text(
                x_pos,
                y_pos,
                f"{int(round(val, 0))}",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )

    # 9) Set y‐ticks at the cluster centers, label them with function names, and invert so top = largest:
    ax.set_yticks(y_centers)
    ax.set_yticklabels(functions, fontsize=10)
    ax.invert_yaxis()

    # 10) Format x‐axis ticks as “k” with at most 5 major ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    # 11) Labels, title, grid, and legend (legend outside)
    ax.set_xlabel(f"{op} Count", fontsize=12)
    ax.set_title(f"Cost Evaluation — {op}", fontsize=14)
    ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.legend(
        title="Section",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=12,
        frameon=False,
    )

    # 12) Save and close
    outfile = f"plots/Cost_{op}_horizontal.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  (2) COST PER FUNCTION — Horizontal Bars, Thicker
# ─────────────────────────────────────────────────────────────────────────────


op_colors = {
    "ADDS": "green",
    "MULS": "blue",
    "DIVS": "orange",
    "SQRT": "red",
}


def thousands_formatter(x, pos):
    val = x / 1000.0
    if val.is_integer():
        return f"{int(val)} k"
    else:
        return f"{val:.1f} k"


for func, data in flops.groupby("Function"):
    # 1) Compute average ADDS/MULS/DIVS/SQRT per Section for this function
    groups = (
        data.groupby("Section")[["ADDS", "MULS", "DIVS", "SQRT"]].mean().reset_index()
    )
    # Separate out the “collide_balls” row (to call it “Total”)
    total_row = groups[groups["Section"] == "collide_balls"].copy()
    sections_except_cb = groups[groups["Section"] != "collide_balls"].copy()

    # Rename “collide_balls” → “Total” in that one‐row DataFrame
    if not total_row.empty:
        total_row.loc[:, "Section"] = "Total"

    # --------------------------------------------------------------------------------
    # A) PLOT 1: All sections except “collide_balls”
    # --------------------------------------------------------------------------------
    if not sections_except_cb.empty:
        sections = sections_except_cb["Section"].tolist()
        operations = ["ADDS", "MULS", "DIVS", "SQRT"]
        N_secs = len(sections)
        N_ops = len(operations)

        # Thicken cluster + a small gap between section clusters
        cluster_height = 1.0
        bar_height = cluster_height / N_ops
        gap_between = 1.0  # inches of whitespace between clusters

        cluster_spacing = cluster_height + gap_between
        y_centers = np.arange(N_secs) * cluster_spacing

        # Compute figure height (plus a small margin)
        fig_width = 8
        fig_height = N_secs * cluster_spacing + 0.5
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Draw bars for each operation within each section cluster
        for op_idx, op in enumerate(operations):
            vals = sections_except_cb[op].values  # length = N_secs
            y_positions = (
                y_centers
                + op_idx * bar_height
                - (cluster_height / 2)
                + (bar_height / 2)
            )
            ax.barh(
                y_positions,
                vals,
                height=bar_height * 0.9,  # small 10% gap between sub-bars
                color=op_colors.get(op, "gray"),
                edgecolor="black",
                label=op,
            )

        # Draw dashed dividers between section clusters
        for i in range(N_secs - 1):
            divider_y = y_centers[i] + (cluster_height / 2) + (gap_between / 2)
            ax.axhline(
                divider_y, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
            )

        # Expand x‐limit so “0” labels fit
        max_count = sections_except_cb[operations].values.max()
        ax.set_xlim(0, max_count * 1.15)

        # Annotate every bar (including zeros)
        for sec_idx, sec in enumerate(sections):
            for op_idx, op in enumerate(operations):
                val = float(sections_except_cb.loc[sec_idx, op])
                if val > 0:
                    x_pos = val * 1.005
                else:
                    x_pos = max_count * 0.005
                y_pos = (
                    y_centers[sec_idx]
                    + op_idx * bar_height
                    - (cluster_height / 2)
                    + (bar_height / 2)
                )
                ax.text(
                    x_pos,
                    y_pos,
                    f"{int(round(val, 0))}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="black",
                )

        # Flip y-axis, label sections (e.g. “Initialization”, “Impulse”…)
        ax.set_yticks(y_centers)
        ax.set_yticklabels(sections, fontsize=10)
        ax.invert_yaxis()

        # Format x-axis in “k”
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

        # Titles, legend, grid
        ax.set_xlabel("Operation Count", fontsize=12)
        ax.set_title(f"Cost Evaluation for {func}", fontsize=14)
        ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

        ax.legend(
            title="Operation",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=10,
            title_fontsize=12,
            frameon=False,
        )

        # Save figure (excluding “collide_balls”)
        safe_name = func.replace(" ", "_")
        outfile = f"plots/Cost_{safe_name}_sections_horizontal.png"
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close(fig)

    # --------------------------------------------------------------------------------
    # B) PLOT 2: The “Total” (formerly “collide_balls”) as its own plot
    # --------------------------------------------------------------------------------
    # Assume `total_row` is a one‐row DataFrame where:
    #   total_row["Section"] == "collide_balls"
    # and you’ve already renamed that row’s section to "Total".
    # Also assume `operations = ["ADDS", "MULS", "DIVS", "SQRT"]` and `op_colors` exist.

    operations = ["ADDS", "MULS", "DIVS", "SQRT"]

    # 1) We only have one “section” cluster (Total), so cluster_center is at y=0.
    y_center = 0.0

    # 2) Thicken the group: increase cluster_height, leaving plenty of space between bars.
    cluster_height = 1.6  # each operation‐bar cluster is now 1.6" tall
    bar_height = cluster_height / len(operations)  # = 0.4" per bar

    # 3) Compute figure size: height = cluster_height + small margin
    fig_width = 8
    fig_height = cluster_height + 0.5  # add 0.5" of top/bottom padding
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    gap = 0.05
    # 4) Draw one horizontal bar per operation, vertically offset so they do not overlap
    for op_idx, op in enumerate(operations):
        val = float(total_row.iloc[0][op])  # the single value for this op

        # Compute y-position for this bar:
        y_pos = (
            y_center
            + op_idx * bar_height
            - (cluster_height / 2)
            + (bar_height / 2)
            + op_idx * gap
        )

        ax.barh(
            y_pos,
            val,
            height=bar_height * 0.9,  # 90% of bar_height gives a small gap between bars
            color=op_colors.get(op, "gray"),
            edgecolor="black",
        )

        # Annotate the bar with its integer value (right‐aligned)
        if val > 0:
            x_pos = val * 1.005
        else:
            x_pos = 0  # or a small epsilon if you want to show "0"

        print(func, x_pos)
        ax.text(
            # 39114.399,
            x_pos + 500.0,
            y_pos + 0.03,
            f"{int(round(val, 0))}",
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    # 5) Label the y-axis with the operation names (vertical positions align with bars)
    y_ticks = [
        y_center
        + op_idx * bar_height
        - (cluster_height / 2)
        + (bar_height / 2)
        + op_idx * gap
        for op_idx in range(len(operations))
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(operations, fontsize=14, fontweight="bold")

    # Invert y so that ADDS appears at the top (or reverse order if you prefer)
    ax.invert_yaxis()

    # 6) Expand x‐limit so annotations never run off
    max_val = float(total_row[operations].values.max())
    ax.set_xlim(0, max_val * 1.15)

    # 7) Remove the legend entirely:
    #    (We never passed a 'label' to barh, so no legend is drawn by default.)

    # 8) Format x-axis in “k”
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    # 9) Titles, labels, grid
    # ax.set_xlabel("Operation Count", fontsize=12)
    ax.set_title(f"Total Cost for {func}", fontsize=14)
    ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    # 10) Save and close
    safe_name = func.replace(" ", "_")
    outfile = f"plots/Cost_{safe_name}_Total_horizontal.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)

# 1) Read in flops.csv and profiling.csv, convert both "Test Case" to "TC X"
# -----------------------------------------------------------------------------
flops_df = pd.read_csv("build/flops.csv")
if flops_df["Test Case"].dtype != object:
    flops_df["Test Case"] = "TC " + flops_df["Test Case"].astype(str)

flops_df = flops_df[flops_df["Section"] == "collide_balls"]

prof_df = pd.read_csv("build/benchmark.csv")
if prof_df["Test Case"].dtype != object:
    prof_df["Test Case"] = "TC " + prof_df["Test Case"].astype(str)

# 2) Merge on (Function, Section, Test Case)
# -----------------------------------------------------------------------------
merged = pd.merge(
    flops_df,
    prof_df,
    on=["Function", "Test Case"],
    suffixes=("_flops", "_prof"),
)


# 3) Compute FlopsPerCycle = Flops / Cycles (no "_prof" suffix)
# -----------------------------------------------------------------------------
merged["FlopsPerCycle"] = merged["Flops"] / merged["Cycles"]

# Strip any whitespace from column names
merged.columns = merged.columns.str.strip()

# 4) Group by (Function, Test Case), taking mean of FlopsPerCycle
# -----------------------------------------------------------------------------
gp = merged.groupby(["Function", "Test Case"])[["FlopsPerCycle"]].mean().reset_index()

# 5) Build a consistent color map for all Functions
# -----------------------------------------------------------------------------
all_funcs = sorted(gp["Function"].unique())
cmap = plt.get_cmap("tab10")
func_to_color_fp = {func: cmap(i % cmap.N) for i, func in enumerate(all_funcs)}


# 6) Formatter for "k" on the x-axis
# -----------------------------------------------------------------------------
def thousands_formatter(x, pos):
    val = x / 1000.0
    if val.is_integer():
        return f"{int(val)} k"
    else:
        return f"{val:.1f} k"


# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# 7) Plot horizontal‐bar chart PER Test Case for FlopsPerCycle
# -----------------------------------------------------------------------------
all_tcs_fp = sorted(gp["Test Case"].unique(), key=lambda s: int(s.split()[-1]))

for tc in all_tcs_fp:
    sub = gp[gp["Test Case"] == tc].copy()
    # Sort ascending so that highest FlopsPerCycle ends up at the top after invert_yaxis
    sub_sorted = sub.sort_values("FlopsPerCycle", ascending=True).reset_index(drop=True)

    funcs = sub_sorted["Function"].tolist()
    values = sub_sorted["FlopsPerCycle"].tolist()
    y_pos = np.arange(len(funcs))

    fig_width = 8
    fig_height = len(funcs) * 0.4 + 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw one horizontal bar per function
    for i, func in enumerate(funcs):
        ax.barh(
            y_pos[i],
            values[i],
            color=func_to_color_fp[func],
            edgecolor="black",
            height=0.35,
        )

    max_val = max(values)
    ax.set_xlim(0, max_val * 1.15)

    # Annotate each bar with FlopsPerCycle to two decimal places
    for i, val in enumerate(values):
        ax.text(val * 1.005, i, f"{val:.2f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(funcs, fontsize=9)
    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    ax.set_title(f"Flops Per Cycle — {tc}", fontsize=12)
    ax.set_xlabel("Flops / Cycle", fontsize=10)
    ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

    outfile = f"plots/flops_per_cycle_{tc.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)

# 8) Plot "Average FlopsPerCycle across all Test Cases"
# -----------------------------------------------------------------------------
avg_fp = gp.groupby("Function")[["FlopsPerCycle"]].mean().reset_index()
avg_sorted = avg_fp.sort_values("FlopsPerCycle", ascending=True).reset_index(drop=True)

funcs = avg_sorted["Function"].tolist()
values = avg_sorted["FlopsPerCycle"].tolist()
y_pos = np.arange(len(funcs))

fig_width = 8
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(funcs):
    ax.barh(
        y_pos[i],
        values[i],
        color=func_to_color_fp[func],
        edgecolor="black",
        height=0.8,
    )

max_val = max(values)
ax.set_xlim(0, max_val * 1.15)

for i, val in enumerate(values):
    ax.text(
        val * 1.005,
        i,
        f"{val:.2f}",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(funcs, fontsize=14, fontweight="bold")
ax.invert_yaxis()

# ─── Remove the thousands_formatter ────────────────────────────────────────
# Just let Matplotlib use the default float formatting:
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

# Optionally, you can set a fixed number of decimal ticks manually, e.g.:
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ───────────────────────────────────────────────────────────────────────────
ax.set_title("Mean Flops/Cycle Across 5 Initial Conditions", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/flops_per_cycle_avg_all_cases.png", dpi=150)
plt.close(fig)


################## BRANCH MISS PLOTS ###################

branch_miss_percent = {
    "Initial C": 0.11,  # 438 769
    "Precompute": 0.10,  # 434,184
    "RSQRT": 0.09,  # 347,270
    "Bitwise Masks": 0.21,  # 498,347
    "SIMD Overhaul": 0.34,  # 429,743
    "RSQRT Double While": 0.11,  # 342,229
    "FMA": 0.11,  # 459,489
    "SIMD": 0.43,  # 555,843
}

branch_miss_absolut = {
    "Initial C": 438769,
    "Precompute": 434184,
    "RSQRT": 347270,
    "Bitwise Masks": 498347,
    "SIMD Overhaul": 429743,
    "RSQRT Double While": 342229,
    "FMA": 459489,
    "SIMD": 555843,
}

sorted_funcs_values = sorted(
    branch_miss_absolut.items(), key=lambda item: item[1], reverse=True
)
funcs = [func for func, val in sorted_funcs_values]
values = [(int)(val) for func, val in sorted_funcs_values]
# funcs = branch_miss_absolut.keys()
# values = list(branch_miss_absolut.values())
y_pos = np.arange(len(funcs))

fig_width = 9.4
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(funcs):
    ax.barh(
        y_pos[i],
        values[i],
        color=func_to_color_fp[func],
        edgecolor="black",
        height=0.8,
    )

max_val = max(values)
ax.set_xlim(0, max_val * 1.15)

for i, val in enumerate(values):
    ax.text(
        val * 1.005,
        i,
        f"{val}",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(funcs, fontsize=14, fontweight="bold")
ax.invert_yaxis()

# ─── Remove the thousands_formatter ────────────────────────────────────────
# Just let Matplotlib use the default float formatting:
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

# Optionally, you can set a fixed number of decimal ticks manually, e.g.:
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ───────────────────────────────────────────────────────────────────────────
ax.set_title("Branch Misses For 50k Iterations", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/branch_misses_absolute.png", dpi=150)
plt.close(fig)

sorted_funcs_values = sorted(
    branch_miss_percent.items(), key=lambda item: item[1], reverse=True
)
funcs = [func for func, val in sorted_funcs_values]
values = [val for func, val in sorted_funcs_values]
# funcs = branch_miss_absolut.keys()
# values = list(branch_miss_absolut.values())
y_pos = np.arange(len(funcs))

fig_width = 9.4
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(funcs):
    ax.barh(
        y_pos[i],
        values[i],
        color=func_to_color_fp[func],
        edgecolor="black",
        height=0.8,
    )

max_val = max(values)
ax.set_xlim(0, max_val * 1.15)

for i, val in enumerate(values):
    ax.text(
        val * 1.005,
        i,
        f"{val:.2f}%",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(funcs, fontsize=14, fontweight="bold")
ax.invert_yaxis()

# ─── Remove the thousands_formatter ────────────────────────────────────────
# Just let Matplotlib use the default float formatting:
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

# Optionally, you can set a fixed number of decimal ticks manually, e.g.:
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ───────────────────────────────────────────────────────────────────────────
ax.set_title("Relative Branch Misses For 50k Iterations", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/branch_misses_percent.png", dpi=150)
plt.close(fig)


################## CACHE MISS PLOTS ###################

branch_miss_percent = {
    "Initial C": 15.65,  # 438 769
    "Precompute": 16.66,  # 434,184
    "RSQRT": 17.44,  # 347,270
    "Bitwise Masks": 17.08,  # 498,347
    "SIMD Overhaul": 16.34,  # 429,743
    "RSQRT Double While": 18.53,  # 342,229
    "FMA": 18.80,  # 459,489
    "SIMD": 11.37,  # 555,843
}

branch_miss_absolut = {
    "Initial C": 387859,
    "Precompute": 247851,
    "RSQRT": 276893,
    "Bitwise Masks": 240375,
    "SIMD Overhaul": 226952,
    "RSQRT Double While": 198975,
    "FMA": 287762,
    "SIMD": 317074,
}

sorted_funcs_values = sorted(
    branch_miss_absolut.items(), key=lambda item: item[1], reverse=True
)
funcs = [func for func, val in sorted_funcs_values]
values = [(int)(val) for func, val in sorted_funcs_values]
# funcs = branch_miss_absolut.keys()
# values = list(branch_miss_absolut.values())
y_pos = np.arange(len(funcs))

fig_width = 9.4
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(funcs):
    ax.barh(
        y_pos[i],
        values[i],
        color=func_to_color_fp[func],
        edgecolor="black",
        height=0.8,
    )

max_val = max(values)
ax.set_xlim(0, max_val * 1.15)

for i, val in enumerate(values):
    ax.text(
        val * 1.005,
        i,
        f"{val}",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(funcs, fontsize=14, fontweight="bold")
ax.invert_yaxis()

# ─── Remove the thousands_formatter ────────────────────────────────────────
# Just let Matplotlib use the default float formatting:
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

# Optionally, you can set a fixed number of decimal ticks manually, e.g.:
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ───────────────────────────────────────────────────────────────────────────
ax.set_title("Cache Misses For 50k Iterations", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/cache_misses_absolute.png", dpi=150)
plt.close(fig)

sorted_funcs_values = sorted(
    branch_miss_percent.items(), key=lambda item: item[1], reverse=True
)
funcs = [func for func, val in sorted_funcs_values]
values = [val for func, val in sorted_funcs_values]
# funcs = branch_miss_absolut.keys()
# values = list(branch_miss_absolut.values())
y_pos = np.arange(len(funcs))

fig_width = 9.4
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(funcs):
    ax.barh(
        y_pos[i],
        values[i],
        color=func_to_color_fp[func],
        edgecolor="black",
        height=0.8,
    )

max_val = max(values)
ax.set_xlim(0, max_val * 1.15)

for i, val in enumerate(values):
    ax.text(
        val * 1.005,
        i,
        f"{val:.2f}%",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(funcs, fontsize=14, fontweight="bold")
ax.invert_yaxis()

# ─── Remove the thousands_formatter ────────────────────────────────────────
# Just let Matplotlib use the default float formatting:
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

# Optionally, you can set a fixed number of decimal ticks manually, e.g.:
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ───────────────────────────────────────────────────────────────────────────
ax.set_title("Relative Cache Misses For 50k Iterations", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/cache_misses_percent.png", dpi=150)
plt.close(fig)


######## MUUUUUUUUUUUUUUUUULTIPLE NNNNNNNNNNNNNNNN###############


df = pd.read_csv("build/N.csv")
avg_cycles = df.groupby(["Function", "N"])[["Cycles"]].mean().reset_index()
avg_sorted = avg_cycles.sort_values("Cycles", ascending=True).reset_index(drop=True)

unique_N = sorted(avg_cycles["N"].unique())

fig_width = 9.4
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.yaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

styles = [
    "--",
    "-.",
]

for i, func in enumerate(all_funcs):
    (line,) = ax.plot(
        unique_N,
        avg_cycles[avg_cycles["Function"] == func]["Cycles"],
        color=func_to_color_fp[func],
        linewidth=2.5,
        linestyle=styles[i % len(styles)],
    )

    N_display_manual = {
        "Initial C": 6000,  # 438 769
        "Precompute": 10000,  # 434,184
        "RSQRT": 10000,  # 347,270
        "Bitwise Masks": 10000,  # 498,347
        "SIMD Overhaul": 8000,  # 429,743
        "RSQRT Double While": 8000,  # 342,229
        "FMA": 9000,  # 459,489
        "SIMD": 10000,  # 555,843
    }

    N_display = N_display_manual[
        func
    ]  # unique_N[int(len(unique_N) / len(all_funcs) * i)]

    import matplotlib.patheffects as path_effects

    ax.text(
        x=N_display,  # Last x-value
        y=avg_cycles[(avg_cycles["Function"] == func) & (avg_cycles["N"] == N_display)][
            "Cycles"
        ].values[-1],
        s=func,  # Label text
        color=func_to_color_fp[func],  # Match the line color
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="left",
        path_effects=[
            path_effects.Stroke(linewidth=2, foreground="black"),  # Border
            path_effects.Normal(),  # Restore normal text rendering on top
        ],
    )

plt.tight_layout()
plt.savefig("plots/N.png", dpi=150)
plt.close(fig)


register_pressure = {
    "Initial C": 57.47,
    "Precompute": 57.47,
    "RSQRT": 95.71,
    "Bitwise Masks": 92.74,
    "SIMD Overhaul": 91.35,
    "RSQRT Double While": 95.60,
    "FMA": 46.54,
    "SIMD": 91.75,
}

sorted_funcs_values = sorted(
    register_pressure.items(), key=lambda item: item[1], reverse=False
)
funcs = [func for func, val in sorted_funcs_values]
values = [(val) for func, val in sorted_funcs_values]
# funcs = branch_miss_absolut.keys()
# values = list(branch_miss_absolut.values())
y_pos = np.arange(len(funcs))

fig_width = 9.4
fig_height = len(funcs) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(funcs):
    ax.barh(
        y_pos[i],
        values[i],
        color=func_to_color_fp[func],
        edgecolor="black",
        height=0.8,
    )

max_val = max(values)
ax.set_xlim(0, max_val * 1.15)

for i, val in enumerate(values):
    ax.text(
        val * 1.005,
        i,
        f"{val:.2f}%",
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(funcs, fontsize=14, fontweight="bold")
ax.invert_yaxis()

# ─── Remove the thousands_formatter ────────────────────────────────────────
# Just let Matplotlib use the default float formatting:
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

# Optionally, you can set a fixed number of decimal ticks manually, e.g.:
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ───────────────────────────────────────────────────────────────────────────
ax.set_title("Simulated Resource Pressure By LLVM-MCA", fontsize=14)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/resource_pressure.png", dpi=150)
plt.close(fig)


"""

# TODO: calcualte peak performance based on COST values and operations / ports, then plot
# TODO: Add port info per instruction

roofline = {
    "AMD Ryzen 7 7735U": {
        "No SIMD": {
            "pi": 32,  # This is with cores included, but per core would be / 16?
            "beta": 28.4,
            "ADDS": 0.5,
            "MULS": 0.5,
            "DIVS": 3.5,
            "SQRT": 5.0,
        },
        "SIMD": {
            "pi": 128,
            "beta": 28.4,
            "ADDS": 0.125,
            "MULS": 0.125,
            "DIVS": 0.875,
            "SQRT": 1.25,
        },
    }
}

oi = np.logspace(-2, 2, 1000)

# Plot setup
plt.figure(figsize=(10, 6))

# Plot each roofline
for cpu, configs in roofline.items():
    for label, vals in configs.items():
        pi = vals["pi"]
        beta = vals["beta"]
        y = np.minimum(beta * oi, pi)
        plt.plot(oi, y, label=f"{cpu} - {label}")

# Axis and styling
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Operational Intensity (FLOPs/Byte)")
plt.ylabel("Performance (FLOPs/Cycle)")
plt.title("Roofline Model")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("plots/roofline.png")
plt.close()
"""
