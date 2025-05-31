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
grouped = (
    df.groupby(["Function", "Test Case"])[["Cycles"]]
      .mean()
      .reset_index()
)


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
    bench_df
    .groupby(["Function", "Test Case"])[["Cycles"]]
    .mean()
    .reset_index()
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
    names=["Function", "Test Case", "Iteration", "Cycles"]
)
# All rows in this file have Function == "Original"
orig_df["Test Case"] = "TC " + orig_df["Test Case"].astype(str)

# Compute mean cycles per (Function="Original", Test Case) for original data:
orig_grouped = (
    orig_df
    .groupby(["Function", "Test Case"])[["Cycles"]]
    .mean()
    .reset_index()
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

func_to_color = {
    func: cmap(i % cmap.N)
    for i, func in enumerate(all_funcs)
}

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

plt.rcParams.update({'figure.autolayout': True})  # auto‐tight layout

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
        ax.barh(
            y_pos[i],
            cycles_vals[i],
            color=func_to_color[func],
            edgecolor="black"
        )

    # f) Expand the x‐limit to leave room for the right‐hand annotations
    max_cycles = max(cycles_vals)
    ax.set_xlim(0, max_cycles * 1.15)

    # g) Annotate each bar with the integer cycle count
    for i, val in enumerate(cycles_vals):
        ax.text(
            val * 1.005,         # slightly to the right of the bar
            i,
            f"{int(round(val, 0))}",
            va="center",
            ha="left",
            fontsize=9
        )

    # h) Flip y‐axis so the slowest bar is at the top
    ax.set_yticks(y_pos)
    ax.set_yticklabels(functions, fontsize=9)
    ax.invert_yaxis()

    # i) Format the x‐axis in “k” with only ~5 ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    # j) Titles, labels, grid
    ax.set_xlabel("Average Cycles", fontsize=10)
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

fig_width = 8
fig_height = len(functions) * 0.4 + 1
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i, func in enumerate(functions):
    ax.barh(
        y_pos[i],
        cycles_vals[i],
        color=func_to_color[func],
        edgecolor="black"
    )

max_cycles = max(cycles_vals)
ax.set_xlim(0, max_cycles * 1.15)

for i, val in enumerate(cycles_vals):
    ax.text(
        val * 1.005,
        i,
        f"{int(round(val, 0))}",
        va="center",
        ha="left",
        fontsize=9
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(functions, fontsize=9)
ax.invert_yaxis()

ax.xaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

ax.set_xlabel("Average Cycles (averaged across all test cases)", fontsize=10)
ax.set_title("Function — Mean Cycles Across All Test Cases", fontsize=12)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/benchmark_avg_all_cases_horizontal.png", dpi=150)
plt.close(fig)
ax.xaxis.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig("plots/benchmark_avg_all_cases_horizontal.png", dpi=150)
plt.close(fig)

## Cost
for opi, op in enumerate(["ADDS", "MULS", "DIVS", "SQRT"]):
    fig, ax = plt.subplots(figsize=(12, 6))  # This gives both fig and ax

    cols = []
    x = np.arange(len(flops["Section"].value_counts()))
    width = 0.25
    multiplier = 0

    for label, group in flops.groupby("Section"):
        cols.append(label)
        data = group.groupby("Function")[op].mean()

        offset = 0  # width * multiplier

        for i, f in enumerate(data.index):
            rects = plt.bar(
                x[multiplier] + offset,
                data.values[i],
                width,
                label=f"{f}",
                color=color_styles[f],
            )
            plt.bar_label(rects, padding=3, rotation=45)
            offset += width

        multiplier += 1

    plt.xlabel("Section")
    plt.xticks(x + width, cols)
    plt.yscale("log")  # Only set the Y-axis to log scale
    plt.ylabel(f"{op} Count")
    plt.title(f"Cost Evaluation {op}")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Function")

    plt.tight_layout()

    plt.savefig(f"plots/Cost{op}.png")
    plt.close()


# Cost per func
#
op_colors = {
    "ADDS": "green",
    "MULS": "blue",
    "DIVS": "orange",
    "SQRT": "RED",
}

section_colors = {
    "Initialization": "green",
    "Impulse": "blue",
    "Delta": "orange",
    "Velocity": "purple",
    "Transform to World Frame": "red",
    "collide_balls": "black",
}


def bar_plot_by_testcase(column, width=0.125, log=True, roundv=True):
    tc = list(df["Test Case"].unique())
    sections = section_colors.keys()
    x = np.arange(len(tc))

    for f, data in df.groupby("Function"):
        groups = data.groupby(["Test Case", "Section"])[[column]].mean()

        fig, ax = plt.subplots(figsize=(20, 6))

        for testcase in tc:
            offset = -len(sections) / 2 * width
            for section in sections:
                rects = plt.bar(
                    x[tc.index(testcase)] + offset,
                    round(groups.loc[(testcase, section), column])
                    if roundv
                    else groups.loc[(testcase, section), column],
                    width=width,
                    label=section,
                    color=section_colors[section],
                )
                plt.bar_label(rects, padding=3, rotation=45)
                offset += width

        plt.xlabel("Testcase")
        plt.xticks(x, tc)
        if log:
            plt.yscale("log")  # Only set the Y-axis to log scale
        plt.ylabel(column)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Section")

        plt.tight_layout()

        plt.savefig(f"plots/{column}_{f}.png")
        plt.close()


bar_plot_by_testcase("Cycles")
bar_plot_by_testcase("FlopsPerCycle", log=False, roundv=False)


for f, data in flops.groupby("Function"):
    groups = data.groupby(["Section"])[["ADDS", "MULS", "DIVS", "SQRT"]].mean()

    fig, ax = plt.subplots(figsize=(12, 6))  # This gives both fig and ax

    width = 0.1

    x = np.arange(len(flops["Section"].value_counts()))

    for i, section in enumerate(section_colors.keys()):
        # for i, section in enumerate(groups.index):
        offset = 0
        for vi, v in enumerate(["ADDS", "MULS", "DIVS", "SQRT"]):
            rects = plt.bar(
                x[i] + offset,
                groups.loc[section, v],
                width=width,
                label=v,
                color=op_colors[v],
            )
            plt.bar_label(rects, padding=3, rotation=45)
            offset += width

    plt.xlabel("Section")
    plt.xticks(x, section_colors.keys())
    plt.yscale("log")  # Only set the Y-axis to log scale
    plt.ylabel("Operation Count")
    plt.title(f"Cost Evaluation for {f}")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Operation")

    plt.tight_layout()

    plt.savefig(f"plots/Cost_{f}.png")
    plt.close()


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
