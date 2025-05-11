import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(
    "Make sure to have run flops and profiling before plotting. Flops needs only be run when flop count changes."
)


flops = pd.read_csv("build/flops.csv")

df = pd.read_csv("build/profiling.csv")

df = pd.merge(flops, df, on=["Function", "Section", "Test Case"])
df["Test Case"] = "TC " + df["Test Case"].astype(str)
df["FlopsPerCycle"] = df["Flops"] / df["Cycles"]

df.columns = df.columns.str.strip()

benchmark = pd.read_csv("build/benchmark.csv")
benchmark["Test Case"] = "TC " + benchmark["Test Case"].astype(str)
benchmark.columns = benchmark.columns.str.strip()

# Group by Function, Section, and Test Case
grouped = (
    df.groupby(["Function", "Section", "Test Case"])[
        ["Cycles", "Flops", "Memory", "ADDS", "MULS", "DIVS", "SQRT"]
    ]
    .mean()
    .reset_index()
)
grouped["Group"] = grouped["Function"] + " | " + grouped["Section"]

# Define line styles for each Function

marker_styles = {
    "Default": "o",
    "Basic Implementation": "^",
    "Code Motion": "s",
    "Less SQRT": "m",
    "Branch Pred": "m",
    "SIMD": "x",
}


color_styles = {
    "Default": "o",
    "Basic Implementation": "blue",
    "Code Motion": "green",
    "Less SQRT": "purple",
    "Branch Pred": "grey",
    "Removed Unused Branches": "black",
    "SIMD": "orange",
}


fig, ax = plt.subplots(figsize=(12, 6))  # ✅ This gives both fig and ax
x = np.arange(len(benchmark["Test Case"].value_counts()))
cols = []
for i, (tc, group) in enumerate(benchmark.groupby("Test Case")):
    sub = group.groupby("Function")[["Cycles"]].mean()
    functions = len(sub.index)
    width = 0.1
    cols.append(tc)

    offset = -functions / 2 * width
    for f in sub.index:
        cycles = sub.loc[f, "Cycles"]

        rects = plt.bar(
            x[i] + offset,
            cycles,
            width,
            label=f"{f}",
            color=color_styles[f],
        )

        plt.bar_label(rects, padding=3, rotation=45)
        offset += width

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Function")

plt.xlabel("")
plt.xticks(x, cols)
# plt.yscale("log")  # Only set the Y-axis to log scale
plt.ylabel(f"Cycles")
plt.title(f"Cycles Benchmark")

plt.savefig(f"plots/benchmark.png")


## Cost
for opi, op in enumerate(["ADDS", "MULS", "DIVS", "SQRT"]):
    fig, ax = plt.subplots(figsize=(12, 6))  # ✅ This gives both fig and ax

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


bar_plot_by_testcase("Cycles")
bar_plot_by_testcase("FlopsPerCycle", log=False, roundv=False)


for f, data in flops.groupby("Function"):
    groups = data.groupby(["Section"])[["ADDS", "MULS", "DIVS", "SQRT"]].mean()

    fig, ax = plt.subplots(figsize=(12, 6))  # ✅ This gives both fig and ax

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
