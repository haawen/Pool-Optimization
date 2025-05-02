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

df.columns = df.columns.str.strip()

# Group by Function, Section, and Test Case
grouped = (
    df.groupby(["Function", "Section", "Test Case"])[
        ["Nanoseconds", "Cycles", "Flops", "Memory", "ADDS", "MULS", "DIVS", "SQRT"]
    ]
    .mean()
    .reset_index()
)
grouped["Group"] = grouped["Function"] + " | " + grouped["Section"]

# Define line styles for each Function
line_styles = {
    "Default": "-",
    "Basic Implementation": "--",
    "Code Motion": "-.",
    "RSQRT": ":",
}

marker_styles = {
    "Default": "o",
    "Basic Implementation": "^",
    "Code Motion": "s",
    "RSQRT": "x",
}


color_styles = {
    "Default": "o",
    "Basic Implementation": "blue",
    "Code Motion": "green",
    "RSQRT": "orange",
}

for col in ["Nanoseconds", "Cycles", "Flops", "Memory"]:
    # Plotting with Y-axis in log scale and different line styles for each function
    plt.figure(figsize=(12, 6))

    for label, group_data in grouped.groupby("Group"):
        function_name = group_data["Function"].iloc[
            0
        ]  # Get the function name for the current group
        line_style = line_styles.get(function_name, "-")
        marker_style = marker_styles.get(function_name, "o")
        plt.plot(
            group_data["Test Case"],
            group_data[col],
            marker=marker_style,
            label=label,
            linestyle=line_style,
        )

    plt.yscale("log")  # Only set the Y-axis to log scale
    plt.xlabel("Test Case")
    plt.ylabel(f"Average {col} (log scale)")
    plt.title(f"Profiling {col}")
    plt.legend(title="Function | Section", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(f"plots/{col}.png")

## FLops per seconds

plt.figure(figsize=(12, 6))

for label, group_data in grouped.groupby("Group"):
    function_name = group_data["Function"].iloc[
        0
    ]  # Get the function name for the current group
    line_style = line_styles.get(function_name, "-")
    marker_style = marker_styles.get(function_name, "o")

    plt.plot(
        group_data["Test Case"],
        group_data["Flops"] / (group_data["Cycles"]),
        marker=marker_style,
        label=label,
        linestyle=line_style,
    )

plt.xlabel("Test Case")
plt.ylabel("Average flops per cycle (log scale)")
plt.title("Profiling Flops Per Cycle")
plt.legend(title="Function | Section", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig("plots/FlopsPerCycle.png")


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
            plt.bar_label(rects, padding=3)
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

for f, data in flops.groupby("Function"):
    groups = data.groupby(["Section"])[["ADDS", "MULS", "DIVS", "SQRT"]].mean()

    fig, ax = plt.subplots(figsize=(12, 6))  # ✅ This gives both fig and ax

    width = 0.1

    x = np.arange(len(flops["Section"].value_counts()))

    for i, section in enumerate(groups.index):
        offset = 0
        for vi, v in enumerate(["ADDS", "MULS", "DIVS", "SQRT"]):
            rects = plt.bar(
                x[i] + offset,
                groups.values[i, vi],
                width=width,
                label=v,
                color=op_colors[v],
            )
            plt.bar_label(rects, padding=3)
            offset += width

    plt.xlabel("Section")
    plt.xticks(x, groups.index)
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
