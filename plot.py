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
        ["Nanoseconds", "Cycles", "Flops", "Memory"]
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
}

marker_styles = {
    "Default": "o",
    "Basic Implementation": "^",
    "Code Motion": "s",
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

    plt.savefig(f"plot_{col}.png")

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

plt.savefig("plot_FlopsPerCycle.png")
