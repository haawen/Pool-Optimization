#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_csv_sections(content):
    lines = content.splitlines()
    sections = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            header = line[1:].strip()
            parts = header.split(",")
            section_title = parts[0].strip()
            # Skip to column header line
            i += 1
            if i >= len(lines):
                break
            # Parse column headers (unused but advance past)
            _ = lines[i].strip().split(",")
            x_vals, y_vals = [], []
            i += 1
            # Read data lines until blank or next section
            while i < len(lines):
                dl = lines[i].strip()
                if not dl or dl.startswith("#") or dl.lower().startswith("summary"):
                    break
                vals = [v.strip() for v in dl.split(",")]
                if len(vals) >= 2:
                    x_raw, y_raw = vals[0], vals[1]
                    try:
                        y = float(y_raw)
                    except ValueError:
                        i += 1
                        continue
                    if section_title == "Collision Scenario Benchmarks":
                        x = x_raw
                    else:
                        try:
                            x = int(x_raw)
                        except ValueError:
                            try:
                                x = float(x_raw)
                            except ValueError:
                                x = x_raw
                    x_vals.append(x)
                    y_vals.append(y)
                i += 1
            if x_vals:
                sections[section_title] = (np.array(x_vals), np.array(y_vals))
        else:
            i += 1
    return sections

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("input_file", help="Benchmark output file")
    parser.add_argument("--output", "-o", help="Prefix for saved plot image")
    parser.add_argument("--show", "-s", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading '{args.input_file}': {e}")
        return

    # Parse CSV-style benchmark sections
    sections = parse_csv_sections(content)

    plt.figure(figsize=(12, 10))
    subplot_index = 1
    for title, data in sections.items():
        if data is None or len(data[0]) == 0:
            continue
        x, y = data
        ax = plt.subplot(2, 2, subplot_index)
        # Convert to milliseconds
        y_ms = y * 1000
        if x.dtype.type is np.str_ or isinstance(x[0], str):
            ax.bar(range(len(x)), y_ms)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, rotation=45, ha='right')
            ax.set_ylabel("Time (ms)")
        else:
            ax.plot(x, y_ms, marker="o")
            ax.set_xlabel(title.split("vs")[-1].strip())
            ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.grid(True, axis="y")
        subplot_index += 1

    plt.tight_layout()
    if args.output:
        plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    if args.show or not args.output:
        plt.show()

if __name__ == "__main__":
    main()