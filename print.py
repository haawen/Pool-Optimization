#!/usr/bin/env python3
import pandas as pd
import numpy as np

def trimmed_mean(s: pd.Series) -> float:
    """
    Drop the bottom 10% and top 10% of samples in `s` before averaging.
    If that removes everything, fall back to the plain mean.
    """
    if s.empty:
        return np.nan
    low, high = s.quantile(0.10), s.quantile(0.90)
    trimmed = s[(s >= low) & (s <= high)]
    return trimmed.mean() if not trimmed.empty else s.mean()

def main():
    # 1) load
    df = pd.read_csv("build/benchmark.csv")
    df["Test Case"] = "TC " + df["Test Case"].astype(str)
    df.columns = df.columns.str.strip()

    # 2) pivot with 10% trimmed mean
    cycles = df.pivot_table(
        index="Function",
        columns="Test Case",
        values="Cycles",
        aggfunc=trimmed_mean
    )

    # 3) round to whole cycles
    cycles = cycles.round(0).astype(int)

    # 4) rank per test‐case (1 = highest cycle count)
    ranks = cycles.rank(ascending=False, method="min")

    # 5) compute mean rank & reverse it so larger is worse
    mean_rank = ranks.mean(axis=1)
    rev_rank  = (len(mean_rank) + 1) - mean_rank

    # 6) assemble & sort
    summary = cycles.copy()
    summary["Mean Rank"] = mean_rank.round(2)
    summary["Reverse Mean Rank"] = rev_rank.round(2)
    summary = summary.sort_values("Reverse Mean Rank", ascending=False)

    # 7) print
    print("\n10%-trimmed mean cycles per Function x TC (rounded), plus rankings:\n")
    print(summary.to_string())

if __name__ == "__main__":
    main()
