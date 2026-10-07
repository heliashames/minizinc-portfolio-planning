"""Figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def make_all(log: pd.DataFrame, bench: pd.DataFrame, wealth: pd.Series, out: Path) -> None:
    log = log.set_index("date")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wealth.index, wealth.values, color="black", lw=2, label="Strategy (MiniZinc)")
    for c in bench.columns:
        ax.plot(bench.index, bench[c], lw=1, alpha=0.8, label=c)
    ax.set_title("Portfolio value over time (USD)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_wealth.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(log.index, log["cash"], log["gold_value"], log["spot_value"], log["margin_equity"], log["bonds"],
                 labels=["cash", "gold", "spot crypto", "margin equity", "bonds"],
                 colors=["#9aa0a6", "#d4a017", "#2a7de1", "#c0392b", "#2e9b5f"])
    ax.set_title("Allocation at the end of each week (USD)"); ax.legend(loc="upper left", fontsize=7, ncol=5)
    fig.tight_layout(); fig.savefig(out / "fig_allocation.png"); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, a in zip(axes, ["gold", "btc", "eth"]):
        ax.plot(log.index, log[f"act_{a}"] * 100, color="black", lw=0.8, label="actual")
        ax.plot(log.index, log[f"fc_{a}"] * 100, color="#2a7de1", lw=1.2, label="forecast")
        ax.set_ylabel(f"{a.upper()} weekly %")
    axes[0].legend(fontsize=7); axes[0].set_title("Weekly return: forecast vs actual")
    fig.tight_layout(); fig.savefig(out / "fig_forecasts.png"); plt.close(fig)
