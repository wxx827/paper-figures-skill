"""Gantt / timeline chart: project phases, progress and milestones.

Horizontal timeline with one row per work package. Each bar shows the planned
window in a light tint with the reported progress filled in on top, the
duration and completion printed at the bar end, and deliverables drawn as
diamonds on a dedicated milestone row. Quarters are shaded so the reader can
locate a date without tracing back to the axis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime
from pubstyle import set_pub_style, savefig, style_axes

# (task, start, end, colour, progress)
TASKS = [
    ("Literature review",  "2024-01-08", "2024-03-29", "#3b5b92", 1.00),
    ("Field campaign",     "2024-02-19", "2024-06-28", "#2e7d5b", 1.00),
    ("Remote-sensing prep", "2024-05-06", "2024-09-27", "#6bbf8a", 0.72),
    ("Model development",  "2024-07-01", "2024-12-20", "#c9a227", 0.34),
    ("Validation",         "2024-10-07", "2025-02-28", "#c0603f", 0.08),
    ("Manuscript",         "2025-01-06", "2025-04-30", "#8e6bbf", 0.00),
]
MILESTONES = [("Protocol frozen", "2024-03-29"),
              ("Field data locked", "2024-06-28"),
              ("Model v1", "2024-12-20"),
              ("Submission", "2025-04-30")]
TODAY = datetime(2024, 9, 15)


def d(s):
    return mdates.date2num(datetime.strptime(s, "%Y-%m-%d"))


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    n = len(TASKS)

    q_start = datetime(2024, 1, 1)
    for q in range(6):
        left = mdates.date2num(datetime(2024 + (q * 3) // 12,
                                        1 + (q * 3) % 12, 1))
        right = mdates.date2num(datetime(2024 + ((q + 1) * 3) // 12,
                                         1 + ((q + 1) * 3) % 12, 1))
        if q % 2 == 0:
            ax.axvspan(left, right, color="#f5f5f5", zorder=0, lw=0)
        ax.text((left + right) / 2, -1.15, f"Q{q % 4 + 1} {2024 + q // 4}",
                ha="center", va="center", fontsize=7, color="#888888")

    for i, (name, s, e, col, prog) in enumerate(TASKS):
        x0, x1 = d(s), d(e)
        ax.barh(i, x1 - x0, left=x0, height=0.52, color=col, alpha=0.28,
                edgecolor=col, linewidth=0.8, zorder=2)
        if prog > 0:
            ax.barh(i, (x1 - x0) * prog, left=x0, height=0.52, color=col,
                    edgecolor="none", zorder=3)
        months = (x1 - x0) / 30.44
        ax.text(x1 + 6, i, f"{months:.1f} mo  -  {prog:.0%}", va="center",
                fontsize=7.5, color="#555555", zorder=4)

    for name, date in MILESTONES:
        ax.scatter(d(date), n, marker="D", s=52, color="#1a1a1a",
                   edgecolors="white", linewidths=0.9, zorder=5)
        ax.text(d(date), n + 0.42, name, ha="center", va="bottom",
                fontsize=7.2, color="#333333")

    today = mdates.date2num(TODAY)
    ax.axvline(today, color="#b23a48", ls="--", lw=1.2, zorder=6)
    ax.text(today, -0.78, " today", color="#b23a48", fontsize=7.5,
            ha="left", va="center", fontweight="bold")

    ax.set_yticks(list(range(n)) + [n])
    ax.set_yticklabels([t[0] for t in TASKS] + ["Milestones"], fontsize=8.5)
    ax.get_yticklabels()[-1].set_style("italic")
    ax.set_ylim(n + 1.0, -1.45)
    ax.set_xlim(d("2023-12-20"), d("2025-06-25"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
             fontsize=7.5)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    style_axes(ax)
    ax.set_title("Research project timeline", fontsize=11, fontweight="bold",
                 loc="left", x=-0.13, pad=12)

    handles = [Patch(facecolor="#808080", alpha=0.28, edgecolor="#808080",
                     label="Planned window"),
               Patch(facecolor="#808080", label="Completed"),
               Line2D([], [], marker="D", color="none", markersize=6,
                      markerfacecolor="#1a1a1a", markeredgecolor="white",
                      label="Deliverable"),
               Line2D([], [], color="#b23a48", ls="--", lw=1.2,
                      label="Reporting date")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    savefig(fig, "51_gantt.png")


if __name__ == "__main__":
    main()
