"""Difference-in-differences event-study plot (AER / QJE house format).

Two outcomes, event time -5 to +5 relative to the reform, period -1
normalised to zero.  Each estimate carries a thick 90% and a thin capped
95% confidence interval, the reform date is a vertical dashed rule, and the
pre-trend joint test plus the average post-reform effect are reported in a
call-out box.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label)

EVENT = np.arange(-5, 6)
BASE = -1

SPECS = [
    dict(
        title="Log employment",
        ylabel="Effect on log employment",
        color="#2a5d8f",
        coef=[0.004, -0.006, 0.003, -0.002, 0.000,
              0.013, 0.029, 0.042, 0.051, 0.056, 0.058],
        se=[0.011, 0.010, 0.009, 0.008, 0.000,
            0.010, 0.012, 0.014, 0.016, 0.018, 0.021],
        pretrend="F = 0.71,  p = 0.62",
        avg="+4.9% (s.e. 1.3)",
    ),
    dict(
        title="Log real wage bill per worker",
        ylabel="Effect on log wage bill p.w.",
        color="#a8493a",
        coef=[-0.008, 0.005, -0.004, 0.006, 0.000,
              0.002, 0.009, 0.021, 0.030, 0.034, 0.033],
        se=[0.013, 0.012, 0.011, 0.009, 0.000,
            0.011, 0.013, 0.015, 0.017, 0.019, 0.022],
        pretrend="F = 1.08,  p = 0.37",
        avg="+2.2% (s.e. 1.2)",
    ),
]


def draw_panel(ax, spec):
    coef = np.asarray(spec["coef"], dtype=float)
    se = np.asarray(spec["se"], dtype=float)
    col = spec["color"]
    is_base = EVENT == BASE

    ax.axvspan(-0.5, 5.6, color="#f4f4f2", zorder=0)
    ax.axhline(0.0, color="#555555", lw=0.8, ls=(0, (5, 4)), zorder=1)
    ax.axvline(-0.5, color="#333333", lw=1.0, ls=(0, (3, 2.5)), zorder=2)

    # 95% (thin, capped) then 90% (thick) intervals
    ax.vlines(EVENT[~is_base], (coef - 1.96 * se)[~is_base],
              (coef + 1.96 * se)[~is_base], color=col, lw=0.9, zorder=3)
    for x, lo, hi in zip(EVENT[~is_base], (coef - 1.96 * se)[~is_base],
                         (coef + 1.96 * se)[~is_base]):
        ax.plot([x - 0.11, x + 0.11], [lo, lo], color=col, lw=0.9, zorder=3)
        ax.plot([x - 0.11, x + 0.11], [hi, hi], color=col, lw=0.9, zorder=3)
    ax.vlines(EVENT[~is_base], (coef - 1.645 * se)[~is_base],
              (coef + 1.645 * se)[~is_base], color=col, lw=2.4, zorder=4,
              alpha=0.85)

    ax.plot(EVENT[~is_base], coef[~is_base], "o", color=col, ms=4.2,
            mec="white", mew=0.7, zorder=5)
    ax.plot([BASE], [0.0], "o", color="white", ms=4.6, mec=col, mew=1.3,
            zorder=6)

    ax.set_xlim(-5.7, 5.7)
    ax.set_xticks(EVENT)
    ax.set_xticklabels([str(t) for t in EVENT], fontsize=7.5)
    ax.set_xlabel("Event time (quarters relative to reform)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"], fontsize=9, loc="left", pad=6, weight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    style_axes(ax)


def main():
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), sharex=True)

    for ax, spec in zip(axes, SPECS):
        draw_panel(ax, spec)
        ax.set_ylim(-0.045, 0.105)
        ax.set_yticks(np.arange(-0.04, 0.101, 0.02))
        ax.set_yticklabels([f"{v:+.2f}" if abs(v) > 1e-9 else "0.00"
                            for v in np.arange(-0.04, 0.101, 0.02)])

    for ax, spec in zip(axes, SPECS):
        ax.text(0.035, 0.965,
                f"Pre-trend joint test:  {spec['pretrend']}\n"
                f"Mean post effect:  {spec['avg']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.36", fc="white",
                          ec="#d5d5d5", lw=0.6, alpha=0.95))
        ax.text(BASE, -0.042, "base\nperiod", ha="center", va="bottom",
                fontsize=6.6, color="#6f6f6f", linespacing=1.25)
        blend = ax.get_xaxis_transform()
        ax.text(-0.68, 0.60, "reform date", rotation=90, fontsize=7,
                color="#333333", ha="right", va="center", transform=blend)

    add_panel_label(axes[0], "(a)", x=-0.155, y=1.03)
    add_panel_label(axes[1], "(b)", x=-0.155, y=1.03)

    handles = [
        Line2D([], [], color="#3f3f3f", marker="o", ls="none", ms=4.2,
               mec="white", mew=0.7, label="Point estimate"),
        Line2D([], [], color="#3f3f3f", lw=2.4, alpha=0.85,
               label="90% CI"),
        Line2D([], [], color="#3f3f3f", lw=0.9, label="95% CI"),
        Line2D([], [], color="white", marker="o", ls="none", ms=4.6,
               mec="#3f3f3f", mew=1.3, label="Normalised to 0"),
    ]
    fig.tight_layout(w_pad=2.6, rect=(0.0, 0.135, 1.0, 1.0))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.4,
               frameon=False, bbox_to_anchor=(0.52, 0.058),
               handlelength=1.5, columnspacing=1.8)
    savefig(fig, "104_did_event_study.png")


if __name__ == "__main__":
    main()
