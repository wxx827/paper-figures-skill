"""Figure 618 - Source-tree growth, refactor events and technical debt.

One synthetic release history drives every panel.  Panel (a) stacks the size
of five modules across thirty releases; the two refactor releases are the
only places where a module shrinks, and the debt curve on the right axis is
the same event series seen from the maintenance side, so the vertical rules
line up with both by construction.  Panel (b) resolves the same tree down to
individual files: churn against a stability index, with the hot files (top
decile of churn and below-median stability) flagged in house red.  Panel (c)
is the commit volume the growth in (a) was integrated from.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import panel_label_inches

MODULES = ["tests", "storage", "core", "api", "ui"]
ACCENT = "#b23a48"          # house red - debt curve, hot files, refactors
BLUE = "#3b5b92"            # house blue - commit volume, cold files
N_REL = 30
REFACTOR = (10, 21)         # release indices where a clean-up landed


def build(rng):
    """Per-module KLOC, technical debt ratio, commits and contributors."""
    start = {"tests": 3.0, "storage": 2.2, "core": 6.5, "api": 2.8, "ui": 3.4}
    pace = {"tests": 1.05, "storage": 0.62, "core": 1.55, "api": 0.78,
            "ui": 1.18}
    cut = {"tests": 0.02, "storage": 0.06, "core": 0.14, "api": 0.05,
           "ui": 0.11}

    rel = np.arange(N_REL)
    kloc = {}
    for m in MODULES:
        step = pace[m] * rng.gamma(3.2, 0.32, N_REL)
        s = start[m] + np.cumsum(step)
        for r in REFACTOR:
            s[r:] -= cut[m] * s[r]
        kloc[m] = np.maximum(s, 0.5)

    debt = 9.0 + 0.62 * rel + rng.normal(0.0, 0.55, N_REL)
    for r in REFACTOR:
        debt[r:] -= 0.31 * debt[r]
    debt = np.clip(debt, 3.0, None)

    commits = np.round(52.0 + 5.1 * rel + 34.0 * rng.random(N_REL)).astype(int)
    commits[list(REFACTOR)] = np.round(commits[list(REFACTOR)] * 1.55)
    contrib = np.round(4.0 + 0.42 * rel + 2.4 * rng.random(N_REL)).astype(int)
    return rel, kloc, debt, commits, contrib


def build_files(rng, kloc):
    """One point per source file: churn, stability and size."""
    n = 96
    churn = np.round(np.exp(rng.normal(2.55, 0.82, n))).astype(float) + 2.0
    size = 0.10 + 0.85 * rng.gamma(1.9, 0.42, n)
    stab = 0.94 - 0.052 * np.log(churn) * (1.0 + 0.30 * rng.random(n))
    stab = np.clip(stab + rng.normal(0.0, 0.055, n), 0.05, 0.99)
    hot = (churn >= np.quantile(churn, 0.90)) & (stab <= np.median(stab))
    return churn, stab, size, hot


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(618)
    rel, kloc, debt, commits, contrib = build(rng)
    churn, stab, size, hot = build_files(rng, kloc)

    ramp = get_palette("cb_sequential_blue", 7)[1:6]
    total = np.sum([kloc[m] for m in MODULES], axis=0)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.062, right=0.935, top=0.885, bottom=0.125,
                          wspace=0.40, hspace=0.52)
    ax = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) stacked module size + technical debt -------------------------
    ax.stackplot(rel, [kloc[m] for m in MODULES], colors=ramp,
                 edgecolor="white", linewidth=0.5, zorder=2)
    for r in REFACTOR:
        ax.axvline(r, color=ACCENT, lw=0.9, ls=(0, (4.5, 2.2)), zorder=6)
        ax.text(r + 0.25, total.max() * 1.03, "refactor",
                fontsize=6.4, color=ACCENT, rotation=90, ha="left",
                va="top", zorder=7)

    # module names on the band they belong to, at the last release
    mid = np.cumsum([kloc[m][-1] for m in MODULES]) - \
        0.5 * np.asarray([kloc[m][-1] for m in MODULES])
    for m, y, c in zip(MODULES, mid, ramp):
        ax.text(N_REL - 1.4, y, m, fontsize=7.0, ha="right", va="center",
                color="white" if c in ramp[3:] else "#1a1a1a",
                fontfamily="monospace", zorder=8)

    ax.set_xlim(0, N_REL - 1)
    ax.set_ylim(0, total.max() * 1.16)
    ax.set_xlabel("release tag (-)", fontsize=8.0)
    ax.set_ylabel("source size (KLOC)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_xticks(np.arange(0, N_REL, 4))
    ax.set_xticklabels(["v%.1f" % (1.0 + 0.1 * k)
                        for k in np.arange(0, N_REL, 4)], fontsize=6.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    axd = ax.twinx()
    axd.plot(rel, debt, color=ACCENT, lw=1.6, zorder=9)
    axd.plot([rel[-1]], [debt[-1]], marker="o", ms=3.8, mfc="white",
             mec=ACCENT, mew=1.1, zorder=10)
    axd.set_ylabel("technical debt ratio (%)", fontsize=8.0, color=ACCENT)
    axd.tick_params(labelsize=7.0, colors=ACCENT)
    axd.set_ylim(0, debt.max() * 1.45)
    axd.spines["top"].set_visible(False)
    axd.spines["right"].set_color(ACCENT)

    ax.text(0.025, 0.965,
            "%d commits, %d contributors\n%.1f KLOC at v%.1f"
            % (commits.sum(), contrib.max(), total[-1],
               1.0 + 0.1 * (N_REL - 1)),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color="#1a1a1a", linespacing=1.35, zorder=11,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.92))
    ax.set_title("Module size across %d releases" % N_REL, fontsize=8.8,
                 fontweight="bold", pad=4.0)

    # ---- (b) churn versus stability ---------------------------------------
    ax_s.scatter(churn[~hot], stab[~hot], s=26.0 * size[~hot] + 4.0,
                 facecolor=BLUE, alpha=0.42, edgecolor="white", lw=0.4,
                 zorder=3)
    ax_s.scatter(churn[hot], stab[hot], s=26.0 * size[hot] + 4.0,
                 facecolor=ACCENT, alpha=0.85, edgecolor="white", lw=0.5,
                 zorder=4)
    ax_s.axhline(float(np.median(stab)), color=COLORS["baseline"], lw=0.7,
                 ls=(0, (3.5, 2.0)), zorder=2)
    ax_s.set_xscale("log")
    ax_s.set_xlabel("change frequency (commits per file)", fontsize=7.4)
    ax_s.set_ylabel("stability index (-)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.6)
    ax_s.set_ylim(min(0.42, stab.min() - 0.05), 1.02)
    ax_s.grid(True, lw=0.45, alpha=0.55)
    ax_s.set_axisbelow(True)
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)
    ax_s.set_title("%d of %d files are hot spots" % (int(hot.sum()), hot.size),
                   fontsize=8.5, fontweight="bold", pad=3.5)
    ax_s.legend(handles=[
        Line2D([0], [0], marker="o", ls="none", ms=4.2, mfc=ACCENT,
               mec="white", label="hot spot"),
        Line2D([0], [0], marker="o", ls="none", ms=4.2, mfc=BLUE,
               mec="white", alpha=0.5, label="other file"),
    ], loc="lower left", frameon=False, fontsize=6.4, handletextpad=0.3,
        borderpad=0.2)

    # ---- (c) commit volume -------------------------------------------------
    cols = [ACCENT if k in REFACTOR else BLUE for k in rel]
    ax_c.bar(rel, commits, width=0.74, color=cols, alpha=0.80,
             edgecolor="none", zorder=3)
    ax_c.set_xlim(-1.0, N_REL)
    ax_c.set_xlabel("release index (-)", fontsize=7.4)
    ax_c.set_ylabel("commits per\nrelease (-)", fontsize=7.4,
                    linespacing=1.15)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(axis="y", lw=0.45, alpha=0.55)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.set_title("Median %d commits per release" % int(np.median(commits)),
                   fontsize=8.5, fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_s, "(b)"), (ax_c, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.42, dy=0.09)

    r2 = REFACTOR[1]

    savefig(fig, "618_swe_code_evolution.png")


if __name__ == "__main__":
    main()
