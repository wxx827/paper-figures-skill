"""Bump chart: how eight world-model methods trade citation rank, 2021-2026.

Ranks are never written down -- they are ``argsort`` of the yearly citation
matrix, so the curves, the numbers inside the markers, the endpoint labels and
the climber call-out cannot drift apart from the data.  A line is drawn in deep
blue when the method ends higher than it started, in pale blue when it ends
lower, and in the single warm accent when it finishes first.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import COLORS, darken, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
RISE, FLAT, FALL = BLUE[1], BLUE[3], BLUE[4]

METHODS = ["Dreamer", "MuZero", "VideoGPT", "IRIS", "DayDreamer",
           "GAIA-1", "Genie", "TD-MPC2"]
YEARS = np.arange(2021, 2027)


def gen_data(seed=1557):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_survey_citation_bump.csv", index_col=0)
        return df.index.tolist(), df.columns.astype(int).to_numpy(), \
               df.to_numpy()
    """
    rng = np.random.default_rng(seed)
    n, t = len(METHODS), YEARS.size
    start = np.array([260, 210, 170, 40, 55, 20, 12, 18], dtype=float)
    # Yearly multiplicative growth; late arrivals compound faster.
    rate = np.array([1.34, 1.28, 1.41, 1.86, 1.72, 2.35, 2.62, 1.95])
    cites = np.zeros((n, t))
    cites[:, 0] = start
    for j in range(1, t):
        step = rate * rng.uniform(0.86, 1.16, n)
        cites[:, j] = cites[:, j - 1] * step
    return METHODS, YEARS, np.rint(cites).astype(int)


def smoothstep(x0, y0, x1, y1, n=60):
    s = np.linspace(0.0, 1.0, n)
    return x0 + (x1 - x0) * s, y0 + (y1 - y0) * (3 * s ** 2 - 2 * s ** 3)


def main():
    set_pub_style(base_font=8)
    methods, years, cites = gen_data()
    n, t = cites.shape

    ranks = np.argsort(np.argsort(-cites, axis=0), axis=0) + 1
    delta = ranks[:, 0] - ranks[:, -1]              # > 0 means climbed
    leader = int(np.argmin(ranks[:, -1]))
    climber = int(np.argmax(delta))

    def colour(i):
        if i == leader:
            return ACCENT
        return RISE if delta[i] > 0 else FALL if delta[i] < 0 else FLAT

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = fig.add_axes([0.155, 0.130, 0.615, 0.705])

    for r in range(1, n + 1):
        ax.plot([-0.42, t - 1 + 0.42], [r, r], color="#eef1f4", lw=0.9,
                zorder=0)
    for j in range(t):
        ax.plot([j, j], [0.45, n + 0.55], color="#f2f4f6", lw=0.8, zorder=0)

    for i in range(n):
        col = colour(i)
        lead = i == leader
        for j in range(t - 1):
            xs, ys = smoothstep(j, ranks[i, j], j + 1, ranks[i, j + 1])
            ax.plot(xs, ys, color=col, lw=2.6 if lead else 2.0, zorder=3,
                    solid_capstyle="round", alpha=1.0 if lead else 0.92)
        ax.scatter(range(t), ranks[i], color=col, s=118 if lead else 100,
                   zorder=4, edgecolors="white", linewidths=1.1)
        for j in range(t):
            ax.text(j, ranks[i, j], f"{ranks[i, j]:d}", ha="center",
                    va="center", fontsize=6.2, color="white",
                    fontweight="bold", zorder=5)
        tcol = darken(col, 0.30) if col == FALL else col
        ax.text(-0.30, ranks[i, 0], methods[i], ha="right", va="center",
                fontsize=8.0, color=tcol, fontweight="bold")
        ax.text(t - 1 + 0.30, ranks[i, -1],
                f"{methods[i]}  ({cites[i, -1]:,d})", ha="left", va="center",
                fontsize=8.0, color=tcol, fontweight="bold")

    # ---- one computed call-out: the biggest climber -------------------------
    ax.annotate(f"{methods[climber]}: rank {ranks[climber, 0]:d} "
                f"$\\rightarrow$ {ranks[climber, -1]:d} "
                f"({delta[climber]:+d} places)",
                xy=(0, ranks[climber, 0]),
                xytext=(2.55, n + 0.80), ha="center", va="center",
                fontsize=7.0, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#c3ccd4", linewidth=0.6),
                arrowprops=dict(arrowstyle="-", color="#8d979f", lw=0.7,
                                shrinkA=2.0, shrinkB=6.0))

    ax.set_xticks(range(t))
    ax.set_xticklabels([str(y) for y in years], fontsize=8.0, color=INK)
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([f"{r:d}" for r in range(1, n + 1)], fontsize=7.4,
                       color=INK)
    ax.set_ylabel(f"Citation rank among the {n:d} methods (1 = most cited)",
                  fontsize=8.0, color=INK, labelpad=1.0)
    ax.set_ylim(n + 1.15, 0.38)
    ax.set_xlim(-1.42, t - 1 + 1.55)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # The leader is drawn in the accent, not in the rise/fall tones, so it must
    # not be counted twice: the three key entries have to add up to n.
    others = np.array([i != leader for i in range(n)])
    n_rise = int((others & (delta > 0)).sum())
    n_fall = int((others & (delta < 0)).sum())
    n_flat = int((others & (delta == 0)).sum())
    keys = [Line2D([], [], color=ACCENT, lw=2.6,
                   label=f"finishes first ({methods[leader]})"),
            Line2D([], [], color=RISE, lw=2.0,
                   label=f"climbed by {years[-1]} ({n_rise:d})"),
            Line2D([], [], color=FALL, lw=2.0,
                   label=f"slipped by {years[-1]} ({n_fall:d})")]
    if n_flat:
        keys.insert(2, Line2D([], [], color=FLAT, lw=2.0,
                              label=f"unchanged ({n_flat:d})"))
    leg = fig.legend(handles=keys, loc="upper left",
                     bbox_to_anchor=(0.790, 0.700), fontsize=7.4,
                     frameon=False, labelspacing=0.85, handlelength=2.0,
                     handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.text(0.790, 0.200,
             f"Total citations in {years[-1]}:\n{cites[:, -1].sum():,d}\n\n"
             f"Top method holds\n"
             f"{cites[leader, -1] / cites[:, -1].sum():.0%} of them",
             ha="left", va="top", fontsize=7.2, color=COLORS["sub"],
             linespacing=1.6)

    # Name the two endpoint years rather than a year count: the axis spans
    # t columns but only t - 1 years, and readers subtract.
    span = f"between {years[0]:d} and {years[-1]:d}"
    if climber == leader:
        headline = (f"{methods[climber]} climbs {delta[climber]:d} places "
                    f"{span} to take the citation lead")
    else:
        headline = (f"{methods[climber]} climbs {delta[climber]:d} places "
                    f"{span}, but {methods[leader]} keeps the lead")

    savefig(fig, "1152_wm_survey_citation_bump.png")


if __name__ == "__main__":
    main()
