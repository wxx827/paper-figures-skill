"""Visual search slopes: a flat feature search against two steep serial ones.

Main panel: mean correct response time against display set size for three
target definitions crossed with target-present and target-absent trials.  The
line through each condition is an ordinary least-squares fit and the search
slope quoted in milliseconds per item is that fit's coefficient, so the
printed efficiency and the drawn line are the same object.

Auxiliary (both at 58 % of the main panel width): the present and absent
slopes side by side with their ratio, the classic serial self-terminating
signature; and miss rate against set size, the accuracy cost that a slope
alone hides.

Fictional 20-observer experiment, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
PURPLE = "#77539f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

SET_SIZES = np.array([4, 8, 12, 16, 24, 32])
N_OBS = 20
N_REP = 24            # correct trials per observer per cell

# (name, short label, intercept ms, target-present slope ms/item, colour)
TASKS = [
    ("colour pop-out", "colour\npop-out", 452.0, 1.6, BLUE),
    ("colour x orientation", "colour x\norientation", 508.0, 18.5, ORANGE),
    ("T among L configuration", "T among L\nconfiguration", 546.0, 31.0,
     PURPLE),
]
ABSENT_FACTOR = 2.0   # target-absent search is exhaustive
MISS_BASE = 0.012
MISS_SLOPE = 0.0024   # extra miss probability per item at the hardest task


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(957)

    obs_offset = rng.normal(0.0, 34.0, N_OBS)
    obs_gain = rng.normal(1.0, 0.17, N_OBS)

    means, sems, fitted = {}, {}, {}
    for name, _, icept, slope, _ in TASKS:
        for present in (True, False):
            s = slope if present else slope * ABSENT_FACTOR
            cell_mu, cell_se = [], []
            for n in SET_SIZES:
                per_obs = np.array([
                    (rng.normal(icept + obs_offset[k] + s * obs_gain[k] * n,
                                34.0, N_REP) +
                     rng.exponential(52.0, N_REP)).mean()
                    for k in range(N_OBS)])
                cell_mu.append(per_obs.mean())
                cell_se.append(per_obs.std(ddof=1) / np.sqrt(N_OBS))
            key = (name, present)
            means[key] = np.array(cell_mu)
            sems[key] = np.array(cell_se)
            b, a0 = np.polyfit(SET_SIZES, means[key], 1)
            fitted[key] = (float(b), float(a0))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: response time against set size -------------------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    xs = np.linspace(2.0, 34.0, 60)
    for name, _, _, _, col in TASKS:
        for present in (True, False):
            key = (name, present)
            b, a0 = fitted[key]
            ax.plot(xs, a0 + b * xs, color=col, lw=1.35 if present else 1.05,
                    ls="-" if present else (0, (3.4, 2.0)), zorder=3)
            ax.errorbar(SET_SIZES, means[key], yerr=1.96 * sems[key],
                        fmt="o" if present else "s",
                        ms=4.2 if present else 3.6,
                        mfc=col if present else "white", mec=col, mew=0.9,
                        ecolor=col, elinewidth=0.8, capsize=1.8, ls="none",
                        zorder=5)

    handles = []
    for name, _, _, _, col in TASKS:
        b_p = fitted[(name, True)][0]
        b_a = fitted[(name, False)][0]
        handles.append(Line2D([], [], color=col, lw=1.5, marker="o", ms=4.2,
                              mfc=col, mec=col,
                              label=f"{name}: {b_p:.1f} present / "
                                    f"{b_a:.1f} absent ms per item"))
    handles += [
        Line2D([], [], color=SUB, lw=1.35, marker="o", ms=4.2, mfc=SUB,
               mec=SUB, label="target present, filled, solid fit"),
        Line2D([], [], color=SUB, lw=1.05, ls=(0, (3.4, 2.0)), marker="s",
               ms=3.6, mfc="white", mec=SUB,
               label="target absent, open, dashed fit"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.9,
              handlelength=2.1, labelspacing=0.38, borderpad=0.12,
              bbox_to_anchor=(-0.004, 1.004))

    ax.text(1.6, 2430.0,
            f"n={N_OBS} observers, {N_REP} correct trials per cell\n"
            "error bars are 95 % CI over observers\n"
            "lines are OLS fits through the plotted means",
            ha="left", va="top", fontsize=6.5, color=MUTED,
            linespacing=1.5)

    ax.set_xlim(1.0, 33.0)
    ax.set_ylim(420.0, 2820.0)
    ax.set_xticks(SET_SIZES)
    ax.set_yticks(np.arange(500, 2801, 400))
    ax.set_xlabel("display set size (items)", fontsize=8.5)
    ax.set_ylabel("mean correct response time (ms)", fontsize=8.5)
    ax.set_title("A flat slope means the target pops out; a steep one means "
                 "it is searched for", fontsize=9.0, fontweight="bold",
                 pad=5.0)

    # ---- aux (b): present and absent slopes side by side --------------------
    names = [t[0] for t in TASKS]
    slope_p = np.array([fitted[(n, True)][0] for n in names])
    slope_a = np.array([fitted[(n, False)][0] for n in names])
    ratio = slope_a / np.maximum(slope_p, 1e-9)

    pos = np.arange(len(names))
    ax_t.grid(axis="y", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_t.bar(pos - 0.19, slope_p, width=0.36, color=[t[4] for t in TASKS],
             edgecolor="white", linewidth=0.7, zorder=3)
    ax_t.bar(pos + 0.19, slope_a, width=0.36,
             color=[lighten(t[4], 0.55) for t in TASKS], edgecolor="white",
             linewidth=0.7, zorder=3)
    for k in pos:
        ax_t.text(k, max(slope_p[k], slope_a[k]) + 2.0,
                  f"absent / present = {ratio[k]:.2f}x", ha="center",
                  va="bottom", fontsize=6.2, color=INK)
        ax_t.text(k - 0.19, slope_p[k] + 1.6, f"{slope_p[k]:.1f}",
                  ha="center", va="bottom", fontsize=6.0, color=TASKS[k][4])
    ax_t.axhline(10.0, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=2)
    ax_t.text(-0.55, 11.0, "10 ms/item efficiency line", ha="left",
              va="bottom", fontsize=6.2, color=MUTED)
    ax_t.set_xticks(pos)
    ax_t.set_xticklabels([t[1] for t in TASKS], fontsize=6.3,
                         linespacing=1.25)
    ax_t.set_xlim(-0.60, len(names) - 0.40)
    ax_t.set_ylim(0.0, 88.0)
    ax_t.tick_params(axis="y", labelsize=6.6)
    ax_t.set_ylabel("search slope (ms / item)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Absent slopes run near twice the present ones",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    # ---- aux (c): miss rate against set size --------------------------------
    hardest = TASKS[-1][3]
    n_miss = N_OBS * N_REP * 5
    ax_b.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    for name, _, _, slope, col in TASKS:
        p = MISS_BASE + MISS_SLOPE * (slope / hardest) * SET_SIZES
        obs = rng.binomial(n_miss, p) / n_miss
        ax_b.plot(SET_SIZES, 100 * obs, marker="o", ms=3.6, lw=1.3, color=col,
                  mfc="white", mec=col, mew=0.9, zorder=4)
        if slope == hardest:
            b_m = float(np.polyfit(SET_SIZES, 100 * obs, 1)[0])
            ax_b.text(32.0, 100 * obs[-1] + 0.55,
                      f"{b_m:.2f} % per item", ha="right", va="bottom",
                      fontsize=6.3, color=col)
    ax_b.set_xlim(1.0, 34.0)
    ax_b.set_ylim(0.0, 11.5)
    ax_b.set_xticks(SET_SIZES)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("display set size (items)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("miss rate (%)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("The hard searches also miss more targets", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.text(2.0, 10.9,
              f"n={n_miss} target-present trials per point; "
              "colours match the main panel", ha="left", va="top",
              fontsize=6.2, color=MUTED)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "957_cog_visual_search.png")


if __name__ == "__main__":
    main()
