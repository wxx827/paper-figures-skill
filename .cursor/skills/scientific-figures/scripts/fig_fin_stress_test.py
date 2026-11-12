"""Multi-scenario stress test of a synthetic trading book over twelve quarters.

(a) portfolio value under four supervisory scenarios against the regulatory
    capital floor, with the breach region highlighted and the simulation band
    of the tail scenario shaded; (b) a waterfall from the baseline value to the
    tail-event value through the three risk drivers; (c) the same
    market / credit / operational split for every stress scenario.

Panels (b) and (c) are computed from the very arrays plotted in (a), so the
waterfall closes exactly on the tail path.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches

BLUE = "#3b5b92"
GOLD = "#c9a227"
RUST = "#c0603f"
DEEPRED = "#8c2f2f"
RED = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

V0 = 1000.0            # USD mn, book value at Q0
FLOOR = 862.0          # USD mn, regulatory capital floor
N_Q = 12
PROF_NOISE = 0.035     # quarterly dispersion of the loss profile

# Peak cumulative loss versus the baseline path, by risk driver (USD mn).
LOSSES = {
    "Baseline": {"Market": 0.0, "Credit": 0.0, "Operational": 0.0},
    "Mild recession": {"Market": 26.0, "Credit": 19.0, "Operational": 7.0},
    "Severe recession": {"Market": 62.0, "Credit": 48.0, "Operational": 14.0},
    "Tail event": {"Market": 104.0, "Credit": 71.0, "Operational": 23.0},
}
SCEN_COLOR = {"Baseline": BLUE, "Mild recession": GOLD,
              "Severe recession": RUST, "Tail event": DEEPRED}
DRIVER_COLOR = {"Market": RED, "Credit": RUST, "Operational": GOLD}


def loss_profiles(rng, q):
    """Unit-peak loss profiles p(t) in [0, 1]; one shape per scenario."""
    t = q / q[-1]
    env = np.sin(np.pi * np.clip(t, 0, 1)) ** 1.5
    prof = {
        "Baseline": np.zeros_like(t),
        "Mild recession": np.clip(1.28 * t ** 0.75 - 0.42 * t ** 2.4, 0, None),
        "Severe recession": np.clip(1.10 * t ** 0.9, 0, None),
        "Tail event": np.clip(1.34 * (1.0 - np.exp(-4.2 * t))
                              - 0.30 * t ** 1.8, 0, None),
    }
    for k, p in prof.items():
        if k == "Baseline":
            continue
        prof[k] = np.clip(p + rng.normal(0, PROF_NOISE, p.size) * env,
                          0.0, None)
        prof[k][0] = 0.0
    return prof


def panel_paths(ax, q, baseline, paths, prof):
    tail = paths["Tail event"]
    band_lo = baseline - (baseline - tail) * 1.16
    band_hi = baseline - (baseline - tail) * 0.84
    ax.fill_between(q, band_lo, band_hi, color=DEEPRED, alpha=0.10, lw=0,
                    zorder=2, label="tail event, $\\pm$1 s.d. of 2000 paths")

    ax.axhline(FLOOR, color=RED, lw=1.3, ls=(0, (5, 2.5)), zorder=5)
    ax.text(q[0] + 0.12, FLOOR - 4.0,
            "regulatory capital floor  %.0f USD mn" % FLOOR,
            fontsize=6.8, color=RED, ha="left", va="top")

    breach = tail < FLOOR
    ax.fill_between(q, tail, FLOOR, where=breach, interpolate=True,
                    color=RED, alpha=0.30, lw=0, zorder=3)

    for name, path in paths.items():
        ax.plot(q, path, color=SCEN_COLOR[name], lw=1.7 if
                name == "Tail event" else 1.3, marker="o", ms=2.8,
                zorder=6, label=name)

    k = int(np.argmin(tail))
    shortfall = FLOOR - tail[k]
    ax.annotate("floor breached in Q%d\nshortfall %.0f USD mn" % (k, shortfall),
                xy=(q[k], tail[k]), xytext=(q[k] - 4.4, tail[k] - 34),
                fontsize=6.9, color=RED, ha="left", va="top", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.75, color=RED,
                                connectionstyle="arc3,rad=0.22"))

    ax.set_xlabel("quarter since stress start (quarters)")
    ax.set_ylabel("portfolio value (USD mn)")
    ax.set_xlim(-0.35, q[-1] + 0.35)
    ax.set_xticks(q[::2])
    ax.set_xticklabels(["Q%d" % v for v in q[::2]])
    ax.set_ylim(min(band_lo.min(), FLOOR) - 42, baseline.max() + 26)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.0, ncol=2, handlelength=1.6,
              columnspacing=1.2, borderaxespad=0.3)
    style_axes(ax)
    return k, shortfall


def panel_waterfall(ax, base_val, drivers, end_val):
    names = ["Baseline"] + list(drivers) + ["Tail event"]
    x = np.arange(len(names))
    lo = min(end_val, base_val) - 78
    ax.bar(x[0], base_val - lo, bottom=lo, width=0.56, color=BLUE,
           alpha=0.9, lw=0)
    ax.bar(x[-1], end_val - lo, bottom=lo, width=0.56, color=DEEPRED,
           alpha=0.9, lw=0)
    top = base_val
    for i, drv in enumerate(drivers, start=1):
        d = drivers[drv]
        ax.bar(x[i], d, bottom=top - d, width=0.56,
               color=DRIVER_COLOR[drv], alpha=0.92, lw=0)
        ax.plot([x[i - 1] + 0.28, x[i] + 0.28], [top, top], color="#9a9a9a",
                lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(x[i], top - d - 5.0, "-%.0f" % d, ha="center", va="top",
                fontsize=6.6, color=DRIVER_COLOR[drv])
        top -= d
    ax.plot([x[-2] + 0.28, x[-1] + 0.28], [top, top], color="#9a9a9a",
            lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(x[0], base_val + 5.0, "%.0f" % base_val, ha="center",
            va="bottom", fontsize=6.6, color=BLUE)
    ax.text(x[-1], end_val + 5.0, "%.0f" % end_val, ha="center", va="bottom",
            fontsize=6.6, color=DEEPRED)
    ax.axhline(FLOOR, color=RED, lw=1.0, ls=(0, (5, 2.5)), zorder=4)
    ax.text(x[-1] + 0.38, FLOOR, "floor", fontsize=6.4, color=RED,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["Base", "Market", "Credit", "Oper.", "Tail"],
                       fontsize=6.8)
    ax.set_ylabel("value (USD mn)")
    ax.set_ylim(lo, base_val + 34)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def panel_split(ax, split):
    names = list(split)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for drv in ("Market", "Credit", "Operational"):
        vals = np.array([split[s][drv] for s in names])
        ax.barh(y, vals, left=left, height=0.56, color=DRIVER_COLOR[drv],
                alpha=0.92, lw=0, label=drv)
        left = left + vals
    for yi, tot in zip(y, left):
        ax.text(tot + 3.5, yi, "%.0f" % tot, va="center", ha="left",
                fontsize=6.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace(" ", "\n") for n in names], fontsize=6.6,
                       linespacing=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("peak loss vs baseline (USD mn)")
    ax.set_xlim(0, left.max() * 1.24)
    ax.grid(axis="x", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=6.6,
              handlelength=1.2, borderaxespad=0.2, ncol=3, columnspacing=1.0)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(571)
    q = np.arange(N_Q + 1)
    baseline = V0 * (1.0 + 0.0055 * q)
    prof = loss_profiles(rng, q)

    paths, totals = {}, {}
    for name, drivers in LOSSES.items():
        tot = sum(drivers.values())
        totals[name] = tot
        paths[name] = baseline - tot * prof[name]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86], hspace=0.46,
                          wspace=0.24, left=0.065, right=0.975, top=0.905,
                          bottom=0.10)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_wf = fig.add_subplot(gs[0, 1])
    ax_sp = fig.add_subplot(gs[1, 1])

    k, shortfall = panel_paths(ax_main, q, baseline, paths, prof)

    # Waterfall is read off the tail path at its worst quarter, so the bars
    # add up to exactly the gap between the two lines in panel (a).
    scale = prof["Tail event"][k]
    tail_drivers = {d: v * scale for d, v in LOSSES["Tail event"].items()}
    panel_waterfall(ax_wf, baseline[k], tail_drivers, paths["Tail event"][k])

    split = {}
    for name in ("Mild recession", "Severe recession", "Tail event"):
        s = prof[name].max()
        split[name] = {d: v * s for d, v in LOSSES[name].items()}
    panel_split(ax_sp, split)

    worst = paths["Tail event"].min()

    panel_labels_inches(fig, [ax_main, ax_wf, ax_sp], dx=-0.42, dy=0.06)
    savefig(fig, "571_fin_stress_test.png")


if __name__ == "__main__":
    main()
