"""Energy efficiency of 27 node configurations, and where the watts go.

The main panel is the trade-off itself: sustained performance against wall
power, with the bubble area carrying the ratio of the two, so the reader can
see that the fastest node and the most efficient node are not the same node.
The Pareto front - configurations that nothing else beats on both axes at
once - is dashed through them. The two side panels explain the shape: clock
frequency buys performance linearly but power super-linearly, so efficiency
peaks well below the nominal clock, and the component breakdown shows how
much of the budget never reaches a floating-point unit at all.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

BLUE = "#3b5b92"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"
GREY = "#808080"
ACCENT = "#b23a48"     # Pareto front, the single emphasis object
SUB = "#555555"
MUTED = "#777777"

FAMILIES = [("CPU only", BLUE, 9, (185.0, 325.0), (410.0, 980.0)),
            ("GPU stock", GOLD, 9, (275.0, 465.0), (2150.0, 4250.0)),
            ("GPU power-capped", GREEN, 9, (195.0, 340.0), (1850.0, 3650.0))]

# Component power in W, in stacking order.
COMPONENTS = [("CPU", BLUE), ("GPU", PURPLE), ("Memory", GREEN),
              ("Network", GOLD), ("Cooling", GREY)]
BREAKDOWN = {
    "CPU only":     [172.0,   0.0,  58.0, 22.0, 44.0],
    "GPU stock":    [ 96.0, 268.0,  72.0, 24.0, 62.0],
    "GPU capped":   [ 88.0, 158.0,  66.0, 24.0, 41.0],
    "GPU + liquid": [ 92.0, 172.0,  68.0, 24.0, 18.0],
}

# Frequency model: performance grows almost linearly with the clock while
# power grows super-linearly on top of a large static floor, so the ratio has
# an interior maximum.
PMAX, F_HALF = 5500.0, 6.00     # GFLOP/s saturation model
P_STATIC, P_DYN, P_EXP = 92.0, 6.0, 2.40
F_NOMINAL = 2.80                # GHz


def pareto_front(power, perf):
    """Indices that no other point beats on both low power and high perf."""
    keep = []
    for i in range(power.size):
        dominated = np.any((power <= power[i]) & (perf >= perf[i]) &
                           ((power < power[i]) | (perf > perf[i])))
        if not dominated:
            keep.append(i)
    return np.array(sorted(keep, key=lambda k: power[k]))


def main():
    set_pub_style()
    rng = np.random.default_rng(5691)

    power, perf, fam = [], [], []
    for k, (name, col, n, pw, pf) in enumerate(FAMILIES):
        p = rng.uniform(*pw, n)
        q = np.interp(p, pw, pf) * np.exp(rng.normal(0.0, 0.085, n))
        power.append(p)
        perf.append(q)
        fam.append(np.full(n, k))
    power = np.concatenate(power)
    perf = np.concatenate(perf)
    fam = np.concatenate(fam)
    eff = perf / power
    front = pareto_front(power, perf)
    best = int(np.argmax(eff))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.90], hspace=0.55, wspace=0.26)
    ax = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) performance vs power ----------------------------------------
    size = 6.0 * eff + 8.0
    for k, (name, col, _, _, _) in enumerate(FAMILIES):
        m = fam == k
        ax.scatter(power[m], perf[m], s=size[m], color=col, alpha=0.82,
                   edgecolors="white", linewidths=0.7, zorder=3, label=name)
    ax.plot(power[front], perf[front], ls="--", lw=1.3, color=ACCENT,
            zorder=4, alpha=0.9)
    ax.scatter(power[front], perf[front], s=size[front], facecolors="none",
               edgecolors=ACCENT, linewidths=1.2, zorder=5)
    ax.scatter([power[best]], [perf[best]], marker="*", s=110, color=ACCENT,
               edgecolors="white", linewidths=0.7, zorder=6)
    ax.annotate(f"best efficiency\n{eff[best]:.1f} GFLOP/W",
                xy=(power[best], perf[best]),
                xytext=(power[best] - 22.0, perf[best] + 620.0),
                ha="center", va="bottom", fontsize=6.8, color=ACCENT,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                shrinkA=2.0, shrinkB=4.0))

    ax.set_xlabel("node wall power (W)", fontsize=8.0)
    ax.set_ylabel("sustained performance (GFLOP/s)", fontsize=8.0)
    ax.set_title("Performance is bought with watts, efficiency is not",
                 fontsize=8.8, fontweight="bold")
    ax.tick_params(labelsize=7.0)
    ax.grid(lw=0.5, alpha=0.45)
    ax.set_ylim(0.0, perf.max() * 1.22)

    ref = np.array([4.0, 8.0, 12.0])
    handles = [Line2D([], [], color=c, marker="o", ms=5.0, ls="none",
                      mec="white", mew=0.7, label=n)
               for n, c, _, _, _ in FAMILIES]
    handles.append(Line2D([], [], color=ACCENT, ls="--", lw=1.3,
                          label="Pareto front"))
    handles += [Line2D([], [], color=MUTED, marker="o", ls="none",
                       mec="white", mew=0.6,
                       ms=np.sqrt(6.0 * r + 8.0), label=f"{r:.0f} GFLOP/W")
                for r in ref]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False,
              handlelength=1.6, labelspacing=0.42, ncol=2, columnspacing=1.4)

    # ---- (b) efficiency against clock frequency --------------------------
    f = np.linspace(1.0, 3.2, 220)
    p_f = PMAX * f / (f + F_HALF)
    w_f = P_STATIC + P_DYN * f ** P_EXP
    e_f = p_f / w_f
    i_opt = int(np.argmax(e_f))
    f_opt, e_opt = float(f[i_opt]), float(e_f[i_opt])
    e_nom = float(np.interp(F_NOMINAL, f, e_f))
    p_nom = float(np.interp(F_NOMINAL, f, p_f))
    p_opt = float(np.interp(f_opt, f, p_f))

    ax_b.plot(f, e_f, lw=1.5, color=BLUE)
    ax_b.axvline(F_NOMINAL, color=GREY, ls="--", lw=0.9)
    ax_b.plot([f_opt], [e_opt], marker="o", ms=5.0, color=ACCENT,
              mec="white", mew=0.8, ls="none", zorder=5)
    ax_b.annotate(f"optimum {f_opt:.2f} GHz\n{e_opt:.1f} GFLOP/W",
                  xy=(f_opt, e_opt), xytext=(f_opt + 0.16, e_opt * 0.90),
                  ha="left", va="top", fontsize=6.3, color=ACCENT,
                  linespacing=1.30,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                  shrinkA=1.5, shrinkB=2.5))
    ax_b.text(F_NOMINAL - 0.05, e_f.min() * 1.02,
              f"nominal {F_NOMINAL:.1f} GHz", rotation=90, ha="right",
              va="bottom", fontsize=6.3, color=SUB)

    ax_b.set_xlim(1.0, 3.2)
    ax_b.set_xlabel("core frequency (GHz)", fontsize=7.4)
    ax_b.set_ylabel("efficiency (GFLOP/W)", fontsize=7.4)
    ax_b.set_title("Efficiency peaks below the nominal clock", fontsize=8.5,
                   fontweight="bold")
    ax_b.tick_params(labelsize=6.8, length=2.4)
    ax_b.grid(lw=0.45, alpha=0.45)

    # ---- (c) where the watts go ------------------------------------------
    names = list(BREAKDOWN)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for j, (label, col) in enumerate(COMPONENTS):
        vals = np.array([BREAKDOWN[n][j] for n in names])
        ax_c.barh(y, vals, left=left, height=0.60, color=col,
                  edgecolor="white", linewidth=0.6, label=label)
        left += vals
    for k, n in enumerate(names):
        ax_c.text(left[k] + 6.0, y[k], f"{left[k]:.0f} W", ha="left",
                  va="center", fontsize=6.3, color=SUB)

    ax_c.set_yticks(y)
    ax_c.set_yticklabels(names, fontsize=6.8)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.0, left.max() * 1.22)
    ax_c.set_xlabel("component power (W)", fontsize=7.4)
    ax_c.set_title("Component power budget", fontsize=8.5, fontweight="bold")
    ax_c.tick_params(labelsize=6.8, length=2.4)
    ax_c.grid(axis="x", lw=0.45, alpha=0.45)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=5,
                fontsize=6.5, frameon=False, handlelength=1.2,
                columnspacing=1.0, handletextpad=0.4)

    cool = np.array([BREAKDOWN[n][-1] / sum(BREAKDOWN[n]) for n in names])

    fig.tight_layout(rect=[0, 0.085, 1, 0.945], w_pad=2.4, h_pad=2.4)
    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.50, dy=0.08)
    savefig(fig, "569_hpc_energy_efficiency.png")


if __name__ == "__main__":
    main()
