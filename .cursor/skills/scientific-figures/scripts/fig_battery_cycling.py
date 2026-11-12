"""Rate-capability and cycling plot for three Li-ion cathodes.

Left axis: discharge specific capacity per cycle; right axis: coulombic
efficiency of the best cell.  The current rate is stepped every 20 cycles and
each step is called out by a tinted background band.  The inset shows the
galvanostatic charge-discharge profiles of selected cycles.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig

RATES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 0.1]        # C-rate per 20-cycle step
SEG = 20

CELLS = [
    dict(name="NCM811@Li$_3$PO$_4$", c0=202.0, col="#b5482f",
         keep=[1.00, 0.945, 0.892, 0.825, 0.715, 0.605, 0.986], fade=1.0e-4),
    dict(name="NCM811", c0=195.0, col="#3f7f8c",
         keep=[1.00, 0.912, 0.845, 0.742, 0.585, 0.438, 0.958], fade=4.0e-4),
    dict(name="bare NCM523", c0=181.0, col="#7a7f4a",
         keep=[1.00, 0.878, 0.784, 0.652, 0.451, 0.286, 0.918], fade=7.0e-4),
]

BAND = {0.1: "#eef3f7", 0.5: "#f3f2ea", 1.0: "#faf0e0",
        2.0: "#f8e6d2", 5.0: "#f4dbc6", 10.0: "#efcdb6"}


def capacity_series(cell, rng):
    cyc, cap = [], []
    for k, r in enumerate(RATES):
        for i in range(SEG):
            n = k * SEG + i + 1
            base = cell["c0"] * cell["keep"][k]
            val = base * (1.0 - cell["fade"] * n) * (1.0 - 0.004 * i / SEG)
            cyc.append(n)
            cap.append(val + rng.normal(0.0, 0.55))
    return np.array(cyc), np.array(cap)


def profile(q_max, fade=0.0):
    """Galvanostatic discharge / charge branch of an NCM cell."""
    s = np.linspace(0.0, 1.0, 400)
    v_d = 3.62 + 0.68 * (1.0 - s) ** 1.2 - 0.85 * s ** 12 - 0.05 * fade
    v_c = (3.62 + 0.68 * s ** 1.2 - 0.85 * (1.0 - s) ** 12) + 0.075 + \
        0.05 * fade
    return s * q_max, v_d, s * q_max * 1.006, v_c


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(11)

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    axr = ax.twinx()

    n_tot = SEG * len(RATES)
    ax.set_xlim(0, n_tot + 1)
    ax.set_ylim(0, 288)
    axr.set_ylim(75, 105)

    # --- rate bands --------------------------------------------------------
    for k, r in enumerate(RATES):
        x0, x1 = k * SEG + 0.5, (k + 1) * SEG + 0.5
        ax.axvspan(x0, x1, facecolor=BAND[r], lw=0, zorder=0)
        if k:
            ax.axvline(x0, color="#c9c9c9", lw=0.6, ls=(0, (3, 2.5)),
                       zorder=1)
        ax.text(0.5 * (x0 + x1), 272, f"{r:g}C", ha="center", va="center",
                fontsize=7.4, fontweight="bold", color="#5c4a3a")

    # --- capacity curves ---------------------------------------------------
    best = None
    for cell in CELLS:
        cyc, cap = capacity_series(cell, rng)
        ax.plot(cyc, cap, ls="none", marker="o", ms=3.0, mfc=cell["col"],
                mec="white", mew=0.35, color=cell["col"], zorder=5,
                label=cell["name"])
        if best is None:
            best = (cyc, cap, cell)

    # --- coulombic efficiency ---------------------------------------------
    cyc, cap, cell = best
    ce = 99.55 - 0.35 * np.exp(-(cyc - 1) / 4.0) + rng.normal(0, 0.055,
                                                              cyc.size)
    ce[0] = 88.6
    for k in range(1, len(RATES)):
        ce[k * SEG] -= 0.45                       # dip when the rate changes
    axr.plot(cyc, ce, ls="none", marker="D", ms=2.1, mfc="none",
             mec="#4d5b6b", mew=0.55, zorder=4)
    axr.set_ylabel("Coulombic efficiency (%)", labelpad=3)
    axr.set_yticks([80, 85, 90, 95, 100])
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color("#4d5b6b")
    axr.tick_params(colors="#4d5b6b", labelsize=7.5)
    axr.annotate("1st-cycle CE 88.6%", xy=(1, 88.6), xytext=(24, 87.4),
                 fontsize=6.4, color="#4d5b6b", ha="left",
                 arrowprops=dict(arrowstyle="-|>", color="#4d5b6b", lw=0.6,
                                 shrinkB=1.5, mutation_scale=6))

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (mAh g$^{-1}$)")
    ax.set_xticks(np.arange(0, n_tot + 1, 20))
    ax.set_yticks(np.arange(0, 241, 40))
    ax.spines["top"].set_visible(False)

    ax.annotate("97.2% recovery\nat 0.1C", xy=(130, 199), xytext=(101, 252),
                fontsize=6.8, color="#b5482f", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#b5482f", lw=0.7,
                                shrinkB=2.0, mutation_scale=7))

    handles = [Line2D([], [], ls="none", marker="o", ms=3.6,
                      mfc=c["col"], mec="white", mew=0.4, label=c["name"])
               for c in CELLS]
    handles.append(Line2D([], [], ls="none", marker="D", ms=2.6, mfc="none",
                          mec="#4d5b6b", mew=0.7,
                          label="coulombic efficiency"))

    # --- inset: charge / discharge profiles --------------------------------
    ins = ax.inset_axes([0.075, 0.055, 0.385, 0.295])
    marks = [(1, 202.0, 0.0, "#c98b6b", 8), (70, 172.0, 0.5, "#c0603f", 82),
             (140, 150.0, 1.0, "#8a2f2f", 156)]
    for cyc_id, q, fade, col, xlab in marks:
        qd, vd, qc, vc = profile(q, fade)
        ins.plot(qd, vd, color=col, lw=0.9)
        ins.plot(qc, vc, color=col, lw=0.9)
        ins.text(xlab, 4.70, f"cycle {cyc_id}", fontsize=6.0, color=col,
                 ha="left", va="center", fontweight="bold")
    ins.set_xlim(0, 232)
    ins.set_ylim(2.6, 4.88)
    ins.set_xticks([0, 100, 200])
    ins.set_yticks([3.0, 3.5, 4.0, 4.5])
    ins.tick_params(labelsize=6.0, length=1.8, pad=1.2)
    ins.set_xlabel("Capacity (mAh g$^{-1}$)", fontsize=6.0, labelpad=0.8)
    ins.set_ylabel("Voltage (V)", fontsize=6.0, labelpad=1.0)
    ins.set_title("charge–discharge profiles, 1C", fontsize=6.0, pad=2.0)
    for s in ins.spines.values():
        s.set_linewidth(0.6)
        s.set_edgecolor("#8a8a8a")
    ins.spines["top"].set_visible(False)
    ins.spines["right"].set_visible(False)
    ins.set_facecolor("white")
    ins.patch.set_alpha(0.94)
    ins.annotate("", xy=(112, 4.28), xytext=(58, 4.28),
                 arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=0.6,
                                 mutation_scale=5))
    ins.text(85, 4.33, "charge", fontsize=6.0, color="#6b6b6b",
             ha="center", va="bottom")
    ins.annotate("", xy=(58, 3.42), xytext=(112, 3.42),
                 arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=0.6,
                                 mutation_scale=5))
    ins.text(85, 3.30, "discharge", fontsize=6.0, color="#6b6b6b",
             ha="center", va="top")

    fig.subplots_adjust(left=0.075, right=0.915, top=0.875, bottom=0.11)
    savefig(fig, "123_battery_cycling.png")


if __name__ == "__main__":
    main()
