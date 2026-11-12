"""Cumulative dose-volume histograms for a synthetic radiotherapy plan.

One voxel-dose sample per structure is drawn once and every curve, every
constraint marker and every bar in this figure is derived from that same
sample.  Because a cumulative DVH is evaluated as "fraction of voxels at or
above dose d" with ``searchsorted`` on the sorted sample, the curves are
monotonically non-increasing by construction rather than by inspection.

The main panel carries the conclusion (target coverage against organ-at-risk
sparing); the two smaller panels on the right carry the evidence: the mean and
maximum dose of every structure against its planning limit, and the change a
sparing re-optimisation buys at every dose level.

Synthetic illustration for a visual reference library - not patient data.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FIGW, FIGH = 10.4, 6.4
RX = 60.0                      # prescription dose (Gy)
N_VOXEL = 24000

# House qualitative order; the target takes the single accent colour.
STRUCTS = [
    ("PTV", "#b23a48"),
    ("Spinal cord", "#3b5b92"),
    ("Lung ipsi", "#2e7d5b"),
    ("Heart", "#c9a227"),
    ("Esophagus", "#8e6bbf"),
]

# Plan B = organ-at-risk sparing re-optimisation of plan A.
SPARE = {"PTV": 1.004, "Spinal cord": 1.07, "Lung ipsi": 0.86,
         "Heart": 0.79, "Esophagus": 0.93}

# (structure, label, dose level in Gy, volume limit in %)
CONSTRAINTS = [
    ("Lung ipsi", "V20", 20.0, 30.0),
    ("Heart", "V25", 25.0, 20.0),
    ("Esophagus", "V50", 50.0, 25.0),
]

# Planning limits used by the bar panel: (structure, metric, limit in Gy)
DOSE_LIMITS = [
    ("Spinal cord", "max", 45.0),
    ("Lung ipsi", "mean", 20.0),
    ("Heart", "mean", 26.0),
    ("Esophagus", "mean", 34.0),
]


def sample_doses(rng, n=N_VOXEL):
    """Per-voxel dose samples (Gy) for each delineated structure."""
    lung_hot = rng.random(n) < 0.42
    return {
        "PTV": np.clip(rng.normal(1.013 * RX, 0.021 * RX, n),
                       0.0, 1.10 * RX),
        "Spinal cord": rng.beta(1.15, 3.1, n) * 0.72 * RX,
        "Lung ipsi": np.where(lung_hot,
                              rng.beta(2.2, 2.7, n) * RX,
                              rng.beta(0.85, 6.0, n) * RX),
        "Heart": rng.beta(1.05, 4.4, n) * 0.86 * RX,
        "Esophagus": rng.beta(1.7, 2.3, n) * 0.97 * RX,
    }


def dvh(doses, grid):
    """Cumulative DVH in %: monotone non-increasing by construction."""
    s = np.sort(np.asarray(doses, dtype=float))
    above = s.size - np.searchsorted(s, grid, side="left")
    return 100.0 * above / s.size


def dose_at_volume(grid, curve, vol):
    """Dose (Gy) covering ``vol`` % of the structure, read off the curve."""
    return float(np.interp(vol, curve[::-1], grid[::-1]))


def volume_at_dose(grid, curve, dose):
    """Volume (%) at or above ``dose``, read off the curve."""
    return float(np.interp(dose, grid, curve))


def main():
    set_pub_style()
    rng = np.random.default_rng(870)
    grid = np.linspace(0.0, 1.12 * RX, 673)

    dose_a = sample_doses(rng)
    dose_b = {k: v * SPARE[k] for k, v in dose_a.items()}
    dvh_a = {k: dvh(v, grid) for k, v in dose_a.items()}
    dvh_b = {k: dvh(v, grid) for k, v in dose_b.items()}

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = axes_inches(fig, (0.82, 1.02, 5.30, 4.34))
    ax_bar = axes_inches(fig, (7.16, 3.70, 2.84, 1.74))
    ax_dif = axes_inches(fig, (7.16, 0.95, 2.84, 1.74))
    for a in (ax, ax_bar, ax_dif):
        despine(a)

    # ---- main panel: cumulative DVH ---------------------------------------
    for name, col in STRUCTS:
        ax.plot(grid, dvh_a[name], color=col, lw=1.7 if name == "PTV" else 1.3,
                label=name, zorder=4 if name == "PTV" else 3)
        ax.plot(grid, dvh_b[name], color=col, lw=0.85, ls=(0, (2.6, 1.5)),
                alpha=0.75, zorder=2)

    ax.axvline(RX, color=GREY, lw=0.9, ls=(0, (4.0, 2.2)), zorder=1)
    ax.text(RX - 0.8, 66.0, f"prescription {RX:.0f} Gy", rotation=90,
            ha="right", va="top", fontsize=6.6, color=SUB)

    d95 = dose_at_volume(grid, dvh_a["PTV"], 95.0)
    d2 = dose_at_volume(grid, dvh_a["PTV"], 2.0)
    ax.plot([d95], [95.0], marker="o", ms=4.2, mfc="white",
            mec="#b23a48", mew=1.2, zorder=6)
    ax.annotate(f"$D_{{95}}$ = {d95:.1f} Gy\n({d95 / RX * 100:.0f} % of Rx)",
                xy=(d95, 95.0), xytext=(0.585, 0.415),
                textcoords=ax.transAxes,
                fontsize=6.6, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED,
                                shrinkA=1.0, shrinkB=2.0))

    lines = []
    for name, tag, lvl, lim in CONSTRAINTS:
        col = dict(STRUCTS)[name]
        val = volume_at_dose(grid, dvh_a[name], lvl)
        ok = val < lim
        ax.plot([lvl], [val], marker="o" if ok else "X", ms=5.0 if ok else 5.6,
                mfc=col if ok else "white", mec=col, mew=1.2, zorder=7)
        lines.append(f"{tag} {name:<11s} {val:5.1f} %  limit {lim:.0f} %  "
                     f"{'PASS' if ok else 'FAIL'}")
    ax.text(0.425, 0.645, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.4, color=SUB,
            family="DejaVu Sans Mono", linespacing=1.75)

    ax.set_xlim(0.0, 1.12 * RX)
    ax.set_ylim(0.0, 101.5)
    ax.set_xlabel("absorbed dose (Gy)", fontsize=8.0)
    ax.set_ylabel("relative structure volume (%)", fontsize=8.0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(labelsize=7.0)
    ax.grid(True, color="#e4e4e4", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=GREY, lw=1.5),
                Line2D([], [], color=GREY, lw=1.0, ls=(0, (2.6, 1.5)))]
    labels += ["plan A", "plan B"]
    leg = ax.legend(handles, labels, loc="lower left", ncol=7, frameon=False,
                    bbox_to_anchor=(-0.005, 1.005), fontsize=6.9,
                    handlelength=1.6, columnspacing=1.15,
                    handletextpad=0.45)
    for h in leg.legend_handles[:5]:
        h.set_linewidth(1.6)

    # ---- aux 1: mean and maximum dose against the planning limit ----------
    names = [s[0] for s in STRUCTS]
    cols = [s[1] for s in STRUCTS]
    dmean = np.array([dose_a[n].mean() for n in names])
    dmax = np.array([dose_a[n].max() for n in names])
    pos = np.arange(len(names), dtype=float)

    ax_bar.bar(pos - 0.20, dmean, width=0.38, color=cols, edgecolor="white",
               linewidth=0.5, zorder=3)
    ax_bar.bar(pos + 0.20, dmax, width=0.38, color="none", edgecolor=cols,
               linewidth=1.0, hatch="////", zorder=3)
    for name, metric, lim in DOSE_LIMITS:
        i = names.index(name)
        off = -0.20 if metric == "mean" else 0.20
        ax_bar.plot([i + off - 0.21, i + off + 0.21], [lim, lim],
                    color="#3a3a3a", lw=1.1, ls=(0, (2.2, 1.3)), zorder=5)
    ax_bar.legend(handles=[
        Patch(facecolor=GREY, edgecolor="white", label="$D_{mean}$"),
        Patch(facecolor="none", edgecolor=GREY, hatch="////",
              label="$D_{max}$"),
        Line2D([], [], color="#3a3a3a", lw=1.1, ls=(0, (2.2, 1.3)),
               label="limit")],
        loc="upper center", ncol=3, frameon=False, fontsize=6.2,
        handlelength=1.4, handletextpad=0.45, columnspacing=1.1,
        borderpad=0.1, bbox_to_anchor=(0.5, 1.035))
    ax_bar.set_xticks(pos)
    ax_bar.set_xticklabels(["PTV", "Spinal\ncord", "Lung\nipsi", "Heart",
                            "Esoph."], fontsize=6.2)
    ax_bar.tick_params(axis="y", labelsize=6.5)
    ax_bar.tick_params(axis="x", pad=1.5, length=0)
    ax_bar.set_ylabel("dose (Gy)", fontsize=7.2)
    ax_bar.set_ylim(0.0, 1.42 * dmax.max())
    ax_bar.set_title("Dose statistics against planning limits",
                     fontsize=8.5, fontweight="bold", color=INK, pad=4.0)

    # ---- aux 2: DVH difference, plan B minus plan A -----------------------
    for name, col in STRUCTS:
        ax_dif.plot(grid, dvh_b[name] - dvh_a[name], color=col, lw=1.15)
    ax_dif.axhline(0.0, color=GREY, lw=0.8)
    ax_dif.axvline(RX, color=HAIR, lw=0.8, ls=(0, (3.0, 2.0)))
    gain = {n: float(np.trapezoid(dvh_a[n] - dvh_b[n], grid)) for n, _ in STRUCTS}
    best = max((k for k in gain if k != "PTV"), key=lambda k: gain[k])
    ax_dif.text(0.975, 0.06,
                f"largest sparing: {best}\n"
                f"integral $\\Delta$DVH = {gain[best]:.0f} % Gy",
                transform=ax_dif.transAxes, ha="right", va="bottom",
                fontsize=6.4, color=SUB, linespacing=1.6)
    ax_dif.set_xlim(0.0, 1.12 * RX)
    ax_dif.set_xlabel("absorbed dose (Gy)", fontsize=7.2)
    ax_dif.set_ylabel("$\\Delta$ volume, B - A (%)", fontsize=7.2)
    ax_dif.tick_params(labelsize=6.5)
    ax_dif.set_title("Volume shifted by the sparing re-plan",
                     fontsize=8.5, fontweight="bold", color=INK, pad=4.0)



    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.36)
    panel_labels_inches(fig, [ax_bar, ax_dif], labels=["(b)", "(c)"],
                        dx=-0.62, dy=0.20)
    savefig(fig, "870_rtx_dvh_curves.png")


if __name__ == "__main__":
    main()
