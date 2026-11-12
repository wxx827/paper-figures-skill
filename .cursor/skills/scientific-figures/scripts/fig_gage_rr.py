"""Gage R&R study (AIAG average-and-range method): operator interaction,
X-bar and R charts by operator, and the variance breakdown."""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
AMBER = "#c9922f"
PALE = "#c3d3e2"
GREY = "#8b949e"
ALERT = "#b23a48"

OPS = [("Operator A", NAVY), ("Operator B", BLUE), ("Operator C", AMBER)]
N_PART, N_TRIAL = 10, 3
A2, D4, D2 = 1.023, 2.574, 1.693          # subgroup size 3
K2, K3 = 0.5231, 0.3146                   # 3 operators, 10 parts


def stamp_labels(fig, axes, labels, dx=0.052, dy=0.014, size=9):
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + dy, lab, fontsize=size,
                 fontweight="bold", ha="left", va="bottom")


def make_data():
    rng = np.random.default_rng(5150)
    parts = 25.000 + np.sort(rng.normal(0, 0.125, N_PART))
    bias = np.array([0.021, -0.015, -0.006])
    m = np.empty((3, N_PART, N_TRIAL))
    for o in range(3):
        for p in range(N_PART):
            m[o, p] = parts[p] + bias[o] + rng.normal(0, 0.022, N_TRIAL)
    return m


def gage_stats(m):
    sub_mean = m.mean(axis=2)                      # operator x part
    sub_rng = np.ptp(m, axis=2)
    rbar = sub_rng.mean()
    ev = rbar / D2
    xdiff = np.ptp(m.mean(axis=(1, 2)))
    av = np.sqrt(max(0.0, (xdiff * K2) ** 2 - ev ** 2 / (N_PART * N_TRIAL)))
    pv = np.ptp(sub_mean.mean(axis=0)) * K3
    grr = np.hypot(ev, av)
    tv = np.hypot(grr, pv)
    return dict(sub_mean=sub_mean, sub_rng=sub_rng, rbar=rbar, ev=ev, av=av,
                pv=pv, grr=grr, tv=tv, pct_rr=100 * grr / tv)


def panel_interaction(ax, s, ylim):
    x = np.arange(1, N_PART + 1)
    for o, (name, col) in enumerate(OPS):
        ax.plot(x, s["sub_mean"][o], "-", color=col, lw=1.3, zorder=3)
        ax.plot(x, s["sub_mean"][o], "o", ms=3.4, mfc="white", mec=col,
                mew=1.0, zorder=4)
    ax.set_xlim(0.5, N_PART + 0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xlabel("Part number", fontsize=8)
    ax.set_ylabel("Measured length (mm)", fontsize=8)
    ax.set_title("Part \u00d7 operator interaction", fontsize=7.6,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.8)
    ax.yaxis.grid(True, color="#eef1f4", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def _operator_sections(ax, ytext):
    for o, (name, col) in enumerate(OPS):
        if o:
            ax.axvline(o * N_PART - 0.5, color="#c8ced4", lw=0.8,
                       ls=(0, (3, 2.5)), zorder=2)
        ax.text(o * N_PART + (N_PART - 1) / 2, ytext, name.split()[-1],
                ha="center", va="top", fontsize=6.6, fontweight="bold",
                color=col, zorder=6)


def panel_xbar(ax, s, ylim):
    v = s["sub_mean"].ravel()
    cl = v.mean()
    ucl, lcl = cl + A2 * s["rbar"], cl - A2 * s["rbar"]
    ax.axhspan(lcl, ucl, color="#f2f6fa", zorder=0)
    for o, (_, col) in enumerate(OPS):
        idx = np.arange(o * N_PART, (o + 1) * N_PART)
        ax.plot(idx, s["sub_mean"][o], "-", color=col, lw=1.1, zorder=3)
        ax.plot(idx, s["sub_mean"][o], "o", ms=2.8, mfc="white", mec=col,
                mew=0.9, zorder=4)
    ax.axhline(cl, color=GREY, lw=1.0, zorder=2)
    for y in (ucl, lcl):
        ax.axhline(y, color=ALERT, lw=0.9, ls=(0, (5, 2.5)), zorder=2)
    for y, lab, col in ((ucl, "UCL", ALERT), (cl, "CL", GREY),
                        (lcl, "LCL", ALERT)):
        ax.text(3 * N_PART - 0.6, y, f"{lab} {y:.3f}", ha="right",
                va="bottom", fontsize=6.2, fontweight="bold", color=col,
                zorder=6, bbox=dict(boxstyle="square,pad=0.16", fc="white",
                                    ec="none", alpha=0.85))
    ax.set_xlim(-0.8, 3 * N_PART - 0.2)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_ylabel("Subgroup mean (mm)", fontsize=8)
    ax.set_title("$\\bar{X}$ chart by operator", fontsize=7.6,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    _operator_sections(ax, ylim[1] - 0.012 * (ylim[1] - ylim[0]))
    ax.text(0.985, 0.035, "most points fall outside the limits \u2014 the "
            "gage resolves the parts", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.1, color=NAVY)


def panel_rchart(ax, s):
    cl = s["rbar"]
    ucl = D4 * cl
    ymax = max(ucl, s["sub_rng"].max()) * 1.30
    for o, (_, col) in enumerate(OPS):
        idx = np.arange(o * N_PART, (o + 1) * N_PART)
        ax.plot(idx, s["sub_rng"][o], "-", color=col, lw=1.1, zorder=3)
        ax.plot(idx, s["sub_rng"][o], "o", ms=2.8, mfc="white", mec=col,
                mew=0.9, zorder=4)
    ax.axhline(cl, color=GREY, lw=1.0, zorder=2)
    ax.axhline(ucl, color=ALERT, lw=0.9, ls=(0, (5, 2.5)), zorder=2)
    box = dict(boxstyle="square,pad=0.16", fc="white", ec="none", alpha=0.85)
    ax.text(3 * N_PART - 0.4, ucl, f"UCL {ucl:.3f}", ha="right", va="bottom",
            fontsize=6.2, fontweight="bold", color=ALERT, zorder=6, bbox=box)
    ax.text(3 * N_PART - 0.4, cl, f"$\\bar{{R}}$ {cl:.3f}", ha="right",
            va="bottom", fontsize=6.2, fontweight="bold", color=GREY,
            zorder=6, bbox=box)
    ax.set_xlim(-0.8, 3 * N_PART - 0.2)
    ax.set_ylim(0, ymax)
    ax.set_xticks([])
    ax.set_ylabel("Subgroup range (mm)", fontsize=8)
    ax.set_title("$R$ chart by operator", fontsize=7.6, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    _operator_sections(ax, ymax * 0.985)


def panel_variance(ax, s):
    ev2, av2, pv2 = s["ev"] ** 2, s["av"] ** 2, s["pv"] ** 2
    tot = ev2 + av2 + pv2
    parts = [("Repeatability", 100 * ev2 / tot, NAVY),
             ("Reproducibility", 100 * av2 / tot, BLUE),
             ("Part-to-part", 100 * pv2 / tot, PALE)]

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Percent (%)", fontsize=8)
    ax.tick_params(labelsize=6.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.text(0, 0.98, "VARIANCE CONTRIBUTION", fontsize=6.6, fontweight="bold",
            color=GREY, ha="left", va="top")
    left = 0.0
    for name, pct, col in parts:
        ax.barh(0.79, pct, left=left, height=0.15, color=col,
                edgecolor="white", linewidth=0.8, zorder=3)
        if pct > 20:
            ax.text(left + pct / 2, 0.79, f"{name}  {pct:.1f}%", ha="center",
                    va="center", fontsize=6.6, fontweight="bold",
                    color=NAVY, zorder=4)
        left += pct
    for i, (name, pct, col) in enumerate(parts[:2]):
        ax.plot([2 + i * 34], [0.60], "s", ms=4.6, color=col, zorder=4)
        ax.text(5 + i * 34, 0.60, f"{name}  {pct:.1f}%", ha="left",
                va="center", fontsize=6.4, color=INK, zorder=4)

    ax.text(0, 0.44, "GAGE R&R AS % OF STUDY VARIATION", fontsize=6.6,
            fontweight="bold", color=GREY, ha="left", va="top")
    for x0, x1, col in ((0, 10, "#eef3f8"), (10, 30, "#fdf5e8"),
                        (30, 100, "#fbf1f2")):
        ax.barh(0.22, x1 - x0, left=x0, height=0.19, color=col,
                edgecolor="none", zorder=2)
    ax.barh(0.22, s["pct_rr"], height=0.11, color=NAVY, edgecolor="none",
            zorder=4)
    for xv in (10, 30):
        ax.plot([xv, xv], [0.125, 0.315], color=ALERT, lw=0.9,
                ls=(0, (3, 2)), zorder=5)
    ax.text(0, 0.085, "acceptable < 10%   \u00b7   marginal 10\u201330%   "
            "\u00b7   unacceptable > 30%", ha="left", va="top", fontsize=6.1,
            color=GREY)
    ax.text(99, 0.235, f"{s['pct_rr']:.1f}%", ha="right", va="center",
            fontsize=15, fontweight="bold", color=AMBER, zorder=6)
    ax.set_title("Variance components", fontsize=7.6, fontweight="bold",
                 pad=4)


def main():
    set_pub_style()
    m = make_data()
    s = gage_stats(m)

    v = s["sub_mean"].ravel()
    cl = v.mean()
    ucl, lcl = cl + A2 * s["rbar"], cl - A2 * s["rbar"]
    lo, hi = min(v.min(), lcl), max(v.max(), ucl)
    pad = 0.14 * (hi - lo)
    ylim = (lo - pad, hi + pad)              # shared by panels (a) and (b)

    fig = plt.figure(figsize=(7.6, 5.3))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.46, left=0.082,
                          right=0.982, top=0.815, bottom=0.095)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    panel_interaction(axes[0], s, ylim)
    panel_xbar(axes[1], s, ylim)
    panel_rchart(axes[2], s)
    panel_variance(axes[3], s)
    stamp_labels(fig, axes, ["(a)", "(b)", "(c)", "(d)"])

    handles = [Line2D([], [], color=col, marker="o", ms=3.4, mfc="white",
                      mec=col, lw=1.3, label=name) for name, col in OPS]

    # The variance line is kept whole and the verdict moved to its own line:
    # as one line it ran past the canvas, so the bbox_inches="tight" export
    # expanded instead of clipping (DESIGN_SYSTEM 12.11).

    savefig(fig, "250_gage_rr.png")


if __name__ == "__main__":
    main()
