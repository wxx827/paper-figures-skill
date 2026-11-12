"""Figure 1193 - Roofline of one world-model inference step.

The roof is a function, not a drawing:

    attainable(I) = min(PEAK, BW * I)          PEAK = 312 TFLOP/s
                                               BW   = 2.04 TB/s
    ridge         = PEAK / BW                  where the two branches meet

Eight kernels are placed at their measured arithmetic intensity, and the
dotted stem on each one runs from what it reached to what the roof allows at
that intensity.  Panel (b) is the length of those stems expressed as the
division that produced it - ``268.0 / 312.0 = 0.859`` - so nothing on the page
is a percentage the reader has to take on faith.

Both roofs are drawn with explicit end points; there is no ``axhline`` in this
file, so the flat branch cannot strike through the y-axis labels.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, style_axes
from _wm_metrics import closed_ratio

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
GRID, CARD = "#ededed", "#e2e8ee"
PEAK = 312.0                       # TFLOP/s, dense FP16 tensor cores
BW_HBM = 2.04                      # TB/s  -> TFLOP/s per FLOP/byte
BW_L2 = 8.20                       # TB/s, the second roof
I_LO, I_HI = 0.28, 520.0
P_LO, P_HI = 0.45, 620.0

KIND = {"gemm": ("#08263f", "o", "GEMM (attention, FFN)"),
        "conv": ("#2e74a8", "s", "convolution (encoder stem)"),
        "band": ("#9cc3dd", "^", "bandwidth-bound (norm, softmax, "
                                 "elementwise)")}
# name, kind, arithmetic intensity (FLOP/byte), achieved TFLOP/s, label offset
OPS = [
    ("FFN GEMM", "gemm", 182.0, 268.0, (1.34, 0.72, "left")),
    # these two points sit 10 % apart in I, so each label goes on the side its
    # own marker is on: swap them and the reader pairs them the wrong way round
    ("QK$^\\top$ GEMM", "gemm", 64.0, 118.0, (1.10, 0.55, "left")),
    ("AV GEMM", "gemm", 58.0, 104.0, (0.50, 0.88, "right")),
    ("Conv stem", "conv", 41.0, 71.0, (0.45, 0.62, "right")),
    ("LayerNorm", "band", 2.10, 3.40, (1.36, 1.30, "left")),
    ("Softmax", "band", 1.60, 2.50, (0.58, 0.58, "right")),
    ("Rotary embed", "band", 0.90, 1.30, (1.42, 1.34, "left")),
    ("Residual add", "band", 0.42, 0.72, (1.44, 1.34, "left")),
]


def attainable(I):
    return np.minimum(PEAK, BW_HBM * np.asarray(I, dtype=float))


def main():
    set_pub_style(base_font=8.5)

    ops = [dict(name=n, kind=k, I=i, got=g, off=o) for (n, k, i, g, o) in OPS]
    for r in ops:
        r["roof"] = float(attainable(r["I"]))
        # the fraction is the division of the two numbers that get printed
        r["num"], r["den"] = f"{r['got']:.1f}", f"{r['roof']:.1f}"
        r["frac"] = float(r["num"]) / float(r["den"])
        # headroom is the subtraction of the two printed numbers as well
        r["gap"] = float(r["den"]) - float(r["num"])
        assert r["got"] <= r["roof"], f"{r['name']} sits above the roof"
    ridge = PEAK / BW_HBM
    rn, rd, rq, _ = closed_ratio(PEAK, BW_HBM)
    worst = min(ops, key=lambda r: r["frac"])
    best = max(ops, key=lambda r: r["frac"])
    fat = max(ops, key=lambda r: r["gap"])
    lever = fat["gap"] / worst["gap"]
    assert abs(lever - round(lever)) < 1e-9, \
        "the headroom ratio no longer closes on the printed gaps"
    # the headline names a kernel *class*, not a size: Residual add has both a
    # smaller intensity and a lower roof than the worst ratio, so "smallest"
    # would be false off the table in (b)
    assert worst["kind"] == "band" and best["kind"] != "band", \
        "the headline no longer describes where the worst ratio sits"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58], left=0.068,
                          right=0.988, top=0.795, bottom=0.150, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) the roofline ---------------------------------------------------
    style_axes(axa)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_axisbelow(True)
    axa.grid(which="major", color=GRID, lw=0.7, zorder=0)
    axa.grid(which="minor", color="#f7f7f7", lw=0.5, zorder=0)

    xs = np.logspace(np.log10(I_LO), np.log10(I_HI), 400)
    roof = attainable(xs)
    axa.fill_between(xs, roof, P_HI, facecolor="#f7f7f7", edgecolor="#dcdcdc",
                     hatch="//", lw=0.0, zorder=1)
    axa.text(30.0, 450.0, "unattainable", ha="center", va="center",
             fontsize=8.8, color="#8a8a8a", style="italic", zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.6,
                       alpha=0.92))
    # second roof: the same peak served from L2 instead of HBM
    xl = np.logspace(np.log10(I_LO), np.log10(PEAK / BW_L2), 120)
    axa.plot(np.append(xl, I_HI), np.append(BW_L2 * xl, PEAK), color="#5f9dc6",
             lw=1.2, ls=(0, (4.0, 2.0)), zorder=3)
    axa.plot(xs, roof, color="#0b3d63", lw=2.0, zorder=4,
             solid_capstyle="round")

    axa.plot([ridge], [PEAK], marker="o", ms=5.6, mfc="white", mec="#0b3d63",
             mew=1.5, zorder=7)
    axa.text(ridge * 1.30, 22.0, f"ridge  $I^*$ = {rn} / {rd}\n"
             f"= {rq} FLOP/byte", ha="left", va="top", fontsize=7.6,
             color="#0b3d63", fontweight="bold", zorder=8, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=HAIR,
                       lw=0.6, alpha=0.96))
    axa.plot([ridge, ridge], [PEAK, 26.0], color="#4d4d4d", lw=0.8,
             ls=(0, (2.4, 2.0)), zorder=5)
    # roof labels sit beside their line, on a white plate, never on it
    axa.text(11.0, 7.6, f"HBM roof   {BW_HBM:.2f} TB/s", ha="left", va="top",
             fontsize=7.6, color="#0b3d63", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.6,
                       alpha=0.96))
    axa.text(1.35, 24.0, f"L2 roof   {BW_L2:.2f} TB/s", ha="left",
             va="bottom", fontsize=7.4, color="#3d6b8e", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.6,
                       alpha=0.96))
    axa.text(I_HI * 0.92, PEAK * 1.13, f"peak {PEAK:.0f} TFLOP/s",
             ha="right", va="bottom", fontsize=7.8, color=INK,
             fontweight="bold", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.94))

    for r in ops:
        col, mk, _ = KIND[r["kind"]]
        axa.plot([r["I"], r["I"]], [r["got"], r["roof"]], color="#8a8a8a",
                 lw=0.8, ls=(0, (1.6, 1.6)), zorder=5)
        axa.plot([r["I"]], [r["got"]], marker=mk, ms=7.4, mfc=col,
                 mec="white", mew=1.0, zorder=8)
        fx, fy, ha = r["off"]
        axa.text(r["I"] * fx, r["got"] * fy, r["name"], ha=ha, va="center",
                 fontsize=7.4, color=INK, zorder=9,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.90))

    axa.set_xlim(I_LO, I_HI)
    axa.set_ylim(P_LO, P_HI)
    axa.set_xticks([0.5, 1, 5, 10, 50, 100, 500])
    axa.set_xticklabels(["0.5", "1", "5", "10", "50", "100", "500"])
    axa.set_yticks([1, 10, 100])
    axa.set_yticklabels(["1", "10", "100"])
    for axis in (axa.xaxis, axa.yaxis):
        axis.set_minor_formatter(mticker.NullFormatter())
    axa.tick_params(which="minor", length=1.6)
    axa.set_xlabel("arithmetic intensity  $I$  (FLOP / byte)", fontsize=8.6)
    axa.set_ylabel("attained performance  (TFLOP / s)", fontsize=8.6)
    axa.tick_params(labelsize=7.4, length=2.4)
    axa.set_title("(a)  Eight kernels under one roof", fontsize=9.4,
                  fontweight="bold", pad=5.0, loc="left")
    axa.text(0.982, 0.045, f"attainable($I$) = min({PEAK:.0f}, "
             f"{BW_HBM:.2f} $\\cdot$ $I$)  TFLOP/s", transform=axa.transAxes,
             ha="right", va="bottom", fontsize=7.8, color=INK, zorder=9,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=HAIR,
                       lw=0.6, alpha=0.96))

    # ---- (b) how much of that roof each kernel reached ----------------------
    style_axes(axb)
    axb.set_axisbelow(True)
    axb.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    order = sorted(ops, key=lambda r: r["frac"])
    ypos = np.arange(len(order), dtype=float)
    for y, r in zip(ypos, order):
        col, _, _ = KIND[r["kind"]]
        axb.barh(y, r["frac"], height=0.58, color=col, edgecolor="white",
                 linewidth=0.8, zorder=3)
        axb.text(r["frac"] - 0.018, y, f"{r['num']} / {r['den']} = "
                 f"{r['frac']:.3f}", ha="right", va="center", fontsize=7.4,
                 color="white" if r["kind"] != "band" else INK,
                 fontweight="bold", zorder=5)
        axb.text(1.055, y, f"{r['gap']:.1f}", ha="left", va="center",
                 fontsize=7.4, zorder=5,
                 color=INK if r is fat else SUB,
                 fontweight="bold" if r is fat else "normal")
    axb.plot([1.0, 1.0], [-0.62, len(order) - 0.38], color="#3a3a3a", lw=1.2,
             zorder=6)
    axb.text(1.055, len(order) - 0.52, "headroom\n(TFLOP/s)", ha="left",
             va="bottom", fontsize=7.2, color=SUB, linespacing=1.4, zorder=7)
    axb.set_xlim(0.0, 1.32)
    axb.set_ylim(-0.62, len(order) - 0.38)
    axb.set_yticks(ypos)
    axb.set_yticklabels([f"{r['name']}   $I$ = {r['I']:.2f}"
                         if r["I"] < 10 else f"{r['name']}   $I$ = {r['I']:.0f}"
                         for r in order], fontsize=7.6)
    axb.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xlabel("attained / attainable at that intensity", fontsize=8.6)
    axb.tick_params(labelsize=7.4, length=2.4)
    axb.set_title("(b)  Attainment ratio and headroom", fontsize=9.4,
                  fontweight="bold", pad=13.0, loc="left")

    handles = [Line2D([], [], ls="none", marker=KIND[k][1], ms=7.0,
                      mfc=KIND[k][0], mec="white", mew=1.0, label=KIND[k][2])
               for k in ("gemm", "conv", "band")]
    handles += [Line2D([], [], color="#0b3d63", lw=2.0,
                       label="HBM roof, min(peak, BW $\\cdot$ $I$)"),
                Line2D([], [], color="#5f9dc6", lw=1.2, ls=(0, (4.0, 2.0)),
                       label="L2 roof"),
                Line2D([], [], color="#8a8a8a", lw=0.8, ls=(0, (1.6, 1.6)),
                       label="gap to the roof")]
    # parked in the empty hatched corner, which no kernel can ever occupy
    leg = axa.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.012, 0.988), ncol=1, frameon=True,
                     facecolor="white", edgecolor=HAIR, framealpha=0.96,
                     fontsize=7.2, handlelength=2.0, borderpad=0.45,
                     labelspacing=0.36)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(10)

    # ---- headline and the arithmetic ---------------------------------------

    band = FancyBboxPatch((0.022, 0.012), 0.956, 0.058,
                          boxstyle="round,pad=0.004,rounding_size=0.008",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#c3cdd6", linewidth=0.6, zorder=-5)
    fig.add_artist(band)
    band.set_in_layout(False)
    savefig(fig, "1193_wm_sys_roofline.png")


if __name__ == "__main__":
    main()
