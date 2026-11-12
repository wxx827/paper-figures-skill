"""Roofline model: which kernels are memory-bound and how close they get.

The upper bound on achievable performance is the lower of two ceilings - the
DRAM bandwidth times the arithmetic intensity, and the vector compute peak -
so the roof is a bent line whose knee (the ridge point) splits the plane into
a memory-bound and a compute-bound half. Measured kernels are placed on that
plane and each one is labelled with the fraction of its own attainable roof
that it actually reaches, which is the number a tuning effort is trying to
move. Every percentage, the ridge point and the footer summary are computed
from the same machine constants that draw the roof.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from pubstyle import set_pub_style, savefig

# ---- machine model (one node) ---------------------------------------------
PEAK = 3200.0        # GFLOP/s, FMA vector peak, double precision
PEAK_NOFMA = 1600.0  # GFLOP/s, same width without fused multiply-add
BW_DRAM = 205.0      # GB/s, streaming triad bandwidth
BW_L2 = 1150.0       # GB/s, aggregate L2 bandwidth

MEM_BG = "#f5f5f5"
COMP_BG = "#f0f0f0"
ROOF = "#333333"
CEIL = "#808080"
KERNEL = "#3b5b92"   # house blue
ACCENT = "#b23a48"   # house red, the single emphasis object
SUB = "#555555"
MUTED = "#777777"

# (name, arithmetic intensity FLOP/Byte, measured GFLOP/s, dx, dy, ha, va)
KERNELS = [
    ("SpMV (CSR)",     0.14,   24.0, -0.14,  1.60, "right", "bottom"),
    ("7-pt stencil",   0.42,   78.0,  0.22,  0.62, "left",  "top"),
    ("SpMM",           0.95,  152.0,  0.18,  0.62, "left",  "top"),
    ("3D FFT",         1.90,  238.0, -0.16,  1.60, "right", "bottom"),
    ("Conv 3x3",       7.50,  905.0,  0.16,  0.70, "left",  "top"),
    ("DGEMM",         38.00, 2640.0, -0.32,  0.42, "right", "top"),
    ("Direct N-body", 160.0, 2955.0, -0.10,  0.38, "right", "top"),
]

XLO, XHI = 0.045, 520.0
YLO, YHI = 9.0, 7000.0


def roof_at(oi, bw=BW_DRAM, peak=PEAK):
    return np.minimum(peak, bw * np.asarray(oi, dtype=float))


def line_angle(ax, x0, y0, x1, y1):
    """On-paper angle of a data-space segment, for text laid along a roof."""
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, y0), (x1, y1)])
    return float(np.degrees(np.arctan2(py1 - py0, px1 - px0)))


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    ridge = PEAK / BW_DRAM
    ridge_nofma = PEAK_NOFMA / BW_DRAM
    ridge_l2 = PEAK / BW_L2

    # ---- bounded regions --------------------------------------------------
    ax.axvspan(XLO, ridge, facecolor=MEM_BG, edgecolor="none", zorder=0)
    ax.axvspan(ridge, XHI, facecolor=COMP_BG, edgecolor="none", zorder=0)
    ax.axvline(ridge, color=CEIL, lw=0.8, ls=":", zorder=1)

    # ---- ceilings ---------------------------------------------------------
    xg = np.logspace(np.log10(XLO), np.log10(XHI), 400)
    ax.plot(xg, roof_at(xg), lw=1.8, color=ROOF, zorder=4,
            solid_joinstyle="miter")
    ax.plot(xg, roof_at(xg, BW_DRAM, PEAK_NOFMA), lw=1.0, ls="--", color=CEIL,
            zorder=3)
    ax.plot(xg, np.minimum(PEAK, BW_L2 * xg), lw=1.0, ls="-.", color=CEIL,
            alpha=0.85, zorder=3)

    # ---- kernels ----------------------------------------------------------
    oi = np.array([k[1] for k in KERNELS])
    perf = np.array([k[2] for k in KERNELS])
    frac = perf / roof_at(oi)
    worst = int(np.argmin(frac))

    for i, (name, x, y, dx, dy, ha, va) in enumerate(KERNELS):
        hot = i == worst
        col = ACCENT if hot else KERNEL
        ax.plot([x], [y], marker="o", ms=6.2 if hot else 5.2, color=col,
                mec="white", mew=0.9, ls="none", zorder=6)
        ax.plot([x, x], [y, roof_at(x)], lw=0.7, ls=":", color=col,
                alpha=0.85, zorder=5)
        ax.text(x * (1.0 + dx), y * dy, f"{name}\n{100.0 * frac[i]:.0f} % of roof",
                ha=ha, va=va, fontsize=6.6, color=ACCENT if hot else SUB,
                linespacing=1.30, zorder=7)

    # ---- axes -------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=40))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=7.2)
    ax.tick_params(which="minor", length=1.8)
    ax.set_xlabel("arithmetic intensity (FLOP/Byte)", fontsize=8.0)
    ax.set_ylabel("attained performance (GFLOP/s)", fontsize=8.0)
    ax.grid(True, which="major", lw=0.5, alpha=0.45, zorder=1)

    ax.text(np.sqrt(XLO * ridge), YHI * 0.90, "memory-bound", ha="center",
            va="top", fontsize=7.4, color=MUTED, style="italic")
    ax.text(np.sqrt(ridge * XHI), YHI * 0.90, "compute-bound", ha="center",
            va="top", fontsize=7.4, color=MUTED, style="italic")
    ax.text(ridge * 0.85, YLO * 1.10,
            f"ridge point\n{ridge:.1f} FLOP/Byte", ha="right", va="bottom",
            fontsize=6.6, color=SUB, linespacing=1.30)

    handles = [
        Line2D([], [], color=ROOF, lw=1.8,
               label=f"Roofline: min({BW_DRAM:.0f} GB/s x I, "
                     f"{PEAK:.0f} GFLOP/s)"),
        Line2D([], [], color=CEIL, lw=1.0, ls="--",
               label=f"No-FMA ceiling ({PEAK_NOFMA:.0f} GFLOP/s)"),
        Line2D([], [], color=CEIL, lw=1.0, ls="-.",
               label=f"L2 bandwidth roof ({BW_L2:.0f} GB/s)"),
        Line2D([], [], color=KERNEL, marker="o", ms=5.2, ls="none",
               mec="white", mew=0.9, label="Measured kernel"),
        Line2D([], [], color=ACCENT, marker="o", ms=6.2, ls="none",
               mec="white", mew=0.9,
               label=f"Largest shortfall ({KERNELS[worst][0]})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.0, frameon=False,
              handlelength=1.9, labelspacing=0.32, borderaxespad=0.7)

    n_mem = int((oi < ridge).sum())
    ax.set_title(f"{n_mem} of {len(KERNELS)} kernels are bandwidth-limited, "
                 f"not compute-limited",
                 fontsize=11.0, fontweight="bold", pad=8)

    fig.tight_layout(rect=[0, 0.075, 1, 1])

    # Roof labels are laid along the lines, so their on-paper angle can only
    # be measured once the axes box is final.
    a_bw = line_angle(ax, 0.30, BW_DRAM * 0.30, 3.0, BW_DRAM * 3.0)
    ax.text(0.60, BW_DRAM * 0.60 * 1.14, f"DRAM {BW_DRAM:.0f} GB/s",
            rotation=a_bw, rotation_mode="anchor", ha="left", va="bottom",
            fontsize=6.8, color=ROOF, zorder=7)
    ax.text(XHI * 0.94, PEAK * 1.07,
            f"vector peak {PEAK:.0f} GFLOP/s", ha="right", va="bottom",
            fontsize=6.8, color=ROOF, zorder=7)

    savefig(fig, "565_hpc_roofline.png")


if __name__ == "__main__":
    main()
