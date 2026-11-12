"""Shewhart X-bar / R control chart pair with sigma zones and rule violations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
GREY = "#9aa3ab"
ALERT = "#b23a48"
ZONE_C, ZONE_B, ZONE_A = "#f0f4f8", "#e0e8f0", "#ccd9e6"
ALARM_WASH = "#fcf3f4"

# Shewhart constants for subgroup size n = 5
A2, D3, D4, D2 = 0.577, 0.0, 2.114, 2.326
NSUB, NGRP = 28, 5


def make_data():
    """Raw subgroups plus Phase-I limits estimated from the first 15 samples."""
    rng = np.random.default_rng(20250812)
    x = rng.normal(25.000, 0.050, size=(NSUB, NGRP))
    cl = x[:15].mean()
    rbar = np.ptp(x[:15], axis=1).mean()
    sigma = A2 * rbar / 3.0

    def put(i, z):
        """Re-centre subgroup ``i`` on ``cl + z*sigma``, keeping its spread."""
        x[i] += cl + z * sigma - x[i].mean()

    put(11, 4.6)                                   # spike beyond UCL
    for k, i in enumerate(range(16, 25)):          # sustained run above CL
        put(i, 1.05 + 0.35 * np.sin(1.1 * k))
    put(21, 2.45)                                  # 2-of-3 beyond 2 sigma
    put(23, 2.30)
    put(25, -4.4)                                  # dip below LCL
    m, dev = x[19].mean(), x[19] - x[19].mean()           # R-chart signal
    x[19] = m + dev * (1.22 * D4 * rbar) / np.ptp(dev)
    return x, cl, rbar, sigma


def flag_rules(v, cl, sigma):
    """Return {index: rule label} for Nelson rules 1, 2 and 5."""
    z = (v - cl) / sigma
    flags = {}
    for i, zi in enumerate(z):
        if abs(zi) > 3:
            flags[i] = "1"
    for i in range(2, len(z)):
        win = z[i - 2:i + 1]
        for s in (1, -1):
            if np.sum(s * win > 2) >= 2 and s * z[i] > 2 and i not in flags:
                flags[i] = "5"
    run, sign = 0, 0
    for i, zi in enumerate(z):
        s = 1 if zi > 0 else -1
        run = run + 1 if s == sign else 1
        sign = s
        if run == 9:                       # flag the point that closes the run
            flags.setdefault(i, "2")
            run = 0
    return flags


def draw_zones(ax, cl, sigma, xlim, ylim):
    """Three symmetric sigma bands plus a faint alarm wash beyond 3 sigma."""
    y0, y1 = ylim
    ax.axhspan(cl + 3 * sigma, y1, color=ALARM_WASH, zorder=0)
    if cl - 3 * sigma > y0:
        ax.axhspan(y0, cl - 3 * sigma, color=ALARM_WASH, zorder=0)
    for k, col in zip((1, 2, 3), (ZONE_C, ZONE_B, ZONE_A)):
        ax.axhspan(cl + (k - 1) * sigma, cl + k * sigma, color=col, zorder=0)
        ax.axhspan(max(cl - k * sigma, y0), max(cl - (k - 1) * sigma, y0),
                   color=col, zorder=0)
    for k, lab in zip((0.5, 1.5, 2.5), ("C", "B", "A")):
        for s in (1, -1):
            y = cl + s * k * sigma
            if y0 + 0.02 * (y1 - y0) < y < y1:
                ax.text(xlim[1] + 0.35, y, lab, ha="left", va="center",
                        fontsize=6.0, color=GREY, clip_on=False)


def chart(ax, idx, v, cl, ucl, lcl, sigma, ylab, flags, fmt="{:.3f}",
          floor=None):
    xlim = (0.3, NSUB + 0.7)
    lo, hi = min(v.min(), lcl), max(v.max(), ucl)
    pad = 0.20 * (hi - lo)
    ylim = (lo - pad if floor is None else max(lo - pad, floor), hi + pad)
    ax.set_ylim(*ylim)
    draw_zones(ax, cl, sigma, xlim, ylim)
    ax.plot(idx, v, "-", color=BLUE, lw=1.1, zorder=3)
    ax.plot(idx, v, "o", ms=3.1, mfc="white", mec=NAVY, mew=0.9, zorder=4)
    ax.axhline(cl, color=NAVY, lw=1.1, zorder=2)
    for lim in (ucl, lcl):
        ax.axhline(lim, color=ALERT, lw=1.0, ls=(0, (5, 2.5)), zorder=2)

    for i, rule in flags.items():
        ax.plot(idx[i], v[i], "o", ms=7.5, mfc="none", mec=ALERT, mew=1.3,
                zorder=5)
        up = v[i] >= cl
        ax.annotate(rule, (idx[i], v[i]),
                    xytext=(0, 9.5 if up else -9.5), textcoords="offset points",
                    ha="center", va="bottom" if up else "top", fontsize=6.4,
                    fontweight="bold", color=ALERT, zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                              ec=ALERT, lw=0.7))

    ax.set_xlim(*xlim)
    ax.set_ylabel(ylab, fontsize=8)
    style_axes(ax)
    ax.tick_params(labelsize=7)
    for lim, lab, col in ((ucl, "UCL", ALERT), (cl, "CL", NAVY),
                          (lcl, "LCL", ALERT)):
        ax.text(xlim[1] + 1.55, lim, f"{lab} {fmt.format(lim)}", ha="left",
                va="center", fontsize=6.5, color=col, fontweight="bold",
                clip_on=False)


def main():
    set_pub_style()
    x, cl_x, rbar, sig_x = make_data()
    idx = np.arange(1, NSUB + 1)
    xbar, rng_ = x.mean(axis=1), np.ptp(x, axis=1)

    ucl_x, lcl_x = cl_x + 3 * sig_x, cl_x - 3 * sig_x
    cl_r = rbar
    ucl_r, lcl_r = D4 * rbar, D3 * rbar
    sig_r = (ucl_r - cl_r) / 3.0

    flags_x = flag_rules(xbar, cl_x, sig_x)
    flags_r = {i: "1" for i, r in enumerate(rng_) if r > ucl_r}

    fig = plt.figure(figsize=(7.4, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.28, 1.0], hspace=0.14,
                          left=0.085, right=0.845, top=0.845, bottom=0.115)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    chart(ax1, idx, xbar, cl_x, ucl_x, lcl_x, sig_x,
          "Subgroup mean  $\\bar{X}$  (mm)", flags_x)
    chart(ax2, idx, rng_, cl_r, ucl_r, lcl_r, sig_r,
          "Subgroup range  $R$  (mm)", flags_r, floor=-0.05 * ucl_r)
    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel("Subgroup index (hourly sample, n = 5)", fontsize=8)
    ax2.set_xticks(np.arange(2, NSUB + 1, 2))

    for ax, lab in ((ax1, "(a)"), (ax2, "(b)")):
        add_panel_label(ax, lab, x=-0.075, y=1.0, size=9)


    handles = [
        Line2D([], [], color=BLUE, marker="o", ms=3.1, mfc="white", mec=NAVY,
               lw=1.1, label="Subgroup statistic"),
        Line2D([], [], color=NAVY, lw=1.1, label="Centre line"),
        Line2D([], [], color=ALERT, lw=1.0, ls=(0, (5, 2.5)),
               label="3$\\sigma$ control limit"),
        Line2D([], [], color="none", marker="o", ms=6.5, mfc="none",
               mec=ALERT, mew=1.3, label="Rule violation"),
        Patch(fc=ZONE_A, ec="none", label="Zone A / B / C ($\\sigma$ bands)"),
    ]


    savefig(fig, "243_control_chart.png")


if __name__ == "__main__":
    main()
