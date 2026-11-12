"""Allometric scaling of an appendage against body length in four clades.

The scatter is drawn on log10 axes with ``set_aspect("equal")`` for one
reason: isometry is the line of slope 1, and on a stretched log-log panel that
line renders at some arbitrary angle, so the reader loses the only visual
reference the figure exists to provide.  Here isometry really is 45 degrees.

Slopes are reduced major axis (RMA) fits, appropriate when both variables
carry measurement error, and their confidence intervals come from a
non-parametric bootstrap over the same points that are plotted.  Line style
encodes the test the intervals imply - solid when the 95 % CI excludes 1,
dashed when it does not - so the main panel, the forest plot and the printed
intervals can never disagree with each other.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"

FIG_W, FIG_H = 10.4, 6.4
SEED = 21
N_BOOT = 3000

# name, n, true slope, intercept, residual sd, log10 body-length range
CLADES = [
    ("Ammonoidea", 46, 1.30, -0.69, 0.052, (1.05, 2.30)),
    ("Trilobita", 40, 0.97, -0.25, 0.100, (1.00, 2.30)),
    ("Brachiopoda", 44, 1.00, -0.45, 0.048, (0.95, 2.45)),
    ("Crinoidea", 38, 0.70, -0.06, 0.055, (1.10, 2.70)),
]
COLS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"]


def rma_slope(x, y):
    """Reduced major axis slope: sign(r) * sd(y) / sd(x)."""
    r = np.corrcoef(x, y)[0, 1]
    return np.sign(r) * y.std(ddof=1) / x.std(ddof=1)


def rma_fit(x, y, rng):
    b = rma_slope(x, y)
    a = y.mean() - b * x.mean()
    boot = np.empty(N_BOOT)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    for i in range(N_BOOT):
        boot[i] = rma_slope(x[idx[i]], y[idx[i]])
    return a, b, np.percentile(boot, [2.5, 97.5])


def fit_equal(ax, xlim, ylim, box_w, box_h):
    (x0, x1), (y0, y1) = xlim, ylim
    want = box_w / box_h
    if (x1 - x0) / (y1 - y0) < want:
        s = (y1 - y0) * want
        c = 0.5 * (x0 + x1)
        x0, x1 = c - s / 2.0, c + s / 2.0
    else:
        s = (x1 - x0) / want
        c = 0.5 * (y0 + y1)
        y0, y1 = c - s / 2.0, c + s / 2.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    data, fits = [], []
    for name, n, slope, icpt, sd, (lo, hi) in CLADES:
        x = np.sort(rng.uniform(lo, hi, n))
        y = icpt + slope * x + rng.normal(scale=sd, size=n)
        a, b, ci = rma_fit(x, y, rng)
        data.append((x, y))
        fits.append((name, n, a, b, ci, ci[0] > 1.0 or ci[1] < 1.0))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    mw, mh = 5.55, 4.55
    ax = axes_inches(fig, (0.68, 0.92, mw, mh))
    ax_f = axes_inches(fig, (7.20, 3.62, 2.90, 1.85))
    ax_r = axes_inches(fig, (7.20, 0.92, 2.90, 1.85))

    # ---- (a) log-log scatter with RMA lines ------------------------------
    ax.set_aspect("equal")
    all_x = np.concatenate([d[0] for d in data])
    all_y = np.concatenate([d[1] for d in data])
    xr = (all_x.min() - 0.10, all_x.max() + 0.10)
    yr = (all_y.min() - 0.12, all_y.max() + 0.32)
    fit_equal(ax, xr, yr, mw, mh)
    # Equal aspect widens x; spend the slack on the right, where the
    # end-of-line slope labels live, instead of on empty space at the left.
    span = float(np.diff(ax.get_xlim())[0])
    ax.set_xlim(xr[0], xr[0] + span)
    x0, x1 = ax.get_xlim()

    # isometry reference, anchored at the grand mean of the pooled cloud
    y0, y1 = ax.get_ylim()
    iso_c = all_y.mean() - all_x.mean()
    ax.plot([x0, x1], [x0 + iso_c, x1 + iso_c], color=GREY, lw=1.0,
            ls=(0, (4.0, 2.2)), zorder=2)

    ends = []
    for (x, y), (name, n, a, b, ci, sig), col in zip(data, fits, COLS):
        ax.scatter(x, y, s=13, color=col, alpha=0.80, edgecolor="white",
                   linewidths=0.35, zorder=4)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, a + b * xs, color=col, lw=1.6 if sig else 1.4,
                ls="-" if sig else (0, (3.4, 2.0)), zorder=5)
        ends.append((a + b * xs[1], xs[1], col,
                     f"{name}  b = {b:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
                     f"{'' if sig else '  n.s.'}"))

    # End-of-line labels: push them apart top-down so two lines that finish at
    # a similar height cannot print their intervals on top of each other.
    ends.sort(key=lambda e: -e[0])
    gap = 0.052 * (y1 - y0)
    y_lab = [ends[0][0]]
    for e in ends[1:]:
        y_lab.append(min(e[0], y_lab[-1] - gap))
    for (y_end, x_end, col, lab), yl in zip(ends, y_lab):
        if abs(yl - y_end) > 1e-3:
            ax.plot([x_end, x_end + 0.028], [y_end, yl], color=col, lw=0.6,
                    alpha=0.8, zorder=5)
        ax.text(x_end + 0.038, yl, lab, fontsize=6.6, color=col, ha="left",
                va="center", zorder=6)

    ax.set_xlabel(r"log$_{10}$ body length (log$_{10}$ mm)", fontsize=8.0)
    ax.set_ylabel(r"log$_{10}$ appendage length (log$_{10}$ mm)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Reduced major axis scaling in four clades", fontsize=9.0,
                 fontweight="bold", color=INK)
    handles = [Line2D([], [], color=c, lw=1.5, marker="o", ms=3.4,
                      label=f"{f[0]} (n = {f[1]})")
               for c, f in zip(COLS, fits)]
    handles += [Line2D([], [], color=GREY, lw=1.0, ls=(0, (4.0, 2.2)),
                       label="isometry (slope = 1)"),
                Line2D([], [], color=GREY, lw=1.4, ls=(0, (3.4, 2.0)),
                       label="slope not distinct from 1")]
    ax.legend(handles=handles, fontsize=6.8, frameon=False, loc="upper left",
              bbox_to_anchor=(0.005, 0.995), handlelength=1.7,
              labelspacing=0.30)

    # ---- (b) forest plot of slopes ---------------------------------------
    ypos = np.arange(len(fits))[::-1]
    ax_f.axvline(1.0, color=GREY, lw=1.0, ls=(0, (4.0, 2.2)), zorder=2)
    for yp, (name, n, a, b, ci, sig), col in zip(ypos, fits, COLS):
        ax_f.plot(ci, [yp, yp], color=col, lw=1.4, solid_capstyle="butt",
                  zorder=3)
        for e in ci:
            ax_f.plot([e, e], [yp - 0.16, yp + 0.16], color=col, lw=1.0,
                      zorder=3)
        ax_f.plot([b], [yp], marker="o", ms=5.0, color=col if sig else "white",
                  mec=col, mew=1.2, zorder=4)
        ax_f.text(ci[1] + 0.035, yp, "*" if sig else "n.s.", fontsize=6.6,
                  color=col if sig else MUTED, ha="left", va="center")
    ax_f.set_yticks(ypos)
    ax_f.set_yticklabels([f[0] for f in fits], fontsize=6.8)
    ax_f.set_ylim(-0.65, len(fits) - 0.35)
    ax_f.set_xlim(0.55, 1.62)
    ax_f.set_xlabel("RMA slope (95 % bootstrap CI)", fontsize=7.4)
    ax_f.tick_params(labelsize=6.6)
    ax_f.text(1.0, -0.60, "isometry", fontsize=6.4, color=SUB, ha="center",
              va="bottom")
    ax_f.set_title("Two clades depart from isometry", fontsize=8.5,
                   fontweight="bold", color=INK)

    # ---- (c) residuals vs body size --------------------------------------
    res_all = []
    for (x, y), (name, n, a, b, ci, sig), col in zip(data, fits, COLS):
        res = y - (a + b * x)
        res_all.append(res)
        ax_r.scatter(x, res, s=11, color=col, alpha=0.78, edgecolor="white",
                     linewidths=0.3, zorder=3)
    ax_r.axhline(0.0, color=GREY, lw=0.9, zorder=2)
    m = 1.15 * np.abs(np.concatenate(res_all)).max()
    ax_r.set_ylim(-m, m)
    ax_r.set_xlim(*xr)
    ax_r.set_xlabel(r"log$_{10}$ body length (log$_{10}$ mm)", fontsize=7.4)
    ax_r.set_ylabel(r"RMA residual (log$_{10}$ mm)", fontsize=7.4)
    ax_r.tick_params(labelsize=6.6)
    sd_txt = "  ".join(f"{f[0][:4]} {r.std(ddof=1):.3f}"
                       for f, r in zip(fits, res_all))
    ax_r.text(0.5, 0.02, "residual SD:  " + sd_txt, transform=ax_r.transAxes,
              ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax_r.set_title("No size-dependent structure left in the residuals",
                   fontsize=8.5, fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.50, dy=0.10)
    panel_labels_inches(fig, [ax_f], labels=["(b)"], dx=-0.60, dy=0.10)
    panel_labels_inches(fig, [ax_r], labels=["(c)"], dx=-0.60, dy=0.10)
    savefig(fig, "824_mph_allometry_scaling.png")


if __name__ == "__main__":
    main()
