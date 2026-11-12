"""Two-dimensional gamma analysis of a patient-specific QA measurement.

A planned dose plane and a deliberately imperfect measurement of it are built
once; the gamma index is then evaluated by an explicit minimum search over a
shifted neighbourhood, exactly as the definition prescribes, rather than
approximated.  Every number on the figure - the pass rate, the mean gamma, the
size of the failing cluster and the four criterion-dependent pass rates - is a
count or an average taken on that same gamma array.

The gamma map is deliberately two-toned: the colour map is a cool ramp for
``gamma <= 1`` welded to a warm ramp above it, so the acceptance boundary is a
hue change rather than a value the reader has to look up on the bar.

Synthetic illustration for a visual reference library - not patient data.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, label, map_coordinates
from scipy.special import erf

from pubstyle import set_pub_style, savefig, despine, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

FIGW, FIGH = 10.4, 6.4
NPIX = 120
SPACING = 1.0                  # detector pitch (mm)
SUBPIX = 4                     # sub-detector steps used by the gamma search
THRESH = 0.10                  # low-dose cut-off, fraction of the maximum
ACTION_LIMIT = 95.0            # departmental pass-rate action limit (%)
CRITERIA = [(3.0, 3.0), (3.0, 2.0), (2.0, 2.0), (2.0, 1.0)]


def gamma_colormap():
    """Cool below the acceptance limit, warm above it, welded at gamma = 1."""
    cool = get_cmap("cb_sequential_blue")
    warm = get_cmap("cb_sequential_warm")
    lo = [cool(t) for t in np.linspace(0.04, 0.92, 128)]
    hi = [warm(t) for t in np.linspace(0.34, 1.00, 128)]
    return ListedColormap(np.vstack([lo, hi]), name="gamma_pass_fail")


def rect(X, Y, x0, x1, y0, y1, soft=2.2):
    """Soft-edged rectangular fluence segment."""
    fx = 0.5 * (erf((X - x0) / soft) - erf((X - x1) / soft))
    fy = 0.5 * (erf((Y - y0) / soft) - erf((Y - y1) / soft))
    return fx * fy


def dose_planes(X, Y, rng):
    """Planned plane and a measurement carrying realistic delivery errors."""
    calc = (1.00 * rect(X, Y, -38.0, 38.0, -34.0, 34.0) +
            0.42 * rect(X, Y, -20.0, 14.0, -18.0, 22.0) +
            0.30 * rect(X, Y, 6.0, 34.0, -30.0, 4.0))
    calc = gaussian_filter(calc, 1.6)
    calc = 200.0 * calc / calc.max()                     # cGy

    # 1.2 mm systematic shift, 1.5 % output error, one hot spot, detector noise
    shifted = (1.000 * rect(X - 1.2, Y - 0.6, -38.0, 38.0, -34.0, 34.0) +
               0.42 * rect(X - 1.2, Y - 0.6, -20.0, 14.0, -18.0, 22.0) +
               0.30 * rect(X - 1.2, Y - 0.6, 6.0, 34.0, -30.0, 4.0))
    shifted = gaussian_filter(shifted, 1.6)
    shifted = 200.0 * shifted / shifted.max()
    hot = 15.0 * np.exp(-(((X + 24.0) ** 2 + (Y + 20.0) ** 2) / (2 * 6.5 ** 2)))
    noise = gaussian_filter(rng.normal(0.0, 1.6, X.shape), 0.7)
    return calc, 1.010 * shifted + hot + noise


def gamma_index(meas, calc, dd_pct, dta_mm, spacing=SPACING, sub=SUBPIX):
    """Global gamma by explicit minimisation over a shifted neighbourhood.

    The search steps in *sub-detector* increments: a whole-pixel search cannot
    resolve a sub-millimetre misalignment, so in a steep penumbra it leaves a
    large residual dose difference and manufactures a ring of failures around
    every field edge.  Real analysers interpolate the reference for the same
    reason.
    """
    ref = float(calc.max())
    dd = dd_pct / 100.0 * ref
    reach = 2.0 * dta_mm
    step = spacing / sub
    r = int(np.ceil(reach / step))
    h, w = meas.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    best = np.full(meas.shape, np.inf)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist = np.hypot(dx * step, dy * step)
            if dist > reach:
                continue
            ref_at = map_coordinates(calc, [yy + dy / sub, xx + dx / sub],
                                     order=1, mode="nearest")
            np.minimum(best, (dist / dta_mm) ** 2 +
                       ((meas - ref_at) / dd) ** 2, out=best)
    return np.sqrt(best)


def main():
    set_pub_style()
    rng = np.random.default_rng(873)

    half = 0.5 * (NPIX - 1) * SPACING
    ax_mm = np.linspace(-half, half, NPIX)
    X, Y = np.meshgrid(ax_mm, ax_mm)
    calc, meas = dose_planes(X, Y, rng)

    mask = calc >= THRESH * calc.max()
    gam = gamma_index(meas, calc, *CRITERIA[0])
    gam_shown = np.where(mask, gam, np.nan)
    passed = gam[mask] <= 1.0
    pass_rate = 100.0 * passed.mean()
    n_eval = int(mask.sum())
    n_fail = int((~passed).sum())
    gam_mean = float(gam[mask].mean())

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = axes_inches(fig, (0.88, 1.00, 4.44, 4.44))
    ax_cum = axes_inches(fig, (7.10, 3.70, 2.90, 1.74))
    ax_cri = axes_inches(fig, (7.10, 0.95, 2.90, 1.74))
    despine(ax_cum)
    despine(ax_cri)

    # ---- main panel: the gamma map ----------------------------------------
    ext = [-half, half, -half, half]
    im = ax.imshow(gam_shown, cmap=gamma_colormap(),
                   norm=mpl.colors.Normalize(0.0, 2.0), extent=ext,
                   origin="lower", interpolation="nearest")
    ax.set_aspect("equal")

    ref = float(calc.max())
    for level in (0.50, 0.90):
        ax.contour(X, Y, calc / ref, levels=[level], colors=["#1a1a1a"],
                   linewidths=0.7)
        ax.contour(X, Y, meas / ref, levels=[level], colors=["#1a1a1a"],
                   linewidths=0.7, linestyles=[(0, (2.2, 1.5))])

    # The worst failing cluster, found on the map rather than assumed.
    blobs, n_blob = label(mask & (gam > 1.0))
    if n_blob:
        sizes = np.bincount(blobs.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        by, bx = np.nonzero(blobs == big)
        cx, cy = float(ax_mm[bx].mean()), float(ax_mm[by].mean())
        ax.annotate(f"largest failing cluster:\n{sizes.max()} of {n_fail} "
                    f"failing points",
                    xy=(cx, cy), xytext=(-half + 2.0, half - 2.0),
                    fontsize=6.4, color=INK, ha="left", va="top",
                    linespacing=1.5,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#3a3a3a",
                                    mutation_scale=7, shrinkB=4.0))
    halo_text(ax, 0.0, -half + 2.5,
              f"pass rate {pass_rate:.1f} %   "
              f"({CRITERIA[0][0]:.0f} % / {CRITERIA[0][1]:.0f} mm global, "
              f"{THRESH * 100:.0f} % threshold)",
              ha="center", va="bottom", fontsize=6.8, color="white",
              halo="#1a1a1a", halo_lw=2.2)

    ax.legend(handles=[
        Line2D([], [], color="#1a1a1a", lw=0.9, label="calculated isodose"),
        Line2D([], [], color="#1a1a1a", lw=0.9, ls=(0, (2.2, 1.5)),
               label="measured isodose"),
        Line2D([], [], color="none",
               label=r"cool $\gamma \leq 1$ / warm $\gamma > 1$")],
        loc="lower right", ncol=3, frameon=False, fontsize=6.5,
        handlelength=1.5, handletextpad=0.5, columnspacing=1.2,
        borderpad=0.1, bbox_to_anchor=(1.005, 1.005))

    ax.set_xlabel("cross-plane position (mm)", fontsize=8.0)
    ax.set_ylabel("in-plane position (mm)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, color="#8a8a8a")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    cb = safe_colorbar(fig, im, ax, label=r"gamma index $\gamma$ (-)",
                       width=0.12, pad=0.09, labelsize=6.8,
                       ticks=[0.0, 0.5, 1.0, 1.5, 2.0], extend="max")
    cb.ax.axhline(1.0, color="#1a1a1a", lw=0.9)

    # ---- aux 1: cumulative gamma histogram --------------------------------
    edges = np.linspace(0.0, 2.0, 201)
    cum = 100.0 * np.array([(gam[mask] <= e).mean() for e in edges])
    ax_cum.fill_between(edges, 0.0, cum, color="#3b5b92", alpha=0.13)
    ax_cum.plot(edges, cum, color="#3b5b92", lw=1.4)
    ax_cum.axvline(1.0, color="#b23a48", lw=1.0, ls=(0, (3.0, 2.0)))
    ax_cum.plot([1.0], [pass_rate], marker="o", ms=4.4, mfc="white",
                mec="#b23a48", mew=1.3, zorder=5)
    ax_cum.annotate(f"{pass_rate:.1f} % pass", xy=(1.0, pass_rate),
                    xytext=(1.18, pass_rate - 26.0), fontsize=6.6,
                    color="#b23a48", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#b23a48"))
    ax_cum.text(0.06, 96.0,
                f"mean $\\gamma$ = {gam_mean:.2f}\n"
                f"$\\gamma_{{95}}$ = {np.percentile(gam[mask], 95.0):.2f}\n"
                f"n = {n_eval} points", fontsize=6.2, color=SUB, ha="left",
                va="top", linespacing=1.7)
    ax_cum.set_xlim(0.0, 2.0)
    ax_cum.set_ylim(0.0, 104.0)
    ax_cum.set_yticks(np.arange(0, 101, 25))
    ax_cum.set_xlabel(r"gamma index $\gamma$ (-)", fontsize=7.2)
    ax_cum.set_ylabel("points with $\\gamma$ below (%)", fontsize=7.2)
    ax_cum.tick_params(labelsize=6.5)
    ax_cum.set_title("Cumulative gamma distribution", fontsize=8.5,
                     fontweight="bold", color=INK, pad=4.0)

    # ---- aux 2: pass rate under four acceptance criteria ------------------
    rates = [pass_rate] + [100.0 * (gamma_index(meas, calc, dd, dta)[mask]
                                    <= 1.0).mean() for dd, dta in CRITERIA[1:]]
    names = [f"{dd:.0f} % /\n{dta:.0f} mm" for dd, dta in CRITERIA]
    pos = np.arange(len(rates), dtype=float)
    cols = ["#2e7d5b" if r >= ACTION_LIMIT else "#b23a48" for r in rates]
    ax_cri.bar(pos, rates, width=0.56, color=cols, edgecolor="white",
               linewidth=0.5, zorder=3)
    for i, r in enumerate(rates):
        ax_cri.text(i, r + 1.0, f"{r:.1f}", ha="center", va="bottom",
                    fontsize=6.3, color=SUB)
    ax_cri.axhline(ACTION_LIMIT, color="#3a3a3a", lw=1.0, ls=(0, (2.4, 1.4)),
                   zorder=4)
    ax_cri.text(len(rates) - 0.45, ACTION_LIMIT - 2.5,
                f"action limit {ACTION_LIMIT:.0f} %", fontsize=6.2,
                color=MUTED, ha="right", va="top")
    ax_cri.set_xticks(pos)
    ax_cri.set_xticklabels(names, fontsize=6.3)
    ax_cri.tick_params(axis="y", labelsize=6.5)
    ax_cri.tick_params(axis="x", length=0, pad=1.5)
    ax_cri.set_ylim(0.0, 114.0)
    ax_cri.set_yticks(np.arange(0, 101, 25))
    ax_cri.set_ylabel("pass rate (%)", fontsize=7.2)
    ax_cri.set_xlabel("acceptance criterion (dose difference / DTA)",
                      fontsize=7.2)
    ax_cri.set_title("Pass rate tightens with the criterion", fontsize=8.5,
                     fontweight="bold", color=INK, pad=4.0)



    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.20)
    panel_labels_inches(fig, [ax_cum, ax_cri], labels=["(b)", "(c)"],
                        dx=-0.62, dy=0.20)
    savefig(fig, "873_rtx_gamma_analysis.png")


if __name__ == "__main__":
    main()
