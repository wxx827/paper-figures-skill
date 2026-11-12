"""Tablet compaction: three blends, one compressibility law, one working window.

Ten tablets per blend are compressed at each of eight main-compression
pressures.  The compactibility curve in the main panel is an exponential
saturation law fitted to those means, and every other number on the page is
read back off it: the operating window in the lower right panel is the stretch
of pressure where the fitted tensile strength lies inside the hardness
specification and stays below the blend's capping onset, so the bars cannot
disagree with the curves above them.  The Heckel panel regresses ln(1/(1-D))
on pressure over the linear stretch of the same tablets and reports the yield
pressure as the reciprocal of that slope.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
GREY = "#808080"
HAIR = "#cccccc"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

PRESS = np.array([25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0])
N_TAB = 10
TS_LSL, TS_USL = 1.70, 3.50      # tensile-strength specification, MPa
P_MAX = 320.0

# name, plateau strength (MPa), rate (1/MPa), Heckel yield pressure (MPa),
# Heckel intercept, capping onset (MPa), colour
BLENDS = [
    ("F1  MCC 70 % (plastic)", 4.55, 0.0122, 95.0, 0.60, 335.0, BLUE),
    ("F2  lactose 60 % (mixed)", 3.30, 0.0104, 135.0, 0.52, 250.0, GREEN),
    ("F3  DCP 55 % (brittle)", 2.30, 0.0089, 180.0, 0.44, 188.0, GOLD),
]


def compactibility(p, smax, k):
    """Exponential saturation of tensile strength with compaction pressure."""
    return smax * (1.0 - np.exp(-k * np.asarray(p, dtype=float)))


def heckel_density(p, py, icept):
    """Relative density predicted by the Heckel equation."""
    return 1.0 - np.exp(-(np.asarray(p, dtype=float) / py + icept))


def press_blend(rng, smax, k, py, icept, cap):
    """Ten tablets per pressure: tensile strength and relative density."""
    ts_true = compactibility(PRESS, smax, k)
    ts = ts_true[None, :] * rng.normal(1.0, 0.062, (N_TAB, PRESS.size))
    # Above the capping onset the compact delaminates and loses strength.
    over = np.clip((PRESS - cap) / 90.0, 0.0, None)
    ts *= np.exp(-1.15 * over)[None, :]
    d_true = heckel_density(PRESS, py, icept)
    dens = np.clip(d_true[None, :] + rng.normal(0.0, 0.0055,
                                                (N_TAB, PRESS.size)),
                   0.30, 0.995)
    return ts, dens


def fit_compactibility(mean, cap):
    """Fit only the pre-capping points; the descent is not part of the law."""
    keep = PRESS <= cap
    popt, _ = curve_fit(compactibility, PRESS[keep], mean[keep],
                        p0=[mean.max() * 1.2, 0.010],
                        bounds=([0.5, 0.001], [12.0, 0.10]), maxfev=40000)
    return popt


def fit_heckel(dens):
    """Slope of ln(1/(1-D)) on P over the linear stretch, 50-250 MPa."""
    y = np.log(1.0 / (1.0 - dens.mean(axis=0)))
    keep = (PRESS >= 50.0) & (PRESS <= 250.0)
    slope, icept = np.polyfit(PRESS[keep], y[keep], 1)
    return y, float(slope), float(icept)


def cross(x, y, level, rising=True):
    """First x where a monotone y passes through ``level``."""
    s = 1.0 if rising else -1.0
    idx = np.flatnonzero(s * y >= s * level)
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(x[0])
    return float(np.interp(s * level, [s * y[i - 1], s * y[i]],
                           [x[i - 1], x[i]]))


def window(popt, cap):
    """Pressure range giving an in-specification, non-capping compact."""
    pg = np.linspace(0.0, P_MAX, 3200)
    fit = compactibility(pg, *popt)
    p_lo = cross(pg, fit, TS_LSL, rising=True)
    p_hi_spec = cross(pg, fit, TS_USL, rising=True)
    if not np.isfinite(p_hi_spec):
        p_hi_spec = P_MAX
    return p_lo, min(p_hi_spec, cap), p_hi_spec


def main_panel(ax, data):
    ax.axhspan(TS_LSL, TS_USL, color="#eef2f6", zorder=0)
    for lim in (TS_LSL, TS_USL):
        ax.axhline(lim, color=RED, lw=1.15, ls=(0, (5.0, 2.2)), zorder=4)
    ax.text(4.0, TS_USL + 0.07,
            "tensile-strength specification {:.2f} - {:.2f} MPa".format(
                TS_LSL, TS_USL),
            ha="left", va="bottom", fontsize=6.6, color=RED, zorder=8,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0.90))

    pg = np.linspace(0.0, P_MAX, 600)
    for (name, _s, _k, _py, _i, cap, col) in BLENDS:
        d = data[name]
        ax.errorbar(PRESS, d["ts_mean"], yerr=d["ts_sd"], ls="none",
                    marker="o", ms=3.6, mfc="white", mec=col, mew=1.0,
                    ecolor=col, elinewidth=0.8, capsize=1.9, zorder=7)
        ax.plot(PRESS, d["ts_mean"], color=col, lw=1.5, alpha=0.35, zorder=5)
        fit = compactibility(pg, *d["popt"])
        ax.plot(pg[pg <= cap], fit[pg <= cap], color=col, lw=1.6, zorder=6)
        ax.plot(pg[pg >= cap], fit[pg >= cap], color=col, lw=1.0,
                ls=(0, (2.0, 2.0)), alpha=0.55, zorder=6)
        if cap <= P_MAX:
            y_cap = float(compactibility(cap, *d["popt"]))
            ax.plot([cap], [y_cap], marker="v", ms=6.0, mfc=RED, mec="white",
                    mew=0.9, zorder=9)
            ax.text(cap, y_cap + 0.13, "capping {:.0f} MPa".format(cap),
                    ha="center", va="bottom", fontsize=6.0, color=RED,
                    zorder=10,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                              edgecolor="none", alpha=0.90))
        ax.text(P_MAX - 4.0, compactibility(P_MAX - 4.0, *d["popt"]),
                name.split()[0], ha="left", va="center", fontsize=6.8,
                color=col, fontweight="bold", clip_on=False, zorder=8)

    handles = []
    for (name, _s, _k, _py, _i, cap, col) in BLENDS:
        d = data[name]
        handles.append(Line2D([], [], color=col, lw=1.6, marker="o", ms=3.6,
                              mfc="white", mec=col, mew=1.0,
                              label="{:s}   $\\sigma_{{max}}$ = {:.2f} MPa"
                                    .format(name, d["popt"][0])))
    handles += [
        Patch(facecolor="#eef2f6", edgecolor=RED, lw=1.0,
              label="specification band (quality standard)"),
        Line2D([], [], color=RED, lw=0.0, marker="v", ms=5.4, mfc=RED,
               mec="white", label="capping onset (defect limit)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, frameon=False,
              bbox_to_anchor=(0.998, 0.020), handlelength=1.9,
              borderaxespad=0.0, labelspacing=0.32)

    ax.set_xlim(0.0, P_MAX)
    ax.set_ylim(0.0, 5.05)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("main compression pressure (MPa)")
    ax.set_ylabel("radial tensile strength (MPa)")
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Compactibility, {:d} tablets per point, mean $\\pm$ 1 SD"
                 .format(N_TAB), fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def heckel_panel(ax, data):
    ax.axvspan(50.0, 250.0, color="#f2f5f8", zorder=0)
    ax.text(150.0, 0.025, "linear region used for the fit", ha="center",
            va="bottom", fontsize=6, color=MUTED, zorder=2,
            transform=ax.get_xaxis_transform())
    pg_in = np.linspace(50.0, 250.0, 40)
    pg_out = np.linspace(0.0, 318.0, 80)
    for (name, _s, _k, _py, _i, _c, col) in BLENDS:
        d = data[name]
        ax.plot(pg_out, d["h_icept"] + d["h_slope"] * pg_out, color=col,
                lw=0.8, ls=(0, (2.2, 2.0)), alpha=0.7, zorder=3)
        ax.plot(pg_in, d["h_icept"] + d["h_slope"] * pg_in, color=col, lw=1.4,
                zorder=4, label="{:s}   $P_y$ = {:.0f} MPa".format(
                    name.split()[0], 1.0 / d["h_slope"]))
        ax.plot(PRESS, d["heckel_y"], ls="none", marker="o", ms=3.2,
                mfc="white", mec=col, mew=0.9, zorder=5)
    ax.set_xlim(0.0, 320.0)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("compression pressure (MPa)", fontsize=7.2)
    ax.set_ylabel("$\\ln[1/(1-D)]$ (dimensionless)", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.1, frameon=False, handlelength=1.4,
              borderaxespad=0.25, labelspacing=0.26)
    ax.set_title("Heckel plot: yield pressure from the slope", fontsize=8.0,
                 fontweight="bold", color=INK)


def window_panel(ax, data):
    y = np.arange(len(BLENDS))[::-1]
    for yi, (name, _s, _k, _py, _i, cap, col) in zip(y, BLENDS):
        lo, hi, hi_spec = data[name]["win"]
        ok = hi > lo
        ax.barh(yi, hi - lo, left=lo, height=0.48,
                color=lighten(col if ok else RED, 0.55),
                edgecolor=col if ok else RED, linewidth=1.0, zorder=4)
        ax.plot([cap], [yi], marker="v", ms=5.4, mfc=RED, mec="white",
                mew=0.8, zorder=6, clip_on=False)
        tgt = 0.5 * (lo + hi)
        ax.plot([tgt], [yi], marker="*", ms=8.0, mfc=INK, mec="white",
                mew=0.6, zorder=7)
        ax.text(hi + 7.0, yi, "{:.0f} - {:.0f} MPa".format(lo, hi),
                ha="left", va="center", fontsize=6.3, color=col,
                fontweight="bold", zorder=7)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0].split()[0] for b in BLENDS], fontsize=6.6)
    ax.set_xlim(0.0, 340.0)
    ax.set_ylim(-0.60, len(BLENDS) - 0.34)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("compression pressure (MPa)", fontsize=7.2)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=INK, lw=0.0, marker="*", ms=7.0, mfc=INK,
                      mec="white", label="target pressure"),
               Line2D([], [], color=RED, lw=0.0, marker="v", ms=5.0, mfc=RED,
                      mec="white", label="capping onset")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.0, frameon=False,
              handlelength=1.0, borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Operating window read off the fitted curves", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Warn if the footnote box overlaps an in-view tick or axis label."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(9431)

    data = {}
    for (name, smax, k, py, icept, cap, col) in BLENDS:
        ts, dens = press_blend(rng, smax, k, py, icept, cap)
        mean, sd = ts.mean(axis=0), ts.std(axis=0, ddof=1)
        popt = fit_compactibility(mean, cap)
        hy, hslope, hicept = fit_heckel(dens)
        data[name] = dict(ts_mean=mean, ts_sd=sd, popt=popt, heckel_y=hy,
                          h_slope=hslope, h_icept=hicept,
                          win=window(popt, cap))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_hk = axes_inches(fig, AUX_T)
    ax_win = axes_inches(fig, AUX_B)

    main_panel(ax_main, data)
    heckel_panel(ax_hk, data)
    window_panel(ax_win, data)

    wid = [data[b[0]]["win"][1] - data[b[0]]["win"][0] for b in BLENDS]
    pys = [1.0 / data[b[0]]["h_slope"] for b in BLENDS]


    panel_labels_inches(fig, [ax_main, ax_hk, ax_win], dx=-0.62, dy=0.14)
    savefig(fig, "943_pha_tablet_compaction.png")


if __name__ == "__main__":
    main()
