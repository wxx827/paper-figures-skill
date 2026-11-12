"""Figure 617 - Probabilistic seismic hazard for a three-fault region.

This is one small probabilistic seismic hazard analysis rendered three ways.
Three fault sources are discretised into point ruptures, each carrying a share
of a truncated Gutenberg-Richter rate; a ground-motion model with log-normal
scatter converts every rupture into an exceedance probability; and the annual
rate of exceeding each ground-motion level is summed over ruptures and
magnitude bins.

Panel (a) is the map of the peak ground acceleration whose annual exceedance
rate equals 1/475 per year - the 10 % in 50 years design level.  Panel (b) is
the same hazard integral evaluated at four sites without collapsing it to one
number, and panel (c) is the ground-motion model that both of them use.  The
design values quoted on the map and in panel (b) are the same array read
twice.
"""
import textwrap

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

from pubstyle import set_pub_style, savefig, lighten, darken, COLORS
from figkit import add_scale_bar, halo_text, panel_label_inches, safe_colorbar

X1, Y1 = 120.0, 90.0                # map extent, km
RETURN_PERIOD = 475.0               # years, i.e. 10 % in 50 years
B_VALUE = 0.90
M_MIN, M_MAX = 5.0, 7.5
N_SEG = 15                          # point ruptures per fault
GM_C = (1.05, 0.85, 1.10, 0.0035)   # c0, c1, c2, c3 of the ground-motion model
GM_H = 7.0                          # pseudo-depth, km
GM_SIGMA = 0.55                     # ln-units scatter

ACCENT = "#b23a48"
# Hazard classes: one house-red ramp with monotone lightness (section 2.2).
PGA_LEVELS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
PGA_COLORS = [lighten(ACCENT, 0.90), lighten(ACCENT, 0.76),
              lighten(ACCENT, 0.60), lighten(ACCENT, 0.44),
              lighten(ACCENT, 0.26), ACCENT, darken(ACCENT, 0.30)]

FAULTS = [
    dict(name="F1", trace=[(22.0, 6.0), (28.0, 40.0), (36.0, 84.0)],
         rate=0.030),
    dict(name="F2", trace=[(8.0, 58.0), (52.0, 64.0), (112.0, 72.0)],
         rate=0.018),
    dict(name="F3", trace=[(78.0, 10.0), (90.0, 30.0), (100.0, 56.0)],
         rate=0.009),
]
SITES = [dict(name="A", xy=(40.0, 48.0)), dict(name="B", xy=(62.0, 40.0)),
         dict(name="C", xy=(100.0, 22.0)), dict(name="D", xy=(96.0, 76.0))]


def gmpe(mw, dist):
    """Median peak ground acceleration in g for magnitude and distance (km)."""
    c0, c1, c2, c3 = GM_C
    r = np.sqrt(np.asarray(dist, dtype=float) ** 2 + GM_H ** 2)
    return np.exp(c0 + c1 * (np.asarray(mw, dtype=float) - 6.0)
                  - c2 * np.log(r) - c3 * np.asarray(dist, dtype=float))


def rupture_set():
    """Point ruptures with their annual rates, and the magnitude bins."""
    xs, ys, rates = [], [], []
    for f in FAULTS:
        p = np.asarray(f["trace"], dtype=float)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        u = (np.arange(N_SEG) + 0.5) / N_SEG * s[-1]
        xs.append(np.interp(u, s, p[:, 0]))
        ys.append(np.interp(u, s, p[:, 1]))
        rates.append(np.full(N_SEG, f["rate"] / N_SEG))
    mags = np.arange(M_MIN + 0.125, M_MAX, 0.25)
    w = 10.0 ** (-B_VALUE * mags)
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(rates),
            mags, w / w.sum())


def hazard_curves(sx, sy, rx, ry, rate, mags, wmag, levels):
    """Annual rate of exceeding each level at each site: shape (sites, levels)."""
    dist = np.hypot(sx[:, None] - rx[None, :], sy[:, None] - ry[None, :])
    out = np.zeros((sx.size, levels.size))
    for m, w in zip(mags, wmag):
        ln_mu = np.log(gmpe(m, dist))
        for j, a in enumerate(levels):
            out[:, j] += (rate * w
                          * norm.sf((np.log(a) - ln_mu) / GM_SIGMA)).sum(axis=1)
    return out


def design_level(curve, levels, target):
    """Ground motion whose exceedance rate equals ``target``, log-interpolated."""
    ok = curve > 0.0
    if ok.sum() < 2:
        return float("nan")
    lg = np.log(curve[ok])[::-1]
    la = np.log(levels[ok])[::-1]
    return float(np.exp(np.interp(np.log(target), lg, la)))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6170)
    rx, ry, rate, mags, wmag = rupture_set()
    levels = np.geomspace(0.005, 2.0, 44)

    nx, ny = 84, 64
    gx = np.linspace(0.0, X1, nx)
    gy = np.linspace(0.0, Y1, ny)
    mx, my = np.meshgrid(gx, gy)
    grid = hazard_curves(mx.ravel(), my.ravel(), rx, ry, rate, mags, wmag,
                         levels)
    pga = np.array([design_level(c, levels, 1.0 / RETURN_PERIOD)
                    for c in grid]).reshape(ny, nx)

    site_xy = np.array([s["xy"] for s in SITES], dtype=float)
    site_curves = hazard_curves(site_xy[:, 0], site_xy[:, 1], rx, ry, rate,
                                mags, wmag, levels)
    for s, c in zip(SITES, site_curves):
        s["pga"] = design_level(c, levels, 1.0 / RETURN_PERIOD)

    # Historic catalogue drawn from the same sources.
    n_hist = 130
    pick = rng.choice(rx.size, n_hist, p=rate / rate.sum())
    hx = rx[pick] + rng.normal(0.0, 3.2, n_hist)
    hy = ry[pick] + rng.normal(0.0, 3.2, n_hist)
    hm = rng.choice(mags, n_hist, p=wmag)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.055, right=0.955, top=0.882, bottom=0.120,
                          wspace=0.40, hspace=0.46)
    ax = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[1, 1])

    # ---- (a) the hazard map -------------------------------------------------
    cmap = mpl.colors.ListedColormap(PGA_COLORS)
    cmap.set_under(lighten(ACCENT, 0.96))
    cmap.set_over(darken(ACCENT, 0.52))
    bnorm = mpl.colors.BoundaryNorm(PGA_LEVELS, cmap.N)
    pm = ax.pcolormesh(mx, my, pga, cmap=cmap, norm=bnorm, shading="auto",
                       rasterized=True, zorder=2)
    ax.contour(mx, my, pga, levels=PGA_LEVELS, colors="white", linewidths=0.4,
               alpha=0.7, zorder=3)

    ax.scatter(hx, hy, s=2.6 * 2.6 ** (hm - M_MIN), facecolor="none",
               edgecolor="#1a1a1a", linewidth=0.55, alpha=0.85, zorder=5)
    for f in FAULTS:
        p = np.asarray(f["trace"], dtype=float)
        ax.plot(p[:, 0], p[:, 1], color="#1a1a1a", lw=1.5, zorder=6,
                solid_capstyle="round")
        halo_text(ax, p[-1, 0], p[-1, 1] + 1.8, f["name"], fontsize=6.6,
                  ha="center", va="bottom", color="#1a1a1a", halo="white",
                  halo_lw=2.2, zorder=8)
    for s in SITES:
        ax.plot([s["xy"][0]], [s["xy"][1]], marker="s", ms=4.6, mfc="white",
                mec="#1a1a1a", mew=1.0, zorder=9)
        halo_text(ax, s["xy"][0] + 2.4, s["xy"][1],
                  f"{s['name']}  {s['pga']:.2f} g", fontsize=6.4, ha="left",
                  va="center", color="#1a1a1a", halo="white", halo_lw=2.2,
                  zorder=10)

    ax.set_xlim(0.0, X1)
    ax.set_ylim(0.0, Y1)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km)", fontsize=8.0)
    ax.set_ylabel("northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title(f"Peak ground acceleration, {RETURN_PERIOD:.0f}-year return "
                 "period", fontsize=8.8, fontweight="bold", pad=4.0)
    add_scale_bar(ax, 20.0, "20 km", loc=(0.035, 0.045), height=0.020,
                  fontsize=6.2, n_seg=2)
    safe_colorbar(fig, pm, ax, label="PGA, 10 % in 50 yr (g)", width=0.12,
                  pad=0.07, labelsize=6.8, ticks=PGA_LEVELS, extend="both")
    ax.legend(handles=[
        Line2D([0], [0], color="#1a1a1a", lw=1.5, label="active fault"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.4,
               markerfacecolor="none", markeredgecolor="#3a3a3a",
               markeredgewidth=0.45, label="historic epicentre (M$_w$)"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=4.0,
               markerfacecolor="white", markeredgecolor="#1a1a1a",
               markeredgewidth=1.0, label="reference site"),
    ], loc="lower right", frameon=True, framealpha=0.92, facecolor="white",
        edgecolor="#cccccc", borderpad=0.4, fontsize=6.3, handlelength=1.7)

    # ---- (b) hazard curves --------------------------------------------------
    cols = [darken(ACCENT, 0.30), ACCENT, "#3b5b92", "#8a8a8a"]
    for s, c, col in zip(SITES, site_curves, cols):
        ax_h.plot(levels, c, color=col, lw=1.4, zorder=4,
                  label=f"{s['name']}  {s['pga']:.2f} g")
        ax_h.plot([s["pga"]], [1.0 / RETURN_PERIOD], marker="o", ms=3.2,
                  mfc="white", mec=col, mew=1.0, zorder=6)
    ax_h.axhline(1.0 / RETURN_PERIOD, color="#1a1a1a", lw=1.0,
                 ls=(0, (4, 2.2)), zorder=5)
    ax_h.text(0.015, 1.0 / RETURN_PERIOD * 1.35,
              f"10 % in 50 yr  (1/{RETURN_PERIOD:.0f} yr)",
              transform=ax_h.get_yaxis_transform(), fontsize=6.3,
              color="#1a1a1a", ha="left", va="bottom")
    ax_h.set_xscale("log")
    ax_h.set_yscale("log")
    ax_h.set_ylim(2.0e-5, 1.2)
    ax_h.set_xlabel("PGA (g)", fontsize=7.6)
    ax_h.set_ylabel("annual exceedance rate (yr$^{-1}$)", fontsize=7.2)
    ax_h.tick_params(labelsize=6.6)
    ax_h.grid(lw=0.5, alpha=0.55, which="major")
    ax_h.legend(loc="lower left", frameon=False, fontsize=6.3,
                handlelength=1.6, labelspacing=0.26, ncol=2,
                columnspacing=1.0)
    ax_h.set_title("Hazard curves at the four sites", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    for sp in ("top", "right"):
        ax_h.spines[sp].set_visible(False)

    # ---- (c) the ground-motion model ---------------------------------------
    dist = np.geomspace(1.0, 200.0, 160)
    for m, col in zip((5.0, 6.0, 7.0), cols[:3]):
        med = gmpe(m, dist)
        ax_g.plot(dist, med, color=col, lw=1.4, zorder=4,
                  label=f"M$_w$ {m:.1f}")
        if m == 6.0:
            ax_g.fill_between(dist, med * np.exp(-GM_SIGMA),
                              med * np.exp(GM_SIGMA), facecolor=col,
                              alpha=0.16, lw=0.0, zorder=2)
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    ax_g.set_xlabel("Joyner-Boore distance (km)", fontsize=7.6)
    ax_g.set_ylabel("median PGA (g)", fontsize=7.2)
    ax_g.tick_params(labelsize=6.6)
    ax_g.grid(lw=0.5, alpha=0.55, which="major")
    ax_g.legend(loc="lower left", frameon=False, fontsize=6.3,
                handlelength=1.6, labelspacing=0.26)
    ax_g.text(0.97, 0.94, f"shaded: $\\pm$1$\\sigma$, $\\sigma$ = "
              f"{GM_SIGMA:.2f} ln units", transform=ax_g.transAxes,
              fontsize=6.3, color=COLORS["sub"], ha="right", va="top")
    ax_g.set_title("Ground-motion attenuation", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    for sp in ("top", "right"):
        ax_g.spines[sp].set_visible(False)

    # Both aux panels carry PGA; one range keeps them comparable (section 13.1).
    lo = min(levels.min(), gmpe(5.0, dist.max()))
    hi = max(levels.max(), gmpe(7.0, dist.min()))
    ax_h.set_xlim(lo, hi)
    ax_g.set_ylim(lo, hi)

    ax.apply_aspect()
    panel_label_inches(fig, ax, "(a)", dx=-0.44, dy=0.09)
    panel_label_inches(fig, ax_h, "(b)", dx=-0.46, dy=0.09)
    panel_label_inches(fig, ax_g, "(c)", dx=-0.46, dy=0.09)

    worst = max(SITES, key=lambda s: s["pga"])
    calm = min(SITES, key=lambda s: s["pga"])

    savefig(fig, "617_eqk_hazard_map.png")


if __name__ == "__main__":
    main()
