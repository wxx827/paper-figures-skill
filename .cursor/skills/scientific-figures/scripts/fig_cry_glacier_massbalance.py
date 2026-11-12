"""Figure 612 - Distributed surface mass balance of a valley glacier.

One balance field drives the whole figure.  The map in panel (a) is a mass
balance grid built from the glacier's own hypsometry (a two-gradient balance
profile plus a smoothed wind-redistribution term); the equilibrium line drawn
on it is the zero contour of that same grid rather than a line typed in at a
convenient elevation, and the profile in panel (b) is the grid binned by
elevation, so the two cross zero at the same place by construction.

The mapped year is the last year of the series in panel (c): the annual
offset applied to the grid is ``offs[-1]``, so the map, the profile and the
right-hand end of the cumulative curve all describe the same balance year.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, COLORS, darken
from figkit import (add_scale_bar, halo_text, panel_label_inches,
                    safe_colorbar, shared_norm)

X0, X1 = 0.0, 5.0            # map extent, km
Y0, Y1 = 0.0, 8.0
Z_TERM = 2540.0              # terminus elevation, m a.s.l.
DZ_DY = 208.0                # centre-line elevation gradient, m per km
ELA_GEOM = 3320.0            # elevation used to anchor the balance profile, m
GRAD_ABL = 0.0072            # ablation gradient, m w.e. per m of elevation
GRAD_ACC = 0.0034            # accumulation gradient, m w.e. per m
ACC_CAP = 2.00               # accumulation saturates at the crest, m w.e.
YEARS = np.arange(1981, 2025)

ICE = "#3b5b92"              # house blue, accumulation
MELT = "#b23a48"             # house red, ablation
ROCK = "#8a8a8a"


def build_grid(rng, nx=300, ny=460):
    """Elevation, glacier mask and the steady balance field."""
    x = np.linspace(X0, X1, nx)
    y = np.linspace(Y0, Y1, ny)
    xg, yg = np.meshgrid(x, y)
    centre = 2.45 + 0.55 * np.sin(2 * np.pi * yg / 12.5) \
        - 0.30 * np.cos(2 * np.pi * yg / 5.2)
    half = 0.30 + 0.105 * yg + 0.16 * np.sin(2 * np.pi * yg / 3.1)
    u = (xg - centre) / half
    mask = np.abs(u) <= 1.0

    # Valley walls rise with the *distance* outside the ice margin, so the
    # contour pattern stays a trough instead of exploding where the tongue
    # is narrow and |u| becomes large.
    out = np.clip(np.abs(xg - centre) - half, 0.0, None)
    z = (Z_TERM + DZ_DY * yg + 55.0 * np.clip(u ** 2, 0.0, 1.0)
         + 22.0 * np.sin(2 * np.pi * yg / 2.7)
         + 250.0 * np.clip(out, 0.0, 1.5))

    dz = z - ELA_GEOM
    b = np.where(dz < 0.0, GRAD_ABL * dz, np.minimum(GRAD_ACC * dz, ACC_CAP))
    wind = gaussian_filter(rng.normal(size=z.shape), 16.0)
    b = b + 0.26 * wind / (wind.std() or 1.0)
    return x, y, xg, yg, z, mask, b


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6120)
    x, y, xg, yg, z, mask, b_steady = build_grid(rng)

    dx = (X1 - X0) / (x.size - 1)
    dy = (Y1 - Y0) / (y.size - 1)
    area = float(mask.sum()) * dx * dy                    # km2

    # Annual offsets: a mildly negative regime that grows more negative.
    offs = (-0.14 - 0.0215 * (YEARS - YEARS[0])
            + rng.normal(0.0, 0.42, YEARS.size))
    base = float(np.nanmean(np.where(mask, b_steady, np.nan)))
    b_ann = base + offs                                   # m w.e. per year
    b_cum = np.cumsum(b_ann)

    # The mapped year IS the last year of the series.
    field = np.where(mask, b_steady + offs[-1], np.nan)
    norm = shared_norm([field], robust=(1, 99), symmetric=True)
    cmap = get_cmap("cb_diverging", reverse=True)

    # Equilibrium line: measured from the plotted grid, not assumed.
    band = mask & (np.abs(field) < 0.07)
    ela = float(np.nanmean(z[band]))
    y_ela = float(np.nanmean(yg[band]))
    acc_frac = float(np.nansum(field > 0.0)) / float(mask.sum())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.055, right=0.965, top=0.885, bottom=0.115,
                          wspace=0.34, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) distributed balance ------------------------------------------
    ax.set_facecolor("#f5f5f5")
    pm = ax.pcolormesh(xg, yg, field, cmap=cmap, norm=norm, shading="auto",
                       rasterized=True, zorder=2)
    cs = ax.contour(xg, yg, z, levels=np.arange(2600, 4401, 200),
                    colors=ROCK, linewidths=0.5, zorder=3)
    ax.clabel(cs, cs.levels[::2], fmt="%d", fontsize=6.0, inline=True,
              inline_spacing=6, colors="#666666")
    ax.contour(xg, yg, mask.astype(float), levels=[0.5], colors=["#3a3a3a"],
               linewidths=1.0, zorder=4)
    ax.contour(xg, yg, np.where(mask, field, np.nan), levels=[0.0],
               colors=["#1a1a1a"], linewidths=2.0, zorder=6)

    halo_text(ax, 1.52, y_ela, f"ELA {ela:.0f} m", fontsize=6.8, ha="right",
              va="center", color="#1a1a1a", halo="white", halo_lw=2.2,
              zorder=8)
    halo_text(ax, 2.45, 7.55, "accumulation area", fontsize=6.6, ha="center",
              va="center", color=darken(ICE, 0.30), halo="white",
              halo_lw=2.2, zorder=8)
    halo_text(ax, 2.05, 0.75, "ablation area", fontsize=6.6, ha="center",
              va="center", color=darken(MELT, 0.20), halo="white",
              halo_lw=2.2, zorder=8)

    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km)", fontsize=8.0)
    ax.set_ylabel("northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title(f"Balance year {YEARS[-1]}, glacier area {area:.1f} "
                 "km$^2$", fontsize=8.8, fontweight="bold", pad=4.0)
    add_scale_bar(ax, 1.0, "1 km", loc=(0.045, 0.045), height=0.020,
                  fontsize=6.2, n_seg=2)
    safe_colorbar(fig, pm, ax,
                  label="surface mass balance (m w.e. yr$^{-1}$)",
                  width=0.12, pad=0.08, labelsize=6.8)

    ax.legend(handles=[
        Line2D([0], [0], color="#1a1a1a", lw=2.0,
               label="equilibrium line (b = 0)"),
        Line2D([0], [0], color=ROCK, lw=0.5, label="elevation, 200 m"),
    ], loc="lower right", frameon=True, framealpha=0.90, facecolor="white",
        edgecolor="#cccccc", borderpad=0.4, fontsize=6.4, handlelength=1.7)

    # ---- (b) balance versus elevation -------------------------------------
    zb, bb = z[mask], field[mask]
    edges = np.arange(2500.0, 4401.0, 100.0)
    idx = np.digitize(zb, edges) - 1
    mids, mean_b, sd_b = [], [], []
    for k in range(edges.size - 1):
        sel = idx == k
        if sel.sum() < 25:
            continue
        mids.append(0.5 * (edges[k] + edges[k + 1]))
        mean_b.append(float(bb[sel].mean()))
        sd_b.append(float(bb[sel].std()))
    mids = np.asarray(mids)
    mean_b = np.asarray(mean_b)
    sd_b = np.asarray(sd_b)

    ax_p.fill_betweenx(mids, mean_b - sd_b, mean_b + sd_b,
                       facecolor="#cccccc", alpha=0.55, lw=0.0, zorder=2)
    ax_p.fill_betweenx(mids, 0.0, mean_b, where=mean_b >= 0.0,
                       facecolor=ICE, alpha=0.30, lw=0.0, zorder=3)
    ax_p.fill_betweenx(mids, 0.0, mean_b, where=mean_b < 0.0,
                       facecolor=MELT, alpha=0.30, lw=0.0, zorder=3)
    ax_p.plot(mean_b, mids, color="#1a1a1a", lw=1.5, zorder=5)
    ax_p.axvline(0.0, color=COLORS["baseline"], lw=0.8, zorder=4)
    ax_p.axhline(ela, color="#1a1a1a", lw=1.6, ls=(0, (5, 2.2)), zorder=6)
    ax_p.text(0.97, ela + 45.0, f"ELA {ela:.0f} m",
              transform=ax_p.get_yaxis_transform(), ha="right", va="bottom",
              fontsize=6.4, color="#1a1a1a")
    ax_p.set_ylim(mids.min() - 60.0, mids.max() + 60.0)
    ax_p.set_xlabel("mass balance (m w.e. yr$^{-1}$)", fontsize=7.6)
    ax_p.set_ylabel("elevation (m a.s.l.)", fontsize=7.6)
    ax_p.tick_params(labelsize=6.6)
    ax_p.grid(axis="x", lw=0.5, alpha=0.6)
    ax_p.set_title("Balance gradient", fontsize=8.5, fontweight="bold",
                   pad=3.5)
    for s in ("top", "right"):
        ax_p.spines[s].set_visible(False)

    # ---- (c) annual and cumulative balance --------------------------------
    cols = [ICE if v >= 0 else MELT for v in b_ann]
    ax_c.bar(YEARS, b_ann, width=0.78, color=cols, alpha=0.65,
             edgecolor="none", zorder=3)
    ax_c.axhline(0.0, color=COLORS["baseline"], lw=0.8, zorder=4)
    ax_c.set_ylabel("annual b (m w.e. yr$^{-1}$)", fontsize=7.2)
    ax_c.set_xlabel("year", fontsize=7.6)
    ax_c.tick_params(labelsize=6.6)
    ax_c.set_xlim(YEARS[0] - 1.2, YEARS[-1] + 1.2)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)

    ax_c2 = ax_c.twinx()
    ax_c2.plot(YEARS, b_cum, color="#1a1a1a", lw=1.6, zorder=6)
    ax_c2.set_ylabel("cumulative (m w.e.)", fontsize=7.2)
    ax_c2.tick_params(labelsize=6.6)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.plot([YEARS[-1]], [b_cum[-1]], marker="o", ms=3.6, mfc="white",
               mec="#1a1a1a", mew=1.0, zorder=7)
    ax_c.set_title(f"Cumulative loss {b_cum[-1]:.1f} m w.e. since "
                   f"{YEARS[0]}", fontsize=8.5, fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_p, "(b)"), (ax_c, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.44 if a is ax else -0.40,
                           dy=0.09)


    savefig(fig, "612_cry_glacier_massbalance.png")


if __name__ == "__main__":
    main()
