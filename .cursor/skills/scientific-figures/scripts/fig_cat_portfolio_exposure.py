"""Figure 723 - Where a catastrophe portfolio is concentrated.

Individual insured locations are drawn around six urban centres with a
lognormal sum insured and one of four lines of business.  Panel (a) plots
them as bubbles over the modelled 100-year gust field, so an accumulation is
visible as a cluster of large bubbles sitting inside a dark hazard band; the
single worst accumulation zone of the 6 x 5 zone grid is outlined.

Panels (b) and (c) aggregate exactly the same locations two ways: by region
and line of business, and as a concentration curve over the thirty zones with
the Gini coefficient and the Herfindahl index computed from those zone
totals.  No number in the figure is written by hand.
"""
import textwrap
import warnings

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, lighten, darken, COLORS
from figkit import (add_scale_bar, axes_inches, halo_text, panel_label_inches,
                    safe_colorbar)

FW, FH = 10.4, 6.4
MAIN_X, MAIN_W, MAIN_TOP = 0.62, 5.30, 5.50
AUX_X, AUX_W, AUX_GAP = 7.05, 2.95, 0.66

X1, Y1 = 600.0, 532.0               # map extent, km
NZX, NZY = 6, 5                     # accumulation-zone grid
N_LOC = 240

ACCENT = "#b23a48"
HAZ_LEVELS = [34.0, 39.0, 44.0, 49.0, 54.0, 59.0]
HAZ_COLORS = [lighten(ACCENT, 0.94), lighten(ACCENT, 0.86),
              lighten(ACCENT, 0.76), lighten(ACCENT, 0.64),
              lighten(ACCENT, 0.50)]
# Red is reserved for hazard here, so the lines of business take positions
# 2-5 of the house categorical order.
LOB = [("Residential", "#3b5b92"), ("Commercial", "#2e7d5b"),
       ("Industrial", "#c9a227"), ("Marine cargo", "#8e6bbf")]
LOB_P = [0.44, 0.28, 0.19, 0.09]

CENTRES = [(120.0, 380.0), (300.0, 300.0), (455.0, 155.0),
           (505.0, 400.0), (215.0, 120.0), (390.0, 455.0)]
CENTRE_W = [0.11, 0.21, 0.20, 0.19, 0.16, 0.13]
REGIONS = [("Coastal east", (505.0, 300.0)), ("Delta south", (300.0, 105.0)),
           ("Central plain", (300.0, 300.0)),
           ("Northern uplands", (300.0, 470.0)),
           ("Western basin", (95.0, 300.0))]


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def hazard(mx, my):
    """Modelled 100-year gust field, m/s: coastal gradient plus two maxima."""
    h = 33.0 + 0.030 * mx + 0.014 * (Y1 - my)
    h += 11.0 * np.exp(-(((mx - 500.0) / 130.0) ** 2
                         + ((my - 210.0) / 105.0) ** 2))
    h += 7.5 * np.exp(-(((mx - 330.0) / 150.0) ** 2
                        + ((my - 430.0) / 90.0) ** 2))
    h -= 5.0 * np.exp(-(((mx - 110.0) / 120.0) ** 2
                        + ((my - 330.0) / 130.0) ** 2))
    return h


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7230)

    # ---- exposure ----------------------------------------------------------
    pick = rng.choice(len(CENTRES), N_LOC, p=np.array(CENTRE_W) /
                      np.sum(CENTRE_W))
    cx = np.array([CENTRES[i][0] for i in pick])
    cy = np.array([CENTRES[i][1] for i in pick])
    px = np.clip(cx + rng.normal(0.0, 58.0, N_LOC), 12.0, X1 - 12.0)
    py = np.clip(cy + rng.normal(0.0, 52.0, N_LOC), 12.0, Y1 - 12.0)
    tiv = rng.lognormal(5.05, 1.05, N_LOC)          # sum insured, USD mn
    lob = rng.choice(len(LOB), N_LOC, p=LOB_P)
    # commercial and industrial risks sit closer to the centres, so they are
    # given the tighter scatter and the larger limits
    tiv[lob >= 1] *= 1.55
    total = float(tiv.sum())

    zx = np.clip((px / X1 * NZX).astype(int), 0, NZX - 1)
    zy = np.clip((py / Y1 * NZY).astype(int), 0, NZY - 1)
    zid = zy * NZX + zx
    zone_tiv = np.bincount(zid, weights=tiv, minlength=NZX * NZY)
    top = int(np.argmax(zone_tiv))
    tzx, tzy = top % NZX, top // NZX

    rcent = np.array([r[1] for r in REGIONS])
    reg = np.argmin((px[:, None] - rcent[None, :, 0]) ** 2
                    + (py[:, None] - rcent[None, :, 1]) ** 2, axis=1)
    table = np.zeros((len(REGIONS), len(LOB)))
    for r in range(len(REGIONS)):
        for l in range(len(LOB)):
            table[r, l] = tiv[(reg == r) & (lob == l)].sum()

    share = np.sort(zone_tiv)[::-1] / total
    hhi = float(np.sum(share ** 2) * 1.0e4)
    asc = np.sort(zone_tiv) / total
    lor = np.concatenate([[0.0], np.cumsum(asc)])
    frac = np.linspace(0.0, 1.0, lor.size)
    gini = float(1.0 - 2.0 * np.trapezoid(lor, frac))

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(FW, FH))
    main_h = MAIN_W * Y1 / X1
    main_y = MAIN_TOP - main_h
    ax = axes_inches(fig, (MAIN_X, main_y, MAIN_W, main_h))
    aux_h = (main_h - AUX_GAP) / 2.0
    ax_r = axes_inches(fig, (AUX_X, main_y + aux_h + AUX_GAP, AUX_W, aux_h))
    ax_c = axes_inches(fig, (AUX_X, main_y, AUX_W, aux_h))

    # ---- (a) exposure over the hazard field --------------------------------
    gx = np.linspace(0.0, X1, 200)
    gy = np.linspace(0.0, Y1, 178)
    mx, my = np.meshgrid(gx, gy)
    hz = hazard(mx, my)

    cmap = mpl.colors.ListedColormap(HAZ_COLORS)
    cmap.set_under(lighten(ACCENT, 0.975))
    cmap.set_over(lighten(ACCENT, 0.36))
    norm = mpl.colors.BoundaryNorm(HAZ_LEVELS, cmap.N)
    cf = ax.contourf(mx, my, hz, levels=HAZ_LEVELS, cmap=cmap, norm=norm,
                     extend="both", zorder=1)
    rasterize(cf)
    cf.set_edgecolor("face")
    ax.contour(mx, my, hz, levels=HAZ_LEVELS, colors="white", linewidths=0.6,
               alpha=0.9, zorder=2)

    for i in range(1, NZX):
        ax.plot([i * X1 / NZX] * 2, [0.0, Y1], color="#b3b3b3", lw=0.5,
                ls=(0, (2.4, 2.4)), zorder=3)
    for j in range(1, NZY):
        ax.plot([0.0, X1], [j * Y1 / NZY] * 2, color="#b3b3b3", lw=0.5,
                ls=(0, (2.4, 2.4)), zorder=3)

    order = np.argsort(tiv)[::-1]
    ax.scatter(px[order], py[order], s=0.085 * tiv[order],
               c=[LOB[k][1] for k in lob[order]], alpha=0.78,
               edgecolor="white", linewidth=0.45, zorder=5)

    ax.add_patch(Rectangle((tzx * X1 / NZX, tzy * Y1 / NZY), X1 / NZX,
                           Y1 / NZY, facecolor="none",
                           edgecolor=darken(ACCENT, 0.35), lw=1.6, zorder=7))
    # the caption has to stay inside the map whichever zone wins
    halo_text(ax, float(np.clip((tzx + 0.5) * X1 / NZX, 0.15 * X1,
                                0.85 * X1)),
              min((tzy + 1) * Y1 / NZY + 7.0, Y1 - 26.0),
              f"largest zone  {zone_tiv[top]:,.0f} USD mn"
              f"  ({zone_tiv[top] / total:.0%})", fontsize=6.5, ha="center",
              va="bottom", color=darken(ACCENT, 0.35), halo="white",
              halo_lw=2.6, zorder=8)

    for name, (rx_, ry_) in REGIONS:
        halo_text(ax, rx_, ry_, name.upper(), fontsize=6.0, ha="center",
                  va="center", color="#777777", halo="white", halo_lw=2.2,
                  zorder=4)

    ax.set_xlim(0.0, X1)
    ax.set_ylim(0.0, Y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title("Sum insured against modelled hazard", fontsize=8.8,
                 fontweight="bold", pad=4.0)
    add_scale_bar(ax, 100.0, "100 km", loc=(0.035, 0.045), height=0.020,
                  fontsize=6.2, n_seg=2)
    safe_colorbar(fig, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax,
                  label="100-year gust (m s$^{-1}$)", width=0.12, pad=0.08,
                  labelsize=6.8, ticks=HAZ_LEVELS, extend="both")

    size_keys = [100.0, 600.0, 2500.0]
    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      markerfacecolor=c, markeredgecolor="white",
                      markeredgewidth=0.45, markersize=6.0, label=n)
               for n, c in LOB]
    handles += [Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor="#d5d5d5", markeredgecolor="#777777",
                       markeredgewidth=0.45,
                       markersize=np.sqrt(0.085 * v),
                       label=f"{v:,.0f} USD mn") for v in size_keys]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              framealpha=0.92, facecolor="white", edgecolor="#cccccc",
              borderpad=0.42, fontsize=6.3, labelspacing=0.42,
              handletextpad=0.7, ncol=2, columnspacing=0.9)

    # ---- (b) exposure by region and line of business ------------------------
    ordr = np.argsort(table.sum(axis=1))
    y = np.arange(len(REGIONS), dtype=float)
    left = np.zeros(len(REGIONS))
    for l, (name, col) in enumerate(LOB):
        w = table[ordr, l] / 1e3                       # USD bn
        ax_r.barh(y, w, left=left, height=0.62, color=col, edgecolor="white",
                  linewidth=0.5, zorder=3, label=name)
        left += w
    for yy, v in zip(y, left):
        ax_r.text(v + left.max() * 0.02, yy,
                  f"{v:.1f}  ({v * 1e3 / total:.0%})", fontsize=6.2,
                  color="#1a1a1a", ha="left", va="center")
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([REGIONS[i][0] for i in ordr], fontsize=6.6)
    ax_r.set_xlim(0.0, left.max() * 1.36)
    ax_r.set_ylim(-0.68, len(REGIONS) - 0.32)
    ax_r.set_xlabel("sum insured (USD bn)", fontsize=7.4)
    ax_r.tick_params(labelsize=6.6)
    ax_r.grid(axis="x", lw=0.5, alpha=0.42)
    ax_r.set_title("Exposure by region and line of business", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_r.legend(loc="lower right", frameon=False, fontsize=6.2,
                handlelength=1.3, handleheight=0.9, labelspacing=0.26,
                ncol=2, columnspacing=0.9)
    for sp in ("top", "right"):
        ax_r.spines[sp].set_visible(False)

    # ---- (c) concentration across the zone grid -----------------------------
    ax_c.plot([0.0, 100.0], [0.0, 100.0], color="#808080", lw=0.9,
              ls=(0, (3.4, 2.2)), zorder=3, label="uniform spread")
    ax_c.plot(frac * 100.0, lor * 100.0, color=ACCENT, lw=1.8, zorder=5,
              label="observed zones")
    ax_c.fill_between(frac * 100.0, lor * 100.0, frac * 100.0,
                      facecolor=ACCENT, alpha=0.13, lw=0.0, zorder=2)
    worst5 = share[:5].sum()
    ax_c.plot([100.0 - 5.0 / (NZX * NZY) * 100.0] * 2, [0.0, 100.0],
              color="#333333", lw=0.8, ls=(0, (2.2, 1.8)), zorder=4)
    ax_c.text(100.0 - 5.0 / (NZX * NZY) * 100.0 - 2.0, 6.0,
              f"top 5 zones hold {worst5:.0%}", fontsize=6.3,
              color="#333333", ha="right", va="bottom")
    ax_c.set_xlim(0.0, 100.0)
    ax_c.set_ylim(0.0, 100.0)
    ax_c.set_xlabel("accumulation zones, least exposed first (%)",
                    fontsize=7.2)
    ax_c.set_ylabel("cumulative sum insured (%)", fontsize=7.2)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(lw=0.5, alpha=0.42)
    ax_c.set_title("Concentration over 30 accumulation zones", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.3,
                handlelength=1.7, labelspacing=0.26)
    ax_c.text(0.045, 0.60,
              f"Gini = {gini:.2f}\nHHI = {hhi:,.0f}", transform=ax_c.transAxes,
              fontsize=6.5, color="#1a1a1a", ha="left", va="bottom",
              linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                        edgecolor="#cccccc", linewidth=0.6, alpha=0.94))
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)

    ax.apply_aspect()
    for a, lab in ((ax, "(a)"), (ax_r, "(b)"), (ax_c, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.30 if a is ax else -0.52,
                           dy=0.09)

    hz_at = hazard(px, py)
    hot = float(tiv[hz_at >= HAZ_LEVELS[3]].sum() / total)

    savefig(fig, "723_cat_portfolio_exposure.png")


if __name__ == "__main__":
    main()
