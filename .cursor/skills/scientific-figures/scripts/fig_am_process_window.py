"""Printability map in the laser power - scan speed plane.

The main panel is the claim: four defect mechanisms carve the plane into
regions, and the printable one is a wedge that narrows as the scan speed rises.
Region membership, the iso-energy lines drawn over it, the density of every
plotted coupon and the two supporting panels all come from the same pair of
model functions, so nothing on the page can drift out of step with anything
else.  The right-hand panels answer the two questions the map provokes: does
volumetric energy density alone predict density, and how much power headroom
does the operator actually have at a given speed.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, lighten, darken, savefig
from figkit import axes_inches, halo_text, lowess, panel_labels_inches
from figkit import safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GOLD = "#c9a227"        # balling
BLUE = "#3b5b92"        # lack of fusion
RED = "#b23a48"         # keyhole porosity
GREEN = "#2e7d5b"       # printable window
GREY = "#808080"

# Trimmed at the pale end: a near-white marker vanishes on the tinted regions.
DENS_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "density", get_cmap("cb_sequential_purple")(np.linspace(0.22, 1.0, 256)))

HATCH_S = 0.11          # mm, hatch spacing
LAYER_T = 0.03          # mm, layer thickness
V0, V1 = 200.0, 1600.0  # mm/s
P0, P1 = 50.0, 400.0    # W

E_LOF = 45.0            # J/mm^3, below this the melt pools stop overlapping
KEYHOLE = 12.5          # W s^0.5 / mm^0.5, depression collapse threshold
V_BALL = 1000.0         # mm/s
E_BALL = 62.0           # J/mm^3

REGION_NAMES = ["Lack of fusion", "Balling", "Keyhole porosity",
                "Printable window"]
REGION_FILL = [lighten(BLUE, 0.62), lighten(GOLD, 0.58), lighten(RED, 0.60),
               lighten(GREEN, 0.52)]
REGION_EDGE = [darken(BLUE, 0.15), darken(GOLD, 0.25), darken(RED, 0.15),
               darken(GREEN, 0.15)]


def energy_density(p, v):
    return p / (HATCH_S * LAYER_T * v)


def classify(p, v):
    """0 lack of fusion, 1 balling, 2 keyhole, 3 printable."""
    e = energy_density(p, v)
    reg = np.full(np.shape(e), 3, dtype=int)
    reg = np.where((v > V_BALL) & (e < E_BALL), 1, reg)
    reg = np.where(e < E_LOF, 0, reg)
    reg = np.where(p / np.sqrt(v) > KEYHOLE, 2, reg)
    return reg


def relative_density(p, v, rng=None):
    e = energy_density(p, v)
    d = 99.88 - 0.155 * np.clip(E_LOF + 6.0 - e, 0.0, None)
    d = d - 0.26 * np.clip(p / np.sqrt(v) - KEYHOLE + 0.8, 0.0, None)
    d = d - 0.055 * (np.clip((v - V_BALL) / 600.0, 0.0, None) *
                     np.clip(E_BALL - e, 0.0, None))
    if rng is not None:
        d = d + rng.normal(0.0, 0.055, np.shape(d))
    return np.clip(d, 96.2, 100.0)


def grids(nv=561, npw=481):
    v = np.linspace(V0, V1, nv)
    p = np.linspace(P0, P1, npw)
    VV, PP = np.meshgrid(v, p, indexing="xy")
    return v, p, VV, PP, classify(PP, VV)


def coupons(n=52, seed=11):
    rng = np.random.default_rng(seed)
    v = rng.uniform(V0 + 40.0, V1 - 40.0, n)
    p = rng.uniform(P0 + 12.0, P1 - 12.0, n)
    return v, p, relative_density(p, v, rng)


def label_point(v, p, reg, k):
    """Centroid of a region, snapped back onto it (regions are L-shaped)."""
    VV, PP = np.meshgrid(v, p, indexing="xy")
    m = reg == k
    if not m.any():
        return None
    cv, cp = VV[m].mean(), PP[m].mean()
    d = ((VV - cv) / (V1 - V0)) ** 2 + ((PP - cp) / (P1 - P0)) ** 2
    d = np.where(m, d, np.inf)
    i = np.unravel_index(np.argmin(d), d.shape)
    return float(VV[i]), float(PP[i])


def map_panel(fig, ax, v, p, reg, cv, cp, cd):
    cmap = mpl.colors.ListedColormap(REGION_FILL)
    ax.imshow(reg, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5,
              vmax=3.5, extent=[V0, V1, P0, P1], interpolation="nearest",
              zorder=0, rasterized=True)
    for k in range(4):
        ax.contour(v, p, (reg == k).astype(float), levels=[0.5],
                   colors=[REGION_EDGE[k]], linewidths=0.8, zorder=2)

    for e in (30.0, 50.0, 70.0, 100.0, 140.0):
        pv = e * HATCH_S * LAYER_T * v
        ok = (pv >= P0) & (pv <= P1)
        if not ok.any():
            continue
        ax.plot(v[ok], pv[ok], color=RULE, lw=0.7, ls=(0, (4.0, 2.4)),
                alpha=0.55, zorder=3)
        tag = f"{e:.0f}" + (r" J/mm$^3$" if e == 30.0 else "")
        halo_text(ax, v[ok][-1] - 12.0, pv[ok][-1] - 4.0, tag, ha="right",
                  va="top", fontsize=6.2, color=RULE, halo="white",
                  halo_lw=2.0, zorder=4)

    sc = ax.scatter(cv, cp, c=cd, cmap=DENS_CMAP, s=26.0, linewidths=0.6,
                    edgecolors="white", zorder=6)
    for k, name in enumerate(REGION_NAMES):
        pt = label_point(v, p, reg, k)
        if pt is None:
            continue
        halo_text(ax, pt[0], pt[1], name, ha="center", va="center",
                  fontsize=7.4, color=REGION_EDGE[k], halo="white",
                  halo_lw=2.6, zorder=7, fontweight="bold")

    ax.set_xlim(V0, V1)
    ax.set_ylim(P0, P1)
    ax.set_xlabel("scan speed v (mm/s)", fontsize=8.0)
    ax.set_ylabel("laser power P (W)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    frac = 100.0 * (reg == 3).mean()
    ax.set_title(f"Only {frac:.0f} % of the parameter plane prints sound "
                 "material", fontsize=9.0, fontweight="bold", color=INK)
    return sc


def energy_panel(ax, cv, cp, cd):
    e = energy_density(cp, cv)
    ax.set_xlim(0.90 * e.min(), 1.04 * e.max())
    ax.scatter(e, cd, s=15.0, c=cd, cmap=DENS_CMAP, linewidths=0.5,
               edgecolors="white", zorder=4)
    xs, ys = lowess(e, cd, frac=0.42, n_out=180)
    ax.plot(xs, ys, color=INK, lw=1.4, zorder=5)

    good = ys >= 99.5
    if good.any():
        lo, hi = xs[good].min(), xs[good].max()
        ax.axvspan(lo, hi, color=GREEN, alpha=0.16, linewidth=0.0, zorder=1)
        ax.text(0.5 * (lo + hi), 96.55, f"{lo:.0f} - {hi:.0f}" +
                r" J/mm$^3$", ha="center", va="bottom", fontsize=6.5,
                color=GREEN)
    ax.axhline(99.5, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax.text(ax.get_xlim()[1], 99.56, "99.5 %", ha="right", va="bottom",
            fontsize=6.3, color=SUB)
    ax.set_xlabel(r"volumetric energy density $E_v$ (J/mm$^3$)", fontsize=7.4)
    ax.set_ylabel("relative density (%)", fontsize=7.4)
    ax.set_ylim(96.3, 100.15)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(r"$E_v$ alone leaves a spread", fontsize=8.2,
                 fontweight="bold", color=INK)


def window_panel(ax, v, p, reg):
    dp = p[1] - p[0]
    width = (reg == 3).sum(axis=0) * dp
    ax.fill_between(v, 0.0, width, color=GREEN, alpha=0.20, linewidth=0.0,
                    zorder=2)
    ax.plot(v, width, color=GREEN, lw=1.6, zorder=4)
    i = int(np.argmax(width))
    ax.plot([v[i]], [width[i]], marker="o", ms=4.0, mfc="white", mec=GREEN,
            mew=1.2, zorder=5)
    ax.annotate(f"widest at {v[i]:.0f} mm/s\n{width[i]:.0f} W of headroom",
                xy=(v[i], width[i]), xytext=(v[i] + 90.0, width[i] * 0.52),
                fontsize=6.5, color=INK, ha="left", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    v_hi = v[width >= 100.0]
    if v_hi.size:
        ax.axvline(v_hi.max(), color=RED, lw=0.9, ls=(0, (2.8, 1.8)),
                   zorder=3)
        ax.text(v_hi.max() - 25.0, ax.get_ylim()[1] * 0.06,
                f"100 W headroom ends\nat {v_hi.max():.0f} mm/s", ha="right",
                va="bottom", fontsize=6.3, color=RED, linespacing=1.3)
    ax.set_xlim(V0, V1)
    ax.set_ylim(0.0, max(width.max() * 1.28, 1.0))
    ax.set_xlabel("scan speed v (mm/s)", fontsize=7.4)
    ax.set_ylabel("window width (W)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Power headroom shrinks with speed", fontsize=8.2,
                 fontweight="bold", color=INK)
    return width


def main():
    set_pub_style()
    v, p, VV, PP, reg = grids()
    cv, cp, cd = coupons()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_map = axes_inches(fig, (0.80, 0.95, 5.00, 4.55))
    ax_en = axes_inches(fig, (7.10, 3.60, 2.75, 1.90))
    ax_win = axes_inches(fig, (7.10, 0.95, 2.75, 1.90))

    sc = map_panel(fig, ax_map, v, p, reg, cv, cp, cd)
    safe_colorbar(fig, sc, ax_map, label="relative density (%)", width=0.11,
                  pad=0.08, labelsize=6.6)
    energy_panel(ax_en, cv, cp, cd)
    width = window_panel(ax_win, v, p, reg)

    v_hi = v[width >= 100.0]
    edge = v_hi.max() if v_hi.size else v[int(np.argmax(width))]
    panel_labels_inches(fig, [ax_map, ax_en, ax_win], dx=-0.50, dy=0.10)
    savefig(fig, "746_am_process_window.png")


if __name__ == "__main__":
    main()
