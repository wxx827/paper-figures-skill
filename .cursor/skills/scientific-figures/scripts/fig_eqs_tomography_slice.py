"""Shear-velocity tomography across a subduction zone, with a resolution test.

The perturbation field is built once and then only read.  A subducting slab is
a Gaussian tube around a dipping polyline that stagnates above the 660 km
discontinuity, the mantle wedge and the plume are Gaussian blobs, and a
smoothed random field stands in for everything the model does not resolve;
the earthquakes plotted on top are placed on the slab surface itself, which is
why the Wadati-Benioff zone follows the fast anomaly rather than merely
sitting near it.  The side panels come from the same arrays: the checkerboard
test is that input pattern passed through the same blurring and depth-fading
operator, with its recovered amplitude measured back off the result, and the
two profiles are columns lifted straight out of the main image.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#22406e"
BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREY = "#8a8a8a"

FIG_W, FIG_H = 11.0, 5.4
MAIN = (0.86, 1.28, 5.96, 2.98)
AUX_T = (7.36, 3.04, 3.46, 1.22)
AUX_B = (7.36, 1.28, 3.46, 1.22)

X_MAX, Z_MAX = 2400.0, 1200.0      # km
DX = 5.0                           # km, grid step
V_LIM = 3.0                        # per cent, colour limit
SLAB = [(1150.0, 0.0), (1235.0, 80.0), (1325.0, 190.0), (1420.0, 320.0),
        (1505.0, 455.0), (1560.0, 580.0), (1625.0, 655.0), (1790.0, 685.0),
        (1980.0, 698.0), (2180.0, 702.0)]
PROFILES = [(1500.0, "through the slab", BLUE),
            (1020.0, "through the wedge", RED)]

DIVERGE = LinearSegmentedColormap.from_list("house_div", [
    "#7d1f2b", "#b23a48", "#d79aa2", "#eed6d9", "#f7f7f7",
    "#d6dfec", "#a8bcd8", "#3b5b92", "#1b3560"])


def halo(ax, x, y, s, color, ha="center", va="center", rot=0.0, size=6.3,
         weight="normal"):
    """Label that stays readable wherever it lands on the colour field."""
    t = ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
                ha=ha, va=va, rotation=rot, zorder=8)
    t.set_path_effects([pe.withStroke(linewidth=1.9, foreground="white")])
    return t


def slab_curve(n=1600):
    p = np.asarray(SLAB, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0),
                                                        axis=1))])
    u = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(u, s, p[:, 0]),
                            np.interp(u, s, p[:, 1])])


def blob(xg, zg, x0, z0, sx, sz, amp, tilt=0.0):
    c, s = np.cos(tilt), np.sin(tilt)
    u = (xg - x0) * c + (zg - z0) * s
    v = -(xg - x0) * s + (zg - z0) * c
    return amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sz) ** 2))


def build_field(rng, xg, zg):
    """Per-cent shear-velocity perturbation on the section grid."""
    curve = slab_curve()
    tree = cKDTree(curve)
    d, _ = tree.query(np.column_stack([xg.ravel(), zg.ravel()]))
    d = d.reshape(xg.shape)
    fade = 1.0 - 0.35 * np.clip(zg / Z_MAX, 0.0, 1.0)
    field = 2.70 * fade * np.exp(-0.5 * (d / 58.0) ** 2)

    field += blob(xg, zg, 1020.0, 150.0, 150.0, 95.0, -2.30)
    field += blob(xg, zg, 1120.0, 330.0, 105.0, 150.0, -1.30)
    field += blob(xg, zg, 380.0, 480.0, 85.0, 330.0, -1.75)
    field += blob(xg, zg, 300.0, 90.0, 260.0, 80.0, 1.55)
    field += blob(xg, zg, 700.0, 900.0, 300.0, 180.0, 0.85)
    field += blob(xg, zg, 1950.0, 120.0, 320.0, 70.0, 0.95)
    field += blob(xg, zg, 2050.0, 420.0, 220.0, 140.0, -0.80)

    rough = gaussian_filter(rng.normal(size=xg.shape), 11.0)
    field += 0.95 * rough / (np.abs(rough).max() + 1e-12)
    return gaussian_filter(field, 3.0)


def wadati_benioff(rng, n=280):
    """Hypocentres on the slab surface, thinning with depth."""
    curve = slab_curve()
    z = curve[:, 1]
    w = np.exp(-z / 260.0) + 0.22 * np.exp(-((z - 590.0) / 55.0) ** 2)
    w = w / w.sum()
    idx = rng.choice(curve.shape[0], size=n, p=w)
    tan = np.gradient(curve, axis=0)
    tan = tan / np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.column_stack([tan[:, 1], -tan[:, 0]])
    off = rng.normal(-14.0, 17.0, n)
    pts = curve[idx] + nrm[idx] * off[:, None]
    return pts[:, 0], np.clip(pts[:, 1], 3.0, 690.0)


def blur_operator(field, zg):
    """What a real inversion returns: smeared, depth-faded, edge-starved."""
    out = (0.55 * gaussian_filter(field, (10.0, 5.0))
           + 0.45 * gaussian_filter(field, (4.0, 14.0)))
    return out * (0.95 - 0.55 * np.clip(zg / Z_MAX, 0.0, 1.0))


def checkerboard(xg, zg, cell=250.0, amp=2.0):
    return amp * (np.sin(np.pi * xg / cell) * np.sin(np.pi * zg / cell))


def main_panel(fig, ax, xg, zg, field, eq, norm):
    im = ax.imshow(field, origin="upper", cmap=DIVERGE, norm=norm,
                   extent=(0.0, X_MAX, Z_MAX, 0.0), interpolation="bilinear",
                   zorder=2)
    ax.contour(xg, zg, field, levels=[1.2], colors=["#12294d"],
               linewidths=0.7, zorder=4)
    ax.contour(xg, zg, field, levels=[-1.2], colors=["#7d1f2b"],
               linewidths=0.7, linestyles=[(0, (3.0, 2.0))], zorder=4)

    for z, lab in [(410.0, "410 km"), (660.0, "660 km")]:
        ax.axhline(z, color="#4a4a4a", lw=0.8, ls=(0, (5.0, 3.0)), zorder=5)
        ax.text(40.0, z - 18.0, lab, fontsize=6.2, color="#3a3a3a",
                ha="left", va="bottom", zorder=7)

    ax.plot(eq[0], eq[1], ls="none", marker="o", ms=1.9, mfc="#141414",
            mec="none", alpha=0.75, zorder=6)

    tx = SLAB[0][0]
    ax.plot([tx], [10.0], marker="v", ms=8.0, mfc="#141414", mec="white",
            mew=0.7, zorder=9)
    ax.annotate("", xy=(tx + 90.0, 52.0), xytext=(tx + 420.0, 52.0),
                arrowprops=dict(arrowstyle="-|>", color="#141414", lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=9),
                zorder=9)
    halo(ax, tx + 440.0, 52.0, "plate motion", "#141414", ha="left")
    halo(ax, tx - 40.0, 52.0, "trench", "#141414", ha="right", weight="bold")

    for x0, lab, col in PROFILES:
        ax.axvline(x0, color=col, lw=0.9, ls=(0, (2.6, 2.0)), zorder=6)
        halo(ax, x0 + 16.0, 1170.0, lab, col, ha="left", va="bottom",
             rot=90.0, size=6.1)

    halo(ax, 1660.0, 500.0, "fast slab", "#0f2547", ha="left", rot=-58.0,
         size=6.6, weight="bold")
    halo(ax, 960.0, 175.0, "slow wedge", "#6d1f2a", ha="right", size=6.6,
         weight="bold")
    halo(ax, 380.0, 590.0, "slow upwelling", "#6d1f2a", rot=90.0, size=6.6,
         weight="bold")
    halo(ax, 1830.0, 760.0, "stagnant slab above 660 km", "#0f2547",
         va="top", size=6.2)

    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(Z_MAX, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("horizontal distance along the section (km)")
    ax.set_ylabel("depth (km, positive down)")
    ax.tick_params(labelsize=6.8)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color("#8a8a8a")
    ax.set_title("Shear-velocity perturbation across an ocean-continent "
                 "subduction zone", fontsize=9.0, fontweight="bold",
                 color=INK, pad=5.0)

    ax.add_patch(Rectangle((0.022, 0.055), 0.315, 0.175,
                           transform=ax.transAxes, facecolor="white",
                           alpha=0.86, edgecolor="#cccccc", linewidth=0.5,
                           zorder=9))
    cax = ax.inset_axes([0.046, 0.088, 0.266, 0.042])
    cax.set_zorder(10)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#8a8a8a")
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.0, pad=1.6)
    cb.ax.set_title("dVs (%): red slow, blue fast", fontsize=6.2, color=INK,
                    pad=3.0)
    return im


def checker_grid(cell):
    x = np.arange(0.0, 1700.0 + DX, DX)
    z = np.arange(0.0, 600.0 + DX, DX)
    xg, zg = np.meshgrid(x, z)
    inp = checkerboard(xg, zg, cell=cell)
    return xg, zg, inp, blur_operator(inp, zg)


def checker_panel(ax, norm, cell, recovery):
    xg, zg, inp, out = checker_grid(cell)
    ax.imshow(out, origin="upper", cmap=DIVERGE, norm=norm,
              extent=(0.0, 1700.0, 600.0, 0.0), interpolation="bilinear",
              zorder=2)
    ax.contour(xg, zg, inp, levels=[0.0], colors=["#3a3a3a"],
               linewidths=0.45, linestyles=[(0, (2.4, 2.0))], zorder=4)
    ax.set_xlim(0.0, 1700.0)
    ax.set_ylim(600.0, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("distance (km)", fontsize=6.9)
    ax.set_ylabel("depth (km)", fontsize=6.9)
    ax.tick_params(labelsize=6.4)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color("#8a8a8a")
    ax.text(0.985, 0.06,
            "{:.0f} km cells, {:.0f} % of the input amplitude recovered"
            .format(cell, 100.0 * recovery), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, color="#2c2c2c",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.82), zorder=8)
    ax.set_title("Checkerboard resolution test", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def profile_panel(ax, x, z, field):
    for x0, lab, col in PROFILES:
        col_idx = int(np.argmin(np.abs(x - x0)))
        ax.plot(field[:, col_idx], z, color=col, lw=1.3, zorder=6,
                label="{:s}, x = {:.0f} km".format(lab, x0))
    ax.axvline(0.0, color=GREY, lw=0.7, zorder=4)
    for zl in (410.0, 660.0):
        ax.axhline(zl, color="#4a4a4a", lw=0.7, ls=(0, (5.0, 3.0)), zorder=5)
    ax.set_xlim(-V_LIM, V_LIM)
    ax.set_ylim(Z_MAX, 0.0)
    ax.set_xlabel("dVs (%)", fontsize=6.9)
    ax.set_ylabel("depth (km, positive down)", fontsize=6.9)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.45, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, labelspacing=0.22, borderaxespad=0.3)
    ax.set_title("Columns read out of the main image", fontsize=8.0,
                 fontweight="bold", color=INK, pad=3.0)


def audit(fig, foot, axes):
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
    rng = np.random.default_rng(77310542)

    x = np.arange(0.0, X_MAX + DX, DX)
    z = np.arange(0.0, Z_MAX + DX, DX)
    xg, zg = np.meshgrid(x, z)
    field = build_field(rng, xg, zg)
    eq = wadati_benioff(rng)

    cell = 250.0
    _, _, inp, out = checker_grid(cell)
    recovery = float(np.abs(out).max() / np.abs(inp).max())

    slab_peak = float(field.max())
    wedge_min = float(field.min())

    assert slab_peak > 1.5, "slab anomaly too weak to see"
    assert wedge_min < -1.5, "wedge anomaly too weak to see"
    assert 0.15 < recovery < 0.95, "resolution test is degenerate"
    assert eq[1].max() < 700.0, "seismicity should stop near 660 km"

    norm = Normalize(vmin=-V_LIM, vmax=V_LIM)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_chk = axes_inches(fig, AUX_T)
    ax_prof = axes_inches(fig, AUX_B)

    main_panel(fig, ax_main, xg, zg, field, eq, norm)
    checker_panel(ax_chk, norm, cell, recovery)
    profile_panel(ax_prof, x, z, field)



    panel_labels_inches(fig, [ax_main, ax_chk, ax_prof], dx=-0.60, dy=0.12)
    savefig(fig, "1054_eqs_tomography_slice.png")


if __name__ == "__main__":
    main()
