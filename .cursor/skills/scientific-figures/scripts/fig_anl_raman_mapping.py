"""Raman map of an exfoliated graphene flake, with the spectra that produced it.

The map is the intensity ratio of the 2D band to the G band, which is the
standard way to count graphene layers, and it is the only field that is
simulated.  Everything else on the page is derived from it: the four point
spectra are *built* from the local ratio, so the 2D band drawn at P1 is tall
because the pixel under P1 is bright, and its width follows the same value
because the 2D band sharpens as the flake thins.  The D band, in turn, is
scaled by the gradient magnitude of the map, so defect intensity appears
exactly where the domains meet - which is where a real flake is damaged.

The histogram is the same array again, binned, with the layer thresholds
drawn on it; the area percentages quoted there are pixel counts, so they and
the coloured regions of the map are two views of one thing.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import axes_inches, panel_labels_inches, halo_text

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

X_UM, Y_UM = 40.0, 30.0
NX, NY = 240, 180
BASE_RATIO = 0.75

# centre x, centre y, radius (um), ratio offset, domain name
BLOBS = [
    (11.0, 20.0, 8.0, 2.15, "monolayer"),
    (26.0, 11.0, 7.0, 0.72, "bilayer"),
    (33.0, 23.5, 5.0, -0.35, "multilayer"),
]

# x um, y um, label, marker colour
POINTS = [
    (11.0, 20.0, "P1", RED),
    (26.0, 11.0, "P2", GOLD),
    (33.0, 23.5, "P3", PURPLE),
    (18.5, 15.5, "P4", GREEN),
]

# threshold on I(2D)/I(G), class name, colour
CLASSES = [
    (2.00, np.inf, "monolayer", RED),
    (1.00, 2.00, "bilayer", GOLD),
    (-np.inf, 1.00, "few / multilayer", BLUE),
]

# centre cm-1, FWHM cm-1, assignment
BANDS = [(1350.0, 42.0, "D"), (1580.0, 18.0, "G"), (1620.0, 15.0, "D'"),
         (2690.0, 0.0, "2D")]
RAMAN_LO, RAMAN_HI = 1200.0, 2900.0
LASER_NM = 532.0

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.30, 5.55, 4.16)
AUX_T = (7.42, 3.62, 3.18, 1.88)
AUX_B = (7.42, 1.28, 3.18, 1.72)


def build_map(rng):
    """The one field: I(2D)/I(G) over the scanned area."""
    x = np.linspace(0.0, X_UM, NX)
    y = np.linspace(0.0, Y_UM, NY)
    xx, yy = np.meshgrid(x, y)
    field = np.full_like(xx, BASE_RATIO)
    for cx, cy, rad, amp, _name in BLOBS:
        r = np.hypot(xx - cx, yy - cy) / rad
        field = field + amp * np.exp(-(r ** 4))
    # Flake-to-flake thickness wobble, then detector noise.
    field = field + 0.14 * gaussian_filter(rng.normal(0.0, 1.0, field.shape),
                                           9.0) * 9.0
    field = gaussian_filter(field, 2.0)
    field = np.clip(field + rng.normal(0.0, 0.020, field.shape), 0.02, None)

    gy, gx = np.gradient(field, y[1] - y[0], x[1] - x[0])
    edge = np.hypot(gx, gy)
    edge = edge / edge.max()
    return x, y, field, edge


def sample(x, y, arr, px, py):
    return float(arr[int(np.argmin(np.abs(y - py))),
                     int(np.argmin(np.abs(x - px)))])


def spectrum(wn, ratio, defect):
    """A point spectrum reconstructed from the two mapped quantities."""
    out = np.zeros_like(wn)
    amps = {"D": 0.06 + 0.90 * defect, "G": 1.00, "D'": 0.30 * defect,
            "2D": ratio}
    for centre, fwhm, name in BANDS:
        # The 2D band sharpens as the flake thins, so its width is read from
        # the same ratio that sets its height.
        w = fwhm if name != "2D" else 30.0 + 26.0 * (
            1.0 - float(np.clip(ratio / 2.9, 0.0, 1.0)))
        out = out + amps[name] / (1.0 + ((wn - centre) / (0.5 * w)) ** 2)
    return out


def map_panel(fig, ax, x, y, field, cmap):
    im = ax.imshow(field, origin="lower", cmap=cmap, interpolation="bilinear",
                   extent=(0.0, X_UM, 0.0, Y_UM), vmin=0.0,
                   vmax=float(field.max()))
    ax.set_aspect("equal")
    ax.contour(x, y, field, levels=[1.0, 2.0], colors=["white"],
               linewidths=0.7, alpha=0.85, zorder=4)

    for px, py, lab, col in POINTS:
        ax.plot([px], [py], marker="o", ms=6.0, mfc="none", mec=col, mew=1.5,
                zorder=6)
        halo_text(ax, px + 1.3, py + 1.1, lab, color=col, halo="white",
                  halo_lw=1.8, fontsize=7.4, fontweight="bold", ha="left",
                  va="bottom", zorder=7)

    ax.set_xlabel(r"x ($\mu$m)")
    ax.set_ylabel(r"y ($\mu$m)")
    ax.set_xticks(np.arange(0.0, 41.0, 10.0))
    ax.set_yticks(np.arange(0.0, 31.0, 10.0))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")

    # Parked just outside the frame rather than on it: the pale end of the
    # ramp fills the lower left of this map, and white bar labels on it were
    # unreadable while dark ones would vanish in the monolayer domain.
    cax = ax.inset_axes([1.022, 0.06, 0.026, 0.46])
    cb = fig.colorbar(im, cax=cax, orientation="vertical",
                      ticks=[0.0, 1.0, 2.0, 3.0])
    cb.set_label(r"I(2D) / I(G) (-)", fontsize=6.6, color=INK, labelpad=2.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.4, color="#333333",
                      pad=1.5)
    halo_text(ax, 0.030, 0.038, "white contours mark the layer thresholds "
              "at 1.0 and 2.0", transform=ax.transAxes, ha="left",
              va="bottom", fontsize=6.2, color=INK, halo="white",
              halo_lw=2.0, zorder=8)
    ax.set_title("Layer map of an exfoliated flake, 40 x 30 $\\mu$m at "
                 "{:.0f} nm excitation".format(LASER_NM), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def spectra_panel(ax, wn, specs, ratios):
    step = 3.7
    for k, ((px, py, lab, col), sp, r) in enumerate(
            zip(POINTS, specs, ratios)):
        base = (len(POINTS) - 1 - k) * step
        ax.plot(wn, base + sp, color=col, lw=0.85, zorder=5)
        ax.axhline(base, color=HAIR, lw=0.45, zorder=2)
        ax.text(RAMAN_LO + 30.0, base + 1.55,
                "{:s}  I(2D)/I(G) = {:.2f}".format(lab, r), ha="left",
                va="bottom", fontsize=6.0, color=col, fontweight="bold")

    top = (len(POINTS) - 1) * step
    for centre, _f, name in BANDS:
        if name == "D'":
            continue
        ax.text(centre, top + 3.35, "{:s}\n{:.0f}".format(name, centre),
                ha="center", va="bottom", fontsize=6, color=SUB,
                linespacing=1.15)
        ax.axvline(centre, color=HAIR, lw=0.5, ls=(0, (3.0, 2.2)), zorder=1)

    ax.set_xlim(RAMAN_LO, RAMAN_HI)
    ax.set_ylim(-0.45, top + 5.4)
    ax.set_yticks([])
    ax.set_xticks(np.arange(1200.0, 2901.0, 400.0))
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Point spectra rebuilt from the mapped ratio, "
                 "intensity in a.u.", fontsize=8.0, fontweight="bold",
                 color=INK, pad=6.0)


def histogram_panel(ax, field):
    flat = field.ravel()
    shares = []
    for lo, hi, name, col in CLASSES:
        frac = 100.0 * float(np.mean((flat >= lo) & (flat < hi)))
        shares.append(frac)
        ax.axvspan(max(lo, 0.0), min(hi, flat.max()), color=lighten(col, 0.86),
                   lw=0.0, zorder=1)
    ax.hist(flat, bins=70, range=(0.0, float(flat.max())), color="#6f7d8c",
            edgecolor="white", linewidth=0.25, zorder=4)
    for thr in (1.0, 2.0):
        ax.axvline(thr, color=INK, lw=0.9, ls=(0, (3.5, 2.0)), zorder=6)

    top = ax.get_ylim()[1]
    centres = [(2.00 + flat.max()) / 2.0, 1.5, 0.5]
    for (lo, hi, name, col), frac, cx in zip(CLASSES, shares, centres):
        ax.text(cx, top * 0.93, "{:s}\n{:.0f} % of area".format(name, frac),
                ha="center", va="top", fontsize=6, color=col,
                fontweight="bold", linespacing=1.3, zorder=7)

    ax.set_xlim(0.0, float(flat.max()))
    ax.set_ylim(0.0, top * 1.05)
    ax.set_xlabel(r"I(2D) / I(G) (-)", fontsize=7.2)
    ax.set_ylabel("pixels per bin (-)", fontsize=7.2)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Distribution of the mapped ratio over 43 200 pixels",
                 fontsize=8.0, fontweight="bold", color=INK, pad=6.0)
    return shares


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
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
    rng = np.random.default_rng(940)
    x, y, field, edge = build_map(rng)

    wn = np.linspace(RAMAN_LO, RAMAN_HI, 1700)
    ratios = [sample(x, y, field, px, py) for px, py, _l, _c in POINTS]
    defects = [sample(x, y, edge, px, py) for px, py, _l, _c in POINTS]
    specs = [spectrum(wn, r, d) for r, d in zip(ratios, defects)]

    cmap = get_cmap("cb_sequential_blue")
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_map = axes_inches(fig, MAIN)
    ax_sp = axes_inches(fig, AUX_T)
    ax_hi = axes_inches(fig, AUX_B)

    map_panel(fig, ax_map, x, y, field, cmap)
    spectra_panel(ax_sp, wn, specs, ratios)
    shares = histogram_panel(ax_hi, field)

    d_over_g = [0.06 + 0.90 * d for d in defects]
    panel_labels_inches(fig, [ax_map], labels=["(a)"], dx=-0.62, dy=0.14)
    # Kept clear of the colorbar that now sits off the right edge of the map.
    panel_labels_inches(fig, [ax_sp, ax_hi], labels=["(b)", "(c)"], dx=-0.34,
                        dy=0.14)
    savefig(fig, "940_anl_raman_mapping.png")


if __name__ == "__main__":
    main()
