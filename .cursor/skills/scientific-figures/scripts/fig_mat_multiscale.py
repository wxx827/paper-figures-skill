"""Figure 393 - Four length scales of one alloy, from coupon to lattice plane.

The chain reads left to right: macro-etched section, optical grain structure,
a decorated grain boundary, and the lattice fringes across that boundary.  Each
tile carries a coloured ROI box and the next tile wears the same colour on its
frame, so the reader never has to guess which square became which picture.

Two things keep the chain honest.  The connector labels state the *true*
magnification step, which is 50x, 50x and 2000x - far too large for a
to-scale ROI box to remain visible - and the log ruler underneath plots the
real field of view of all four tiles on one axis, next to the feature sizes
they resolve.  The Hall-Petch panel then ties the grain size actually measured
in tile (b) to a mechanical property, which is the reason for climbing down
the scales in the first place.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle
from scipy.ndimage import gaussian_filter, binary_dilation

from pubstyle import (set_pub_style, savefig, get_cmap, lighten, darken,
                      style_axes, COLORS, CATEGORICAL)
from figkit import (image_axes, add_scale_bar, halo_text, corner_stamp,
                    panel_labels_inches, roi_rect)

NPX = 300                       # every tile is rendered at NPX x NPX
SEED = 23
ROI_FRAC = 1.0 / 11.0           # on-paper ROI size; true steps are printed

# name, field of view in metres, scale-bar length in metres, bar caption
LEVELS = [
    ("Macro-etched section", 20.0e-3, 5.0e-3, "5 mm"),
    ("Optical grain structure", 400.0e-6, 100.0e-6, "100 \u00b5m"),
    ("Decorated grain boundary", 8.0e-6, 2.0e-6, "2 \u00b5m"),
    ("Lattice fringes at the boundary", 4.0e-9, 1.0e-9, "1 nm"),
]
# The colour of a box is the colour of the frame it produces.
LINK_C = [CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2]]

A_LATT = 0.405e-9               # fcc aluminium lattice parameter, m
HP_S0, HP_K = 28.0, 68.0        # only used to synthesise the trial points


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(3, 9, 24), weights=(0.30, 0.34, 0.36)):
    out = sum(w * _sm(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _unit(n=NPX):
    """Coordinates in 0..1 across the tile."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return xx / (n - 1), yy / (n - 1)


# ---------------------------------------------------------------------------
# Level 1 - macro-etched cross-section of the coupon
# ---------------------------------------------------------------------------
def macro_section(rng, n=NPX):
    """Mounted and macro-etched coupon: resin, coupon edge, casting bands."""
    u, v = _unit(n)
    # Directional solidification: bands parallel to the chill face on top.
    band = np.sin(2 * np.pi * (v * 6.0 + 0.22 * _sm(rng, n, 26)))
    etch = (0.50 + 0.115 * band * (0.30 + 0.70 * v)
            + 0.085 * _fbm(rng, n, (3, 10, 30), (0.26, 0.34, 0.40)))
    # Columnar zone against the chill face, equiaxed core underneath.
    etch += 0.075 * np.exp(-(v / 0.20) ** 2) * _sm(rng, n, 2.2)

    inside = ((np.abs(u - 0.5) / 0.375) ** 6
              + (np.abs(v - 0.5) / 0.400) ** 6) <= 1.0
    resin = 0.885 + 0.020 * _sm(rng, n, 1.6)
    img = np.where(inside, etch, resin)
    rim = binary_dilation(~inside, np.ones((3, 3), bool)) & inside
    img = np.where(rim, 0.20, img)
    return np.clip(gaussian_filter(img, 0.7), 0.03, 0.97)


# ---------------------------------------------------------------------------
# Level 2 - optical grain structure (weighted Voronoi, etched look)
# ---------------------------------------------------------------------------
def grain_structure(rng, n=NPX, n_grains=120, relax=2):
    pts = np.c_[rng.random(n_grains) * n, rng.random(n_grains) * n]
    radii = np.exp(0.30 * rng.standard_normal(n_grains))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    wx = (xx + 3.2 * _sm(rng, (n), 8.0)).ravel()
    wy = (yy + 3.2 * _sm(rng, (n), 8.0)).ravel()
    w = (radii * 0.55 * (n * n / n_grains) ** 0.5) ** 2

    lab = np.zeros(wx.size, dtype=np.int32)
    for _ in range(relax + 1):
        best = np.full(wx.size, np.inf)
        for k in range(n_grains):
            d = (wx - pts[k, 0]) ** 2 + (wy - pts[k, 1]) ** 2 - w[k]
            m = d < best
            best[m] = d[m]
            lab[m] = k
        cnt = np.maximum(np.bincount(lab, minlength=n_grains), 1).astype(float)
        pts = np.c_[np.bincount(lab, weights=wx, minlength=n_grains) / cnt,
                    np.bincount(lab, weights=wy, minlength=n_grains) / cnt]
    lab = lab.reshape(n, n)

    bnd = np.zeros((n, n), bool)
    bnd[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    bnd[:-1, :] |= lab[:-1, :] != lab[1:, :]

    # Etch contrast: each grain takes a tone from its own orientation.
    tone = 0.32 + 0.52 * rng.random(n_grains)
    img = tone[lab] + 0.045 * _sm(rng, n, 1.4)
    img[binary_dilation(bnd, np.ones((2, 2), bool))] = 0.10
    return np.clip(gaussian_filter(img, 0.6), 0, 1), lab, bnd


def grain_diameters(lab, m_per_px):
    n = int(lab.max()) + 1
    area = np.bincount(lab.ravel(), minlength=n).astype(float)
    edge = np.zeros(n, bool)
    for ids in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        edge[np.unique(ids)] = True
    d = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi) * m_per_px
    keep = (~edge) & (area > 0)
    return d[keep]


# ---------------------------------------------------------------------------
# Level 3 - grain boundary decorated with precipitates
# ---------------------------------------------------------------------------
def boundary_tile(rng, n=NPX, fov=8.0e-6):
    u, v = _unit(n)
    path = (0.50 + 0.085 * np.sin(2.1 * np.pi * v)
            + 0.035 * np.sin(5.3 * np.pi * v))
    dist = np.abs(u - path)
    left = u < path
    img = np.where(left, 0.66, 0.50) + 0.038 * _fbm(rng, n, (2, 6, 16))
    # Diffraction contrast: the two grains carry differently oriented fringes.
    img += 0.018 * np.sin(2 * np.pi * (16 * u + 6 * v)) * left
    img += 0.018 * np.sin(2 * np.pi * (5 * u - 19 * v)) * (~left)
    img -= 0.36 * np.exp(-(dist / 0.0075) ** 2)         # boundary groove

    # Precipitates: bigger and denser on the boundary than in the matrix.
    prec = np.zeros((n, n))
    nm_px = fov / n
    cent, radii = [], []
    for _ in range(26):
        t = rng.random()
        cy = t * (n - 1)
        cx = (0.50 + 0.085 * np.sin(2.1 * np.pi * t)
              + 0.035 * np.sin(5.3 * np.pi * t)) * (n - 1)
        cx += rng.normal(0.0, 2.6)
        rr = rng.uniform(2.8, 6.4)
        cent.append((cx, cy))
        radii.append(rr)
    for _ in range(22):                                  # matrix dispersoids
        cent.append((rng.random() * n, rng.random() * n))
        radii.append(rng.uniform(1.4, 2.9))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    for (cx, cy), rr in zip(cent, radii):
        prec = np.maximum(prec, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                       / (2 * (rr * 0.62) ** 2)))
    img = img - 0.34 * prec
    img += 0.016 * rng.standard_normal((n, n))
    d_prec = 2.0 * np.array(radii[:26]) * nm_px
    return np.clip(gaussian_filter(img, 0.7), 0, 1), d_prec


# ---------------------------------------------------------------------------
# Level 4 - HRTEM lattice fringes across the same boundary
# ---------------------------------------------------------------------------
def lattice_tile(rng, n=NPX, fov=4.0e-9, a=A_LATT, tilt=(-8.0, 7.0)):
    m_px = fov / n
    step = a / np.sqrt(2.0) / m_px                       # {111} column spacing
    acc = np.zeros((n, n))
    boundary = 0.50 * n + 0.045 * n * np.sin(np.linspace(0, 2.4, n))
    for side, ang in zip((-1, 1), tilt):
        th = np.deg2rad(ang)
        k = int(2.2 * n / step)
        ii, jj = np.mgrid[-k:k + 1, -k:k + 1].astype(float)
        px = (ii * step * np.cos(th) - jj * step * np.sin(th)) + n / 2
        py = (ii * step * np.sin(th) + jj * step * np.cos(th)) + n / 2
        px, py = px.ravel(), py.ravel()
        ok = (px > -6) & (px < n + 6) & (py > -6) & (py < n + 6)
        px, py = px[ok], py[ok]
        edge = np.interp(py, np.arange(n), boundary)
        keep = (px - edge) * side > 0
        px, py = px[keep] + rng.normal(0, 0.30, keep.sum()), \
            py[keep] + rng.normal(0, 0.30, keep.sum())
        gi = np.clip(np.round(py).astype(int), 0, n - 1)
        gj = np.clip(np.round(px).astype(int), 0, n - 1)
        np.add.at(acc, (gi, gj), 1.0)
    img = gaussian_filter(acc, step * 0.26)
    img = img / (img.max() + 1e-9)
    img = img + 0.045 * _sm(rng, n, 1.0)
    return np.clip(0.10 + 0.90 * img, 0, 1), boundary, step * m_px


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    fov = [lv[1] for lv in LEVELS]
    mpp = [f / NPX for f in fov]

    img1 = macro_section(rng)
    img2, lab2, bnd2 = grain_structure(rng)
    img3, d_prec = boundary_tile(rng, fov=fov[2])
    img4, bpath, d_col = lattice_tile(rng, fov=fov[3])

    d_grain = grain_diameters(lab2, mpp[1])              # m
    d_mean_um = float(d_grain.mean()) * 1e6
    prec_nm = float(np.median(d_prec)) * 1e9
    steps = [fov[i] / fov[i + 1] for i in range(3)]

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.55],
                          left=0.035, right=0.988, top=0.880, bottom=0.118,
                          wspace=0.155, hspace=0.40)
    tiles = [fig.add_subplot(gs[0, j]) for j in range(4)]
    ax_rule = fig.add_subplot(gs[1, 0:3])
    ax_hp = fig.add_subplot(gs[1, 3])

    grey = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
    imgs = [img1, img2, img3, img4]

    # ---- the four tiles ----------------------------------------------------
    # The ROI centre of each tile is the feature the next tile shows.
    roi_centre = [(0.615, 0.335), (0.435, 0.585), (0.505, 0.470)]
    for j, ((name, f, bar_m, bar_txt), ax) in enumerate(zip(LEVELS, tiles)):
        frame_c = LINK_C[j - 1] if j > 0 else COLORS["rule"]
        image_axes(ax, color=frame_c, lw=1.5 if j > 0 else 0.8)
        ax.imshow(imgs[j], **grey)
        ax.set_xlim(-0.5, NPX - 0.5)
        ax.set_ylim(NPX - 0.5, -0.5)
        ax.set_title(name, fontsize=9.0, fontweight="bold", pad=4.0,
                     color=frame_c if j > 0 else COLORS["ink"])
        add_scale_bar(ax, bar_m / (f / NPX), bar_txt, loc=(0.040, 0.055),
                      height=0.020, fontsize=6.4)
        corner_stamp(ax, f"field of view {_fmt_len(f)}", loc="upper left",
                     fontsize=6.4)

    # boundary trace on the atomic tile, so the last link is unambiguous
    tiles[3].plot(bpath, np.arange(NPX), color=LINK_C[2], lw=0.9,
                  ls=(0, (4, 3)), alpha=0.85, zorder=5)
    halo_text(tiles[3], 0.962, 0.055,
              f"{{111}} spacing {d_col * 1e9:.3f} nm",
              transform=tiles[3].transAxes, ha="right", va="bottom",
              fontsize=6.4)

    # ---- ROI boxes and the connectors between tiles ------------------------
    box = ROI_FRAC * NPX
    for j in range(3):
        cx, cy = roi_centre[j]
        x0, y0 = cx * NPX - box / 2, cy * NPX - box / 2
        # White under-stroke: the box has to survive both a dark tile and a
        # light one, and these tiles carry both.
        roi_rect(tiles[j], (x0, y0, box, box), color="white", lw=2.4,
                 zorder=8)
        roi_rect(tiles[j], (x0, y0, box, box), color=LINK_C[j], lw=1.2,
                 zorder=9)
        for corner, target in (((x0 + box, y0), (0.0, 1.0)),
                               ((x0 + box, y0 + box), (0.0, 0.0))):
            con = ConnectionPatch(
                xyA=corner, coordsA=tiles[j].transData,
                xyB=target, coordsB=tiles[j + 1].transAxes,
                color=LINK_C[j], lw=0.7, alpha=0.85, zorder=10)
            fig.add_artist(con)
        bb0, bb1 = tiles[j].get_position(), tiles[j + 1].get_position()
        fig.text(0.5 * (bb0.x1 + bb1.x0), bb0.y0 - 0.014,
                 f"x {steps[j]:,.0f}".replace(",", "\u2009"), ha="center",
                 va="top", fontsize=7.0, color=LINK_C[j], fontweight="bold")

    # ---- (e) where the four tiles sit on one log ruler ---------------------
    for j, (name, f, _, _) in enumerate(LEVELS):
        col = LINK_C[j - 1] if j > 0 else COLORS["rule"]
        y = 3.0 - j
        ax_rule.plot([f / NPX, f], [y, y], lw=4.6, solid_capstyle="butt",
                     color=lighten(col, 0.50), zorder=2)
        ax_rule.plot([f], [y], marker="|", ms=7.0, mew=1.2, color=col,
                     zorder=4)
        ax_rule.text(f / NPX, y + 0.24,
                     f"({'abcd'[j]}) {name.lower()} - {_fmt_len(f)} wide, "
                     f"{_fmt_len(f / NPX)} per pixel",
                     fontsize=6.6, va="bottom", ha="left", color=col)

    feats = [(f"{{111}} plane spacing\n{d_col * 1e9:.3f} nm", d_col),
             (f"boundary precipitate\n{prec_nm:.0f} nm", prec_nm * 1e-9),
             (f"mean grain diameter\n{d_mean_um:.1f} \u00b5m",
              float(d_grain.mean())),
             ("coupon thickness\n12 mm", 12.0e-3)]
    for lbl, val in feats:
        ax_rule.plot([val], [-0.62], marker="v", ms=4.2,
                     color=COLORS["accent"], zorder=5)
        ax_rule.text(val, -0.92, lbl, fontsize=6.2, ha="center", va="top",
                     color=COLORS["sub"], linespacing=1.3)
    ax_rule.axhline(-0.62, color=COLORS["grid"], lw=0.7, zorder=1)

    ax_rule.set_xscale("log")
    ax_rule.set_xlim(4.0e-12, 6.0e-1)
    ax_rule.set_ylim(-2.30, 3.85)
    ax_rule.set_yticks([])
    ax_rule.set_xlabel("length (m)", fontsize=8.0)
    ax_rule.tick_params(axis="x", labelsize=7.0)
    ax_rule.set_title("Every tile on one ruler: eight decades in three steps",
                      fontsize=8.6, fontweight="bold", pad=3.5)
    for s in ("left", "right", "top"):
        ax_rule.spines[s].set_visible(False)
    ax_rule.grid(axis="x", linewidth=0.5, alpha=0.6)

    # ---- (f) Hall-Petch: the measured grain size becomes a strength --------
    d_trial = np.array([2.4, 4.8, 9.0, 16.0, 27.0, 44.0, 72.0])       # um
    sy = HP_S0 + HP_K / np.sqrt(d_trial) + rng.normal(0.0, 1.1,
                                                      d_trial.size)
    xs = 1.0 / np.sqrt(d_trial)
    slope, icept = np.polyfit(xs, sy, 1)
    xg = np.linspace(0.0, xs.max() * 1.12, 60)
    ax_hp.plot(xg, icept + slope * xg, color=COLORS["baseline"], lw=1.0,
               ls="--", zorder=2,
               label=f"fit: {icept:.0f} + {slope:.0f} d$^{{-1/2}}$")
    ax_hp.plot(xs, sy, marker="o", ms=3.4, ls="none", mfc="white",
               mec=CATEGORICAL[1], mew=1.0, zorder=3,
               label=f"heat-treatment trials (n = {d_trial.size})")
    x_here = 1.0 / np.sqrt(d_mean_um)
    y_here = icept + slope * x_here
    ax_hp.plot([x_here], [y_here], marker="*", ms=10.0,
               color=COLORS["accent"], zorder=5,
               label="this section")
    ax_hp.annotate(f"d = {d_mean_um:.1f} \u00b5m\n"
                   f"$\\sigma_y$ = {y_here:.0f} MPa",
                   xy=(x_here, y_here), xytext=(x_here + 0.055, y_here - 4.0),
                   fontsize=6.6, color=COLORS["accent"], ha="left",
                   va="top", linespacing=1.35,
                   arrowprops=dict(arrowstyle="->", color=COLORS["accent"],
                                   lw=0.7, shrinkA=0, shrinkB=2))
    ax_hp.set_xlabel("d$^{-1/2}$ (\u00b5m$^{-1/2}$)", fontsize=8.0)
    ax_hp.set_ylabel("yield strength (MPa)", fontsize=8.0)
    ax_hp.set_xlim(0.0, xs.max() * 1.12)
    ax_hp.set_ylim(min(sy.min(), y_here) - 13.0, sy.max() + 7.0)
    ax_hp.tick_params(labelsize=7.0)
    ax_hp.grid(linewidth=0.5, alpha=0.6)
    ax_hp.legend(loc="upper left", frameon=False, fontsize=6.5,
                 handlelength=1.5, borderaxespad=0.25)
    ax_hp.set_title("Grain size becomes a strength", fontsize=8.6,
                    fontweight="bold", pad=3.5)
    style_axes(ax_hp)

    panel_labels_inches(fig, tiles + [ax_rule, ax_hp], dx=-0.20, dy=0.11)
    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).

    savefig(fig, "393_mat_multiscale.png")


def _fmt_len(m):
    """Metres -> the unit a microscopist would actually print."""
    for scale, unit in ((1e-3, "mm"), (1e-6, "\u00b5m"), (1e-9, "nm"),
                        (1e-12, "pm")):
        if m >= scale:
            v = m / scale
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
    return f"{m:.2e} m"


if __name__ == "__main__":
    main()
