"""Figure 385 - EBSD inverse-pole-figure map with a misorientation spectrum.

Every grain of a Laguerre polycrystal carries a cubic orientation; grains that
share a parent get that parent's orientation plus a few degrees of scatter, so
the map contains both sub-grain (low-angle) and true (high-angle) boundaries.
Boundary segments are classified from the real misorientation between the two
grains they separate, and the same numbers drive the histogram.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from pubstyle import set_pub_style, savefig, annotate_box, style_axes, COLORS
from figkit import add_scale_bar, image_axes, panel_labels_inches

UM_PER_PX = 0.20
SHAPE = (350, 470)
LAGB_CUT = 15.0                      # deg, conventional LAGB / HAGB split
LAGB_C = "#c9a227"                   # house gold
HAGB_C = "#3b5b92"                   # house blue


# ---------------------------------------------------------------------------
# Microstructure
# ---------------------------------------------------------------------------
def smooth_noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def laguerre_grains(shape, n_grains, seed, relax=2, sigma_r=0.30,
                    warp=4.0, warp_sigma=8.5):
    ny, nx = shape
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.random(n_grains) * nx, rng.random(n_grains) * ny]
    radii = np.exp(sigma_r * rng.standard_normal(n_grains))
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    wx = (xx + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    wy = (yy + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    w = (radii * 0.55 * (nx * ny / n_grains) ** 0.5) ** 2
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
    return lab.reshape(shape)


# ---------------------------------------------------------------------------
# Crystallography
# ---------------------------------------------------------------------------
def cubic_symmetry():
    ops = []
    for perm in itertools.permutations(range(3)):
        for sgn in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for i, p in enumerate(perm):
                m[i, p] = sgn[i]
            if np.linalg.det(m) > 0:
                ops.append(m)
    return np.array(ops)


SYM = cubic_symmetry()


def misorientation_deg(ra, rb):
    """Smallest cubic-symmetry-equivalent rotation angle between two frames."""
    g = ra @ rb.T
    tr = np.einsum("kij,ji->k", SYM, g)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)).min())


def ipf_rgb(dirs):
    """IPF-Z colour key for cubic symmetry (red 001, green 101, blue 111)."""
    d = np.abs(np.atleast_2d(dirs))
    d = np.sort(d, axis=1)                       # h <= k <= l
    h, k, l = d[:, 0], d[:, 1], d[:, 2]
    rgb = np.stack([l - k, (k - h) * 1.0, h * np.sqrt(2.0)], axis=1)
    rgb /= np.maximum(rgb.max(axis=1, keepdims=True), 1e-9)
    return np.clip(rgb, 0.0, 1.0)


def triangle_key(ax, n=420):
    """Draw the standard stereographic triangle, filled with the IPF key."""
    xs = np.linspace(-0.01, 0.44, n)
    ys = np.linspace(-0.01, 0.40, n)
    X, Y = np.meshgrid(xs, ys)
    r2 = X ** 2 + Y ** 2
    z = (1 - r2) / (1 + r2)
    x = X * (1 + z)
    y = Y * (1 + z)
    inside = (y >= -1e-6) & (x >= y - 1e-6) & (z >= x - 1e-6)
    v = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    nrm = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    rgba = np.ones((v.shape[0], 4))
    rgba[:, :3] = ipf_rgb(v / nrm)
    rgba[:, 3] = inside.ravel().astype(float)
    ax.imshow(rgba.reshape(n, n, 4), origin="lower",
              extent=[xs[0], xs[-1], ys[0], ys[-1]], interpolation="bilinear")

    t = np.linspace(0, 1, 200)
    arc = np.stack([np.ones_like(t), t, np.ones_like(t)], axis=1)
    arc /= np.linalg.norm(arc, axis=1, keepdims=True)
    ax.plot(arc[:, 0] / (1 + arc[:, 2]), arc[:, 1] / (1 + arc[:, 2]),
            color=COLORS["rule"], lw=0.9)
    ax.plot([0, np.sqrt(2) - 1], [0, 0], color=COLORS["rule"], lw=0.9)
    ax.plot([0, arc[-1, 0] / (1 + arc[-1, 2])],
            [0, arc[-1, 1] / (1 + arc[-1, 2])], color=COLORS["rule"], lw=0.9)
    for (px, py, txt, ha, va) in [
            (0.0, 0.0, "001", "right", "top"),
            (np.sqrt(2) - 1, 0.0, "101", "left", "top"),
            (arc[-1, 0] / (1 + arc[-1, 2]), arc[-1, 1] / (1 + arc[-1, 2]),
             "111", "left", "bottom")]:
        ax.text(px, py, "  " + txt, ha=ha, va=va, fontsize=6.6,
                color=COLORS["ink"])
    ax.set_xlim(-0.035, 0.475)
    ax.set_ylim(-0.055, 0.415)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    n_grains = 210
    lab = laguerre_grains(SHAPE, n_grains, seed=7)
    parent = laguerre_grains(SHAPE, 13, seed=21, relax=3, sigma_r=0.18,
                             warp=9.0, warp_sigma=26.0)

    ny, nx = SHAPE
    yy, xx = np.mgrid[0:ny, 0:nx]
    cnt = np.maximum(np.bincount(lab.ravel(), minlength=n_grains), 1)
    cy = (np.bincount(lab.ravel(), weights=yy.ravel().astype(float),
                      minlength=n_grains) / cnt).astype(int)
    cx = (np.bincount(lab.ravel(), weights=xx.ravel().astype(float),
                      minlength=n_grains) / cnt).astype(int)
    g_parent = parent[np.clip(cy, 0, ny - 1), np.clip(cx, 0, nx - 1)]

    rng = np.random.default_rng(3)
    u = rng.random((13, 3))
    q = np.stack([np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1]),
                  np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1]),
                  np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2]),
                  np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])], axis=1)
    parent_R = Rotation.from_quat(q).as_matrix()

    axis = rng.standard_normal((n_grains, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    spread = np.radians(rng.gamma(2.0, 2.6, n_grains))
    dev = Rotation.from_rotvec(axis * spread[:, None]).as_matrix()
    R = np.einsum("nij,njk->nik", dev, parent_R[g_parent])

    dirs = R[:, 2, :]                       # sample Z in crystal frame
    grain_rgb = ipf_rgb(dirs)

    # ---- boundary segments and their misorientations ----------------------
    mis = np.zeros((n_grains, n_grains))
    hb = lab[:, :-1] != lab[:, 1:]
    vb = lab[:-1, :] != lab[1:, :]
    pair = np.unique(np.concatenate([
        np.stack([lab[:, :-1][hb], lab[:, 1:][hb]], axis=1),
        np.stack([lab[:-1, :][vb], lab[1:, :][vb]], axis=1)]), axis=0)
    for a, b in pair:
        if mis[a, b] == 0.0:
            m = misorientation_deg(R[a], R[b])
            mis[a, b] = mis[b, a] = m

    ang = np.full(SHAPE, np.nan)
    ang[:, :-1] = np.where(hb, mis[lab[:, :-1], lab[:, 1:]], ang[:, :-1])
    fill = vb & ~np.isfinite(ang[:-1, :])
    ang[:-1, :] = np.where(fill, mis[lab[:-1, :], lab[1:, :]], ang[:-1, :])

    seg = np.isfinite(ang)
    lagb = seg & (ang < LAGB_CUT)
    hagb = seg & (ang >= LAGB_CUT)
    lagb_frac = lagb.sum() / max(seg.sum(), 1)

    rgb = grain_rgb[lab]
    rgb[lagb] = np.array([0.62, 0.62, 0.62])
    rgb[hagb] = np.array([0.10, 0.10, 0.10])

    # ---- layout -----------------------------------------------------------
    fig = plt.figure(figsize=(7.8, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.60],
                          height_ratios=[0.90, 1.0], wspace=0.30, hspace=0.50,
                          left=0.035, right=0.985, top=0.885, bottom=0.115)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax_his = fig.add_subplot(gs[1, 1])

    ax_map.imshow(rgb, interpolation="nearest")
    image_axes(ax_map, color=COLORS["rule"], lw=0.7)
    ax_map.set_title("IPF-Z orientation map, step 0.20 \u00b5m",
                     fontsize=8.5, fontweight="bold", pad=3.5)
    add_scale_bar(ax_map, 20.0 / UM_PER_PX, label="20 \u00b5m",
                  loc=(0.028, 0.048), height=0.022, fontsize=6.5)
    ax_map.plot([], [], color="#9e9e9e", lw=1.4,
                label=f"LAGB  2\u2013{LAGB_CUT:.0f}\u00b0")
    ax_map.plot([], [], color="#1a1a1a", lw=1.4,
                label=f"HAGB  >{LAGB_CUT:.0f}\u00b0")
    ax_map.legend(loc="upper right", frameon=True, facecolor="white",
                  edgecolor=COLORS["grid"], framealpha=0.92, fontsize=6.8,
                  handlelength=1.5, borderpad=0.35)

    triangle_key(ax_key)
    ax_key.set_title("IPF colour key (cubic)", fontsize=8.0,
                     fontweight="bold", pad=2.0)

    edges = np.arange(0, 65, 2.5)
    a_lo = ang[lagb]
    a_hi = ang[hagb]
    ax_his.hist([a_lo, a_hi], bins=edges, stacked=True,
                color=[LAGB_C, HAGB_C], edgecolor="white", linewidth=0.35,
                label=["low-angle", "high-angle"])
    ax_his.axvline(LAGB_CUT, color=COLORS["accent"], lw=1.0, ls="--")
    ax_his.text(LAGB_CUT + 1.2, ax_his.get_ylim()[1] * 0.93,
                f"{LAGB_CUT:.0f}\u00b0", fontsize=6.6,
                color=COLORS["accent"], ha="left", va="top")
    ax_his.set_xlabel("misorientation angle (deg)", fontsize=7.5)
    ax_his.set_ylabel("boundary length (px)", fontsize=7.5)
    ax_his.set_xlim(0, 63)
    ax_his.tick_params(labelsize=6.5, length=2.5)
    ax_his.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_his)
    ax_his.legend(loc="upper right", frameon=False, fontsize=6.5,
                  handlelength=1.2, borderaxespad=0.2)
    ax_his.set_title("Boundary misorientation spectrum", fontsize=8.0,
                     fontweight="bold", pad=2.5)
    annotate_box(ax_his,
                 f"LAGB fraction = {100 * lagb_frac:.0f}%\n"
                 f"mean = {np.nanmean(ang):.1f}\u00b0",
                 loc="center right", fontsize=6.4)

    panel_labels_inches(fig, [ax_map, ax_key, ax_his], dx=-0.26, dy=0.08)

    savefig(fig, "385_mat_ebsd_orientation.png")


if __name__ == "__main__":
    main()
