"""Figure 389 - Fatigue crack advancing along grain boundaries, with Paris law.

Four snapshots of one crack in one field of view: the path is pinned to the
grain-boundary network, so it zig-zags and deflects at triple junctions.  The
cycle count and stress intensity printed on each panel come from integrating
the growth law drawn underneath, so the strip and the curve tell one story.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation, distance_transform_edt,
                           gaussian_filter)

from pubstyle import (set_pub_style, savefig, annotate_box, style_axes,
                      COLORS, CATEGORICAL)
from figkit import add_scale_bar, halo_text, image_axes, panel_labels_inches

UM_PER_PX = 1.0
SHAPE = (340, 360)
A0_M = 4.0e-5                 # crack length already grown before the window
D_SIGMA = 420.0               # MPa
Y_GEOM = 1.12
DK_TH = 5.2                   # MPa sqrt(m)
K_C = 24.0
PARIS_C = 3.2e-11
PARIS_M = 3.0


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


def boundary_mask(lab, width=1):
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return binary_dilation(b, np.ones((width, width), bool)) if width > 1 else b


def boundary_path(lab, y_start, seed=1):
    """Greedy left-to-right walk that prefers to sit on grain boundaries."""
    ny, nx = lab.shape
    dist = distance_transform_edt(~boundary_mask(lab, 1))
    rng = np.random.default_rng(seed)
    drift = 7.0 * smooth_noise(rng, (1, nx), 30.0).ravel()
    y = float(y_start)
    ys = np.empty(nx)
    for x in range(nx):
        lo, hi = int(max(y - 8, 3)), int(min(y + 9, ny - 3))
        cand = np.arange(lo, hi)
        cost = (dist[cand, x] + 0.13 * np.abs(cand - y)
                + 0.07 * np.abs(cand - (y_start + drift[x])))
        y = float(cand[np.argmin(cost)])
        ys[x] = y
    return gaussian_filter(ys, 1.2)


def dadn(dk):
    dk = np.asarray(dk, float)
    return (PARIS_C * dk ** PARIS_M * (1.0 - (DK_TH / dk) ** 2)
            / (1.0 - (dk / K_C) ** 2))


def dk_of(a_m):
    return D_SIGMA * Y_GEOM * np.sqrt(np.pi * np.asarray(a_m, float))


def build_matrix(lab, seed=5):
    rng = np.random.default_rng(seed)
    grey = 0.66 + 0.055 * (rng.random(int(lab.max()) + 1) - 0.5)[lab]
    grey -= 0.16 * boundary_mask(lab, 1)
    tex = (0.6 * smooth_noise(rng, SHAPE, 0.9)
           + 0.4 * smooth_noise(rng, SHAPE, 3.0))
    grey *= 1.0 + 0.045 * tex
    yg, xg = np.mgrid[0:SHAPE[0], 0:SHAPE[1]] / (SHAPE[0] - 1.0) - 0.5
    grey *= 1.0 - 0.08 * (xg ** 2 + yg ** 2) / 0.5
    grey += 0.010 * rng.standard_normal(SHAPE)
    return np.clip(grey, 0.0, 1.0)


def render_crack(base, path, x_tip, open_px, seed=0):
    """Dark crack opening tapering to zero at the tip."""
    img = base.copy()
    ny, nx = SHAPE
    rng = np.random.default_rng(seed)
    rough = 0.6 * smooth_noise(rng, (1, nx), 2.0).ravel()
    yy = np.arange(ny)[:, None]
    xs = np.arange(nx)[None, :]
    frac = np.clip(xs / max(x_tip, 1.0), 0.0, 1.0)
    hw = (0.55 + open_px * np.sqrt(np.maximum(1.0 - frac ** 2, 0.0)) ** 0.55)
    hw = np.where(xs <= x_tip, hw, 0.0)
    d = np.abs(yy - (path[None, :] + rough[None, :]))
    crack = d <= hw
    img = np.where(crack, 0.08, img)
    flank = (d <= hw + 1.4) & ~crack & (xs <= x_tip)
    img = np.where(flank, np.clip(img * 1.18, 0, 1), img)
    return img, crack


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    lab = laguerre_grains(SHAPE, 240, seed=6)
    base = build_matrix(lab)
    path = boundary_path(lab, SHAPE[0] * 0.52)

    tips = np.array([0.22, 0.46, 0.71, 0.94]) * (SHAPE[1] - 1)
    a_m = A0_M + tips * UM_PER_PX * 1e-6
    dk = dk_of(a_m)

    grid = np.linspace(A0_M, a_m[-1], 4000)
    cycles = np.concatenate([[0.0], np.cumsum(
        np.diff(grid) / dadn(dk_of(0.5 * (grid[1:] + grid[:-1]))))])
    n_cyc = np.interp(a_m, grid, cycles)
    rate = dadn(dk)

    fig = plt.figure(figsize=(10.4, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.60], wspace=0.085,
                          hspace=0.30, left=0.030, right=0.988, top=0.885,
                          bottom=0.125)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_p = fig.add_subplot(gs[1, :])

    for i, ax in enumerate(axes):
        img, crack = render_crack(base, path, tips[i],
                                  open_px=1.1 + 3.4 * (dk[i] / dk[-1]) ** 3,
                                  seed=i)
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
        image_axes(ax, color=COLORS["rule"], lw=0.7)
        ax.set_title(f"N = {n_cyc[i] / 1e3:.1f}k cycles\n"
                     f"$\\Delta K$ = {dk[i]:.1f} MPa$\\sqrt{{m}}$",
                     fontsize=7.8, fontweight="bold", pad=2.5,
                     linespacing=1.35)
        ty = path[int(tips[i])]
        side = -1 if tips[i] > 0.72 * SHAPE[1] else 1
        ax.annotate("", xy=(tips[i] + 3 * side, ty),
                    xytext=(tips[i] + 36 * side, ty - 36),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                    color=COLORS["accent"],
                                    shrinkA=0, shrinkB=1))
        halo_text(ax, tips[i] + 38 * side, ty - 38, "tip", fontsize=6.2,
                  color="white", ha="left" if side > 0 else "right",
                  va="bottom")
        add_scale_bar(ax, 50.0 / UM_PER_PX, label="50 \u00b5m",
                      loc=(0.045, 0.045), height=0.026, fontsize=6.2)

    # ---- Paris curve ------------------------------------------------------
    dkg = np.linspace(DK_TH * 1.015, K_C * 0.97, 600)
    rg = dadn(dkg)
    ax_p.loglog(dkg, rg * 1e3, color=CATEGORICAL[1], lw=1.8, zorder=4)
    k1, k2 = 1.65 * DK_TH, 0.62 * K_C
    tr = ax_p.get_xaxis_transform()
    for x0, x1, fc, name in [(dkg[0], k1, COLORS["card_alt"], "I  threshold"),
                             (k1, k2, "white", "II  Paris regime"),
                             (k2, dkg[-1], COLORS["card_alt"],
                              "III  unstable")]:
        ax_p.axvspan(x0, x1, facecolor=fc, edgecolor="none", zorder=0)
        ax_p.text(np.sqrt(x0 * x1), 0.955, name, transform=tr, ha="center",
                  va="top", fontsize=6.8, color=COLORS["sub"])
    for xk in (k1, k2):
        ax_p.axvline(xk, color=COLORS["grid"], lw=0.7, zorder=1)

    band = (dkg > k1) & (dkg < k2)
    slope = np.polyfit(np.log10(dkg[band]), np.log10(rg[band]), 1)[0]
    ax_p.plot(dk, rate * 1e3, marker="D", ms=5.0, ls="none",
              color=COLORS["accent"], zorder=6)
    ax_p.text(0.045, 0.64, "NASGRO-type growth law",
              transform=ax_p.transAxes, fontsize=7.0, color=CATEGORICAL[1],
              ha="left", va="center")
    ax_p.text(0.045, 0.50, "\u25c6  imaged states (a)\u2013(d)",
              transform=ax_p.transAxes, fontsize=7.0,
              color=COLORS["accent"], ha="left", va="center")
    for i in range(4):
        ax_p.annotate(f"({chr(97 + i)})", xy=(dk[i], rate[i] * 1e3),
                      xytext=(3.0, -9.0), textcoords="offset points",
                      fontsize=6.8, color=COLORS["accent"],
                      fontweight="bold")
    ax_p.axvline(DK_TH, color=COLORS["baseline"], lw=0.8, ls="--", zorder=2)
    ax_p.text(DK_TH * 1.03, 0.045, f"$\\Delta K_{{th}}$ = {DK_TH:.1f}",
              transform=tr, fontsize=6.8, color=COLORS["sub"], ha="left",
              va="bottom")
    ax_p.axvline(K_C, color=COLORS["baseline"], lw=0.8, ls="--", zorder=2)
    ax_p.text(K_C * 0.97, 0.30, f"$K_{{c}}$ = {K_C:.0f}", transform=tr,
              fontsize=6.8, color=COLORS["sub"], ha="right", va="center")

    ax_p.set_xlabel(r"stress-intensity range $\Delta K$ (MPa$\sqrt{m}$)",
                    fontsize=8.0)
    ax_p.set_ylabel("da/dN (mm/cycle)", fontsize=8.0)
    ax_p.set_xlim(4.6, 27.0)
    ax_p.set_ylim(rg.min() * 1e3 * 0.45, rg.max() * 1e3 * 3.2)
    ax_p.tick_params(labelsize=7.0, length=2.5, which="both")
    ax_p.grid(which="major", linewidth=0.5, alpha=0.7)
    ax_p.set_axisbelow(True)
    style_axes(ax_p)
    ax_p.text(0.985, 0.055,
              f"apparent Paris exponent  m = {slope:.2f}\n"
              f"fitted over {k1:.1f}\u2013{k2:.1f} MPa$\\sqrt{{m}}$",
              transform=ax_p.transAxes, fontsize=6.8, color=COLORS["sub"],
              ha="right", va="bottom", linespacing=1.35)

    panel_labels_inches(fig, axes + [ax_p], dx=-0.20, dy=0.07)
    last = 100.0 * (n_cyc[-1] - n_cyc[-2]) / n_cyc[-1]

    savefig(fig, "389_mat_crack_propagation.png")


if __name__ == "__main__":
    main()
