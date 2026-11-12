"""Uniform random sampling against Poisson-disc sampling, and their spectra.

Main panel: two point sets of the same count generated over the same frame,
one by independent uniform draws and one by Bridson's Poisson-disc algorithm.
The frame shows the left half of the first and the right half of the second so
the eye can compare clumping directly; the exclusion discs of the Poisson set
are drawn at half the minimum separation, and the largest empty disc that fits
inside each half is found by a distance search and circled.

Auxiliary (both at 58 % of the main panel width): the radially averaged power
spectrum of each full set, computed by binning the samples onto a lattice and
taking a two-dimensional FFT, with the two-dimensional spectra inset; and the
nearest-neighbour distance distributions, which is where the exclusion radius
of the Poisson set shows up as a hard left edge.

Illustrative sampling, no physical units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

from pubstyle import set_pub_style, savefig, style_axes, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
BLUE = "#3b5b92"
DEEP = "#134a76"
RUST = "#c0603f"
ACCENT = "#b23a48"
TEAL = "#2e7d5b"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 1.02, 5.54, 4.44)
AUX_T_RECT = (7.62, 3.56, 3.16, 1.90)
AUX_B_RECT = (7.62, 1.02, 3.16, 1.90)
ASPECT = MAIN_RECT[2] / MAIN_RECT[3]

SEED = 33127
HALF_Y = 2.30
HALF_X = HALF_Y * ASPECT
RADIUS = 0.185
GRID_N = 240


def poisson_disc(rng, w, h, r, k=30):
    """Bridson's algorithm on a background grid of cell size r/sqrt(2)."""
    cell = r / np.sqrt(2.0)
    nx, ny = int(np.ceil(w / cell)), int(np.ceil(h / cell))
    grid = -np.ones((ny, nx), dtype=int)
    pts = []
    active = []

    def insert(p):
        pts.append(p)
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        grid[min(gy, ny - 1), min(gx, nx - 1)] = len(pts) - 1
        active.append(len(pts) - 1)

    def ok(p):
        if not (0.0 <= p[0] < w and 0.0 <= p[1] < h):
            return False
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        for j in range(max(gy - 2, 0), min(gy + 3, ny)):
            for i in range(max(gx - 2, 0), min(gx + 3, nx)):
                q = grid[j, i]
                if q >= 0 and np.hypot(*(pts[q] - p)) < r:
                    return False
        return True

    insert(np.array([rng.uniform(0, w), rng.uniform(0, h)]))
    while active:
        a = int(rng.integers(0, len(active)))
        base = pts[active[a]]
        placed = False
        for _ in range(k):
            ang = rng.uniform(0.0, 2 * np.pi)
            rad = r * np.sqrt(rng.uniform(1.0, 4.0))
            cand = base + rad * np.array([np.cos(ang), np.sin(ang)])
            if ok(cand):
                insert(cand)
                placed = True
                break
        if not placed:
            active.pop(a)
    return np.array(pts)


def radial_spectrum(pts, w, h, n=GRID_N):
    """Radially averaged power spectrum of a point set, white noise = 1."""
    ny = int(round(n * h / w))
    hist, xe, ye = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=(n, ny),
        range=[[0.0, w], [0.0, h]])
    f = np.fft.fftshift(np.fft.fft2(hist - hist.mean()))
    power = (np.abs(f) ** 2) / len(pts)
    kx = np.fft.fftshift(np.fft.fftfreq(n, d=w / n))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=h / ny))
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kr = np.hypot(kxx, kyy)
    k_max = min(kx.max(), ky.max())
    bins = np.linspace(0.0, k_max, 120)
    which = np.digitize(kr.ravel(), bins) - 1
    prof = np.array([power.ravel()[which == b].mean() if (which == b).any()
                     else np.nan for b in range(len(bins) - 1)])
    ctr = 0.5 * (bins[1:] + bins[:-1])
    return ctr, prof, power, (kx, ky), k_max


def crop_spectrum(power, kx, ky, k_half):
    """Square window of the 2-D spectrum in frequency, ready for imshow."""
    ix = np.abs(kx) <= k_half
    iy = np.abs(ky) <= k_half
    return power[np.ix_(ix, iy)].T


def largest_gap(pts, x0, x1, y0, y1, n=340):
    """Radius and centre of the largest empty disc inside a sub-rectangle."""
    gx = np.linspace(x0, x1, n)
    gy = np.linspace(y0, y1, int(n * (y1 - y0) / (x1 - x0)))
    xx, yy = np.meshgrid(gx, gy)
    d, _ = cKDTree(pts).query(np.column_stack([xx.ravel(), yy.ravel()]))
    i = int(np.argmax(d))
    return float(d[i]), np.array([xx.ravel()[i], yy.ravel()[i]])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    w, h = 2 * HALF_X, 2 * HALF_Y
    blue = poisson_disc(rng, w, h, RADIUS)
    n_pts = len(blue)
    white = np.column_stack([rng.uniform(0.0, w, n_pts),
                             rng.uniform(0.0, h, n_pts)])
    blue_d = blue - np.array([HALF_X, HALF_Y])
    white_d = white - np.array([HALF_X, HALF_Y])

    nn_w = cKDTree(white).query(white, k=2)[0][:, 1]
    nn_b = cKDTree(blue).query(blue, k=2)[0][:, 1]

    gap_w, cw = largest_gap(white_d, -HALF_X, 0.0, -HALF_Y, HALF_Y)
    gap_b, cb = largest_gap(blue_d, 0.0, HALF_X, -HALF_Y, HALF_Y)

    kw, pw, pow_w, kaxis, kmax = radial_spectrum(white, w, h)
    kb, pb, pow_b, _, _ = radial_spectrum(blue, w, h)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the two sets, half a frame each -----------------------------
    ax.set_aspect("equal")
    left = white_d[white_d[:, 0] < 0.0]
    right = blue_d[blue_d[:, 0] >= 0.0]

    ax.add_collection(PatchCollection(
        [Circle(tuple(p), RADIUS / 2.0) for p in right],
        facecolor=lighten(BLUE, 0.80), edgecolor=lighten(BLUE, 0.55),
        linewidths=0.4, zorder=2))
    ax.plot(left[:, 0], left[:, 1], ls="none", marker="o", ms=2.7, mfc=RUST,
            mec="none", zorder=4)
    ax.plot(right[:, 0], right[:, 1], ls="none", marker="o", ms=2.7, mfc=DEEP,
            mec="none", zorder=4)

    for ctr, rad in ((cw, gap_w), (cb, gap_b)):
        ax.add_patch(Circle(tuple(ctr), rad, facecolor="none",
                            edgecolor=ACCENT, lw=1.2, ls=(0, (4.0, 2.2)),
                            zorder=6))
        below = ctr[1] > 0.0
        ty = ctr[1] + (-1.0 if below else 1.0) * (rad + 0.07)
        tx = float(np.clip(ctr[0], -HALF_X + 0.62, HALF_X - 0.62))
        ax.text(tx, ty, f"largest empty disc\n{rad:.3f} units", ha="center",
                va="top" if below else "bottom", fontsize=6.2, color=ACCENT,
                linespacing=1.35, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                          pad=1.2))

    ax.axvline(0.0, color=INK, lw=1.0, zorder=8)
    ax.text(-HALF_X * 0.5, HALF_Y - 0.10,
            f"uniform random draws\n{len(left)} points in this half",
            ha="center", va="top", fontsize=7.4, color=RUST, zorder=9,
            linespacing=1.4, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      pad=2.0))
    ax.text(HALF_X * 0.5, HALF_Y - 0.10,
            f"Poisson-disc sampling\n{len(right)} points in this half",
            ha="center", va="top", fontsize=7.4, color=DEEP, zorder=9,
            linespacing=1.4, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      pad=2.0))

    ax.set_xlim(-HALF_X, HALF_X)
    ax.set_ylim(-HALF_Y, HALF_Y)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("x (dimensionless length units)", fontsize=8.5)
    ax.set_ylabel("y (dimensionless length units)", fontsize=8.5)
    ax.set_title(f"Same density, different neighbourhood structure: "
                 f"{n_pts} samples per set", fontsize=9.0, fontweight="bold",
                 pad=5.0)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc=RUST, mec="none",
               label=f"uniform: nearest neighbour {nn_w.mean():.3f} units "
                     f"on average"),
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc=DEEP, mec="none",
               label=f"Poisson disc: {nn_b.mean():.3f} units, none closer "
                     f"than {nn_b.min():.3f}"),
        Line2D([], [], color=lighten(BLUE, 0.55), lw=1.0,
               label=f"exclusion discs, radius {RADIUS / 2:.3f} units "
                     f"(never overlapping)"),
        Line2D([], [], color=ACCENT, lw=1.2, ls=(0, (4.0, 2.2)),
               label="largest empty disc inside each half"),
    ]
    leg = ax.legend(handles=handles, loc="lower center", frameon=True,
                    fontsize=6.5, handlelength=1.6, labelspacing=0.30,
                    borderpad=0.30, framealpha=0.94, edgecolor="#d5d5d5",
                    facecolor="white", bbox_to_anchor=(0.50, 0.004))
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(20)

    # ---- aux (b): radially averaged power spectrum -------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_t.axhline(1.0, color=FAINT, lw=0.8, ls=(0, (3.0, 2.2)), zorder=2)
    ax_t.plot(kw, pw, color=RUST, lw=1.2, zorder=4, label="uniform random")
    ax_t.plot(kb, pb, color=DEEP, lw=1.4, zorder=5, label="Poisson disc")
    k_ring = float(kb[np.nanargmax(pb)])
    k_view = 11.0
    ax_t.axvline(k_ring, color=ACCENT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    ax_t.text(k_ring - 0.18, 2.44, f"ring peak at {k_ring:.2f}\ncycles/unit",
              ha="right", va="top", fontsize=6.0, color=ACCENT,
              linespacing=1.35)
    ax_t.set_xlim(0.0, k_view)
    ax_t.set_ylim(0.0, 2.55)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("radial frequency (cycles per length unit)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("power (white noise = 1)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Blue noise is quiet at low frequency", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.legend(loc="lower right", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.24, borderpad=0.10,
                bbox_to_anchor=(1.012, -0.015))

    cmap = get_cmap("cb_sequential_blue")
    k_half = 0.34 * kmax
    box_w = 0.616 / AUX_T_RECT[2]
    box_h = 0.616 / AUX_T_RECT[3]
    for n, (p, lab) in enumerate(((pow_w, "uniform"), (pow_b, "Poisson"))):
        sub = crop_spectrum(p, kaxis[0], kaxis[1], k_half)
        axi = ax_t.inset_axes([0.545 + (box_w + 0.020) * n,
                               0.975 - box_h, box_w, box_h])
        axi.imshow(np.log10(sub + 1e-3), cmap=cmap, origin="lower",
                   extent=(-k_half, k_half, -k_half, k_half), aspect="equal",
                   interpolation="nearest", vmin=-1.4, vmax=0.9)
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_linewidth(0.6)
            s.set_color("#8a8a8a")
        axi.set_title(lab, fontsize=6.0, color=SUB, pad=1.6)
    ax_t.text(0.995, 0.975 - box_h - 0.035,
              f"2-D power, log scale, |k| < {k_half:.0f} cycles/unit",
              transform=ax_t.transAxes, ha="right", va="top", fontsize=6,
              color=MUTED)

    # ---- aux (c): nearest neighbour distances ------------------------------
    ax_b.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    bins = np.linspace(0.0, max(nn_w.max(), nn_b.max()) * 1.02, 42)
    ax_b.hist(nn_w, bins=bins, color=lighten(RUST, 0.55), edgecolor="white",
              linewidth=0.5, zorder=3,
              label=f"uniform, coefficient of variation "
                    f"{nn_w.std() / nn_w.mean():.2f}")
    ax_b.hist(nn_b, bins=bins, histtype="step", color=DEEP, linewidth=1.5,
              zorder=5,
              label=f"Poisson disc, coefficient of variation "
                    f"{nn_b.std() / nn_b.mean():.2f}")
    ax_b.axvline(RADIUS, color=ACCENT, lw=1.1, zorder=6)
    ax_b.text(RADIUS - 0.008, 0.62, f"exclusion radius\n{RADIUS:.3f} units",
              transform=ax_b.get_xaxis_transform(), ha="right", va="top",
              fontsize=6.0, color=ACCENT, linespacing=1.35)
    ax_b.set_xlim(0.0, bins[-1])
    ax_b.set_ylim(0.0, float(np.histogram(nn_b, bins=bins)[0].max()) * 1.34)
    ax_b.tick_params(labelsize=6.6)
    ax_b.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax_b.set_xlabel("distance to the nearest sample (length units)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel(f"samples of {n_pts} (count)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("The exclusion radius is a hard floor", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.24, borderpad=0.10,
                bbox_to_anchor=(-0.012, 1.045))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "977_cgx_bluenoise_sampling.png")


if __name__ == "__main__":
    main()
