"""Figure 374 - CT display windows: one HU volume slice, five renderings.

The whole point of the layout is that the five tiles are the *same* float array
of Hounsfield units; only ``vmin`` / ``vmax`` change.  The subordinate strip
underneath shows the slice's HU histogram with the five window spans drawn on
the same axis, so the reader can see which tissues each window throws away.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, CATEGORICAL, COLORS
from figkit import image_axes, corner_stamp, add_scale_bar

N = 256
FOV_MM = 240.0                       # square field of view -> mm per pixel


# ---------------------------------------------------------------------------
# Synthetic head phantom (layered ellipses + gyral folding, as in figure 253)
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _sym(a):
    return 0.5 * (a + a[:, ::-1])


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def _rad(x, y, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.hypot(xr / rx, yr / ry)


# tissue code -> (mean HU, HU spread)
TISSUE_HU = {
    0: (-1000.0, 12.0),    # air outside the head
    1: (48.0, 9.0),        # scalp / temporalis muscle
    2: (980.0, 130.0),     # cortical + diploic skull
    3: (10.0, 5.0),        # CSF in sulci and cisterns
    4: (38.0, 5.0),        # grey matter
    5: (26.0, 4.0),        # white matter
    6: (4.0, 3.0),         # ventricular CSF
    7: (-960.0, 20.0),     # air in the frontal / mastoid sinuses
    8: (190.0, 14.0),      # contrast-filled vessels
}


def head_labels(n=N, seed=5):
    """Integer tissue map for one axial head slice."""
    rng = np.random.default_rng(seed)
    x, y = _grid(n)
    wob = 1.0 + 0.015 * _sym(_sm(rng, n, 9))

    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob
    lab = np.zeros((n, n), dtype=int)
    lab[rr <= 1.000] = 1
    lab[rr <= 0.962] = 2
    lab[rr <= 0.900] = 3
    brain = rr <= 0.875
    lab[brain] = 4

    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    lab[brain & (rw <= 1.0)] = 5

    g1, g2 = _sym(_sm(rng, n, 7.0)), _sym(_sm(rng, n, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.42) & brain & (rr > 0.50)
    lab[sulci] = 3

    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13),
                                 (0.0, 0.10, 0.030, 0.10, 0)]:
        lab[_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0] = 6
    lab[(np.abs(x) < 0.010) & brain] = 3

    for sx in (-1, 1):                                  # paranasal air cells
        lab[_rad(x, y, sx * 0.20, -0.66, 0.13, 0.10, sx * 12) <= 1.0] = 7
    for sx in (-1, 1):                                  # mastoid air cells
        lab[_rad(x, y, sx * 0.62, 0.30, 0.10, 0.09, sx * 24) <= 1.0] = 7

    for vx, vy, vrx, vry, va in [(0.0, -0.50, 0.020, 0.020, 0),
                                 (0.34, -0.14, 0.014, 0.075, 26),
                                 (-0.34, -0.14, 0.014, 0.075, -26),
                                 (0.0, 0.60, 0.026, 0.018, 0)]:
        lab[_rad(x, y, vx, vy, vrx, vry, va) <= 1.0] = 8
    return lab, rng


def head_ct(n=N, seed=5):
    """Axial head slice in Hounsfield units - the single source array."""
    lab, rng = head_labels(n, seed)
    hu = np.zeros((n, n), dtype=float)
    for code, (mu, sd) in TISSUE_HU.items():
        m = lab == code
        if m.any():
            hu[m] = mu + sd * _sm(rng, n, 2.0)[m]
    # diploe: the skull is not a uniform slab
    hu = np.where(lab == 2, hu - 260.0 * np.clip(_sm(rng, n, 1.6), 0, None), hu)
    hu = gaussian_filter(hu, 1.0)
    return hu + 3.0 * rng.standard_normal((n, n))


# ---------------------------------------------------------------------------
# Display windows (name, window width, window level) - clinical presets
# ---------------------------------------------------------------------------
WINDOWS = [
    ("Brain", 80.0, 40.0),
    ("Bone", 2000.0, 400.0),
    ("Lung", 1500.0, -600.0),
    ("Soft tissue", 400.0, 50.0),
    ("Angio", 600.0, 150.0),
]


def main():
    set_pub_style(base_font=8)
    hu = head_ct()
    mm_px = FOV_MM / N

    fig = plt.figure(figsize=(10.4, 4.6))
    gs = fig.add_gridspec(2, len(WINDOWS), height_ratios=[1.0, 0.55],
                          wspace=0.035, hspace=0.26,
                          left=0.030, right=0.986, top=0.880, bottom=0.115)

    for j, (name, ww, wl) in enumerate(WINDOWS):
        ax = fig.add_subplot(gs[0, j])
        image_axes(ax, color="#d5d5d5", lw=0.6)
        ax.imshow(hu, cmap="gray", vmin=wl - ww / 2, vmax=wl + ww / 2,
                  interpolation="bilinear")
        ax.set_title(f"{name} window", fontsize=8.6, fontweight="bold",
                     color=CATEGORICAL[j], pad=3.5)
        corner_stamp(ax, f"WW {ww:.0f} / WL {wl:+.0f} HU", loc="lower left",
                     fontsize=6.8)
        corner_stamp(ax, f"displayed range  {wl - ww / 2:+.0f} .. "
                         f"{wl + ww / 2:+.0f} HU", loc="upper right",
                     fontsize=6.2, color="#e0e0e0")
        if j == 0:
            add_scale_bar(ax, 50.0 / mm_px, label="50 mm", fontsize=6.2,
                          loc=(0.035, 0.115))

    # -- subordinate strip: the histogram every tile above is drawn from -----
    axh = fig.add_subplot(gs[1, :])
    lo, hi = -1450.0, 1550.0
    counts, edges = np.histogram(np.clip(hu, lo, hi), bins=300, range=(lo, hi))
    centres = 0.5 * (edges[1:] + edges[:-1])
    dens = counts.astype(float) ** 0.40
    dens /= dens.max()
    axh.fill_between(centres, 0.0, 0.50 * dens, color="#d5d5d5",
                     edgecolor=COLORS["baseline"], linewidth=0.6, zorder=2)

    for lbl, hu_at in (("air", -1000), ("brain", 34), ("skull", 830)):
        axh.text(hu_at, 0.53, lbl, fontsize=6.2, color=COLORS["muted"],
                 ha="center", va="bottom", zorder=3)

    for j, (name, ww, wl) in enumerate(WINDOWS):
        yb = 0.72 + 0.10 * j
        x0, x1 = wl - ww / 2, wl + ww / 2
        axh.plot([x0, x1], [yb, yb], lw=2.6, solid_capstyle="butt",
                 color=CATEGORICAL[j], zorder=4)
        axh.plot([wl], [yb], marker="|", ms=5.5, mew=1.0, color="white",
                 zorder=5)
        txt = f"{name}  ({ww:.0f} / {wl:+.0f} HU)"
        if x1 > 700:                       # bone: label on the free left side
            axh.text(x0 - 55, yb, txt, fontsize=6.5, va="center", ha="right",
                     color=COLORS["sub"])
        else:
            axh.text(x1 + 55, yb, txt, fontsize=6.5, va="center", ha="left",
                     color=COLORS["sub"])

    axh.set_xlim(lo, hi)
    axh.set_ylim(-0.03, 1.22)
    axh.set_yticks([])
    axh.set_xlabel("CT number (HU)", fontsize=8.0)
    axh.text(560, 0.30, "HU histogram of the slice\n(power-scaled voxel "
             "count, a.u.)", fontsize=6.5, color=COLORS["muted"], ha="left",
             va="center")
    axh.tick_params(labelsize=7.0)
    axh.set_xticks([-1000, -500, 0, 500, 1000, 1500])
    for s in ("left", "right", "top"):
        axh.spines[s].set_visible(False)


    savefig(fig, "374_med_ct_windowing.png")


if __name__ == "__main__":
    main()
