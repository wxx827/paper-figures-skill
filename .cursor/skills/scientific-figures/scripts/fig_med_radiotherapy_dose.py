"""Figure 379 - Radiotherapy plan: isodose wash on CT beside its DVH.

The main panel is the spatial read (where the dose goes), the subordinate panel
is the quantitative one (how much of each structure receives it).  Both come
from the same dose array and the same structure masks, so every constraint
quoted on the figure is measured, not asserted.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_palette, darken, COLORS
from figkit import image_axes, axes_inches, panel_labels_inches, halo_text

N = 224
MM_PX = 220.0 / N
RX_DOSE = 60.0                      # prescription dose, Gy

# isodose levels as a fraction of the prescription
FRACTIONS = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 1.05]
LEVELS = [f * RX_DOSE for f in FRACTIONS]
# a single warm sequential ramp: low dose barely tints the CT, high dose is
# the most saturated thing on the panel
ISO_COLORS = get_palette("cb_sequential_warm", 6)

STRUCTS = [
    ("PTV", "#b23a48", 2.0, (0.26, -0.18, 0.170, 0.150, -12)),
    ("Brainstem", "#3b5b92", 1.3, (0.00, 0.44, 0.105, 0.115, 0)),
    ("Optic chiasm", "#c9a227", 1.3, (0.00, 0.10, 0.095, 0.038, 0)),
    ("L hippocampus", "#2e7d5b", 1.3, (-0.24, 0.17, 0.115, 0.045, -22)),
]
BEAM_ANGLES = [18.0, 78.0, 200.0, 262.0, 320.0]


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


def planning_ct(seed=6):
    rng = np.random.default_rng(seed)
    x, y = _grid(N)
    wob = 1.0 + 0.014 * _sym(_sm(rng, N, 9))
    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob

    img = np.full((N, N), 0.03)
    img = np.where(rr <= 1.000, 0.42, img)
    img = np.where(rr <= 0.962, 0.95, img)                    # bone is bright
    img = np.where(rr <= 0.900, 0.30, img)
    brain = rr <= 0.875
    img = np.where(brain, 0.46 + 0.025 * _sym(_sm(rng, N, 3)), img)
    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    img = np.where(brain & (rw <= 1.0), 0.41 + 0.020 * _sm(rng, N, 5), img)
    g1, g2 = _sym(_sm(rng, N, 7.0)), _sym(_sm(rng, N, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.62) & brain & (rr > 0.58)
    img = np.where(sulci, 0.33, img)
    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13)]:
        img = np.where(_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0, 0.28, img)
    img = gaussian_filter(img, 0.9) + 0.008 * rng.standard_normal((N, N))
    return np.clip(img, 0, 1), rr <= 1.0, brain


def structure_masks():
    x, y = _grid(N)
    return {name: _rad(x, y, *geom) <= 1.0 for name, _, _, geom in STRUCTS}


def plan_dose(head, ptv):
    """Five coplanar beams focused on the PTV centroid, in Gy."""
    x, y = _grid(N)
    cx, cy = STRUCTS[0][3][0], STRUCTS[0][3][1]
    total = np.zeros((N, N))
    for ang in BEAM_ANGLES:
        t = np.deg2rad(ang)
        s = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)      # off-axis
        d = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)       # along the beam
        field = np.exp(-(np.abs(s) / 0.185) ** 6)             # flat-topped
        atten = np.exp(-0.55 * (d + 1.35))                    # depth falloff
        total += field * atten
    total = gaussian_filter(total, 3.0) * head
    dose = RX_DOSE * total / np.mean(total[ptv])
    return gaussian_filter(dose, 1.5)


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def main():
    set_pub_style(base_font=8)
    ct, head, brain = planning_ct()
    masks = structure_masks()
    dose = plan_dose(head, masks["PTV"])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.45, 0.95, 4.90, 4.90))
    image_axes(ax, color="#d5d5d5", lw=0.6)
    ax.imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

    cf = ax.contourf(dose, levels=LEVELS, colors=ISO_COLORS, alpha=0.40,
                     zorder=2)
    rasterize(cf)
    cf.set_edgecolor("face")
    cl = ax.contour(dose, levels=LEVELS[:-1],
                    colors=[darken(c, 0.30) for c in ISO_COLORS],
                    linewidths=0.95, zorder=3)
    rasterize(cl)

    for name, colour, lw, _ in STRUCTS:
        ax.contour(masks[name].astype(float), levels=[0.5], colors=[colour],
                   linewidths=lw, zorder=5)
    for name, colour, _, geom in STRUCTS:
        px = geom[0] * N / 2 + (N - 1) / 2
        py = geom[1] * N / 2 + (N - 1) / 2 + geom[3] * N / 2 + 8
        halo_text(ax, px, py, name, ha="center", va="top", fontsize=6.8,
                  color=colour, halo="white", halo_lw=1.9, zorder=8)
    ax.set_title("Planning CT with the isodose wash, five coplanar beams",
                 fontsize=9.0, fontweight="bold", pad=4.0)

    cax = ax.inset_axes([0.09, -0.090, 0.82, 0.028])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal",
                      ticks=LEVELS[:-1])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")
    cb.set_label("absorbed dose (Gy)", fontsize=7.0, color=COLORS["ink"])
    cb.ax.set_xticklabels([f"{v:.0f}" for v in LEVELS[:-1]])
    for f, v in zip(FRACTIONS[:-1], LEVELS[:-1]):
        cb.ax.text(v, 1.45, f"{f:.0%}", ha="center", va="bottom",
                   fontsize=6.0, color=COLORS["muted"])

    # -- subordinate panel: the same dose, as a cumulative DVH ---------------
    axd = axes_inches(fig, (6.55, 1.35, 3.40, 3.05))
    grid = np.linspace(0.0, 1.20 * RX_DOSE, 260)
    stats = []
    for name, colour, _, _ in STRUCTS:
        vals = dose[masks[name]]
        curve = [(vals >= g).mean() * 100.0 for g in grid]
        axd.plot(grid, curve, color=colour, lw=1.7 if name == "PTV" else 1.2,
                 label=name)
        stats.append((name, colour, np.percentile(vals, 5), vals.max(),
                      vals.mean()))

    d95 = stats[0][2]
    axd.plot([d95], [95.0], marker="o", ms=4.6, mfc="white", mec="#b23a48",
             mew=1.4, zorder=6)
    axd.annotate(f"PTV D95 = {d95:.1f} Gy", xy=(d95, 95.0), xytext=(25.0, 46.0),
                 fontsize=6.8, color="#b23a48", ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7,
                                 color=COLORS["baseline"]))
    axd.axvline(RX_DOSE, color=COLORS["baseline"], lw=0.8, ls="--", zorder=1)
    axd.text(RX_DOSE + 1.6, 56.0, f"prescription {RX_DOSE:.0f} Gy",
             fontsize=6.5, color=COLORS["sub"], ha="left", va="center",
             rotation=90)

    axd.set_xlim(0, 1.20 * RX_DOSE)
    axd.set_ylim(0, 104)
    axd.set_xlabel("dose (Gy)", fontsize=8.0)
    axd.set_ylabel("volume receiving at least that dose (%)", fontsize=8.0)
    axd.tick_params(labelsize=7.0)
    axd.grid(True, lw=0.5, alpha=0.7)
    axd.legend(loc="lower left", bbox_to_anchor=(0.32, 0.02), frameon=False,
               fontsize=7.0)
    axd.set_title("Dose-volume histogram", fontsize=9.0, fontweight="bold",
                  pad=4.0)

    rows = "\n".join(f"{n:<14s} {mx:5.1f}      {mn:5.1f}"
                     for n, _, _, mx, mn in stats)

    for a in (ax, axd):
        a.apply_aspect()
    panel_labels_inches(fig, [ax, axd], dx=-0.22, dy=0.14)


    savefig(fig, "379_med_radiotherapy_dose.png")


if __name__ == "__main__":
    main()
