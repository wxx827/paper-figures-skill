"""Figure 397 - Leaf-level disease severity grading across three fields.

One synthetic nadir canopy patch per field, read left to right: the RGB tile,
the lesion segmentation, the per-leaf severity grade on the standard five-step
scale, and the grade histogram.  The red magnifier box sits on the worst leaf
of each row and is repeated at exactly the same pixel coordinates in every
image column, so the three views of that leaf can be compared directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig
from figkit import add_zoom_inset, roi_rect, corner_stamp

SEED = 7719
N = 200                     # tile side, px
MMPP = 1.4                  # mm per pixel

# The five-step severity scale, healthy -> destroyed.  Same ramp family as the
# yield map in 394: green for good, gold for marginal, red for bad.
GRADES = [
    ("0  healthy", 0.000, "#2e7d5b"),
    ("1  trace  (<5%)", 0.005, "#a8c07a"),
    ("2  mild  (5-15%)", 0.050, "#c9a227"),
    ("3  moderate  (15-35%)", 0.150, "#c0603f"),
    ("4  severe  (>35%)", 0.350, "#8c2f2f"),
]
CUTS = [g[1] for g in GRADES][1:]
GRADE_COLORS = [g[2] for g in GRADES]
SOIL_GREY = "#e4e0d6"
LEAF_GREY = "#c4c9bd"
LESION_RED = "#b23a48"
ZOOM_C = "#b23a48"

FIELDS = [("Field A", "low pressure", 0.10),
          ("Field B", "moderate pressure", 0.45),
          ("Field C", "high pressure", 1.45)]
COLS = ["RGB canopy tile", "Lesion segmentation", "Per-leaf severity grade",
        "Grade distribution"]


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def make_tile(rng, pressure, n=N, n_leaf=54):
    """Nadir canopy patch -> (rgb, leaf id map, lesion mask, per-leaf stats)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    tex = _noise(rng, n, 2.2)
    rgb = np.stack([0.46 + 0.085 * tex, 0.38 + 0.075 * tex,
                    0.29 + 0.060 * tex], axis=-1)

    leaf = np.zeros((n, n), int)
    lesion = np.zeros((n, n), bool)
    stats = []

    for _ in range(n_leaf):
        cx, cy = rng.uniform(-10, n + 10), rng.uniform(-10, n + 10)
        ang = rng.uniform(0.0, np.pi)
        a, b = rng.uniform(19.0, 32.0), rng.uniform(5.2, 8.8)
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        m = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        area = int(m.sum())
        if area < 110:
            continue
        lid = len(stats) + 1
        leaf[m] = lid

        base = np.array([0.145 + 0.085 * rng.random(),
                         0.360 + 0.185 * rng.random(),
                         0.105 + 0.075 * rng.random()])
        shade = (1.0 - 0.34 * (v / b) ** 2 + 0.06 * _noise(rng, n, 1.4))
        rib = np.abs(v) < 0.95
        col = base[None, None, :] * shade[..., None]
        col = np.where(rib[..., None], col * 1.30, col)
        rgb[m] = np.clip(col[m], 0.0, 1.0)

        # Lesions: necrotic core inside a chlorotic halo.
        idx = np.argwhere(m)
        n_les = rng.poisson(pressure * area / 190.0)
        for _ in range(int(n_les)):
            ly, lx = idx[rng.integers(idx.shape[0])]
            r = rng.uniform(1.9, 4.1)
            d = np.hypot(xx - lx, yy - ly) * (1.0 + 0.30 * _noise(rng, n, 1.1))
            halo = m & (d < r)
            core = m & (d < 0.60 * r)
            rgb[halo] = np.array([0.66, 0.60, 0.27]) * (
                0.86 + 0.22 * rng.random())
            rgb[core] = np.array([0.40, 0.27, 0.15]) * (
                0.86 + 0.26 * rng.random())
            lesion |= m & (d < 0.82 * r)

        frac = float((lesion & m).sum()) / area
        stats.append(dict(lid=lid, area=area, frac=frac,
                          cy=float(idx[:, 0].mean()),
                          cx=float(idx[:, 1].mean())))

    vig = 1.0 - 0.10 * (((xx / (n - 1) - 0.5) ** 2
                         + (yy / (n - 1) - 0.5) ** 2) / 0.5)
    rgb = np.clip(rgb * vig[..., None] + 0.012 * rng.standard_normal(
        (n, n, 3)), 0.0, 1.0)
    return rgb, leaf, lesion, stats


def grade_of(frac):
    return int(np.digitize(frac, CUTS))


def severity_image(leaf, stats, n=N):
    """Per-leaf grade painted as an RGB tile with hairline leaf outlines."""
    from matplotlib.colors import to_rgb
    img = np.tile(np.array(to_rgb(SOIL_GREY)), (n, n, 1))
    for s in stats:
        m = leaf == s["lid"]
        img[m] = to_rgb(GRADE_COLORS[grade_of(s["frac"])])
    edge = np.zeros((n, n), bool)
    edge[:, :-1] |= leaf[:, :-1] != leaf[:, 1:]
    edge[:-1, :] |= leaf[:-1, :] != leaf[1:, :]
    img[edge] *= 0.74
    return img


def mask_image(leaf, lesion, n=N):
    from matplotlib.colors import to_rgb
    img = np.tile(np.array(to_rgb(SOIL_GREY)), (n, n, 1))
    img[leaf > 0] = to_rgb(LEAF_GREY)
    img[lesion] = to_rgb(LESION_RED)
    return img


def bare(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#8a8a8a")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    tiles = []
    for _, _, pressure in FIELDS:
        rgb, leaf, lesion, stats = make_tile(rng, pressure)
        tiles.append((rgb, leaf, lesion, stats))

    # Shared x range for the three histogram panels (rule 13.1).
    all_pct = []
    for _, _, _, stats in tiles:
        g = np.array([grade_of(s["frac"]) for s in stats])
        all_pct.append(np.array([100.0 * (g == k).mean() for k in range(5)]))
    x_hi = max(p.max() for p in all_pct) * 1.22

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 0.92],
                          left=0.052, right=0.988, top=0.878, bottom=0.130,
                          wspace=0.10, hspace=0.075)

    for r, ((rgb, leaf, lesion, stats), (fname, fnote, _)) in enumerate(
            zip(tiles, FIELDS)):
        seg = mask_image(leaf, lesion)
        sev = severity_image(leaf, stats)
        grades = np.array([grade_of(s["frac"]) for s in stats])
        areas = np.array([s["area"] for s in stats], float)

        # Magnifier locked on the worst leaf of this row.
        worst = max(stats, key=lambda s: (s["frac"], s["area"]))
        half = 26
        bx = int(np.clip(worst["cx"] - half, 0, N - 2 * half))
        by = int(np.clip(worst["cy"] - half, 0, N - 2 * half))
        box = (bx, by, 2 * half, 2 * half)

        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(rgb, interpolation="bilinear")
        bare(ax0)
        add_zoom_inset(ax0, rgb, box, (0.605, 0.020, 0.375, 0.375),
                       color=ZOOM_C, interpolation="bilinear")
        corner_stamp(ax0, f"{len(stats)} leaves segmented", loc="lower left",
                     fontsize=6.2)

        ax1 = fig.add_subplot(gs[r, 1])
        ax1.imshow(seg, interpolation="nearest")
        bare(ax1)
        roi_rect(ax1, box, color=ZOOM_C, lw=1.15)
        les_pct = 100.0 * lesion.sum() / max((leaf > 0).sum(), 1)
        corner_stamp(ax1, f"lesion area {les_pct:.1f}% of leaf",
                     loc="lower left", fontsize=6.2, color="#333333",
                     halo="white")

        ax2 = fig.add_subplot(gs[r, 2])
        ax2.imshow(sev, interpolation="nearest")
        bare(ax2)
        add_zoom_inset(ax2, sev, box, (0.605, 0.020, 0.375, 0.375),
                       color=ZOOM_C, interpolation="nearest")
        corner_stamp(ax2, f"worst leaf grade {grade_of(worst['frac'])}",
                     loc="lower left", fontsize=6.2)

        # ---- grade distribution -------------------------------------------
        ax3 = fig.add_subplot(gs[r, 3])
        pct = all_pct[r]
        ax3.barh(np.arange(5), pct, height=0.66, color=GRADE_COLORS,
                 edgecolor="white", linewidth=0.6, zorder=3)
        for k, p in enumerate(pct):
            ax3.text(p + x_hi * 0.022, k, f"{p:.0f}%", va="center",
                     ha="left", fontsize=6.4, color="#333333")
        ax3.set_yticks(np.arange(5))
        ax3.set_yticklabels(["0", "1", "2", "3", "4"], fontsize=7.0)
        # Inverted, with an empty band above grade 0 to seat the summary note.
        ax3.set_ylim(4.62, -1.30)
        ax3.set_xlim(0, x_hi)
        ax3.tick_params(axis="x", labelsize=6.8)
        ax3.grid(axis="x", lw=0.5, alpha=0.5)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.set_ylabel("grade", fontsize=7.4, labelpad=1.5)
        if r == 2:
            ax3.set_xlabel("share of leaves (%)", fontsize=8.0)

        dsi = 100.0 * float((grades * areas).sum() / (4.0 * areas.sum()))
        inc = 100.0 * float((grades > 0).mean())
        ax3.text(0.975, 0.985,
                 f"incidence {inc:.0f}%\nseverity index {dsi:.1f}%",
                 transform=ax3.transAxes, ha="right", va="top",
                 fontsize=6.6, color="#555555", linespacing=1.45)

        bb = ax0.get_position()
        fig.text(bb.x0 - 0.012, bb.y0 + bb.height / 2.0,
                 f"{fname}\n{fnote}", rotation=90, ha="right", va="center",
                 fontsize=7.6, color="#333333", linespacing=1.5)

        if r == 0:
            for c, name in enumerate(COLS):
                axc = fig.add_subplot(gs[0, c])
                axc.set_axis_off()
                pb = axc.get_position()
                axc.remove()

    handles = [Patch(facecolor=c, edgecolor="white", label=l)
               for l, _, c in GRADES]
    handles += [Patch(facecolor=LESION_RED, edgecolor="white",
                      label="segmented lesion"),
                Patch(facecolor=LEAF_GREY, edgecolor="white",
                      label="healthy leaf area"),
                Patch(facecolor="none", edgecolor=ZOOM_C, linewidth=1.2,
                      label="magnifier on worst leaf")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.028), ncol=8, frameon=False,
               fontsize=7.2, columnspacing=1.3, handlelength=1.3)


    savefig(fig, "397_agr_disease_severity.png")


if __name__ == "__main__":
    main()
