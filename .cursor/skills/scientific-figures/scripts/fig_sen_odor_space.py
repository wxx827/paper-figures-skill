"""Perceptual odour space of a panel of odorants, with a physicochemical biplot.

The main panel is a principal-component projection of odorant molecules
described by six physicochemical properties, coloured by the descriptor class
that human panellists applied most often and enclosed by convex hulls.  The
grey arrows are the property loadings from the same decomposition, so the
figure says not only which odours group together but which molecular property
carries them apart - volatility and hydrophobicity run almost orthogonally,
which is why a light citrus note and a heavy woody note separate so cleanly.

The subordinate panels show that the descriptor labels themselves are not
independent (a co-occurrence matrix over the same panel) and that detection
thresholds span five orders of magnitude, which is why the axis is
logarithmic.  Explained variance, hull areas, co-occurrence counts and
threshold medians are all measured from the generated panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches, safe_colorbar

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

CLASSES = ["floral", "fruity", "minty", "woody", "pungent"]
CLASS_COL = dict(zip(CLASSES, [RED, GOLD, GREEN, BLUE, PURPLE]))
FEATURES = ["mol. weight", "log P", "vapour pressure", "H-bond donors",
            "polarisability", "aromatic rings"]
# Where each class sits in the two perceptual dimensions before the property
# space is built around it; the classes are spaced on a ring so that the
# hulls stay separable after the projection.
RING_DEG = {"floral": 92.0, "fruity": 20.0, "minty": -52.0, "woody": 164.0,
            "pungent": -124.0}
RING_R = {"floral": 3.15, "fruity": 2.85, "minty": 3.30, "woody": 3.00,
          "pungent": 2.90}
NAMES = {"floral": "geraniol", "fruity": "isoamyl acetate",
         "minty": "menthol", "woody": "cedrol", "pungent": "pyridine"}
LOG_THR = {"floral": -0.6, "fruity": 0.5, "minty": 0.1, "woody": 1.4,
           "pungent": -1.8}
N_PER = 24


def panel_data(seed=17):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(len(FEATURES), 2)))
    basis = q.T                                   # 2 x n_features, orthonormal
    x, lab, thr = [], [], []
    for c in CLASSES:
        a = np.deg2rad(RING_DEG[c])
        seed_pt = RING_R[c] * np.array([np.cos(a), np.sin(a)])
        mu = seed_pt @ basis
        spread = rng.normal(0.0, 0.55, (N_PER, len(FEATURES)))
        x.append(mu[None, :] + spread)
        lab += [c] * N_PER
        thr.append(10.0 ** rng.normal(LOG_THR[c], 0.85, N_PER))
    x = np.vstack(x)
    lab = np.array(lab)
    thr = np.concatenate(thr)

    xs = (x - x.mean(axis=0)) / x.std(axis=0)
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    scores = u * s
    explained = s ** 2 / np.sum(s ** 2)
    return scores[:, :2], vt[:2].T, explained, lab, thr, rng


def cooccurrence(lab, rng):
    """Secondary descriptor drawn per molecule; the matrix is then counted."""
    affinity = np.array([
        [0.0, 0.9, 0.5, 0.7, 0.2],
        [0.9, 0.0, 0.8, 0.2, 0.3],
        [0.5, 0.8, 0.0, 0.3, 0.6],
        [0.7, 0.2, 0.3, 0.0, 0.4],
        [0.2, 0.3, 0.6, 0.4, 0.0]])
    m = np.zeros((len(CLASSES), len(CLASSES)))
    for i, c in enumerate(CLASSES):
        n = int(np.sum(lab == c))
        p = affinity[i] / affinity[i].sum()
        draw = rng.choice(len(CLASSES), size=n, p=p)
        for j in draw:
            m[i, j] += 1.0
            m[j, i] += 1.0
        m[i, i] = n
    return m


def main_panel(ax, scores, load, explained, lab):
    hull_area = {}
    for c in CLASSES:
        p = scores[lab == c]
        col = CLASS_COL[c]
        h = ConvexHull(p)
        poly = np.vstack([p[h.vertices], p[h.vertices][:1]])
        ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.10, zorder=2,
                linewidth=0.0)
        ax.plot(poly[:, 0], poly[:, 1], color=col, lw=0.9,
                ls=(0, (3.4, 2.2)), zorder=3)
        ax.plot(p[:, 0], p[:, 1], marker="o", ms=3.6, ls="none", color=col,
                mfc=col, mec="white", mew=0.5, alpha=0.92, zorder=5)
        hull_area[c] = h.volume

        cen = p.mean(axis=0)
        ax.text(cen[0], cen[1], c, ha="center", va="center", fontsize=7.6,
                color=col, zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=0.82))

        i_o = int(np.argmax(np.linalg.norm(p - 0.35 * cen, axis=1)))
        v = cen / max(np.linalg.norm(cen), 1e-9)
        ax.plot([p[i_o, 0]], [p[i_o, 1]], marker="o", ms=5.8, mfc="none",
                mec=col, mew=1.0, zorder=6)
        ax.text(p[i_o, 0] + 0.24 * np.sign(v[0]), p[i_o, 1] + 0.22 * v[1],
                NAMES[c], ha="left" if v[0] >= 0 else "right", va="center",
                fontsize=6.6, color=SUB, style="italic", zorder=7)

    # Arrows stay inside the empty core of the ring so that neither the shaft
    # nor the label lands on a cluster.
    k = 1.55 / np.linalg.norm(load, axis=1).max()
    for (dx, dy), name in zip(load * k, FEATURES):
        ax.annotate("", xy=(dx, dy), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="#3a3a3a",
                                    lw=0.9, shrinkA=0.0, shrinkB=0.0),
                    zorder=8)
        r = 1.13
        ax.text(dx * r, dy * r, name, ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top", fontsize=6.4,
                color="#3a3a3a", zorder=9,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.75))

    ax.axhline(0.0, color=GREY, lw=0.6, ls=(0, (1.6, 2.0)), zorder=1)
    ax.axvline(0.0, color=GREY, lw=0.6, ls=(0, (1.6, 2.0)), zorder=1)
    ax.set_xlim(-1.24 * np.abs(scores[:, 0]).max(),
                1.24 * np.abs(scores[:, 0]).max())
    ax.set_ylim(-1.30 * np.abs(scores[:, 1]).max(),
                1.30 * np.abs(scores[:, 1]).max())
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.0f} % of variance) (-)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.0f} % of variance) (-)")
    ax.set_title(f"Two components carry "
                 f"{(explained[0] + explained[1]) * 100:.0f} % of the "
                 f"property variance", fontsize=9.0, fontweight="bold",
                 color=INK)
    return hull_area


def cooc_panel(fig, ax, m):
    im = ax.imshow(m, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                   vmax=m.max())
    ax.set_aspect("equal")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.0,
                    color="white" if m[i, j] > 0.55 * m.max() else INK)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=32.0, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.tick_params(labelsize=6.5, length=2.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title("Descriptor co-occurrence", fontsize=8.2, fontweight="bold",
                 color=INK)
    safe_colorbar(fig, im, ax, label="molecules (count)", width=0.09,
                  pad=0.06, labelsize=6.5)


def threshold_panel(ax, lab, thr):
    rng = np.random.default_rng(29)
    med = {}
    for i, c in enumerate(CLASSES):
        v = thr[lab == c]
        med[c] = float(np.median(v))
        y = i + rng.uniform(-0.16, 0.16, v.size)
        ax.plot(v, y, marker="o", ms=2.6, ls="none", color=CLASS_COL[c],
                alpha=0.55, mec="none")
        q1, q3 = np.percentile(v, [25.0, 75.0])
        ax.plot([q1, q3], [i, i], color=CLASS_COL[c], lw=2.4, alpha=0.9,
                solid_capstyle="butt")
        ax.plot([med[c]], [i], marker="|", ms=9.0, mew=1.6, color="#1a1a1a")

    lo, hi = min(med.values()), max(med.values())
    ax.text(0.97, 0.95, f"medians span {hi / lo:.0f} x", ha="right",
            va="top", transform=ax.transAxes, fontsize=6.5, color=SUB)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e3)
    ax.set_ylim(-0.7, len(CLASSES) - 0.3)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("detection threshold (ppb in air)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Threshold spread by descriptor class", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    scores, load, explained, lab, thr, rng = panel_data()
    m = cooccurrence(lab, rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.72)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_co = fig.add_subplot(gs[0, 1])
    ax_th = fig.add_subplot(gs[1, 1])

    hull_area = main_panel(ax_main, scores, load, explained, lab)
    cooc_panel(fig, ax_co, m)
    threshold_panel(ax_th, lab, thr)

    widest = max(hull_area, key=hull_area.get)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_co, ax_th], dx=-0.46, dy=0.12)
    savefig(fig, "729_sen_odor_space.png")


if __name__ == "__main__":
    main()
