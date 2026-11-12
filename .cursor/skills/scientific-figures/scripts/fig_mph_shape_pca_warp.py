"""Shape space of a Procrustes-aligned landmark set, with thin-plate warps.

The landmark population is built once, aligned by generalized Procrustes
analysis, then handed to ``sklearn.decomposition.PCA``; the axis labels, the
scree bars and the warp amplitudes all read their percentages and standard
deviations straight out of that fitted object rather than from typed-in
numbers.  The four corner insets are true thin-plate-spline warps of the
consensus shape driven by the eigenvectors, so a reader can check that the
scatter and the deformations tell the same story.  Every shape panel is
equal-aspect: a warp grid on a stretched axes would invent an anisotropy the
model never produced.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"

K = 16
N_PER = 34
GROUPS = ("Morphotype A", "Morphotype B")
COLS = (RED, BLUE)
FIG_W, FIG_H = 10.4, 6.4
SEED = 5


# ---------------------------------------------------------------------------
# Shapes and Procrustes alignment
# ---------------------------------------------------------------------------
def template():
    th = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    r = 1.0 + 0.22 * np.cos(th) + 0.11 * np.cos(2.0 * th) + 0.05 * np.sin(3 * th)
    return np.column_stack([1.15 * r * np.cos(th), 0.80 * r * np.sin(th)])


def shape_modes(base):
    """Three unit-norm deformation modes: elongation, bending, shear."""
    elong = np.column_stack([
        np.sign(base[:, 0]) * np.abs(base[:, 0]) ** 1.3, -0.75 * base[:, 1]])
    bend = np.column_stack([
        np.zeros(K), base[:, 0] ** 2 - (base[:, 0] ** 2).mean()])
    shear = np.column_stack([base[:, 1], np.zeros(K)])
    return [m / np.linalg.norm(m) for m in (elong, bend, shear)]


def make_specimens(rng):
    """Group difference lives on mode 1; modes 2-3 vary within both groups."""
    base = template()
    m1, m2, m3 = shape_modes(base)
    amp = np.linalg.norm(base)
    configs, labels = [], []
    for g in range(2):
        for _ in range(N_PER):
            a1 = (0.086 * g + rng.normal(scale=0.020)) * amp
            a2 = rng.normal(scale=0.030) * amp
            a3 = rng.normal(scale=0.019) * amp
            cfg = (base + a1 * m1 + a2 * m2 + a3 * m3
                   + rng.normal(scale=0.0075 * amp, size=base.shape))
            a = rng.uniform(-np.pi, np.pi)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            configs.append(cfg @ rot * rng.uniform(0.7, 1.4)
                           + rng.normal(scale=0.7, size=(1, 2)))
            labels.append(g)
    return np.asarray(configs), np.asarray(labels)


def center_scale(x):
    xc = x - x.mean(axis=0)
    return xc / np.sqrt((xc ** 2).sum())


def rotate_to(x, ref):
    u, _, vt = np.linalg.svd(x.T @ ref)
    d = np.sign(np.linalg.det(u @ vt))
    return x @ (u @ np.diag([1.0, d]) @ vt)


def gpa(configs, n_iter=12):
    aligned = np.asarray([center_scale(c) for c in configs])
    consensus = center_scale(aligned[0])
    for _ in range(n_iter):
        aligned = np.asarray([rotate_to(c, consensus) for c in aligned])
        consensus = center_scale(aligned.mean(axis=0))
    return aligned, consensus


# ---------------------------------------------------------------------------
# Thin-plate spline
# ---------------------------------------------------------------------------
def _u(r2):
    return 0.5 * r2 * np.log(np.maximum(r2, 1e-14))


def tps_fit(src, dst):
    n = src.shape[0]
    r2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    p = np.hstack([np.ones((n, 1)), src])
    lmat = np.zeros((n + 3, n + 3))
    lmat[:n, :n] = _u(r2)
    lmat[:n, n:] = p
    lmat[n:, :n] = p.T
    rhs = np.vstack([dst, np.zeros((3, 2))])
    return np.linalg.solve(lmat + 1e-9 * np.eye(n + 3), rhs)


def tps_apply(w, src, pts):
    n = src.shape[0]
    r2 = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    p = np.hstack([np.ones((pts.shape[0], 1)), pts])
    return _u(r2) @ w[:n] + p @ w[n:]


def warp_inset(ax, consensus, target, color, label):
    """Deformation grid plus outline for one extreme of a principal component."""
    w = tps_fit(consensus, target)
    span = 1.30 * np.abs(consensus).max()
    grid = np.linspace(-span, span, 9)
    fine = np.linspace(-span, span, 70)
    for g in grid:
        for pts in (np.column_stack([fine, np.full_like(fine, g)]),
                    np.column_stack([np.full_like(fine, g), fine])):
            q = tps_apply(w, consensus, pts)
            ax.plot(q[:, 0], q[:, 1], color=HAIR, lw=0.4, zorder=1)
    loop = np.vstack([target, target[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color=color, lw=1.2, zorder=3)
    ax.scatter(target[:, 0], target[:, 1], s=4.0, color=color, zorder=4,
               linewidths=0.0)
    ax.set_aspect("equal")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(HAIR)
    ax.set_facecolor("white")
    ax.text(0.5, 0.012, label, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.2, color=SUB, zorder=5)


def fit_equal(ax, xlim, ylim, box_w, box_h):
    (x0, x1), (y0, y1) = xlim, ylim
    want = box_w / box_h
    if (x1 - x0) / (y1 - y0) < want:
        s = (y1 - y0) * want
        c = 0.5 * (x0 + x1)
        x0, x1 = c - s / 2.0, c + s / 2.0
    else:
        s = (x1 - x0) / want
        c = 0.5 * (y0 + y1)
        y0, y1 = c - s / 2.0, c + s / 2.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    configs, labels = make_specimens(rng)
    aligned, consensus = gpa(configs)

    flat = aligned.reshape(len(aligned), -1)
    pca = PCA(n_components=8).fit(flat)
    scores = pca.transform(flat)
    var = pca.explained_variance_ratio_ * 100.0
    sd1, sd2 = scores[:, 0].std(ddof=1), scores[:, 1].std(ddof=1)
    mean_shape = pca.mean_.reshape(K, 2)

    def warped(pc, k_sd):
        sd = (sd1, sd2)[pc]
        return mean_shape + (k_sd * sd * pca.components_[pc]).reshape(K, 2)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    mw, mh = 5.55, 4.55
    ax = axes_inches(fig, (0.62, 0.92, mw, mh))
    ax_s = axes_inches(fig, (7.05, 3.62, 3.05, 1.85))
    ax_w = axes_inches(fig, (7.05, 0.92, 3.05, 1.85))

    # ---- (a) shape space --------------------------------------------------
    ax.axhline(0.0, color=HAIR, lw=0.6, zorder=0)
    ax.axvline(0.0, color=HAIR, lw=0.6, zorder=0)
    for g in range(2):
        pts = scores[labels == g][:, :2]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=COLS[g], alpha=0.80,
                   edgecolor="white", linewidths=0.4, zorder=3)
        hull = ConvexHull(pts)
        loop = np.vstack([pts[hull.vertices], pts[hull.vertices][:1]])
        ax.fill(loop[:, 0], loop[:, 1], color=COLS[g], alpha=0.09, zorder=1)
        ax.plot(loop[:, 0], loop[:, 1], color=COLS[g], lw=0.8, alpha=0.75,
                zorder=2)
        ax.scatter(*pts.mean(axis=0), s=70, marker="X", color=COLS[g],
                   edgecolor="white", linewidths=0.9, zorder=5)

    lim = 1.55 * np.abs(scores[:, :2]).max()
    fit_equal(ax, (-lim, lim), (-lim, lim), mw, mh)
    ax.set_xlabel(f"PC 1 shape score ({var[0]:.1f} % of shape variance)",
                  fontsize=8.0)
    ax.set_ylabel(f"PC 2 shape score ({var[1]:.1f} % of shape variance)",
                  fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title(f"Shape space of {2 * N_PER} Procrustes-aligned "
                 f"{K}-landmark configurations",
                 fontsize=9.0, fontweight="bold", color=INK)

    # thin-plate warps at the four corners; square on paper, not in fractions
    hf = 0.255
    wf = hf * mh / mw
    corners = [((0.012, 0.012), 0, -3.0, r"PC1  $-$3 SD"),
               ((1.0 - wf - 0.012, 0.012), 0, +3.0, r"PC1  $+$3 SD"),
               ((0.012, 1.0 - hf - 0.012), 1, -3.0, r"PC2  $-$3 SD"),
               ((1.0 - wf - 0.012, 1.0 - hf - 0.012), 1, +3.0,
                r"PC2  $+$3 SD")]
    for (x0, y0), pc, k_sd, lab in corners:
        axi = ax.inset_axes([x0, y0, wf, hf])
        axi.set_zorder(6)
        warp_inset(axi, mean_shape, warped(pc, k_sd),
                   RED if pc == 0 else BLUE, lab)

    handles = [Line2D([], [], marker="o", ls="none", ms=4.0, color=COLS[g],
                      label=f"{GROUPS[g]} (n = {N_PER})") for g in range(2)]
    handles += [Line2D([], [], marker="X", ls="none", ms=6.0, color=GREY,
                       label="group centroid"),
                Line2D([], [], ls="-", lw=0.8, color=GREY,
                       label="convex hull")]
    ax.legend(handles=handles, fontsize=7.0, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 0.995), ncol=2, handlelength=1.4,
              columnspacing=1.4, labelspacing=0.30)

    # ---- (b) scree --------------------------------------------------------
    n_show = 8
    pos = np.arange(1, n_show + 1)
    cols = [RED, BLUE] + [GREY] * (n_show - 2)
    ax_s.bar(pos, var[:n_show], width=0.66, color=cols, edgecolor="white",
             linewidth=0.5, zorder=3)
    ax_s.set_xticks(pos)
    ax_s.set_xlim(0.4, n_show + 0.6)
    ax_s.set_xlabel("principal component", fontsize=7.4)
    ax_s.set_ylabel("variance explained (%)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.6)
    ax_s.set_ylim(0.0, var[0] * 1.28)
    ax_c = ax_s.twinx()
    ax_c.plot(pos, np.cumsum(var[:n_show]), color=INK, lw=1.0, marker="o",
              ms=2.8, zorder=4)
    ax_c.set_ylim(0.0, 105.0)
    ax_c.set_ylabel("cumulative (%)", fontsize=7.4, color=SUB)
    ax_c.tick_params(labelsize=6.6, colors=SUB)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_color(HAIR)
    ax_s.text(0.97, 0.94, f"PC1 + PC2 = {var[0] + var[1]:.1f} %",
              transform=ax_s.transAxes, ha="right", va="top", fontsize=6.8,
              color=SUB, zorder=6)
    ax_s.set_title("Two components carry the shape signal", fontsize=8.5,
                   fontweight="bold", color=INK)

    # ---- (c) shape series along PC1 --------------------------------------
    ramp = get_palette("cb_sequential_blue", 8)[3:8:2]
    steps = np.array([-3.0, 0.0, 3.0])
    for c, k_sd in zip(ramp, steps):
        s = warped(0, k_sd)
        loop = np.vstack([s, s[:1]])
        ax_w.plot(loop[:, 0], loop[:, 1], color=c, lw=1.2, zorder=3)
    lo, hi = warped(0, -3.0), warped(0, 3.0)
    for a, b in zip(lo, hi):
        ax_w.annotate("", xy=b, xytext=a, zorder=4, arrowprops=dict(
            arrowstyle="-|>", color=RED, lw=0.8, shrinkA=0, shrinkB=0,
            mutation_scale=6.0))
    ax_w.set_aspect("equal")
    sp = 1.02 * np.abs(np.vstack([lo, hi])).max()
    fit_equal(ax_w, (-sp, sp), (-sp, sp), 3.05, 1.85)
    ax_w.set_xticks([])
    ax_w.set_yticks([])
    for s in ax_w.spines.values():
        s.set_visible(False)
    ax_w.text(0.015, 0.02, r"$-$3 SD", transform=ax_w.transAxes, ha="left",
              va="bottom", fontsize=7.0, color=ramp[0])
    ax_w.text(0.985, 0.02, r"$+$3 SD", transform=ax_w.transAxes, ha="right",
              va="bottom", fontsize=7.0, color=ramp[-1])
    ax_w.text(0.5, 0.02, "arrows: landmark displacement",
              transform=ax_w.transAxes, ha="center", va="bottom",
              fontsize=6.4, color=MUTED)
    ax_w.set_title("Shape change along PC 1", fontsize=8.5,
                   fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.44, dy=0.10)
    panel_labels_inches(fig, [ax_s], labels=["(b)"], dx=-0.46, dy=0.10)
    panel_labels_inches(fig, [ax_w], labels=["(c)"], dx=-0.46, dy=0.10)
    savefig(fig, "823_mph_shape_pca_warp.png")


if __name__ == "__main__":
    main()
