"""Generalized Procrustes superimposition of a two-morphotype landmark set.

One synthetic landmark population is built once - a 12-landmark outline, a
smooth between-group deformation, isotropic digitising noise, then a random
similarity transform per specimen - and every panel reads from that single
array.  The main panel is the superimposed scatter, so it carries the
equal-aspect constraint that any shape plot needs: a Procrustes cloud drawn on
a stretched axes reports a landmark variance that does not exist.  The two
subordinate panels on the right carry the evidence for the superimposition
itself: how far the configurations sat from the consensus before and after
alignment, and which landmarks actually hold the residual variance.

Every number printed on the figure - Procrustes distances, their medians, the
per-landmark variances, the variance share of the top landmark - is computed
from the arrays that are drawn.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"

K = 12                       # landmarks per configuration
N_PER = 26                   # specimens per morphotype
GROUPS = ("Morphotype A", "Morphotype B")
COLS = (RED, BLUE)
FIG_W, FIG_H = 10.4, 6.4
SEED = 12


# ---------------------------------------------------------------------------
# Shape construction
# ---------------------------------------------------------------------------
def template():
    """Twelve landmarks on a smooth closed outline (a generic valve shape)."""
    th = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
    r = 1.0 + 0.26 * np.cos(th) + 0.10 * np.cos(2.0 * th)
    return np.column_stack([1.18 * r * np.cos(th), 0.82 * r * np.sin(th)])


def deformation(base):
    """Between-group shape difference: anterior stretch, dorsoventral squash."""
    dx = 0.17 * np.sign(base[:, 0]) * np.abs(base[:, 0]) ** 1.25
    dy = -0.13 * base[:, 1]
    return np.column_stack([dx, dy])


def make_specimens(rng):
    base = template()
    defo = deformation(base)
    configs, labels = [], []
    for g in range(2):
        for _ in range(N_PER):
            cfg = base + g * defo
            cfg = cfg + rng.normal(scale=0.030, size=cfg.shape)
            a = rng.uniform(-np.pi, np.pi)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            cfg = cfg @ rot * rng.uniform(0.65, 1.55)
            cfg = cfg + rng.normal(scale=0.8, size=(1, 2))
            configs.append(cfg)
            labels.append(g)
    return np.asarray(configs), np.asarray(labels)


# ---------------------------------------------------------------------------
# Generalized Procrustes analysis
# ---------------------------------------------------------------------------
def center_scale(x):
    xc = x - x.mean(axis=0)
    return xc / np.sqrt((xc ** 2).sum())


def rotate_to(x, ref):
    """Ordinary Procrustes rotation of ``x`` onto ``ref`` (reflection barred)."""
    u, _, vt = np.linalg.svd(x.T @ ref)
    d = np.sign(np.linalg.det(u @ vt))
    return x @ (u @ np.diag([1.0, d]) @ vt)


def gpa(configs, n_iter=12):
    aligned = np.asarray([center_scale(c) for c in configs])
    pre = aligned.copy()
    consensus = center_scale(aligned[0])
    for _ in range(n_iter):
        aligned = np.asarray([rotate_to(c, consensus) for c in aligned])
        consensus = center_scale(aligned.mean(axis=0))
    return pre, aligned, consensus


def procrustes_distance(configs, consensus):
    return np.sqrt(((configs - consensus) ** 2).sum(axis=(1, 2)))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def fit_equal(ax, xlim, ylim, box_w, box_h):
    """Pad the tighter axis so the data aspect equals the axes-box aspect."""
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


def conf_ellipse(ax, pts, color):
    """95 % confidence ellipse of a landmark cluster (chi-square, 2 d.o.f.)."""
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * np.sqrt(np.maximum(vals, 0.0) * 5.991)
    ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=ang, facecolor=color,
                         alpha=0.14, edgecolor=color, linewidth=0.7,
                         zorder=3))


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    configs, labels = make_specimens(rng)
    pre, post, consensus = gpa(configs)

    d_pre = procrustes_distance(pre, consensus)
    d_post = procrustes_distance(post, consensus)
    # Per-landmark Procrustes variance: mean squared deviation from consensus.
    lm_var = ((post - consensus) ** 2).sum(axis=2).mean(axis=0)
    top_lm = int(np.argmax(lm_var))
    top_share = 100.0 * lm_var[top_lm] / lm_var.sum()
    drop = 100.0 * (1.0 - np.median(d_post) / np.median(d_pre))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    mw, mh = 5.55, 4.55
    ax = axes_inches(fig, (0.62, 0.92, mw, mh))
    ax_d = axes_inches(fig, (7.05, 3.62, 3.05, 1.85))
    ax_v = axes_inches(fig, (7.05, 0.92, 3.05, 1.85))

    # ---- (a) superimposed landmark cloud ---------------------------------
    ax.set_aspect("equal")
    ax.axhline(0.0, color=HAIR, lw=0.6, zorder=0)
    ax.axvline(0.0, color=HAIR, lw=0.6, zorder=0)
    for g in range(2):
        pts = post[labels == g]
        ax.scatter(pts[:, :, 0].ravel(), pts[:, :, 1].ravel(), s=3.2,
                   color=COLS[g], alpha=0.42, linewidths=0.0, zorder=2)
        for k in range(K):
            conf_ellipse(ax, pts[:, k, :], COLS[g])

    loop = np.vstack([consensus, consensus[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color=INK, lw=1.1, zorder=5,
            solid_joinstyle="round")
    ax.scatter(consensus[:, 0], consensus[:, 1], s=26, color="white",
               edgecolor=INK, linewidths=1.0, zorder=6)
    for k in range(K):
        v = consensus[k] / np.linalg.norm(consensus[k])
        ax.text(consensus[k, 0] + 0.052 * v[0], consensus[k, 1] + 0.052 * v[1],
                str(k + 1), fontsize=6.4, color=INK, ha="center", va="center",
                zorder=7)

    fit_equal(ax, (-0.44, 0.44), (-0.30, 0.30), mw, mh)
    ax.set_xlabel("Procrustes shape coordinate $x$ (unit centroid size)",
                  fontsize=8.0)
    ax.set_ylabel("Procrustes shape coordinate $y$ (unit centroid size)",
                  fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Generalized Procrustes superimposition, "
                 f"{2 * N_PER} specimens x {K} landmarks",
                 fontsize=9.0, fontweight="bold", color=INK)
    handles = [Line2D([], [], marker="o", ls="none", ms=4.0, color=COLS[g],
                      label=f"{GROUPS[g]} (n = {N_PER})") for g in range(2)]
    handles += [Line2D([], [], marker="o", ls="-", ms=4.6, lw=1.1,
                       color=INK, mfc="white", mew=1.0,
                       label="consensus (mean) shape"),
                Line2D([], [], ls="-", lw=0.8, color=GREY,
                       label="95 % ellipse per landmark")]
    ax.legend(handles=handles, fontsize=7.0, frameon=False, loc="lower left",
              bbox_to_anchor=(0.005, 0.005), handlelength=1.5,
              labelspacing=0.32)

    # ---- (b) Procrustes distance before / after --------------------------
    lo = min(d_pre.min(), d_post.min()) * 0.75
    hi = max(d_pre.max(), d_post.max()) * 1.35
    edges = np.logspace(np.log10(lo), np.log10(hi), 28)
    ax_d.set_xscale("log")
    ax_d.hist(d_pre, bins=edges, color=GREY, alpha=0.45, edgecolor=GREY,
              linewidth=0.7, label="centred + scaled only")
    ax_d.hist(d_post, bins=edges, color=BLUE, alpha=0.72, edgecolor=BLUE,
              linewidth=0.7, label="after full GPA")
    for val, col in ((np.median(d_pre), GREY), (np.median(d_post), BLUE)):
        ax_d.axvline(val, color=col, lw=1.0, ls=(0, (2.6, 1.6)), zorder=4)
    ax_d.set_xlim(lo, hi)
    ax_d.text(0.97, 0.95,
              f"median {np.median(d_pre):.3f} -> {np.median(d_post):.3f}\n"
              f"({drop:.0f} % reduction)", transform=ax_d.transAxes,
              ha="right", va="top", fontsize=6.6, color=SUB)
    ax_d.set_xlabel("Procrustes distance to consensus (log10 axis)",
                    fontsize=7.4)
    ax_d.set_ylabel("specimens (n)", fontsize=7.4)
    ax_d.tick_params(labelsize=6.6)
    ax_d.legend(fontsize=6.6, frameon=False, loc="upper right",
                bbox_to_anchor=(1.0, 0.80), handlelength=1.3)
    ax_d.set_title("Alignment removes the nuisance variation",
                   fontsize=8.5, fontweight="bold", color=INK)

    # ---- (c) per-landmark variance ---------------------------------------
    pos = np.arange(1, K + 1)
    cols = [RED if k == top_lm else BLUE for k in range(K)]
    ax_v.bar(pos, lm_var * 1e3, width=0.68, color=cols, edgecolor="white",
             linewidth=0.5)
    ax_v.axhline(lm_var.mean() * 1e3, color=GREY, lw=0.9,
                 ls=(0, (3.0, 1.8)), zorder=4)
    ax_v.text(0.52, lm_var.mean() * 1e3, "mean", fontsize=6.4,
              color=SUB, ha="left", va="bottom")
    ax_v.text(top_lm + 1, lm_var[top_lm] * 1e3, f" {top_share:.0f} % ",
              fontsize=6.6, color=RED, ha="center", va="bottom")
    ax_v.set_xticks(pos)
    ax_v.set_xlim(0.4, K + 0.6)
    ax_v.set_xlabel("landmark index", fontsize=7.4)
    ax_v.set_ylabel(r"Procrustes variance ($\times 10^{-3}$)", fontsize=7.4)
    ax_v.tick_params(labelsize=6.6)
    ax_v.set_title("Residual variance is not spread evenly",
                   fontsize=8.5, fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.44, dy=0.10)
    panel_labels_inches(fig, [ax_d], labels=["(b)"], dx=-0.46, dy=0.10)
    panel_labels_inches(fig, [ax_v], labels=["(c)"], dx=-0.46, dy=0.10)
    savefig(fig, "822_mph_landmark_gpa.png")


if __name__ == "__main__":
    main()
