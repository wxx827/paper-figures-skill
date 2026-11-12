"""Valence-arousal plane for four genres, with marginals and rater agreement.

Every track is rated by five simulated annotators; the point plotted on the
plane is the mean of those five ratings, the ellipse is the 95% contour of the
sample covariance of the plotted points, and the agreement bars are the mean
pairwise Pearson correlation between annotators.  Genres whose ellipse is wide
are therefore also the genres whose agreement bar is short, because both
numbers come from the same rating tensor.
"""
import textwrap
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#cccccc"

# name, short code, (valence, arousal) centre, spread, annotator noise
GENRES = [
    ("Electronic", "EL", (0.46, 0.54), (0.20, 0.16), 0.13, "#b23a48"),
    ("Metal", "MT", (-0.30, 0.62), (0.17, 0.14), 0.16, "#c9a227"),
    ("Jazz", "JZ", (0.28, -0.16), (0.24, 0.21), 0.22, "#3b5b92"),
    ("Ambient", "AM", (-0.08, -0.58), (0.22, 0.15), 0.18, "#2e7d5b"),
]
N_TRACKS = 26
N_RATERS = 5
CHI2_95 = 5.991      # two degrees of freedom

QUADRANTS = [
    (0.5, 0.5, "excited\nelated", "#3b5b92"),
    (-0.5, 0.5, "angry\ntense", "#b23a48"),
    (-0.5, -0.5, "sad\ndepressed", "#c0603f"),
    (0.5, -0.5, "content\ncalm", "#2e7d5b"),
]


def rate(rng):
    """Latent track ratings plus per-annotator ratings for each genre."""
    data = {}
    for name, code, (mv, ma), (sv, sa), noise, col in GENRES:
        latent = np.column_stack([
            rng.normal(mv, sv, N_TRACKS),
            rng.normal(ma, sa, N_TRACKS)])
        latent = np.clip(latent, -0.97, 0.97)
        raters = np.clip(
            latent[None, :, :] + rng.normal(0.0, noise,
                                            (N_RATERS, N_TRACKS, 2)),
            -1.0, 1.0)
        data[name] = dict(code=code, color=col, raters=raters,
                          mean=raters.mean(axis=0))
    return data


def mean_pairwise_r(raters, dim):
    vals = [np.corrcoef(raters[i, :, dim], raters[j, :, dim])[0, 1]
            for i, j in combinations(range(N_RATERS), 2)]
    return float(np.mean(vals))


def cov_ellipse(ax, pts, color):
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * np.sqrt(CHI2_95 * vals)
    ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=color, alpha=0.10,
                         edgecolor=color, lw=1.1, zorder=3))
    return mu, h, ang


def main():
    set_pub_style()
    rng = np.random.default_rng(850)
    data = rate(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 0.92, 4.68, 4.68))
    ax_marg = axes_inches(fig, (6.65, 3.30, 2.85, 2.30))
    ax_agr = axes_inches(fig, (6.65, 0.92, 2.85, 1.90))

    # ---- valence-arousal plane -------------------------------------------
    ax.set_aspect("equal")
    for qx, qy, word, col in QUADRANTS:
        ax.add_patch(plt.Rectangle((0.0 if qx > 0 else -1.0,
                                    0.0 if qy > 0 else -1.0), 1.0, 1.0,
                                   facecolor=col, alpha=0.055,
                                   edgecolor="none", zorder=0))
        ax.text(0.96 * qx / abs(qx), 0.96 * qy / abs(qy), word,
                ha="right" if qx > 0 else "left",
                va="top" if qy > 0 else "bottom", fontsize=7.0,
                color=col, linespacing=1.35, zorder=1)
    ax.axhline(0.0, color=RULE, lw=0.9, zorder=2)
    ax.axvline(0.0, color=RULE, lw=0.9, zorder=2)

    n_total = 0
    for name, code, _, _, _, _ in GENRES:
        d = data[name]
        pts = d["mean"]
        n_total += pts.shape[0]
        ax.plot(pts[:, 0], pts[:, 1], ls="none", marker="o", ms=3.4,
                mfc=d["color"], mec="white", mew=0.4, alpha=0.9, zorder=4)
        mu, h, _ = cov_ellipse(ax, pts, d["color"])
        ax.text(mu[0], min(mu[1] + 0.5 * h + 0.03, 0.88), name, ha="center",
                va="bottom", fontsize=7.4, color=d["color"], zorder=6)

    # three leader-line callouts: the extreme of three different genres
    picks = [
        ("Metal", int(np.argmax(data["Metal"]["mean"][:, 1])),
         (-0.98, 0.26), "left"),
        ("Jazz", int(np.argmax(data["Jazz"]["mean"][:, 0])),
         (0.98, -0.46), "right"),
        ("Ambient", int(np.argmin(data["Ambient"]["mean"][:, 1])),
         (-0.46, -0.92), "center"),
    ]
    for name, idx, at, ha in picks:
        d = data[name]
        x, y = d["mean"][idx]
        ax.annotate("Track %s-%02d\nv %+.2f, a %+.2f" % (d["code"], idx, x, y),
                    xy=(x, y), xytext=at, fontsize=6.4, color=INK, ha=ha,
                    va="center", linespacing=1.35, zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED,
                                    shrinkA=1.0, shrinkB=2.0))

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
    ax.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax.set_xlabel("valence (-, negative to positive)", fontsize=8.0)
    ax.set_ylabel("arousal (-, calm to activated)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Affect plane, n=%d tracks, %d annotators each"
                 % (n_total, N_RATERS), fontsize=9.0, fontweight="bold",
                 pad=4.0)

    # ---- marginal distributions -------------------------------------------
    pos_v = np.arange(4) * 1.0 + 1.0
    pos_a = np.arange(4) * 1.0 + 6.0
    for dim, pos in ((0, pos_v), (1, pos_a)):
        parts = ax_marg.violinplot(
            [data[g[0]]["mean"][:, dim] for g in GENRES], positions=pos,
            widths=0.78, showextrema=False, showmedians=True)
        for body, g in zip(parts["bodies"], GENRES):
            body.set_facecolor(g[5])
            body.set_alpha(0.42)
            body.set_edgecolor(g[5])
            body.set_linewidth(0.8)
        parts["cmedians"].set_color(RULE)
        parts["cmedians"].set_linewidth(0.9)
    ax_marg.axhline(0.0, color="#808080", lw=0.8, ls=(0, (4, 3)))
    ax_marg.axvline(5.0, color=GRID, lw=0.8)
    ax_marg.text(2.5, 1.02, "valence", ha="center", va="bottom", fontsize=7.0,
                 color=SUB)
    ax_marg.text(7.5, 1.02, "arousal", ha="center", va="bottom", fontsize=7.0,
                 color=SUB)
    ax_marg.set_xticks(np.r_[pos_v, pos_a])
    ax_marg.set_xticklabels([g[1] for g in GENRES] * 2, fontsize=6.5)
    ax_marg.set_xlim(0.3, 9.7)
    ax_marg.set_ylim(-1.05, 1.22)
    ax_marg.set_ylabel("rating (-)", fontsize=7.5)
    ax_marg.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_marg.spines[s].set_visible(False)
    ax_marg.set_title("Marginal ratings by genre", fontsize=8.5,
                      fontweight="bold", pad=3.0)

    # ---- annotator agreement ----------------------------------------------
    rv = [mean_pairwise_r(data[g[0]]["raters"], 0) for g in GENRES]
    ra = [mean_pairwise_r(data[g[0]]["raters"], 1) for g in GENRES]
    xs = np.arange(4)
    ax_agr.bar(xs - 0.19, rv, width=0.34, color="#3b5b92", alpha=0.88,
               edgecolor="white", lw=0.5, label="valence")
    ax_agr.bar(xs + 0.19, ra, width=0.34, color="#c0603f", alpha=0.88,
               edgecolor="white", lw=0.5, label="arousal")
    for x, v, a in zip(xs, rv, ra):
        ax_agr.text(x - 0.19, v + 0.015, "%.2f" % v, ha="center", va="bottom",
                    fontsize=6.0, color=SUB)
        ax_agr.text(x + 0.19, a + 0.015, "%.2f" % a, ha="center", va="bottom",
                    fontsize=6.0, color=SUB)
    ax_agr.set_xticks(xs)
    ax_agr.set_xticklabels([g[1] for g in GENRES], fontsize=6.5)
    ax_agr.set_ylim(0.0, max(rv + ra) * 1.25)
    ax_agr.set_ylabel("mean pairwise r (-)", fontsize=7.5)
    ax_agr.set_xlabel("genre", fontsize=7.5)
    ax_agr.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_agr.spines[s].set_visible(False)
    ax_agr.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, 1.03), handlelength=1.1,
                  columnspacing=1.2)
    ax_agr.set_title("Annotator agreement, %d pairs per bar"
                     % (N_RATERS * (N_RATERS - 1) // 2), fontsize=8.5,
                     fontweight="bold", pad=3.0)

    panel_labels_inches(fig, [ax, ax_marg, ax_agr], dx=-0.52, dy=0.12)


    savefig(fig, "850_aff_valence_arousal.png")


if __name__ == "__main__":
    main()
