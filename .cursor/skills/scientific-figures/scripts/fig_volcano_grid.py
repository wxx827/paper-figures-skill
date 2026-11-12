"""Volcano plot grid: six differential-expression contrasts.

3 x 2 panels of log2 fold-change vs -log10(P). Up / down / not significant
are colour coded, effect-size and significance thresholds are dashed, the
strongest hits are labelled along the outer margins with a vertical
de-collision pass, and every panel carries a boxed up/down tally. One shared
legend serves the whole grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

UP, DOWN, NS = "#b23a48", "#2c6fad", "#c8ccd2"
FC_CUT, P_CUT = 1.0, 0.01

CONTRASTS = [
    ("Treated vs Control", 0.060, 1.80),
    ("Relapse vs Control", 0.090, 2.30),
    ("Relapse vs Treated", 0.042, 1.45),
    ("Responder vs Non-responder", 0.075, 2.00),
    ("Drug 24 h vs 0 h", 0.032, 1.25),
    ("Drug 72 h vs 0 h", 0.100, 2.50),
]

PREFIX = ["ARL", "BCL", "CDK", "DUSP", "EGR", "FOX", "GADD", "HSP", "IFI",
          "JUN", "KLF", "LGALS", "MMP", "NFK", "PLA", "RGS", "SOX", "TNF",
          "USP", "ZNF", "SERPIN", "COL", "CCL", "CXCL", "ATF", "MYC"]


def gene_names(rng, n):
    return [f"{PREFIX[rng.integers(len(PREFIX))]}{rng.integers(1, 40)}"
            for _ in range(n)]


def simulate(rng, de_frac, strength, n=4200):
    """Null genes plus a heavy-tailed DE component; dispersion grows with
    effect size so the volcano keeps its funnel instead of saturating."""
    lfc = rng.normal(0, 0.30, n)
    is_de = rng.random(n) < de_frac
    k = int(is_de.sum())
    sign = rng.choice([-1, 1], k, p=[0.44, 0.56])
    lfc[is_de] = sign * rng.gamma(2.4, strength / 2.4, k)
    se = 0.22 + 0.085 * np.abs(lfc) + rng.gamma(1.6, 0.070, n)
    p = 2 * stats.norm.sf(np.abs(lfc / se))
    return lfc, np.maximum(p, 1e-30)


def repel(y, min_dy=0.062, rounds=120):
    """Push label anchors apart vertically in axes-fraction space."""
    pos = np.asarray(y, dtype=float).copy()
    for _ in range(rounds):
        moved = False
        for i in range(pos.size):
            for j in range(i + 1, pos.size):
                d = pos[j] - pos[i]
                if abs(d) < min_dy:
                    shift = (min_dy - abs(d)) / 2 + 0.005
                    s = 1.0 if d >= 0 else -1.0
                    pos[i] -= s * shift
                    pos[j] += s * shift
                    moved = True
        if not moved:
            break
    return pos


def panel(ax, rng, title, de_frac, strength):
    lfc, pval = simulate(rng, de_frac, strength)
    y = -np.log10(pval)
    sig = pval < P_CUT
    up = sig & (lfc > FC_CUT)
    dn = sig & (lfc < -FC_CUT)
    ns = ~(up | dn)

    ax.scatter(lfc[ns], y[ns], s=3.0, c=NS, lw=0, alpha=0.6)
    ax.scatter(lfc[dn], y[dn], s=6.0, c=DOWN, lw=0, alpha=0.8)
    ax.scatter(lfc[up], y[up], s=6.0, c=UP, lw=0, alpha=0.8)

    xmax = np.ceil(max(3.0, np.abs(lfc).max() * 1.05))
    ymax = y.max()
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax * 0.04, ymax * 1.34)
    ax.axhline(-np.log10(P_CUT), color="#666666", ls=(0, (4, 3)), lw=0.7)
    for v in (-FC_CUT, FC_CUT):
        ax.axvline(v, color="#666666", ls=(0, (4, 3)), lw=0.7)

    # ---- label the strongest hits on each side ----------------------------
    names = gene_names(rng, lfc.size)
    score = y * np.abs(lfc)
    for mask, right in ((up, True), (dn, False)):
        idx = np.where(mask)[0]
        if not idx.size:
            continue
        picks = idx[np.argsort(score[idx])[::-1][:5]]
        # labels are parked against the outer margin so they never sit on the
        # point cloud; only their vertical order needs de-collision
        fy = repel((y[picks] + ymax * 0.04) / (ymax * 1.38))
        for k, gi in enumerate(picks):
            ax.annotate(
                names[gi], xy=(lfc[gi], y[gi]),
                xytext=(0.988 if right else 0.012, np.clip(fy[k], 0.06, 0.86)),
                textcoords="axes fraction", fontsize=6.0, color="#1a1a1a",
                ha="right" if right else "left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.45, color="#9aa0a6",
                                shrinkA=1.0, shrinkB=1.5))

    box = dict(boxstyle="round,pad=0.3", fc="white", ec="#c8ccd2", lw=0.5,
               alpha=0.92)
    ax.text(0.025, 0.97, f"Down  n={dn.sum()}", transform=ax.transAxes,
            fontsize=6.8, color=DOWN, fontweight="bold", ha="left", va="top",
            bbox=box)
    ax.text(0.975, 0.97, f"Up  n={up.sum()}", transform=ax.transAxes,
            fontsize=6.8, color=UP, fontweight="bold", ha="right", va="top",
            bbox=box)
    ax.set_title(title, fontsize=8.5, fontweight="bold", color="#1a1a1a")
    style_axes(ax)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 9.2))
    for k, (ax, (title, frac, strength)) in enumerate(zip(axes.ravel(), CONTRASTS)):
        panel(ax, rng, title, frac, strength)
        add_panel_label(ax, f"({chr(97 + k)})", x=-0.145, y=1.02)
        if k >= 4:
            ax.set_xlabel("log$_2$ fold-change")
        if k % 2 == 0:
            ax.set_ylabel("-log$_{10}$ $P$")

    handles = [
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=DOWN, mec="none",
               label=f"Down  (log$_2$FC < -{FC_CUT:.0f}, $P$ < {P_CUT})"),
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=NS, mec="none",
               label="Not significant"),
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=UP, mec="none",
               label=f"Up  (log$_2$FC > {FC_CUT:.0f}, $P$ < {P_CUT})"),
        Line2D([], [], color="#666666", ls=(0, (4, 3)), lw=0.9,
               label="Significance threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.006), columnspacing=1.6,
               handletextpad=0.5)
    fig.tight_layout(rect=[0.01, 0.035, 0.99, 0.950], h_pad=2.0, w_pad=2.4)
    savefig(fig, "72_volcano_grid.png")


if __name__ == "__main__":
    main()
