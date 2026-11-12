"""Figure 487 - Client drift under non-IID partitions.

The main panel is the parameter space: a shared loss basin whose minimum is
the global optimum, the four local optima each client would reach if it were
allowed to train to convergence, and the sample-weighted average of those
local optima.  The dashed spokes are the drift vectors; the red spoke is the
residual bias that survives averaging, which is what the two subordinate
panels then track over training - accuracy under IID against non-IID
partitions, and the mean pairwise cosine similarity between client updates.

Every number quoted in the annotations is recomputed from the arrays plotted
here, so re-seeding the generator moves the text with the geometry.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, get_cmap

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
TEAL = "#2a7f7a"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

FIG_W, FIG_H = 10.4, 6.4
MAIN = [0.055, 0.115, 0.545, 0.755]
ACC = [0.690, 0.580, 0.290, 0.290]
COS = [0.690, 0.115, 0.290, 0.290]

XLIM, YLIM = (-3.0, 3.4), (-2.6, 2.9)
THETA = np.deg2rad(24.0)
CURV = (0.36, 1.05)                    # basin curvature along its two axes

CLIENTS = [("Client 1", GREEN, 1.70, 120.0, 1240),
           ("Client 2", GOLD, 2.10, 168.0, 2110),
           ("Client 3", PURPLE, 1.50, 232.0, 860),
           ("Client 4", TEAL, 1.90, 305.0, 1475)]
ROUNDS = np.arange(1, 101)


def loss_surface(x, y):
    """Rotated quadratic basin centred on the global optimum."""
    u = np.cos(THETA) * x + np.sin(THETA) * y
    v = -np.sin(THETA) * x + np.cos(THETA) * y
    return CURV[0] * u ** 2 + CURV[1] * v ** 2


def curves(rng):
    """Accuracy and update-agreement traces for the two partition regimes."""
    r = ROUNDS
    acc_iid = 91.5 * (1.0 - np.exp(-r / 15.0)) + rng.normal(0, 0.45, r.size)
    acc_niid = (83.0 * (1.0 - np.exp(-r / 27.0))
                + 2.1 * np.sin(r / 4.0) * np.exp(-r / 55.0)
                + rng.normal(0, 1.05, r.size))
    cos_iid = 0.84 - 0.07 * (1.0 - np.exp(-r / 30.0)) + rng.normal(0, .014, r.size)
    cos_niid = 0.18 + 0.62 * np.exp(-r / 34.0) + rng.normal(0, 0.022, r.size)
    return acc_iid, acc_niid, cos_iid, cos_niid


def aux_axes(ax, title):
    ax.tick_params(labelsize=6.8, length=2.2)
    ax.grid(True, axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4.0)


def main():
    set_pub_style()
    rng = np.random.default_rng(487)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---- parameter space --------------------------------------------------
    ax = fig.add_axes(MAIN)
    gx = np.linspace(*XLIM, 320)
    gy = np.linspace(*YLIM, 300)
    X, Y = np.meshgrid(gx, gy)
    Z = loss_surface(X, Y)
    levels = np.linspace(0.0, Z.max(), 13)
    ax.contourf(X, Y, Z, levels=levels, cmap=get_cmap("cb_sequential_blue"),
                alpha=0.42, zorder=0)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.5,
               zorder=1)

    pts, weights, cols = [], [], []
    for name, col, rad, ang, nk in CLIENTS:
        a = np.deg2rad(ang)
        pts.append((rad * np.cos(a), rad * np.sin(a) * 0.92))
        weights.append(nk)
        cols.append(col)
    pts = np.asarray(pts, dtype=float)
    weights = np.asarray(weights, dtype=float)
    agg = (weights[:, None] * pts).sum(axis=0) / weights.sum()

    for (name, col, _, _, nk), p in zip(CLIENTS, pts):
        ax.annotate("", xy=tuple(p), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.15, color=col,
                                    ls=(0, (4, 2.2)), shrinkA=6.0,
                                    shrinkB=6.5, mutation_scale=11))
        cloud = p + rng.normal(0.0, 0.20, size=(26, 2))
        ax.scatter(cloud[:, 0], cloud[:, 1], s=9, facecolor=col, alpha=0.28,
                   edgecolor="none", zorder=3)
        ax.scatter(*p, s=52, marker="o", facecolor=col, edgecolor="white",
                   linewidth=1.0, zorder=6)
        off = 0.30 if p[1] >= 0 else -0.30
        ax.text(p[0], p[1] + off, f"{name}  $n_k$={nk:,}", ha="center",
                va="bottom" if off > 0 else "top", fontsize=6.6, color=col,
                fontweight="bold", zorder=7)

    ax.annotate("", xy=tuple(agg), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color=ACCENT,
                                shrinkA=7.0, shrinkB=6.0, mutation_scale=13))
    ax.scatter(*agg, s=78, marker="o", facecolor=ACCENT, edgecolor="white",
               linewidth=1.1, zorder=7)
    ax.scatter(0.0, 0.0, s=210, marker="*", facecolor=BLUE, edgecolor="white",
               linewidth=1.0, zorder=8)
    ax.text(0.16, 0.12, r"global optimum $w^\star$", ha="left", va="bottom",
            fontsize=7.0, color=BLUE, fontweight="bold", zorder=8)
    ax.text(agg[0], agg[1] - 0.20, r"aggregate $\bar{w}$", ha="center",
            va="top", fontsize=7.0, color=ACCENT, fontweight="bold",
            zorder=8)

    drift = np.linalg.norm(pts, axis=1)
    bias = float(np.linalg.norm(agg))
    ax.text(0.018, 0.030,
            f"mean drift $\\|w^k-w^\\star\\|$ = {drift.mean():.2f}   ·   "
            f"residual bias $\\|\\bar{{w}}-w^\\star\\|$ = {bias:.2f}",
            transform=ax.transAxes, fontsize=6.8, color=SUB, ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"parameter direction $w_1$ (a.u.)", fontsize=8.0,
                  labelpad=2.5)
    ax.set_ylabel(r"parameter direction $w_2$ (a.u.)", fontsize=8.0,
                  labelpad=2.5)
    ax.tick_params(labelsize=7.0, length=2.5)
    style_axes(ax)
    ax.set_title("Local optima pull the average off the shared minimum",
                 fontsize=8.5, fontweight="bold", pad=4.0)

    handles = [Line2D([], [], ls="none", marker="*", ms=9, mfc=BLUE,
                      mec="white", mew=0.8, label=r"global optimum $w^\star$"),
               Line2D([], [], ls="none", marker="o", ms=5.6, mfc=ACCENT,
                      mec="white", mew=0.9, label=r"FedAvg aggregate $\bar{w}$"),
               Line2D([], [], color=MUTED, ls=(0, (4, 2.2)), lw=1.1,
                      label="client drift vector")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.9,
              handletextpad=0.6, labelspacing=0.34, borderaxespad=0.35)

    # ---- accuracy over rounds --------------------------------------------
    acc_iid, acc_niid, cos_iid, cos_niid = curves(rng)
    axa = fig.add_axes(ACC)
    axa.plot(ROUNDS, acc_iid, color=BLUE, lw=1.5, label="IID partition")
    axa.plot(ROUNDS, acc_niid, color=ACCENT, lw=1.5,
             label=r"non-IID  ($\alpha$ = 0.1)")
    axa.fill_between(ROUNDS, acc_niid, acc_iid, color=ACCENT, alpha=0.09,
                     lw=0)
    gap = float(acc_iid[-10:].mean() - acc_niid[-10:].mean())
    axa.annotate(f"{gap:.1f} pt gap", xy=(ROUNDS[-1], (acc_iid[-1] +
                                                       acc_niid[-1]) / 2),
                 xytext=(-6, 0), textcoords="offset points", ha="right",
                 va="center", fontsize=6.6, color=ACCENT, fontweight="bold")
    axa.set_xlim(0, 102)
    axa.set_ylim(0, 100)
    axa.set_yticks([0, 25, 50, 75, 100])
    axa.set_xlabel("communication rounds", fontsize=7.5, labelpad=1.8)
    axa.set_ylabel("global accuracy (%)", fontsize=7.5, labelpad=2.0)
    axa.legend(loc="lower right", frameon=False, fontsize=6.6,
               handlelength=1.4, labelspacing=0.3, borderaxespad=0.3)
    aux_axes(axa, "Skew costs accuracy and stability")

    # ---- cosine similarity between client updates ------------------------
    axc = fig.add_axes(COS)
    axc.plot(ROUNDS, cos_iid, color=BLUE, lw=1.5)
    axc.plot(ROUNDS, cos_niid, color=ACCENT, lw=1.5)
    for y, txt, col, dy in ((cos_iid[-1], "IID", BLUE, 6),
                            (cos_niid[-1], "non-IID", ACCENT, 11)):
        axc.annotate(txt, xy=(ROUNDS[-1], y), xytext=(-4, dy),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=6.6, color=col, fontweight="bold")
    fall = float(cos_niid[:5].mean() - cos_niid[-5:].mean())
    axc.text(0.03, 0.05, f"non-IID agreement falls {fall:.2f} over "
             f"{ROUNDS[-1]} rounds", transform=axc.transAxes, fontsize=6.4,
             color=SUB, ha="left", va="bottom")
    axc.set_xlim(0, 102)
    axc.set_ylim(0.0, 1.0)
    axc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axc.set_xlabel("communication rounds", fontsize=7.5, labelpad=1.8)
    axc.set_ylabel("update cosine similarity (-)", fontsize=7.5, labelpad=2.0)
    aux_axes(axc, "Client updates stop agreeing")

    for tag, rect in (("(a)", MAIN), ("(b)", ACC), ("(c)", COS)):
        fig.text(rect[0] - 0.40 / FIG_W, rect[1] + rect[3] + 0.07 / FIG_H,
                 tag, fontsize=11, fontweight="bold", ha="left", va="bottom")

    savefig(fig, "487_fed_client_drift.png")


if __name__ == "__main__":
    main()
