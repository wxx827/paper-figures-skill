"""Loss surface along two random directions, with optimiser trajectories.

Left: the 3-D surface with its contour projection on the floor.  Right: the
same landscape from above with two optimisation paths actually integrated on
the analytic loss (momentum SGD versus the proposed adaptive step), arrow
heads showing the direction of travel and epoch stamps along the way.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj.)
from matplotlib.lines import Line2D

from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label,
                      get_cmap, add_colorbar)

OURS = "#3b5b92"
GREY = "#8c8c8c"
ORANGE = "#c9a227"

START = np.array([-2.55, 2.35])
STEPS = 120
GMIN = np.array([1.15, -0.95])
LMIN = np.array([-1.55, 1.25])


def loss(x, y):
    """Two basins separated by a central ridge; the deeper one is global."""
    return (0.085 * (x ** 2 + y ** 2)
            - 1.45 * np.exp(-((x - GMIN[0]) ** 2 + (y - GMIN[1]) ** 2) / 0.75)
            - 0.95 * np.exp(-((x - LMIN[0]) ** 2 + (y - LMIN[1]) ** 2) / 0.70)
            + 0.60 * np.exp(-(x ** 2 + y ** 2) / 1.10)
            + 0.10 * np.sin(1.8 * x) * np.cos(1.8 * y) + 1.75)


def grad(p, h=1e-4):
    x, y = p
    return np.array([(loss(x + h, y) - loss(x - h, y)) / (2 * h),
                     (loss(x, y + h) - loss(x, y - h)) / (2 * h)])


def run_sgd(lr=0.10, mom=0.82, noise=0.055, seed=11):
    """Plain momentum: falls into the first basin it meets and stays."""
    rng = np.random.default_rng(seed)
    p, v = START.copy(), np.zeros(2)
    path = [p.copy()]
    for _ in range(STEPS):
        g = grad(p) + rng.normal(0, noise, 2)
        v = mom * v - lr * g
        p = p + v
        path.append(p.copy())
    return np.array(path)


def run_ours(lr=0.29, beta=0.90, eps=2e-2, noise=0.020, seed=4):
    """RMSProp-style preconditioning: normalised steps clear the ridge."""
    rng = np.random.default_rng(seed)
    p, s, v = START.copy(), np.zeros(2), np.zeros(2)
    path = [p.copy()]
    for k in range(STEPS):
        g = grad(p) + rng.normal(0, noise, 2)
        s = beta * s + (1 - beta) * g ** 2
        decay = 0.12 + 0.88 * np.exp(-k / 30.0)      # annealed step size
        v = 0.70 * v - lr * decay * g / (np.sqrt(s) + eps)
        p = p + v
        path.append(p.copy())
    return np.array(path)


def main():
    set_pub_style(base_font=8)
    cmap = get_cmap("cb_sequential_blue")

    g = np.linspace(-3.0, 3.0, 240)
    X, Y = np.meshgrid(g, g)
    Z = loss(X, Y)
    levels = np.linspace(Z.min(), Z.max(), 20)

    sgd, ours = run_sgd(), run_ours()
    imin = np.unravel_index(np.argmin(Z), Z.shape)
    pmin = (X[imin], Y[imin])

    fig = plt.figure(figsize=(7.4, 3.5))
    ax3 = fig.add_axes([-0.015, 0.055, 0.50, 0.82], projection="3d")
    ax2 = fig.add_axes([0.545, 0.145, 0.335, 0.73])

    # ---- (a) 3-D surface ---------------------------------------------------
    ax3.plot_surface(X, Y, Z, cmap=cmap, rstride=3, cstride=3, linewidth=0,
                     antialiased=True, alpha=0.94, vmin=Z.min(), vmax=Z.max())
    zfloor = Z.min() - 1.35
    ax3.contour(X, Y, Z, levels=levels, zdir="z", offset=zfloor,
                cmap=cmap, linewidths=0.5, alpha=0.9)
    ax3.plot([pmin[0]], [pmin[1]], [zfloor], marker="*", ms=8,
             color="#b23a48", zorder=10)
    ax3.plot([LMIN[0]], [LMIN[1]], [zfloor], marker="o", ms=3.4, mfc="white",
             mec="#666666", mew=0.9, zorder=10)
    ax3.set_zlim(zfloor, Z.max() * 1.02)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.view_init(elev=30, azim=-56)
    ax3.set_box_aspect((1, 1, 0.62))
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#dddddd")
        pane.pane.set_alpha(1.0)
        pane._axinfo["grid"].update(color="#eeeeee", linewidth=0.5)
    ax3.tick_params(labelsize=6.2, pad=-1.5, colors="#555555")
    ax3.set_xlabel(r"direction  $\delta_1$", fontsize=7.2, labelpad=-5)
    ax3.set_ylabel(r"direction  $\delta_2$", fontsize=7.2, labelpad=-5)
    ax3.set_zlabel("loss  $L$", fontsize=7.2, labelpad=-6)
    ax3.set_zticks([1, 2, 3])
    ax3.set_xticks([-2, 0, 2])
    ax3.set_yticks([-2, 0, 2])
    ax3.set_title("Loss surface with contour projection", fontsize=8.5,
                  pad=-14)

    # ---- (b) top view with trajectories ------------------------------------
    cf = ax2.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.95, zorder=0)
    ax2.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.35,
                alpha=0.6, zorder=1)

    for path, col, lw, marks, arrows in (
            (sgd, ORANGE, 1.4, (0, 10, 30, 120), (4, 12)),
            (ours, OURS, 2.0, (0, 4, 10, 120), (2, 6))):
        ax2.plot(path[:, 0], path[:, 1], color="white", lw=lw + 1.4,
                 zorder=4, solid_capstyle="round", alpha=0.7)
        ax2.plot(path[:, 0], path[:, 1], color=col, lw=lw, zorder=5,
                 solid_capstyle="round")
        for i in arrows:
            ax2.annotate("", xy=path[i + 2], xytext=path[i],
                         arrowprops=dict(arrowstyle="-|>", lw=0, color=col,
                                         mutation_scale=10), zorder=6)
        for ep in marks:
            ax2.plot(*path[ep], marker="o", ms=3.2, mfc="white", mec=col,
                     mew=1.1, zorder=7)

    halo = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82)
    ax2.annotate("ep 120\n(trapped)", xy=tuple(sgd[120]), xytext=(-0.55, 1.75),
                 fontsize=6.1, color=ORANGE, ha="left", va="center",
                 linespacing=1.25, zorder=8, bbox=halo,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE,
                                 shrinkA=1, shrinkB=3))
    ax2.text(ours[10][0] + 0.38, ours[10][1] - 0.26, "ep 10", fontsize=6.1,
             color=OURS, ha="left", va="center", zorder=8, bbox=halo)
    ax2.text(ours[120][0] + 0.34, ours[120][1] - 0.62, "ep 120",
             fontsize=6.3, color=OURS, ha="left", va="top", zorder=8,
             fontweight="bold", bbox=halo)

    ax2.plot(*START, marker="s", ms=5.0, mfc="white", mec="#333333", mew=1.0,
             zorder=8)
    ax2.text(START[0] + 0.22, START[1] + 0.10, "init", fontsize=6.5,
             color="#333333", ha="left", va="bottom", zorder=8, bbox=halo)
    ax2.plot(*pmin, marker="*", ms=10, mfc="#b23a48", mec="white", mew=0.7,
             zorder=9)
    ax2.annotate("global\nminimum", xy=tuple(pmin), xytext=(1.95, 0.85),
                 fontsize=6.2, color="#b23a48", ha="center", va="bottom",
                 linespacing=1.25, zorder=9, bbox=halo,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#b23a48",
                                 shrinkA=1, shrinkB=4))
    ax2.text(-2.45, 0.55, "local\ntrap", fontsize=6.3, color="#3d3d3d",
             ha="center", va="center", zorder=9, style="italic",
             linespacing=1.25, bbox=halo)
    ax2.text(0.03, 0.035,
             f"final $L$:  {loss(*ours[-1]):.2f} (Ours)  vs. "
             f"{loss(*sgd[-1]):.2f} (SGD)", transform=ax2.transAxes,
             ha="left", va="bottom", fontsize=6.4, color="#1a1a1a", zorder=10,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                       lw=0.6, alpha=0.93))

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_xticks([-2, 0, 2])
    ax2.set_yticks([-2, 0, 2])
    ax2.set_xlabel(r"direction  $\delta_1$", fontsize=8, labelpad=1.5)
    ax2.set_ylabel(r"direction  $\delta_2$", fontsize=8, labelpad=0)
    ax2.set_title("Optimiser trajectories (top view)", fontsize=8.5, pad=3)
    add_panel_label(ax2, "(b)", x=-0.155, y=1.01, size=9.5)
    for s in ax2.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")

    cb = add_colorbar(fig, cf, ax2, label="loss  $L$  (-)", fraction=0.045,
                      pad=0.035, shrink=1.0, aspect=26)
    cb.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0])

    fig.legend(handles=[
        Line2D([], [], color=ORANGE, lw=1.3, label="SGD + momentum"),
        Line2D([], [], color=OURS, lw=2.0, label="Ours (adaptive)"),
        Line2D([], [], color="#333333", lw=0, marker="s", ms=4.4,
               mfc="white", mew=1.0, label="initialisation"),
        Line2D([], [], color="#b23a48", lw=0, marker="*", ms=8,
               label="global minimum")],
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.012),
        frameon=False, fontsize=7.2, handlelength=1.9, columnspacing=1.9)

    print(f"  final loss  SGD  {loss(*sgd[-1]):.4f} at "
          f"({sgd[-1][0]:+.2f}, {sgd[-1][1]:+.2f})")
    print(f"  final loss  Ours {loss(*ours[-1]):.4f} at "
          f"({ours[-1][0]:+.2f}, {ours[-1][1]:+.2f})")
    print(f"  true minimum {Z.min():.4f} at ({pmin[0]:+.2f}, {pmin[1]:+.2f})")
    savefig(fig, "307_loss_landscape.png")


if __name__ == "__main__":
    main()
