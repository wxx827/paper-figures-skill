"""Eight snapshots of the ISOA population collapsing onto the global optimum.

The 2x4 grid freezes the swarm at iterations 1, 5, 10, 20, 50, 100, 150 and
200 on the Rastrigin landscape. Agents are drawn over the filled contour map
and coloured by their own fitness, the incumbent best is marked with a red
star, and each panel reports the iteration, the incumbent fitness and the
mean spread of the population. Once the swarm has collapsed, a magnified
inset keeps the residual structure readable.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pubstyle import set_pub_style, savefig

DOM = 5.12
ITERS = 200
N_AGENTS = 40
SNAPS = [1, 5, 10, 20, 50, 100, 150, 200]
ZOOM_FROM = 150                     # panels at or after this get an inset
CMAP_BG = "Blues"
CMAP_PT = "YlOrRd"


def rastrigin(x, y):
    return (x ** 2 - 10 * np.cos(2 * np.pi * x) + 10
            + y ** 2 - 10 * np.cos(2 * np.pi * y) + 10)


def simulate(seed=31415):
    """Population positions per iteration for a converging swarm."""
    rng = np.random.default_rng(seed)
    t = np.arange(ITERS) / (ITERS - 1)
    radius = DOM * (0.98 * np.exp(-3.6 * t) + 0.004)
    c0 = np.array([-1.55, 1.15])
    centre = c0[None, :] * np.exp(-3.0 * t)[:, None]
    centre += np.stack([
        np.convolve(rng.normal(0, 1, ITERS + 20), np.ones(21) / 21.0,
                    mode="same")[:ITERS] * radius * 0.40,
        np.convolve(rng.normal(0, 1, ITERS + 20), np.ones(21) / 21.0,
                    mode="same")[:ITERS] * radius * 0.40], axis=1)

    ang = rng.uniform(0, 2 * np.pi, (ITERS, N_AGENTS))
    rad = radius[:, None] * np.sqrt(rng.uniform(0, 1, (ITERS, N_AGENTS)))
    px = np.clip(centre[:, 0:1] + rad * np.cos(ang), -DOM, DOM)
    py = np.clip(centre[:, 1:2] + rad * np.sin(ang), -DOM, DOM)

    fit = rastrigin(px, py)
    idx = fit.argmin(axis=1)
    raw_best = fit[np.arange(ITERS), idx]
    best = np.minimum.accumulate(raw_best)
    keep = np.maximum.accumulate(np.where(raw_best <= best, np.arange(ITERS),
                                          -1))
    bx = px[np.arange(ITERS), idx][keep]
    by = py[np.arange(ITERS), idx][keep]
    return px, py, fit, best, bx, by, radius


def panel(ax, X, Y, Z, px, py, fit, best, bx, by, it, norm, label):
    ax.contourf(X, Y, Z, levels=26, cmap=CMAP_BG, alpha=0.42)
    ax.contour(X, Y, Z, levels=10, colors="#ffffff", linewidths=0.3,
               alpha=0.75)
    ax.plot(0, 0, marker="+", ms=9, color="#222222", mew=1.1, zorder=4)

    spread = float(np.hypot(px - px.mean(), py - py.mean()).mean())
    ax.add_patch(Circle((px.mean(), py.mean()), max(spread, 0.02),
                        facecolor="none", edgecolor="#444444", lw=0.7,
                        ls=(0, (3, 2)), zorder=5))
    sc = ax.scatter(px, py, c=fit, cmap=CMAP_PT, norm=norm, s=18,
                    edgecolors="#33333366", linewidths=0.35, zorder=6)
    ax.plot(bx, by, marker="*", ms=14, mfc="#b23a48", mec="#ffffff",
            mew=0.9, zorder=8)

    ax.set_xlim(-DOM, DOM)
    ax.set_ylim(-DOM, DOM)
    ax.set_aspect("equal")
    ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_yticks([-5, -2.5, 0, 2.5, 5])
    ax.tick_params(labelsize=6.2, length=2.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    ax.set_title(f"{label}  iteration {it}", fontsize=8.4, fontweight="bold",
                 loc="left", pad=3.0)
    ax.text(0.035, 0.035, f"best $f$ = {best:.3g}\nspread = {spread:.3g}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
            color="#1a1a1a", linespacing=1.4, zorder=9,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                      edgecolor="#d0d0d0", lw=0.5, alpha=0.92))

    # ---- magnified inset once the population has collapsed ---------------
    if it >= ZOOM_FROM:
        cx, cy = (float(px.mean()) + bx) / 2.0, (float(py.mean()) + by) / 2.0
        reach = np.hypot(np.r_[px, bx] - cx, np.r_[py, by] - cy).max()
        half = max(reach * 1.30, 0.05)
        axi = ax.inset_axes([0.575, 0.575, 0.395, 0.395])
        gz = np.linspace(-half, half, 90)
        Xz, Yz = np.meshgrid(gz + cx, gz + cy)
        axi.contourf(Xz, Yz, rastrigin(Xz, Yz), levels=18, cmap=CMAP_BG,
                     alpha=0.42)
        axi.scatter(px, py, c=fit, cmap=CMAP_PT, norm=norm, s=13,
                    edgecolors="#33333366", linewidths=0.3)
        axi.plot(bx, by, marker="*", ms=11, mfc="#b23a48", mec="#ffffff",
                 mew=0.8)
        axi.plot(0, 0, marker="+", ms=7, color="#222222", mew=1.0)
        axi.set_xlim(cx - half, cx + half)
        axi.set_ylim(cy - half, cy + half)
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.9)
            s.set_edgecolor("#b23a48")
        axi.set_title(f"{DOM / half:.0f}$\\times$ zoom", fontsize=6.0,
                      pad=1.6, color="#b02020")
    return sc


def main():
    set_pub_style()
    px, py, fit, best, bx, by, radius = simulate()

    g = np.linspace(-DOM, DOM, 340)
    X, Y = np.meshgrid(g, g)
    Z = rastrigin(X, Y)
    norm = Normalize(vmin=0.0, vmax=float(np.percentile(Z, 99)))

    fig = plt.figure(figsize=(13.6, 7.6))
    gs = GridSpec(2, 4, figure=fig, left=0.040, right=0.918, top=0.858,
                  bottom=0.092, hspace=0.28, wspace=0.20)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    sc = None
    for i, it in enumerate(SNAPS):
        k = it - 1
        ax = fig.add_subplot(gs[i // 4, i % 4])
        sc = panel(ax, X, Y, Z, px[k], py[k], fit[k], best[k], bx[k], by[k],
                   it, norm, labels[i])
        if i % 4 == 0:
            ax.set_ylabel("$x_2$", fontsize=8.2, labelpad=1.5)
        if i // 4 == 1:
            ax.set_xlabel("$x_1$", fontsize=8.2, labelpad=1.5)

    cax = fig.add_axes([0.932, 0.16, 0.012, 0.62])
    cb = fig.colorbar(sc, cax=cax, extend="max")
    cb.set_label("Fitness of the individual,  $F_4(x)$ Rastrigin",
                 fontsize=7.6, labelpad=3)
    cb.ax.tick_params(labelsize=6.6, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    handles = [
        Line2D([], [], marker="*", ls="none", mfc="#b23a48", mec="white",
               mew=0.8, ms=11, label="incumbent best"),
        Line2D([], [], marker="+", ls="none", color="#222222", mew=1.1,
               ms=8, label="true global optimum $(0, 0)$"),
        Line2D([], [], color="#444444", lw=0.9, ls=(0, (3, 2)),
               label="mean spread of the population"),
    ]

    savefig(fig, "149_swarm_snapshots.png")


if __name__ == "__main__":
    main()
