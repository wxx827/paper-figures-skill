"""Qualitative search-behaviour analysis of ISOA on the Rastrigin landscape.

The four panels follow the layout established by the SOA / GWO papers:
(a) the path traced by the first search agent over the 2-D contour map,
(b) the oscillation of its first coordinate, (c) the average fitness of the
whole population and (d) the decay of the mean pairwise distance, i.e. the
loss of population diversity.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pubstyle import set_pub_style, savefig, add_panel_label

RED = "#b23a48"
ITERS = 200
DOM = 5.12
N_AGENTS = 30


def rastrigin(x, y):
    return (x ** 2 - 10 * np.cos(2 * np.pi * x) + 10
            + y ** 2 - 10 * np.cos(2 * np.pi * y) + 10)


def simulate(seed=20250812):
    """Contracting random walk that mimics a converging swarm."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, ITERS)
    radius = DOM * (0.95 * np.exp(-4.0 * t) + 0.0025)

    # leader path: spirals in toward the global optimum at the origin
    theta = np.cumsum(rng.normal(0.52, 0.34, ITERS)) + 1.1
    wobble = np.convolve(rng.normal(0, 1, ITERS + 30),
                         np.ones(9) / 9.0, mode="same")[:ITERS]
    lx = radius * np.cos(theta) * (1.0 + 0.30 * wobble)
    ly = radius * np.sin(theta) * (1.0 + 0.30 * wobble)
    lx = np.clip(lx, -DOM, DOM)
    ly = np.clip(ly, -DOM, DOM)

    # whole population: agents scattered inside the shrinking radius
    ang = rng.uniform(0, 2 * np.pi, (ITERS, N_AGENTS))
    rad = radius[:, None] * np.sqrt(rng.uniform(0, 1, (ITERS, N_AGENTS)))
    px = np.clip(lx[:, None] * 0.35 + rad * np.cos(ang), -DOM, DOM)
    py = np.clip(ly[:, None] * 0.35 + rad * np.sin(ang), -DOM, DOM)

    raw = rastrigin(px, py).mean(axis=1)
    ker = np.ones(11) / 11.0
    sm = np.convolve(np.r_[np.full(5, raw[0]), raw, np.full(5, raw[-1])],
                     ker, mode="valid")
    avg_fit = np.minimum.accumulate(sm)
    # mean pairwise distance as the diversity proxy
    dist = np.sqrt(((px - px.mean(1, keepdims=True)) ** 2
                    + (py - py.mean(1, keepdims=True)) ** 2).mean(axis=1)) * 2
    return lx, ly, avg_fit, dist


def panel_a(ax, lx, ly):
    g = np.linspace(-DOM, DOM, 320)
    X, Y = np.meshgrid(g, g)
    Z = rastrigin(X, Y)
    ax.contourf(X, Y, Z, levels=34, cmap="Blues", alpha=0.42)
    ax.contour(X, Y, Z, levels=14, colors="#ffffff", linewidths=0.35,
               alpha=0.75)

    pts = np.stack([lx, ly], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.plot(lx, ly, color="#2b2b2b", lw=2.3, alpha=0.30, zorder=4,
            solid_capstyle="round")
    lc = LineCollection(segs, cmap="viridis", norm=Normalize(0, ITERS),
                        lw=1.35, alpha=0.98, zorder=5)
    lc.set_array(np.arange(ITERS - 1))
    line = ax.add_collection(lc)

    ax.plot(lx[0], ly[0], marker="s", ms=7, mfc="#ffffff", mec="#1a1a1a",
            mew=1.1, zorder=8)
    ax.plot(lx[-1], ly[-1], marker="o", ms=7, mfc="#111111", mec="#ffffff",
            mew=1.0, zorder=8)
    ax.plot(0, 0, marker="*", ms=15, mfc="#c9a227", mec="#1a1a1a", mew=0.9,
            zorder=9)
    ax.annotate("start", xy=(lx[0], ly[0]), xytext=(lx[0] - 1.7, ly[0] + 0.7),
                fontsize=6.8, color="#1a1a1a", zorder=10,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333"))
    ax.annotate("end", xy=(lx[-1], ly[-1]), xytext=(1.35, -1.55),
                fontsize=6.8, color="#1a1a1a", zorder=10,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#333333"))
    ax.annotate("global optimum $(0, 0)$", xy=(0, 0), xytext=(0.6, 3.4),
                fontsize=6.8, color="#1a1a1a", zorder=10,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))

    ax.set_xlim(-DOM, DOM)
    ax.set_ylim(-DOM, DOM)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("$x_2$", fontsize=8.4, labelpad=1.5)
    ax.tick_params(labelsize=7.2, length=2.4)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    ax.set_title("Search path of agent #1 on $F_4(x)$ Rastrigin",
                 fontsize=8.8, fontweight="bold", loc="left", pad=3.5)
    return line


def panel_b(ax, lx):
    it = np.arange(1, ITERS + 1)
    ax.axhline(0, color="#999999", lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.plot(it, lx, color=RED, lw=0.95, zorder=4)
    ax.fill_between(it, 0, lx, color=RED, alpha=0.13, lw=0, zorder=2)
    env = DOM * (0.95 * np.exp(-4.0 * (it - 1) / (ITERS - 1)) + 0.0025) * 1.32
    ax.plot(it, env, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=3)
    ax.plot(it, -env, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=3)
    ax.text(ITERS * 0.52, env[int(ITERS * 0.30)] * 1.15,
            "contracting envelope", fontsize=6.4, color="#555555")
    ax.set_xlim(1, ITERS)
    ax.set_ylim(-DOM * 1.12, DOM * 1.12)
    ax.set_xlabel("Iteration", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("$x_1$ of agent #1", fontsize=8.4, labelpad=1.5)
    ax.tick_params(labelsize=7.2, length=2.4)
    ax.grid(True, lw=0.4, color="#eeeeee", zorder=0)
    ax.set_title("Trajectory of the first dimension", fontsize=8.8,
                 fontweight="bold", loc="left", pad=3.5)


def panel_c(ax, avg_fit):
    it = np.arange(1, ITERS + 1)
    ax.plot(it, avg_fit, color=RED, lw=1.8, zorder=4)
    ax.fill_between(it, avg_fit.min() * 0.5, avg_fit, color=RED, alpha=0.12,
                    lw=0, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(avg_fit.min() * 0.55, avg_fit.max() * 1.9)
    ax.set_xlim(1, ITERS)
    ax.set_xlabel("Iteration", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("Average fitness of the swarm", fontsize=8.4, labelpad=1.5)
    ax.tick_params(labelsize=7.2, length=2.4)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, lw=0.4, color="#eeeeee", zorder=0)
    ax.annotate(f"final  {avg_fit[-1]:.3g}", xy=(ITERS, avg_fit[-1]),
                xytext=(ITERS * 0.60, avg_fit.min() * 3.0), fontsize=6.8,
                color="#333333",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#d5d5d5", lw=0.5))
    ax.set_title("Average fitness of the population", fontsize=8.8,
                 fontweight="bold", loc="left", pad=3.5)


def panel_d(ax, dist):
    it = np.arange(1, ITERS + 1)
    ax.plot(it, dist, color="#2e7d5b", lw=1.6, zorder=4)
    ax.fill_between(it, dist.min() * 0.5, dist, color="#2e7d5b", alpha=0.15,
                    lw=0, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(dist.min() * 0.6, dist.max() * 1.8)
    ax.set_xlim(1, ITERS)
    ax.set_xlabel("Iteration", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("Mean pairwise distance", fontsize=8.4, labelpad=1.5)
    ax.tick_params(labelsize=7.2, length=2.4)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, lw=0.4, color="#eeeeee", zorder=0)
    half = it[np.argmax(dist <= dist[0] / 2.0)]
    ax.axvline(half, color="#555555", lw=0.8, ls=(0, (2, 2)), zorder=3)
    ax.text(half + 4, dist.max() * 1.1,
            f"half-diversity\nat iter {half}", fontsize=6.5, va="top",
            color="#333333")
    ax.set_title("Population diversity (mean distance)", fontsize=8.8,
                 fontweight="bold", loc="left", pad=3.5)


def main():
    set_pub_style()
    lx, ly, avg_fit, dist = simulate()

    fig = plt.figure(figsize=(10.4, 8.4))
    gs = GridSpec(2, 2, figure=fig, left=0.068, right=0.955, top=0.895,
                  bottom=0.145, hspace=0.30, wspace=0.24)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    line = panel_a(axa, lx, ly)
    panel_b(axb, lx)
    panel_c(axc, avg_fit)
    panel_d(axd, dist)
    for ax, lab in zip((axa, axb, axc, axd), ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, lab, x=-0.115, y=1.02, size=11)

    cax = fig.add_axes([0.115, 0.082, 0.300, 0.012])
    cb = fig.colorbar(line, cax=cax, orientation="horizontal")
    cb.set_label("Iteration index along the search path of panel (a)",
                 fontsize=7.2, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.6, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "148_search_trajectory.png")


if __name__ == "__main__":
    main()
