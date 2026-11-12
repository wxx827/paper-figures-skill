"""Benchmark landscape + convergence board for a swarm-intelligence study.

Four rows by two macro-columns give eight benchmark blocks. Every block pairs
the 3-D landscape of the test function (surface floating above a filled
contour plane, own vertical colourbar) with the averaged convergence curves of
five metaheuristics. Two blocks carry a zoom inset on the first ten iterations
linked to the main axes by a magenta callout arrow.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from pubstyle import set_pub_style, savefig

# ---- algorithm identity ----------------------------------------------------
ALGOS = ["ISOA", "SOA", "PSO", "SSA", "GWO"]
COLORS = {"ISOA": "#b23a48", "SOA": "#3b5b92", "PSO": "#2e7d5b",
          "SSA": "#8e6bbf", "GWO": "#e0812f"}
LSTYLE = {"ISOA": "-", "SOA": "-", "PSO": (0, (5, 2)), "SSA": (0, (6, 2, 1, 2)),
          "GWO": (0, (1.4, 1.6))}
LWIDTH = {"ISOA": 2.0, "SOA": 1.25, "PSO": 1.25, "SSA": 1.25, "GWO": 1.45}
KIND = {"ISOA": "isoa", "SOA": "soa", "PSO": "pso", "SSA": "ssa", "GWO": "gwo"}

ITERS = 200
N_AVG = 12
PINK = "#b23a48"


# ---- benchmark functions ---------------------------------------------------
def f_sphere(x, y):
    return x ** 2 + y ** 2


def f_schwefel222(x, y):
    return np.abs(x) + np.abs(y) + np.abs(x) * np.abs(y)


def f_rosenbrock(x, y):
    return 100.0 * (y - x ** 2) ** 2 + (x - 1.0) ** 2


def f_rastrigin(x, y):
    return (x ** 2 - 10 * np.cos(2 * np.pi * x) + 10
            + y ** 2 - 10 * np.cos(2 * np.pi * y) + 10)


def f_ackley(x, y):
    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + 20.0 + np.e)


def f_griewank(x, y):
    return (x ** 2 + y ** 2) / 4000.0 - np.cos(x) * np.cos(y / np.sqrt(2)) + 1.0


def f_schaffer(x, y):
    r2 = x ** 2 + y ** 2
    return 0.5 + (np.sin(np.sqrt(r2)) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2


def f_dropwave(x, y):
    r2 = x ** 2 + y ** 2
    return -(1.0 + np.cos(12.0 * np.sqrt(r2))) / (0.5 * r2 + 2.0)


FUNCS = [
    dict(key="F1", name="Sphere", fn=f_sphere, vdom=(-5, 5),
         dom="[-100, 100]", fmin=0.0, gstart=3.0e4, mode="Unimodal"),
    dict(key="F2", name="Schwefel 2.22", fn=f_schwefel222, vdom=(-10, 10),
         dom="[-10, 10]", fmin=0.0, gstart=7.5e1, mode="Unimodal"),
    dict(key="F3", name="Rosenbrock", fn=f_rosenbrock, vdom=(-2, 2),
         dom="[-30, 30]", fmin=0.0, gstart=4.2e6, mode="Unimodal"),
    dict(key="F4", name="Rastrigin", fn=f_rastrigin, vdom=(-5.12, 5.12),
         dom="[-5.12, 5.12]", fmin=0.0, gstart=3.4e2, mode="Multimodal"),
    dict(key="F5", name="Ackley", fn=f_ackley, vdom=(-32, 32),
         dom="[-32, 32]", fmin=0.0, gstart=2.05e1, mode="Multimodal"),
    dict(key="F6", name="Griewank", fn=f_griewank, vdom=(-6, 6),
         dom="[-600, 600]", fmin=0.0, gstart=2.8e2, mode="Multimodal"),
    dict(key="F7", name="Schaffer", fn=f_schaffer, vdom=(-10, 10),
         dom="[-100, 100]", fmin=0.0, gstart=9.7e-1, mode="Fixed-dim"),
    dict(key="F8", name="Drop-wave", fn=f_dropwave, vdom=(-5.12, 5.12),
         dom="[-5.12, 5.12]", fmin=-1.0, gstart=8.6e-1, mode="Fixed-dim"),
]
BY_KEY = {f["key"]: f for f in FUNCS}
KEYS = [f["key"] for f in FUNCS]

# Final error above the true optimum. ISOA leads on six of eight problems;
# GWO edges it out on Rosenbrock and PSO on the 2-D Schaffer function.
ENDS = {
    "F1": {"ISOA": 2.1e-95, "SOA": 5.6e-42, "PSO": 3.4e-9,
           "SSA": 8.8e-24, "GWO": 1.7e-50},
    "F2": {"ISOA": 3.7e-52, "SOA": 1.2e-26, "PSO": 6.1e-5,
           "SSA": 4.4e-14, "GWO": 9.3e-30},
    "F3": {"ISOA": 2.67e+1, "SOA": 2.84e+1, "PSO": 4.13e+1,
           "SSA": 2.91e+1, "GWO": 2.55e+1},
    "F4": {"ISOA": 2.4e-13, "SOA": 7.6e-6, "PSO": 3.9e+1,
           "SSA": 1.1e+1, "GWO": 3.4e+0},
    "F5": {"ISOA": 8.9e-16, "SOA": 4.4e-14, "PSO": 2.3e+0,
           "SSA": 1.9e-7, "GWO": 1.5e-13},
    "F6": {"ISOA": 3.3e-16, "SOA": 2.7e-3, "PSO": 1.9e-2,
           "SSA": 1.4e-8, "GWO": 4.2e-3},
    "F7": {"ISOA": 3.6e-16, "SOA": 9.7e-11, "PSO": 1.2e-16,
           "SSA": 5.2e-12, "GWO": 3.1e-9},
    "F8": {"ISOA": 4.1e-16, "SOA": 2.2e-11, "PSO": 6.3e-12,
           "SSA": 8.7e-10, "GWO": 1.5e-8},
}
SIGMA = {"F1": 3.00, "F2": 2.20, "F3": 0.020, "F4": 1.80,
         "F5": 0.22, "F6": 2.00, "F7": 0.28, "F8": 0.33}
# Drop-wave has a negative optimum, so its trace is interpolated linearly in
# fitness space; the concentric rings trap PSO/GWO short of the true minimum.
LINEAR = {"F8"}
ENDS["F8"] = {"ISOA": 1.0e-5, "SOA": 2.2e-4, "PSO": 6.4e-2,
              "SSA": 1.45e-2, "GWO": 3.2e-2}


def _progress(rng, T, kind):
    """Monotone 0 -> 1 progress profile with algorithm-specific character."""
    t = np.arange(T, dtype=float) / max(T - 1, 1)
    if kind == "isoa":
        w = np.exp(-3.0 * t) + 0.30 + 0.30 * np.exp(-((t - 0.45) / 0.12) ** 2)
        mask = rng.random(T) < 0.85
    elif kind == "soa":
        w = np.exp(-4.5 * t) + 0.05
        mask = rng.random(T) < 0.60
    elif kind == "pso":
        w = np.exp(-9.0 * t) + 0.004
        mask = rng.random(T) < 0.50
    elif kind == "ssa":
        w = np.exp(-3.0 * t) + 0.10
        mask = rng.random(T) < 0.12          # sparse jumps -> staircase
    else:
        w = np.exp(-2.0 * t) + 0.18
        mask = rng.random(T) < 0.75
    inc = w * rng.gamma(1.6, 1.0, T) * mask
    inc[0] = 0.0
    p = np.cumsum(inc)
    return p / max(p[-1], 1e-12)


def curve(fkey, algo, run=0, iters=ITERS):
    """Best-so-far fitness trace; monotone non-increasing by construction."""
    rng = np.random.default_rng(
        9001 + KEYS.index(fkey) * 977 + ALGOS.index(algo) * 131 + run * 7919)
    p = _progress(rng, iters, KIND[algo])
    g0 = BY_KEY[fkey]["gstart"] * 10 ** rng.normal(0, 0.06)
    g1 = ENDS[fkey][algo] * 10 ** rng.normal(0, SIGMA[fkey])
    g1 = min(g1, g0 * 1e-2)
    if fkey in LINEAR:
        g = g0 + p * (g1 - g0)
    else:
        g = 10 ** (np.log10(g0) + p * (np.log10(g1) - np.log10(g0)))
    return g + BY_KEY[fkey]["fmin"]


def mean_curve(fkey, algo, iters=ITERS, n=N_AVG):
    """Average over independent runs (stays monotone non-increasing)."""
    fmin = BY_KEY[fkey]["fmin"]
    errs = [curve(fkey, algo, r, iters) - fmin for r in range(n)]
    if fkey in LINEAR:
        return np.mean(errs, axis=0) + fmin
    return 10 ** np.mean([np.log10(e) for e in errs], axis=0) + fmin


# ---- drawing ---------------------------------------------------------------
def draw_surface(fig, ax, cax, spec):
    lo, hi = spec["vdom"]
    g = np.linspace(lo, hi, 190)
    X, Y = np.meshgrid(g, g)
    Z = spec["fn"](X, Y)
    zlo, zhi = float(Z.min()), float(Z.max())
    span = zhi - zlo or 1.0
    floor = zlo - 0.52 * span

    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", rstride=2, cstride=2,
                           linewidth=0.08, edgecolor=(0, 0, 0, 0.12),
                           antialiased=True, shade=True)
    ax.contourf(X, Y, Z, zdir="z", offset=floor, levels=28, cmap="coolwarm",
                alpha=0.9)
    ax.set_zlim(floor, zhi + 0.05 * span)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.view_init(elev=34, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 0.74), zoom=1.04)
    except TypeError:
        ax.set_box_aspect((1, 1, 0.74))

    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor("#d0d0d0")
        pane.pane.set_alpha(1.0)
        pane._axinfo["grid"]["color"] = "#e4e4e4"
        pane._axinfo["grid"]["linewidth"] = 0.45
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("$x_1$", fontsize=7.5, labelpad=-2)
    ax.set_ylabel("$x_2$", fontsize=7.5, labelpad=-2)
    ax.set_zlabel("")                 # named on the colourbar instead
    ax.tick_params(axis="x", labelsize=6.0, pad=-2.6)
    ax.tick_params(axis="y", labelsize=6.0, pad=-2.6)
    ax.tick_params(axis="z", labelsize=6.0, pad=1.0)
    ax.text2D(-0.05, 1.02, f"${spec['key'][0]}_{spec['key'][1]}(x)$",
              transform=ax.transAxes, fontsize=10.5, fontweight="bold",
              color="#1a1a1a")
    ax.text2D(-0.05, 0.945, f"{spec['name']}  |  {spec['mode']}",
              transform=ax.transAxes, fontsize=6.8, color="#5a5a5a")
    ax.text2D(-0.05, 0.015, f"$x_i \\in$ {spec['dom']}",
              transform=ax.transAxes, fontsize=6.4, color="#5a5a5a")

    cb = fig.colorbar(surf, cax=cax)
    cb.ax.tick_params(labelsize=6.0, length=1.8, width=0.6, pad=1.4)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#555555")
    cb.locator = plt.MaxNLocator(5)
    cb.update_ticks()
    cb.set_label("$f(x_1, x_2)$", fontsize=7.0, labelpad=2.0)


def draw_convergence(ax, spec, inset_mode=None):
    fkey = spec["key"]
    it = np.arange(1, ITERS + 1)
    traces = {a: mean_curve(fkey, a) for a in ALGOS}
    for a in ALGOS:
        ax.plot(it, traces[a], color=COLORS[a], ls=LSTYLE[a], lw=LWIDTH[a],
                label=a, zorder=6 if a == "ISOA" else 4,
                solid_capstyle="round")

    logy = spec["fmin"] == 0.0
    if logy:
        ax.set_yscale("log")
        lo = min(t.min() for t in traces.values())
        hi = max(t.max() for t in traces.values())
        ax.set_ylim(lo * 10 ** -1.2, hi * 10 ** 0.8)
        ax.yaxis.set_major_locator(
            plt.LogLocator(base=10, numticks=7))
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_ylim(-1.035, -0.02)
    ax.set_xlim(0, ITERS)
    ax.set_xlabel("Iteration", fontsize=8, labelpad=1.5)
    ax.set_ylabel("Fitness value", fontsize=8, labelpad=1.5)
    ax.tick_params(labelsize=6.8, length=2.4)
    ax.grid(True, which="major", lw=0.4, color="#e6e6e6", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.2, ncol=1, frameon=True,
              framealpha=0.9, edgecolor="#cccccc", borderpad=0.28,
              labelspacing=0.24, handlelength=1.7, handletextpad=0.45)
    ax.set_title(f"Convergence on ${fkey[0]}_{fkey[1]}(x)$", fontsize=8,
                 fontweight="bold", loc="left", pad=3)

    if inset_mode is None:
        return
    # ---- zoom on the first ten iterations, linked by a magenta arrow -----
    zx = 10
    axi = ax.inset_axes([0.40, 0.40, 0.33, 0.29])
    for a in ALGOS:
        axi.plot(it[:zx], traces[a][:zx], color=COLORS[a], ls=LSTYLE[a],
                 lw=LWIDTH[a] * 0.85, zorder=6 if a == "ISOA" else 4)
    if logy:
        axi.set_yscale("log")
    axi.set_xlim(1, zx)
    axi.tick_params(labelsize=6.0, length=1.6, pad=1.0)
    axi.set_title("first 10 iterations", fontsize=6.0, pad=1.6,
                  color="#333333")
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.9)
        s.set_edgecolor(PINK)
    axi.set_facecolor("#fffafc")
    axi.grid(False)

    # zoom box drawn in data coordinates so it really frames the magnified part
    seg = np.array([traces[a][:zx] for a in ALGOS])
    ylo, yhi = float(seg.min()), float(seg.max())
    if logy:
        ylo, yhi = ylo * 0.35, yhi * 2.6
    axi.set_ylim(ylo, yhi)
    ax.add_patch(Rectangle((0.0, ylo), zx * 1.6, yhi - ylo,
                           facecolor="none", edgecolor=PINK, lw=1.2,
                           zorder=7))
    arr = FancyArrowPatch(posA=(0.395, 0.60), posB=(0.085, 0.80),
                          transform=ax.transAxes, arrowstyle="-|>",
                          mutation_scale=17, lw=2.6, color=PINK,
                          shrinkA=3, shrinkB=3, zorder=9,
                          connectionstyle="arc3,rad=-0.24")
    ax.add_patch(arr)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(13.0, 15.4))
    gs = GridSpec(4, 6, figure=fig,
                  width_ratios=[1.34, 0.035, 1.00, 1.34, 0.035, 1.00],
                  left=0.030, right=0.985, top=0.945, bottom=0.048,
                  hspace=0.42, wspace=0.40)

    insets = {"F1", "F5"}
    for i, spec in enumerate(FUNCS):
        r, c = i // 2, (i % 2) * 3
        ax3 = fig.add_subplot(gs[r, c], projection="3d")
        cax = fig.add_subplot(gs[r, c + 1])
        pos = cax.get_position()
        cax.set_position([pos.x0 - 0.014, pos.y0 + pos.height * 0.26,
                          pos.width, pos.height * 0.48])
        draw_surface(fig, ax3, cax, spec)
        ax2 = fig.add_subplot(gs[r, c + 2])
        draw_convergence(ax2, spec,
                         inset_mode=True if spec["key"] in insets else None)

    savefig(fig, "143_benchmark_convergence.png")


if __name__ == "__main__":
    main()
