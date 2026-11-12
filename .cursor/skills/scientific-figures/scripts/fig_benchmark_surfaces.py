"""Gallery of the eight two-dimensional benchmark landscapes.

A 2x4 grid of 3-D surfaces rendered from one common viewpoint, each floating
above its own filled contour projection. Because the eight functions span very
different magnitudes, the shared colourbar encodes the min-max normalised
objective so that a single key applies to every panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from pubstyle import set_pub_style, savefig

CMAP = "viridis"
ELEV, AZIM = 35, -60
N = 220


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
         dom="[-100,\\,100]", fmin="0", mode="Unimodal", xopt=(0, 0)),
    dict(key="F2", name="Schwefel 2.22", fn=f_schwefel222, vdom=(-10, 10),
         dom="[-10,\\,10]", fmin="0", mode="Unimodal", xopt=(0, 0)),
    dict(key="F3", name="Rosenbrock", fn=f_rosenbrock, vdom=(-2, 2),
         dom="[-30,\\,30]", fmin="0", mode="Unimodal", xopt=(1, 1)),
    dict(key="F4", name="Rastrigin", fn=f_rastrigin, vdom=(-5.12, 5.12),
         dom="[-5.12,\\,5.12]", fmin="0", mode="Multimodal", xopt=(0, 0)),
    dict(key="F5", name="Ackley", fn=f_ackley, vdom=(-32, 32),
         dom="[-32,\\,32]", fmin="0", mode="Multimodal", xopt=(0, 0)),
    dict(key="F6", name="Griewank", fn=f_griewank, vdom=(-6, 6),
         dom="[-600,\\,600]", fmin="0", mode="Multimodal", xopt=(0, 0)),
    dict(key="F7", name="Schaffer", fn=f_schaffer, vdom=(-10, 10),
         dom="[-100,\\,100]", fmin="0", mode="Fixed-dim", xopt=(0, 0)),
    dict(key="F8", name="Drop-wave", fn=f_dropwave, vdom=(-5.12, 5.12),
         dom="[-5.12,\\,5.12]", fmin="-1", mode="Fixed-dim", xopt=(0, 0)),
]

MODE_COLOR = {"Unimodal": "#3b5b92", "Multimodal": "#c0603f",
              "Fixed-dim": "#2e7d5b"}


def panel(fig, cell, spec, label):
    ax = fig.add_subplot(cell, projection="3d")
    lo, hi = spec["vdom"]
    g = np.linspace(lo, hi, N)
    X, Y = np.meshgrid(g, g)
    Z = spec["fn"](X, Y)
    zlo, zhi = float(Z.min()), float(Z.max())
    Zn = (Z - zlo) / (zhi - zlo or 1.0)
    floor = -0.55

    ax.plot_surface(Zn * 0 + X, Y, Zn, facecolors=plt.get_cmap(CMAP)(Zn),
                    rstride=2, cstride=2, linewidth=0.06,
                    edgecolor=(0.15, 0.15, 0.15, 0.10), antialiased=True,
                    shade=False)
    ax.contourf(X, Y, Zn, zdir="z", offset=floor, levels=32, cmap=CMAP,
                alpha=0.92)
    ax.contour(X, Y, Zn, zdir="z", offset=floor, levels=10,
               colors="#ffffff", linewidths=0.28, alpha=0.55)
    ax.scatter([spec["xopt"][0]], [spec["xopt"][1]], [floor], marker="*",
               s=55, c="white", edgecolors="#1a1a1a", linewidths=0.7,
               depthshade=False, zorder=12)

    ax.set_zlim(floor, 1.05)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.view_init(elev=ELEV, azim=AZIM)
    try:
        ax.set_box_aspect((1, 1, 0.80), zoom=1.16)
    except TypeError:
        ax.set_box_aspect((1, 1, 0.80))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#d5d5d5")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = "#e8e8e8"
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.set_xlabel("$x_1$", fontsize=7.6, labelpad=-3)
    ax.set_ylabel("$x_2$", fontsize=7.6, labelpad=-3)
    ax.tick_params(axis="x", labelsize=6.0, pad=-2.6)
    ax.tick_params(axis="y", labelsize=6.0, pad=-2.6)
    ax.tick_params(axis="z", labelsize=6.0, pad=0.6)

    ax.text2D(-0.03, 1.12, f"{label}  ${spec['key'][0]}_{spec['key'][1]}(x)$"
              f"  {spec['name']}", transform=ax.transAxes, fontsize=8.6,
              fontweight="bold", color="#1a1a1a")
    ax.text2D(1.02, 1.125, f"$f \\in [{zlo:.3g},\\,{zhi:.3g}]$",
              transform=ax.transAxes, fontsize=6.4, color="#8a8a8a",
              ha="right")
    ax.text2D(-0.03, 1.038, spec["mode"], transform=ax.transAxes,
              fontsize=6.8, color=MODE_COLOR[spec["mode"]], fontweight="bold")
    ax.text2D(1.02, 1.038,
              f"$x_i \\in {spec['dom']}$   $f^* = {spec['fmin']}$",
              transform=ax.transAxes, fontsize=6.6, color="#5a5a5a",
              ha="right")
    return ax


def main():
    set_pub_style()
    fig = plt.figure(figsize=(14.2, 7.6))
    gs = GridSpec(2, 4, figure=fig, left=0.012, right=0.925, top=0.880,
                  bottom=0.082, hspace=0.34, wspace=0.10)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for i, spec in enumerate(FUNCS):
        panel(fig, gs[i // 4, i % 4], spec, labels[i])

    cax = fig.add_axes([0.945, 0.18, 0.012, 0.60])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Normalised objective  $(f - f_{\\min}) / "
                 "(f_{\\max} - f_{\\min})$", fontsize=7.6, labelpad=4)
    cb.ax.tick_params(labelsize=6.6, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "144_benchmark_surfaces.png")


if __name__ == "__main__":
    main()
