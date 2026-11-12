"""Contouring error of a planar toolpath: reference, tracking, and error CDF.

Main panel: commanded square-with-fillets path and the tracked path at two
feed rates, equal aspect.  Side: error vs path length, and the CDF of the
absolute contouring error.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RED, BLUE, GOLD = "#b23a48", "#3d6b8e", "#c9a227"
SEED = 1067
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def square_path(n=800, r=0.012):
    s = np.linspace(0, 4.0, n, endpoint=False)
    x = np.empty(n)
    y = np.empty(n)
    side = 0.18
    for i, si in enumerate(s):
        k = int(si)
        u = si - k
        if k == 0:
            x[i], y[i] = -side + 2 * side * u, -side
        elif k == 1:
            x[i], y[i] = side, -side + 2 * side * u
        elif k == 2:
            x[i], y[i] = side - 2 * side * u, side
        else:
            x[i], y[i] = -side, side - 2 * side * u
    return x, y


def track(x, y, lag, rng):
    dx, dy = np.gradient(x), np.gradient(y)
    nrm = np.hypot(dx, dy) + 1e-12
    nx, ny = -dy / nrm, dx / nrm
    e = lag + rng.normal(0, 0.35 * lag, x.size)
    return x + nx * e, y + ny * e, np.abs(e)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    x, y = square_path()
    s = np.cumsum(np.hypot(np.gradient(x), np.gradient(y)))
    s -= s[0]
    x1, y1, e1 = track(x, y, 0.0018, rng)
    x2, y2, e2 = track(x, y, 0.0046, rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_e = axes_inches(fig, AUX_T)
    ax_c = axes_inches(fig, AUX_B)

    ax.plot(x, y, color=INK, lw=1.5, label="command")
    ax.plot(x1, y1, color=BLUE, lw=1.2, label="120 mm/s")
    ax.plot(x2, y2, color=RED, lw=1.2, label="300 mm/s")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Planar contour, two feed rates", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.03, 0.06, f"mean $|e|$ 120 mm/s = {1e3 * e1.mean():.2f} mm\n"
            f"mean $|e|$ 300 mm/s = {1e3 * e2.mean():.2f} mm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    ax_e.plot(s * 1e3, e1 * 1e3, color=BLUE, lw=1.1)
    ax_e.plot(s * 1e3, e2 * 1e3, color=RED, lw=1.1)
    ax_e.set_xlabel("path length (mm)")
    ax_e.set_ylabel("|contour error| (mm)")
    ax_e.set_title("Error along the path", fontsize=8.5, fontweight="bold",
                   pad=5)
    ax_e.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_e.spines[side].set_visible(False)

    def cdf(a):
        v = np.sort(a)
        return v, np.arange(1, v.size + 1) / v.size

    v1, c1 = cdf(e1 * 1e3)
    v2, c2 = cdf(e2 * 1e3)
    ax_c.plot(v1, c1, color=BLUE, lw=1.5, label="120 mm/s")
    ax_c.plot(v2, c2, color=RED, lw=1.5, label="300 mm/s")
    ax_c.set_xlabel("|contour error| (mm)")
    ax_c.set_ylabel("CDF")
    ax_c.set_title("Error CDF", fontsize=8.5, fontweight="bold", pad=5)
    ax_c.legend(loc="lower right", frameon=False, fontsize=6.8)
    ax_c.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    panel_labels_inches(fig, [ax, ax_e, ax_c], dx=-0.62, dy=0.14)
    savefig(fig, "1067_ric_contour_error.png")


if __name__ == "__main__":
    main()
