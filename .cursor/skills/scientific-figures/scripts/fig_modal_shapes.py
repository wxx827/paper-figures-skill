"""2x3 grid of the first six bending modes of a clamped-free steel beam.

Every panel shows the undeformed beam axis (grey dashed), the vibration
envelope, the mirrored half-cycle and the mode shape itself drawn as a
line whose colour encodes the local normalised displacement.  Interior
nodes are circled, the clamped root is hatched, and each panel is captioned
with the mode order and its natural frequency in Hz.  A single diverging
colorbar is shared by the whole grid.

Physics: exact Euler-Bernoulli cantilever eigenfunctions,
phi_n(x) = cosh(b x) - cos(b x) - sigma_n [sinh(b x) - sin(b x)] with
cos(bL) cosh(bL) = -1 (roots found numerically) and
f_n = (bL)^2 / (2 pi L^2) sqrt(E I / rho A) for a 1000 x 30 x 6 mm steel
beam, so the frequencies and the node positions are the real ones.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.optimize import brentq
from pubstyle import set_pub_style, savefig

LEN = 1.000        # beam length [m]
WID, THK = 0.030, 0.006
E = 2.10e11        # Young's modulus [Pa]
RHO = 7850.0       # density [kg/m^3]
NMODES = 6
CMAP = "coolwarm"


def beta_roots(n):
    """Roots of cos(bL) cosh(bL) + 1 = 0."""
    out = []
    k = 1
    while len(out) < n:
        guess = (2 * k - 1) * np.pi / 2.0
        lo, hi = guess - 0.45, guess + 0.45
        try:
            out.append(brentq(lambda z: np.cos(z) * np.cosh(z) + 1.0,
                              lo, hi, xtol=1e-13))
        except ValueError:
            pass
        k += 1
    return np.array(out)


def mode_shape(bl, xi):
    b = bl
    sig = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
    p = (np.cosh(b * xi) - np.cos(b * xi)
         - sig * (np.sinh(b * xi) - np.sin(b * xi)))
    return p / np.max(np.abs(p))


def main():
    set_pub_style(base_font=8)
    ii = WID * THK ** 3 / 12.0
    ra = RHO * WID * THK
    c = np.sqrt(E * ii / ra)
    bls = beta_roots(NMODES)
    freqs = bls ** 2 / (2.0 * np.pi * LEN ** 2) * c

    xi = np.linspace(0.0, 1.0, 900)
    norm = Normalize(-1.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.05), sharex=True,
                             sharey=True)
    lc = None
    for k, ax in enumerate(axes.ravel()):
        y = mode_shape(bls[k], xi)
        ax.add_patch(Rectangle((-0.075, -1.35), 0.075, 2.70,
                               facecolor="#e2e5e8", edgecolor="#5f6870",
                               lw=0.7, hatch="////", zorder=2))
        ax.axhline(0.0, color="#9aa3ad", lw=0.9, ls=(0, (4, 2)), zorder=3)
        ax.fill_between(xi, -np.abs(y), np.abs(y), color="#c9ced3",
                        alpha=0.22, lw=0, zorder=1)
        ax.plot(xi, -y, color="#aeb4bc", lw=0.8, ls=(0, (3, 2)), zorder=4)

        # pale halo keeps the near-zero (white) part of the ramp visible
        ax.plot(xi, y, color="#c2c8cd", lw=3.4, solid_capstyle="round",
                zorder=5)
        pts = np.array([xi, y]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=CMAP, norm=norm, zorder=6)
        lc.set_array(0.5 * (y[:-1] + y[1:]))
        lc.set_linewidth(2.3)
        lc.set_capstyle("round")
        ax.add_collection(lc)

        s = np.sign(y)
        nodes = xi[:-1][(s[:-1] * s[1:] < 0)]
        nodes = nodes[nodes > 0.02]
        ax.plot(nodes, np.zeros_like(nodes), ls="none", marker="o", ms=4.0,
                mfc="white", mec="#333333", mew=0.8, zorder=7)

        ax.set_title(r"$\bf{Mode\ %d}$    $f_%d$ = %.1f Hz"
                     % (k + 1, k + 1, freqs[k]), fontsize=7.5,
                     color="#111418", pad=3)
        ax.text(0.115, 0.045, "%d node%s" % (nodes.size,
                                             "" if nodes.size == 1 else "s"),
                transform=ax.transAxes, fontsize=6.3, color="#6b7280",
                ha="left", va="bottom", zorder=9)

        ax.set_xlim(-0.085, 1.03)
        ax.set_ylim(-1.42, 1.42)
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, axis="x", lw=0.4, color="#eeeeee")
        ax.tick_params(length=2.4)
        ax.text(-0.037, -1.30, "clamped", fontsize=6.0, color="#4a5259",
                rotation=90, ha="center", va="bottom", zorder=9)
        ax.text(0.008, 1.06, "(%s)" % "abcdef"[k], transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for ax in axes[1, :]:
        ax.set_xlabel(r"axial position  $x / L$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\phi_n$  (norm.)")

    fig.subplots_adjust(left=0.062, right=0.985, top=0.925, bottom=0.185,
                        wspace=0.13, hspace=0.30)
    cax = fig.add_axes([0.325, 0.055, 0.36, 0.028])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("normalised modal displacement  $\\phi_n$  (-)",
                 fontsize=7.2, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "100_modal_shapes.png")


if __name__ == "__main__":
    main()
