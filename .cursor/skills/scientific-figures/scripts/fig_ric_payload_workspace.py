"""Reachable workspace of a 6-DoF arm, a mid-plane slice, and a payload curve.

Main panel: XY projection of reachable tool poses at three payload masses,
equal aspect.  Side: a vertical slice through the same cloud, and the
reachable volume against payload from the same kinematic samples.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RED, BLUE, GOLD = "#b23a48", "#3d6b8e", "#c9a227"
SEED = 1065
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)
L = np.array([0.35, 0.40, 0.32])  # link lengths, m


def fk(q):
    x = (L[0] * np.cos(q[:, 0])
         + L[1] * np.cos(q[:, 0] + q[:, 1])
         + L[2] * np.cos(q[:, 0] + q[:, 1] + q[:, 2]))
    z = (L[0] * np.sin(q[:, 0])
         + L[1] * np.sin(q[:, 0] + q[:, 1])
         + L[2] * np.sin(q[:, 0] + q[:, 1] + q[:, 2]))
    y = 0.18 * np.sin(q[:, 3])
    return np.column_stack([x, y, z])


def sample(rng, n, payload):
    span = max(0.28, 0.95 - 0.055 * payload)
    q = rng.uniform(-span, span, size=(n, 6)) * np.array([1.1, 1.0, 1.0, 0.8, 0.6, 0.4])
    return fk(q)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    masses = [0.0, 4.0, 8.0]
    cols = [BLUE, GOLD, RED]
    clouds = [sample(rng, 1800, m) for m in masses]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX_T)
    ax_v = axes_inches(fig, AUX_B)

    for pts, col, m in zip(clouds, cols, masses):
        ax.scatter(pts[:, 0], pts[:, 1], s=3.5, color=col, alpha=0.22,
                   linewidths=0, rasterized=True, zorder=3,
                   label=f"{m:.0f} kg")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Reachable XY cloud vs payload", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.legend(loc="lower left", frameon=False, fontsize=7.2, markerscale=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    def aabb_l(p):
        return 1e3 * np.prod(np.ptp(p, axis=0))

    ax.text(0.97, 0.96,
            f"V(0 kg) = {aabb_l(clouds[0]):.0f} L\n"
            f"V(8 kg) = {aabb_l(clouds[2]):.0f} L",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    for pts, col in zip(clouds, cols):
        sl = np.abs(pts[:, 1]) < 0.04
        ax_s.scatter(pts[sl, 0], pts[sl, 2], s=4.0, color=col, alpha=0.35,
                     linewidths=0, rasterized=True)
    ax_s.set_xlabel("x (m)")
    ax_s.set_ylabel("z (m)")
    ax_s.set_title("Mid-plane slice $|y|$ < 40 mm", fontsize=8.5,
                   fontweight="bold", pad=5)
    ax_s.set_aspect("equal")
    for side in ("top", "right"):
        ax_s.spines[side].set_visible(False)

    pay = np.linspace(0, 10, 9)
    vol = []
    for m in pay:
        p = sample(rng, 1400, m)
        vol.append(np.prod(np.ptp(p, axis=0)))
    vol = np.array(vol)
    ax_v.plot(pay, vol * 1e3, color=BLUE, marker="o", ms=4.5, lw=1.5)
    ax_v.set_xlabel("payload (kg)")
    ax_v.set_ylabel("axis-aligned volume (L)")
    ax_v.set_title("Envelope volume vs payload", fontsize=8.5,
                   fontweight="bold", pad=5)
    ax_v.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_v.spines[side].set_visible(False)

    panel_labels_inches(fig, [ax, ax_s, ax_v], dx=-0.62, dy=0.14)
    savefig(fig, "1065_ric_payload_workspace.png")


if __name__ == "__main__":
    main()
