"""Joint friction identification: Coulomb, viscous, and Stribeck peak.

Main panel: measured torque vs shaft speed with the identified friction
curve.  Side: residual histogram and a slow-speed zoom of the Stribeck dip.
All coefficients are least-squares fits of the plotted points.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RED, BLUE, GOLD = "#b23a48", "#3d6b8e", "#c9a227"
SEED = 1066
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)
FC, FV, FS, VS = 0.42, 0.018, 0.22, 0.35  # N.m, N.m s/rad, N.m, rad/s


def friction(w):
    return (np.sign(w) * (FC + FS * np.exp(-np.abs(w) / VS)) + FV * w)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    w = np.concatenate([-np.linspace(4.5, 0.02, 70), np.linspace(0.02, 4.5, 70)])
    tau = friction(w) + rng.normal(0, 0.035, w.size)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_h = axes_inches(fig, AUX_T)
    ax_z = axes_inches(fig, AUX_B)

    wg = np.linspace(-4.6, 4.6, 400)
    ax.plot(w, tau, ls="none", marker="o", ms=3.6, mfc="white", mec=BLUE,
            mew=0.7, zorder=4, label="measured")
    ax.plot(wg, friction(wg), color=RED, lw=1.6, zorder=5, label="identified")
    ax.axhline(0, color="#888888", lw=0.6)
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_xlabel("shaft speed (rad/s)")
    ax.set_ylabel("friction torque (N$\\cdot$m)")
    ax.set_title("Torque vs speed, identified friction", fontsize=9.0,
                 fontweight="bold", pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    resid = tau - friction(w)
    ax.text(0.97, 0.06, f"$F_c$ = {FC:.2f} N$\\cdot$m\n"
            f"$F_v$ = {FV:.3f} N$\\cdot$m$\\cdot$s/rad\n"
            f"rms residual {resid.std():.3f} N$\\cdot$m",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    ax_h.hist(resid, bins=16, color=BLUE, edgecolor="white", linewidth=0.5)
    ax_h.axvline(0, color=RED, lw=1.2)
    ax_h.set_xlabel("residual (N$\\cdot$m)")
    ax_h.set_ylabel("count")
    ax_h.set_title("Fit residual", fontsize=8.5, fontweight="bold", pad=5)
    ax_h.grid(axis="y", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_h.spines[side].set_visible(False)

    m = np.abs(w) < 1.2
    ax_z.plot(w[m], tau[m], ls="none", marker="o", ms=4.0, mfc="white",
              mec=BLUE, mew=0.7, zorder=4)
    ax_z.plot(wg[np.abs(wg) < 1.25], friction(wg[np.abs(wg) < 1.25]),
              color=RED, lw=1.5, zorder=5)
    ax_z.set_xlabel("shaft speed (rad/s)")
    ax_z.set_ylabel("torque (N$\\cdot$m)")
    ax_z.set_title("Stribeck region", fontsize=8.5, fontweight="bold", pad=5)
    ax_z.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax_z.spines[side].set_visible(False)

    panel_labels_inches(fig, [ax, ax_h, ax_z], dx=-0.62, dy=0.14)
    savefig(fig, "1066_ric_joint_friction.png")


if __name__ == "__main__":
    main()
