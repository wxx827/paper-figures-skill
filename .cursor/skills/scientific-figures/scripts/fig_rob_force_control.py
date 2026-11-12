"""Impedance-controlled contact task: one time base, three coupled channels.

The three stacked rows are the same run seen through position, force and
controller gains, so the reader can follow one event - the tool touching the
surface at 1.4 s, the gain switch at 3.0 s, the release at 4.6 s - straight
down the page.  Phase bands are drawn on every row so the eye does not have
to re-find the boundaries.  The side panel is not a second experiment: it
re-plots the contact samples of the same run as force against penetration, so
the identified environment stiffness is a slope through points that are
already on the page.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"        # measured / commanded robot signal
GREEN = "#2e7d5b"       # target, reference, ground truth
RED = "#b23a48"         # limit violation, peak
GREY = "#808080"
BAND = "#f0f0f0"

T_END = 6.0
DT = 0.002
T_TOUCH, T_SWITCH, T_RELEASE = 1.4, 3.0, 4.6
F_STIFF, F_SOFT = 12.0, 6.0        # target normal force, N
F_TOL = 1.2                        # tolerance band, N
K_ENV = 8.0                        # environment stiffness, N/mm
KP = (2.5, 0.8)                    # impedance stiffness, N/mm
DAMP = (45.0, 130.0)               # impedance damping, N s/m


def step(t, t0, amp, tau, zeta):
    """Underdamped second-order step, zero before ``t0``."""
    d = np.maximum(t - t0, 0.0)
    wn = 1.0 / tau
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 1e-6))
    env = np.exp(-zeta * wn * d)
    y = 1.0 - env * (np.cos(wd * d)
                     + zeta / np.sqrt(max(1.0 - zeta ** 2, 1e-6))
                     * np.sin(wd * d))
    return amp * np.where(t >= t0, y, 0.0)


def run(seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T_END + DT, DT)

    force = (step(t, T_TOUCH, F_STIFF, 0.085, 0.34)
             + step(t, T_SWITCH, F_SOFT - F_STIFF, 0.150, 0.80)
             + step(t, T_RELEASE, -F_SOFT, 0.110, 0.90))
    noise = rng.normal(0.0, 1.0, t.size)
    kern = np.exp(-0.5 * (np.arange(-30, 31) / 9.0) ** 2)
    noise = np.convolve(noise, kern / kern.sum(), mode="same")
    amp = np.where(t < T_SWITCH, 0.30, 0.11) * (t > T_TOUCH)
    force = np.clip(force + amp * noise, 0.0, None)

    # Penetration is not force / stiffness alone: the surface recovers with a
    # lag, so unloading follows a shifted path and the F-d curve is a loop.
    runmax = np.maximum.accumulate(force)
    resid = (0.38 * (runmax - force) / K_ENV
             * np.exp(-np.maximum(t - T_RELEASE, 0.0) / 0.45))
    pen = force / K_ENV + resid                           # penetration, mm
    approach = 40.0 * (1.0 - np.clip((t - 0.30) / 1.10, 0.0, 1.0))
    lift = 30.0 * np.clip((t - 4.90) / 0.90, 0.0, 1.0)
    z = np.where(t < T_TOUCH, approach, -pen + lift)

    z_cmd = np.where(t < T_TOUCH, approach,
                     -np.where(t < T_SWITCH, F_STIFF,
                               np.where(t < T_RELEASE, F_SOFT, 0.0)) / K_ENV
                     + lift)
    target = np.where(t < T_TOUCH, 0.0,
                      np.where(t < T_SWITCH, F_STIFF,
                               np.where(t < T_RELEASE, F_SOFT, 0.0)))
    return t, z, z_cmd, force, pen, target


def phase_bands(ax, label=False):
    for x0, x1 in [(T_TOUCH, T_SWITCH), (T_SWITCH, T_RELEASE)]:
        ax.axvspan(x0, x1, facecolor=BAND, edgecolor="none", zorder=0)
    if label:
        top = ax.get_ylim()[1]
        for x, s in [(T_TOUCH / 2.0, "free space"),
                     ((T_TOUCH + T_SWITCH) / 2.0, "contact, stiff"),
                     ((T_SWITCH + T_RELEASE) / 2.0, "contact, compliant"),
                     ((T_RELEASE + T_END) / 2.0, "retract")]:
            ax.text(x, top * 0.99, s, ha="center", va="top", fontsize=6.6,
                    color=SUB)


def pos_panel(ax, t, z, z_cmd):
    phase_bands(ax)
    ax.axhline(0.0, color=GREY, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
    ax.plot(t, z_cmd, color=GREEN, lw=1.0, ls=(0, (4, 2.4)), zorder=3,
            label="commanded $z_d$")
    ax.plot(t, z, color=BLUE, lw=1.5, zorder=4, label="measured $z$")
    ax.set_ylim(-4.0, 46.0)
    ax.set_ylabel("TCP height $z$ (mm)")
    ax.text(0.10, 2.0, "surface", fontsize=6.5, color=GREY, va="bottom")
    ax.legend(fontsize=7.0, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.84), handlelength=1.6,
              borderaxespad=0.2)
    phase_bands(ax, label=True)
    ax.set_title("End-effector approach, dwell and retract", fontsize=8.5,
                 fontweight="bold", color=INK)


def force_panel(ax, t, force, target):
    phase_bands(ax)
    hold = (t > T_TOUCH) & (t < T_RELEASE)
    ax.fill_between(t, target - F_TOL, target + F_TOL,
                    where=hold, color=GREEN, alpha=0.16, linewidth=0.0,
                    zorder=2)
    ax.plot(t, target, color=GREEN, lw=1.0, ls=(0, (4, 2.4)), zorder=3)
    ax.plot(t, force, color=BLUE, lw=1.2, zorder=4)

    out = hold & (np.abs(force - target) > F_TOL)
    ax.plot(t[out], force[out], ls="none", marker="o", ms=1.4, color=RED,
            zorder=5)
    ax.text(T_RELEASE - 0.12, F_SOFT - F_TOL - 0.6,
            r"target $\pm$ " f"{F_TOL:.1f} N tolerance", ha="right",
            va="top", fontsize=6.8, color=GREEN)

    ipk = int(np.argmax(force))
    peak = float(force[ipk])
    ax.annotate(f"peak {peak:.1f} N,  "
                f"{(peak / F_STIFF - 1.0) * 100:.0f} % overshoot",
                xy=(t[ipk], peak), xytext=(t[ipk] + 0.30, peak + 3.4),
                fontsize=6.8, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_ylim(-1.5, 21.0)
    ax.set_ylabel("contact force $F_z$ (N)")
    ax.set_title("Force tracking through the gain switch", fontsize=8.5,
                 fontweight="bold", color=INK)
    return peak, float(out.sum()) / float(max(hold.sum(), 1))


def gain_panel(ax, t):
    phase_bands(ax)
    kp = np.where(t < T_SWITCH, KP[0], KP[1])
    dd = np.where(t < T_SWITCH, DAMP[0], DAMP[1])
    ax.step(t, kp, where="post", color=BLUE, lw=1.5, zorder=4)
    ax.set_ylim(0.0, 3.6)
    ax.set_ylabel("stiffness $K_p$ (N/mm)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)

    axd = ax.twinx()
    axd.spines["top"].set_visible(False)
    axd.step(t, dd, where="post", color=GREEN, lw=1.5, ls=(0, (5, 2.2)),
             zorder=4)
    axd.set_ylim(0.0, 190.0)
    axd.set_ylabel(r"damping $D$ (N$\cdot$s/m)", color=GREEN)
    axd.tick_params(axis="y", colors=GREEN, labelsize=7.0)

    ax.annotate(f"gain switch: $K_p$ {KP[0]:.1f} " r"$\rightarrow$ "
                f"{KP[1]:.1f} N/mm", xy=(T_SWITCH, KP[1] + 0.15),
                xytext=(T_SWITCH + 0.22, 2.6), fontsize=6.8, color=INK,
                ha="left", arrowprops=dict(arrowstyle="-", color=INK,
                                           lw=0.7))
    ax.set_xlabel("time $t$ (s)")
    ax.set_xlim(0.0, T_END)


def fd_panel(ax, t, pen, force):
    load = (t >= T_TOUCH) & (t < T_SWITCH)
    rel = (t >= T_SWITCH) & (t < T_RELEASE + 0.9)
    ax.plot(pen[load], force[load], color=BLUE, lw=0.9, alpha=0.85,
            label="press")
    ax.plot(pen[rel], force[rel], color=GREEN, lw=0.9, alpha=0.85,
            label="release")

    # Stiffness is a slope through the origin: an unconstrained line would
    # buy a better residual with a physically meaningless offset force.
    fit = (t >= T_TOUCH) & (t < T_RELEASE)
    k = float(np.sum(force[fit] * pen[fit]) / np.sum(pen[fit] ** 2))
    xs = np.linspace(0.0, float(pen.max()) * 1.05, 40)
    ax.plot(xs, k * xs, color=RED, lw=1.2, ls=(0, (4, 2.2)),
            label=f"fit  {k:.2f} N/mm")

    ax.set_xlabel("penetration depth $d$ (mm)", fontsize=7.4)
    ax.set_ylabel("contact force $F_z$ (N)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0.0, float(pen.max()) * 1.08)
    ax.set_ylim(0.0, 21.0)
    ax.grid(lw=0.45, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              handlelength=1.5, borderaxespad=0.25)
    ax.set_title("Stiffness identification", fontsize=8.2,
                 fontweight="bold", color=INK)
    return k


def main():
    set_pub_style()
    t, z, z_cmd, force, pen, target = run()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0, 0.62], wspace=0.28,
                          hspace=0.22)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0], sharex=ax_p)
    ax_g = fig.add_subplot(gs[2, 0], sharex=ax_p)
    ax_d = fig.add_subplot(gs[0:2, 1])

    for ax in (ax_p, ax_f):
        ax.tick_params(labelbottom=False)
    for ax in (ax_p, ax_f, ax_g):
        ax.set_axisbelow(True)
        ax.grid(axis="y", lw=0.45, alpha=0.65)

    pos_panel(ax_p, t, z, z_cmd)
    peak, out_frac = force_panel(ax_f, t, force, target)
    gain_panel(ax_g, t)
    k_hat = fd_panel(ax_d, t, pen, force)

    hold = (t > T_TOUCH + 0.35) & (t < T_RELEASE)
    rms = float(np.sqrt(np.mean((force[hold] - target[hold]) ** 2)))

    handles = [Patch(facecolor=BAND, edgecolor="none", label="contact phase"),
               Line2D([], [], color=BLUE, lw=1.5, label="measured"),
               Line2D([], [], color=GREEN, lw=1.2, ls=(0, (4, 2.2)),
                      label="target / reference"),
               Line2D([], [], color=RED, lw=1.2, label="out of tolerance")]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.985, 0.075), frameon=False, fontsize=7.0,
               handlelength=1.6, labelspacing=0.45)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=1.4, w_pad=2.4)
    panel_labels_inches(fig, [ax_p, ax_f, ax_g, ax_d], dx=-0.44, dy=0.06)
    savefig(fig, "584_rob_force_control.png")


if __name__ == "__main__":
    main()
