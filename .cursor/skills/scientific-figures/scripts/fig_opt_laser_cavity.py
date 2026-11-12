"""A Q-switched solid-state cavity: layout, mode comb and threshold.

The main panel is the claim: everything the laser does is set by the round
trip between two mirrors - the standing wave that fits inside it, the comb of
frequencies that fit inside it, and the pump level at which round-trip gain
finally beats round-trip loss.  The schematic is drawn on an inch canvas so
that every gap and radius is a paper distance, and the intracavity intensity
trace underneath shares its horizontal axis with the layout above it.  The two
smaller panels carry the consequences: the longitudinal mode comb whose
spacing is computed from the cavity length drawn above, and a measured
input-output curve whose threshold and slope efficiency come from a fit.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

from pubstyle import set_pub_style, savefig
from figkit import inch_axes, axes_inches, panel_card, panel_labels_inches

RED = "#b23a48"        # the laser field
BLUE = "#3b5b92"       # gain medium
GOLD = "#c9a227"       # pump and modulator (loss elements)
DARK = "#3a3a3a"       # mirrors and mounts
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

C_MS = 2.998e8
L_CAV_MM = 300.0        # mirror separation
N_ROD = 1.82            # refractive index of the gain rod
ROD_MM = 60.0
LAM_NM = 1064.0

MW, MH = 5.30, 4.85     # main panel, inches
X_HR, X_OC = 0.62, 4.72
Y_AX = 3.45


def optical_length_mm():
    return L_CAV_MM + (N_ROD - 1.0) * ROD_MM


def fsr_mhz():
    return C_MS / (2.0 * optical_length_mm() * 1e-3) / 1e6


def mode_radius(x, w0=0.088, z_r=1.15):
    return w0 * np.sqrt(1.0 + ((x - 0.5 * (X_HR + X_OC)) / z_r) ** 2)


def cavity_panel(ax):
    z = np.linspace(X_HR, X_OC, 900)
    w = mode_radius(z)

    ax.fill_between(z, Y_AX - w, Y_AX + w, color=RED, alpha=0.16,
                    linewidth=0.0, zorder=3)
    ax.plot(z, Y_AX + w, color=RED, lw=0.9, zorder=4)
    ax.plot(z, Y_AX - w, color=RED, lw=0.9, zorder=4)
    stand = np.sin(14.0 * np.pi * (z - X_HR) / (X_OC - X_HR))
    ax.plot(z, Y_AX + 0.80 * w * stand, color=RED, lw=0.7, alpha=0.85,
            zorder=5)

    ax.add_patch(Arc((X_HR + 0.55, Y_AX), 1.20, 1.28, angle=0.0,
                     theta1=160.0, theta2=200.0, color=DARK, lw=2.4,
                     zorder=6))
    ax.add_patch(Rectangle((X_OC, Y_AX - 0.32), 0.055, 0.64,
                           facecolor=DARK, edgecolor="none", zorder=6))
    ax.text(X_HR - 0.02, Y_AX + 0.46, "HR mirror\n$R$ > 99.9 %", ha="left",
            va="bottom", fontsize=6.8, color=DARK, linespacing=1.35)
    ax.text(X_OC + 0.05, Y_AX + 0.46, "output\ncoupler\n$T$ = 12 %",
            ha="right", va="bottom", fontsize=6.8, color=DARK,
            linespacing=1.35)

    ax.add_patch(Rectangle((1.42, Y_AX - 0.24), 0.95, 0.48, facecolor=BLUE,
                           alpha=0.30, edgecolor=BLUE, lw=1.0, zorder=6))
    ax.text(1.895, Y_AX - 0.34, f"Nd:YAG rod\n{ROD_MM:.0f} mm, "
            + r"$n$ = " + f"{N_ROD:.2f}", ha="center", va="top",
            fontsize=6.8, color=BLUE, linespacing=1.35)
    for xp in (1.60, 1.895, 2.19):
        ax.annotate("", xy=(xp, Y_AX + 0.26), xytext=(xp, Y_AX + 0.86),
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.0,
                                    mutation_scale=8), zorder=6)
    ax.text(1.895, Y_AX + 0.94, "808 nm diode pump", ha="center",
            va="bottom", fontsize=6.8, color=GOLD)

    for s in (+1.0, -1.0):
        ax.add_patch(Rectangle((2.74, Y_AX + s * 0.155), 0.05, s * 0.30,
                               facecolor=DARK, edgecolor="none", zorder=6))
    ax.text(2.765, Y_AX + 0.50, "aperture", ha="center", va="bottom",
            fontsize=6.8, color=SUB)

    ax.add_patch(Rectangle((3.22, Y_AX - 0.26), 0.42, 0.52, facecolor=GOLD,
                           alpha=0.32, edgecolor=GOLD, lw=1.0, zorder=6))
    ax.text(3.43, Y_AX - 0.36, "Q-switch", ha="center", va="top",
            fontsize=6.8, color=GOLD)

    ax.annotate("", xy=(X_OC + 0.52, Y_AX), xytext=(X_OC + 0.08, Y_AX),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                mutation_scale=11), zorder=7)
    ax.text(X_OC + 0.30, Y_AX - 0.14, "pulsed\noutput", ha="center",
            va="top", fontsize=6.8, color=RED, linespacing=1.35)

    y_dim = 2.62
    ax.annotate("", xy=(X_OC, y_dim), xytext=(X_HR, y_dim),
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.8,
                                mutation_scale=7), zorder=6)
    for xv in (X_HR, X_OC):
        ax.plot([xv, xv], [y_dim - 0.07, y_dim + 0.07], color=SUB, lw=0.8,
                zorder=6)
    ax.text(0.5 * (X_HR + X_OC), y_dim - 0.09,
            f"cavity length $L$ = {L_CAV_MM:.0f} mm,  optical length "
            f"{optical_length_mm():.0f} mm", ha="center", va="top",
            fontsize=6.8, color=SUB)

    card = panel_card(ax, 0.5 * (X_HR + X_OC) + 0.02, 1.28,
                      X_OC - X_HR + 0.30, 1.52, fc="#f5f5f5", ec=HAIR,
                      lw=0.7, zorder=1, rounding=0.05)
    env = (mode_radius(0.5 * (X_HR + X_OC)) / w) ** 2
    inten = env * stand ** 2
    base = card["y"] - 0.44
    ax.plot(z, base + 0.86 * inten, color=RED, lw=1.0, zorder=4)
    ax.fill_between(z, base, base + 0.86 * inten, color=RED, alpha=0.16,
                    linewidth=0.0, zorder=3)
    ax.plot([X_HR, X_OC], [base, base], color=GREY, lw=0.7, zorder=3)
    ax.text(X_HR - 0.06, card["y"] + 0.66,
            "intracavity standing-wave intensity (-),  node spacing "
            r"exaggerated; true spacing is $\lambda/2$ = "
            f"{LAM_NM / 2.0:.0f} nm", ha="left", va="top", fontsize=6.5,
            color=SUB)

    ax.set_title(f"Round trip of {2 * optical_length_mm():.0f} mm sets both "
                 "the standing wave and the mode comb", fontsize=9.0,
                 fontweight="bold", color=INK, pad=5.0)


def mode_panel(ax):
    fsr = fsr_mhz()
    nu = np.linspace(-2.6, 2.6, 800)                # GHz from line centre
    gain = np.exp(-0.5 * (nu / 0.92) ** 2)
    loss = 0.42
    orders = np.arange(-int(2.6e3 / fsr), int(2.6e3 / fsr) + 1)
    nu_m = orders * fsr / 1e3
    g_m = np.exp(-0.5 * (nu_m / 0.92) ** 2)
    lase = g_m >= loss

    ax.plot(nu, gain, color=BLUE, lw=1.3, zorder=3)
    ax.fill_between(nu, 0.0, gain, color=BLUE, alpha=0.10, linewidth=0.0,
                    zorder=2)
    ax.axhline(loss, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=3)
    ax.vlines(nu_m[~lase], 0.0, g_m[~lase], color=GREY, lw=1.0, zorder=4)
    ax.vlines(nu_m[lase], 0.0, g_m[lase], color=RED, lw=1.6, zorder=5)
    ax.plot(nu_m[lase], g_m[lase], ls="none", marker="o", ms=2.6, mfc="white",
            mec=RED, mew=0.8, zorder=6)

    ax.annotate("", xy=(nu_m[len(nu_m) // 2 + 1], 1.06),
                xytext=(nu_m[len(nu_m) // 2], 1.06),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8,
                                mutation_scale=6), zorder=6)
    ax.text(nu_m[len(nu_m) // 2] + 0.03, 1.10,
            r"$\Delta\nu$ = " + f"{fsr:.0f} MHz", ha="left", va="bottom",
            fontsize=6.6, color=INK)
    ax.text(-2.5, loss + 0.03, "round-trip loss", ha="left", va="bottom",
            fontsize=6.4, color=SUB)
    ax.text(2.55, 1.40, f"{int(lase.sum())} modes above threshold\n"
            r"single-mode $\delta\nu$ = 1.2 MHz", ha="right", va="top",
            fontsize=6.6, color=RED, linespacing=1.4)

    ax.set_xlabel(r"detuning from line centre (GHz)", fontsize=7.4)
    ax.set_ylabel("relative gain (-)", fontsize=7.4)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(0.0, 1.44)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Longitudinal mode comb under the gain curve", fontsize=8.2,
                 fontweight="bold", color=INK)
    return fsr, int(lase.sum())


def threshold_panel(ax, seed=11):
    rng = np.random.default_rng(seed)
    p_th, eta = 2.35, 0.415
    pump = np.linspace(0.0, 12.0, 25)
    out = np.where(pump > p_th, eta * (pump - p_th), 0.0)
    meas = out + rng.normal(0.0, 0.045, pump.size)
    meas[pump < p_th] = np.abs(rng.normal(0.0, 0.018, (pump < p_th).sum()))

    above = pump > p_th + 0.6
    slope, icept = np.polyfit(pump[above], meas[above], 1)
    p_fit = -icept / slope
    line = np.linspace(p_fit, pump[-1], 60)

    ax.plot(line, slope * line + icept, color=RED, lw=1.5, zorder=3)
    ax.plot(pump, meas, ls="none", marker="o", ms=3.2, mfc="white", mec=BLUE,
            mew=0.9, zorder=4)
    ax.axvline(p_fit, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=2)
    ax.plot([p_fit], [0.0], marker="o", ms=4.0, mfc="white", mec=RED,
            mew=1.1, zorder=5)
    ax.annotate(f"threshold {p_fit:.2f} W", xy=(p_fit, 0.0),
                xytext=(p_fit + 0.45, 2.85), ha="left", fontsize=6.8,
                color=RED, arrowprops=dict(arrowstyle="-", color=RED,
                                           lw=0.7))
    ax.text(11.8, 0.30, f"slope efficiency {slope * 100:.1f} %", ha="right",
            va="bottom", fontsize=6.8, color=INK)

    ax.set_xlabel("diode pump power (W)", fontsize=7.4)
    ax.set_ylabel("output power (W)", fontsize=7.4)
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(-0.12, 4.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Threshold and slope efficiency", fontsize=8.2,
                 fontweight="bold", color=INK)
    return p_fit, slope


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    ax_cav = inch_axes(fig, (0.42, 0.55, MW, MH))
    ax_mode = axes_inches(fig, (6.62, 3.60, 3.10, 1.62))
    ax_thr = axes_inches(fig, (6.62, 0.92, 3.10, 1.62))

    cavity_panel(ax_cav)
    fsr, n_lase = mode_panel(ax_mode)
    p_th, slope = threshold_panel(ax_thr)

    panel_labels_inches(fig, [ax_cav, ax_mode, ax_thr], dx=-0.30, dy=0.14)
    savefig(fig, "700_opt_laser_cavity.png")


if __name__ == "__main__":
    main()
