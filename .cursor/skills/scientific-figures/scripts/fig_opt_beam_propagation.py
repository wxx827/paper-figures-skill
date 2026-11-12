"""Gaussian beam propagation: waist, Rayleigh range and beam quality.

The main panel is the claim: a real beam expands along a hyperbola whose two
free parameters - the waist radius and the far-field divergence - are locked
together by the wavelength, so measuring one fixes the other.  Peak-normalised
transverse profiles are laid over the envelope at three stations to show what
the hyperbola actually means at a plane.  The two smaller panels carry the
evidence: an ISO-style caustic fit that returns the beam propagation factor,
and the envelopes the same waist would produce at three values of that factor.

Every annotated number - waist, Rayleigh range, divergence, fitted factor - is
computed from the arrays that are plotted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

RED = "#b23a48"        # the beam itself (laser convention)
BLUE = "#3b5b92"       # measured samples, transverse profiles
GOLD = "#c9a227"       # degraded beam quality
GREY = "#808080"       # ideal / reference
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

LAM_UM = 1.064          # wavelength in um (Nd:YAG fundamental)
W0_UM = 120.0           # waist radius in um
Z_MM = 150.0            # half span of the propagation axis in mm
PROFILE_Z = (-108.0, 0.0, 108.0)


def w_of_z(z_mm, m2=1.0, w0=W0_UM):
    """Beam radius in um at axial position ``z_mm`` (mm) for factor ``m2``."""
    z_r = np.pi * w0 ** 2 / LAM_UM          # Rayleigh range in um
    return w0 * np.sqrt(1.0 + (m2 * z_mm * 1e3 / z_r) ** 2)


def caustic_samples(seed=7, m2_true=1.35, n=13):
    rng = np.random.default_rng(seed)
    z = np.linspace(-Z_MM, Z_MM, n)
    w = w_of_z(z, m2=m2_true, w0=W0_UM * 1.04)
    return z, w * (1.0 + rng.normal(0.0, 0.017, z.size))


def fit_caustic(z, w):
    """ISO 11146 style parabolic fit of w^2(z); returns (w0, m2, curve)."""
    a, b, c = np.polyfit(z, w ** 2, 2)
    z_fit = np.linspace(-Z_MM, Z_MM, 400)
    w_fit = np.sqrt(np.maximum(a * z_fit ** 2 + b * z_fit + c, 1.0))
    w0 = float(w_fit.min())
    m2 = float(np.sqrt(a) * np.pi * w0 / (LAM_UM * 1e3))
    return w0, m2, z_fit, w_fit


def main_panel(ax):
    z = np.linspace(-Z_MM, Z_MM, 600)
    w = w_of_z(z)
    z_r = np.pi * W0_UM ** 2 / LAM_UM / 1e3          # mm
    theta = float((w[-1] - w[-2]) / (z[-1] - z[-2]))  # um/mm = mrad

    ax.fill_between(z, -w, w, color=RED, alpha=0.16, linewidth=0.0, zorder=2)
    ax.plot(z, w, color=RED, lw=1.7, zorder=4)
    ax.plot(z, -w, color=RED, lw=1.7, zorder=4)
    ax.axhline(0.0, color=RED, lw=0.7, ls=(0, (5, 3)), alpha=0.55, zorder=3)

    ax.plot(z, theta * z, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=3)
    ax.plot(z, -theta * z, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=3)

    y = np.linspace(-470.0, 470.0, 500)
    for k, zk in enumerate(PROFILE_Z):
        wk = w_of_z(zk)
        prof = np.exp(-2.0 * (y / wk) ** 2)
        ax.plot([zk, zk], [-470.0, 470.0], color=BLUE, lw=0.55,
                ls=(0, (2, 2.2)), alpha=0.65, zorder=4)
        ax.fill_betweenx(y, zk, zk + 27.0 * prof, color=BLUE, alpha=0.20,
                         linewidth=0.0, zorder=5)
        ax.plot(zk + 27.0 * prof, y, color=BLUE, lw=1.0, zorder=6)
        ax.text(zk + 3.0, 442.0, f"$w$ = {wk:.0f} " + r"$\mathrm{\mu m}$",
                ha="left", va="top", fontsize=6.4, color=BLUE)

    ax.annotate("", xy=(0.0, W0_UM), xytext=(0.0, -W0_UM),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8,
                                mutation_scale=7), zorder=7)
    ax.text(-6.0, 0.0, r"$2w_0$ = " + f"{2 * W0_UM:.0f} " + r"$\mathrm{\mu m}$",
            ha="right", va="center", fontsize=7.0, color=INK, zorder=7)

    ax.annotate("", xy=(z_r, -300.0), xytext=(0.0, -300.0),
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.8,
                                mutation_scale=7), zorder=7)
    ax.text(z_r / 2.0, -318.0, r"$z_R$ = " + f"{z_r:.1f} mm", ha="center",
            va="top", fontsize=7.0, color=SUB, zorder=7)

    ax.annotate(r"asymptote  $\theta$ = " + f"{theta:.2f} mrad",
                xy=(Z_MM * 0.76, theta * Z_MM * 0.76),
                xytext=(Z_MM * 0.16, 388.0), ha="left", fontsize=7.2,
                color=SUB, zorder=7,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))

    ax.set_xlabel("axial position $z$ (mm)")
    ax.set_ylabel(r"transverse coordinate $r$ ($\mathrm{\mu m}$)")
    ax.set_xlim(-Z_MM, Z_MM)
    ax.set_ylim(-470.0, 470.0)
    ax.set_yticks([-400, -200, 0, 200, 400])
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title(f"One waist of {W0_UM:.0f} " + r"$\mathrm{\mu m}$"
                 + f" fixes both $z_R$ = {z_r:.0f} mm and "
                 + r"$\theta$ = " + f"{theta:.2f} mrad", fontsize=9.0,
                 fontweight="bold", color=INK)
    return z_r, theta


def caustic_panel(ax):
    z, w = caustic_samples()
    w0, m2, z_fit, w_fit = fit_caustic(z, w)

    ax.plot(z_fit, w_fit ** 2 / 1e3, color=RED, lw=1.5, zorder=3,
            label=r"parabolic fit")
    ax.plot(z, w ** 2 / 1e3, ls="none", marker="o", ms=3.2, mfc="white",
            mec=BLUE, mew=0.9, zorder=4, label="measured planes")
    ax.plot(z_fit, w_of_z(z_fit) ** 2 / 1e3, color=GREY, lw=0.9,
            ls=(0, (4, 2.6)), zorder=2, label=r"ideal $M^2$ = 1")

    ax.text(0.035, 0.94, r"$M^2$ = " + f"{m2:.2f}\n"
            + r"$w_0$ = " + f"{w0:.0f} " + r"$\mathrm{\mu m}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=INK, linespacing=1.5)

    ax.set_xlabel("axial position $z$ (mm)", fontsize=7.4)
    ax.set_ylabel(r"$w^2$ ($10^3\ \mathrm{\mu m}^2$)", fontsize=7.4)
    ax.set_xlim(-Z_MM, Z_MM)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="center",
              handlelength=1.5, borderaxespad=0.2, labelspacing=0.28)
    ax.set_title(r"Caustic fit returns the beam quality", fontsize=8.2,
                 fontweight="bold", color=INK)
    return m2


def quality_panel(ax):
    z = np.linspace(0.0, Z_MM, 400)
    cases = [(1.0, GREY, (0, (4, 2.6))), (1.8, RED, "-"), (3.0, GOLD, "-")]
    for m2, color, ls in cases:
        w = w_of_z(z, m2=m2)
        ax.plot(z, w, color=color, lw=1.5, ls=ls, zorder=3)
        theta = float((w[-1] - w[-2]) / (z[-1] - z[-2]))
        ax.text(Z_MM * 1.01, w[-1], r"$M^2$ = " + f"{m2:.1f}\n"
                + f"{theta:.1f} mrad", ha="left", va="center", fontsize=6.4,
                color=color, linespacing=1.4, clip_on=False)

    ax.set_xlabel("axial position $z$ (mm)", fontsize=7.4)
    ax.set_ylabel(r"beam radius $w$ ($\mathrm{\mu m}$)", fontsize=7.4)
    ax.set_xlim(0.0, Z_MM)
    ax.set_ylim(0.0, w_of_z(Z_MM, m2=3.0) * 1.06)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Same waist, three beam qualities", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_caustic = fig.add_subplot(gs[0, 1])
    ax_quality = fig.add_subplot(gs[1, 1])

    z_r, theta = main_panel(ax_main)
    m2 = caustic_panel(ax_caustic)
    quality_panel(ax_quality)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_caustic, ax_quality],
                        dx=-0.42, dy=0.12)
    savefig(fig, "696_opt_beam_propagation.png")


if __name__ == "__main__":
    main()
