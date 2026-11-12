"""Galaxy rotation curve: the flat outer disc only closes with a dark halo.

The main panel is the claim.  Four mass components are drawn separately - a
Plummer bulge, an exponential stellar disc from the Freeman/Bessel solution,
an extended gas disc and a pseudo-isothermal halo - and added in quadrature.
The baryons alone turn over into a Keplerian decline past the disc scale
length and fall tens of km/s short of the measured points; adding the halo is
what flattens the model onto them.  The halo's share of the outer circular
speed is integrated from the plotted curves, not asserted.

Side panels carry the supporting evidence: how sensitive the fit is to the
stellar mass-to-light ratio, and the enclosed-mass profile the same curves
imply.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1, k0, k1

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GREY = "#808080"

RED = "#b23a48"        # total model, the emphasis object
BLUE = "#3b5b92"       # stellar disc
GREEN = "#2e7d5b"      # gas disc
GOLD = "#c9a227"       # bulge
PURPLE = "#8e6bbf"     # dark halo

G_KPC = 4.301e-6       # (km/s)^2 kpc / Msun

M_BULGE, A_BULGE = 8.0e9, 0.60          # Msun, kpc
M_DISK, RD_DISK = 4.2e10, 2.60
M_GAS, RD_GAS = 1.2e10, 6.00
V_HALO, RC_HALO = 175.0, 6.00
R_MIN, R_MAX = 0.20, 21.0


def v_bulge(r, mass=M_BULGE):
    return np.sqrt(G_KPC * mass * r ** 2 / (r ** 2 + A_BULGE ** 2) ** 1.5)


def v_exp_disc(r, mass, r_d):
    """Freeman's exponential-disc rotation curve (Bessel-function form)."""
    sigma0 = mass / (2.0 * np.pi * r_d ** 2)
    y = np.clip(r / (2.0 * r_d), 1e-6, None)
    kernel = i0(y) * k0(y) - i1(y) * k1(y)
    v2 = 4.0 * np.pi * G_KPC * sigma0 * r_d * y ** 2 * kernel
    return np.sqrt(np.clip(v2, 0.0, None))


def v_halo(r, v_inf=V_HALO, r_c=RC_HALO):
    """Pseudo-isothermal sphere: rises linearly, then flattens to v_inf."""
    return v_inf * np.sqrt(np.clip(1.0 - (r_c / r) * np.arctan(r / r_c),
                                   0.0, None))


def components(r, ml=1.0):
    return dict(bulge=v_bulge(r, M_BULGE * ml),
                disc=v_exp_disc(r, M_DISK * ml, RD_DISK),
                gas=v_exp_disc(r, M_GAS, RD_GAS),
                halo=v_halo(r))


def quadrature(parts, keys):
    return np.sqrt(sum(parts[k] ** 2 for k in keys))


def observations(seed=23, n=26):
    rng = np.random.default_rng(seed)
    r = np.linspace(0.9, R_MAX - 0.6, n)
    r = r + rng.normal(0.0, 0.12, n)
    r.sort()
    model = quadrature(components(r), ["bulge", "disc", "gas", "halo"])
    err = 5.0 + 5.5 * (r / R_MAX) + 3.0 * rng.random(n)
    v = model + rng.normal(0.0, 1.0, n) * err
    return r, v, err


def main_panel(ax, r, obs_r, obs_v, obs_e):
    parts = components(r)
    total = quadrature(parts, ["bulge", "disc", "gas", "halo"])
    baryon = quadrature(parts, ["bulge", "disc", "gas"])

    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt="o", ms=4.0, mfc="white",
                mec=RULE, mew=1.0, ecolor=RULE, elinewidth=0.9, capsize=2.0,
                linestyle="none", zorder=6, label="observed H I rotation")
    ax.plot(r, total, color=RED, lw=2.0, zorder=5,
            label="total model (all four components)")
    ax.plot(r, baryon, color=RULE, lw=1.3, ls=(0, (5, 2.6)), zorder=4,
            label="baryons only, no halo")

    style = [("disc", BLUE, "stellar disc"), ("bulge", GOLD, "bulge"),
             ("gas", GREEN, "gas disc"), ("halo", PURPLE, "dark halo")]
    for key, color, name in style:
        ax.plot(r, parts[key], color=color, lw=1.25, ls=(0, (2.6, 1.8)),
                zorder=3, label=name)

    gap = total[-1] - baryon[-1]
    ax.annotate("", xy=(r[-1] - 0.35, total[-1]),
                xytext=(r[-1] - 0.35, baryon[-1]),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.0))
    ax.text(r[-1] - 0.75, 0.5 * (total[-1] + baryon[-1]),
            f"{gap:.0f} km/s missing\nwithout a halo", ha="right",
            va="center", fontsize=7.2, color=PURPLE, linespacing=1.35)

    f_halo = parts["halo"][-1] ** 2 / total[-1] ** 2
    ax.set_xlim(0.0, R_MAX)
    ax.set_ylim(0.0, max(total.max(), (obs_v + obs_e).max()) * 1.20)
    ax.set_xlabel("galactocentric radius $R$ (kpc)")
    ax.set_ylabel("circular velocity $v_c$ (km/s)")
    ax.tick_params(labelsize=7.2)
    ax.grid(lw=0.45, alpha=0.55)
    ax.legend(fontsize=7.0, frameon=False, loc="lower right", ncol=2,
              handlelength=1.7, borderaxespad=0.4, columnspacing=1.1)
    return total, baryon, parts, f_halo


def ml_panel(ax, r, obs_r, obs_v, obs_e, v_span):
    factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    shades = ["#cfe1ee", "#9cc3dd", "#5f9dc6", "#2e74a8", "#134a76"]
    chi2 = []
    for f, col in zip(factors, shades):
        tot = quadrature(components(r, ml=f), ["bulge", "disc", "gas", "halo"])
        wide = abs(f - 1.0) < 1e-9
        ax.plot(r, tot, color=col, lw=1.9 if wide else 1.15, zorder=4 if wide
                else 3)
        pred = quadrature(components(obs_r, ml=f),
                          ["bulge", "disc", "gas", "halo"])
        chi2.append(float(np.sum(((obs_v - pred) / obs_e) ** 2) /
                          (obs_r.size - 1)))
        # Only the two extremes and the fiducial are labelled; five labels at
        # the curve ends collide at this panel height.
        if f in (factors[0], 1.0, factors[-1]):
            ax.text(r[-1] + 0.5, tot[-1], f"{f:.2f}", ha="left", va="center",
                    fontsize=6.4, color=col)
    chi2 = np.array(chi2)
    best = factors[int(np.argmin(chi2))]
    ax.text(r[-1] + 0.5, v_span[1] * 0.97, r"$M/L$ x", ha="left",
            va="top", fontsize=6.4, color=SUB)

    ax.errorbar(obs_r, obs_v, yerr=obs_e, fmt="o", ms=2.4, mfc="white",
                mec=RULE, mew=0.6, ecolor=RULE, elinewidth=0.5, capsize=0.0,
                linestyle="none", zorder=5)
    ax.text(0.04, 0.06, f"best $\\chi^2_\\nu$ = {chi2.min():.2f} at "
                        f"$M/L$ x {best:.2f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=INK)
    ax.set_xlim(0.0, R_MAX * 1.18)
    ax.set_ylim(*v_span)
    ax.set_xlabel("$R$ (kpc)", fontsize=7.4)
    ax.set_ylabel("$v_c$ (km/s)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(lw=0.45, alpha=0.6)
    ax.set_title(r"Sensitivity to the stellar mass-to-light ratio",
                 fontsize=8.4, fontweight="bold", color=INK)
    return best, chi2.min()


def mass_panel(ax, r, total, baryon):
    m_tot = total ** 2 * r / G_KPC
    m_bar = baryon ** 2 * r / G_KPC
    ax.fill_between(r, m_bar, m_tot, color=PURPLE, alpha=0.20, linewidth=0.0,
                    zorder=2, label="dark matter")
    ax.plot(r, m_tot, color=RED, lw=1.5, zorder=4, label="total")
    ax.plot(r, m_bar, color=RULE, lw=1.2, ls=(0, (5, 2.6)), zorder=4,
            label="baryons")
    f_dm = 1.0 - m_bar[-1] / m_tot[-1]
    ax.text(0.04, 0.93,
            f"{f_dm * 100:.0f} % dark within {r[-1]:.0f} kpc",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=PURPLE)
    ax.set_yscale("log")
    ax.set_xlim(0.0, R_MAX)
    ax.set_ylim(1e9, m_tot.max() * 3.0)
    ax.set_xlabel("$R$ (kpc)", fontsize=7.4)
    ax.set_ylabel(r"$M(<R)$ ($M_\odot$)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(lw=0.45, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right",
              handlelength=1.4, borderaxespad=0.3)
    ax.set_title("Enclosed mass implied by the same curves", fontsize=8.4,
                 fontweight="bold", color=INK)
    return f_dm


def main():
    set_pub_style()
    r = np.linspace(R_MIN, R_MAX, 500)
    obs_r, obs_v, obs_e = observations()

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    ax_main = axes_inches(fig, (0.90, 0.88, 5.45, 4.58))
    ax_ml = axes_inches(fig, (7.05, 3.30, 3.00, 2.16))
    ax_mass = axes_inches(fig, (7.05, 0.88, 3.00, 1.76))

    total, baryon, parts, f_halo = main_panel(ax_main, r, obs_r, obs_v, obs_e)
    v_span = shared_limits([total, obs_v, np.array([0.0])], pad=0.08)
    best, chi2 = ml_panel(ax_ml, r, obs_r, obs_v, obs_e, v_span)
    f_dm = mass_panel(ax_mass, r, total, baryon)

    flat = total[r > 8.0]
    ax_main.set_title(
        f"Beyond 8 kpc the model stays flat to within "
        f"{100 * np.ptp(flat) / flat.mean():.1f} %, with the halo supplying "
        f"{f_halo * 100:.0f} % of $v_c^2$", fontsize=9.0, fontweight="bold",
        color=INK)


    panel_labels_inches(fig, [ax_main, ax_ml, ax_mass], dx=-0.66, dy=0.30)
    savefig(fig, "800_ast_rotation_curve.png")


if __name__ == "__main__":
    main()
