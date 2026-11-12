"""Wind shear profile: log-law versus power-law fit across the rotor disk.

The main panel plots measured wind speed against height (height on the y
axis) for a synthetic 200 m met mast.  Both a logarithmic profile and a
power-law profile are fitted to the same ten measured levels, so the
reported roughness length, friction velocity and shear exponent all come
from the plotted data rather than from the caption.  A shaded band marks
the swept range of a 126 m rotor at 100 m hub height.

Two smaller companion panels carry the evidence: a family of profiles for
three atmospheric stability classes (sharing the main panel's speed and
height scales) and the distribution of the fitted shear exponent within
each class.
"""
import textwrap

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
GRID = "#d9d9d9"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

# cold -> mid -> warm for stable / neutral / unstable stratification
STAB = [
    ("Stable", "#3d6b8e", 0.30),
    ("Neutral", "#c9a227", 0.16),
    ("Unstable", "#b23a48", 0.09),
]

KARMAN = 0.40
HUB = 100.0
DIAM = 126.0
Z_TOP = HUB + DIAM / 2.0
Z_BOT = HUB - DIAM / 2.0
Z_MAX = 205.0


def synth_mast(rng):
    """Ten measured levels drawn about a log profile with level-wise scatter."""
    z = np.array([10., 20., 40., 60., 80., 100., 120., 140., 160., 180.])
    u_true = (0.45 / KARMAN) * np.log(z / 0.085)
    sigma = 0.10 + 0.022 * np.sqrt(z)
    u_obs = u_true + rng.normal(0.0, 0.55 * sigma)
    return z, u_obs, sigma


def fit_profiles(z, u):
    """Least-squares log-law and power-law fits to the same measurements."""
    lnz = np.log(z)
    a, b = np.polyfit(lnz, u, 1)
    log_par = dict(ustar=a * KARMAN, z0=float(np.exp(-b / a)), a=a, b=b)
    alpha, c = np.polyfit(lnz, np.log(u), 1)
    pow_par = dict(alpha=alpha, amp=float(np.exp(c)))
    return log_par, pow_par


def u_log(z, p):
    return p["a"] * np.log(z) + p["b"]


def u_pow(z, p):
    return p["amp"] * z ** p["alpha"]


def main():
    set_pub_style()
    rng = np.random.default_rng(714)

    z_obs, u_obs, sig = synth_mast(rng)
    log_par, pow_par = fit_profiles(z_obs, u_obs)

    zg = np.linspace(4.0, Z_MAX, 320)
    fit_log = u_log(zg, log_par)
    fit_pow = u_pow(zg, pow_par)

    rmse_log = float(np.sqrt(np.mean((u_log(z_obs, log_par) - u_obs) ** 2)))
    rmse_pow = float(np.sqrt(np.mean((u_pow(z_obs, pow_par) - u_obs) ** 2)))
    u_hub = float(u_log(np.array([HUB]), log_par)[0])

    # stability family: pinned to the fitted hub speed, so the panels agree
    fam = {name: u_hub * (zg / HUB) ** al for name, _, al in STAB}
    alpha_samp = {name: np.clip(rng.normal(al, 0.035, 320), 0.01, None)
                  for name, _, al in STAB}

    ulo, uhi = shared_limits([u_obs, fit_log, fit_pow] + list(fam.values()),
                             pad=0.07)
    ulo = max(ulo, 0.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.88, 0.82, 5.02, 4.68))
    ax1 = axes_inches(fig, (6.72, 3.30, 3.00, 2.20))
    ax2 = axes_inches(fig, (6.72, 0.82, 3.00, 1.95))

    # ---- main panel: measured profile and the two fits -------------------
    ax.axhspan(Z_BOT, Z_TOP, facecolor=lighten(BLUE, 0.90), edgecolor="none",
               zorder=0)
    for zz, lab, ls in ((Z_TOP, "upper blade tip", (0, (3, 2))),
                        (HUB, "hub height", (0, (5, 2))),
                        (Z_BOT, "lower blade tip", (0, (3, 2)))):
        ax.axhline(zz, color=BLUE, lw=0.85, ls=ls, alpha=0.75, zorder=1)
        ax.text(ulo + 0.12, zz + 3.0, f"{lab}, {zz:.0f} m", fontsize=6.6,
                color=BLUE, ha="left", va="bottom")
    ax.annotate("", xy=(ulo + 0.55, Z_TOP), xytext=(ulo + 0.55, Z_BOT),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=BLUE,
                                shrinkA=0, shrinkB=0))
    ax.text(ulo + 0.66, HUB - 12.0, f"rotor swept\nrange, D = {DIAM:.0f} m",
            fontsize=6.6, color=BLUE, ha="left", va="top", linespacing=1.25)

    ax.plot(fit_log, zg, color=BLUE, lw=1.7, zorder=4,
            label="Logarithmic law fit")
    ax.plot(fit_pow, zg, color=RED, lw=1.5, ls=(0, (5, 2)), zorder=4,
            label="Power law fit")
    ax.errorbar(u_obs, z_obs, xerr=sig, fmt="o", ms=3.8, mfc="white",
                mec=INK, mew=0.9, ecolor="#8a8a8a", elinewidth=0.8,
                capsize=2.0, zorder=5, label="Cup anemometer, 10 min mean")
    ax.plot([u_hub], [HUB], marker="D", ms=4.4, mfc=GOLD, mec=INK, mew=0.7,
            zorder=6)
    ax.annotate(f"$u$({HUB:.0f} m) = {u_hub:.2f} m/s",
                xy=(u_hub, HUB), xytext=(u_hub + 0.55, HUB - 26.0),
                fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY))

    ax.set_xlim(ulo, uhi)
    ax.set_ylim(0.0, Z_MAX)
    ax.set_xlabel("Wind speed $u$ (m/s)", fontsize=8.0)
    ax.set_ylabel("Height above ground $z$ (m)", fontsize=8.0)
    ax.set_title("Measured profile and two fitted shear models",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    ax.grid(True, color=GRID, lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)

    stat = (f"Log law   $u_*$ = {log_par['ustar']:.3f} m/s, "
            f"$z_0$ = {log_par['z0']:.3f} m, RMSE = {rmse_log:.3f} m/s\n"
            f"Power law   $\\alpha$ = {pow_par['alpha']:.3f} (-), "
            f"RMSE = {rmse_pow:.3f} m/s")
    ax.text(0.975, 0.035, stat, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.8, color=INK, linespacing=1.5, zorder=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))

    # ---- aux 1: profile family by stability class ------------------------
    ax1.axhspan(Z_BOT, Z_TOP, facecolor=lighten(BLUE, 0.92), edgecolor="none",
                zorder=0)
    for name, col, al in STAB:
        ax1.plot(fam[name], zg, color=col, lw=1.5, zorder=3,
                 label=f"{name}, $\\alpha$ = {al:.2f}")
    ax1.axhline(HUB, color=GREY, lw=0.8, ls=(0, (5, 2)), zorder=2)
    ax1.text(uhi - 0.15, HUB + 4.0, "hub", fontsize=6.2, color=MUTED,
             ha="right", va="bottom")
    ax1.set_xlim(ulo, uhi)
    ax1.set_ylim(0.0, Z_MAX)
    ax1.set_xlabel("Wind speed $u$ (m/s)", fontsize=7.2)
    ax1.set_ylabel("Height $z$ (m)", fontsize=7.2)
    ax1.set_title("Profile family by stability class", fontsize=8.5,
                  fontweight="bold")
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", frameon=False, fontsize=6.5,
               handlelength=1.4, labelspacing=0.28)

    # ---- aux 2: distribution of the fitted shear exponent ----------------
    ag = np.linspace(0.0, 0.45, 320)
    for name, col, _ in STAB:
        s = alpha_samp[name]
        dens = gaussian_kde(s)(ag)
        ax2.fill_between(ag, dens, color=col, alpha=0.22, lw=0.0, zorder=2)
        ax2.plot(ag, dens, color=col, lw=1.3, zorder=3, label=name)
        ax2.plot([np.median(s)], [0.0], marker="^", ms=3.6, color=col,
                 clip_on=False, zorder=4)
    ax2.axvline(pow_par["alpha"], color=INK, lw=0.9, ls=(0, (4, 2)), zorder=5)
    ax2.text(pow_par["alpha"] + 0.006, 0.96,
             f"site fit {pow_par['alpha']:.3f}", transform=ax2.get_xaxis_transform(),
             fontsize=6.5, color=INK, ha="left", va="top")
    ax2.set_xlim(0.0, 0.45)
    ax2.set_ylim(0.0, None)
    ax2.set_xlabel("Shear exponent $\\alpha$ (-)", fontsize=7.2)
    ax2.set_ylabel("Probability density (-)", fontsize=7.2)
    ax2.set_title("Shear exponent by stability class", fontsize=8.5,
                  fontweight="bold")
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, fontsize=6.5,
               handlelength=1.4, labelspacing=0.28)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.50, dy=0.10)


    savefig(fig, "714_wnd_shear_profile.png")


if __name__ == "__main__":
    main()
