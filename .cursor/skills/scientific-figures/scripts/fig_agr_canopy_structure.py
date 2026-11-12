"""Figure 396 - Canopy architecture and light capture in a maize stand.

Three views of one synthetic canopy, read left to right: how the leaves are
tilted in each stratum, where the leaf area actually sits, and how much
photosynthetically active radiation survives to each height.  The strata keep
the same three greens in all three panels, and the two profile panels share the
height axis so the reader can trace one horizon straight across.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist

from pubstyle import set_pub_style, savefig, add_panel_label

SEED = 604
H_TOP = 1.12                       # canopy height, m
DZ = 0.02                          # profile layer thickness, m

# Lower / middle / upper stratum, dark to light in the house green ramp.
STRATA = [
    ("Lower", 0.00, 0.36, "#a4d7b8", (2.6, 5.4)),      # planophile
    ("Middle", 0.36, 0.74, "#3d9469", (4.6, 4.6)),     # plagiophile
    ("Upper", 0.74, H_TOP, "#0b3d2e", (5.6, 2.4)),     # erectophile
]
GOLD = "#c9a227"
ACCENT = "#b23a48"


def lad_profile(z):
    """Leaf area density, m2 leaf per m3 canopy volume."""
    return 2.35 * (3.05 * np.exp(-0.5 * ((z - 0.70) / 0.165) ** 2)
                   + 1.85 * np.exp(-0.5 * ((z - 0.40) / 0.190) ** 2)
                   + 0.35 * np.exp(-0.5 * ((z - 0.12) / 0.130) ** 2))


def angle_pdf(theta_deg, a, b):
    """Leaf inclination density on 0-90 deg, normalised to integrate to 1."""
    return beta_dist.pdf(np.clip(theta_deg / 90.0, 1e-4, 1 - 1e-4), a, b) / 90.0


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    z = np.arange(DZ / 2, H_TOP, DZ)
    lad = lad_profile(z)
    lai_layer = lad * DZ
    lai_total = lai_layer.sum()
    # Cumulative leaf area counted downward from the top of the canopy.
    lai_down = np.cumsum(lai_layer[::-1])[::-1]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 0.72, 1.0],
                          left=0.055, right=0.975, top=0.790, bottom=0.115,
                          wspace=0.40)
    # A 0-90 deg wedge only fills the lower-left of its box, so the polar axes
    # is boxed into the vertical middle of the column instead of stretched.
    gsl = gs[0, 0].subgridspec(3, 1, height_ratios=[0.16, 1.0, 0.24],
                               hspace=0.0)
    ax_pol = fig.add_subplot(gsl[1, 0], projection="polar")
    ax_lad = fig.add_subplot(gs[0, 1])
    ax_lgt = fig.add_subplot(gs[0, 2])

    # ---- (a) leaf inclination distribution --------------------------------
    th = np.linspace(0.0, 90.0, 181)
    mta = {}
    for name, z0, z1, col, (a, b) in STRATA:
        pdf = angle_pdf(th, a, b)
        pct = pdf * 10.0 * 100.0                      # % of leaves per 10 deg
        mta[name] = float(np.trapezoid(th * pdf, th) / np.trapezoid(pdf, th))
        ax_pol.plot(np.deg2rad(th), pct, color=col, lw=1.6, zorder=4)
        ax_pol.fill_between(np.deg2rad(th), 0, pct, color=col, alpha=0.22,
                            zorder=2)
        ax_pol.plot([np.deg2rad(mta[name])], [np.interp(mta[name], th, pct)],
                    marker="o", ms=4.0, mfc="white", mec=col, mew=1.1,
                    zorder=6)

    ax_pol.set_thetamin(0)
    ax_pol.set_thetamax(90)
    ax_pol.set_theta_zero_location("E")
    ax_pol.set_thetagrids([0, 15, 30, 45, 60, 75, 90],
                          labels=["0\u00b0\nhorizontal", "15\u00b0", "30\u00b0",
                                  "45\u00b0", "60\u00b0", "75\u00b0",
                                  "90\u00b0\nvertical"], fontsize=6.8)
    ax_pol.set_rlabel_position(88)
    ax_pol.set_ylim(0, 26)
    ax_pol.set_yticks([5, 10, 15, 20, 25])
    ax_pol.tick_params(labelsize=6.6)
    ax_pol.grid(lw=0.5, alpha=0.6)
    ax_pol.text(0.5, -0.115, "radius: leaf frequency (% per 10\u00b0)\n"
                "open marker: mean tip angle",
                transform=ax_pol.transAxes, ha="center", va="top",
                fontsize=7.0, color="#555555", linespacing=1.5)

    # ---- (b) vertical leaf area density profile ---------------------------
    x_note = lad.max() * 1.06
    for name, z0, z1, col, _ in STRATA:
        m = (z >= z0) & (z < z1)
        ax_lad.barh(z[m], lad[m], height=DZ * 0.94, color=col,
                    edgecolor="white", linewidth=0.25, zorder=3)
        lai_s = lai_layer[m].sum()
        ax_lad.text(x_note, 0.5 * (z0 + min(z1, H_TOP)),
                    f"{name}\nLAI {lai_s:.2f}\nMTA {mta[name]:.0f}\u00b0",
                    ha="left", va="center", fontsize=6.4, color="#333333",
                    linespacing=1.4)

    ax_top = ax_lad.twiny()
    ax_top.plot(lai_down, z, color=ACCENT, lw=1.5, zorder=5)
    ax_top.set_xlim(0, lai_total * 1.06)
    ax_top.set_xlabel("cumulative LAI from top (m$^2$/m$^2$)", fontsize=7.6,
                      color=ACCENT, labelpad=3)
    ax_top.tick_params(axis="x", labelsize=6.8, colors=ACCENT)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["top"].set_color(ACCENT)
    ax_top.spines["right"].set_visible(False)

    ax_lad.set_xlim(0, lad.max() * 1.62)
    ax_lad.set_xlabel("leaf area density (m$^2$/m$^3$)", fontsize=8.0)
    ax_lad.set_ylabel("height above ground (m)", fontsize=8.0)

    # ---- (c) Beer-Lambert light extinction --------------------------------
    k_true = 0.63
    trans = np.exp(-k_true * lai_down)

    z_obs = np.array([1.06, 0.94, 0.82, 0.70, 0.58, 0.46, 0.34, 0.22, 0.08])
    l_obs = np.interp(z_obs, z, lai_down)
    t_obs = np.clip(np.exp(-k_true * l_obs) * np.exp(rng.normal(0, 0.085,
                                                                z_obs.size)),
                    0.004, 1.0)
    t_err = np.clip(0.055 * t_obs + 0.006, 0.004, None)
    # Extinction coefficient refitted from the plotted observations only.
    k_fit = float(-(l_obs * np.log(t_obs)).sum() / (l_obs ** 2).sum())

    ax_lgt.fill_betweenx(z, 0, trans, color=GOLD, alpha=0.20, zorder=1)
    ax_lgt.fill_betweenx(z, trans, 1.0, color="#2e7d5b", alpha=0.10, zorder=1)
    ax_lgt.plot(np.exp(-k_fit * lai_down), z, color="#333333", lw=1.0,
                ls=(0, (3.2, 1.8)), zorder=4)
    ax_lgt.plot(trans, z, color=GOLD, lw=1.9, zorder=5)
    ax_lgt.errorbar(t_obs, z_obs, xerr=t_err, fmt="o", ms=3.6, mfc="white",
                    mec="#8a6b1f", mew=1.0, ecolor="#8a6b1f", elinewidth=0.8,
                    capsize=2.0, ls="none", zorder=6)

    for name, z0, z1, col, _ in STRATA[:-1]:
        ax_lgt.axhline(z1, color="#cccccc", lw=0.7, ls="-", zorder=2)
        ax_lad.axhline(z1, color="#cccccc", lw=0.7, ls="-", zorder=1)

    t_floor = float(np.exp(-k_fit * lai_total))
    ax_lgt.annotate(f"{1 - t_floor:.1%} of incident PAR\nintercepted by the "
                    f"canopy",
                    xy=(t_floor, z[0]), xytext=(0.36, 0.50),
                    textcoords="axes fraction", fontsize=6.8, color=ACCENT,
                    linespacing=1.4, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ACCENT,
                                    shrinkA=2, shrinkB=3))
    ax_lgt.text(0.965, 0.265,
                f"Beer-Lambert fit\n$T = e^{{-kL}}$,  k = {k_fit:.3f}\n"
                f"total LAI = {lai_total:.2f} m$^2$/m$^2$",
                transform=ax_lgt.transAxes, ha="right", va="top",
                fontsize=6.8, color="#333333", linespacing=1.45)
    ax_lgt.text(0.048, 0.36, "transmitted", fontsize=6.6, color="#8a6b1f",
                rotation=90, ha="center", va="center",
                transform=ax_lgt.transAxes)
    ax_lgt.text(0.80, 0.70, "intercepted", fontsize=6.6, color="#2e7d5b",
                ha="center", va="center", transform=ax_lgt.transAxes)

    ax_lgt.set_xlim(0, 1.02)
    ax_lgt.set_xlabel("fraction of above-canopy PAR remaining (-)",
                      fontsize=8.0)
    ax_lgt.set_ylabel("height above ground (m)", fontsize=8.0)

    for ax in (ax_lad, ax_lgt):
        ax.set_ylim(0.0, H_TOP)
        ax.tick_params(labelsize=7.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_lgt.grid(axis="x", lw=0.5, alpha=0.5)

    handles = [Line2D([0], [0], color=c, lw=2.6,
                      label=f"{n} stratum ({z0:.2f}-{z1:.2f} m)")
               for n, z0, z1, c, _ in STRATA]
    handles += [Line2D([0], [0], color=ACCENT, lw=1.5,
                       label="cumulative LAI from top"),
                Line2D([0], [0], color=GOLD, lw=1.9,
                       label="modelled PAR transmittance"),
                Line2D([0], [0], color="#8a6b1f", lw=0, marker="o", ms=3.6,
                       mfc="white", mec="#8a6b1f",
                       label="ceptometer readings (\u00b11 SE)")]

    # Panel titles and letters are stamped at one figure-level height: the
    # twin top axis on (b) would otherwise push its own title far above the
    # other two.
    titles = [(ax_pol, "(a)", "Leaf inclination by stratum"),
              (ax_lad, "(b)", "Where the leaf area sits"),
              (ax_lgt, "(c)", "Light extinction through the canopy")]
    for ax, lab, title in titles:
        bb = ax.get_position()
        fig.text(bb.x0 + bb.width / 2.0, 0.845, title, ha="center",
                 va="bottom", fontsize=8.5, fontweight="bold")
        fig.text(bb.x0 - 0.034, 0.842, lab, ha="left", va="bottom",
                 fontsize=11, fontweight="bold")


    savefig(fig, "396_agr_canopy_structure.png")


if __name__ == "__main__":
    main()
