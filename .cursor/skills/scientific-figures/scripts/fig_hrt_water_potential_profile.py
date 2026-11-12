"""Water potential down the soil-plant-atmosphere path of a trained tomato.

One hydraulic resistance density is laid out along the flow path, from the
bulk soil through the rhizosphere, the root and stem xylem, the petiole and
finally the leaf mesophyll.  Integrating it gives the cumulative resistance,
and the profile for any transpiration rate is the soil potential minus the
flow times that resistance minus the gravitational head.  Because all three
profiles come from the same integral, the share of the total drop attributed
to each organ in the annotations is measured on the curve rather than
asserted, and the transpiration rate that first pushes the leaf past its
turgor loss point is found by bisection on the same expression.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.optimize import brentq

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
GOLD = "#c9a227"

SEED = 1035
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

PSI_SOIL = -0.05        # bulk soil matric potential, MPa
PSI_TLP = -1.60         # turgor loss point, MPa
RHO_G = 0.00978         # gravitational head, MPa per m
Z_LO, Z_HI = -0.45, 2.30

# Segment name, lower and upper bound in m, resistance density in
# MPa per (mmol m-2 s-1) per m, band colour.
SEGMENTS = [
    ("Bulk soil", -0.45, -0.30, 0.020, "#e8e0d4"),
    ("Rhizosphere", -0.30, -0.25, 0.900, "#d8c9b0"),
    ("Root xylem", -0.25, 0.00, 0.030, "#cfe0ea"),
    ("Stem xylem", 0.00, 1.60, 0.035, "#e3eef5"),
    ("Petiole", 1.60, 2.10, 0.120, "#cfe0ea"),
    ("Leaf mesophyll", 2.10, 2.30, 1.400, "#dbe9d9"),
]

FLOWS = [(0.5, "#9cc3dd", "night, 0.5"),
         (2.0, "#3d6b8e", "morning, 2.0"),
         (4.5, "#0b3d63", "midday, 4.5")]

UNIT_E = r"mmol m$^{-2}$ s$^{-1}$"


def resistance(z):
    """Cumulative hydraulic resistance from the bulk soil up to height z."""
    dens = np.zeros_like(z)
    for (_n, lo, hi, r, _c) in SEGMENTS:
        dens = np.where((z >= lo) & (z < hi), r, dens)
    dens[-1] = SEGMENTS[-1][3]
    dz = np.diff(z, prepend=z[0])
    return np.cumsum(dens * dz)


def profile(z, rcum, flow):
    return PSI_SOIL - flow * rcum - RHO_G * (z - z[0])


def segment_shares(z, rcum, flow):
    """Fraction of the whole potential drop contributed by each segment."""
    psi = profile(z, rcum, flow)
    total = psi[0] - psi[-1]
    out = []
    for (name, lo, hi, _r, _c) in SEGMENTS:
        i0 = int(np.argmin(np.abs(z - lo)))
        i1 = int(np.argmin(np.abs(z - hi)))
        out.append((name, 100.0 * (psi[i0] - psi[i1]) / total))
    return out, total


def main_panel(ax, z, rcum, e_crit):
    for (name, lo, hi, _r, col) in SEGMENTS:
        ax.axhspan(lo, hi, color=col, zorder=1)
        ax.text(-2.36, 0.5 * (lo + hi), name, ha="left", va="center",
                fontsize=6.4, color=SUB, zorder=4)
    ax.axhline(0.0, color="#8a6b4f", lw=1.0, zorder=5)
    ax.text(-0.08, 0.03, "soil surface", ha="right", va="bottom",
            fontsize=6.2, color="#7a5c3f", zorder=6)

    for (flow, col, label) in FLOWS:
        psi = profile(z, rcum, flow)
        ax.plot(psi, z, color=col, lw=2.0, zorder=7)
        ax.plot([psi[-1]], [z[-1]], marker="o", ms=4.8, mfc="white", mec=col,
                mew=1.2, zorder=9)
        ax.text(psi[-1], Z_HI + 0.045, "{:.2f} MPa".format(psi[-1]),
                ha="center", va="bottom", fontsize=6.3, color=col,
                fontweight="bold", zorder=9)

    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    ax.axvline(PSI_TLP, color=RED, lw=1.1, ls=(0, (4.5, 2.6)), zorder=6)
    ax.text(PSI_TLP - 0.05, 2.02, "turgor loss point\n{:.2f} MPa"
            .format(PSI_TLP), ha="right", va="center", fontsize=6.5,
            color=RED, fontweight="bold", linespacing=1.32,
            zorder=9).set_path_effects(halo)
    ax.annotate("leaves lose turgor above a transpiration rate of\n{:.2f} "
                "{:s}, solved on the same profile".format(e_crit, UNIT_E),
                xy=(PSI_TLP, 1.55), xytext=(-2.32, 1.22), fontsize=6.4,
                color=RED, ha="left", va="center", linespacing=1.34,
                zorder=10, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                           shrinkA=2, shrinkB=4)
                ).set_path_effects(halo)

    shares, total = segment_shares(z, rcum, FLOWS[-1][0])
    biggest = max(shares, key=lambda s: s[1])
    second = sorted(shares, key=lambda s: -s[1])[1]
    ax.text(-0.10, 1.76, "at midday the {:s} takes {:.0f} % of the\n{:.2f} "
            "MPa total drop and the {:s} {:.0f} %"
            .format(biggest[0].lower(), biggest[1], total, second[0].lower(),
                    second[1]), ha="right", va="center", fontsize=6.4,
            color=INK, linespacing=1.34,
            zorder=9).set_path_effects(halo)

    handles = [Line2D([], [], color=c, lw=2.0,
                      label="{:s} {:s}".format(lab, UNIT_E))
               for (_f, c, lab) in FLOWS]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, borderaxespad=0.4, labelspacing=0.26,
              title="transpiration rate", title_fontsize=6.2,
              bbox_to_anchor=(0.015, 0.11))

    ax.set_xlim(-2.40, 0.10)
    ax.set_ylim(Z_LO, Z_HI + 0.16)
    ax.set_xlabel("water potential, $\\Psi$ (MPa)")
    ax.set_ylabel("height above the soil surface (m)")
    ax.grid(axis="x", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Potential falls fastest where the resistance is, not where "
                 "the path is long", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def stomata_panel(ax, vpd, gs_set):
    for (psi_s, col, lab), g in zip(gs_set[0], gs_set[1]):
        ax.plot(vpd, g, color=col, lw=1.6, zorder=6, label=lab)
        half = g[0] / 2.0
        idx = int(np.argmin(np.abs(g - half)))
        ax.plot([vpd[idx]], [g[idx]], marker="o", ms=4.0, mfc="white",
                mec=col, mew=1.0, zorder=8)
        ax.text(vpd[idx] + 0.10, g[idx] + 12.0, "{:.1f} kPa"
                .format(vpd[idx]), ha="left", va="bottom", fontsize=6.0,
                color=col, zorder=9)
    ax.text(3.95, 345.0, "circles mark the half-maximum\nvapour pressure "
            "deficit", ha="right", va="top", fontsize=6.0, color=SUB,
            linespacing=1.32, zorder=9)

    ax.set_xlim(0.2, 4.0)
    ax.set_ylim(0.0, 600.0)
    ax.set_xlabel("vapour pressure deficit (kPa)", fontsize=6.8)
    ax.set_ylabel("stomatal conductance (" + UNIT_E + ")", fontsize=6.4)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.3, borderaxespad=0.3, labelspacing=0.2)
    ax.set_title("Stomata close on dry air, sooner on dry soil", fontsize=8.0,
                 fontweight="bold", color=INK)


def diurnal_panel(ax, hours, e_day, psi_leaf):
    ax.plot(hours, psi_leaf, color="#0b3d63", lw=1.8, zorder=6)
    ax.axhline(PSI_TLP, color=RED, lw=0.8, ls=(0, (3.6, 2.2)), zorder=4)
    i_min = int(np.argmin(psi_leaf))
    ax.plot([hours[i_min]], [psi_leaf[i_min]], marker="v", ms=4.6,
            mfc="white", mec="#0b3d63", mew=1.0, zorder=8)
    ax.annotate("midday minimum\n{:.2f} MPa at {:.1f} h"
                .format(psi_leaf[i_min], hours[i_min]),
                xy=(hours[i_min], psi_leaf[i_min]), xytext=(23.5, -1.05),
                fontsize=6.0, color="#0b3d63", ha="right", va="center",
                linespacing=1.32,
                zorder=9, arrowprops=dict(arrowstyle="-", color="#0b3d63",
                                          lw=0.7, shrinkA=2, shrinkB=4))
    ax.text(0.4, PSI_TLP + 0.06, "turgor loss point", ha="left", va="bottom",
            fontsize=6.0, color=RED, zorder=9)
    ax.text(0.4, psi_leaf[0] + 0.05, "predawn {:.2f} MPa".format(psi_leaf[0]),
            ha="left", va="bottom", fontsize=6.0, color=SUB, zorder=9)

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(-2.65, 0.12)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("time of day (h)", fontsize=6.8)
    ax.set_ylabel("leaf $\\Psi$ (MPa)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    axr = ax.twinx()
    axr.fill_between(hours, 0.0, e_day, color="#cfe0ea", alpha=0.75, zorder=2)
    axr.plot(hours, e_day, color="#3d6b8e", lw=1.2, zorder=5)
    axr.set_ylim(0.0, 6.6)
    axr.set_ylabel("transpiration (" + UNIT_E + ")", fontsize=6.4,
                   color="#3d6b8e")
    axr.tick_params(axis="y", labelsize=6.2, colors="#3d6b8e")
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color("#3d6b8e")
    axr.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax.set_title("One clear day of leaf water status", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    z = np.linspace(Z_LO, Z_HI, 1101)
    rcum = resistance(z)
    r_tot = float(rcum[-1])

    for (flow, _c, _l) in FLOWS:
        psi = profile(z, rcum, flow)
        assert np.all(np.diff(psi) <= 1e-12), \
            "water potential must fall monotonically along the flow path"
    e_crit = brentq(lambda e: profile(z, rcum, e)[-1] - PSI_TLP, 0.01, 20.0,
                    xtol=1e-12)

    vpd = np.linspace(0.2, 4.0, 260)
    soils = [(-0.05, "#9cc3dd", "wet soil, $\\Psi_s$ -0.05 MPa"),
             (-0.35, "#3d6b8e", "drying, $\\Psi_s$ -0.35 MPa"),
             (-0.75, "#0b3d63", "dry soil, $\\Psi_s$ -0.75 MPa")]
    gs_lines = []
    for (psi_s, _c, _l) in soils:
        g0 = 560.0 * np.exp(1.35 * psi_s)
        gs_lines.append(g0 / (1.0 + (vpd / (1.55 + 1.4 * psi_s)) ** 1.8))

    hours = np.linspace(0.0, 24.0, 289)
    day = np.clip(np.sin(np.pi * (hours - 6.0) / 12.0), 0.0, None) ** 1.25
    e_day = 0.18 + 4.9 * day + rng.normal(0.0, 0.055, hours.size) * day
    e_day = np.clip(e_day, 0.05, None)
    psi_leaf = PSI_SOIL - e_day * r_tot - RHO_G * (Z_HI - Z_LO)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_gs = axes_inches(fig, AUX_T)
    ax_di = axes_inches(fig, AUX_B)

    main_panel(ax_main, z, rcum, e_crit)
    stomata_panel(ax_gs, vpd, (soils, gs_lines))
    diurnal_panel(ax_di, hours, e_day, psi_leaf)

    shares, total = segment_shares(z, rcum, FLOWS[-1][0])
    leaf_share = [s for s in shares if s[0] == "Leaf mesophyll"][0][1]


    panel_labels_inches(fig, [ax_main, ax_gs, ax_di], dx=-0.62, dy=0.14)
    savefig(fig, "1035_hrt_water_potential_profile.png")


if __name__ == "__main__":
    main()
