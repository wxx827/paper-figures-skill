"""Neutron-density lithology crossplot with M-N and porosity companions.

The main panel is the chart every log analyst reaches for first: neutron
porosity across, bulk density *down* (the axis is inverted by convention, so
tight rock sits at the bottom), with the sandstone, limestone and dolomite
matrix lines drawn through it and ticked every five porosity units.

Samples are synthesised by picking a porosity and a matrix, evaluating the
same three parameterised lines the chart draws, then adding scatter - so a
point that plots on the limestone line really was built as limestone.  The
gas arrow, the M and N mineral anchors, the group means and every number in
the annotations are computed from those arrays, never typed in.

Companions are 56 % of the main panel's width: an M-N crossplot that separates
the same three matrices without needing porosity, and the porosity histogram
that the main panel implies.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, safe_colorbar, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

SAND = "#c9a227"
LIME = "#3b5b92"
DOLO = "#8e6bbf"
GAS = "#b23a48"

# (matrix density g/cm3, neutron offset v/v, sonic dtma us/ft, colour, marker)
MATRIX = {
    "Sandstone": (2.650, -0.035, 55.5, SAND, "o"),
    "Limestone": (2.710, 0.000, 47.6, LIME, "s"),
    "Dolomite": (2.870, 0.045, 43.5, DOLO, "^"),
}
RHO_FL, DT_FL, NPHI_FL = 1.00, 189.0, 1.00

X_LIM = (-0.06, 0.46)          # neutron porosity, limestone units
Y_LIM = (3.02, 1.92)           # bulk density, inverted by convention
PHI_TICKS = (0.10, 0.20, 0.30, 0.40)


def matrix_line(name, phi):
    """Chart line for one matrix: apparent neutron and bulk density."""
    rho_ma, off, _, _, _ = MATRIX[name]
    nphi = phi + off * (1.0 - 1.4 * phi)      # offset closes as porosity grows
    rhob = rho_ma - phi * (rho_ma - RHO_FL)
    return nphi, rhob


def wyllie(name, phi):
    _, _, dt_ma, _, _ = MATRIX[name]
    return dt_ma + phi * (DT_FL - dt_ma)


def mn(nphi, rhob, dt):
    m = 0.01 * (DT_FL - dt) / (rhob - RHO_FL)
    n = (NPHI_FL - nphi) / (rhob - RHO_FL)
    return m, n


def sample_group(rng, name, n, phi_lo, phi_hi, gr_mu, gr_sd, gas=0.0):
    phi = rng.uniform(phi_lo, phi_hi, n)
    nphi, rhob = matrix_line(name, phi)
    if gas:
        nphi = nphi - gas
        rhob = rhob - 0.55 * gas
    nphi = nphi + rng.normal(0.0, 0.014, n)
    rhob = rhob + rng.normal(0.0, 0.028, n)
    dt = wyllie(name, phi) + rng.normal(0.0, 2.4, n)
    gr = np.clip(rng.normal(gr_mu, gr_sd, n), 6.0, 145.0)
    return dict(name=name, phi=phi, nphi=nphi, rhob=rhob, dt=dt, gr=gr)


def main():
    set_pub_style()
    rng = np.random.default_rng(8317)

    groups = [
        sample_group(rng, "Sandstone", 210, 0.06, 0.31, 42.0, 15.0),
        sample_group(rng, "Limestone", 150, 0.03, 0.22, 22.0, 8.0),
        sample_group(rng, "Dolomite", 120, 0.04, 0.24, 28.0, 10.0),
    ]
    gas_grp = sample_group(rng, "Sandstone", 95, 0.10, 0.28, 30.0, 9.0,
                           gas=0.115)
    gas_grp["name"] = "Gas sandstone"

    # shaly tail: high GR points drifting off the sandstone line
    shaly = sample_group(rng, "Sandstone", 90, 0.14, 0.30, 96.0, 16.0)
    shaly["nphi"] = shaly["nphi"] + rng.uniform(0.03, 0.11, shaly["gr"].size)
    shaly["rhob"] = shaly["rhob"] + rng.uniform(0.08, 0.22, shaly["gr"].size)
    shaly["name"] = "Shaly sandstone"

    allg = groups + [gas_grp, shaly]
    gr_all = np.concatenate([g["gr"] for g in allg])
    cmap = get_cmap("cb_sequential_warm")
    vmin, vmax = float(gr_all.min()), float(gr_all.max())

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.80, 0.82, 4.55, 4.56))
    ax_mn = axes_inches(fig, (6.95, 3.28, 2.65, 2.10))
    ax_h = axes_inches(fig, (6.95, 0.82, 2.65, 2.10))

    # ---- main crossplot -----------------------------------------------------
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.grid(color=GRID, lw=0.45, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("Neutron porosity, limestone units (v/v)", fontsize=8.0)
    ax.set_ylabel(r"Bulk density ($g/cm^3$)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)

    phi_line = np.linspace(0.0, 0.45, 200)
    for name, (_, _, _, col, _) in MATRIX.items():
        nx, ny = matrix_line(name, phi_line)
        ax.plot(nx, ny, color=col, lw=1.5, zorder=4)
        for p in PHI_TICKS:
            px, py = matrix_line(name, p)
            ax.plot([px], [py], marker="|", ms=5.0, mew=1.3, color=col,
                    zorder=5)
            if name == "Dolomite":
                ax.text(px + 0.007, py + 0.010, f"{p * 100:.0f}", ha="left",
                        va="top", fontsize=6.2, color=col, zorder=5)
        # label where the line still has room inside the frame
        nx_l, ny_l = matrix_line(name, phi_line)
        ok = ((nx_l < X_LIM[1] - 0.09) & (ny_l > Y_LIM[1] + 0.09)
              & (ny_l < Y_LIM[0]))
        j = int(np.flatnonzero(ok)[-1])
        ax.text(nx_l[j] + 0.008, ny_l[j] + 0.012, name, ha="left", va="top",
                fontsize=6.8, color=col, fontweight="bold", zorder=5)

    for g in allg:
        mk = MATRIX[g["name"]][4] if g["name"] in MATRIX else (
            "D" if g["name"].startswith("Gas") else "v")
        ax.scatter(g["nphi"], g["rhob"], c=g["gr"], cmap=cmap, vmin=vmin,
                   vmax=vmax, s=11, marker=mk, linewidths=0.25,
                   edgecolors="#555555", alpha=0.92, zorder=3)
    sc = ax.scatter([], [], c=[], cmap=cmap, vmin=vmin, vmax=vmax)
    safe_colorbar(fig, sc, ax, label="gamma ray (API)", width=0.11, pad=0.07)

    # gas arrow: measured shift between the wet and gas sandstone clouds
    wet = groups[0]
    band = (wet["phi"] > 0.12) & (wet["phi"] < 0.26)
    p0 = (float(wet["nphi"][band].mean()), float(wet["rhob"][band].mean()))
    p1 = (float(gas_grp["nphi"].mean()), float(gas_grp["rhob"].mean()))
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=GAS, lw=1.3, shrinkA=0, shrinkB=0,
                                 zorder=6))
    ax.text(p1[0] - 0.010, p1[1] - 0.030,
            "gas effect\nNPHI %+.3f, RHOB %+.2f" % (p1[0] - p0[0],
                                                    p1[1] - p0[1]),
            ha="right", va="bottom", fontsize=6.4, color=GAS, zorder=6,
            linespacing=1.35)

    handles = [Line2D([], [], ls="none", marker=MATRIX[k][4], ms=4.2,
                      mfc="#bbbbbb", mec="#555555", mew=0.4, label=k)
               for k in MATRIX]
    handles += [Line2D([], [], ls="none", marker="D", ms=4.2, mfc="#bbbbbb",
                       mec="#555555", mew=0.4, label="Gas sandstone"),
                Line2D([], [], ls="none", marker="v", ms=4.2, mfc="#bbbbbb",
                       mec="#555555", mew=0.4, label="Shaly sandstone")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.0,
              handletextpad=0.5, labelspacing=0.35)
    ax.set_title("Matrix lines ticked every 5 porosity units; colour is "
                 "shaliness", fontsize=8.8, fontweight="bold")

    # ---- M-N crossplot ------------------------------------------------------
    ax_mn.grid(color=GRID, lw=0.45, alpha=0.9)
    ax_mn.set_axisbelow(True)
    ax_mn.set_xlabel("N (-)", fontsize=7.4)
    ax_mn.set_ylabel("M (-)", fontsize=7.4)
    ax_mn.tick_params(labelsize=6.6)
    for g in allg:
        m, n = mn(g["nphi"], g["rhob"], g["dt"])
        mk = MATRIX[g["name"]][4] if g["name"] in MATRIX else (
            "D" if g["name"].startswith("Gas") else "v")
        col = (MATRIX[g["name"]][3] if g["name"] in MATRIX
               else (GAS if g["name"].startswith("Gas") else "#8a8a8a"))
        ax_mn.scatter(n, m, s=6, marker=mk, color=col, alpha=0.65,
                      linewidths=0.0, zorder=3)
    for name, (rho_ma, off, dt_ma, col, _) in MATRIX.items():
        m0, n0 = mn(off, rho_ma, dt_ma)
        ax_mn.plot([n0], [m0], marker="*", ms=9.0, mfc=col, mec="white",
                   mew=0.6, zorder=6)
        t = ax_mn.annotate(name[:4], (n0, m0), textcoords="offset points",
                           xytext=(5, 5), fontsize=6.2, color=col,
                           fontweight="bold", zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
    ax_mn.set_title("M-N separates matrix without porosity", fontsize=8.5,
                    fontweight="bold")

    # ---- porosity histogram -------------------------------------------------
    ax_h.grid(axis="y", color=GRID, lw=0.45, alpha=0.9)
    ax_h.set_axisbelow(True)
    bins = np.linspace(0.0, 0.36, 25)
    stack = [g["phi"] for g in groups] + [gas_grp["phi"]]
    cols = [MATRIX[g["name"]][3] for g in groups] + [GAS]
    labs = [g["name"] for g in groups] + ["Gas sandstone"]
    ax_h.hist(stack, bins=bins, stacked=True, color=cols, label=labs,
              edgecolor="white", linewidth=0.35)
    phi_pool = np.concatenate(stack)
    ax_h.axvline(float(phi_pool.mean()), color=RULE, lw=1.0, ls=(0, (3, 2)))
    ax_h.text(float(phi_pool.mean()) + 0.006, 0.96,
              "mean %.3f v/v" % phi_pool.mean(), transform=ax_h.get_xaxis_transform(),
              ha="left", va="top", fontsize=6.4, color=RULE)
    ax_h.set_xlabel("Porosity (v/v)", fontsize=7.4)
    ax_h.set_ylabel("Samples (n)", fontsize=7.4)
    ax_h.tick_params(labelsize=6.6)
    ax_h.legend(frameon=False, fontsize=6.2, loc="upper left",
                labelspacing=0.3, handlelength=1.1, handletextpad=0.5)
    ax_h.set_title("Porosity by assigned matrix", fontsize=8.5,
                   fontweight="bold")

    means = ", ".join("%s %.1f p.u." % (g["name"][:4], 100.0 * g["phi"].mean())
                      for g in groups)
    panel_labels_inches(fig, [ax, ax_mn, ax_h], dx=-0.60, dy=0.30)
    savefig(fig, "811_wlg_crossplot_lithology.png")


if __name__ == "__main__":
    main()
