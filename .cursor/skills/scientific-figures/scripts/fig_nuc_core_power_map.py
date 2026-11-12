"""Assembly-wise power distribution of a PWR core at hot full power.

The main panel is the claim: a quarter-symmetric loading pattern in which the
hottest assembly sits on the second ring rather than at the centre, because
the low-leakage pattern depresses the middle.  Every annotated number - the
peak assembly factor, its power in MW, the radial and axial peaking factors -
is measured from the same field that is painted, so a change of seed moves the
text with the colours.  The two smaller panels carry the supporting evidence:
how assembly power falls with radius, and how the axial shape flattens from a
clean cosine at start of cycle into a burnup-distorted saddle at end of cycle.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"        # start of cycle / nominal
RED = "#b23a48"         # the single emphasis: hottest assembly and limits
POWER_CMAP = get_cmap("cb_sequential_warm")

CORE_MW = 3411.0        # core thermal power
N_LAT = 15              # assemblies across the flat of the lattice
PITCH = 0.215           # assembly pitch, m
HEIGHT = 3.66           # active fuel height, m
F_LIMIT = 1.55          # technical-specification limit on F_xy
ROD_SEED = [(0, 0), (4, 0), (0, 4), (2, 2), (4, 4), (6, 2), (2, 6), (6, 6)]


def core_field(seed=7):
    """Assembly peaking factors on the lattice; NaN outside the barrel."""
    rng = np.random.default_rng(seed)
    idx = np.arange(N_LAT) - (N_LAT - 1) / 2.0
    x, y = np.meshgrid(idx, idx)
    r = np.hypot(x, y)
    inside = r <= 7.15

    rn = np.clip(r / 7.8, 0.0, 1.0)
    shape = np.cos(0.5 * np.pi * rn ** 1.45)
    shape *= 1.0 - 0.26 * np.exp(-(rn / 0.30) ** 2)      # low-leakage dip
    shape *= 1.0 + 0.05 * rng.normal(size=shape.shape)   # loading asymmetry
    field = np.where(inside, shape, np.nan)
    return x, y, inside, field / np.nanmean(field)


def rod_cells(inside):
    """Symmetry-expanded control-rod bank locations that fall inside the core."""
    pts = set()
    for a, b in ROD_SEED:
        for sa in (1, -1):
            for sb in (1, -1):
                pts.add((sa * a, sb * b))
    half = (N_LAT - 1) / 2.0
    keep = []
    for cx, cy in sorted(pts):
        i, j = int(cy + half), int(cx + half)
        if 0 <= i < N_LAT and 0 <= j < N_LAT and inside[i, j]:
            keep.append((cx, cy))
    return keep


def axial_shapes(n=241):
    """Cosine start-of-cycle shape and its burnup-distorted end-of-cycle twin."""
    z = np.linspace(0.0, HEIGHT, n)
    u = (z - HEIGHT / 2.0) / (HEIGHT / 2.0)
    boc = np.cos(0.5 * np.pi * u / 1.12)
    eoc = boc * (1.0 - 0.34 * np.exp(-(u / 0.42) ** 2))
    boc /= boc.mean()
    eoc /= eoc.mean()
    return z, boc, eoc


def map_panel(fig, ax, x, y, inside, field, rods):
    norm = mpl.colors.Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    for i in range(N_LAT):
        for j in range(N_LAT):
            if not inside[i, j]:
                continue
            ax.add_patch(Rectangle(
                ((x[i, j] - 0.5) * PITCH, (y[i, j] - 0.5) * PITCH),
                PITCH, PITCH, facecolor=POWER_CMAP(norm(field[i, j])),
                edgecolor="white", linewidth=0.5, zorder=2))

    rx = [c[0] * PITCH for c in rods]
    ry = [c[1] * PITCH for c in rods]
    ax.plot(rx, ry, ls="none", marker="o", ms=3.6, mfc="white", mec=RULE,
            mew=0.8, zorder=4)

    k = np.unravel_index(np.nanargmax(field), field.shape)
    hx, hy = x[k] * PITCH, y[k] * PITCH
    n_asm = int(inside.sum())
    hot_mw = float(np.nanmax(field)) * CORE_MW / n_asm
    ax.add_patch(Rectangle((hx - 0.5 * PITCH, hy - 0.5 * PITCH), PITCH, PITCH,
                           facecolor="none", edgecolor=RED, linewidth=1.5,
                           zorder=6))
    ax.annotate(f"hottest assembly\n$F_{{xy}}$ = {np.nanmax(field):.2f},"
                f"  {hot_mw:.1f} MW",
                xy=(hx + 0.45 * PITCH, hy + 0.45 * PITCH),
                xytext=(1.58, 1.36), ha="right", va="center", fontsize=7.0,
                color=RED, linespacing=1.35, zorder=7,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    lim = 7.7 * PITCH
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("core x (m)")
    ax.set_ylabel("core y (m)")
    ax.set_title(f"{n_asm} assemblies, quarter-core symmetry",
                 fontsize=9.0, fontweight="bold", color=INK)

    handles = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white",
                      mec=RULE, mew=0.8, label="control rod bank"),
               Line2D([], [], color=RED, lw=1.5, label="limiting assembly")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False,
              handlelength=1.4, borderaxespad=0.2)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=POWER_CMAP)
    safe_colorbar(fig, sm, ax, label=r"assembly peaking factor $F_{xy}$ (-)",
                  width=0.11, pad=0.08, labelsize=7.0)
    return norm, n_asm, hot_mw


def radial_panel(ax, x, y, inside, field, n_asm):
    r = np.hypot(x[inside], y[inside]) * PITCH
    mw = field[inside] * CORE_MW / n_asm
    ax.plot(r, mw, ls="none", marker="o", ms=2.6, mfc=BLUE, mec="white",
            mew=0.35, alpha=0.85, zorder=3)

    edges = np.linspace(0.0, r.max() * 1.001, 9)
    mid = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([mw[(r >= a) & (r < b)].mean() if ((r >= a) & (r < b)).any()
                     else np.nan for a, b in zip(edges[:-1], edges[1:])])
    ax.plot(mid, prof, color=RULE, lw=1.4, zorder=4)

    lim_mw = F_LIMIT * CORE_MW / n_asm
    ax.axhline(lim_mw, color=RED, lw=1.0, ls=(0, (4, 2.4)), zorder=5)
    ax.text(r.max(), lim_mw, f"  limit {lim_mw:.1f} MW", ha="right",
            va="bottom", fontsize=6.5, color=RED)
    ax.set_ylim(0.0, lim_mw * 1.16)
    ax.set_xlim(-0.05, r.max() + 0.10)
    ax.set_xlabel("distance from core axis (m)", fontsize=7.4)
    ax.set_ylabel("assembly power (MW)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(f"Radial power falls to {prof[-1] / prof[0]:.2f} of the "
                 f"inner ring", fontsize=8.2, fontweight="bold", color=INK)


def axial_panel(ax, z, boc, eoc):
    ax.plot(boc, z, color=BLUE, lw=1.6, zorder=4, label="start of cycle")
    ax.plot(eoc, z, color=RED, lw=1.6, ls=(0, (5, 2.2)), zorder=4,
            label="end of cycle")
    ax.axvline(1.0, color=GREY, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)

    fz_b, fz_e = float(boc.max()), float(eoc.max())
    z_b = float(z[np.argmax(boc)])
    ax.plot([fz_b], [z_b], marker="o", ms=3.4, mfc="white", mec=BLUE, mew=1.0,
            zorder=5)
    ax.text(fz_b + 0.05, z_b, f"$F_z$ = {fz_b:.2f}", ha="left", va="center",
            fontsize=6.5, color=BLUE)
    ax.text(1.76, HEIGHT * 0.045,
            f"burnup flattens\nthe peak to {fz_e:.2f}", ha="right",
            va="bottom", fontsize=6.5, color=RED, linespacing=1.35)

    ax.set_ylim(0.0, HEIGHT)
    ax.set_xlim(0.0, 1.80)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel(r"axial peaking factor $F_z$ (-)", fontsize=7.4)
    ax.set_ylabel("elevation above core inlet (m)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.2)
    ax.set_title("Axial shape, cosine to saddle", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    x, y, inside, field = core_field()
    rods = rod_cells(inside)
    z, boc, eoc = axial_shapes()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], left=0.075, right=0.895,
                          top=0.895, bottom=0.115, wspace=0.42, hspace=0.66)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_rad = fig.add_subplot(gs[0, 1])
    ax_ax = fig.add_subplot(gs[1, 1])

    _, n_asm, hot_mw = map_panel(fig, ax_map, x, y, inside, field, rods)
    radial_panel(ax_rad, x, y, inside, field, n_asm)
    axial_panel(ax_ax, z, boc, eoc)

    dens = CORE_MW / (n_asm * PITCH ** 2 * HEIGHT)
    panel_labels_inches(fig, [ax_map, ax_rad, ax_ax], dx=-0.44, dy=0.12)
    savefig(fig, "654_nuc_core_power_map.png")


if __name__ == "__main__":
    main()
