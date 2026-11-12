"""Axonometric structural frame coloured by demand-capacity ratio.

Columns, beams and slabs come from one grid definition, and every member is
shaded by the same dimensionless demand-capacity ratio so that a column in
kilonewtons and a beam in kilonewton-metres can honestly share one colour
scale.  Members that fail that ratio are re-drawn heavy in house red - the
single emphasis of the figure.  The two smaller panels are the code checks
that follow from the same analysis: inter-storey drift against the 1/550
limit, and the distribution of utilisation across all members.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from pubstyle import set_pub_style, get_palette, add_colorbar, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"
GREY = "#808080"
BLUE = "#3b5b92"

NX, NY, NZ = 4, 3, 5              # grid lines in x, y and number of storeys
DX, DY, DZ = 6.0, 7.5, 3.6        # bay widths and storey height, metres
LOAD = 12.0                       # gravity load on the floor plate, kN/m2
COL_CAP = 2800.0                  # column axial capacity, kN
BEAM_CAP = 330.0                  # beam moment capacity, kN m
DRIFT_LIMIT = 1000.0 / 550.0      # 1/550 expressed in per mille
# The two palest stops of the house blue ramp vanish against white paper, and
# a lightly loaded member still has to be visible as a line in the frame.
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "frame_load", get_palette("cb_sequential_blue")[2:], N=256)


def frame(seed=23):
    """Member list of (p0, p1, ratio, demand, unit) for the whole frame."""
    rng = np.random.default_rng(seed)
    xs = np.arange(NX) * DX
    ys = np.arange(NY) * DY
    members = []

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            fx = 0.5 if i in (0, NX - 1) else 1.0
            fy = 0.5 if j in (0, NY - 1) else 1.0
            trib = fx * fy * DX * DY
            # Lateral demand grows toward the leeward face and the perimeter.
            lever = (0.22 * x / xs.max()
                     + 0.12 * abs(y - ys.mean()) / ys.mean())
            for s in range(NZ):
                axial = trib * LOAD * (NZ - s) * (1.0 + lever)
                axial *= 1.0 + rng.normal(0.0, 0.045)
                members.append(((x, y, s * DZ), (x, y, (s + 1) * DZ),
                                axial / COL_CAP, axial, "kN"))

    for s in range(1, NZ + 1):
        z = s * DZ
        height = 1.0 - 0.42 * (s - 1) / (NZ - 1)      # upper floors lighter
        for j, y in enumerate(ys):
            for i in range(NX - 1):
                m = LOAD * DY * DX ** 2 / 11.0 * height
                m *= 1.0 + rng.normal(0.0, 0.06)
                members.append(((xs[i], y, z), (xs[i + 1], y, z),
                                m / BEAM_CAP, m, "kN m"))
        for i, x in enumerate(xs):
            for j in range(NY - 1):
                m = LOAD * DX * DY ** 2 / 13.0 * height
                m *= 1.0 + rng.normal(0.0, 0.06)
                members.append(((x, ys[j], z), (x, ys[j + 1], z),
                                m / BEAM_CAP, m, "kN m"))
    return members


def drift_profile(seed=9):
    rng = np.random.default_rng(seed)
    floors = np.arange(1, NZ + 1)
    shape = np.array([1.00, 1.22, 1.14, 0.92, 0.66])   # shear-type response
    dx_ = 1.72 * shape * (1.0 + rng.normal(0.0, 0.03, NZ))
    dy_ = 1.34 * shape * (1.0 + rng.normal(0.0, 0.03, NZ))
    return floors, dx_, dy_


def frame_panel(fig, ax, members):
    ratios = np.array([m[2] for m in members])
    norm = mpl.colors.Normalize(vmin=float(ratios.min()),
                                vmax=float(ratios.max()))

    ok = [m for m in members if m[2] <= 1.0]
    bad = [m for m in members if m[2] > 1.0]
    if ok:
        ax.add_collection3d(Line3DCollection(
            [(m[0], m[1]) for m in ok], linewidths=1.5,
            colors=[SEQ_CMAP(norm(m[2])) for m in ok], zorder=3))
    if bad:
        ax.add_collection3d(Line3DCollection(
            [(m[0], m[1]) for m in bad], linewidths=2.9, colors=ACCENT,
            zorder=5))

    span_x, span_y = (NX - 1) * DX, (NY - 1) * DY
    slabs = [[(0.0, 0.0, s * DZ), (span_x, 0.0, s * DZ),
              (span_x, span_y, s * DZ), (0.0, span_y, s * DZ)]
             for s in range(1, NZ + 1)]
    pc = Poly3DCollection(slabs, facecolor="#b3b3b3", alpha=0.13,
                          edgecolor="#8a8a8a", linewidths=0.4)
    pc.set_zsort("min")
    ax.add_collection3d(pc)

    for s in range(1, NZ + 1):
        ax.text(-2.4, 0.0, s * DZ, f"L{s}", fontsize=6.4, color=MUTED,
                ha="right", va="center")

    ax.set_xlim(0.0, span_x)
    ax.set_ylim(0.0, span_y)
    ax.set_zlim(0.0, NZ * DZ)
    ax.set_box_aspect((span_x, span_y, NZ * DZ))
    ax.view_init(elev=17.0, azim=-62.0)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

    worst = max(members, key=lambda m: m[2])
    ax.text2D(0.015, 0.085,
              f"{len(bad)} of {len(members)} members exceed capacity; worst "
              f"ratio {worst[2]:.2f} at {worst[3]:.0f} {worst[4]}",
              transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
              color=ACCENT)
    ax.text2D(0.015, 0.030,
              f"{NX - 1} x {NY - 1} bays at {DX:.1f} m x {DY:.1f} m, "
              f"{NZ} storeys at {DZ:.2f} m", transform=ax.transAxes,
              ha="left", va="bottom", fontsize=6.8, color=MUTED)
    ax.set_title("Frame members shaded by demand-capacity ratio",
                 fontsize=9.0, fontweight="bold", color=INK, y=0.99)
    return mpl.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP), len(bad)


def drift_panel(ax, floors, dx_, dy_):
    for vals, color, name in ((dx_, BLUE, "X direction"),
                              (dy_, "#2e7d5b", "Y direction")):
        ax.plot(vals, floors, marker="o", ms=3.2, lw=1.5, color=color,
                mfc="white", mew=0.9, label=name)
    ax.axvline(DRIFT_LIMIT, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)))
    ax.text(DRIFT_LIMIT, NZ + 0.45, "1/550 limit", ha="center", va="bottom",
            fontsize=6.4, color=ACCENT)
    over = int(np.count_nonzero(dx_ > DRIFT_LIMIT))
    ax.set_yticks(floors)
    ax.set_yticklabels([f"L{f}" for f in floors], fontsize=6.5)
    ax.set_xlim(0.0, max(dx_.max(), DRIFT_LIMIT) * 1.28)
    ax.set_ylim(0.4, NZ + 1.0)
    ax.set_xlabel("inter-storey drift ratio (\u2030)", fontsize=7.4)
    ax.set_ylabel("storey (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right",
              handlelength=1.4, borderaxespad=0.2)
    ax.set_title(f"Drift exceeds the limit on {over} storeys", fontsize=8.2,
                 fontweight="bold", color=INK)


def util_panel(ax, members):
    ratios = np.array([m[2] for m in members])
    bins = np.linspace(0.0, np.ceil(ratios.max() * 10) / 10.0, 24)
    counts, edges = np.histogram(ratios, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    colors = [ACCENT if c > 1.0 else BLUE for c in centres]
    ax.bar(centres, counts, width=np.diff(edges) * 0.9, color=colors,
           edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)))
    ax.text(0.98, ax.get_ylim()[1] * 0.94, "capacity", ha="right", va="top",
            fontsize=6.4, color=ACCENT)
    ax.set_xlabel("demand-capacity ratio (-)", fontsize=7.4)
    ax.set_ylabel("members (count)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title(f"Median utilisation {np.median(ratios):.2f}", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    members = frame()
    floors, dx_, dy_ = drift_profile()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_dr = fig.add_subplot(gs[0, 1])
    ax_ut = fig.add_subplot(gs[1, 1])

    sm, n_bad = frame_panel(fig, ax3d, members)
    drift_panel(ax_dr, floors, dx_, dy_)
    util_panel(ax_ut, members)

    bad = [m for m in members if m[2] > 1.0]
    n_lvl = int(np.ceil(max(max(m[0][2], m[1][2]) for m in bad) / DZ))
    where = "storey" if n_lvl == 1 else f"{n_lvl} storeys"
    fig.tight_layout(rect=[0, 0.040, 1, 0.945], h_pad=2.6, w_pad=2.6)
    cb = add_colorbar(fig, sm, ax3d, label="demand-capacity ratio (-)",
                      fraction=0.030, pad=0.00, shrink=0.62, aspect=16,
                      labelsize=6.6)
    cb.ax.set_zorder(6)
    panel_labels_inches(fig, [ax3d, ax_dr, ax_ut], dx=-0.30, dy=0.06)
    savefig(fig, "591_bim_structural_frame.png")


if __name__ == "__main__":
    main()
