"""Spread mooring of a semi-submersible: layout, offset and line tension.

The main panel is the plan view - eight chain legs in four bundles, drawn to
scale with their azimuths and pretensions, and the platform's offset track in
an inset because a 130 m excursion is invisible next to an 800 m anchor
radius.  Everything downstream is derived from that same track: each leg's
tension history is the track projected onto the leg azimuth and pushed through
the tension-offset curve of the right panel, so the box statistics and the
quoted safety factors cannot drift away from the geometry that produced them.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"        # intact condition
RED = "#b23a48"         # damaged condition and the governing leg
GOLD = "#c9a227"        # offset track
STEEL = "#c4c4c4"

DEPTH = 1000.0          # water depth, m
ANCHOR_R = 800.0        # anchor radius, m
HULL_R = 55.0           # hull half-breadth, m
T_PRE = 1200.0          # pretension per leg, kN
MBL = 9000.0            # minimum breaking load, kN
SF_INTACT, SF_DAMAGED = 1.67, 1.25
WATCH = 150.0           # design watch circle radius, m
STIFFEN = 150.0         # offset that triples the tension, m
DAMAGED_GAIN = 1.45     # load redistribution after one leg parts
BUNDLES = [0.0, 90.0, 180.0, 270.0]
SPREAD = 11.0           # half angle inside a bundle, deg


def legs():
    out = []
    for b in BUNDLES:
        for s in (-SPREAD, SPREAD):
            out.append((b + s) % 360.0)
    return np.array(sorted(out))


def tension(offset, gain=1.0):
    """Chain tension in kN for a surge along the leg, catenary stiffening."""
    return gain * T_PRE * np.exp(np.log(3.0) * offset / STIFFEN)


def track(seed=11, n=4200):
    """Slow drift plus wave-frequency surge, in metres."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 3.0 * 3600.0, n)
    mean_dir = np.deg2rad(212.0)
    drift = 78.0 + 16.0 * np.sin(2.0 * np.pi * t / 900.0)
    slow = np.cumsum(rng.normal(0.0, 1.0, n))
    slow = 22.0 * slow / np.abs(slow).max()
    wave = 9.0 * np.sin(2.0 * np.pi * t / 12.5 + 0.6 * np.cos(t / 240.0))
    r = drift + slow + wave
    ang = mean_dir + np.deg2rad(9.0) * np.sin(2.0 * np.pi * t / 1500.0)
    ang = ang + np.deg2rad(3.0) * np.sin(2.0 * np.pi * t / 130.0)
    return t, r * np.cos(ang), r * np.sin(ang)


def leg_tensions(x, y, az):
    """Tension history of every leg, kN, from the plotted offset track."""
    out = {}
    for a in az:
        u = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        out[a] = tension(-(x * u[0] + y * u[1]))
    return out


def layout_panel(ax, az, x, y):
    lim = ANCHOR_R * 1.16
    ax.add_patch(Circle((0.0, 0.0), WATCH, facecolor="none", edgecolor=GOLD,
                        lw=0.9, ls=(0, (4, 2.4)), zorder=3))
    for a in az:
        ang = np.deg2rad(a)
        ax.plot([HULL_R * np.cos(ang), ANCHOR_R * np.cos(ang)],
                [HULL_R * np.sin(ang), ANCHOR_R * np.sin(ang)],
                color=STEEL, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([ANCHOR_R * np.cos(ang)], [ANCHOR_R * np.sin(ang)],
                marker="^", ms=4.6, mfc="white", mec=RULE, mew=0.9, zorder=4)
    for b in BUNDLES:
        ang = np.deg2rad(b)
        rx, ry = 0.92 * lim * np.cos(ang), 0.92 * lim * np.sin(ang)
        ax.text(rx, ry, f"{b:03.0f}" r"$^{\circ}$" "\n"
                f"2 x {T_PRE:.0f} kN",
                ha="center", va="center", fontsize=6.8, color=SUB,
                linespacing=1.35, zorder=5)

    hull = np.array([[HULL_R * np.cos(t), HULL_R * np.sin(t)]
                     for t in np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8])
    ax.add_patch(Polygon(hull, closed=True, facecolor="#e2e2e2",
                         edgecolor="#555555", lw=0.9, zorder=6))
    ax.plot(x, y, color=GOLD, lw=0.45, alpha=0.85, zorder=7)

    env = np.deg2rad(212.0)
    ax.annotate("environment 212" r"$^{\circ}$",
                xy=(0.80 * lim * np.cos(env), 0.80 * lim * np.sin(env)),
                xytext=(0.36 * lim * np.cos(env), 0.36 * lim * np.sin(env)),
                ha="center", va="bottom", fontsize=7.0, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0,
                                mutation_scale=9))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("east offset from field centre (m)")
    ax.set_ylabel("north offset from field centre (m)")
    ax.set_title(f"{len(az)} chain legs, {ANCHOR_R:.0f} m anchor radius in "
                 f"{DEPTH:.0f} m water", fontsize=9.0, fontweight="bold",
                 color=INK)
    handles = [Line2D([], [], color=STEEL, lw=1.4, label="chain leg"),
               Line2D([], [], ls="none", marker="^", ms=4.6, mfc="white",
                      mec=RULE, mew=0.9, label="anchor"),
               Line2D([], [], color=GOLD, lw=1.0, label="offset track"),
               Line2D([], [], color=GOLD, lw=0.9, ls=(0, (4, 2.4)),
                      label=f"watch circle {WATCH:.0f} m")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, frameon=False,
              handlelength=1.6, borderaxespad=0.2)


def offset_inset(ax, x, y):
    axi = ax.inset_axes([0.655, 0.030, 0.330, 0.330])
    axi.set_zorder(8)
    axi.patch.set_visible(True)
    axi.add_patch(Circle((0.0, 0.0), WATCH, facecolor="none", edgecolor=GOLD,
                         lw=0.9, ls=(0, (4, 2.4)), zorder=3))
    axi.plot(x, y, color=GOLD, lw=0.4, alpha=0.9, zorder=4)
    r = np.hypot(x, y)
    k = int(np.argmax(r))
    axi.plot([x[k]], [y[k]], marker="o", ms=3.6, mfc="white", mec=RED,
             mew=1.1, zorder=6)
    axi.text(x[k] - 8.0, y[k] - 8.0, f"max {r[k]:.0f} m", ha="right",
             va="top", fontsize=6.2, color=RED, zorder=6)
    axi.plot([0.0], [0.0], marker="+", ms=5.0, color=RULE, mew=0.9, zorder=5)
    lim = WATCH * 1.12
    axi.set_xlim(-lim, lim)
    axi.set_ylim(-lim, lim)
    axi.set_aspect("equal")
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.set_xticks([-100, 0, 100])
    axi.set_yticks([-100, 0, 100])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    axi.set_title("offset detail (m)", fontsize=6.5, color=SUB, pad=2.0)
    return r[k]


def curve_panel(ax, r_max):
    s = np.linspace(-40.0, 230.0, 300)
    intact = tension(s)
    damaged = tension(s, DAMAGED_GAIN)
    ax.plot(s, intact, color=BLUE, lw=1.6, zorder=4, label="intact")
    ax.plot(s, damaged, color=RED, lw=1.6, ls=(0, (5, 2.2)), zorder=4,
            label="one leg parted")

    for lab, sf, color in [("intact allowable", SF_INTACT, BLUE),
                           ("damaged allowable", SF_DAMAGED, RED)]:
        ax.axhline(MBL / sf, color=color, lw=0.9, ls=(0, (2.2, 2.0)),
                   zorder=3)
        ax.text(s[0] + 4.0, MBL / sf, f"{lab} {MBL / sf:.0f} kN", ha="left",
                va="bottom", fontsize=6.2, color=color)
    ax.axvline(r_max, color=GREY, lw=0.9, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(r_max - 6.0, MBL / SF_DAMAGED * 1.30,
            f"track maximum\n{r_max:.0f} m offset", ha="right", va="top",
            fontsize=6.5, color=SUB, linespacing=1.35)

    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(0.0, MBL / SF_DAMAGED * 1.34)
    ax.set_xlabel("surge along the leg (m)", fontsize=7.4)
    ax.set_ylabel("leg tension (kN)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.2)
    ax.set_title("Tension stiffens with offset", fontsize=8.2,
                 fontweight="bold", color=INK)


def box_panel(ax, hist):
    az = list(hist)
    data = [hist[a] for a in az]
    peak = int(np.argmax([d.max() for d in data]))
    bp = ax.boxplot(data, widths=0.62, patch_artist=True, whis=(5, 95),
                    showfliers=False,
                    medianprops=dict(color="white", lw=1.0),
                    whiskerprops=dict(color=RULE, lw=0.7),
                    capprops=dict(color=RULE, lw=0.7))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(RED if i == peak else BLUE)
        box.set_edgecolor("white")
        box.set_linewidth(0.5)

    allow = MBL / SF_INTACT
    ax.axhline(allow, color=BLUE, lw=0.9, ls=(0, (2.2, 2.0)), zorder=3)
    sf = MBL / max(d.max() for d in data)
    ax.text(len(az) + 0.35, allow, f"allowable {allow:.0f} kN", ha="right",
            va="bottom", fontsize=6.2, color=BLUE)
    left = peak < len(az) / 2.0
    ax.annotate(f"governing leg {az[peak]:03.0f}" r"$^{\circ}$" "\n"
                f"SF = {sf:.2f}",
                xy=(peak + 1, data[peak].max()),
                xytext=(peak + 1 + (1.4 if left else -1.4),
                        data[peak].max() * 1.55),
                ha="left" if left else "right", va="bottom", fontsize=6.5,
                color=RED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    ax.set_xticks(np.arange(1, len(az) + 1))
    ax.set_xticklabels([f"{a:03.0f}" for a in az], fontsize=6.2, rotation=45)
    ax.set_xlabel(r"leg azimuth ($^{\circ}$)", fontsize=7.4)
    ax.set_ylabel("tension (kN)", fontsize=7.4)
    ax.set_ylim(0.0, allow * 1.20)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Three hours of tension per leg", fontsize=8.2,
                 fontweight="bold", color=INK)
    return sf, az[peak]


def main():
    set_pub_style()
    az = legs()
    t, x, y = track()
    hist = leg_tensions(x, y, az)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], left=0.070, right=0.985,
                          top=0.890, bottom=0.125, wspace=0.26, hspace=0.66)
    ax_plan = fig.add_subplot(gs[:, 0])
    ax_cur = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[1, 1])

    layout_panel(ax_plan, az, x, y)
    r_max = offset_inset(ax_plan, x, y)
    curve_panel(ax_cur, r_max)
    sf, gov = box_panel(ax_box, hist)

    annotate_box(ax_plan, f"maximum offset {r_max:.0f} m "
                          f"({r_max / DEPTH * 100:.1f} % of depth)\n"
                          f"governing leg {gov:03.0f}"
                          r"$^{\circ}$" f", safety factor {sf:.2f}\n"
                          f"pretension {T_PRE:.0f} kN per leg, MBL "
                          f"{MBL / 1e3:.1f} MN",
                 loc="lower left", fontsize=6.8)

    panel_labels_inches(fig, [ax_plan, ax_cur, ax_box], dx=-0.44, dy=0.12)
    savefig(fig, "658_oce_mooring_analysis.png")


if __name__ == "__main__":
    main()
