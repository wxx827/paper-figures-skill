"""Sewer long section: the water surface is computed, not sketched.

Nine reaches of a gravity sewer are drawn as a long section - ground line,
pipe soffit and invert, manholes and the water surface - with the vertical
scale exaggerated and the exaggeration factor stated on the panel, because a
long section at true scale is an unreadable horizontal line.  Normal depth in
each reach is the root of the Manning discharge equation solved for the
part-full circular section, so the filling ratio, the velocity and the water
surface are three views of the same root: the section geometry uses the exact
circular-segment formulas rather than an approximation.  The side panels check
each reach against the two rules that decide whether a sewer works, a maximum
filling ratio and a self-cleansing velocity.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
DEEP = "#134a76"
PALE = "#9cc3dd"
GREEN = "#2e7d5b"
BROWN = "#7a6a5d"
SOIL = "#d8cbb8"
RED = "#b23a48"
ORANGE = "#c0603f"

SEED = 1007
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

MANNING = 0.013            # roughness of a concrete sewer, s per m^(1/3)
FILL_MAX = 0.75            # maximum filling ratio for this pipe size
V_MIN = 0.60               # self-cleansing velocity, m/s
V_MAX = 4.0                # scour limit, m/s

# chainage (m), length (m), diameter (m), gradient (-), local inflow (L/s)
REACHES = [
    (0.0, 220.0, 0.300, 0.00165, 26.0),
    (220.0, 175.0, 0.300, 0.00367, 18.0),
    (395.0, 190.0, 0.400, 0.00286, 34.0),
    (585.0, 205.0, 0.400, 0.00319, 22.0),
    (790.0, 160.0, 0.500, 0.00269, 41.0),
    (950.0, 185.0, 0.500, 0.00188, 15.0),
    (1135.0, 210.0, 0.600, 0.00202, 37.0),
    (1345.0, 175.0, 0.600, 0.00183, 12.0),
    (1520.0, 195.0, 0.600, 0.00192, 24.0),
]
START_INVERT = 41.20       # invert level at the head manhole, m
Y_LO, Y_HI = 36.2, 46.2    # level axis of the long section, m


def part_full(theta, dia):
    """Area (m2), wetted perimeter (m) and depth (m) for a filling angle."""
    area = dia ** 2 / 8.0 * (theta - np.sin(theta))
    perim = dia * theta / 2.0
    depth = dia / 2.0 * (1.0 - np.cos(theta / 2.0))
    return area, perim, depth


def manning_q(theta, dia, slope):
    """Discharge (m3/s) of a part-full circular pipe at a filling angle."""
    area, perim, _d = part_full(theta, dia)
    if perim <= 0.0:
        return 0.0
    return area / MANNING * (area / perim) ** (2.0 / 3.0) * np.sqrt(slope)


def normal_depth(q, dia, slope):
    """Filling angle that carries q; the root of Manning's equation."""
    full = manning_q(2.0 * np.pi - 1e-9, dia, slope)
    if q >= full:
        return 2.0 * np.pi - 1e-9, True
    f = lambda th: manning_q(th, dia, slope) - q
    return brentq(f, 1e-6, 2.0 * np.pi - 1e-9, xtol=1e-12), False


def solve_line():
    """Invert levels, depths, velocities and filling ratios reach by reach."""
    rows = []
    invert = START_INVERT
    q_cum = 0.0
    for (x0, length, dia, slope, inflow) in REACHES:
        q_cum += inflow / 1000.0
        drop = slope * length
        theta, surcharged = normal_depth(q_cum, dia, slope)
        area, perim, depth = part_full(theta, dia)
        vel = q_cum / area
        rows.append(dict(x0=x0, x1=x0 + length, length=length, dia=dia,
                         slope=slope, q=q_cum, theta=theta, depth=depth,
                         area=area, vel=vel, fill=depth / dia,
                         inv0=invert, inv1=invert - drop,
                         surcharged=surcharged, inflow=inflow))
        invert -= drop
    return rows


def ground_line(x, rng):
    """Ground surface: a gentle fall with a hill the sewer has to pass."""
    return (44.9 - 0.00095 * x + 0.85 * np.exp(-0.5 * ((x - 640.0) / 210.0)**2)
            - 0.45 * np.exp(-0.5 * ((x - 1420.0) / 260.0) ** 2)
            + 0.05 * np.sin(x / 95.0))


def main_panel(ax, rows, rng, vex):
    x = np.linspace(0.0, rows[-1]["x1"], 900)
    gl = ground_line(x, rng)

    inv = np.concatenate([[r["inv0"], r["inv1"]] for r in rows])
    xs = np.concatenate([[r["x0"], r["x1"]] for r in rows])
    sof = np.concatenate([[r["inv0"] + r["dia"], r["inv1"] + r["dia"]]
                          for r in rows])

    ax.fill_between(x, np.interp(x, xs, inv) - 0.9, gl, facecolor=SOIL,
                    alpha=0.45, edgecolor="none", zorder=1)
    ax.plot(x, gl, color=BROWN, lw=1.4, zorder=6)
    ax.plot(xs, sof, color=SUB, lw=1.0, zorder=5)
    ax.plot(xs, inv, color=SUB, lw=1.0, zorder=5)

    for r in rows:
        wx = np.array([r["x0"], r["x1"]])
        wy = np.array([r["inv0"] + r["depth"], r["inv1"] + r["depth"]])
        col = ORANGE if r["fill"] > FILL_MAX else PALE
        ax.fill_between(wx, [r["inv0"], r["inv1"]], wy, facecolor=col,
                        alpha=0.85, edgecolor="none", zorder=4)
        ax.plot(wx, wy, color=DEEP, lw=1.1, zorder=7)
        xm = 0.5 * (r["x0"] + r["x1"])
        ax.text(xm, 0.5 * (r["inv0"] + r["inv1"]) - 0.30,
                "DN{:.0f}\n1 in {:.0f}".format(1000.0 * r["dia"],
                                               1.0 / r["slope"]),
                fontsize=6.0, color=SUB, ha="center", va="top",
                linespacing=1.28, zorder=8)
        ax.text(xm, 0.5 * (r["inv0"] + r["inv1"] + 2 * r["depth"]) + 0.10,
                "{:.0f} %".format(100.0 * r["fill"]), fontsize=6.0,
                color=RED if r["fill"] > FILL_MAX else DEEP, ha="center",
                va="bottom", zorder=9)

    for r in rows + [None]:
        if r is None:
            xm, top, bot = rows[-1]["x1"], None, rows[-1]["inv1"]
        else:
            xm, bot = r["x0"], r["inv0"]
        gtop = float(ground_line(np.array([xm]), rng)[0])
        ax.add_patch(Rectangle((xm - 11.0, bot - 0.12), 22.0,
                               gtop - bot + 0.12, facecolor="white",
                               edgecolor=SUB, linewidth=0.8, zorder=3))

    deep = max(rows, key=lambda r: ground_line(np.array([r["x0"]]), rng)[0]
               - r["inv0"])
    cover = float(ground_line(np.array([deep["x0"]]), rng)[0]) - \
        deep["inv0"] - deep["dia"]
    ax.annotate("deepest manhole: {:.2f} m of cover\nover the soffit"
                .format(cover), xy=(deep["x0"], deep["inv0"] + deep["dia"]),
                xytext=(deep["x0"] + 90.0, deep["inv0"] + deep["dia"] + 1.9),
                fontsize=6.4, color=BROWN, ha="left", va="bottom",
                linespacing=1.32, zorder=10,
                arrowprops=dict(arrowstyle="-", color=BROWN, lw=0.8,
                                shrinkA=2, shrinkB=4))
    bad = [r for r in rows if r["fill"] > FILL_MAX]
    if bad:
        r = bad[0]
        ax.annotate("reach {:d} runs {:.0f} % full, over the {:.0f} % limit"
                    .format(rows.index(r) + 1, 100.0 * r["fill"],
                            100.0 * FILL_MAX),
                    xy=(0.5 * (r["x0"] + r["x1"]),
                        0.5 * (r["inv0"] + r["inv1"]) + r["depth"]),
                    xytext=(r["x1"] + 95.0,
                            0.5 * (r["inv0"] + r["inv1"]) + 1.5),
                    fontsize=6.4, color=RED, ha="left", va="bottom",
                    linespacing=1.32, zorder=10,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                    shrinkA=2, shrinkB=4))

    handles = [
        Line2D([], [], color=BROWN, lw=1.4, label="ground level"),
        Line2D([], [], color=SUB, lw=1.0, label="pipe soffit and invert"),
        Line2D([], [], color=DEEP, lw=1.1, label="water surface at design flow"),
        Patch(facecolor=PALE, edgecolor="none", label="flow within the limit"),
        Patch(facecolor=ORANGE, edgecolor="none",
              label="filling ratio above {:.0f} %".format(100.0 * FILL_MAX)),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, ncol=2, columnspacing=1.0,
              borderaxespad=0.3, labelspacing=0.26)

    ax.set_xlim(-30.0, rows[-1]["x1"] + 30.0)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("chainage along the sewer (m)")
    ax.set_ylabel("level above datum (m), axis increases upward")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Long section of {:d} reaches, vertical scale exaggerated "
                 "{:.0f} times".format(len(rows), vex), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def fill_panel(ax, rows):
    idx = np.arange(1, len(rows) + 1)
    fill = np.array([100.0 * r["fill"] for r in rows])
    cols = [ORANGE if f > 100.0 * FILL_MAX else BLUE for f in fill]
    ax.bar(idx, fill, width=0.66, color=cols, edgecolor="white", linewidth=0.6,
           zorder=4)
    ax.axhline(100.0 * FILL_MAX, color=RED, lw=1.0, ls=(0, (4.5, 2.4)),
               zorder=6)
    ax.text(len(rows) + 0.42, 100.0 * FILL_MAX + 2.0,
            "limit {:.0f} %".format(100.0 * FILL_MAX), fontsize=6.1,
            color=RED, ha="right", va="bottom", zorder=7)
    for i, r in zip(idx, rows):
        ax.text(i, 2.5, "{:.0f}".format(1000.0 * r["q"]), fontsize=6,
                color="white", ha="center", va="bottom", zorder=7)
    ax.set_xlim(0.4, len(rows) + 0.6)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(idx)
    ax.set_xticklabels(["{:d}".format(i) for i in idx], fontsize=6.2)
    ax.set_xlabel("reach number, white figure is the flow in L/s",
                  fontsize=6.8)
    ax.set_ylabel("filling ratio (%)", fontsize=6.6)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Depth of flow against the design limit", fontsize=8.0,
                 fontweight="bold", color=INK)


def velocity_panel(ax, rows):
    slope = np.array([1000.0 * r["slope"] for r in rows])
    vel = np.array([r["vel"] for r in rows])
    dia = np.array([r["dia"] for r in rows])
    sizes = 22.0 + 240.0 * (dia - dia.min() + 0.05)
    cols = [ORANGE if v < V_MIN else GREEN for v in vel]
    ax.axhspan(0.0, V_MIN, facecolor="#f6ded0", edgecolor="none", zorder=1)
    ax.scatter(slope, vel, s=sizes, c=cols, edgecolor="white", linewidth=0.7,
               zorder=5)
    for s, v, d, i in zip(slope, vel, dia, range(1, len(rows) + 1)):
        off = 0.020 + 0.055 * (d - dia.min() + 0.10) * 6.0
        ax.text(s + off, v, "{:d}".format(i), fontsize=6, color=SUB,
                ha="left", va="center", zorder=6)
    ax.axhline(V_MIN, color=RED, lw=1.0, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(slope.max() * 1.02, V_MIN - 0.05,
            "self-cleansing {:.2f} m/s".format(V_MIN), fontsize=6.1,
            color=RED, ha="right", va="top", zorder=7)
    ax.set_xlim(0.0, slope.max() * 1.12)
    ax.set_ylim(0.0, max(V_MAX * 0.42, vel.max() * 1.30))
    ax.set_xlabel("gradient (m per km), marker area is diameter",
                  fontsize=6.8)
    ax.set_ylabel("mean velocity (m/s)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Velocity check against gradient", fontsize=8.0,
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
    rows = solve_line()

    for r in rows:
        assert not r["surcharged"], "a reach has gone surcharged"
        assert abs(manning_q(r["theta"], r["dia"], r["slope"]) - r["q"]) \
            < 1e-12, "normal depth must reproduce the reach discharge"
        assert abs(r["vel"] * r["area"] - r["q"]) < 1e-12, \
            "continuity between velocity, area and discharge"
    q_in = sum(r[4] for r in REACHES) / 1000.0
    assert abs(rows[-1]["q"] - q_in) < 1e-12, "flow must accumulate exactly"

    span_x = rows[-1]["x1"] + 60.0
    vex = (span_x / MAIN[2]) / ((Y_HI - Y_LO) / MAIN[3])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_fill = axes_inches(fig, AUX_T)
    ax_vel = axes_inches(fig, AUX_B)

    main_panel(ax_main, rows, rng, vex)
    fill_panel(ax_fill, rows)
    velocity_panel(ax_vel, rows)

    worst = max(rows, key=lambda r: r["fill"])
    slow = min(rows, key=lambda r: r["vel"])


    panel_labels_inches(fig, [ax_main, ax_fill, ax_vel], dx=-0.62, dy=0.14)
    savefig(fig, "1007_wsn_sewer_profile.png")


if __name__ == "__main__":
    main()
