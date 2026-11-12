"""Maintained-load test on a 25 m bored pile, read as a load-transfer problem.

Main panel: the head load-settlement curve with its maintained load steps, the
elastic unloading branch and the residual settlement left at zero load, cut by
the Davisson offset line that defines the failure load.  Auxiliary (both at
58 % of the main panel width): axial force down the shaft at four load levels
over the soil column, and the unit shaft friction those forces imply against
the ultimate values assumed for each stratum.

One load-transfer march produces all three panels.  For a trial toe movement
the toe resistance follows a hyperbolic q-w law, and the axial force and pile
shortening are integrated upward with dN/dz = -pi D f and dw/dz = -N / EA,
the local friction being hyperbolic in the local movement.  Head load and head
settlement are the top values of that march, so the Q-s curve, the axial force
profiles and the friction bars are the same solution read three ways; the
axial force therefore decreases from head to toe by construction.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"

DIA = 0.80                      # m
LEN = 25.0                      # m
E_PILE = 30.0e9                 # Pa
AREA = np.pi * DIA ** 2 / 4.0   # m^2
W_REF = 0.003                   # m, shaft mobilisation
W_TOE = 0.030                   # m, toe mobilisation
Q_TOE_ULT = 4200.0              # kPa
NZ = 501

# top, base, ultimate unit shaft friction (kPa), name, colour
LAYERS = [(0.0, 6.0, 28.0, "soft clay", lighten("#8c6d4f", 0.62)),
          (6.0, 15.0, 62.0, "stiff clay", lighten("#7a6a5d", 0.50)),
          (15.0, LEN, 105.0, "dense sand", lighten("#c9a227", 0.58))]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def f_ultimate(z):
    """Ultimate unit shaft friction at depth ``z``, in kPa."""
    out = np.zeros_like(z)
    for top, base, fu, _n, _c in LAYERS:
        out[(z >= top) & (z <= base)] = fu
    return out


def march(w_toe):
    """Integrate one load-transfer solution from a trial toe movement.

    Returns depth (m), axial force (kN), local movement (m) and unit shaft
    friction (kPa), all as arrays running from the head down to the toe.
    """
    z = np.linspace(0.0, LEN, NZ)
    dz = z[1] - z[0]
    fu = f_ultimate(z)
    n = np.zeros(NZ)
    w = np.zeros(NZ)
    fs = np.zeros(NZ)

    q_b = Q_TOE_ULT * w_toe / (w_toe + W_TOE)
    n[-1] = q_b * AREA
    w[-1] = w_toe
    for i in range(NZ - 1, 0, -1):
        fs[i] = fu[i] * w[i] / (w[i] + W_REF)
        n[i - 1] = n[i] + np.pi * DIA * fs[i] * dz
        w[i - 1] = w[i] + n[i] * 1000.0 * dz / (E_PILE * AREA)
    fs[0] = fu[0] * w[0] / (w[0] + W_REF)
    return z, n, w, fs


def build_curve(n_step=170):
    """Head load (kN) and head settlement (mm) over a sweep of toe movements."""
    toes = np.logspace(np.log10(2.0e-5), np.log10(0.075), n_step)
    heads = np.array([march(t)[1][0] for t in toes])
    setts = np.array([1000.0 * march(t)[2][0] for t in toes])
    return heads, setts, toes


def davisson(load):
    """Davisson offset limit line, in mm, for the plotted load array."""
    elastic = load * 1000.0 * LEN / (E_PILE * AREA) * 1000.0
    return elastic + 4.0 + DIA * 1000.0 / 120.0


def main_panel(ax, q_load, s_load, steps, q_un, s_un, q_dav, s_dav, s_res):
    ax.plot(q_load, s_load, color=TEAL, lw=1.9, zorder=5, label="loading")
    ax.plot(q_un, s_un, color=BLUE, lw=1.4, ls=(0, (4.0, 2.2)), zorder=6,
            label="unloading (elastic rebound)")
    ax.plot(*zip(*steps), lw=0.0, marker="o", ms=4.0, mfc="white", mec=TEAL,
            mew=1.0, zorder=8, label="maintained load step")

    qq = np.linspace(0.0, q_load.max() * 1.02, 60)
    ax.plot(qq, davisson(qq), color=MUTED, lw=1.1, ls=(0, (2.5, 2.0)),
            zorder=4, label="Davisson offset line")
    ax.plot([q_dav], [s_dav], marker="D", ms=5.0, mfc=RED, mec="white",
            mew=0.7, zorder=9)
    ax.annotate("failure load {:.0f} kN\nat {:.1f} mm".format(q_dav, s_dav),
                xy=(q_dav, s_dav), xytext=(q_dav - 420.0, s_dav + 11.0),
                fontsize=6.8, color=RED, fontweight="bold", ha="right",
                va="top", zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=2,
                                shrinkB=4))
    ax.annotate("residual {:.1f} mm".format(s_res), xy=(0.0, s_res),
                xytext=(560.0, s_res - 5.0), fontsize=6.6, color=BLUE,
                ha="left", va="bottom", zorder=10,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7,
                                shrinkA=2, shrinkB=3))

    ax.set_xlim(0.0, q_load.max() * 1.03)
    ax.set_ylim(s_load.max() * 1.06, 0.0)
    ax.set_xlabel("head load (kN)")
    ax.set_ylabel("head settlement (mm), positive downward")
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Head load-settlement curve with the unload rebound",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    ax.legend(loc="lower left", fontsize=6.4, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.94, handlelength=1.7,
              borderaxespad=0.4, labelspacing=0.3)


def shade_layers(ax, xlo, xhi, label=False):
    for top, base, _fu, name, col in LAYERS:
        ax.axhspan(top, base, facecolor=col, alpha=0.55, lw=0.0, zorder=1)
        ax.axhline(base, color="#9a9a9a", lw=0.6, zorder=2)
        if label:
            ax.text(xhi - 0.02 * (xhi - xlo), 0.5 * (top + base), name,
                    fontsize=6, color=SUB, ha="right", va="center",
                    zorder=6)


def axial_panel(ax, z, profiles, colors):
    shade_layers(ax, 0.0, 6600.0, label=True)
    # Head-load labels sit at four different depths: at the pile head the
    # four curves are only a few hundred kN apart and the labels would stack.
    z_lab = np.linspace(2.6, 11.6, len(profiles))
    for (q, n), c, zl in zip(profiles, colors, z_lab):
        ax.plot(n, z, color=c, lw=1.5, zorder=5)
        ax.text(float(np.interp(zl, z, n)) + 130.0, zl, "{:.0f} kN".format(q),
                color=c, fontsize=6, ha="left", va="center", zorder=7)
    q_top, n_top = profiles[-1]
    ax.annotate("toe takes {:.0f} kN\n({:.0f} % of the head load)"
                .format(n_top[-1], 100.0 * n_top[-1] / q_top),
                xy=(n_top[-1], LEN), xytext=(2450.0, 21.6), fontsize=6.0,
                color=INK, ha="left", va="center", zorder=8, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=2,
                                shrinkB=3))
    ax.set_xlim(0.0, 6600.0)
    ax.set_ylim(LEN, 0.0)
    ax.set_xlabel("axial force (kN)", fontsize=6.8)
    ax.set_ylabel("depth (m), positive downward", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Axial force decays from head to toe", fontsize=8.0,
                 fontweight="bold", color=INK)


def friction_panel(ax, z, fs, fu):
    shade_layers(ax, 0.0, 125.0)
    ax.fill_betweenx(z, 0.0, fs, facecolor=ORANGE, alpha=0.40, lw=0.0,
                     zorder=3)
    ax.plot(fs, z, color=ORANGE, lw=1.5, zorder=5, label="mobilised at "
                                                        "maximum load")
    ax.plot(fu, z, color=RED, lw=1.1, ls=(0, (3.5, 2.0)), zorder=6,
            label="ultimate value assumed")
    ratio = np.trapezoid(fs, z) / np.trapezoid(fu, z)
    ax.text(52.0, 1.0, "{:.0f} % of the available shaft\nfriction is "
            "mobilised".format(100.0 * ratio), fontsize=6.0, color=INK,
            ha="left", va="top", zorder=8, linespacing=1.35)
    ax.set_xlim(0.0, 125.0)
    ax.set_ylim(LEN, 0.0)
    ax.set_xlabel("unit shaft friction (kPa)", fontsize=6.8)
    ax.set_ylabel("depth (m), positive downward", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.3, borderaxespad=0.3, labelspacing=0.24)
    ax.set_title("Friction read from the force gradient", fontsize=8.0,
                 fontweight="bold", color=INK)
    return ratio


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
    np.random.default_rng(5502)

    heads, setts, toes = build_curve()
    # The test is taken to a round load close to a 45 mm head settlement,
    # which is where a maintained-load test is normally stopped.
    q_max = 200.0 * np.floor(float(np.interp(45.0, setts, heads)) / 200.0)
    keep = heads <= q_max
    q_load, s_load = heads[keep], setts[keep]

    # Maintained load steps, read off the computed curve.
    q_steps = np.arange(1, 9) * q_max / 8.0
    steps = [(q, float(np.interp(q, q_load, s_load))) for q in q_steps]

    # State at the top of the test, and the elastic shortening it stores.
    toe_max = float(np.interp(q_max, heads, toes))
    z, n_max, w_max, fs_max = march(toe_max)
    s_max = float(np.interp(q_max, q_load, s_load))
    s_elastic = 1000.0 * np.trapezoid(n_max, z) * 1000.0 / (E_PILE * AREA)
    s_res = s_max - s_elastic
    q_un = np.linspace(q_max, 0.0, 40)
    s_un = s_max - s_elastic * (1.0 - q_un / q_max) ** 1.15

    diff = s_load - davisson(q_load)
    k = int(np.argmax(diff > 0.0))
    q_dav = float(np.interp(0.0, [diff[k - 1], diff[k]],
                            [q_load[k - 1], q_load[k]]))
    s_dav = float(np.interp(q_dav, q_load, s_load))

    profiles = [(q, march(float(np.interp(q, heads, toes)))[1])
                for q in (1500.0, 3000.0, 4500.0, q_max)]
    assert all(np.all(np.diff(p) <= 1e-9) for _q, p in profiles), \
        "axial force must decrease with depth"
    assert s_res > 0.0, "unloading cannot recover more than it stored"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_ax = axes_inches(fig, AUX_T)
    ax_fs = axes_inches(fig, AUX_B)

    colors = [lighten(BLUE, 0.55), lighten(BLUE, 0.30), BLUE, RED]
    main_panel(ax_main, q_load, s_load, steps, q_un, s_un, q_dav, s_dav,
               s_res)
    axial_panel(ax_ax, z, profiles, colors)
    ratio = friction_panel(ax_fs, z, fs_max, f_ultimate(z))



    panel_labels_inches(fig, [ax_main, ax_ax, ax_fs], dx=-0.62, dy=0.14)
    savefig(fig, "986_shm_pile_load_test.png")


if __name__ == "__main__":
    main()
