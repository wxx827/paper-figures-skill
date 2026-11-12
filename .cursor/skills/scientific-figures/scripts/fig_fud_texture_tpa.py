"""Texture profile analysis: every TPA parameter read off the curve that
defines it.

The two-bite trace is built once, as a piecewise force-time law under a
triangular probe stroke, and then *measured* - nothing on the page is asserted.
Hardness is the height of the first peak, A1 and A2 are trapezoidal integrals
over exactly the boolean masks that the two shaded areas are filled with, so
the printed cohesiveness A2/A1 cannot disagree with the geometry the reader
sees.  Springiness is the ratio of the two contact-to-peak intervals, which
differ only because the sample does not fully recover its height between
bites, so the probe meets it later on the second descent; adhesiveness is the
negative lobe left behind when the probe pulls away, and resilience is the
unloading area of the first bite over its loading area.  Gumminess and
chewiness are then products of those measurements.  The right-hand column runs
the same measurement function over four synthetic products and over thirty
jittered replicates of them, which is why the radar and the scatter share one
definition of every axis.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
RED = "#b23a48"
OCHRE = "#c0603f"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

C1_ON, C1_PK, C1_ZERO, C1_NEG = 1.0, 3.0, 4.60, 5.60
C2_ON, C2_PK, C2_ZERO, C2_NEG = 6.60, 8.60, 10.20, 11.00
SPLIT = 6.10
T_END = 13.0
STROKE = C1_PK - C1_ON              # s, probe descent of one bite
DEPTH_MM = 12.5                     # mm, 50 % strain on a 25 mm specimen

#          name              hardness ratio2 recovery adhesion  colour
PRODUCTS = [("process cheese", 24.0, 0.55, 0.775, 2.60, BLUE),
            ("set yoghurt gel", 8.5, 0.78, 0.930, 0.95, GREEN),
            ("bread crumb", 4.2, 0.62, 0.860, 0.35, GOLD),
            ("gummy confection", 31.0, 0.88, 0.955, 0.45, PURPLE)]
MAIN_ID = 0
PARAMS = ["hardness", "springiness", "cohesiveness", "gumminess", "chewiness"]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.82, 3.61, 2.68, 1.89)
AUX_B = (7.82, 1.02, 2.68, 1.82)


def tpa_curve(hardness, ratio2, recovery, adhesion, rng, dt=0.002):
    """Piecewise two-bite force trace under a fixed triangular probe stroke.

    ``recovery`` is the fraction of specimen height regained between bites;
    the probe therefore touches the sample ``(1 - recovery) * STROKE`` seconds
    later on the second descent, which is the only reason springiness comes
    out below one.
    """
    t = np.arange(0.0, T_END, dt)
    f = np.zeros_like(t)
    h2 = hardness * ratio2
    on2 = C2_PK - recovery * STROKE

    m = (t >= C1_ON) & (t <= C1_PK)
    f[m] = hardness * ((t[m] - C1_ON) / STROKE) ** 1.7
    m = (t > C1_PK) & (t <= C1_ZERO)
    f[m] = hardness * ((C1_ZERO - t[m]) / (C1_ZERO - C1_PK)) ** 2.2
    m = (t > C1_ZERO) & (t <= C1_NEG)
    f[m] = -adhesion * np.sin(np.pi * (t[m] - C1_ZERO) /
                              (C1_NEG - C1_ZERO)) ** 1.4
    m = (t >= on2) & (t <= C2_PK)
    f[m] = h2 * ((t[m] - on2) / (C2_PK - on2)) ** 1.7
    m = (t > C2_PK) & (t <= C2_ZERO)
    f[m] = h2 * ((C2_ZERO - t[m]) / (C2_ZERO - C2_PK)) ** 2.2
    m = (t > C2_ZERO) & (t <= C2_NEG)
    f[m] = -0.55 * adhesion * np.sin(np.pi * (t[m] - C2_ZERO) /
                                     (C2_NEG - C2_ZERO)) ** 1.4
    return t, f + rng.normal(0.0, 0.005 * hardness, t.size)


def probe_mm(t):
    """Probe penetration depth in mm; two identical triangular strokes."""
    d = np.zeros_like(t)
    for on, pk in ((C1_ON, C1_PK), (C2_ON, C2_PK)):
        up = (t >= on) & (t <= pk)
        d[up] = DEPTH_MM * (t[up] - on) / STROKE
        dn = (t > pk) & (t <= pk + STROKE)
        d[dn] = DEPTH_MM * (pk + STROKE - t[dn]) / STROKE
    return d


def tpa_metrics(t, f):
    """Measure the standard TPA parameters off a two-bite trace."""
    thr = 0.02 * float(f.max())
    m1 = (t < SPLIT) & (f > thr)
    m2 = (t >= SPLIT) & (f > thr)
    mneg = (t < SPLIT) & (f < -thr)
    i1 = int(np.argmax(np.where(t < SPLIT, f, -np.inf)))
    i2 = int(np.argmax(np.where(t >= SPLIT, f, -np.inf)))
    on1 = float(t[m1][0])
    on2 = float(t[m2][0])
    a1 = float(np.trapezoid(f[m1], t[m1]))
    a2 = float(np.trapezoid(f[m2], t[m2]))
    adh = float(np.trapezoid(f[mneg], t[mneg]))
    load = m1 & (t <= t[i1])
    unload = m1 & (t > t[i1])
    hardness = float(f[i1])
    cohesive = a2 / a1
    springy = (float(t[i2]) - on2) / (float(t[i1]) - on1)
    gummy = hardness * cohesive
    return dict(hardness=hardness, hardness2=float(f[i2]), a1=a1, a2=a2,
                cohesiveness=cohesive, springiness=springy,
                adhesiveness=adh, gumminess=gummy, chewiness=gummy * springy,
                resilience=(float(np.trapezoid(f[unload], t[unload])) /
                            float(np.trapezoid(f[load], t[load]))),
                t_pk1=float(t[i1]), t_pk2=float(t[i2]), on1=on1, on2=on2,
                m1=m1, m2=m2, mneg=mneg)


def main_panel(ax, t, f, met, colour):
    ax_d = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_d.set_zorder(1)
    ax_d.plot(t, probe_mm(t), color=MUTED, lw=1.0, ls=(0, (4.5, 2.2)),
              zorder=2)
    ax_d.fill_between(t, 0.0, probe_mm(t), color="#f2f2f2", lw=0.0, zorder=1)

    h = met["hardness"]
    ax.axhline(0.0, color=SUB, lw=0.8, zorder=3)
    ax.fill_between(t, 0.0, f, where=met["m1"], color=lighten(BLUE, 0.74),
                    lw=0.0, zorder=4)
    ax.fill_between(t, 0.0, f, where=met["m2"], color=lighten(GREEN, 0.68),
                    lw=0.0, zorder=4)
    ax.fill_between(t, 0.0, f, where=met["mneg"], color=lighten(RED, 0.60),
                    lw=0.0, zorder=4)
    ax.plot(t, f, color=colour, lw=1.5, zorder=6)

    ax.annotate("", xy=(met["t_pk1"], 0.0), xytext=(met["t_pk1"], h),
                zorder=8, arrowprops=dict(arrowstyle="<|-|>", color=INK,
                                          lw=0.9, mutation_scale=8))
    ax.text(met["t_pk1"] - 0.14, h * 0.55,
            "hardness\n$F_1$ = {:.1f} N".format(h), ha="right", va="center",
            fontsize=6.8, color=INK, linespacing=1.30, zorder=9)
    ax.plot([met["t_pk2"]], [met["hardness2"]], marker="v", ms=4.4,
            color=colour, zorder=8)
    ax.text(met["t_pk2"] + 0.14, met["hardness2"],
            "$F_2$ = {:.1f} N".format(met["hardness2"]), ha="left",
            va="center", fontsize=6.6, color=SUB, zorder=9)

    ax.text(met["t_pk1"] + 0.28, h * 0.20,
            "$A_1$ = {:.1f} N$\\cdot$s".format(met["a1"]), ha="left",
            va="center", fontsize=6.8, color=BLUE, zorder=9)
    ax.text(met["t_pk2"] + 0.30, met["hardness2"] * 0.22,
            "$A_2$ = {:.1f} N$\\cdot$s".format(met["a2"]), ha="left",
            va="center", fontsize=6.8, color=GREEN, zorder=9)
    ax.annotate("adhesiveness $A_3$ = {:.2f} N$\\cdot$s".format(
        met["adhesiveness"]),
        xy=(0.5 * (C1_ZERO + C1_NEG), -0.055 * h),
        xytext=(C1_NEG + 0.35, -0.26 * h), ha="left", va="center",
        fontsize=6.6, color=RED, zorder=9,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    y_s = -0.145 * h
    for x0, x1, lab in ((met["on1"], met["t_pk1"], "$L_1$"),
                        (met["on2"], met["t_pk2"], "$L_2$")):
        ax.annotate("", xy=(x0, y_s), xytext=(x1, y_s), zorder=8,
                    arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.8,
                                    mutation_scale=7))
        ax.text(0.5 * (x0 + x1), y_s + 0.022 * h,
                "{:s} = {:.2f} s".format(lab, x1 - x0), ha="center",
                va="bottom", fontsize=6.3, color=SUB, zorder=9)

    ax.text(0.12, 1.385 * h,
            "cohesiveness  $A_2/A_1$ = {:.3f}\n"
            "springiness  $L_2/L_1$ = {:.3f}\n"
            "resilience  = {:.3f}\n"
            "gumminess  $F_1\\times A_2/A_1$ = {:.1f} N\n"
            "chewiness  = {:.1f} N".format(
                met["cohesiveness"], met["springiness"], met["resilience"],
                met["gumminess"], met["chewiness"]),
            ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.42,
            zorder=9)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(-0.40 * h, 1.42 * h)
    ax_d.set_ylim(0.0, 3.1 * DEPTH_MM)
    ax_d.set_yticks([0.0, 4.0, 8.0, 12.0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("compression force (N)", color=colour)
    ax_d.set_ylabel("probe penetration (mm)", color=MUTED)
    ax.tick_params(axis="y", colors=colour)
    ax_d.tick_params(axis="y", colors=MUTED, labelsize=6.8)
    ax.spines["left"].set_color(colour)
    ax_d.spines["right"].set_color(HAIR)
    for a, sides in ((ax, ("top", "right")), (ax_d, ("top", "left"))):
        for s in sides:
            a.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("Two-bite compression of process cheese, 50 % strain",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def radar_panel(ax, table):
    n = len(PARAMS)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.concatenate([ang, ang[:1]])
    mat = np.array([[table[i][p] for p in PARAMS]
                    for i in range(len(PRODUCTS))])
    mat = mat / mat.max(axis=0)
    for i, (name, _h, _r, _c, _a, col) in enumerate(PRODUCTS):
        v = np.concatenate([mat[i], mat[i][:1]])
        ax.plot(ring, v, color=col, lw=1.3, zorder=5,
                ls="-" if i == MAIN_ID else (0, (4.0, 1.8)))
        if i == MAIN_ID:
            ax.fill(ring, v, color=col, alpha=0.15, lw=0.0, zorder=3)
    ax.set_xticks(ang)
    ax.set_xticklabels(PARAMS, fontsize=6.2, color=SUB)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6,
                       color=MUTED)
    ax.set_ylim(0.0, 1.10)
    ax.tick_params(axis="x", pad=0.5)
    ax.set_rlabel_position(104.0)
    ax.grid(lw=0.5, color=HAIR, alpha=0.9)
    ax.spines["polar"].set_color(HAIR)
    ax.spines["polar"].set_linewidth(0.7)
    ax.set_title("Scaled to the largest of the four (-)", fontsize=8.0,
                 fontweight="bold", color=INK, pad=9.0)
    handles = [Line2D([], [], color=c, lw=1.3, label=n)
               for n, _h, _r, _co, _a, c in PRODUCTS]
    ax.legend(handles=handles, loc="lower center", fontsize=6,
              frameon=False, ncol=2, handlelength=1.4, columnspacing=0.9,
              bbox_to_anchor=(0.5, -0.28), labelspacing=0.20)


def scatter_panel(ax, table, cloud):
    for i, (name, _h, _r, _c, _a, col) in enumerate(PRODUCTS):
        hx = [m["hardness"] for m in cloud[i]]
        cy = [m["chewiness"] for m in cloud[i]]
        ax.plot(hx, cy, marker="o", ms=3.0, mfc=lighten(col, 0.45),
                mec="white", mew=0.4, ls="none", zorder=4)
        ax.plot([table[i]["hardness"]], [table[i]["chewiness"]], marker="D",
                ms=5.0, mfc=col, mec="white", mew=0.8, ls="none", zorder=6)
        right = table[i]["hardness"] > 15.0    # keep labels off the frame
        ax.annotate(name, xy=(table[i]["hardness"], table[i]["chewiness"]),
                    xytext=(-4.5 if right else 4.5, 4.0),
                    textcoords="offset points",
                    ha="right" if right else "left", va="bottom",
                    fontsize=6, color=col, zorder=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.4, 90.0)
    ax.set_ylim(0.9, 40.0)
    ax.set_xlabel("hardness $F_1$ (N)", fontsize=7.4)
    ax.set_ylabel("chewiness (N)", fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Hard is not the same as chewy", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
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
    rng = np.random.default_rng(913)

    table, cloud = [], []
    for name, h, r2, rec, adh, _col in PRODUCTS:
        t, f = tpa_curve(h, r2, rec, adh, rng)
        table.append(tpa_metrics(t, f))
        reps = []
        for _ in range(8):
            tt, ff = tpa_curve(h * rng.normal(1.0, 0.20),
                               float(np.clip(r2 * rng.normal(1.0, 0.07),
                                             0.3, 0.97)),
                               float(np.clip(rec * rng.normal(1.0, 0.05),
                                             0.5, 0.99)),
                               adh * rng.normal(1.0, 0.25), rng, dt=0.005)
            reps.append(tpa_metrics(tt, ff))
        cloud.append(reps)

    t, f = tpa_curve(*PRODUCTS[MAIN_ID][1:5], rng=rng)
    met = tpa_metrics(t, f)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_rad = axes_inches(fig, AUX_T, projection="polar")
    ax_sc = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, f, met, PRODUCTS[MAIN_ID][5])
    radar_panel(ax_rad, table)
    scatter_panel(ax_sc, table, cloud)

    panel_labels_inches(fig, [ax_main, ax_rad, ax_sc], dx=-0.62, dy=0.14)
    savefig(fig, "913_fud_texture_tpa.png")


if __name__ == "__main__":
    main()
