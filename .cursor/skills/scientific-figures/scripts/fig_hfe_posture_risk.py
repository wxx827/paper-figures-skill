"""Whole-body posture scoring, in the style of a REBA worksheet.

Main panel: the worst posture of a simulated packing shift, drawn as a stick
figure from the very joint angles that are being scored, with the six body
segments laid out as banded angle scales.  Each band is cut into the score
intervals, the observed angle is marked on it, and the segment scores feed the
two grids (group A: trunk, neck, legs; group B: upper arm, lower arm, wrist).
The group scores, the load and coupling adders and the final score are
propagated through the grids that are drawn, so the number in the result box
is the cell the arrows point at.

Auxiliary (both at 58 % of the main panel width): the same scoring pipeline
applied to every posture sample of an eight-hour shift, with the spells above
the action threshold shaded; and the mean score each body segment contributed
over that shift.

Illustrative scoring grids inspired by REBA, not the published tables;
fictional task, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches, arrow

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#cccccc"
GRID = "#d5d5d5"
SOFT = "#f5f5f5"
BLUE = "#3b5b92"
TEAL = "#2e7d5b"
AMBER = "#c9a227"
RUST = "#c0603f"
DEEP = "#8c2f2f"
ACCENT = "#b23a48"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.62, 0.72, 5.86, 4.72)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

SCORE_COLORS = {1: "#6bbf8a", 2: "#c9a227", 3: "#c0603f", 4: "#8c2f2f"}
BAND_COLORS = [(1, 1, "#bfe3cd", "negligible"), (2, 3, "#6bbf8a", "low"),
               (4, 7, "#c9a227", "medium"), (8, 10, "#c0603f", "high"),
               (11, 15, "#8c2f2f", "very high")]

# segment name, angle axis range, score intervals, axis unit label
SEGMENTS = [
    ("trunk flexion", (-20.0, 80.0),
     [(-20, 5, 1), (5, 20, 2), (20, 60, 3), (60, 80, 4)]),
    ("neck flexion", (-15.0, 45.0),
     [(-15, 0, 2), (0, 20, 1), (20, 45, 2)]),
    ("knee flexion", (0.0, 90.0),
     [(0, 30, 1), (30, 60, 2), (60, 90, 3)]),
    ("upper arm elevation", (-25.0, 130.0),
     [(-25, 20, 1), (20, 45, 2), (45, 90, 3), (90, 130, 4)]),
    ("lower arm angle", (20.0, 150.0),
     [(20, 60, 2), (60, 100, 1), (100, 150, 2)]),
    ("wrist deviation", (-30.0, 30.0),
     [(-30, -15, 2), (-15, 15, 1), (15, 30, 2)]),
]
LOAD_ADDER = 1          # > 5 kg handled
COUPLING_ADDER = 1      # poor handle
ACTIVITY_ADDER = 1      # repeated small-range action
ACTION_LIMIT = 8        # score at or above which the task needs redesign


def seg_score(idx, angle):
    """Score an angle off the interval table that the band panel draws."""
    for lo, hi, s in SEGMENTS[idx][2]:
        if lo <= angle < hi:
            return s
    return SEGMENTS[idx][2][-1][2] if angle >= SEGMENTS[idx][1][1] else \
        SEGMENTS[idx][2][0][2]


def group_a(trunk, neck, legs):
    return int(np.clip(round(1.05 * trunk + 0.85 * neck + 0.75 * legs - 1.2),
                       1, 12))


def group_b(upper, lower, wrist):
    return int(np.clip(round(1.05 * upper + 0.80 * lower + 0.75 * wrist - 1.1),
                       1, 9))


def table_c(a, b):
    return int(np.clip(round(0.62 * (a + b) + 0.30), 1, 15))


def score_posture(angles):
    """Angles -> segment scores -> group scores -> final REBA-style score."""
    s = [seg_score(i, a) for i, a in enumerate(angles)]
    a_raw = group_a(s[0], s[1], s[2])
    b_raw = group_b(s[3], s[4], s[5])
    a_tot = a_raw + LOAD_ADDER
    b_tot = b_raw + COUPLING_ADDER
    c = table_c(a_tot, b_tot)
    return dict(seg=s, a_raw=a_raw, b_raw=b_raw, a=a_tot, b=b_tot, c=c,
                total=c + ACTIVITY_ADDER)


def risk_band(total):
    for lo, hi, col, lab in BAND_COLORS:
        if lo <= total <= hi:
            return col, lab
    return DEEP, "very high"


def shift_samples(rng, n=32):
    """One posture sample every 15 min of an eight-hour packing shift."""
    t = np.linspace(0.0, 8.0, n)
    drift = 18.0 * (1.0 - np.exp(-t / 3.0))
    base = np.array([32.0, 11.0, 16.0, 33.0, 86.0, 7.0])
    slope = np.array([1.30, 0.95, 0.85, 1.45, -1.55, 0.80])
    ang = (base[None, :] + drift[:, None] * slope[None, :]
           + rng.normal(0.0, np.array([11.0, 7.5, 9.0, 13.0, 11.0, 7.5]),
                        (n, 6)))
    for k, (_, (lo, hi), _) in enumerate(SEGMENTS):
        ang[:, k] = np.clip(ang[:, k], lo + 0.5, hi - 0.5)
    ang[:, 2] = np.maximum(ang[:, 2], 0.0)
    return t, ang


def stick_figure(ax, x0, y0, ang):
    """Side-view linkage drawn from the scored angles; inch coordinates."""
    t, neck, knee, ua, la, _ = ang
    d = np.deg2rad
    hip = np.array([x0, y0])
    trunk_dir = np.array([np.sin(d(t)), np.cos(d(t))])
    shoulder = hip + 0.80 * trunk_dir
    head_dir = np.array([np.sin(d(t + neck)), np.cos(d(t + neck))])
    head_c = shoulder + 0.27 * head_dir
    base = np.degrees(np.arctan2(-np.cos(d(t)), -np.sin(d(t))))
    arm_a = base + ua
    elbow = shoulder + 0.40 * np.array([np.cos(d(arm_a)), np.sin(d(arm_a))])
    fore_a = arm_a + (180.0 - la)
    hand = elbow + 0.34 * np.array([np.cos(d(fore_a)), np.sin(d(fore_a))])
    thigh_a = -84.0
    knee_p = hip + 0.52 * np.array([np.cos(d(thigh_a)), np.sin(d(thigh_a))])
    shank_a = thigh_a - knee
    ankle = knee_p + 0.52 * np.array([np.cos(d(shank_a)),
                                      np.sin(d(shank_a))])
    toe = ankle + np.array([0.22, -0.05])

    ax.plot([ankle[0] - 0.42, ankle[0] + 0.58], [ankle[1] - 0.05] * 2,
            color="#b8b8b8", lw=1.0, zorder=1)
    for p0, p1, col, lw in [(hip, shoulder, INK, 3.2), (hip, knee_p, INK, 3.0),
                            (knee_p, ankle, INK, 3.0), (ankle, toe, INK, 2.2),
                            (shoulder, elbow, ACCENT, 2.6),
                            (elbow, hand, ACCENT, 2.6)]:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=lw,
                solid_capstyle="round", zorder=6)
    ax.plot([shoulder[0], head_c[0]], [shoulder[1], head_c[1]], color=INK,
            lw=2.4, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle(tuple(head_c), 0.13, facecolor="white",
                        edgecolor=INK, lw=1.3, zorder=7))
    for p, col in ((hip, INK), (shoulder, INK), (knee_p, INK),
                   (elbow, ACCENT)):
        ax.add_patch(Circle(tuple(p), 0.036, facecolor="white",
                            edgecolor=col, lw=1.0, zorder=8))

    # angle arcs against the local reference direction
    ax.plot([hip[0], hip[0]], [hip[1], hip[1] + 0.70], color=FAINT, lw=0.7,
            ls=(0, (2.6, 2.0)), zorder=4)
    ax.add_patch(Arc(tuple(hip), 0.92, 0.92, angle=0.0, theta1=90.0 - t,
                     theta2=90.0, color=SUB, lw=0.9, zorder=5))
    ax.text(hip[0] - 0.07, hip[1] + 0.58, f"trunk {t:.0f} deg", fontsize=6.2,
            color=SUB, ha="right", va="bottom")
    ext = shoulder - 0.30 * trunk_dir
    ax.plot([shoulder[0], ext[0]], [shoulder[1], ext[1]], color=FAINT,
            lw=0.7, ls=(0, (2.6, 2.0)), zorder=4)
    ax.add_patch(Arc(tuple(shoulder), 0.50, 0.50, angle=0.0, theta1=base,
                     theta2=arm_a, color=SUB, lw=0.9, zorder=5))
    mid = d(0.5 * (base + arm_a))
    ax.text(shoulder[0] + 0.42 * np.cos(mid), shoulder[1] + 0.42
            * np.sin(mid), f"{ua:.0f} deg", fontsize=6.2, color=SUB,
            ha="center", va="center")
    ax.add_patch(Arc(tuple(elbow), 0.36, 0.36, angle=0.0, theta1=fore_a,
                     theta2=arm_a + 180.0, color=SUB, lw=0.9, zorder=5))
    ax.text(elbow[0] + 0.22, elbow[1] - 0.16, f"{la:.0f} deg", fontsize=6.2,
            color=SUB, ha="left", va="top")
    ax.add_patch(Arc(tuple(knee_p), 0.44, 0.44, angle=0.0, theta1=shank_a,
                     theta2=thigh_a, color=SUB, lw=0.9, zorder=5))
    ax.text(knee_p[0] - 0.14, knee_p[1] - 0.26, f"{knee:.0f} deg",
            fontsize=6.2, color=SUB, ha="right", va="center")
    return hip, shoulder


def grid_table(ax, x0, y0, cell, rows, cols, values, hit, title,
               row_lab, col_lab):
    """Small score grid with the selected cell outlined."""
    vmax = float(np.max(values))
    for i in range(rows):
        for j in range(cols):
            v = values[i, j]
            col = lighten(RUST if v >= 0.72 * vmax else
                          (AMBER if v >= 0.45 * vmax else TEAL),
                          0.72 - 0.42 * v / vmax)
            ax.add_patch(Rectangle((x0 + j * cell, y0 + (rows - 1 - i) * cell),
                                   cell, cell, facecolor=col,
                                   edgecolor="white", lw=0.6, zorder=3))
            ax.text(x0 + (j + 0.5) * cell,
                    y0 + (rows - 1 - i + 0.5) * cell, f"{v:.0f}",
                    ha="center", va="center", fontsize=6, color=INK,
                    zorder=4)
    i, j = hit
    ax.add_patch(Rectangle((x0 + j * cell, y0 + (rows - 1 - i) * cell), cell,
                           cell, facecolor="none", edgecolor=INK, lw=1.5,
                           zorder=6))
    ax.text(x0 + cols * cell / 2.0, y0 + rows * cell + 0.20, title,
            ha="center", va="bottom", fontsize=6.8, fontweight="bold",
            color=INK)
    ax.text(x0 + cols * cell / 2.0, y0 + rows * cell + 0.06, col_lab,
            ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.text(x0 - 0.05, y0 + rows * cell / 2.0, row_lab, ha="center",
            va="center", fontsize=6, color=MUTED, rotation=90)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(952)

    tt, ang = shift_samples(rng)
    res = [score_posture(a) for a in ang]
    totals = np.array([r["total"] for r in res])
    worst = int(np.argmax(totals))
    peak = res[worst]
    pang = ang[worst]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = inch_axes(fig, MAIN_RECT, aspect="equal")
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    W, H = MAIN_RECT[2], MAIN_RECT[3]
    ax.add_patch(Rectangle((0.02, 0.02), W - 0.04, H - 0.04, facecolor="none",
                           edgecolor=HAIR, lw=0.8, zorder=0))

    # ---- main, left: the posture itself ------------------------------------
    stick_figure(ax, 0.50, 1.66, pang)
    ax.text(0.16, H - 0.22, "worst sampled posture", ha="left", va="top",
            fontsize=7.6, fontweight="bold", color=INK)
    ax.text(0.16, H - 0.42,
            f"shift hour {tt[worst]:.2f} h of 8.00 h,\nload handled > 5 kg",
            ha="left", va="top", fontsize=6.4, color=MUTED, linespacing=1.45)
    ax.text(0.16, H - 0.92, "segment score key", ha="left", va="top",
            fontsize=6.4, color=SUB)
    for s in range(1, 5):
        ax.add_patch(Rectangle((0.16 + (s - 1) * 0.30, H - 1.22), 0.24, 0.14,
                               facecolor=lighten(SCORE_COLORS[s], 0.28),
                               edgecolor="white", lw=0.6, zorder=3))
        ax.text(0.28 + (s - 1) * 0.30, H - 1.15, f"{s}", ha="center",
                va="center", fontsize=6, color="white", fontweight="bold",
                zorder=4)
    ax.text(0.16, H - 1.28, "1 acceptable  ->  4 investigate now", ha="left",
            va="top", fontsize=6, color=MUTED)

    # ---- main, middle: banded angle scales ---------------------------------
    bx0, bw = 2.05, 1.45
    top = H - 0.62
    row_h = 0.60
    for k, (name, (lo, hi), ivs) in enumerate(SEGMENTS):
        y = top - k * row_h
        for a0, a1, s in ivs:
            x = bx0 + bw * (a0 - lo) / (hi - lo)
            w = bw * (a1 - a0) / (hi - lo)
            ax.add_patch(Rectangle((x, y), w, 0.155,
                                   facecolor=lighten(SCORE_COLORS[s], 0.28),
                                   edgecolor="white", lw=0.6, zorder=3))
            if w > 0.19:
                ax.text(x + w / 2.0, y + 0.077, f"{s}", ha="center",
                        va="center", fontsize=6, color="white",
                        fontweight="bold", zorder=4)
        val = pang[k]
        xm = bx0 + bw * (np.clip(val, lo, hi) - lo) / (hi - lo)
        ax.plot([xm], [y + 0.205], marker="v", ms=5.0, mfc=INK, mec="white",
                mew=0.6, zorder=6)
        ax.text(bx0, y + 0.245, name, ha="left", va="bottom", fontsize=6.4,
                color=INK)
        ax.text(bx0 + bw, y + 0.245, f"{val:.0f} deg -> {peak['seg'][k]}",
                ha="right", va="bottom", fontsize=6.2, color=SUB)
        ax.text(bx0 - 0.04, y + 0.077, f"{lo:.0f}", ha="right", va="center",
                fontsize=6, color=MUTED)
        ax.text(bx0 + bw + 0.04, y + 0.077, f"{hi:.0f}", ha="left",
                va="center", fontsize=6, color=MUTED)
    ax.text(bx0 + bw / 2.0, top + 0.32,
            "segment angle (deg), banded by score", ha="center", va="bottom",
            fontsize=6.6, color=MUTED)
    ax.plot([bx0 - 0.30, bx0 - 0.30], [top - 2.4 * row_h + 0.12,
                                       top + 0.20], color=BLUE, lw=2.2,
            solid_capstyle="round", zorder=2)
    ax.text(bx0 - 0.36, top - 0.70, "group A", rotation=90, ha="center",
            va="center", fontsize=6.6, color=BLUE, fontweight="bold")
    ax.plot([bx0 - 0.30, bx0 - 0.30], [top - 5.4 * row_h + 0.12,
                                       top - 2.8 * row_h + 0.20],
            color=TEAL, lw=2.2, solid_capstyle="round", zorder=2)
    ax.text(bx0 - 0.36, top - 3.70, "group B", rotation=90, ha="center",
            va="center", fontsize=6.6, color=TEAL, fontweight="bold")

    # ---- main, right: the two grids and the propagation ---------------------
    a_vals = np.array([[group_a(t + 1, n + 1, peak["seg"][2])
                        for n in range(3)] for t in range(4)], dtype=float)
    b_vals = np.array([[group_b(u + 1, l + 1, peak["seg"][5])
                        for l in range(2)] for u in range(4)], dtype=float)
    gx, gy, cell = 3.68, H - 1.58, 0.215
    grid_table(ax, gx, gy, cell, 4, 3, a_vals,
               (peak["seg"][0] - 1, peak["seg"][1] - 1), "table A",
               "trunk score", "neck score 1-3")
    grid_table(ax, gx + 1.30, gy, cell, 4, 2, b_vals,
               (peak["seg"][3] - 1, peak["seg"][4] - 1), "table B",
               "upper arm score", "lower arm 1-2")
    ax.text(gx + 3 * cell / 2.0, gy - 0.22,
            f"legs score {peak['seg'][2]} fixed", ha="center", va="top",
            fontsize=6, color=MUTED)
    ax.text(gx + 1.30 + 2 * cell / 2.0, gy - 0.22,
            f"wrist score {peak['seg'][5]} fixed", ha="center", va="top",
            fontsize=6, color=MUTED)

    def box(x, y, w, h, text, fc, ec, fs=6.6, tc=INK, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=fc, edgecolor=ec, lw=0.9,
                                    mutation_aspect=1.0, zorder=4))
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight, zorder=6,
                linespacing=1.4)
        return (x + w / 2.0, y), (x + w / 2.0, y + h)

    ya = gy - 0.72
    ba, ta = box(gx - 0.05, ya, 0.98, 0.40,
                 f"A {peak['a_raw']} + load {LOAD_ADDER}\n= {peak['a']}",
                 lighten(BLUE, 0.84), BLUE, fs=6.4)
    bb, tb = box(gx + 1.24, ya, 0.98, 0.40,
                 f"B {peak['b_raw']} + coupling {COUPLING_ADDER}\n"
                 f"= {peak['b']}", lighten(TEAL, 0.84), TEAL, fs=6.4)
    arrow(ax, (gx + 3 * cell / 2.0, gy - 0.26), ta, color=BLUE, lw=0.9, ms=7)
    arrow(ax, (gx + 1.30 + cell, gy - 0.26), tb, color=TEAL, lw=0.9, ms=7)

    yc = ya - 0.62
    bc, tc_ = box(gx + 0.44, yc, 1.32, 0.38,
                  f"table C ({peak['a']}, {peak['b']})  ->  {peak['c']}",
                  SOFT, HAIR, fs=6.6)
    arrow(ax, ba, (gx + 0.80, yc + 0.38), color=BLUE, lw=0.9, ms=7)
    arrow(ax, bb, (gx + 1.40, yc + 0.38), color=TEAL, lw=0.9, ms=7)

    col, lab = risk_band(peak["total"])
    yd = yc - 0.60
    bd, td = box(gx + 0.28, yd, 1.64, 0.40,
                 f"+ activity {ACTIVITY_ADDER}   ->   REBA score "
                 f"{peak['total']}", lighten(col, 0.78), col, fs=7.0,
                 weight="bold")
    arrow(ax, bc, td, color=SUB, lw=0.9, ms=7)

    # risk band ruler
    rx0, rw, ry = gx - 0.05, 2.20, yd - 0.62
    for lo, hi, bcol, blab in BAND_COLORS:
        x = rx0 + rw * (lo - 1) / 15.0
        w = rw * (hi - lo + 1) / 15.0
        ax.add_patch(Rectangle((x, ry), w, 0.17, facecolor=bcol,
                               edgecolor="white", lw=0.6, zorder=3))
        if w > 0.24:
            ax.text(x + w / 2.0, ry - 0.05, blab, ha="center", va="top",
                    fontsize=6, color=SUB)
    xm = rx0 + rw * (peak["total"] - 0.5) / 15.0
    ax.plot([xm], [ry + 0.245], marker="v", ms=6.0, mfc=INK, mec="white",
            mew=0.7, zorder=6)
    ax.text(xm, ry + 0.30, f"{peak['total']} = {lab} risk", ha="center",
            va="bottom", fontsize=6.6, color=INK, fontweight="bold")
    ax.text(rx0, ry + 0.02, "1", ha="right", va="bottom", fontsize=6,
            color=MUTED)
    ax.text(rx0 + rw + 0.03, ry + 0.02, "15", ha="left", va="bottom",
            fontsize=6, color=MUTED)

    # ---- aux (b): the score through the shift ------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    over = totals >= ACTION_LIMIT
    edges = np.concatenate([[tt[0]], 0.5 * (tt[1:] + tt[:-1]), [tt[-1]]])
    for k in range(len(tt)):
        if over[k]:
            ax_t.axvspan(edges[k], edges[k + 1], color=lighten(RUST, 0.80),
                         lw=0, zorder=1)
    ax_t.step(tt, totals, where="mid", color=BLUE, lw=1.4, zorder=4)
    ax_t.plot(tt[over], totals[over], ls="none", marker="o", ms=3.0,
              mfc=RUST, mec="white", mew=0.4, zorder=5)
    ax_t.axhline(ACTION_LIMIT, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)),
                 zorder=3)
    ax_t.plot([tt[worst]], [totals[worst]], marker="v", ms=5.0, mfc=INK,
              mec="white", mew=0.6, zorder=6)
    ax_t.text(tt[worst] + 0.18, totals[worst] + 0.55, "panel (a)",
              ha="left", va="bottom", fontsize=6.1, color=INK)
    ax_t.text(-0.05, ACTION_LIMIT + 0.25, f"action limit {ACTION_LIMIT}",
              ha="left", va="bottom", fontsize=6.1, color=ACCENT)
    ax_t.set_xlim(-0.15, 8.15)
    ax_t.set_ylim(0.0, 14.0)
    ax_t.set_xticks(np.arange(0, 9, 1))
    ax_t.set_yticks([0, 4, 8, 12])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("time into shift (h)", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("REBA-style score", fontsize=7.6, labelpad=2.0)
    ax_t.set_title(f"Posture drifts as the shift wears on, n={len(tt)} "
                   "samples", fontsize=8.4, fontweight="bold", pad=4.0)
    ax_t.text(-0.05, 0.4, f"{100.0 * over.mean():.0f} % of samples at or "
              "above the action limit", ha="left", va="bottom", fontsize=6.1,
              color=MUTED)

    # ---- aux (c): mean contribution of each segment ------------------------
    seg_mat = np.array([r["seg"] for r in res], dtype=float)
    means = seg_mat.mean(axis=0)
    order = np.argsort(means)
    labels = [SEGMENTS[i][0] for i in order]
    vals = means[order]
    cols = [SCORE_COLORS[int(np.clip(round(v), 1, 4))] for v in vals]
    ypos = np.arange(len(vals))
    ax_b.grid(axis="x", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.barh(ypos, vals, height=0.60, color=cols, edgecolor="white",
              linewidth=0.7, zorder=3)
    for k in ypos:
        ax_b.text(vals[k] + 0.04, k, f"{vals[k]:.2f}", ha="left", va="center",
                  fontsize=6.3, color=INK)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(labels, fontsize=6.4)
    ax_b.set_ylim(-0.7, len(vals) - 0.3)
    ax_b.set_xlim(0.0, 4.0)
    ax_b.set_xticks([0, 1, 2, 3, 4])
    ax_b.tick_params(axis="x", labelsize=6.6)
    ax_b.set_xlabel("mean segment score over the shift", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_title("Where the score comes from", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.44, dy=0.10)
    savefig(fig, "952_hfe_posture_risk.png")


if __name__ == "__main__":
    main()
