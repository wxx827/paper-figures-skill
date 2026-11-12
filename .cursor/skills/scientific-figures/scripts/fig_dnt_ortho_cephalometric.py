"""Lateral cephalometric tracing with the four classical skeletal angles.

Everything numerical in this figure is derived from one landmark table.  SNA
and SNB are signed angles measured from the same N->S reference ray, so ANB is
computed as their difference and cannot drift away from them; FMA is the angle
between the Frankfort and mandibular planes and is drawn as an arc at gonion
against a construction line parallel to Frankfort, which is the same angle by
corresponding angles.  The post-treatment tracing is a rigid rotation of the
mandibular outline about the condyle plus an incisor movement, so the dashed
outline and the post-treatment landmark coordinates are one and the same
object.  Panel (b) reports every measurement in standard-deviation units
because the six angles differ by an order of magnitude in absolute value;
panel (c) keeps degrees, where the changes are directly comparable.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.86, 5.58, 4.56)
AUX_T = (7.00, 3.52, 3.12, 1.90)
AUX_B = (7.00, 0.86, 3.12, 1.76)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#d5d5d5"
RED, BLUE, GREEN, GOLD, PURPLE = ("#b23a48", "#3b5b92", "#2e7d5b",
                                  "#c9a227", "#8e6bbf")
BONE = "#8a8a8a"            # hard-tissue tracing
SOFT = "#b0a496"            # soft-tissue profile

CONDYLE = np.array([-45.0, 13.0])
ROT_POST = 3.0              # mandibular autorotation after treatment, deg

LANDMARKS = {
    "S": (-42.0, 18.0), "N": (12.0, 34.0), "Po": (-58.0, 12.0),
    "Or": (-2.0, 12.0), "ANS": (22.0, -4.0), "PNS": (-18.0, -6.0),
    "A": (20.0, -12.0), "B": (17.0, -44.0), "Pog": (21.0, -57.0),
    "Gn": (17.0, -63.0), "Me": (12.0, -66.0), "Go": (-42.0, -42.0),
}
MANDIBULAR = ("B", "Pog", "Gn", "Me", "Go")

LABEL_OFF = {
    "S": (-4, 4, "right", "bottom"), "N": (4, 3, "left", "bottom"),
    "Po": (-4, -3, "right", "top"), "Or": (1, 5, "left", "bottom"),
    "ANS": (5, 3, "left", "bottom"), "PNS": (-5, 0, "right", "center"),
    "A": (6, -1, "left", "top"), "B": (6, 0, "left", "center"),
    "Pog": (6, 0, "left", "center"), "Gn": (6, -3, "left", "top"),
    "Me": (0, -6, "center", "top"), "Go": (-6, -3, "right", "top"),
}

VAULT = [(4, 60), (-14, 66), (-38, 60), (-58, 44), (-68, 24), (-69, 6),
         (-62, -8), (-56, -16)]
PROFILE = [(-58, -26), (-72, -6), (-80, 16), (-76, 42), (-58, 64), (-34, 76),
           (-8, 78), (14, 70), (23, 55), (26, 40), (23, 31), (32, 22),
           (48, 5), (38, 0), (36, -8), (42, -16), (37, -24), (42, -31),
           (36, -41), (34, -52), (27, -68), (10, -76), (-8, -78), (-24, -70),
           (-34, -54)]
MAXILLA = [(-18, -6), (2, -7), (22, -4), (20, -12), (16, -25), (-2, -26),
           (-18, -22)]
MANDIBLE = [(-45, 13), (-48, 6), (-50, -5), (-51, -17), (-49, -29),
            (-45, -37), (-42, -42), (-31, -53), (-16, -61), (-2, -65),
            (12, -66), (17, -63), (21, -57), (19, -50), (17, -44),
            (15, -36), (2, -34), (-12, -33), (-24, -32), (-31, -27),
            (-33, -16), (-33, 2), (-38, -10), (-42, 0), (-44, 7)]
U1 = [(16.0, -14.0), (30.0, -32.0)]     # apex, incisal tip
L1 = [(20.0, -50.0), (27.0, -34.0)]

NORMS = {"SNA": (82.0, 3.5), "SNB": (80.0, 3.0), "ANB": (2.0, 2.0),
         "FMA": (25.0, 4.5), "U1-SN": (103.0, 5.5), "IMPA": (95.0, 7.0)}
ORDER = ["SNA", "SNB", "ANB", "FMA", "U1-SN", "IMPA"]
ARC_COLOR = {"SNA": BLUE, "SNB": GREEN, "ANB": RED, "FMA": GOLD}

FOOT = ("Synthetic lateral cephalometric tracing, not a patient radiograph. "
        "Solid outline is pre-treatment, dashed is post-treatment; the "
        "post-treatment mandible is a rigid rotation of the pre-treatment "
        "outline about the condyle, so its landmarks and its silhouette stay "
        "consistent by construction. ANB is computed as SNA minus SNB from "
        "signed angles about the N-S ray, and FMA is drawn at gonion against "
        "a line parallel to Frankfort. Normal bands are population mean "
        "+/- 1 SD.")


def fit_limits(ax, xlim, ylim, rect):
    """Pad the shorter data axis so an equal-aspect panel fills its box."""
    want = rect[2] / rect[3]
    (x0, x1), (y0, y1) = xlim, ylim
    if (x1 - x0) / (y1 - y0) < want:
        e = (y1 - y0) * want - (x1 - x0)
        x0, x1 = x0 - e / 2.0, x1 + e / 2.0
    else:
        e = (x1 - x0) / want - (y1 - y0)
        y0, y1 = y0 - e / 2.0, y1 + e / 2.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


def spline(pts, n=16, closed=False, tension=0.5):
    """Cardinal spline through the control points.

    ``tension=0.5`` is plain Catmull-Rom; lower values tighten the curve.  A
    sharp corner such as the coronoid process makes Catmull-Rom overshoot into
    a visible loop, which on an anatomical tracing reads as a second structure
    that is not there, so the bone outlines use a reduced tension.
    """
    p = np.asarray(pts, dtype=float)
    p = (np.vstack([p[-1], p, p[0], p[1]]) if closed
         else np.vstack([p[0], p, p[-1]]))
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    out = []
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i:i + 4]
        out.append(h00 * p1 + h10 * tension * (p2 - p0)
                   + h01 * p2 + h11 * tension * (p3 - p1))
    return np.vstack(out)


def rotate(pts, centre, deg):
    p = np.atleast_2d(np.asarray(pts, dtype=float)) - centre
    a = np.deg2rad(deg)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return p @ m.T + centre


def signed_angle(v_ref, v):
    """Angle from ``v_ref`` to ``v`` in degrees, in (-180, 180]."""
    a = np.arctan2(v[1], v[0]) - np.arctan2(v_ref[1], v_ref[0])
    return float(np.degrees((a + np.pi) % (2.0 * np.pi) - np.pi))


def measure(lm, u1, l1):
    """The six cephalometric angles, all from one landmark table."""
    p = {k: np.asarray(v, dtype=float) for k, v in lm.items()}
    ns = p["S"] - p["N"]
    sna = signed_angle(ns, p["A"] - p["N"])
    snb = signed_angle(ns, p["B"] - p["N"])
    u1v = np.asarray(u1[1], dtype=float) - np.asarray(u1[0], dtype=float)
    l1v = np.asarray(l1[1], dtype=float) - np.asarray(l1[0], dtype=float)
    return {"SNA": sna, "SNB": snb, "ANB": sna - snb,
            "FMA": abs(signed_angle(p["Or"] - p["Po"], p["Me"] - p["Go"])),
            "U1-SN": abs(signed_angle(ns, u1v)),
            "IMPA": abs(signed_angle(p["Me"] - p["Go"], l1v))}


def post_treatment():
    """Rigidly rotate the mandible, retract U1, procline L1."""
    lm = dict(LANDMARKS)
    for k in MANDIBULAR:
        lm[k] = tuple(rotate(LANDMARKS[k], CONDYLE, ROT_POST)[0])
    mand = rotate(MANDIBLE, CONDYLE, ROT_POST)
    l1 = rotate(L1, CONDYLE, ROT_POST)
    l1 = rotate(l1, l1[0], 5.0)                    # slight proclination
    u1 = np.array([[16.5, -14.0], [26.2, -31.4]])  # retracted incisal tip
    return lm, mand, u1, l1


def incisor(ax, axis_pts, color, ls="-", lw=1.2, zorder=6):
    """Draw an incisor as a tapered blade along its long axis."""
    a, b = np.asarray(axis_pts, dtype=float)
    v = b - a
    n = np.array([-v[1], v[0]]) / np.hypot(*v)
    poly = np.array([a + 0.8 * n, a + 0.45 * v + 2.2 * n, b + 1.5 * n,
                     b - 1.5 * n, a + 0.45 * v - 2.2 * n, a - 0.8 * n])
    ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor=color,
                         lw=lw, ls=ls, zorder=zorder))


def angle_arc(ax, vertex, v1, v2, r, color):
    """Unlabelled arc between two rays; the value lives in the key column."""
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    if (a2 - a1 + np.pi) % (2 * np.pi) - np.pi < 0:
        a1, a2 = a2, a1
    t = np.linspace(a1, a1 + (a2 - a1) % (2 * np.pi), 90)
    ax.plot(vertex[0] + r * np.cos(t), vertex[1] + r * np.sin(t),
            color=color, lw=1.1, zorder=7)


def key_column(ax, x0, y0, rows, dy=7.4, head_dy=10.6):
    """Stack the tracing key in the space equal aspect leaves on the left."""
    y = y0
    for kind, text, style in rows:
        if kind == "head":
            y -= head_dy
            ax.text(x0, y, text, ha="left", va="center", fontsize=7.2,
                    fontweight="bold", color=INK)
        else:
            y -= dy
            col, ls, lw = style
            ax.plot([x0, x0 + 12.0], [y, y], color=col, ls=ls, lw=lw,
                    solid_capstyle="butt")
            ax.text(x0 + 16.0, y, text, ha="left", va="center", fontsize=6.4,
                    color=SUB)
    return y


def main():
    set_pub_style()
    fig = plt.figure(figsize=(FW, FH))

    pre = measure(LANDMARKS, U1, L1)
    lm_post, mand_post, u1_post, l1_post = post_treatment()
    post = measure(lm_post, u1_post, l1_post)

    # ---- (a) tracing --------------------------------------------------------
    ax = axes_inches(fig, MAIN)
    ax.axis("off")
    P = {k: np.asarray(v, dtype=float) for k, v in LANDMARKS.items()}

    ax.plot(*spline(PROFILE).T, color=SOFT, lw=1.4, zorder=3)
    ax.plot(*spline(VAULT).T, color=BONE, lw=1.1, zorder=3)
    ax.add_patch(Polygon(spline(MAXILLA, closed=True, tension=0.34),
                         closed=True, facecolor="none", edgecolor=BONE,
                         lw=1.0, zorder=3))
    ax.add_patch(Polygon(spline(MANDIBLE, closed=True, tension=0.30),
                         closed=True, facecolor="none", edgecolor=BONE,
                         lw=1.2, zorder=3))
    incisor(ax, U1, BONE)
    incisor(ax, L1, BONE)

    ax.add_patch(Polygon(spline(mand_post, closed=True, tension=0.30),
                         closed=True, facecolor="none", edgecolor=PURPLE,
                         lw=1.0, ls=(0, (3.2, 2.0)), zorder=4))
    incisor(ax, u1_post, PURPLE, ls=(0, (2.4, 1.8)), lw=1.0, zorder=5)
    incisor(ax, l1_post, PURPLE, ls=(0, (2.4, 1.8)), lw=1.0, zorder=5)

    # Reference planes.
    ax.plot([P["S"][0], P["N"][0]], [P["S"][1], P["N"][1]], color=RULE,
            lw=0.9, zorder=5)
    ax.plot([-70, 10], [12, 12], color=RULE, lw=0.8, ls=(0, (4.0, 2.4)),
            zorder=5)
    mp = np.vstack([P["Go"], P["Me"] + 0.34 * (P["Me"] - P["Go"])])
    ax.plot(mp[:, 0], mp[:, 1], color=GOLD, lw=1.1, zorder=5)
    ax.plot([P["ANS"][0], P["PNS"][0]], [P["ANS"][1], P["PNS"][1]],
            color=MUTED, lw=0.7, ls=(0, (2.4, 2.0)), zorder=5)
    for k, col in (("A", BLUE), ("B", GREEN)):
        ax.plot([P["N"][0], P[k][0]], [P["N"][1], P[k][1]], color=col,
                lw=0.7, ls=(0, (3.0, 2.0)), zorder=5)

    ns = P["S"] - P["N"]
    angle_arc(ax, P["N"], ns, P["A"] - P["N"], 36.0, BLUE)
    angle_arc(ax, P["N"], ns, P["B"] - P["N"], 26.0, GREEN)
    angle_arc(ax, P["N"], P["A"] - P["N"], P["B"] - P["N"], 50.0, RED)
    ax.plot([P["Go"][0] - 3, P["Go"][0] + 42], [P["Go"][1]] * 2, color=GOLD,
            lw=0.7, ls=(0, (2.4, 2.0)), zorder=5)
    angle_arc(ax, P["Go"], np.array([1.0, 0.0]), P["Me"] - P["Go"], 27.0,
              GOLD)

    for k, (x, y) in LANDMARKS.items():
        ax.plot([x], [y], marker="o", ms=3.0, mfc="white", mec=INK, mew=0.8,
                zorder=9)
        dx, dy, ha, va = LABEL_OFF[k]
        ax.text(x + dx, y + dy, k, ha=ha, va=va, fontsize=6.6, color=INK,
                zorder=9)
    for k in ("B", "Pog", "Me"):
        x, y = lm_post[k]
        ax.plot([x], [y], marker="o", ms=2.8, mfc=PURPLE, mec="white",
                mew=0.5, zorder=9)

    rows = [("head", "Skeletal angles (pre-treatment)", None)]
    rows += [("row", f"{n}   {pre[n]:.1f} deg", (ARC_COLOR[n], "-", 1.4))
             for n in ("SNA", "SNB", "ANB", "FMA")]
    rows += [("head", "Reference planes", None),
             ("row", "SN (sella-nasion)", (RULE, "-", 0.9)),
             ("row", "Frankfort (Po-Or)", (RULE, (0, (4.0, 2.4)), 0.8)),
             ("row", "mandibular (Go-Me)", (GOLD, "-", 1.1)),
             ("row", "palatal (ANS-PNS)", (MUTED, (0, (2.4, 2.0)), 0.7)),
             ("head", "Tracing", None),
             ("row", "pre-treatment bone", (BONE, "-", 1.2)),
             ("row", "post-treatment bone", (PURPLE, (0, (3.2, 2.0)), 1.0)),
             ("row", "soft-tissue profile", (SOFT, "-", 1.4))]
    key_column(ax, -158.0, 84.0, rows)

    fit_limits(ax, (-162.5, 48.0), (-86.0, 86.0), MAIN)
    ax.set_title("Lateral tracing with skeletal and dental angles",
                 fontsize=8.8, fontweight="bold", color=INK, pad=4)

    # ---- (b) measurements against the normal band --------------------------
    ax_b = axes_inches(fig, AUX_T)
    yv = np.arange(len(ORDER), dtype=float)[::-1]
    ax_b.axvspan(-1.0, 1.0, color=GREEN, alpha=0.14, lw=0)
    ax_b.axvline(0.0, color=RULE, lw=0.8)
    for i, name in enumerate(ORDER):
        mu, sd = NORMS[name]
        zp, zq = (pre[name] - mu) / sd, (post[name] - mu) / sd
        ax_b.annotate("", xy=(zq, yv[i]), xytext=(zp, yv[i]),
                      arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                      shrinkA=1.5, shrinkB=1.5))
        ax_b.plot([zp], [yv[i]], marker="o", ms=4.0, color=BLUE, mec="white",
                  mew=0.5, zorder=5)
        ax_b.plot([zq], [yv[i]], marker="s", ms=4.0, color=PURPLE,
                  mec="white", mew=0.5, zorder=5)
        ax_b.text(3.85, yv[i], f"{pre[name]:.1f} / {post[name]:.1f}",
                  ha="right", va="center", fontsize=6.0, color=SUB)
    ax_b.set_yticks(yv)
    ax_b.set_yticklabels(ORDER, fontsize=6.6)
    ax_b.set_ylim(-0.6, len(ORDER) - 0.15)
    ax_b.set_xlim(-4.0, 4.0)
    ax_b.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax_b.set_xlabel("deviation from normal mean (SD)", fontsize=8.0)
    ax_b.tick_params(labelsize=6.6)
    ax_b.text(0.0, len(ORDER) - 0.22, r"normal band $\pm$1 SD", ha="center",
              va="top", fontsize=6.2, color=GREEN)
    ax_b.text(3.85, len(ORDER) - 0.22, "pre / post (deg)", ha="right",
              va="top", fontsize=6.0, color=SUB)
    ax_b.set_title("Measurements against the normal band", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    # ---- (c) treatment change ----------------------------------------------
    ax_c = axes_inches(fig, AUX_B)
    delta = np.array([post[n] - pre[n] for n in ORDER])
    cols = [RED if d < 0 else BLUE for d in delta]
    ax_c.barh(yv, delta, height=0.58, color=cols, edgecolor="white",
              linewidth=0.5)
    ax_c.axvline(0.0, color=RULE, lw=0.8)
    lim = float(np.abs(delta).max()) * 1.55
    for i, d in enumerate(delta):
        s = 1.0 if d >= 0 else -1.0
        ax_c.text(d + s * lim * 0.035, yv[i], f"{d:+.1f}",
                  ha="left" if d >= 0 else "right", va="center", fontsize=6.2,
                  color=SUB)
    ax_c.set_yticks(yv)
    ax_c.set_yticklabels(ORDER, fontsize=6.6)
    ax_c.set_ylim(-0.6, len(ORDER) - 0.4)
    ax_c.set_xlim(-lim, lim)
    ax_c.set_xlabel("change after treatment (deg)", fontsize=8.0)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(axis="x", color=HAIR, lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.set_title("Post minus pre", fontsize=8.5, fontweight="bold",
                   color=INK, pad=3)

    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.48, dy=0.10)
    savefig(fig, "819_dnt_ortho_cephalometric.png")
    for n in ORDER:
        print(f"  {n:6s} pre {pre[n]:7.2f}  post {post[n]:7.2f}")


if __name__ == "__main__":
    main()
