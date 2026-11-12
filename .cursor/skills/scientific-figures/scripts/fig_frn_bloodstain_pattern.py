"""Bloodstain pattern analysis: directionality, area of convergence, origin.

Every stain is drawn as an ellipse whose axes are the ones the arithmetic then
reads back: the impact angle is recomputed as asin(minor / major) from the
drawn patch, the area of convergence is the least-squares intersection of the
drawn major axes, and the height of origin is the median of
distance x tan(angle) over all stains.  Changing the seed moves every number on
the page.

Synthetic illustration for a visual reference library; not a case record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

ORANGE = "#c0603f"          # reconstruction / estimate
BLUE = "#3b5b92"            # measured population
STAIN = "#8c2f2f"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

FW, FH = 10.4, 6.4
MAIN = (0.78, 0.88, 5.12, 4.46)
AUX_T = (6.88, 3.59, 2.96, 1.75)
AUX_B = (6.88, 0.88, 2.96, 1.75)

XMAX, YMAX = 52.0, 45.2      # substrate window, cm
SRC = np.array([25.0, 22.0])  # ground projection used to synthesise, cm
H_TRUE = 12.0                 # height used to synthesise, cm
N_STAIN = 18


def make_stains(rng):
    ang = np.sort(rng.uniform(-0.62, 3.76, N_STAIN))
    r = rng.uniform(7.0, 22.0, N_STAIN)
    pos = SRC[None, :] + np.c_[r * np.cos(ang), r * np.sin(ang)]
    minor = rng.uniform(1.30, 2.00, N_STAIN)
    major = minor / np.sin(np.arctan2(H_TRUE, r)) * rng.uniform(0.94, 1.06,
                                                                N_STAIN)
    major = np.maximum(major, minor * 1.02)
    orient = ang + rng.normal(0.0, np.deg2rad(3.2), N_STAIN)
    alpha = np.arcsin(np.clip(minor / major, 0.0, 1.0))     # rad, from patch
    return pos, minor, major, orient, alpha


def converge(pos, orient):
    d = np.c_[np.cos(orient), np.sin(orient)]
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, u in zip(pos, d):
        P = np.eye(2) - np.outer(u, u)
        A += P
        b += P @ p
    return np.linalg.solve(A, b), d


def pair_spread(pos, d, conv):
    pts = []
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            cr = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
            if abs(cr) < 0.18:
                continue
            dp = pos[j] - pos[i]
            t = (dp[0] * d[j, 1] - dp[1] * d[j, 0]) / cr
            q = pos[i] + t * d[i]
            if np.hypot(*(q - conv)) < 11.0:
                pts.append(q)
    return np.array(pts)


def draw_scene(ax, pos, minor, major, orient, alpha, conv, cov, heights):
    ax.set_facecolor("#fbfbfb")
    ax.grid(True, color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)

    for p, u, c in zip(pos, np.c_[np.cos(orient), np.sin(orient)], conv - pos):
        t = float(np.dot(c, u)) * 1.10
        ax.plot([p[0], p[0] + t * u[0]], [p[1], p[1] + t * u[1]],
                color=GREY, lw=0.6, ls=(0, (3.0, 2.2)), zorder=2)

    for p, w, L, o in zip(pos, minor, major, orient):
        ax.add_patch(Ellipse(p, L, w, angle=np.rad2deg(o), facecolor=STAIN,
                             edgecolor="#5f1f1f", linewidth=0.4, alpha=0.88,
                             zorder=4))

    vals, vecs = np.linalg.eigh(cov)
    k = np.sqrt(5.991)
    ax.add_patch(Ellipse(conv, 2 * k * np.sqrt(vals[1]),
                         2 * k * np.sqrt(vals[0]),
                         angle=np.rad2deg(np.arctan2(vecs[1, 1], vecs[0, 1])),
                         facecolor=ORANGE, alpha=0.16, edgecolor=ORANGE,
                         linewidth=1.0, linestyle=(0, (3.0, 2.0)), zorder=5))
    ax.plot([conv[0]], [conv[1]], marker="x", ms=8.0, mew=1.6, color=ORANGE,
            zorder=6)

    order = np.argsort(alpha)
    for k_i in (order[0], order[len(order) // 2], order[-1]):
        p, a = pos[k_i], np.rad2deg(alpha[k_i])
        ax.text(p[0], p[1] + 0.55 * major[k_i] + 0.4, f"{a:.0f} deg",
                ha="center", va="bottom", fontsize=6.2, color=INK, zorder=7,
                bbox=dict(boxstyle="square,pad=0.14", facecolor="white",
                          edgecolor="none", alpha=0.85))

    q1, q3 = np.percentile(heights, [25.0, 75.0])
    ax.text(0.025, 0.975,
            f"area of convergence\n  x = {conv[0]:.1f} cm, "
            f"y = {conv[1]:.1f} cm\n"
            f"height of origin (median)\n  {np.median(heights):.1f} cm  "
            f"(IQR {q1:.1f}-{q3:.1f} cm)\n"
            f"stains used  n = {len(pos)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=INK, linespacing=1.45, zorder=8,
            bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.94))

    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("substrate x (cm)", fontsize=8.0)
    ax.set_ylabel("substrate y (cm)", fontsize=8.0)
    ax.set_title("Stain long axes projected back to the area of convergence",
                 fontsize=9.0, fontweight="bold", color=INK)


def draw_angles(ax, alpha):
    deg = np.rad2deg(alpha)
    ax.hist(deg, bins=np.arange(16.0, 72.0, 4.0), color=BLUE,
            edgecolor="white", linewidth=0.4)
    m = float(np.mean(deg))
    ax.axvline(m, color=ORANGE, lw=1.0, ls=(0, (3.0, 2.0)))
    ax.text(m + 1.0, ax.get_ylim()[1] * 0.94,
            f"mean {m:.1f} deg\nrange {deg.min():.0f}-{deg.max():.0f} deg",
            ha="left", va="top", fontsize=6.2, color=SUB, linespacing=1.35)
    ax.set_xlim(16.0, 70.0)
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("impact angle asin(minor/major) (deg)", fontsize=7.0)
    ax.set_ylabel("stains (count)", fontsize=7.0)
    ax.set_title("Impact angle distribution", fontsize=8.5,
                 fontweight="bold", color=INK)


def draw_size_speed(ax, rng):
    n = 150
    dia = rng.uniform(2.4, 5.6, n)
    vel = 9.6 * dia ** (-0.72) * np.exp(rng.normal(0.0, 0.085, n))
    lg = np.polyfit(np.log(dia), np.log(vel), 1)
    xs = np.linspace(dia.min(), dia.max(), 120)
    r = float(np.corrcoef(np.log(dia), np.log(vel))[0, 1])
    ax.scatter(dia, vel, s=8.0, color=BLUE, alpha=0.55, linewidths=0.0)
    ax.plot(xs, np.exp(lg[1]) * xs ** lg[0], color=ORANGE, lw=1.4)
    ax.text(0.97, 0.94, f"fit exponent {lg[0]:.2f}\nPearson r = {r:.3f}  "
                        f"(n = {n})", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.2, color=SUB, linespacing=1.35)
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("drop diameter (mm)", fontsize=7.0)
    ax.set_ylabel("impact velocity (m/s)", fontsize=7.0)
    ax.set_title("Drop size against impact speed", fontsize=8.5,
                 fontweight="bold", color=INK)
    return lg[0], r


def main():
    set_pub_style()
    rng = np.random.default_rng(715)

    pos, minor, major, orient, alpha = make_stains(rng)
    conv, d = converge(pos, orient)
    pts = pair_spread(pos, d, conv)
    cov = np.cov(pts.T) if len(pts) > 8 else np.eye(2) * 0.5
    dist = np.hypot(*(pos - conv[None, :]).T)
    heights = dist * np.tan(alpha)

    fig = plt.figure(figsize=(FW, FH))
    ax_main = axes_inches(fig, MAIN)
    ax_ang = axes_inches(fig, AUX_T)
    ax_sz = axes_inches(fig, AUX_B)

    draw_scene(ax_main, pos, minor, major, orient, alpha, conv, cov, heights)
    draw_angles(ax_ang, alpha)
    expo, r = draw_size_speed(ax_sz, rng)

    q1, q3 = np.percentile(heights, [25.0, 75.0])

    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.62, dy=0.14)
    panel_labels_inches(fig, [ax_ang, ax_sz], labels=["(b)", "(c)"],
                        dx=-0.54, dy=0.14)
    savefig(fig, "795_frn_bloodstain_pattern.png")


if __name__ == "__main__":
    main()
