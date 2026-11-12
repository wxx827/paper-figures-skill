"""Post-mortem interval from body cooling, rigor, livor and entomology.

The Henssge double-exponential is evaluated once and inverted numerically for
the measured deep body temperature, so the cooling interval drawn as a bar is
the same one the curve crosses.  The joint window is the intersection of the
four method bars, computed from the bar list, and the posterior panel
multiplies soft versions of exactly those windows.

Synthetic illustration for a visual reference library; not a case record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches

ORANGE = "#c0603f"          # reconstruction / joint estimate
BLUE = "#3b5b92"            # individual methods
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
CARD = "#f5f5f5"

FW, FH = 10.4, 6.4
MAIN = (0.80, 0.88, 5.10, 4.46)
AUX_T = (6.88, 3.59, 2.96, 1.75)
AUX_B = (6.88, 0.88, 2.96, 1.75)

T_START = 37.2              # rectal temperature at death, deg C
T_AMB = 18.0                # scene ambient, deg C
T_MEAS = 27.4               # measured deep body temperature, deg C
MASS = 75.0                 # body mass, kg
COOL_TOL = 2.8              # Henssge 95 % tolerance, h
TMAX = 48.0

DIVIDE = 16.0               # boundary between the curve and the bar band
YLO, YHI = 1.6, 39.5
BAR_Y = [13.4, 10.2, 7.0, 3.8]
BAR_H = 2.2


def henssge(t, t_amb, mass=MASS):
    b = -1.2815 * mass ** -0.625 + 0.0284
    q = 1.25 * np.exp(b * t) - 0.25 * np.exp(5.0 * b * t)
    return t_amb + (T_START - t_amb) * q


def invert(t_amb, t_meas=T_MEAS):
    t = np.linspace(0.05, 120.0, 24000)
    c = henssge(t, t_amb)
    k = int(np.argmin(np.abs(c - t_meas)))
    return float(t[k])


def soft_window(t, lo, hi, s=0.9):
    return (1.0 / (1.0 + np.exp(-(t - lo) / s)) *
            1.0 / (1.0 + np.exp(-(hi - t) / s)))


def draw_main(ax, tg, curve, t_cool, methods, joint):
    ax.axhspan(YLO, DIVIDE, facecolor=CARD, edgecolor="none", zorder=0)
    ax.axhline(DIVIDE, color=HAIR, lw=0.8, zorder=1)
    ax.axvspan(joint[0], joint[1], facecolor=ORANGE, alpha=0.18,
               edgecolor="none", zorder=1)
    for v in joint:
        ax.plot([v, v], [YLO, YHI], color=ORANGE, lw=0.9,
                ls=(0, (3.0, 2.0)), zorder=2)

    ax.plot(tg, curve, color=ORANGE, lw=1.7, zorder=5)
    ax.axhline(T_AMB, color=GREY, lw=0.8, ls=(0, (4.0, 2.5)), zorder=3)
    ax.plot([0.0, t_cool], [T_MEAS, T_MEAS], color=INK, lw=0.9,
            ls=(0, (2.4, 1.8)), zorder=4)
    ax.plot([t_cool], [T_MEAS], marker="o", ms=4.6, color=INK, mec="white",
            mew=0.6, zorder=6)
    ax.text(0.6, T_MEAS + 0.45, f"measured {T_MEAS:.1f} " + r"$^\circ$C",
            ha="left", va="bottom", fontsize=6.6, color=INK, zorder=6)
    ax.text(TMAX - 0.6, T_AMB - 0.5, f"ambient {T_AMB:.1f} " + r"$^\circ$C",
            ha="right", va="top", fontsize=6.6, color=SUB, zorder=6)
    ax.text(t_cool + 0.6, 34.6, f"Henssge cooling model\n{MASS:.0f} kg, "
            f"standard conditions", ha="left", va="top", fontsize=6.6,
            color=ORANGE, linespacing=1.35, zorder=6)

    for y, (name, lo, hi, col) in zip(BAR_Y, methods):
        ax.add_patch(plt.Rectangle((lo, y - BAR_H / 2.0), hi - lo, BAR_H,
                                   facecolor=col, alpha=0.80,
                                   edgecolor="white", linewidth=0.6,
                                   zorder=4))
        ax.text(0.6, y, name, ha="left", va="center", fontsize=6.8,
                color=INK, zorder=6)
        ax.text(hi + 0.5, y, f"{lo:.1f} - {hi:.1f} h", ha="left",
                va="center", fontsize=6.4, color=SUB, zorder=6)

    ax.text(0.5 * (joint[0] + joint[1]), 38.9,
            f"joint window {joint[0]:.1f} - {joint[1]:.1f} h", ha="center",
            va="top", fontsize=7.0, fontweight="bold", color=ORANGE,
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor=ORANGE, linewidth=0.7, alpha=0.94))
    ax.text(TMAX - 0.6, DIVIDE - 0.6, "method intervals", ha="right",
            va="top", fontsize=6.6, color=MUTED, zorder=6)

    ax.set_xlim(0.0, TMAX)
    ax.set_ylim(YLO, YHI)
    ax.set_xticks(np.arange(0.0, TMAX + 0.1, 6.0))
    ax.set_yticks(np.arange(18.0, 38.1, 4.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("time since death (h)", fontsize=8.0)
    ax.set_ylabel(r"deep body temperature ($^\circ$C)", fontsize=8.0)
    ax.set_title("Four independent methods narrowed to one window",
                 fontsize=9.0, fontweight="bold", color=INK)


def draw_sensitivity(ax, tg, ambs, t_hits):
    cols = get_palette("cb_sequential_blue", len(ambs) + 2)[1:-1]
    for ta, col in zip(ambs, cols):
        ax.plot(tg, henssge(tg, ta), color=col, lw=1.2)
        ax.text(TMAX - 0.5, henssge(TMAX, ta) + 0.35, f"{ta:.0f}", ha="right",
                va="bottom", fontsize=6.0, color=col, zorder=5,
                bbox=dict(boxstyle="square,pad=0.10", facecolor="white",
                          edgecolor="none", alpha=0.85))
    ax.axhline(T_MEAS, color=INK, lw=0.8, ls=(0, (2.4, 1.8)))
    ax.plot(t_hits, np.full(len(t_hits), T_MEAS), marker="o", ms=3.2,
            ls="none", color=ORANGE, mec="white", mew=0.4)
    ax.text(0.97, 0.95, "ambient (" + r"$^\circ$C" + ") labelled at 48 h\n"
            f"PMI spread {min(t_hits):.1f} - {max(t_hits):.1f} h",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=SUB, linespacing=1.35)
    ax.set_xlim(0.0, TMAX)
    ax.set_ylim(6.0, 39.0)
    ax.set_xticks(np.arange(0.0, TMAX + 0.1, 12.0))
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("time since death (h)", fontsize=7.0)
    ax.set_ylabel(r"body temperature ($^\circ$C)", fontsize=7.0)
    ax.set_title("Ambient sensitivity of the cooling curve", fontsize=8.5,
                 fontweight="bold", color=INK)


def draw_posterior(ax, methods, joint):
    t = np.linspace(0.0, TMAX, 2400)
    post = np.ones_like(t)
    for name, lo, hi, _c in methods:
        w = soft_window(t, lo, hi)
        ax.plot(t, w * 0.92, color=BLUE, lw=0.8, alpha=0.55)
        post = post * w
    post = post / np.trapezoid(post, t)
    ax.fill_between(t, 0.0, post / post.max() * 0.92, color=ORANGE,
                    alpha=0.28, lw=0.0)
    ax.plot(t, post / post.max() * 0.92, color=ORANGE, lw=1.5)

    cdf = np.cumsum(post) / np.sum(post)
    lo95 = float(np.interp(0.025, cdf, t))
    hi95 = float(np.interp(0.975, cdf, t))
    for v in (lo95, hi95):
        ax.plot([v, v], [0.0, 0.98], color=ORANGE, lw=0.8,
                ls=(0, (3.0, 2.0)))
    ax.text(0.97, 0.95, f"95 % credible\n{lo95:.1f} - {hi95:.1f} h\n"
                        f"intersection {joint[0]:.1f} - {joint[1]:.1f} h",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=SUB, linespacing=1.35)
    ax.set_xlim(0.0, TMAX)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks(np.arange(0.0, TMAX + 0.1, 12.0))
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("time since death (h)", fontsize=7.0)
    ax.set_ylabel("relative support (-)", fontsize=7.0)
    ax.set_title("Single-method windows and their product", fontsize=8.5,
                 fontweight="bold", color=INK)
    return lo95, hi95


def main():
    set_pub_style()

    tg = np.linspace(0.0, TMAX, 1200)
    curve = henssge(tg, T_AMB)
    t_cool = invert(T_AMB)

    methods = [("body cooling", t_cool - COOL_TOL, t_cool + COOL_TOL,
                ORANGE),
               ("rigor mortis", 9.0, 24.0, BLUE),
               ("livor mortis", 10.0, 20.5, BLUE),
               ("entomology (ADH)", 14.5, 32.0, BLUE)]
    joint = (max(m[1] for m in methods), min(m[2] for m in methods))

    ambs = [10.0, 14.0, 18.0, 22.0, 26.0]
    t_hits = [invert(ta) for ta in ambs]

    fig = plt.figure(figsize=(FW, FH))
    ax_main = axes_inches(fig, MAIN)
    ax_sen = axes_inches(fig, AUX_T)
    ax_post = axes_inches(fig, AUX_B)

    draw_main(ax_main, tg, curve, t_cool, methods, joint)
    draw_sensitivity(ax_sen, tg, ambs, t_hits)
    lo95, hi95 = draw_posterior(ax_post, methods, joint)


    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.64, dy=0.14)
    panel_labels_inches(fig, [ax_sen, ax_post], labels=["(b)", "(c)"],
                        dx=-0.54, dy=0.14)
    savefig(fig, "797_frn_pmi_estimation.png")


if __name__ == "__main__":
    main()
