"""Process capability study: histogram against specification limits, index
cards, a reduced trend strip and the predicted defect rate in ppm."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
FILL = "#c3d3e2"
GREY = "#8b949e"
ALERT = "#b23a48"
AMBER = "#c9922f"
SOFT_RED = "#e6bcc2"
CARD_BG = "#f4f6f9"

LSL, USL, TARGET = 9.950, 10.170, 10.060


def rating(v):
    """Capability verdict colour: capable / marginal / not capable."""
    if v >= 1.33:
        return NAVY, "capable"
    if v >= 1.00:
        return AMBER, "marginal"
    return ALERT, "not capable"


def stamp_labels(fig, axes, labels, dx=0.052, dy=0.014, size=9):
    """Panel letters at a constant figure-space offset from each axes box."""
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + dy, lab, fontsize=size,
                 fontweight="bold", ha="left", va="bottom")


def make_data():
    rng = np.random.default_rng(4271)
    n = 240
    t = np.arange(n)
    drift = 0.055 * (t / n) + 0.012 * np.sin(t / 26.0)
    return 10.052 + drift + rng.normal(0, 0.026, n)


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def phi(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def draw_hist(ax, v, mu, s_within, s_overall):
    lo, hi = LSL - 0.038, USL + 0.038
    edges = np.linspace(lo, hi, 37)
    counts, _ = np.histogram(v, bins=edges)
    width = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    cols = [SOFT_RED if (c < LSL or c > USL) else FILL for c in centres]
    ax.bar(centres, counts, width=width * 0.94, color=cols, edgecolor="white",
           linewidth=0.5, zorder=2)

    xg = np.linspace(lo, hi, 500)
    scale = len(v) * width
    ax.plot(xg, scale * normal_pdf(xg, mu, s_overall), color=NAVY, lw=1.5,
            zorder=4)
    ax.plot(xg, scale * normal_pdf(xg, mu, s_within), color=BLUE, lw=1.1,
            ls=(0, (4, 2)), zorder=4)

    for side in (xg < LSL, xg > USL):
        ax.fill_between(xg[side], 0, scale * normal_pdf(xg[side], mu, s_overall),
                        color=ALERT, alpha=0.45, lw=0, zorder=3)

    top = counts.max() * 1.42
    ax.set_ylim(0, top)

    # Label the two fits in place instead of using a legend box.
    xw = mu - 1.85 * s_within
    ax.annotate("within-$\\sigma$ fit", (xw, scale * normal_pdf(xw, mu, s_within)),
                xytext=(-40, 16), textcoords="offset points", fontsize=6.5,
                color=BLUE, ha="center", va="bottom", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=BLUE,
                                shrinkA=1, shrinkB=2))
    xo = mu + 1.95 * s_overall
    ax.annotate("overall-$\\sigma$ fit", (xo, scale * normal_pdf(xo, mu, s_overall)),
                xytext=(40, 20), textcoords="offset points", fontsize=6.5,
                color=NAVY, ha="center", va="bottom", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=NAVY,
                                shrinkA=1, shrinkB=2))
    # Limit flags share one top row; T sits left of its line and the sample
    # mean right of its own, so the two central labels cannot collide.
    for xv, lab, col, ls, ha, off in (
            (LSL, f"LSL {LSL:.3f}", ALERT, "-", "center", 0.0),
            (USL, f"USL {USL:.3f}", ALERT, "-", "center", 0.0),
            (TARGET, f"T {TARGET:.3f}", GREY, (0, (3, 2)), "right", -0.002)):
        ax.axvline(xv, color=col, lw=1.2, ls=ls, zorder=5)
        ax.text(xv + off, top * 0.995, lab, ha=ha, va="top", fontsize=6.6,
                fontweight="bold", color=col, zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col,
                          lw=0.6))
    ax.axvline(mu, color=NAVY, lw=1.0, ls=(0, (1, 1.6)), zorder=5)
    ax.text(mu + 0.002, top * 0.995, f"$\\bar{{x}}$ {mu:.3f}", ha="left",
            va="top", fontsize=6.6, fontweight="bold", color=NAVY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=NAVY, lw=0.6))

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Bore diameter (mm)", fontsize=8)
    ax.set_ylabel("Frequency (count)", fontsize=8)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def draw_cards(ax, items):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.005, "CAPABILITY INDICES", fontsize=7.0, fontweight="bold",
            color=GREY, ha="left", va="bottom")
    h, gap = 0.222, 0.037
    for k, (name, val, note) in enumerate(items):
        y0 = 1.0 - (k + 1) * h - k * gap
        col, verdict = rating(val)
        ax.add_patch(FancyBboxPatch(
            (0.0, y0), 1.0, h, boxstyle="round,pad=0,rounding_size=0.02",
            facecolor=CARD_BG, edgecolor="#dfe4ea", lw=0.7, zorder=1))
        ax.add_patch(Rectangle((0.0, y0), 0.022, h, facecolor=col,
                               edgecolor="none", zorder=2))
        ax.text(0.072, y0 + h * 0.62, name, fontsize=9.0, fontweight="bold",
                color=INK, ha="left", va="center")
        ax.text(0.072, y0 + h * 0.26, note, fontsize=6.0, color=GREY,
                ha="left", va="center")
        ax.text(0.955, y0 + h * 0.64, f"{val:.2f}", fontsize=14.0,
                fontweight="bold", color=col, ha="right", va="center")
        ax.text(0.955, y0 + h * 0.24, verdict, fontsize=6.0, color=col,
                ha="right", va="center")


def draw_trend(ax, v, mu):
    t = np.arange(1, len(v) + 1)
    ax.axhspan(USL, USL + 0.05, color="#fcf3f4", zorder=0)
    ax.axhspan(LSL - 0.05, LSL, color="#fcf3f4", zorder=0)
    ax.plot(t, v, ".", ms=2.0, color=GREY, alpha=0.75, zorder=2)
    k = 15
    roll = np.convolve(v, np.ones(k) / k, mode="valid")
    ax.plot(t[k - 1:], roll, color=NAVY, lw=1.4, zorder=4)
    for y, col in ((USL, ALERT), (LSL, ALERT)):
        ax.axhline(y, color=col, lw=0.9, ls=(0, (5, 2.5)), zorder=3)
    ax.axhline(TARGET, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.set_xlim(0, len(v) + 1)
    ax.set_ylim(LSL - 0.022, USL + 0.022)
    ax.set_xlabel("Measurement order", fontsize=7.5)
    ax.set_ylabel("mm", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_yticks([LSL, TARGET, USL])
    style_axes(ax)
    ax.text(0.015, 0.885, "15-point rolling mean drifting toward USL",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            color=NAVY)


def draw_ppm(ax, ppm_lo, ppm_hi, z_bench):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.02), 1.0, 0.96, boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=CARD_BG, edgecolor="#dfe4ea", lw=0.7, zorder=1))
    total = ppm_lo + ppm_hi
    ax.text(0.5, 0.88, "PREDICTED DEFECT RATE", fontsize=6.6,
            fontweight="bold", color=GREY, ha="center", va="top")
    ax.text(0.5, 0.50, f"{total:,.0f}", fontsize=19, fontweight="bold",
            color=ALERT, ha="center", va="center")
    ax.text(0.5, 0.335, "ppm out of specification", fontsize=6.4, color=GREY,
            ha="center", va="center")
    ax.plot([0.08, 0.92], [0.245, 0.245], color="#dfe4ea", lw=0.8, zorder=2)
    ax.text(0.08, 0.135, f"< LSL  {ppm_lo:,.0f}", fontsize=6.3, color=INK,
            ha="left", va="center")
    ax.text(0.92, 0.135, f"{ppm_hi:,.0f}  > USL", fontsize=6.3, color=ALERT,
            ha="right", va="center")
    ax.text(0.5, 0.735, f"Z-bench = {z_bench:.2f} $\\sigma$", fontsize=7.4,
            fontweight="bold", color=NAVY, ha="center", va="center")


def main():
    set_pub_style()
    v = make_data()
    mu = v.mean()
    s_overall = v.std(ddof=1)
    s_within = np.abs(np.diff(v)).mean() / 1.128      # moving-range estimate

    cp = (USL - LSL) / (6 * s_within)
    cpk = min(USL - mu, mu - LSL) / (3 * s_within)
    pp = (USL - LSL) / (6 * s_overall)
    ppk = min(USL - mu, mu - LSL) / (3 * s_overall)
    ppm_hi = (1 - phi((USL - mu) / s_overall)) * 1e6
    ppm_lo = phi((LSL - mu) / s_overall) * 1e6
    z_bench = 3 * ppk

    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.40],
                          height_ratios=[1.0, 0.46], wspace=0.30, hspace=0.50,
                          left=0.075, right=0.975, top=0.845, bottom=0.125)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 1])

    draw_hist(ax_h, v, mu, s_within, s_overall)
    draw_cards(ax_c, [
        ("$C_p$", cp, "potential \u00b7 within $\\sigma$"),
        ("$C_{pk}$", cpk, "actual \u00b7 within $\\sigma$"),
        ("$P_p$", pp, "potential \u00b7 overall $\\sigma$"),
        ("$P_{pk}$", ppk, "actual \u00b7 overall $\\sigma$"),
    ])
    draw_trend(ax_t, v, mu)
    draw_ppm(ax_p, ppm_lo, ppm_hi, z_bench)

    stamp_labels(fig, [ax_h, ax_c, ax_t, ax_p],
                 ["(a)", "(b)", "(c)", "(d)"])


    savefig(fig, "244_process_capability.png")


if __name__ == "__main__":
    main()
