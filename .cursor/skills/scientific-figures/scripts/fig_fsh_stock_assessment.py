"""Stock assessment told through a Kobe plot driven by a Schaefer model.

One surplus-production model generates everything on the page, so the three
panels cannot disagree.  Biomass is stepped forward with
B(t+1) = B(t) + r B (1 - B/K) - F B, which fixes B_MSY = K/2, F_MSY = r/2 and
MSY = rK/4; the Kobe axes are then just B/B_MSY and F/F_MSY of that same run.
The terminal point carries a 95% confidence ellipse whose axes come from the
eigen-decomposition of an assessment cloud drawn around it, not from a
hand-picked radius, and it sits exactly on the last value of the time series
in panel (b) because both read the same array.

Panel (c) plots the production parabola and the *realised* surplus,
B(t+1) - B(t) + C(t), recovered from the simulated trajectory; MSY and B_MSY
are read off with argmax of the plotted curve rather than from the analytic
formula, which is the check that the curve and the caption agree.

Quadrant shading avoids a red/green pairing: healthy is house teal, the two
single-breach quadrants are gold and ochre, and only the doubly-failing
quadrant is red - and that one is hatched as well, so the ranking survives a
greyscale print.  Illustrative numbers; this is a layout reference.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from pubstyle import (set_pub_style, savefig, get_cmap, lighten, darken,
                      annotate_box)
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY = "#808080"
TEAL, GOLD, OCHRE, RED = "#2a7f7a", "#c9a227", "#c0603f", "#b23a48"
BIO, PRESS = "#3b5b92", "#b23a48"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.04, 4.86, 4.16)
AUX1 = (6.62, 3.14, 2.78, 2.06)
AUX2 = (6.62, 1.04, 2.78, 1.44)

K_CAP = 900.0            # carrying capacity, kt
R_INT = 0.38             # intrinsic growth rate, per year
YEAR0, YEAR1 = 1970, 2024
XMAX, YMAX = 2.30, 2.20


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_stock(rng):
    """Step a Schaefer biomass forward under a rise-and-fall in fishing."""
    years = np.arange(YEAR0, YEAR1 + 1)
    t = (years - years[0]) / (years[-1] - years[0])
    f_mult = (0.42 + 1.72 * logistic((t - 0.33) / 0.11)
              - 1.42 * logistic((t - 0.70) / 0.10))
    f_mult = f_mult * np.exp(rng.normal(0.0, 0.045, f_mult.size))
    f_msy = R_INT / 2.0
    f = f_mult * f_msy

    b = np.empty(years.size)
    b[0] = 0.82 * K_CAP
    for i in range(years.size - 1):
        growth = R_INT * b[i] * (1.0 - b[i] / K_CAP)
        nxt = b[i] + growth - f[i] * b[i]
        b[i + 1] = max(nxt * float(np.exp(rng.normal(0.0, 0.022))), 20.0)
    catch = f * b
    surplus = np.diff(b) + catch[:-1]
    return dict(year=years, biomass=b, f=f, catch=catch, surplus=surplus,
                b_msy=K_CAP / 2.0, f_msy=f_msy,
                b_rel=b / (K_CAP / 2.0), f_rel=f / f_msy)


def confidence_ellipse(rng, st, n=6000):
    """95% ellipse from the covariance of an assessment cloud (chi2, 2 d.f.)."""
    mu = np.array([st["b_rel"][-1], st["f_rel"][-1]])
    cov = np.array([[0.0180, 0.0042], [0.0042, 0.0091]])
    cloud = rng.multivariate_normal(mu, cov, size=n)
    emp = np.cov(cloud.T)
    vals, vecs = np.linalg.eigh(emp)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w, h = 2.0 * np.sqrt(5.991 * vals)
    d = cloud - mu
    inside = float(np.mean(np.einsum("ij,jk,ik->i", d,
                                     np.linalg.inv(emp), d) <= 5.991))
    return mu, w, h, ang, inside


def draw_kobe(ax, st, ell):
    quads = [((1.0, 0.0), (XMAX - 1.0, 1.0), TEAL, None,
              "healthy\nB > B$_{MSY}$,  F < F$_{MSY}$", (0.975, 0.035),
              "right", "bottom"),
             ((1.0, 1.0), (XMAX - 1.0, YMAX - 1.0), GOLD, None,
              "overfishing\n(pressure too high)", (0.975, 0.965),
              "right", "top"),
             ((0.0, 0.0), (1.0, 1.0), OCHRE, None,
              "overfished\n(rebuilding)", (0.030, 0.035), "left", "bottom"),
             ((0.0, 1.0), (1.0, YMAX - 1.0), RED, "////",
              "overfished and\noverfishing", (0.030, 0.965), "left", "top")]
    for (x0, y0), (w, h), col, hatch, name, loc, ha, va in quads:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=lighten(col, 0.84),
                               edgecolor="none", zorder=0))
        if hatch:
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none",
                                   edgecolor=lighten(col, 0.55), hatch=hatch,
                                   linewidth=0.0, zorder=1))
        ax.text(loc[0], loc[1], name, transform=ax.transAxes, ha=ha, va=va,
                fontsize=6.8, color=darken(col, 0.26), linespacing=1.32,
                zorder=4)
    ax.axvline(1.0, color=GREY, lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)
    ax.axhline(1.0, color=GREY, lw=0.9, ls=(0, (4.0, 2.4)), zorder=2)

    x, y = st["b_rel"], st["f_rel"]
    ax.plot(x, y, color="#4a4a4a", lw=1.05, alpha=0.9, zorder=5)
    cmap = get_cmap("cb_sequential_blue")
    ax.scatter(x, y, s=13.0, c=cmap(np.linspace(0.24, 1.0, x.size)),
               edgecolors="white", linewidths=0.32, zorder=6)

    mu, ew, eh, ang, _ = ell
    ax.add_patch(Ellipse(mu, ew, eh, angle=ang, facecolor=lighten(BIO, 0.72),
                         edgecolor=darken(BIO, 0.15), lw=0.9, alpha=0.55,
                         zorder=7))
    ax.plot([mu[0]], [mu[1]], marker="o", ms=8.0, color=darken(BIO, 0.25),
            markeredgecolor="white", markeredgewidth=0.9, zorder=9)

    # Year callouts placed along the local outward normal, so the label leans
    # away from the track instead of landing on top of the next segment.
    marks = [0, 10, 20, 30, 40, x.size - 1]
    for k in marks:
        i0, i1 = max(k - 2, 0), min(k + 2, x.size - 1)
        tx, ty = x[i1] - x[i0], y[i1] - y[i0]
        n = np.hypot(tx, ty) or 1.0
        nx, ny = -ty / n, tx / n
        if k == x.size - 1:
            nx, ny = 0.72, -0.70
        off = (nx * 22.0, ny * 22.0)
        ax.annotate(str(int(st["year"][k])), xy=(x[k], y[k]), xytext=off,
                    textcoords="offset points", fontsize=6.6, color=INK,
                    ha="center", va="center", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.80),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=RULE,
                                    shrinkA=0.0, shrinkB=1.5))

    annotate_box(ax, "\n".join([
        "terminal year " + str(int(st["year"][-1])),
        "B/B$_{MSY}$ = " + "{:.2f}".format(x[-1]),
        "F/F$_{MSY}$ = " + "{:.2f}".format(y[-1]),
        "95% ellipse, 2 d.f."]),
        loc="upper right", fontsize=6.6, margin=0.026)

    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(0.0, YMAX)
    ax.set_xlabel("stock status  B / B$_{MSY}$ (-)", fontsize=8.0)
    ax.set_ylabel("fishing pressure  F / F$_{MSY}$ (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Kobe phase plane, {:d}-{:d} trajectory"
                 .format(YEAR0, YEAR1), fontsize=8.5, fontweight="bold")


def draw_series(ax, st):
    yr = st["year"]
    ax.plot(yr, st["biomass"], color=BIO, lw=1.6, zorder=6)
    ax.axhline(st["b_msy"], color=BIO, lw=0.9, ls=(0, (3.0, 2.2)), alpha=0.8,
               zorder=4)
    ax.text(yr[0] + 0.6, st["b_msy"] + 16.0,
            "B$_{MSY}$ = " + "{:.0f} kt".format(st["b_msy"]), fontsize=6.4,
            color=BIO, ha="left", va="bottom", zorder=7)
    ax.set_ylim(0.0, K_CAP * 1.10)
    ax.set_xlim(yr[0] - 1.0, yr[-1] + 1.0)
    ax.set_ylabel("biomass (kt)", fontsize=7.8, color=BIO)
    ax.tick_params(labelsize=6.8)
    ax.tick_params(axis="y", colors=BIO)
    ax.spines["left"].set_color(BIO)

    axf = ax.twinx()
    axf.plot(yr, st["f"], color=PRESS, lw=1.4, zorder=6)
    axf.axhline(st["f_msy"], color=PRESS, lw=0.9, ls=(0, (3.0, 2.2)),
                alpha=0.8, zorder=4)
    axf.text(yr[-1] - 0.6, st["f_msy"] - 0.014,
             "F$_{MSY}$ = " + "{:.2f} /yr".format(st["f_msy"]), fontsize=6.4,
             color=PRESS, ha="right", va="top", zorder=7)
    axf.set_ylim(0.0, st["f"].max() * 1.42)
    axf.set_ylabel("fishing mortality F (1/yr)", fontsize=7.8, color=PRESS)
    axf.tick_params(labelsize=6.8, colors=PRESS)
    axf.spines["top"].set_visible(False)
    axf.spines["right"].set_visible(True)
    axf.spines["right"].set_color(PRESS)
    axf.spines["right"].set_linewidth(0.8)
    ax.set_title("Biomass and fishing mortality vs reference points",
                 fontsize=8.5, fontweight="bold")


def draw_production(ax, st):
    grid = np.linspace(0.0, K_CAP, 400)
    prod = R_INT * grid * (1.0 - grid / K_CAP)
    k = int(np.argmax(prod))
    msy, b_at_msy = float(prod[k]), float(grid[k])

    ax.fill_between(grid, 0.0, prod, color=lighten(TEAL, 0.86), zorder=1)
    ax.plot(grid, prod, color=darken(TEAL, 0.12), lw=1.5, zorder=4)
    ax.scatter(st["biomass"][:-1], st["surplus"], s=8.0,
               facecolors=lighten(BIO, 0.35), edgecolors="white",
               linewidths=0.25, alpha=0.9, zorder=5)
    ax.plot([b_at_msy], [msy], marker="o", ms=6.5, color=RED,
            markeredgecolor="white", markeredgewidth=0.8, zorder=7)
    ax.vlines(b_at_msy, 0.0, msy, color=RED, lw=0.8, ls=(0, (2.6, 2.0)),
              zorder=6)
    ax.annotate("MSY = {:.0f} kt/yr at B = {:.0f} kt".format(msy, b_at_msy),
                xy=(b_at_msy, msy), xytext=(0, 20),
                textcoords="offset points", fontsize=6.5, color=RED,
                ha="center", va="bottom", zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=RED,
                                shrinkA=0.0, shrinkB=2.0))
    ax.plot([st["biomass"][-1]], [0.0], marker="^", ms=5.0, color=BIO,
            clip_on=False, zorder=8)
    ax.text(st["biomass"][-1], msy * 0.10,
            "{:d} B\n{:.0f} kt".format(int(st["year"][-1]),
                                       st["biomass"][-1]),
            fontsize=6.3, color=BIO, ha="center", va="bottom",
            linespacing=1.25, zorder=8)
    ax.set_xlim(0.0, K_CAP)
    ax.set_ylim(0.0, msy * 1.36)
    ax.set_xlabel("biomass (kt)", fontsize=7.8)
    ax.set_ylabel("surplus production (kt/yr)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Surplus production curve", fontsize=8.5, fontweight="bold")
    return msy, b_at_msy


def main():
    set_pub_style()
    rng = np.random.default_rng(9000)
    st = build_stock(rng)
    ell = confidence_ellipse(rng, st)

    fig = plt.figure(figsize=(FW, FH))
    ax_k = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX1)
    ax_p = axes_inches(fig, AUX2)

    draw_kobe(ax_k, st, ell)
    draw_series(ax_s, st)
    msy, b_at_msy = draw_production(ax_p, st)

    panel_labels_inches(fig, [ax_k, ax_s, ax_p], dx=-0.46, dy=0.20)

    over_f = float(np.mean(st["f_rel"] > 1.0))
    over_b = float(np.mean(st["b_rel"] < 1.0))
    k_pk = int(np.argmax(st["f_rel"]))
    k_lo = int(np.argmin(st["b_rel"]))

    print("terminal b_rel={:.3f} f_rel={:.3f} msy={:.1f} b_at_msy={:.0f} "
          "cover={:.3f}".format(st["b_rel"][-1], st["f_rel"][-1], msy,
                                b_at_msy, ell[4]))
    savefig(fig, "900_fsh_stock_assessment.png")


if __name__ == "__main__":
    main()
