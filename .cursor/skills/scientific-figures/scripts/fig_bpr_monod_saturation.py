"""Monod saturation kinetics and the double-reciprocal linearisation.

The main panel is one hyperbola, mu = mu_max S/(Ks+S), drawn together with the
Haldane form that adds a substrate-inhibition term: both share mu_max and Ks,
so the gap between them is the inhibition penalty and nothing else.  The
mu_max asymptote and the half-saturation point are not typed in - the
asymptote is the model constant and the marked point is evaluated at S = Ks,
which is why it lands exactly on mu_max/2.  The Lineweaver-Burk panel takes
the double reciprocal of the same model samples and regresses them, so the
mu_max recovered from the intercept and the Ks recovered from the slope come
back as the constants the main curve was drawn with; the inhibited strain
leaves the straight line, which is the classical diagnostic.  The bottom panel
puts the reference strain next to three others on the rate / affinity plane.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

GREEN = "#2e7d5b"        # biomass / growth rate
BLUE = "#3b5b92"         # substrate
RED = "#b23a48"          # inhibited strain
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

MU_MAX = 0.62            # h^-1
KS = 1.80                # g/L
KI = 12.0                # g/L, substrate inhibition constant
S_MAX = 30.0             # g/L

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 0.95, 4.62, 4.55)
TOP = (7.82, 3.61, 2.68, 1.89)
BOT = (7.82, 0.95, 2.68, 1.89)

SAMPLES = np.array([0.45, 0.75, 1.20, 2.00, 3.40, 6.00, 12.00, 24.00])

STRAINS = [("A ref", MU_MAX, KS), ("B", 0.48, 0.65),
           ("C", 0.71, 4.20), ("D", 0.36, 0.22)]


def monod(s):
    return MU_MAX * s / (KS + s)


def haldane(s):
    return MU_MAX * s / (KS + s + s * s / KI)


def main_panel(ax, rng):
    s = np.linspace(0.0, S_MAX, 1400)
    mu_m, mu_h = monod(s), haldane(s)

    ax.axvspan(0.0, KS, facecolor=lighten(BLUE, 0.93), edgecolor="none",
               zorder=0)
    ax.axvspan(10.0 * KS, S_MAX, facecolor=lighten(GREEN, 0.93),
               edgecolor="none", zorder=0)
    ax.text(KS / 2.0, 0.030, "first order\nin S", ha="center", va="bottom",
            fontsize=6.6, color=SUB, linespacing=1.30)
    ax.text((10.0 * KS + S_MAX) / 2.0, 0.030, "zero order,\nS saturating",
            ha="center", va="bottom", fontsize=6.6, color=SUB,
            linespacing=1.30)

    ax.axhline(MU_MAX, color=INK, lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    ax.text(S_MAX - 0.4, MU_MAX + 0.007,
            r"$\mu_{max}$ = " + "{:.2f}".format(MU_MAX) +
            r" h$^{-1}$   (horizontal asymptote)", ha="right", va="bottom",
            fontsize=7.0, color=INK)

    ax.plot(s, mu_m, color=GREEN, lw=2.0, zorder=6)
    ax.plot(s, mu_h, color=RED, lw=1.8, ls=(0, (5.0, 2.2)), zorder=6)

    obs_m = monod(SAMPLES) * (1.0 + rng.normal(0.0, 0.028, SAMPLES.size))
    obs_h = haldane(SAMPLES) * (1.0 + rng.normal(0.0, 0.034, SAMPLES.size))
    ax.plot(SAMPLES, obs_m, ls="none", marker="o", ms=4.2, mfc="white",
            mec=GREEN, mew=1.0, zorder=7)
    ax.plot(SAMPLES, obs_h, ls="none", marker="s", ms=3.8, mfc="white",
            mec=RED, mew=1.0, zorder=7)

    half = MU_MAX / 2.0
    ax.plot([0.0, KS, KS], [half, half, 0.0], color=BLUE, lw=0.9,
            ls=(0, (2.6, 2.0)), zorder=4)
    ax.plot([KS], [half], marker="o", ms=5.2, color=BLUE, zorder=8)
    ax.annotate(
        r"$K_s$ = " + "{:.2f} g/L".format(KS) + "\nhalf-saturation, "
        r"$\mu$ = $\mu_{max}$/2 = " + "{:.3f}".format(half) + r" h$^{-1}$",
        xy=(KS, half), xytext=(KS + 3.4, half - 0.115), ha="left", va="top",
        fontsize=7.0, color=BLUE, linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))

    i_pk = int(np.argmax(mu_h))
    s_pk, mu_pk = float(s[i_pk]), float(mu_h[i_pk])
    ax.plot([s_pk], [mu_pk], marker="^", ms=5.2, color=RED, zorder=8)
    ax.annotate(
        "inhibition optimum\n{:.2f} g/L,  ".format(s_pk) +
        r"$\mu$ = " + "{:.3f}".format(mu_pk) + r" h$^{-1}$" +
        "\n({:.0f} % of ".format(100.0 * mu_pk / MU_MAX) +
        r"$\mu_{max}$)",
        xy=(s_pk, mu_pk), xytext=(s_pk + 4.6, mu_pk + 0.085), ha="left",
        va="bottom", fontsize=7.0, color=RED, linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    ax.set_xlim(0.0, S_MAX)
    ax.set_ylim(0.0, MU_MAX * 1.20)
    ax.set_xlabel("residual substrate S (g/L)")
    ax.set_ylabel(r"specific growth rate $\mu$ (h$^{-1}$)")
    ax.grid(axis="y", lw=0.5, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([], [], color=GREEN, lw=2.0, label="Monod, no inhibition"),
        Line2D([], [], color=RED, lw=1.8, ls=(0, (5.0, 2.2)),
               label=r"Haldane, $K_i$ = " + "{:.0f} g/L".format(KI)),
        Line2D([], [], color=GREEN, lw=0.0, marker="o", ms=4.2, mfc="white",
               mec=GREEN, label="assay points, 2.8 % noise"),
        Line2D([], [], color=RED, lw=0.0, marker="s", ms=3.8, mfc="white",
               mec=RED, label="assay points, inhibited strain")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False,
              bbox_to_anchor=(0.020, 0.812), handlelength=1.6,
              borderaxespad=0.0, labelspacing=0.32)
    ax.set_title("One hyperbola fixes both constants; inhibition bends it "
                 "back down", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)
    return s_pk, mu_pk


def lb_panel(ax):
    inv_s = 1.0 / SAMPLES
    inv_mu = 1.0 / monod(SAMPLES)
    slope, icept = np.polyfit(inv_s, inv_mu, 1)
    mu_fit = 1.0 / icept
    ks_fit = slope * mu_fit
    resid = inv_mu - (slope * inv_s + icept)
    r2 = 1.0 - resid.var() / inv_mu.var()

    x_lo = -1.0 / KS - 0.32
    xs = np.linspace(x_lo, inv_s.max() * 1.10, 120)
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=1)
    ax.axvline(0.0, color=GREY, lw=0.7, zorder=1)
    ax.plot(xs, slope * xs + icept, color=GREEN, lw=1.5, zorder=4)
    ax.plot(inv_s, inv_mu, ls="none", marker="o", ms=3.9, mfc="white",
            mec=GREEN, mew=1.0, zorder=6)

    inv_h = 1.0 / haldane(SAMPLES)
    ax.plot(inv_s, inv_h, color=RED, lw=1.0, ls=(0, (3.0, 1.8)), marker="s",
            ms=3.2, mfc="white", mec=RED, mew=0.9, zorder=5)
    j = int(np.argmin(np.abs(inv_s - 1.0 / 6.0)))
    ax.annotate("inhibited strain\nleaves the line",
                xy=(inv_s[j], inv_h[j]), xytext=(0.34, inv_h[0] * 0.80),
                ha="left", va="bottom", fontsize=6.3, color=RED,
                linespacing=1.30,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    ax.plot([0.0], [icept], marker="o", ms=4.4, color=INK, zorder=7)
    ax.plot([-1.0 / KS], [0.0], marker="o", ms=4.4, color=BLUE, zorder=7)
    ax.text(0.10, icept + 0.28, r"intercept 1/$\mu_{max}$" +
            " = {:.3f}".format(icept), ha="left", va="bottom", fontsize=6.3,
            color=INK)
    ax.text(-1.0 / KS - 0.05, 0.42, r"$-1/K_s$" + " = {:.3f}".format(
        -1.0 / KS), ha="right", va="bottom", fontsize=6.3, color=BLUE)

    ax.text(0.985, 0.965,
            r"1/$\mu$ = " + "{:.3f}".format(slope) + " (1/S) + " +
            "{:.3f}".format(icept) + "\n" +
            r"$\mu_{max}$ = 1/intercept = " + "{:.3f}".format(mu_fit) +
            r" h$^{-1}$" + "\n" + r"$K_s$ = slope $\times$ $\mu_{max}$ = " +
            "{:.2f} g/L".format(ks_fit) + "\n" +
            r"$R^2$ = " + "{:.4f}".format(r2) + " (matches main panel)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=INK, linespacing=1.42)

    ax.set_xlim(x_lo, inv_s.max() * 1.10)
    ax.set_ylim(-0.55, max(inv_mu.max(), inv_h.max()) * 1.55)
    ax.set_xlabel("1 / S (L/g)", fontsize=7.6)
    ax.set_ylabel(r"1 / $\mu$ (h)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.5, alpha=0.55)
    ax.set_title("Lineweaver-Burk returns the same constants",
                 fontsize=8.4, fontweight="bold", color=INK)
    return slope, icept, mu_fit, ks_fit, r2


def strain_panel(ax):
    names = [n for n, _m, _k in STRAINS]
    mus = np.array([m for _n, m, _k in STRAINS])
    kss = np.array([k for _n, _m, k in STRAINS])
    pos = np.arange(len(STRAINS), dtype=float)
    wid = 0.36

    ax_k = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_k.set_zorder(1)

    ax.bar(pos - wid / 2, mus, width=wid, color=GREEN, edgecolor="white",
           linewidth=0.7, zorder=3)
    ax_k.bar(pos + wid / 2, kss, width=wid, color="white", edgecolor=BLUE,
             linewidth=1.0, hatch="////", zorder=3)
    for px, v in zip(pos - wid / 2, mus):
        ax.text(px, v + mus.max() * 0.030, "{:.2f}".format(v), ha="center",
                va="bottom", fontsize=6.2, color=GREEN)
    for px, v in zip(pos + wid / 2, kss):
        ax_k.text(px, v + kss.max() * 0.085, "{:.2f}".format(v), ha="center",
                  va="bottom", fontsize=6.2, color=BLUE)

    ax.set_xticks(pos)
    ax.set_xticklabels(names, fontsize=7.0)
    ax.set_xlim(-0.62, len(STRAINS) - 0.38)
    ax.set_ylim(0.0, mus.max() * 1.40)
    ax_k.set_ylim(0.0, kss.max() * 1.40)
    ax.set_xlabel("production strain", fontsize=7.6)
    ax.set_ylabel(r"$\mu_{max}$ (h$^{-1}$)", fontsize=7.6, color=GREEN)
    ax_k.set_ylabel(r"$K_s$ (g/L)", fontsize=7.6, color=BLUE)
    ax.tick_params(axis="y", colors=GREEN, labelsize=6.8)
    ax_k.tick_params(axis="y", colors=BLUE, labelsize=6.8)
    ax.spines["left"].set_color(GREEN)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_k.spines["right"].set_color(BLUE)
    ax_k.spines["top"].set_visible(False)
    ax_k.spines["left"].set_visible(False)

    ax.text(-wid / 2, mus[0] * 0.50, "main\npanel", ha="center", va="center",
            fontsize=6.2, color="white", linespacing=1.25)
    best = int(np.argmax(mus / kss))
    ax.text(0.5, 0.985, "strain {:s} has the best ".format(names[best]) +
            r"$\mu_{max}/K_s$ = " + "{:.2f} L/(g h)".format(
                (mus / kss)[best]), transform=ax.transAxes, ha="center",
            va="top", fontsize=6.5, color=INK)
    ax.set_title("Rate against affinity across four strains", fontsize=8.4,
                 fontweight="bold", color=INK)
    return float((mus / kss)[best]), names[best]


def main():
    set_pub_style()
    rng = np.random.default_rng(878)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_lb = axes_inches(fig, TOP)
    ax_str = axes_inches(fig, BOT)

    s_pk, mu_pk = main_panel(ax_main, rng)
    slope, icept, mu_fit, ks_fit, r2 = lb_panel(ax_lb)
    ratio, best = strain_panel(ax_str)

    panel_labels_inches(fig, [ax_main, ax_lb, ax_str], dx=-0.62, dy=0.14)
    savefig(fig, "878_bpr_monod_kinetics.png")


if __name__ == "__main__":
    main()
