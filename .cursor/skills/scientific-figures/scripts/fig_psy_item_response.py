"""Two-parameter IRT: item characteristic curves, item and test information.

Five items are defined once by their discrimination and difficulty, and all
three panels are computed from that single parameter table on one shared
theta grid.  The shared grid matters here more than usual: an item
information curve read against a different theta axis than its own ICC is
worse than no panel at all, so the axis limits are set from one variable and
the aux panels reuse the main panel's ticks.  The precision window quoted in
the footnote is measured off the test information curve that is drawn.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_palette, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
ACCENT = "#b23a48"

# Discrimination a_i and difficulty b_i of the five items drawn in full.
A_PAR = np.array([1.65, 1.20, 0.85, 1.40, 0.95])
B_PAR = np.array([-1.60, -0.70, 0.10, 0.85, 1.75])
N_FORM = 40                # the five above are exemplars from this form
THETA_LO, THETA_HI = -3.5, 3.5
SE_TARGET = 0.316          # SE = 0.316 logits <=> marginal reliability 0.90


def icc(theta, a, b):
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def form(seed=597):
    """The whole test: five exemplar items plus the rest of the blueprint."""
    rng = np.random.default_rng(seed)
    n_fill = N_FORM - A_PAR.size
    b = np.linspace(-2.4, 2.4, n_fill) + rng.normal(0.0, 0.11, n_fill)
    a = rng.uniform(1.00, 1.90, n_fill)
    return (np.concatenate([A_PAR, a]), np.concatenate([B_PAR, b]))


def main():
    set_pub_style()
    theta = np.linspace(THETA_LO, THETA_HI, 601)
    probs = np.array([icc(theta, a, b) for a, b in zip(A_PAR, B_PAR)])
    info = np.array([a ** 2 * p * (1.0 - p)
                     for a, p in zip(A_PAR, probs)])
    a_all, b_all = form()
    p_all = np.array([icc(theta, a, b) for a, b in zip(a_all, b_all)])
    test_info = (a_all[:, None] ** 2 * p_all * (1.0 - p_all)).sum(axis=0)
    se = 1.0 / np.sqrt(test_info)

    # Precision window measured off the curve that is actually plotted.
    ok = se <= SE_TARGET
    win = (theta[ok][0], theta[ok][-1]) if ok.any() else (np.nan, np.nan)
    peak = float(theta[np.argmax(test_info)])

    # Dark blue = hardest item: one hue ordered by difficulty.
    ramp = get_palette("cb_sequential_blue", 7)
    cols = [ramp[i] for i in (2, 3, 4, 5, 6)]
    order = np.argsort(B_PAR)
    item_col = {int(k): cols[i] for i, k in enumerate(order)}

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.24,
                          hspace=0.50)
    ax_icc = fig.add_subplot(gs[:, 0])
    ax_iif = fig.add_subplot(gs[0, 1])
    ax_tif = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.062, right=0.955, bottom=0.155, top=0.862)

    # ---- main: item characteristic curves --------------------------------
    for i in range(A_PAR.size):
        c = item_col[i]
        ax_icc.plot(theta, probs[i], color=c, lw=1.8, zorder=4,
                    label=(f"item {i + 1}   $a_i$ = {A_PAR[i]:.2f},  "
                           f"$b_i$ = {B_PAR[i]:+.2f}"))
        ax_icc.plot([B_PAR[i], B_PAR[i]], [0.0, 0.5], color=c, lw=0.8,
                    ls=(0, (2.2, 1.8)), alpha=0.8, zorder=2)
        ax_icc.plot([B_PAR[i]], [0.5], marker="o", ms=4.4, mfc="white",
                    mec=c, mew=1.2, zorder=5)

    ax_icc.axhline(0.5, color=GREY, lw=0.9, ls=(0, (4.0, 2.2)), zorder=1)
    ax_icc.text(THETA_LO + 0.08, 0.52,
                r"$P(\theta)=0.5$ at $\theta=b_i$", ha="left", va="bottom",
                fontsize=6.9, color=SUB)
    ax_icc.set_xlim(THETA_LO, THETA_HI)
    ax_icc.set_ylim(0.0, 1.02)
    ax_icc.set_xticks(np.arange(-3, 4))
    ax_icc.set_xlabel(r"latent ability $\theta$ (logits)", fontsize=8.0)
    ax_icc.set_ylabel(r"probability of a correct response $P(\theta)$ (-)",
                      fontsize=8.0)
    ax_icc.tick_params(labelsize=7.2)
    ax_icc.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_icc.set_title("Item characteristic curves: $b_i$ shifts the curve, "
                     "$a_i$ steepens it", fontsize=9.0, fontweight="bold",
                     color=INK, pad=11.0)
    ax_icc.legend(loc="upper left", frameon=False, fontsize=6.9,
                  handlelength=1.6, labelspacing=0.34, borderaxespad=0.6)

    # Difficulties go on a secondary top axis: under the curves they would
    # land on the x-axis label, and beside the markers they collide with the
    # neighbouring curve.
    ax_b = ax_icc.secondary_xaxis("top")
    ax_b.set_xticks(B_PAR[order])
    ax_b.set_xticklabels([f"$b_{{{int(k) + 1}}}$" for k in order],
                         fontsize=6.8)
    ax_b.tick_params(length=2.4, width=0.8)
    for lab, k in zip(ax_b.get_xticklabels(), order):
        lab.set_color(item_col[int(k)])
    ax_b.spines["top"].set_color("#8a8a8a")

    # ---- aux 1: item information -----------------------------------------
    for i in range(A_PAR.size):
        ax_iif.plot(theta, info[i], color=item_col[i], lw=1.4)
        k = int(np.argmax(info[i]))
        ax_iif.plot([theta[k]], [info[i][k]], marker="o", ms=3.0,
                    mfc="white", mec=item_col[i], mew=0.9)
    ax_iif.set_xlim(THETA_LO, THETA_HI)
    ax_iif.set_ylim(0.0, info.max() * 1.16)
    ax_iif.set_xticks(np.arange(-3, 4))
    ax_iif.set_xlabel(r"$\theta$ (logits)", fontsize=7.4)
    ax_iif.set_ylabel(r"item information $I_i(\theta)$ (-)", fontsize=7.4)
    ax_iif.tick_params(labelsize=6.6)
    ax_iif.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_iif.set_title(r"Each item informs near its own $b_i$", fontsize=8.5,
                     fontweight="bold", color=INK)
    ax_iif.text(0.98, 0.94, r"peak height $\propto a_i^{2}$",
                transform=ax_iif.transAxes, ha="right", va="top",
                fontsize=6.4, color=MUTED)

    # ---- aux 2: test information and standard error ----------------------
    ax_tif.plot(theta, test_info, color=INK, lw=1.6, zorder=4,
                label=r"$I(\theta)$")
    ax_tif.fill_between(theta, 0.0, test_info, color="#cfe1ee", alpha=0.55,
                        linewidth=0.0, zorder=2)
    ax_tif.axvspan(win[0], win[1], color=ACCENT, alpha=0.10, zorder=1,
                   linewidth=0.0)
    ax_tif.set_xlim(THETA_LO, THETA_HI)
    ax_tif.set_ylim(0.0, test_info.max() * 1.28)
    ax_tif.set_xticks(np.arange(-3, 4))
    ax_tif.set_xlabel(r"$\theta$ (logits)", fontsize=7.4)
    ax_tif.set_ylabel(r"test information $I(\theta)$ (-)", fontsize=7.4)
    ax_tif.tick_params(labelsize=6.6)

    ax_se = ax_tif.twinx()
    ax_se.plot(theta, se, color=ACCENT, lw=1.4, ls=(0, (3.6, 1.8)), zorder=5,
               label=r"SE$(\theta)$")
    ax_se.axhline(SE_TARGET, color=GREY, lw=0.8, ls=(0, (1.8, 1.8)))
    ax_se.set_ylim(0.0, float(se.max()) * 1.30)
    ax_se.set_ylabel(r"SE$(\theta)$ (logits)", fontsize=7.4, color=ACCENT)
    ax_se.tick_params(axis="y", labelsize=6.6, colors=ACCENT)
    ax_se.spines["top"].set_visible(False)
    ax_se.spines["right"].set_color(ACCENT)

    ax_tif.set_title("Test information and where it is precise",
                     fontsize=8.5, fontweight="bold", color=INK)
    ax_tif.text(np.mean(win), test_info.max() * 1.13,
                f"SE $\\leq$ {SE_TARGET:.3f} over "
                f"$\\theta \\in$ [{win[0]:+.2f}, {win[1]:+.2f}]",
                ha="center", va="center", fontsize=6.5, color=ACCENT)
    h1, l1 = ax_tif.get_legend_handles_labels()
    h2, l2 = ax_se.get_legend_handles_labels()
    ax_tif.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False,
                  fontsize=6.4, handlelength=1.5, labelspacing=0.28,
                  borderaxespad=0.3)


    note = (f"Two-parameter logistic model "
            f"$P(\\theta)=[1+e^{{-a_i(\\theta-b_i)}}]^{{-1}}$. Panels (a) "
            f"and (b) show five exemplar items with $a_i$ = "
            + " / ".join(f"{v:.2f}" for v in A_PAR) +
            " and $b_i$ = " + " / ".join(f"{v:+.2f}" for v in B_PAR) +
            f" logits; panel (c) sums over the whole {N_FORM}-item form. "
            f"Test information peaks at $\\theta$ = {peak:+.2f} "
            f"logits ($I$ = {test_info.max():.2f}); the shaded window is "
            f"where SE$(\\theta)=I(\\theta)^{{-1/2}}$ stays at or below "
            f"{SE_TARGET:.3f} logits, i.e. marginal reliability "
            f"{1.0 - SE_TARGET ** 2:.2f}.")

    panel_labels_inches(fig, [ax_icc, ax_iif, ax_tif],
                        labels=["(a)", "(b)", "(c)"], dx=-0.44, dy=0.10)
    savefig(fig, "597_psy_item_response.png")


if __name__ == "__main__":
    main()
