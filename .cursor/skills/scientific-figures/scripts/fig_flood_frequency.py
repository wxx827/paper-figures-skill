"""Flood frequency analysis on a return-period axis.

Annual maximum discharges are plotted at Gringorten plotting positions against a
logarithmic return-period axis, with a Gumbel fit, its 90 % confidence band and
a Log-Pearson III alternative. Design floods for the 5-, 10-, 50- and 100-year
events are dimensioned on the Gumbel curve. A smaller goodness-of-fit panel on
a grey card compares fitted and observed quantiles for both distributions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"
C_WARM = "#d9a441"

DESIGN_T = (5, 10, 50, 100)


def panel_tag(fig, ax, text, dx=-0.058, dy=0.012, size=10.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def gumbel_q(T, mu, sigma):
    return mu - sigma * np.log(-np.log(1.0 - 1.0 / T))


def lp3_q(T, m, s, g):
    """Log-Pearson III quantile via the Wilson-Hilferty approximation."""
    z = stats.norm.ppf(1.0 - 1.0 / T)
    k = (2.0 / g) * ((1.0 + g * z / 6.0 - g ** 2 / 36.0) ** 3 - 1.0)
    return 10.0 ** (m + k * s)


def main():
    set_pub_style()
    rng = np.random.default_rng(2024)

    n = 48
    am = rng.gumbel(520.0, 168.0, n)
    am[np.argmax(am)] *= 1.16          # one outlier flood of record
    am = np.sort(am)[::-1]
    pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)   # Gringorten
    T_obs = 1.0 / pp

    sigma = am.std(ddof=1) * np.sqrt(6.0) / np.pi
    mu = am.mean() - 0.5772 * sigma
    lg = np.log10(am)
    m_l, s_l = lg.mean(), lg.std(ddof=1)
    g_l = float(stats.skew(lg, bias=False)) or 0.05

    T = np.logspace(np.log10(1.02), np.log10(220), 400)
    q_gum = gumbel_q(T, mu, sigma)
    q_lp3 = lp3_q(T, m_l, s_l, g_l)

    y = -np.log(-np.log(1.0 - 1.0 / T))
    k = (y - 0.5772) * np.sqrt(6.0) / np.pi
    se = sigma * np.sqrt((1.0 + 1.1396 * k + 1.1 * k ** 2) / n)
    lo, hi = q_gum - 1.645 * se, q_gum + 1.645 * se

    fig = plt.figure(figsize=(9.6, 5.6))
    ax = fig.add_axes([0.072, 0.130, 0.545, 0.680])
    ax_s = fig.add_axes([0.770, 0.268, 0.196, 0.392])
    ax.set_zorder(2)
    ax_s.set_zorder(2)
    ax.patch.set_alpha(0.0)
    ax_s.patch.set_alpha(0.0)

    # ---- (a) frequency curve ---------------------------------------------
    ax.fill_between(T, lo, hi, color=C_MAIN, alpha=0.16, lw=0, zorder=2,
                    label="90 % confidence band (Gumbel)")
    ax.plot(T, q_gum, color=C_DEEP, lw=1.9, zorder=5, label="Gumbel (EV1) fit")
    ax.plot(T, q_lp3, color=C_WARM, lw=1.6, ls=(0, (5, 2.2)), zorder=5,
            label="Log-Pearson III fit")
    ax.plot(T_obs, am, ls="none", marker="o", ms=3.8, mfc="white",
            mec=C_MAIN, mew=0.9, zorder=6,
            label=f"Annual maxima, {n} years")

    ax.set_xscale("log")
    ax.set_xlim(1.02, 220)
    ax.set_ylim(150, 1780)
    ax.set_xticks([1.05, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1.05", "2", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("Return period  $T$  (years)")
    ax.set_ylabel("Annual maximum discharge (m$^{3}$ s$^{-1}$)")
    ax.grid(color="#ececec", lw=0.55, zorder=1)
    ax.set_axisbelow(True)
    style_axes(ax)

    # mirrored exceedance-probability axis
    ax_t = ax.twiny()
    ax_t.set_xscale("log")
    ax_t.set_xlim(ax.get_xlim())
    probs = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    ax_t.set_xticks([1.0 / q for q in probs])
    ax_t.set_xticklabels([f"{q:g}" for q in probs])
    ax_t.minorticks_off()
    ax_t.set_xlabel("Annual exceedance probability", fontsize=8.2,
                    labelpad=3.5)
    ax_t.tick_params(length=2.6, labelsize=7.6)
    ax_t.spines["top"].set_color("#333333")
    ax_t.spines["right"].set_visible(False)

    # ---- design floods ----------------------------------------------------
    rows = []
    for i, Tv in enumerate(DESIGN_T):
        qv = gumbel_q(np.array([Tv]), mu, sigma)[0]
        ax.plot([Tv, Tv], [ax.get_ylim()[0], qv], color=C_ACC, lw=0.75,
                ls=(0, (3, 2)), zorder=3)
        ax.plot([Tv], [qv], marker="s", ms=4.6, mfc=C_ACC, mec="white",
                mew=0.9, zorder=7)
        rows.append(f"$Q_{{{Tv}}}$   {qv:>6.0f}")
    ax.text(0.030, 0.955, "Design floods (m$^{3}$ s$^{-1}$)\n" + "\n".join(rows),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color="#333333", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#d5d5d5", linewidth=0.6, alpha=0.95))

    q100 = gumbel_q(np.array([100.0]), mu, sigma)[0]
    ax.annotate("100-year design flood", xy=(97, q100 * 0.99),
                xytext=(88, 1105), fontsize=7.8, color=C_ACC,
                ha="right", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=C_ACC, lw=0.8,
                                connectionstyle="arc3,rad=0.18"))
    ax.annotate("flood of record", xy=(T_obs[0] * 0.94, am[0]),
                xytext=(43, 1665), fontsize=7.4,
                color="#5a5a5a", ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                connectionstyle="arc3,rad=-0.12"))

    handles, labels = ax.get_legend_handles_labels()
    order = [3, 1, 2, 0]
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                    loc="lower right", bbox_to_anchor=(0.995, 0.012),
                    fontsize=7.8, handlelength=1.9, labelspacing=0.45)
    leg.set_clip_on(False)

    # ---- (b) goodness of fit ---------------------------------------------
    box = ax_s.get_position()
    card = FancyBboxPatch(
        (box.x0 - 0.062, box.y0 - 0.082), box.width + 0.082,
        box.height + 0.176, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor="#f2f5f6", edgecolor="#dde3e6", linewidth=0.7, zorder=-5)
    fig.patches.append(card)

    fit_g = gumbel_q(T_obs, mu, sigma)
    fit_l = lp3_q(T_obs, m_l, s_l, g_l)
    lim = (150, 1700)
    ax_s.plot(lim, lim, color="#b0b0b0", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax_s.plot(am, fit_g, ls="none", marker="o", ms=3.4, mfc="white",
              mec=C_DEEP, mew=0.9, zorder=4, label="Gumbel")
    ax_s.plot(am, fit_l, ls="none", marker="^", ms=3.4, mfc="white",
              mec=C_WARM, mew=0.9, zorder=3, label="Log-Pearson III")

    ax_s.set_xlim(lim)
    ax_s.set_ylim(lim)
    ax_s.set_xticks([500, 1000, 1500])
    ax_s.set_yticks([500, 1000, 1500])
    ax_s.tick_params(labelsize=7.0, length=2.4)
    ax_s.set_xlabel("Observed (m$^{3}$ s$^{-1}$)", fontsize=7.4, labelpad=2.0)
    ax_s.set_ylabel("Fitted (m$^{3}$ s$^{-1}$)", fontsize=7.4, labelpad=2.0)
    ax_s.grid(color="#e2e6e8", lw=0.5)
    ax_s.set_axisbelow(True)
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)

    rmse_g = float(np.sqrt(np.mean((fit_g - am) ** 2)))
    rmse_l = float(np.sqrt(np.mean((fit_l - am) ** 2)))
    ax_s.text(0.045, 0.955,
              f"RMSE   Gumbel  {rmse_g:.0f}\n"
              f"              LP3  {rmse_l:.0f}",
              transform=ax_s.transAxes, ha="left", va="top", fontsize=6.9,
              color="#444444", linespacing=1.5)
    leg_s = ax_s.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0),
                        fontsize=6.9, handlelength=1.0, labelspacing=0.3,
                        borderpad=0.2)
    leg_s.set_clip_on(False)

    ax_s.text(0.0, 1.16, "Goodness of fit", transform=ax_s.transAxes,
              fontsize=8.0, fontweight="bold", ha="left", va="bottom",
              color="#1a1a1a")
    ax_s.text(0.0, 1.045, "fitted vs observed quantiles, 1:1 dashed",
              transform=ax_s.transAxes, fontsize=6.9, ha="left", va="bottom",
              color="#6b6b6b")


    panel_tag(fig, ax, "(a)", dy=0.052)
    panel_tag(fig, ax_s, "(b)", dx=-0.062, dy=0.096)
    savefig(fig, "215_flood_frequency.png")


if __name__ == "__main__":
    main()
