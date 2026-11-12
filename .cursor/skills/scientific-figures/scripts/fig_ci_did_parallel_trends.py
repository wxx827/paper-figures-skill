"""Figure 768 - Difference-in-differences read off a parallel-trends plot.

Main panel: treated and control group means of a log employment index over
fifteen years with 95 % bands, the policy year marked and the post-policy
window shaded.  The dashed counterfactual carries the treated pre-policy
level forward along the control trajectory, so the vertical double arrow at
the right edge *is* the difference-in-differences estimate rather than a
decoration placed near it.

Auxiliary (both at 58 % of the main panel width): the canonical 2x2 table of
group-by-period means, and a forest of the same estimand under six
specifications.  Every quoted number - the four cell means, the DiD point
estimate and interval, the t statistic, the pre-trend slope and its p value -
is computed from the simulated unit-level panel that the curves are drawn
from, so changing the seed changes the captions.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, inch_axes, panel_label_inches, text_card

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID, BASE = "#cccccc", "#d9d9d9", "#808080"
CARD = "#f5f5f5"

TREAT = "#b23a48"        # house red: the treated group, the single emphasis
CTRL = "#3b5b92"         # house blue: the comparison group
COUNTER = "#8a8a8a"      # grey: the counterfactual, deliberately weakest

YEARS = np.arange(2009, 2024)
T0 = 2018                # first treated period
N_T, N_C = 64, 96        # units per arm
TAU = 0.058              # true saturated effect, in log units
SHOCK = 2020             # common shock year dropped by one specification


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ar1(rng, n, phi=0.55, sd=0.0052):
    """Stationary AR(1) shocks: visible wobble, no accumulated drift."""
    e = rng.normal(0.0, sd, n)
    out = np.empty(n)
    out[0] = e[0] / np.sqrt(1.0 - phi ** 2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out - out.mean()


def make_panel(rng):
    """Unit-level log outcome panel for both arms plus the treatment matrix."""
    k = np.arange(YEARS.size)
    # Common time effects: a trend, a mild cycle and a one-year shock.
    period = 0.019 * k + 0.026 * np.sin(0.85 * k) + rng.normal(0, 0.006, k.size)
    period[YEARS == SHOCK] -= 0.041

    ramp = np.clip((YEARS - T0 + 1) / 3.0, 0.0, 1.0) * (YEARS >= T0)

    a_t = rng.normal(2.62, 0.155, N_T)
    a_c = rng.normal(2.55, 0.170, N_C)
    # Group-specific wobble: without it the pre-period looks impossibly flat,
    # because the group mean of 60-odd units would average the noise away.
    # It has to be stationary rather than a random walk - a walk would drift
    # the two arms apart on its own and quietly bias the estimand.
    w_t, w_c = ar1(rng, k.size), ar1(rng, k.size)

    y_t = (a_t[:, None] + period[None, :] + w_t[None, :]
           + TAU * ramp[None, :] + rng.normal(0, 0.050, (N_T, YEARS.size)))
    y_c = (a_c[:, None] + period[None, :] + w_c[None, :]
           + rng.normal(0, 0.054, (N_C, YEARS.size)))
    return y_t, y_c, ramp


def twfe(y, d, cluster_units=False):
    """Two-way fixed-effects estimate of the coefficient on ``d``.

    Within-transforming both sides removes the unit and period effects
    exactly, so the slope of the residualised regression is the TWFE
    coefficient; the cluster option swaps the homoskedastic meat for the
    unit-clustered one, which is what widens the interval in the forest.
    """
    def demean(a):
        return a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()

    yd, dd = demean(y), demean(d)
    sxx = float((dd * dd).sum())
    b = float((dd * yd).sum() / sxx)
    e = yd - b * dd
    n, t = y.shape
    dof = max(n * t - n - t, 1)
    if cluster_units:
        g = (dd * e).sum(axis=1)
        se = float(np.sqrt(n / (n - 1.0) * (g ** 2).sum()) / sxx)
    else:
        se = float(np.sqrt((e ** 2).sum() / dof / sxx))
    return b, se


def stack(y_t, y_c, ramp, cols=None):
    """Pooled outcome matrix and treatment matrix, optionally sub-setting."""
    y = np.vstack([y_t, y_c])
    d = np.vstack([np.tile(ramp, (y_t.shape[0], 1)),
                   np.zeros((y_c.shape[0], YEARS.size))])
    if cols is not None:
        y, d = y[:, cols], d[:, cols]
    return y, d


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def trends_panel(ax, m_t, s_t, m_c, s_c, cf, did_bar, pre_p, pre_slope):
    ax.axvspan(T0 - 0.5, YEARS[-1] + 0.5, facecolor=CARD, edgecolor="none",
               zorder=0)
    ax.axvline(T0 - 0.5, color="#3a3a3a", lw=1.0, ls=(0, (3.4, 2.2)), zorder=5)

    for m, s, col, lab in ((m_c, s_c, CTRL, f"control (n = {N_C})"),
                           (m_t, s_t, TREAT, f"treated (n = {N_T})")):
        ax.fill_between(YEARS, m - 1.96 * s, m + 1.96 * s,
                        facecolor=lighten(col, 0.78), edgecolor="none",
                        alpha=0.85, zorder=2)
        ax.plot(YEARS, m, lw=1.7, color=col, marker="o", ms=3.2, mew=0.0,
                zorder=4, label=lab)

    keep = YEARS >= T0 - 1
    ax.plot(YEARS[keep], cf[keep], lw=1.3, color=COUNTER, ls=(0, (4.5, 2.4)),
            zorder=3, label="counterfactual for treated")

    # The double arrow is the estimate: it spans exactly the plotted gap.
    xg = YEARS[-1]
    lo, hi = cf[-1], m_t[-1]
    ax.annotate("", xy=(xg + 0.30, hi), xytext=(xg + 0.30, lo),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=8),
                annotation_clip=False, zorder=6)
    ax.text(xg + 0.46, 0.5 * (lo + hi),
            f"gap {xg}\n{100 * (hi - lo):+.1f} log pt", ha="left",
            va="center", fontsize=6.8, color=INK, clip_on=False)

    ax.text(T0 - 0.7, 0.965, "pre-policy", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=6.6, color=MUTED)
    ax.text(T0 - 0.3, 0.965, "post-policy", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=6.6, color=SUB)

    ax.set_xlim(YEARS[0] - 0.5, YEARS[-1] + 1.25)
    ax.set_xticks(YEARS[::2])
    ax.set_xlabel("year (policy enacted 2018)", fontsize=8.0)
    ax.set_ylabel("log employment index (log points)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Common trend before 2018, sustained divergence after",
                 fontsize=9.0, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    h, l = ax.get_legend_handles_labels()
    order = [1, 0, 2]                      # treated first: it is the subject
    leg = ax.legend([h[i] for i in order], [l[i] for i in order],
                    loc="upper left", fontsize=7.0, frameon=False,
                    handlelength=1.6, borderaxespad=0.35, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK)

    ax.text(0.030, 0.615,
            f"pre-trend slope {100 * pre_slope:+.2f} log pt/yr, p = {pre_p:.2f}"
            f"\naverage post-policy DiD {100 * did_bar:+.1f} log pt",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB, linespacing=1.45)


def table_panel(ax, w, h, cells, did, se, tval, pval):
    """2x2 group-by-period means, with the DiD cell called out as a card."""
    ax.text(0.06, h - 0.05, "Group-by-period means (log index)", ha="left",
            va="top", fontsize=8.5, fontweight="bold", color=INK)

    xs = (1.32, 1.98, 2.66)
    heads = ("pre\n2009-17", "post\n2018-23", "difference\n(log pt)")
    y_head, rows_y = h - 0.46, (h - 0.85, h - 1.16)
    y_diff = h - 1.52

    for x, head in zip(xs, heads):
        ax.text(x, y_head, head, ha="center", va="center", fontsize=6.6,
                color=SUB, linespacing=1.35)
    ax.plot([0.06, w - 0.06], [y_head - 0.20] * 2, color=HAIR, lw=0.7)

    names = (f"treated (n = {N_T})", f"control (n = {N_C})")
    for y, name, col, row in zip(rows_y, names, (TREAT, CTRL), cells[:2]):
        ax.text(0.06, y, name, ha="left", va="center", fontsize=6.8,
                color=col)
        for x, v, fmt in zip(xs, row, ("{:.3f}", "{:.3f}", "{:+.1f}")):
            ax.text(x, y, fmt.format(v), ha="center", va="center",
                    fontsize=6.8, color=INK)
    ax.plot([0.06, w - 0.06], [rows_y[1] - 0.155] * 2, color=HAIR, lw=0.7)

    ax.text(0.06, y_diff, "difference", ha="left", va="center", fontsize=6.8,
            color=SUB)
    for x, v in zip(xs[:2], cells[2][:2]):
        ax.text(x, y_diff, f"{100 * v:+.1f}", ha="center", va="center",
                fontsize=6.8, color=SUB)
    text_card(ax, xs[2] - 0.36, y_diff - 0.17, 0.72, 0.34,
              f"{100 * did:+.1f}", fc=lighten(TREAT, 0.86), ec=TREAT,
              fs=7.6, color=TREAT, weight="bold", ha="center", lw=0.8,
              rounding=0.05)

    ax.text(0.06, 0.06,
            f"DiD = {100 * did:+.1f} log pt  (SE {100 * se:.1f}, "
            f"t = {tval:.2f}, p = {pval:.1e})", ha="left", va="bottom",
            fontsize=6.6, color=SUB)


def forest_panel(ax, specs, truth):
    ys = np.arange(len(specs))[::-1]
    ax.axvline(0.0, color=BASE, lw=0.8, ls=(0, (3.0, 2.2)), zorder=1)
    for y, (name, est, se) in zip(ys, specs):
        lo, hi = 100 * (est - 1.96 * se), 100 * (est + 1.96 * se)
        sig = lo > 0 or hi < 0
        col = TREAT if name.startswith("Two-way FE") else (CTRL if sig
                                                           else "#9a9a9a")
        ax.plot([lo, hi], [y, y], lw=1.9 if sig else 1.1, color=col,
                solid_capstyle="butt", zorder=3)
        ax.plot([100 * est], [y], marker="o", ms=4.0, color=col,
                mfc=col if sig else "white", mec=col, mew=1.0, zorder=4)

    ax.axvline(100 * truth, color=BASE, lw=0.7, ls=(0, (1.6, 1.8)), zorder=1)
    ax.text(100 * truth, len(specs) - 0.42, "design value", ha="center",
            va="bottom", fontsize=6.2, color=MUTED)

    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in specs], fontsize=6.5)
    ax.set_ylim(-0.7, len(specs) - 0.05)
    ax.set_xlabel("DiD estimate (log points, 95 % CI)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("Six specifications, one estimand", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(768)
    y_t, y_c, ramp = make_panel(rng)

    m_t, m_c = y_t.mean(0), y_c.mean(0)
    s_t = y_t.std(0, ddof=1) / np.sqrt(N_T)
    s_c = y_c.std(0, ddof=1) / np.sqrt(N_C)

    pre, post = YEARS < T0, YEARS >= T0
    cf = m_t[pre].mean() + (m_c - m_c[pre].mean())
    did_bar = float((m_t[post] - cf[post]).mean())

    # Unit-level 2x2: each unit contributes one post-minus-pre difference, so
    # the two-sample t test below is clustered at the unit by construction.
    d_t = y_t[:, post].mean(1) - y_t[:, pre].mean(1)
    d_c = y_c[:, post].mean(1) - y_c[:, pre].mean(1)
    tt = stats.ttest_ind(d_t, d_c, equal_var=False)
    did22 = float(d_t.mean() - d_c.mean())
    se22 = float(abs(did22 / tt.statistic))

    pre_fit = stats.linregress(YEARS[pre], (m_t - m_c)[pre])

    cells = [
        (y_t[:, pre].mean(), y_t[:, post].mean(), 100 * d_t.mean()),
        (y_c[:, pre].mean(), y_c[:, post].mean(), 100 * d_c.mean()),
        (y_t[:, pre].mean() - y_c[:, pre].mean(),
         y_t[:, post].mean() - y_c[:, post].mean(), did22),
    ]

    y_all, d_all = stack(y_t, y_c, ramp)
    b_fe, se_fe = twfe(y_all, d_all)
    b_cl, se_cl = twfe(y_all, d_all, cluster_units=True)

    keep = np.where(YEARS != SHOCK)[0]
    b_ns, se_ns = twfe(*stack(y_t, y_c, ramp, keep))

    win = np.where(YEARS >= T0 - 4)[0]
    b_w, se_w = twfe(*stack(y_t, y_c, ramp, win))

    # Trim the 5 % most extreme pre-period unit means in each arm.
    def trim(y):
        mu = y[:, pre].mean(1)
        lo, hi = np.percentile(mu, [2.5, 97.5])
        return y[(mu >= lo) & (mu <= hi)]

    yt2, yc2 = trim(y_t), trim(y_c)
    d2 = np.vstack([np.tile(ramp, (yt2.shape[0], 1)),
                    np.zeros((yc2.shape[0], YEARS.size))])
    b_tr, se_tr = twfe(np.vstack([yt2, yc2]), d2)

    specs = [
        ("2x2 group means", did22, se22),
        ("Two-way FE", b_fe, se_fe),
        ("Two-way FE, unit cluster", b_cl, se_cl),
        (f"Drop {SHOCK} shock", b_ns, se_ns),
        ("Pre-window 4 years", b_w, se_w),
        ("Trim 5 % outlier units", b_tr, se_tr),
    ]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.82, 5.24, 4.52))
    ax_tab = inch_axes(fig, (7.05, 3.39, 3.05, 1.95))
    ax_for = axes_inches(fig, (7.05, 0.82, 3.05, 1.86))

    trends_panel(ax_main, m_t, s_t, m_c, s_c, cf, did_bar,
                 pre_fit.pvalue, pre_fit.slope)
    table_panel(ax_tab, 3.05, 1.95, cells, did22, se22,
                float(tt.statistic), float(tt.pvalue))
    forest_panel(ax_for, specs, TAU * float(ramp[post].mean()))


    for ax, lab in ((ax_main, "(a)"), (ax_tab, "(b)"), (ax_for, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.10)
    savefig(fig, "768_ci_did_parallel_trends.png")


if __name__ == "__main__":
    main()
