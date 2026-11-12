"""Additive decomposition and breakpoint detection of a 20-year NDVI record.

Panels (a)-(d) share the time axis: the monthly observations with the
reconstructed signal, the trend component with a two-breakpoint segmented fit,
the seasonal component and the remainder.  Panel (e) is a deliberately
subordinate side panel giving the variance partition of the three components.

The decomposition is a classical additive one (centred 12-month moving average
for the trend, month-of-year means for the season).  Breakpoints are located
by exhaustive least-squares search over a segmented model that allows both
level jumps and slope changes; their uncertainty comes from a residual
bootstrap.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

T_START = 2003.0
N_MONTH = 240                      # 20 years, monthly


def design(t, taus):
    """Segmented model: intercept, slope, then a jump and a slope change at
    every breakpoint."""
    cols = [np.ones_like(t), t]
    for tau in taus:
        cols.append(np.clip(t - tau, 0, None))
        cols.append((t > tau).astype(float))
    return np.column_stack(cols)


def fit_seg(t, y, taus):
    X = design(t, taus)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def search_breaks(t, y, cand):
    best, ssr_best = None, np.inf
    prof = {}
    for i, t1 in enumerate(cand):
        for t2 in cand[i + 1:]:
            if t2 - t1 < 2.5:
                continue
            _, ssr = fit_seg(t, y, (t1, t2))
            if ssr < ssr_best:
                best, ssr_best = (t1, t2), ssr
            prof[(t1, t2)] = ssr
    return best, ssr_best, prof


def bootstrap_breaks(rng, t, y, taus, cand, n_boot=200, half_width=1.5):
    beta, _ = fit_seg(t, y, taus)
    fit = design(t, taus) @ beta
    res = y - fit
    c1 = cand[np.abs(cand - taus[0]) <= half_width]
    c2 = cand[np.abs(cand - taus[1]) <= half_width]
    out = np.zeros((n_boot, 2))
    for b in range(n_boot):
        yb = fit + res[rng.integers(0, res.size, res.size)]
        best, sb = None, np.inf
        for t1 in c1:
            for t2 in c2:
                if t2 - t1 < 2.5:
                    continue
                _, ssr = fit_seg(t, yb, (t1, t2))
                if ssr < sb:
                    best, sb = (t1, t2), ssr
        out[b] = best
    return np.percentile(out, [2.5, 97.5], axis=0)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240191)

    t = T_START + np.arange(N_MONTH) / 12.0

    # ---- synthetic NDVI record -------------------------------------------
    tau_true = (2010.42, 2016.75)
    trend_true = 0.545 + 0.0021 * (t - T_START)
    trend_true = np.where(t > tau_true[0],
                          0.545 + 0.0021 * (tau_true[0] - T_START) - 0.048
                          + 0.0135 * (t - tau_true[0]), trend_true)
    lvl2 = (0.545 + 0.0021 * (tau_true[0] - T_START) - 0.048
            + 0.0135 * (tau_true[1] - tau_true[0]))
    trend_true = np.where(t > tau_true[1],
                          lvl2 + 0.0012 * (t - tau_true[1]), trend_true)

    phase = 2 * np.pi * (t - T_START)
    seas_true = (0.088 * np.sin(phase - 1.05) + 0.026 * np.sin(2 * phase + 0.4)
                 + 0.011 * np.sin(3 * phase - 0.8))
    noise = np.zeros(N_MONTH)
    for i in range(1, N_MONTH):                     # AR(1) remainder
        noise[i] = 0.42 * noise[i - 1] + rng.normal(0, 0.014)
    obs = trend_true + seas_true + noise

    # ---- classical additive decomposition --------------------------------
    w = np.ones(13) / 12.0          # centred 12-month MA (half weights at ends)
    w[0] = w[-1] = 0.5 / 12.0
    pad = np.r_[obs[:6][::-1], obs, obs[-6:][::-1]]
    trend = np.convolve(pad, w, mode="same")[6:-6]
    detr = obs - trend
    month = np.arange(N_MONTH) % 12
    seas = np.array([detr[month == m].mean() for m in range(12)])[month]
    seas -= seas.mean()
    resid = obs - trend - seas

    # ---- breakpoint search on the deseasonalised series ------------------
    # (running it on the raw deseasonalised record, not on the smoothed trend,
    # keeps the full remainder variance in the breakpoint uncertainty)
    deseas = obs - seas
    cand = np.arange(T_START + 2.0, T_START + 18.0, 1.0 / 12.0)
    taus, _, _ = search_breaks(t, deseas, cand)
    beta, _ = fit_seg(t, deseas, taus)
    seg_fit = design(t, taus) @ beta
    ci = bootstrap_breaks(rng, t, deseas, taus, cand)

    slopes = [beta[1], beta[1] + beta[2], beta[1] + beta[2] + beta[4]]
    jumps = [beta[3], beta[5]]

    # ---- variance partition ----------------------------------------------
    parts = np.array([trend.var(), seas.var(), resid.var()])
    parts = 100 * parts / parts.sum()

    fig = plt.figure(figsize=(7.4, 6.0))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[3.25, 1.0],
                  height_ratios=[1.15, 1.15, 0.85, 0.85],
                  hspace=0.24, wspace=0.16)

    col_obs, col_tr, col_se, col_re = "#333333", "#1f3f6b", "#2e7d5b", "#8a6d3b"
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    for i, ax in enumerate(axes):
        ax.set_xlim(T_START, T_START + 20)
        ax.set_xticks(np.arange(2004, 2023, 3))
        ax.set_xticklabels([str(y) for y in range(2004, 2023, 3)])
        ax.grid(axis="y", lw=0.45, alpha=0.5)
        style_axes(ax)
        if i < 3:
            ax.tick_params(labelbottom=False)
        for tau, c in zip(taus, ("#b23a48", "#b23a48")):
            ax.axvline(tau, color=c, lw=1.0, ls=(0, (4, 2.5)), zorder=2)

    # (a) observations
    axes[0].plot(t, obs, color=col_obs, lw=0.8, marker="o", ms=1.6,
                 mfc="white", mew=0.3, zorder=4, label="Monthly NDVI")
    axes[0].plot(t, seg_fit + seas, color="#c0603f", lw=1.0, alpha=0.9,
                 zorder=5, label="Segmented + seasonal reconstruction")
    axes[0].set_ylabel("NDVI (\u2013)")
    axes[0].legend(loc="lower right", fontsize=6.2, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc", handlelength=1.4,
                   labelspacing=0.2, borderpad=0.3, borderaxespad=0.3)
    add_panel_label(axes[0], "(a)", x=-0.075, y=1.02, size=10)
    axes[0].set_title("Observed monthly NDVI and its additive decomposition",
                      fontsize=8.6)

    # (b) trend + segmented fit
    axes[1].plot(t, trend, color=col_tr, lw=1.3, zorder=4, label="Trend component")
    axes[1].plot(t, seg_fit, color="#b23a48", lw=1.1, ls="--", zorder=5,
                 label="Segmented fit")
    for j, tau in enumerate(taus):
        axes[1].axvspan(ci[0, j], ci[1, j], color="#b23a48", alpha=0.14, lw=0,
                        zorder=1)
    axes[1].set_ylabel("Trend (\u2013)")
    ylo, yhi = trend.min(), trend.max()
    axes[1].set_ylim(ylo - 0.035, yhi + 0.055)
    for j, tau in enumerate(taus):
        axes[1].text(tau, yhi + 0.048,
                     f"BP{j + 1}  {tau:.2f}\n95% CI [{ci[0, j]:.2f}, "
                     f"{ci[1, j]:.2f}]",
                     fontsize=6.0, color="#8c2f2f", ha="center", va="top",
                     fontweight="bold", linespacing=1.2, zorder=6,
                     bbox=dict(boxstyle="round,pad=0.22", fc="white",
                               ec="#e0c4c4", lw=0.5, alpha=0.95))
    axes[1].annotate("", xy=(taus[0], seg_fit[t > taus[0]][0]),
                     xytext=(taus[0], seg_fit[t <= taus[0]][-1]),
                     arrowprops=dict(arrowstyle="->", color="#8c2f2f", lw=1.0))
    axes[1].text(taus[0] - 0.30, seg_fit[t <= taus[0]][-1] + 0.022,
                 f"jump {jumps[0]:+.3f}".replace("-", "\u2212"),
                 fontsize=6.0, color="#8c2f2f", ha="right", va="bottom",
                 fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=6.2, frameon=True,
                   framealpha=0.9, edgecolor="#cccccc", handlelength=1.4,
                   labelspacing=0.2, borderpad=0.3, borderaxespad=0.3)
    add_panel_label(axes[1], "(b)", x=-0.075, y=1.02, size=10)
    for j, (x0, x1) in enumerate(((T_START, taus[0]), (taus[0], taus[1]),
                                  (taus[1], T_START + 20))):
        xm = 0.5 * (x0 + x1)
        ym = float(np.interp(xm, t, seg_fit)) - 0.016
        axes[1].text(xm, ym,
                     f"{slopes[j] * 1000:+.1f}".replace("-", "\u2212")
                     + r" $\times10^{-3}$ yr$^{-1}$",
                     fontsize=6.0, color=col_tr, ha="center", va="top")

    # (c) seasonal
    axes[2].plot(t, seas, color=col_se, lw=0.9)
    axes[2].axhline(0, color="#555555", lw=0.7)
    axes[2].set_ylabel("Seasonal (\u2013)")
    axes[2].text(0.012, 0.94,
                 f"peak-to-peak {seas.max() - seas.min():.3f} NDVI",
                 transform=axes[2].transAxes, fontsize=6.1, va="top",
                 color=col_se, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec="none", alpha=0.85))
    add_panel_label(axes[2], "(c)", x=-0.075, y=1.04, size=10)

    # (d) remainder
    sd = resid.std()
    axes[3].axhspan(-2 * sd, 2 * sd, color="#d9d9d9", alpha=0.45, lw=0)
    axes[3].plot(t, resid, color=col_re, lw=0.8)
    out = np.abs(resid) > 2 * sd
    axes[3].plot(t[out], resid[out], ls="none", marker="o", ms=2.6,
                 mfc="#b23a48", mec="white", mew=0.4, zorder=5)
    axes[3].axhline(0, color="#555555", lw=0.7)
    axes[3].set_ylabel("Remainder (\u2013)")
    axes[3].set_xlabel("Year", labelpad=1.5)
    axes[3].text(0.012, 0.92,
                 f"$\\sigma$ = {sd:.3f},  {out.sum()} points beyond 2$\\sigma$",
                 transform=axes[3].transAxes, fontsize=6.1, va="top",
                 color=col_re, fontweight="bold")
    add_panel_label(axes[3], "(d)", x=-0.075, y=1.04, size=10)

    # ---- (e) subordinate variance panel -----------------------------------
    host = fig.add_subplot(gs[:, 1])
    host.set_xticks([])
    host.set_yticks([])
    for s in host.spines.values():
        s.set_visible(False)
    host.add_patch(FancyBboxPatch(
        (0.06, 0.40), 0.94, 0.40, boxstyle="round,pad=0.015",
        transform=host.transAxes, facecolor="#f1f1ec", edgecolor="#d5d5cc",
        linewidth=0.7, zorder=0, clip_on=False))
    host.text(0.53, 0.815, "Variance partition", transform=host.transAxes,
              ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    add_panel_label(host, "(e)", x=0.0, y=0.835, size=10)

    sub = host.inset_axes([0.30, 0.445, 0.62, 0.29])
    names = ["Trend", "Seasonal", "Remainder"]
    cols = [col_tr, col_se, col_re]
    ypos = np.arange(3)[::-1]
    sub.barh(ypos, parts, height=0.62, color=cols, edgecolor="white",
             linewidth=0.5)
    for yv, v in zip(ypos, parts):
        sub.text(v + 1.8, yv, f"{v:.1f}%", va="center", fontsize=6.1,
                 fontweight="bold", color="#1a1a1a")
    sub.set_yticks(ypos)
    sub.set_yticklabels(names, fontsize=6.3)
    sub.set_xlim(0, 118)
    sub.set_xticks([0, 50, 100])
    sub.set_xlabel("share of total variance (%)", fontsize=6.1, labelpad=1.0)
    sub.tick_params(labelsize=6.0)
    sub.patch.set_alpha(0.0)
    style_axes(sub)
    sub.grid(axis="x", lw=0.45, alpha=0.5)

    fig.subplots_adjust(top=0.905, bottom=0.065, left=0.085, right=0.985)
    savefig(fig, "191_rs_timeseries_decomp.png")


if __name__ == "__main__":
    main()
