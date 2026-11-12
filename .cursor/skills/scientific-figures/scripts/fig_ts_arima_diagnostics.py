"""Four-panel residual diagnostic sheet for a fitted ARIMA(1,1,1) model.

(a) standardised residuals over time, (b) their autocorrelation with the
+-1.96/sqrt(n) band, (c) a normal Q-Q plot against the 45-degree reference and
(d) Ljung-Box p-values by lag against the 5 % line.  Every number printed on
the figure is recomputed from the residual array that is being drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, annotate_box
from figkit import panel_labels_inches

BLUE = "#3b5b92"       # house blue: the model's own quantities
RED = "#b23a48"        # house red: the single emphasis (thresholds)
INK = "#3a3a3a"        # observed / residual traces
GREY = "#808080"
N = 180
N_LAG = 22
N_PARAM = 2            # ARIMA(1,1,1): one AR + one MA coefficient


def acf(x, nlags):
    """Sample autocorrelation r_1..r_nlags of a mean-centred series."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / denom
                     for k in range(1, nlags + 1)])


def ljung_box(r, n, dof_used):
    """Q statistic and p-value at every lag m = 1..len(r)."""
    m = np.arange(1, r.size + 1)
    terms = r ** 2 / (n - m)
    q = n * (n + 2) * np.cumsum(terms)
    dof = np.maximum(m - dof_used, 1)
    return q, 1.0 - stats.chi2.cdf(q, dof)


def panel_residuals(ax, resid):
    ax.axhline(0.0, color=GREY, lw=0.7)
    for lim in (-2.0, 2.0):
        ax.axhline(lim, color=RED, lw=0.7, ls=(0, (2.4, 2.0)), zorder=2)
    ax.plot(np.arange(resid.size), resid, color=INK, lw=0.9, zorder=3)
    out = np.abs(resid) > 2.0
    ax.plot(np.arange(resid.size)[out], resid[out], ls="none", marker="o",
            ms=3.4, mfc="none", mec=RED, mew=0.9, zorder=4)
    ax.text(resid.size - 2, 2.08, r"$\pm2\sigma$", ha="right", va="bottom",
            fontsize=6.6, color=RED)
    ax.set_xlim(-3, resid.size + 2)
    ax.set_ylim(-4.3, 3.6)
    ax.set_xlabel("Observation index (-)")
    ax.set_ylabel("Standardised residual (-)")
    ax.set_title("Residuals show no drift or variance change", fontsize=9,
                 fontweight="bold")
    annotate_box(ax, f"mean = {resid.mean():+.3f}\n"
                     f"s.d. = {resid.std(ddof=1):.3f}\n"
                     f"|z| > 2 : {int(out.sum())} of {resid.size}",
                 loc="lower left", fontsize=6.6)


def panel_acf(ax, r, n):
    lags = np.arange(1, r.size + 1)
    band = 1.96 / np.sqrt(n)
    ax.fill_between([0.2, r.size + 0.8], -band, band, color=BLUE, alpha=0.11,
                    lw=0, zorder=1)
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=2)
    sig = np.abs(r) > band
    ax.vlines(lags, 0.0, r, color=np.where(sig, RED, BLUE), lw=1.5, zorder=3)
    ax.plot(lags, r, ls="none", marker="o", ms=2.6,
            mfc="white", mec=BLUE, mew=0.8, zorder=4)
    ax.plot(lags[sig], r[sig], ls="none", marker="o", ms=2.9, color=RED,
            zorder=5)
    ax.text(r.size - 0.3, band + 0.012, r"$\pm1.96/\sqrt{n}$", ha="right",
            va="bottom", fontsize=6.6, color=BLUE)
    ax.set_xlim(0.2, r.size + 0.8)
    ax.set_ylim(-0.30, 0.30)
    ax.set_xticks(np.arange(0, r.size + 1, 4))
    ax.set_xlabel("Lag $k$ (months)")
    ax.set_ylabel(r"Autocorrelation $\rho_k$ (-)")
    ax.set_title(f"Only {int(sig.sum())} of {r.size} lags leave the band",
                 fontsize=9, fontweight="bold")


def panel_qq(ax, resid):
    n = resid.size
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    samp = np.sort(resid)
    lo, hi = -3.3, 3.3
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls=(0, (3.0, 2.2)),
            zorder=2)
    ax.plot(theo, samp, ls="none", marker="o", ms=2.9, mfc="none", mec=BLUE,
            mew=0.75, zorder=3)
    tail = np.abs(theo) > 2.2
    ax.plot(theo[tail], samp[tail], ls="none", marker="o", ms=3.1, color=RED,
            zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Theoretical quantile (-)")
    ax.set_ylabel("Sample quantile (-)")
    ax.set_title("Tails sit close to the normal line", fontsize=9,
                 fontweight="bold")
    w, p = stats.shapiro(resid)
    ax.text(0.97, 0.05, f"Shapiro-Wilk $W$ = {w:.3f}\n$p$ = {p:.2f}\n"
                        f"skew = {stats.skew(resid):+.2f}, "
                        f"kurt = {stats.kurtosis(resid):+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color="#555555", linespacing=1.35)


def panel_ljung(ax, q, p, dof_used):
    lags = np.arange(1, p.size + 1)
    ax.fill_between([0.2, p.size + 0.8], 0.0, 0.05, color=RED, alpha=0.10,
                    lw=0, zorder=1)
    ax.axhline(0.05, color=RED, lw=0.9, ls=(0, (2.6, 2.0)), zorder=2)
    ax.plot(lags, p, color=BLUE, lw=1.3, zorder=3)
    ax.plot(lags, p, ls="none", marker="o", ms=3.0, mfc="white", mec=BLUE,
            mew=0.85, zorder=4)
    ax.set_xlim(0.2, p.size + 0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, p.size + 1, 4))
    ax.set_xlabel("Lag $m$ tested jointly (months)")
    ax.set_ylabel("Ljung-Box $p$-value (-)")
    ax.text(p.size * 0.985, 0.062, r"reject white noise below $p=0.05$",
            ha="right", va="bottom", fontsize=6.6, color=RED)
    ax.set_title(f"Minimum $p$ = {p.min():.2f} at lag "
                 f"{int(lags[np.argmin(p)])}", fontsize=9, fontweight="bold")
    ax.text(0.025, 0.965, f"$Q({p.size})$ = {q[-1]:.1f},  "
                          f"d.f. = {p.size - dof_used}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color="#555555")


def main():
    set_pub_style()
    rng = np.random.default_rng(63)
    raw = rng.normal(0.0, 1.0, N)
    raw = raw + 0.05 * np.roll(raw, 1)
    resid = raw / raw.std(ddof=1)

    r = acf(resid, N_LAG)
    q, p = ljung_box(r, N, N_PARAM)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.1))
    panel_residuals(axes[0, 0], resid)
    panel_acf(axes[0, 1], r, N)
    panel_qq(axes[1, 0], resid)
    panel_ljung(axes[1, 1], q, p, N_PARAM)
    for ax in axes.ravel():
        ax.grid(axis="y", lw=0.5, color="#f0f0f0")
        ax.set_axisbelow(True)
        style_axes(ax)
    fig.tight_layout(rect=[0, 0.028, 1, 0.925], h_pad=3.0, w_pad=2.8)
    panel_labels_inches(fig, axes, dx=-0.76, dy=0.18)
    savefig(fig, "355_ts_arima_diagnostics.png")


if __name__ == "__main__":
    main()
