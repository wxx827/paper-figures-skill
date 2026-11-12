"""Weekly demand forecast with a quantile fan and an aligned error track.

The main panel holds two years of weekly history, a seasonal-plus-trend
forecast drawn as nested 50 / 80 / 90 % quantile bands, and the eight weeks
of realised demand that have since arrived, so the fan can be judged rather
than admired.  Directly underneath, sharing the same x axis, is the signed
error track: light bars for in-sample residuals, solid bars for the holdout
weeks.  The auxiliary column carries the portfolio view - an ABC x XYZ
classification matrix and how accuracy decays with horizon.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches, safe_colorbar

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"

N_HIST = 104        # weeks of history
N_FCST = 16         # weeks of forecast
N_VALID = 8         # holdout weeks already realised
QUANTILES = [(0.05, 0.95, 0.16, "90 % interval"),
             (0.10, 0.90, 0.24, "80 % interval"),
             (0.25, 0.75, 0.34, "50 % interval")]


def signal(t):
    """Trend plus annual and quarterly seasonality, in units per week."""
    return (1450.0 + 4.6 * t
            + 235.0 * np.sin(2 * np.pi * (t - 6.0) / 52.0)
            + 78.0 * np.sin(2 * np.pi * t / 13.0 + 0.6))


def main():
    set_pub_style()
    rng = np.random.default_rng(561)

    t_hist = np.arange(N_HIST)
    t_fcst = np.arange(N_HIST, N_HIST + N_FCST)
    hist = signal(t_hist) + rng.normal(0.0, 96.0, N_HIST)
    fitted = signal(t_hist)
    resid = hist - fitted
    sigma = float(resid.std())

    median = signal(t_fcst) - 18.0                    # slight model bias
    grow = np.sqrt(1.0 + np.arange(1, N_FCST + 1) / 5.0)
    actual = (signal(t_fcst[:N_VALID])
              + rng.normal(30.0, 108.0, N_VALID))
    err = actual - median[:N_VALID]
    mape = 100.0 * np.mean(np.abs(err) / actual)
    bias = 100.0 * np.mean(err) / actual.mean()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.46],
                          wspace=0.24, hspace=0.30)
    axm = fig.add_subplot(gs[0, 0])
    axe = fig.add_subplot(gs[1, 0], sharex=axm)
    axk = fig.add_subplot(gs[0, 1])
    axh = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------ (a) history + fan
    axm.axvspan(N_HIST - 0.5, N_HIST + N_VALID - 0.5, color="#f5f5f5",
                zorder=0)
    axm.plot(t_hist, hist, color=BLUE, lw=1.15, zorder=4,
             label="observed demand")
    for lo, hi, alpha, lab in QUANTILES:
        zlo, zhi = norm.ppf(lo), norm.ppf(hi)
        axm.fill_between(t_fcst, median + zlo * sigma * grow,
                         median + zhi * sigma * grow, color=GREEN,
                         alpha=alpha, linewidth=0.0, zorder=2, label=lab)
    axm.plot(t_fcst, median, color=GREEN, lw=1.5, ls="--", zorder=5,
             label="forecast median")
    axm.plot(t_fcst[:N_VALID], actual, color=INK, lw=1.0, marker="o", ms=3.2,
             mfc="white", mec=INK, mew=0.9, zorder=6,
             label="realised (holdout)")
    axm.axvline(N_HIST - 0.5, color=RULE, lw=0.9, ls=":", zorder=5)
    axm.text(N_HIST - 1.5, axm.get_ylim()[1], "forecast origin", fontsize=6.6,
             color=RULE, ha="right", va="top", rotation=90)

    axm.text(0.012, 0.972,
             "holdout MAPE %.1f%%\nbias %+.1f%% of mean demand\n"
             "in-sample sigma %.0f units/week" % (mape, bias, sigma),
             transform=axm.transAxes, fontsize=6.8, color=INK, ha="left",
             va="top", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc",
                       lw=0.6, alpha=0.94))
    axm.set_ylabel("demand (units/week)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, labelbottom=False)
    axm.grid(axis="y", color="#e9e7e2", lw=0.55)
    axm.set_axisbelow(True)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)
    axm.legend(fontsize=6.6, frameon=False, loc="lower right", ncol=2,
               labelspacing=0.32, handlelength=1.6, columnspacing=1.1)
    axm.set_title("Two years of weekly demand and the next 16 weeks",
                  fontsize=9.0, fontweight="bold", color=INK, pad=5)

    # ---------------------------------------------------- (b) error track
    axe.axhline(0.0, color="#808080", lw=0.8, zorder=3)
    axe.bar(t_hist, resid, width=0.85, color=BLUE, alpha=0.35,
            edgecolor="none", zorder=2, label="in-sample residual")
    axe.bar(t_fcst[:N_VALID], err, width=0.85, color=GREEN,
            edgecolor="white", linewidth=0.4, zorder=4,
            label="holdout error")
    axe.axvline(N_HIST - 0.5, color=RULE, lw=0.9, ls=":", zorder=5)
    axe.set_xlim(-1.0, N_HIST + N_FCST)
    axe.set_xlabel("week index (weeks)", fontsize=8.0)
    axe.set_ylabel("error (units/week)", fontsize=8.0)
    axe.tick_params(labelsize=7.0)
    axe.grid(axis="y", color="#e9e7e2", lw=0.55)
    axe.set_axisbelow(True)
    axe.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2,
               labelspacing=0.3, handlelength=1.3, columnspacing=1.0)
    for sp in ("top", "right"):
        axe.spines[sp].set_visible(False)

    # ------------------------------------------------- (c) ABC-XYZ matrix
    counts = np.array([[46.0, 22.0, 9.0],
                       [58.0, 71.0, 34.0],
                       [37.0, 96.0, 152.0]])
    abc = ["A (top 80 % value)", "B (next 15 %)", "C (last 5 %)"]
    xyz = ["X (stable)", "Y (seasonal)", "Z (erratic)"]
    im = axk.imshow(counts, cmap=get_cmap("cb_sequential_blue"),
                    aspect="auto", vmin=0.0, vmax=counts.max())
    for i in range(3):
        for j in range(3):
            v = counts[i, j]
            axk.text(j, i, "%d" % v, ha="center", va="center", fontsize=7.0,
                     color="white" if v > 0.55 * counts.max() else INK)
    axk.set_xticks(range(3))
    axk.set_xticklabels(xyz, fontsize=6.6)
    axk.set_yticks(range(3))
    axk.set_yticklabels(abc, fontsize=6.6)
    axk.tick_params(length=0)
    for sp in axk.spines.values():
        sp.set_visible(False)
    axk.set_title("Where the forecastable value sits", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    # --------------------------------------------- (d) accuracy by horizon
    horizon = np.arange(1, N_FCST + 1)
    mape_h = mape * grow / grow[0]
    axh.bar(horizon, mape_h, width=0.72, color=BLUE, edgecolor="white",
            linewidth=0.4, zorder=3, label="extrapolated horizon")
    axh.bar(horizon[:N_VALID], mape_h[:N_VALID], width=0.72, color=GREEN,
            edgecolor="white", linewidth=0.4, zorder=4,
            label="validated by holdout")
    axh.axhline(mape, color=RED, lw=1.0, ls="--", zorder=5,
                label="holdout MAPE %.1f%%" % mape)
    axh.legend(fontsize=6.4, frameon=False, loc="upper left",
               labelspacing=0.30, handlelength=1.3)
    axh.set_xlabel("forecast horizon (weeks ahead)", fontsize=8.0)
    axh.set_ylabel("expected MAPE (%)", fontsize=8.0)
    axh.set_xticks([1, 4, 8, 12, 16])
    axh.set_ylim(0, mape_h.max() * 1.52)
    axh.tick_params(labelsize=6.8)
    axh.grid(axis="y", color="#e9e7e2", lw=0.55)
    axh.set_axisbelow(True)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    axh.set_title("Accuracy decays with horizon", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    band = sigma * grow[:N_VALID]
    inside = int(np.sum((actual >= median[:N_VALID] + norm.ppf(0.10) * band)
                        & (actual <= median[:N_VALID]
                           + norm.ppf(0.90) * band)))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.4, w_pad=2.6)
    safe_colorbar(fig, im, axk, label="SKUs (count)", width=0.10, pad=0.07,
                  labelsize=6.5)
    panel_labels_inches(fig, [axm, axe, axk, axh], dx=-0.46, dy=0.12)
    savefig(fig, "561_scm_demand_forecast.png")


if __name__ == "__main__":
    main()
