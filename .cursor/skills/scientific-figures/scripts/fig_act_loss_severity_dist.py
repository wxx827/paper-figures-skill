"""Loss severity: three candidate laws, and the right tail that separates them.

Main panel: the claim-size histogram on a logarithmic money axis, plotted as
density per decade of USD so that the body and the tail are legible in one
frame, with lognormal, gamma and Pareto maximum-likelihood fits drawn over
it.  Each fit carries its own AIC and Kolmogorov-Smirnov statistic and the
lowest-AIC law is drawn heavy; the three curves are indistinguishable in the
body and disagree by orders of magnitude past the 99th percentile.  The two
narrow right-hand panels are the tail diagnostics: a log-log quantile-quantile
plot and the empirical mean excess function.

Severities come from ``gen_losses``; every fitted parameter, information
criterion, test statistic and tail index is estimated from that sample at run
time.  Values are illustrative.
"""
import textwrap

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

LOGN = "#3b5b92"
GAMMA = "#c9a227"
PARETO = "#8e6bbf"
ACCENT = "#b23a48"
BARS = lighten("#c0603f", 0.74)
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.86, 0.74, 5.24, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

N_CLAIMS = 1400


def gen_losses(seed=884):
    """Claim severities in USD: a lognormal body with a Pareto tail graft."""
    rng = np.random.default_rng(seed)
    n_tail = int(round(0.14 * N_CLAIMS))
    body = rng.lognormal(mean=np.log(8600.0), sigma=0.95,
                         size=N_CLAIMS - n_tail)
    tail = 26000.0 * (1.0 - rng.random(n_tail)) ** (-1.0 / 1.55)
    x = np.concatenate([body, tail])
    rng.shuffle(x)
    return x


def money(v, _pos=None):
    if v >= 1e6:
        return f"{v / 1e6:.3g}M"
    if v >= 1e3:
        return f"{v / 1e3:.3g}k"
    return f"{v:.3g}"


def fit_all(x):
    """MLE fits with location pinned at zero, plus AIC and the KS statistic."""
    out = {}

    s, _, sc = stats.lognorm.fit(x, floc=0.0)
    out["lognormal"] = stats.lognorm(s, 0.0, sc)
    a, _, sc = stats.gamma.fit(x, floc=0.0)
    out["gamma"] = stats.gamma(a, 0.0, sc)
    xm = float(x.min())
    b = x.size / np.sum(np.log(x / xm))          # closed-form Pareto MLE
    out["Pareto"] = stats.pareto(b, 0.0, xm)

    stat = {}
    for name, d in out.items():
        ll = float(np.sum(d.logpdf(x)))
        ks = float(stats.kstest(x, d.cdf).statistic)
        stat[name] = dict(aic=2 * 2 - 2 * ll, ks=ks)
    return out, stat


def mean_excess(x, thresholds):
    """Empirical e(u) = E[X - u | X > u] and the exceedance count."""
    e, n = [], []
    for u in thresholds:
        tail = x[x > u]
        e.append(float(np.mean(tail - u)) if tail.size > 4 else np.nan)
        n.append(tail.size)
    return np.array(e), np.array(n)


def main():
    set_pub_style(base_font=8)
    x = gen_losses()
    fits, stat = fit_all(x)
    best = min(stat, key=lambda k: stat[k]["aic"])
    col = {"lognormal": LOGN, "gamma": GAMMA, "Pareto": PARETO}

    q99 = float(np.percentile(x, 99))
    mean_loss = float(np.mean(x))
    median_loss = float(np.median(x))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_q = axes_inches(fig, AUX_T_RECT)
    ax_m = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_q, ax_m):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: histogram as density per decade, plus the three fits --------
    lo, hi = 10.0 ** np.floor(np.log10(x.min())), 10.0 ** np.ceil(
        np.log10(x.max()))
    edges = np.logspace(np.log10(lo), np.log10(hi), 33)
    counts, _ = np.histogram(x, bins=edges)
    width_dec = np.diff(np.log10(edges))
    dens = counts / (x.size * width_dec)          # share of claims per decade

    ax.stairs(dens, edges, fill=True, color=BARS, edgecolor="white",
              linewidth=0.5, zorder=2, label=f"observed claims (n = {x.size})")
    grid = np.logspace(np.log10(lo), np.log10(hi), 600)
    for name, d in fits.items():
        y = d.pdf(grid) * grid * np.log(10.0)
        heavy = name == best
        ax.plot(grid, y, color=col[name], lw=2.2 if heavy else 1.3,
                ls="-" if heavy else (0, (4.0, 1.8)), zorder=6 if heavy else 5,
                label=f"{name} fit" + (" (lowest AIC)" if heavy else ""))

    ax.axvline(q99, color=FAINT, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(q99 * 0.88, 0.985, f"99th percentile\n{money(q99)} USD",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=6.6, color=SUB, linespacing=1.45)

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max(dens.max(), 1.0) * 1.30)
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("claim severity (USD, log scale)", fontsize=8.5)
    ax.set_ylabel("share of claims per decade of USD (-)", fontsize=8.5)
    ax.set_title("Three laws agree on the body and disagree on the tail",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2,
              handlelength=2.2, labelspacing=0.42, borderpad=0.30,
              bbox_to_anchor=(0.005, 0.995))

    rows = [f"{'law':<10}{'AIC':>10}{'KS':>8}"]
    for name in ("lognormal", "gamma", "Pareto"):
        rows.append(f"{name:<10}{stat[name]['aic']:>10,.0f}"
                    f"{stat[name]['ks']:>8.3f}")
    ax.text(0.985, 0.615, "\n".join(rows), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.6, color=SUB,
            family="DejaVu Sans Mono", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))
    ax.text(0.985, 0.885,
            f"mean {money(mean_loss)} USD is {mean_loss / median_loss:.1f}x "
            f"the median\n{money(median_loss)} USD: the tail, not the body, "
            f"sets the price", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.8, color=ACCENT, linespacing=1.5)

    # ---- aux (b): log-log QQ plot ------------------------------------------
    xs = np.sort(x)
    pp = (np.arange(1, xs.size + 1) - 0.5) / xs.size
    for name, d in fits.items():
        ax_q.plot(d.ppf(pp), xs, color=col[name], lw=0.0, marker="o",
                  ms=1.5, alpha=0.75, zorder=4,
                  mfc=col[name], mec="none")
    ref = np.array([xs.min() * 0.8, xs.max() * 1.25])
    ax_q.plot(ref, ref, color=FAINT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=5)
    ax_q.set_xscale("log")
    ax_q.set_yscale("log")
    ax_q.set_xlim(ref[0], ref[1])
    ax_q.set_ylim(ref[0], ref[1])
    ax_q.xaxis.set_major_formatter(FuncFormatter(money))
    ax_q.yaxis.set_major_formatter(FuncFormatter(money))
    ax_q.tick_params(labelsize=6.8)
    ax_q.set_xlabel("fitted quantile (USD)", fontsize=7.8, labelpad=1.5)
    ax_q.set_ylabel("observed quantile (USD)", fontsize=7.8)
    ax_q.set_title("Tail departure on the QQ plot", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_q.text(0.035, 0.955,
              "points above the diagonal:\nthe fit is too light in the tail",
              transform=ax_q.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): mean excess function -------------------------------------
    us = np.percentile(x, np.linspace(2, 96, 45))
    e_u, n_u = mean_excess(x, us)
    ax_m.plot(us, e_u, color=ACCENT, lw=1.5, zorder=5)
    ax_m.plot(us[::4], e_u[::4], ls="none", marker="o", ms=2.6, mfc="white",
              mec=ACCENT, mew=0.9, zorder=6)
    ok = np.isfinite(e_u) & (us > np.percentile(x, 60))
    slope, icept = np.polyfit(us[ok], e_u[ok], 1)
    alpha_tail = 1.0 + 1.0 / slope
    ax_m.plot(us[ok], icept + slope * us[ok], color=SUB, lw=0.9,
              ls=(0, (3.0, 2.0)), zorder=7)
    ax_m.axhline(float(np.mean(x)), color=FAINT, lw=0.8, ls=(0, (2.0, 2.0)),
                 zorder=3)
    ax_m.set_xscale("log")
    ax_m.xaxis.set_major_formatter(FuncFormatter(money))
    ax_m.yaxis.set_major_formatter(FuncFormatter(money))
    ax_m.tick_params(labelsize=6.8)
    ax_m.set_xlabel("threshold $u$ (USD, log scale)", fontsize=7.8,
                    labelpad=1.5)
    ax_m.set_ylabel("mean excess $e(u)$ (USD)", fontsize=7.8)
    ax_m.set_title("Mean excess rises: the tail is heavy", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_m.text(0.035, 0.955,
              f"upward slope {slope:.2f} implies a Pareto\n"
              f"tail index of {alpha_tail:.2f}; a flat line would\n"
              f"mean an exponential tail (dashed grey = mean)",
              transform=ax_m.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_q, ax_m], dx=-0.56, dy=0.10)
    savefig(fig, "884_act_loss_severity_dist.png")


if __name__ == "__main__":
    main()
