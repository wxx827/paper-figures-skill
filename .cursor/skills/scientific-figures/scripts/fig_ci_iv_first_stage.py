"""Figure 772 - Instrumental variables: first stage, reduced form, 2SLS.

Main panel: the first stage.  Individual observations sit behind binned means
with 95 % intervals and an OLS fit with its confidence band, and the inset
repeats the exercise for the reduced form, because the ratio of those two
slopes is the 2SLS estimate - showing one without the other asks the reader
to take the identification on trust.  The stats box carries the first-stage F
against the conventional weak-instrument threshold of 10.

Auxiliary (both at 58 % of the main panel width): OLS against 2SLS with
Wald and Anderson-Rubin intervals, which is where the endogeneity bias
becomes visible; and the same estimate re-computed as the instrument is
weakened, plotted against the first-stage F on a log axis, so the reader can
watch the Anderson-Rubin set blow up exactly where the rule of thumb says it
should.

Every slope, standard error, F statistic and interval endpoint is computed
from the simulated micro data that the points are drawn from.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID, BASE = "#cccccc", "#d9d9d9", "#808080"
CARD = "#f5f5f5"

FS = "#b23a48"           # first stage / 2SLS: the emphasis
RF = "#3b5b92"           # reduced form
OLSC = "#8a8a8a"
AR = "#c9a227"           # Anderson-Rubin, the weak-identification-robust set

N_OBS = 1500
BETA = 0.085             # true return per year of schooling, log units
PI = 0.62                # first-stage strength at the reference design
CHI2_95 = float(stats.chi2.ppf(0.95, 1))
B_GRID = np.linspace(-0.35, 0.45, 1601)     # candidate slopes for the AR set


# ---------------------------------------------------------------------------
def simulate(rng, pi=PI, n=N_OBS, draws=None):
    """One instrument, one endogenous regressor, one omitted confounder."""
    if draws is None:
        z = rng.normal(0.0, 1.0, n)
        u = rng.normal(0.0, 1.0, n)
        ex = rng.normal(0.0, 0.85, n)
        ey = rng.normal(0.0, 0.30, n)
    else:
        z, u, ex, ey = draws
    x = 12.6 + pi * z + 0.95 * u + ex          # years of schooling
    y = 1.90 + BETA * x + 0.33 * u + ey        # log hourly wage
    return z, x, y, (z, u, ex, ey)


def ols(a, b):
    """Slope of ``b`` on ``a`` with intercept: estimate, SE, t, R-squared."""
    ac, bc = a - a.mean(), b - b.mean()
    sxx = float((ac ** 2).sum())
    beta = float((ac * bc).sum() / sxx)
    e = bc - beta * ac
    s2 = float((e ** 2).sum() / (a.size - 2))
    se = float(np.sqrt(s2 / sxx))
    r2 = 1.0 - float((e ** 2).sum()) / float((bc ** 2).sum())
    return beta, se, beta / se, r2


def tsls(z, x, y):
    """Just-identified 2SLS: the ratio of reduced form to first stage."""
    zc, xc, yc = z - z.mean(), x - x.mean(), y - y.mean()
    pi = float((zc * xc).sum() / (zc ** 2).sum())
    xhat = pi * zc
    beta = float((xhat * yc).sum() / (xhat ** 2).sum())
    e = yc - beta * xc
    s2 = float((e ** 2).sum() / (z.size - 2))
    se = float(np.sqrt(s2 / (xhat ** 2).sum()))
    return beta, se


def ar_set(z, x, y, grid=B_GRID):
    """Anderson-Rubin confidence set: valid however weak the instrument is.

    For each candidate slope the structural residual is regressed on the
    instrument; candidates whose residual is still correlated with it are
    rejected.  Unlike a Wald interval this set is allowed to be empty or
    unbounded, which is precisely the honest answer under weak instruments.
    """
    zc = z - z.mean()
    szz = float((zc ** 2).sum())
    r = y[None, :] - grid[:, None] * x[None, :]
    rc = r - r.mean(axis=1, keepdims=True)
    pi = (rc * zc[None, :]).sum(axis=1) / szz
    e = rc - pi[:, None] * zc[None, :]
    s2 = (e ** 2).sum(axis=1) / (z.size - 2)
    stat = pi ** 2 * szz / s2
    ok = stat <= CHI2_95
    if not ok.any():
        return np.nan, np.nan, False
    lo, hi = float(grid[ok][0]), float(grid[ok][-1])
    bounded = not (ok[0] or ok[-1])
    return lo, hi, bounded


def binned(a, b, n_bins=14):
    edges = np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.digitize(a, edges) - 1
    out = []
    for k in range(n_bins):
        m = idx == k
        if m.sum() >= 5:
            out.append((a[m].mean(), b[m].mean(),
                        b[m].std(ddof=1) / np.sqrt(m.sum())))
    return np.array(out)


def fit_band(a, b, col_x):
    """Fitted line and its 95 % confidence band over ``col_x``."""
    beta, se, _, _ = ols(a, b)
    alpha = b.mean() - beta * a.mean()
    ac = a - a.mean()
    sxx = float((ac ** 2).sum())
    e = b - (alpha + beta * a)
    s2 = float((e ** 2).sum() / (a.size - 2))
    band = np.sqrt(s2 * (1.0 / a.size + (col_x - a.mean()) ** 2 / sxx))
    return alpha + beta * col_x, 1.96 * band


# ---------------------------------------------------------------------------
def first_stage_panel(ax, z, x, y, fs, rf):
    grid = np.linspace(z.min(), z.max(), 120)
    ax.scatter(z, x, s=5.0, facecolor=lighten(FS, 0.70), edgecolor="none",
               alpha=0.55, zorder=1, rasterized=True)
    line, band = fit_band(z, x, grid)
    ax.fill_between(grid, line - band, line + band, facecolor=FS, alpha=0.22,
                    edgecolor="none", zorder=3)
    ax.plot(grid, line, lw=2.0, color=FS, zorder=4)
    b = binned(z, x)
    ax.errorbar(b[:, 0], b[:, 1], yerr=1.96 * b[:, 2], fmt="o", ms=4.4,
                color=INK, mfc="white", mec=INK, mew=1.1, ecolor=INK,
                elinewidth=0.9, capsize=2.2, capthick=0.8, zorder=5)

    ax.set_xlim(z.min() - 0.12, z.max() + 0.12)
    ax.set_xlabel("instrument: reform exposure index (SD units)", fontsize=8.0)
    ax.set_ylabel("years of schooling (years)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("First stage: the instrument really does move schooling",
                 fontsize=9.0, fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    pi_hat, pi_se, pi_t, pi_r2 = fs
    f_stat = pi_t ** 2
    ax.text(0.026, 0.965,
            f"first stage  d(schooling)/d(Z) = {pi_hat:.3f} years/SD\n"
            f"SE {pi_se:.3f},  F = {f_stat:.1f}  (rule of thumb F > 10)\n"
            f"partial R2 = {pi_r2:.3f},  n = {N_OBS:,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color=INK, linespacing=1.5)
    ax.text(0.026, 0.775, "open circles: means of 14 equal-count bins "
            "with 95 % intervals", transform=ax.transAxes, ha="left",
            va="top", fontsize=6.5, color=SUB)

    axi = ax.inset_axes([0.615, 0.075, 0.360, 0.335])
    axi.set_zorder(7)
    axi.patch.set_visible(True)
    br = binned(z, y)
    lr, bandr = fit_band(z, y, grid)
    axi.fill_between(grid, lr - bandr, lr + bandr, facecolor=RF, alpha=0.22,
                     edgecolor="none", zorder=2)
    axi.plot(grid, lr, lw=1.5, color=RF, zorder=3)
    axi.errorbar(br[:, 0], br[:, 1], yerr=1.96 * br[:, 2], fmt="o", ms=2.8,
                 color=INK, mfc="white", mec=INK, mew=0.8, ecolor=INK,
                 elinewidth=0.7, capsize=1.4, capthick=0.6, zorder=4)
    axi.set_xlim(z.min() - 0.12, z.max() + 0.12)
    axi.set_xlabel("instrument (SD units)", fontsize=6.2, labelpad=1.5)
    axi.set_ylabel("log hourly wage", fontsize=6.2, labelpad=1.5)
    axi.tick_params(labelsize=6.0, length=1.8, pad=1.2)
    axi.set_title(f"Reduced form: {rf[0]:.4f} log/SD (SE {rf[1]:.4f})",
                  fontsize=6.6, fontweight="bold", pad=2.0, color=RF)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    for s in axi.spines.values():
        s.set_linewidth(0.8)
        s.set_color("#333333")
    axi.grid(axis="y", color=GRID, linewidth=0.5)


def compare_panel(ax, rows):
    ys = np.arange(len(rows))[::-1]
    ax.axvline(100 * BETA, color=BASE, lw=0.8, ls=(0, (2.4, 2.0)), zorder=1)
    ax.text(100 * BETA, len(rows) - 0.40, "design value", ha="center",
            va="bottom", fontsize=6.2, color=MUTED)
    for y, (name, est, lo, hi, col) in zip(ys, rows):
        ax.plot([100 * lo, 100 * hi], [y, y], lw=2.0, color=col,
                solid_capstyle="butt", zorder=3)
        ax.plot([100 * est], [y], marker="o", ms=4.6, color=col, mfc=col,
                mec=col, mew=1.0, zorder=4)
        ax.text(100 * hi, y + 0.26, f"  {100 * est:.1f} "
                f"[{100 * lo:.1f}, {100 * hi:.1f}]", ha="right", va="bottom",
                fontsize=6.2, color=SUB)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.set_xlabel("return to a year of schooling (log points)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_title("OLS is biased upward; 2SLS is not", fontsize=8.5,
                 fontweight="bold", pad=3.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def strength_panel(ax, fs, est, wald, ar_lo, ar_hi, bounded, ylim):
    ax.fill_between(fs, 100 * ar_lo, 100 * ar_hi, facecolor=AR, alpha=0.24,
                    edgecolor="none", zorder=1, label="Anderson-Rubin set")
    ax.fill_between(fs, 100 * (est - 1.96 * wald), 100 * (est + 1.96 * wald),
                    facecolor=FS, alpha=0.26, edgecolor="none", zorder=2,
                    label="2SLS Wald interval")
    ax.plot(fs, 100 * est, lw=1.6, color=FS, marker="o", ms=3.0, mew=0.0,
            zorder=4, label="2SLS estimate")
    ax.axhline(100 * BETA, color=BASE, lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    ax.axvline(10.0, color="#3a3a3a", lw=0.9, ls=(0, (2.6, 2.0)), zorder=3)

    for f, ok in zip(fs, bounded):
        if not ok:
            ax.plot([f], [ylim[1]], marker="^", ms=3.4, color=AR,
                    clip_on=False, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(fs[0] * 0.75, fs[-1] * 1.35)
    ax.set_ylim(*ylim)
    ax.text(10.0, ylim[0] + 0.045 * (ylim[1] - ylim[0]), " F = 10",
            ha="left", va="bottom", fontsize=6.3, color=SUB)
    ax.text(0.985, 0.965, "carets: unbounded AR set", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=MUTED)
    ax.set_xlabel("first-stage F statistic (-)", fontsize=7.4)
    ax.set_ylabel("estimate (log points)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, length=2.2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_title("Weak instruments widen the honest interval",
                 fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.3, frameon=False,
              handlelength=1.2, borderaxespad=0.3, labelspacing=0.26)


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(772)
    z, x, y, draws = simulate(rng)

    fs = ols(z, x)
    rf = ols(z, y)
    b_ols = ols(x, y)
    b_iv, se_iv = tsls(z, x, y)
    lo_ar, hi_ar, _ = ar_set(z, x, y)

    rows = [
        ("OLS", b_ols[0], b_ols[0] - 1.96 * b_ols[1],
         b_ols[0] + 1.96 * b_ols[1], OLSC),
        ("2SLS, Wald CI", b_iv, b_iv - 1.96 * se_iv, b_iv + 1.96 * se_iv, FS),
        ("2SLS, Anderson-Rubin", b_iv, lo_ar, hi_ar, AR),
    ]

    # Sweep the instrument strength on the *same* draws, so only pi changes.
    pis = np.array([0.045, 0.065, 0.09, 0.13, 0.19, 0.28, 0.40, 0.62, 0.90])
    f_stat, est, wald, a_lo, a_hi, bnd = [], [], [], [], [], []
    for p in pis:
        zz, xx, yy, _ = simulate(rng, pi=p, draws=draws)
        f_stat.append(ols(zz, xx)[2] ** 2)
        b, s = tsls(zz, xx, yy)
        est.append(b)
        wald.append(s)
        lo, hi, ok = ar_set(zz, xx, yy)
        a_lo.append(lo)
        a_hi.append(hi)
        bnd.append(ok)
    f_stat = np.array(f_stat)
    est, wald = np.array(est), np.array(wald)
    a_lo, a_hi = np.array(a_lo), np.array(a_hi)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.82, 5.24, 4.52))
    ax_cmp = axes_inches(fig, (7.05, 3.48, 3.05, 1.86))
    ax_str = axes_inches(fig, (7.05, 0.82, 3.05, 1.86))

    first_stage_panel(ax_main, z, x, y, fs, rf)
    compare_panel(ax_cmp, rows)
    strength_panel(ax_str, f_stat, est, wald, a_lo, a_hi, bnd,
                   (100 * B_GRID[0] * 0.62, 100 * B_GRID[-1] * 0.62))


    for ax, lab in ((ax_main, "(a)"), (ax_cmp, "(b)"), (ax_str, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.10)
    savefig(fig, "772_ci_iv_first_stage.png")


if __name__ == "__main__":
    main()
