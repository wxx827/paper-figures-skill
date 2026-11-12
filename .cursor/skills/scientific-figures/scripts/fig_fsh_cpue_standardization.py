"""Why a nominal catch rate has to be standardised before it is believed.

One synthetic haul-by-haul data set drives the whole page.  Each record gets a
year, a quarter, a region and a vessel class, and its log catch rate is the sum
of those effects plus lognormal noise.  The trick is in the *effort*: the fleet
migrates from the shelf onto the productive offshore ground over the record, so
the arithmetic mean of the raw rates drifts upward even though the year term
that generated the data is falling.  That is the fake trend the figure exists
to kill.

Panel (a) plots the nominal mean against the standardised index.  The
standardised series is a set of least-squares means from one OLS fit of
log CPUE on year + quarter + region + vessel, evaluated with the non-year
factors pinned at their overall observed proportions, so the fleet's changing
composition can no longer move it.  The band is a proper contrast interval:
for each year a contrast vector c is built and the variance is
sigma^2 * c' (X'X)^-1 c, back-transformed with the same exponential as the
point estimate, rather than a fixed percentage of it.

Panels (b) and (c) are the diagnostics that decide whether panel (a) may be
believed: multiplicative effect sizes exp(beta) per factor level with the
sequential deviance each factor explains, and a normal QQ plot of the
standardised residuals.  Illustrative values; this is a layout reference.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY = "#808080"
STD, NOM, PRESS = "#2a7f7a", "#c9a227", "#b23a48"
BLUE = "#3b5b92"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.04, 4.86, 4.16)
AUX1 = (6.62, 3.14, 2.78, 2.06)
AUX2 = (6.62, 1.04, 2.78, 1.44)

YEAR0, YEAR1 = 1995, 2024
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
REGIONS = ["shelf", "slope", "offshore"]
VESSELS = ["small", "medium", "large"]
B_Q = np.array([0.00, 0.18, 0.34, 0.11])      # log effects, first level = ref
B_R = np.array([0.00, 0.42, 0.78])
B_V = np.array([0.00, 0.22, 0.40])
SIGMA = 0.34


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_records(rng):
    """Haul records whose region mix drifts while the true year term falls."""
    years = np.arange(YEAR0, YEAR1 + 1)
    t = (years - years[0]) / (years[-1] - years[0])
    true_year = (np.log(0.62) * logistic((t - 0.42) / 0.20)
                 + rng.normal(0.0, 0.035, years.size))
    true_year -= true_year[0]

    n_haul = np.rint(150.0 + 90.0 * logistic((t - 0.35) / 0.18)
                     + rng.normal(0.0, 9.0, years.size)).astype(int)

    yi, qi, ri, vi, logc = [], [], [], [], []
    for k, _ in enumerate(years):
        n = int(n_haul[k])
        w_r = np.array([0.70 - 0.52 * t[k] ** 0.8,
                        0.22 + 0.06 * t[k],
                        0.08 + 0.46 * t[k] ** 0.8])
        w_r = w_r / w_r.sum()
        w_v = np.array([0.52 - 0.30 * t[k], 0.33, 0.15 + 0.30 * t[k]])
        w_v = w_v / w_v.sum()
        q = rng.integers(0, 4, size=n)
        r = rng.choice(3, size=n, p=w_r)
        v = rng.choice(3, size=n, p=w_v)
        lc = (np.log(0.46) + true_year[k] + B_Q[q] + B_R[r] + B_V[v]
              + rng.normal(0.0, SIGMA, n))
        yi.append(np.full(n, k))
        qi.append(q)
        ri.append(r)
        vi.append(v)
        logc.append(lc)
    return dict(year=years, t=t, true_year=true_year, n_haul=n_haul,
                yi=np.concatenate(yi), qi=np.concatenate(qi),
                ri=np.concatenate(ri), vi=np.concatenate(vi),
                logc=np.concatenate(logc),
                effort=n_haul * 2.4)


def dummies(idx, n_lev):
    """Treatment-coded columns, first level dropped as the reference."""
    d = np.zeros((idx.size, n_lev - 1))
    for j in range(1, n_lev):
        d[:, j - 1] = (idx == j)
    return d


def fit_glm(rec):
    """OLS on log CPUE; returns least-squares means and their contrast SEs."""
    ny = rec["year"].size
    x_y = dummies(rec["yi"], ny)
    x_q = dummies(rec["qi"], 4)
    x_r = dummies(rec["ri"], 3)
    x_v = dummies(rec["vi"], 3)
    one = np.ones((rec["yi"].size, 1))
    X = np.hstack([one, x_y, x_q, x_r, x_v])
    y = rec["logc"]

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = y.size - np.linalg.matrix_rank(X)
    sigma2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)

    # Non-year factors pinned at their overall observed proportions, so a
    # change in fleet composition cannot move the index.
    p_q = x_q.mean(axis=0)
    p_r = x_r.mean(axis=0)
    p_v = x_v.mean(axis=0)

    idx = np.empty(ny)
    lo = np.empty(ny)
    hi = np.empty(ny)
    for k in range(ny):
        c = np.zeros(X.shape[1])
        c[0] = 1.0
        if k > 0:
            c[k] = 1.0
        c[ny:ny + 3] = p_q
        c[ny + 3:ny + 5] = p_r
        c[ny + 5:ny + 7] = p_v
        mu = float(c @ beta)
        se = float(np.sqrt(sigma2 * (c @ xtx_inv @ c)))
        idx[k] = np.exp(mu)
        lo[k] = np.exp(mu - 1.96 * se)
        hi[k] = np.exp(mu + 1.96 * se)

    std_res = resid / np.sqrt(sigma2)
    return dict(beta=beta, index=idx, lo=lo, hi=hi, sigma=np.sqrt(sigma2),
                resid=std_res, ny=ny,
                b_q=beta[ny:ny + 3], b_r=beta[ny + 3:ny + 5],
                b_v=beta[ny + 5:ny + 7],
                b_year=np.concatenate([[0.0], beta[1:ny]]),
                blocks=[("year", x_y), ("quarter", x_q), ("region", x_r),
                        ("vessel", x_v)])


def sequential_deviance(rec, fit):
    """Type-I deviance explained: R2 increment as each factor block enters."""
    y = rec["logc"]
    sst = float(np.sum((y - y.mean()) ** 2))
    X = np.ones((y.size, 1))
    prev = 0.0
    out = []
    for name, blk in fit["blocks"]:
        X = np.hstack([X, blk])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        r2 = 1.0 - float(r @ r) / sst
        out.append((name, r2 - prev))
        prev = r2
    return out, prev


def draw_index(ax, rec, fit):
    yr = rec["year"]
    nominal = np.array([np.exp(rec["logc"][rec["yi"] == k]).mean()
                        for k in range(yr.size)])

    axe = ax.twinx()
    axe.bar(yr, rec["effort"], width=0.74, color=lighten(PRESS, 0.72),
            edgecolor="white", linewidth=0.3, zorder=1)
    axe.set_ylim(0.0, rec["effort"].max() * 3.15)
    axe.set_ylabel("effort (10$^3$ hooks)", fontsize=7.8, labelpad=1.5,
                   color=darken(PRESS, 0.05))
    axe.tick_params(labelsize=6.8, pad=1.5, colors=darken(PRESS, 0.05))
    axe.spines["top"].set_visible(False)
    axe.spines["right"].set_visible(True)
    axe.spines["right"].set_color(darken(PRESS, 0.05))
    axe.spines["right"].set_linewidth(0.8)
    axe.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax.fill_between(yr, fit["lo"], fit["hi"], color=lighten(STD, 0.78),
                    edgecolor="none", zorder=3)
    ax.plot(yr, nominal, color=NOM, lw=1.7, marker="o", ms=3.2,
            markeredgecolor="white", markeredgewidth=0.35, zorder=6)
    ax.plot(yr, fit["index"], color=STD, lw=1.9, marker="s", ms=3.2,
            markeredgecolor="white", markeredgewidth=0.35, zorder=7)

    k0, k1 = 3, yr.size - 3
    d_nom = nominal[k1:].mean() / nominal[:k0].mean() - 1.0
    d_std = fit["index"][k1:].mean() / fit["index"][:k0].mean() - 1.0
    ax.annotate("nominal drifts {:+.0%}\nwith the fleet".format(d_nom),
                xy=(yr[-4], nominal[-4]), xytext=(-6, 26),
                textcoords="offset points", fontsize=6.6,
                color=darken(NOM, 0.22), ha="right", va="bottom", zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=darken(NOM, 0.2),
                                shrinkA=0.0, shrinkB=2.0))
    ax.annotate("standardised {:+.0%}".format(d_std),
                xy=(yr[-4], fit["index"][-4]), xytext=(-4, -24),
                textcoords="offset points", fontsize=6.6,
                color=darken(STD, 0.15), ha="right", va="top", zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=darken(STD, 0.2),
                                shrinkA=0.0, shrinkB=2.0))

    ax.set_xlim(yr[0] - 1.0, yr[-1] + 1.0)
    ax.set_ylim(0.0, max(nominal.max(), fit["hi"].max()) * 1.42)
    ax.set_xlabel("year", fontsize=8.0)
    ax.set_ylabel("catch rate (kg/hook)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Nominal versus standardised catch rate, with annual effort",
                 fontsize=8.5, fontweight="bold")

    handles = [
        Line2D([], [], color=NOM, lw=1.7, marker="o", ms=3.2,
               label="nominal mean CPUE (kg/hook)"),
        Line2D([], [], color=STD, lw=1.9, marker="s", ms=3.2,
               label="standardised index, GLM least-squares means"),
        Line2D([], [], color=lighten(STD, 0.78), lw=5.0,
               label="95% contrast interval"),
        Line2D([], [], color=lighten(PRESS, 0.72), lw=5.0,
               label="effort (10$^3$ hooks, right axis)")]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.6,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    handlelength=1.8, labelspacing=0.34)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(12)
    return nominal, d_nom, d_std


def draw_effects(ax, fit, dev):
    groups = [("quarter", QUARTERS, np.concatenate([[0.0], fit["b_q"]]), BLUE),
              ("region", REGIONS, np.concatenate([[0.0], fit["b_r"]]), STD),
              ("vessel", VESSELS, np.concatenate([[0.0], fit["b_v"]]), PRESS)]
    dev_map = dict(dev)

    # Level names live *inside* the axes, left of the exp(beta) = 1 rule, so
    # the panel needs no tick-label gutter and cannot crowd the main panel's
    # right-hand effort axis.
    labels, vals, cols, ticks = [], [], [], []
    y = 0.0
    heads = []
    for name, lev, coef, col in groups:
        heads.append((y - 0.92, name, col))
        for j, lv in enumerate(lev):
            labels.append(lv + (" (ref)" if j == 0 else ""))
            vals.append(float(np.exp(coef[j])))
            cols.append(lighten(col, 0.30 if j else 0.66))
            ticks.append(y)
            y += 1.0
        y += 1.05

    vals = np.array(vals)
    x_lo, x_hi = 0.42, float(vals.max()) * 1.16
    ax.barh(ticks, vals - 1.0, left=1.0, height=0.62, color=cols,
            edgecolor="white", linewidth=0.4, zorder=4)
    ax.axvline(1.0, color=GREY, lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    for yy, v, lab in zip(ticks, vals, labels):
        ax.text(x_lo + 0.02, yy, lab, fontsize=6.2, color=INK, ha="left",
                va="center", zorder=6)
        ax.text(v + 0.020, yy, "{:.2f}x".format(v), fontsize=6.2, color=SUB,
                ha="left", va="center", zorder=6)
    for yy, name, col in heads:
        ax.text(x_lo + 0.02, yy, "{:s}  -  {:.0%} of deviance"
                .format(name, dev_map[name]), fontsize=6.4,
                color=darken(col, 0.20), ha="left", va="center",
                fontweight="bold", zorder=6)

    ax.set_yticks([])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(max(ticks) + 0.9, -1.9)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("multiplicative effect on CPUE, exp($\\beta$) (-)",
                  fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.spines["left"].set_visible(False)
    ax.set_title("GLM factor effects and deviance explained", fontsize=8.5,
                 fontweight="bold")


def draw_qq(ax, fit):
    r = np.sort(fit["resid"])
    n = r.size
    q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    corr = float(np.corrcoef(q, r)[0, 1])
    lim = 3.9
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.9, ls=(0, (3.4, 2.2)),
            zorder=3)
    ax.scatter(q, r, s=4.0, facecolors=lighten(BLUE, 0.42),
               edgecolors="none", alpha=0.75, zorder=4)
    ax.text(0.035, 0.95,
            "n = {:d} hauls\nresidual s.d. = {:.3f} (log)\n"
            "QQ correlation = {:.4f}".format(n, fit["sigma"], corr),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=SUB, linespacing=1.40, zorder=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("theoretical normal quantile (-)", fontsize=7.6)
    ax.set_ylabel("standardised residual (-)", fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title("Residual QQ plot of the log-CPUE model", fontsize=8.5,
                 fontweight="bold")
    return corr


def main():
    set_pub_style()
    rng = np.random.default_rng(9020)
    rec = build_records(rng)
    fit = fit_glm(rec)
    dev, total_r2 = sequential_deviance(rec, fit)

    fig = plt.figure(figsize=(FW, FH))
    ax_i = axes_inches(fig, MAIN)
    ax_e = axes_inches(fig, AUX1)
    ax_q = axes_inches(fig, AUX2)

    nominal, d_nom, d_std = draw_index(ax_i, rec, fit)
    draw_effects(ax_e, fit, dev)
    corr = draw_qq(ax_q, fit)

    panel_labels_inches(fig, [ax_i, ax_e, ax_q], dx=-0.46, dy=0.20)

    off_share0 = float(np.mean(rec["ri"][rec["yi"] < 3] == 2))
    off_share1 = float(np.mean(rec["ri"][rec["yi"] >= rec["year"].size - 3] == 2))
    dev_txt = ", ".join("{:s} {:.0%}".format(k, v) for k, v in dev)


    print("n={:d} d_nom={:+.3f} d_std={:+.3f} r2={:.3f} sigma={:.3f} "
          "qq={:.4f}".format(rec["logc"].size, d_nom, d_std, total_r2,
                             fit["sigma"], corr))
    savefig(fig, "902_fsh_cpue_standardization.png")


if __name__ == "__main__":
    main()
