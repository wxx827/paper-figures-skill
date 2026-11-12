"""Spline exposure-response curve with the exposure distribution beneath it.

A cubic B-spline logistic model is fitted to simulated individual outcomes, and
the relative risk is plotted against the reference exposure with a 95%
confidence band from the model covariance. The threshold (lowest exposure whose
lower bound clears RR = 1) and the inflection of the fitted curve are both read
off the fit. The lower panel shows where the population actually sits on the
exposure axis, on a shared x-axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#134a76"
BAND = "#cfe1ee"
BAR = "#9cc3dd"
ALERT = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))

N = 20000
REF = 5.0                      # reference exposure, ug/m3
LO, HI = 3.0, 34.0
BASE_RISK = 0.08


def truth(x):
    return 0.85 / (1.0 + np.exp(-(x - 17.0) / 4.0))


def basis_factory(x_all, n_internal=4):
    q = np.linspace(0, 1, n_internal + 2)[1:-1]
    inner = np.quantile(x_all, q)
    t = np.r_[[LO] * 4, inner, [HI] * 4]
    return lambda x: BSpline.design_matrix(
        np.clip(x, LO, HI), t, 3).toarray()


def irls(B, y, n_iter=40):
    beta = np.zeros(B.shape[1])
    w = None
    for _ in range(n_iter):
        eta = B @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        z = eta + (y - p) / w
        A = B.T @ (w[:, None] * B)
        beta = np.linalg.solve(A + 1e-8 * np.eye(B.shape[1]), B.T @ (w * z))
    cov = np.linalg.inv(B.T @ (w[:, None] * B) + 1e-8 * np.eye(B.shape[1]))
    return beta, cov


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(17)

    x = np.clip(rng.lognormal(np.log(13.0), 0.40, N), LO, HI)
    p = np.clip(BASE_RISK * np.exp(truth(x) - truth(REF)), 0, 0.95)
    y = (rng.random(N) < p).astype(float)

    basis = basis_factory(x)
    beta, cov = irls(basis(x), y)

    xg = np.linspace(REF - 0.5, 31.0, 400)      # drawn over the populated range
    C = basis(xg) - basis(np.array([REF]))
    log_rr = C @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", C, cov, C))
    rr, rr_lo, rr_hi = np.exp(log_rr), np.exp(log_rr - 1.96 * se), \
        np.exp(log_rr + 1.96 * se)

    # threshold: first exposure above the reference whose lower bound clears 1
    right = xg > REF
    sig = np.flatnonzero(right & (rr_lo > 1.0))
    x_thr = float(xg[sig[0]]) if sig.size else np.nan
    # inflection: sign change of the second derivative of the fitted log-RR
    d2 = np.gradient(np.gradient(log_rr, xg), xg)
    flips = np.flatnonzero(np.diff(np.sign(d2)) != 0)
    flips = flips[(xg[flips] > REF + 2) & (xg[flips] < HI - 4)]
    x_inf = float(xg[flips[np.argmax(np.abs(np.gradient(log_rr, xg))[flips])]]) \
        if flips.size else np.nan

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.2, 4.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 0.34], hspace=0.19,
                         left=0.088, right=0.985, top=0.845, bottom=0.115))

    # ---- (a) exposure-response ---------------------------------------------
    ax.fill_between(xg, rr_lo, rr_hi, color=BAND, lw=0, zorder=2,
                    label="95% confidence band")
    ax.plot(xg, rr, color=DEEP, lw=1.9, zorder=5, label="spline estimate")
    ax.axhline(1.0, color=RULE, ls=DASH, lw=0.9, zorder=3)
    ax.text(31.0, 1.07, "no excess risk, RR = 1", fontsize=6.6,
            color=RULE, ha="right", va="bottom")
    ax.axvline(REF, color=MUTED, ls=DASH, lw=0.8, zorder=3)
    ax.text(REF + 0.5, 0.025, f"reference {REF:.0f} " + r"$\mu$g/m$^3$",
            transform=ax.get_xaxis_transform(), fontsize=6.6, color=MUTED,
            ha="left", va="bottom")

    if np.isfinite(x_thr):
        k = int(np.argmin(np.abs(xg - x_thr)))
        ax.plot([x_thr], [rr[k]], marker="o", ms=4.6, mfc="white", mec=ALERT,
                mew=1.3, zorder=7)
        ax.annotate(f"threshold {x_thr:.1f} " + r"$\mu$g/m$^3$"
                    + f"\nlower bound clears 1 (RR = {rr[k]:.2f})",
                    xy=(x_thr, rr[k]), xytext=(x_thr - 10.5, rr[k] + 0.80),
                    fontsize=6.6, color=ALERT, linespacing=1.35, ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=ALERT))
    if np.isfinite(x_inf):
        k = int(np.argmin(np.abs(xg - x_inf)))
        ax.plot([x_inf], [rr[k]], marker="s", ms=4.2, mfc="white", mec=DEEP,
                mew=1.2, zorder=7)
        ax.annotate(f"inflection {x_inf:.1f} " + r"$\mu$g/m$^3$",
                    xy=(x_inf, rr[k]), xytext=(x_inf + 7.0, 0.86),
                    fontsize=6.6, color=DEEP, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP))

    ax.set_xlim(4.0, 32.0)
    ax.set_ylim(0.70, min(3.3, rr_hi.max() * 1.06))
    ax.set_ylabel("Relative risk vs reference (RR)")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False, handlelength=1.5,
              borderaxespad=0.4)
    style_axes(ax)
    add_panel_label(ax, "(a)", x=-0.078, y=1.04)

    # ---- (b) exposure distribution -----------------------------------------
    axb.hist(x, bins=np.arange(LO, HI + 0.5, 0.75), color=BAR,
             edgecolor="white", linewidth=0.3, zorder=3)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    axb.axvline(med, color=DEEP, lw=1.1, zorder=5)
    axb.axvline(REF, color=MUTED, ls=DASH, lw=0.8, zorder=4)
    ymax = axb.get_ylim()[1]
    axb.annotate(f"median {med:.1f}, IQR {q1:.1f}-{q3:.1f} "
                 + r"$\mu$g/m$^3$",
                 xy=(med, ymax * 0.92), xytext=(med + 1.6, ymax * 0.92),
                 fontsize=6.6, color=DEEP, va="center", ha="left")
    share = 100.0 * float((x > x_thr).mean()) if np.isfinite(x_thr) else np.nan
    axb.text(0.985, 0.88, f"{share:.0f}% of the cohort sits above the threshold",
             transform=axb.transAxes, ha="right", va="top", fontsize=6.6,
             color=MUTED)
    axb.set_ylabel("Count (n)")
    axb.set_xlabel(r"Annual mean PM$_{2.5}$ exposure ($\mu$g/m$^3$)")
    axb.tick_params(labelsize=7.2)
    style_axes(axb)
    add_panel_label(axb, "(b)", x=-0.078, y=1.06)

    savefig(fig, "431_epi_exposure_response.png")


if __name__ == "__main__":
    main()
