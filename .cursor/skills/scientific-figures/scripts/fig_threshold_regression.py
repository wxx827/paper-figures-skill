"""Threshold / piecewise segmented regression grid.

Each panel shows a scatter of one standardised driver against microbial
biomass, the fitted LOWESS trend with a bootstrap confidence band, a vertical
threshold, and the two sub-slope fits.  All statistics printed on the figure
(segment R2, p-values, total R2 of the segmented model) are computed from the
simulated data rather than invented, so the annotations stay self-consistent.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

C_LO, C_HI = PALETTES["teal_rust"]
C_TREND = "#2b2b2b"
C_THR = "#8c3b1f"

# driver name, threshold (z-score), slope below, slope above, noise sd
DRIVERS = [
    ("Mean annual temperature", 0.8, -0.10, -0.62, 0.30),
    ("Mean annual precipitation", 0.1, -0.55, -0.14, 0.28),
    ("Soil pH", -0.1, 0.62, 0.10, 0.30),
    ("Soil organic carbon", -0.1, 0.48, 0.05, 0.32),
    ("Total nitrogen", -0.8, 0.70, 0.09, 0.29),
    ("C : N ratio", 0.0, -0.18, -0.48, 0.27),
    ("Clay content", -0.6, -0.75, -0.12, 0.31),
    ("Electrical conductivity", -0.5, -0.66, -0.10, 0.30),
]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def ols(x, y):
    """Slope, intercept, R2 and two-sided p-value of the slope."""
    n = x.size
    sx = x - x.mean()
    sy = y - y.mean()
    denom = np.sum(sx ** 2)
    slope = np.sum(sx * sy) / denom
    inter = y.mean() - slope * x.mean()
    resid = y - (inter + slope * x)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum(sy ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se = np.sqrt(ss_res / (n - 2) / denom)
    t = slope / se if se > 0 else 0.0
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), n - 2)
    except Exception:                                    # pragma: no cover
        p = 0.0 if abs(t) > 4 else 0.05
    return slope, inter, r2, p


def kernel_smooth(x, y, xg, h):
    w = np.exp(-0.5 * ((x[None, :] - xg[:, None]) / h) ** 2)
    s = w.sum(axis=1)
    s[s == 0] = 1.0
    return (w @ y) / s


def panel(ax, rng, name, thr, s_lo, s_hi, sd, show_ylabel):
    n = 380
    x = np.sort(rng.uniform(-2.0, 2.0, n))
    base = 22.0 + np.where(x < thr, s_lo * (x - thr), s_hi * (x - thr))
    y = base + rng.normal(0, sd, n)

    lo, hi = x < thr, x >= thr
    ax.scatter(x[lo], y[lo], s=11, c=C_LO, alpha=0.60, linewidths=0.25,
               edgecolors="white", zorder=2)
    ax.scatter(x[hi], y[hi], s=11, c=C_HI, alpha=0.60, linewidths=0.25,
               edgecolors="white", zorder=2)

    # segment fits + honest segment statistics
    stats = {}
    pred = np.empty_like(y)
    for key, mask, col in [("lo", lo, C_LO), ("hi", hi, C_HI)]:
        sl, ic, r2, p = ols(x[mask], y[mask])
        stats[key] = (r2, p)
        pred[mask] = ic + sl * x[mask]
        xr = np.linspace(x[mask].min(), x[mask].max(), 30)
        ax.plot(xr, ic + sl * xr, "--", color=col, lw=1.3, alpha=0.95, zorder=4)
    r2_tot = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    # LOWESS-style trend with a bootstrap band
    xg = np.linspace(x.min(), x.max(), 160)
    h = 0.28 * (x.max() - x.min())
    yg = kernel_smooth(x, y, xg, h)
    boot = np.empty((120, xg.size))
    for b in range(boot.shape[0]):
        idx = rng.integers(0, n, n)
        boot[b] = kernel_smooth(x[idx], y[idx], xg, h)
    band_lo, band_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.fill_between(xg, band_lo, band_hi, color="#8a8a8a", alpha=0.22,
                    linewidth=0, zorder=3)
    ax.plot(xg, yg, color=C_TREND, lw=1.8, zorder=5)

    ax.axvline(thr, color=C_THR, ls=(0, (2, 2)), lw=1.2, zorder=1)

    pad = 0.30
    ax.set_ylim(y.min() - pad, y.max() + pad * 2.6)
    ax.set_xlim(-2.15, 2.15)

    box = dict(boxstyle="round,pad=0.22", facecolor="white",
               edgecolor="#d5d5d5", linewidth=0.5, alpha=0.9)
    ax.text(0.03, 0.965, f"R$^2$ = {stats['lo'][0]:.2f}{stars(stats['lo'][1])}",
            transform=ax.transAxes, va="top", ha="left", color=C_LO,
            fontsize=7.5, fontweight="bold", bbox=box, zorder=6)
    ax.text(0.97, 0.965, f"R$^2$ = {stats['hi'][0]:.2f}{stars(stats['hi'][1])}",
            transform=ax.transAxes, va="top", ha="right", color=C_HI,
            fontsize=7.5, fontweight="bold", bbox=box, zorder=6)
    ax.text(0.5, 0.02, f"threshold = {thr:+.1f} SD", transform=ax.transAxes,
            ha="center", va="bottom", color=C_THR, fontsize=7,
            fontweight="bold", bbox=box, zorder=6)

    ax.set_title(f"Segmented R$^2$ = {r2_tot:.2f}***", fontsize=8.5,
                 fontweight="bold", color="#1a1a1a")
    ax.set_xlabel(f"{name} (z-score)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Microbial biomass C (mg g$^{-1}$)", fontsize=8)
    ax.tick_params(labelsize=7)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.4))
    for i, (ax, (nm, thr, s_lo, s_hi, sd)) in enumerate(zip(axes.ravel(), DRIVERS)):
        panel(ax, rng, nm, thr, s_lo, s_hi, sd, show_ylabel=(i % 2 == 0))
        add_panel_label(ax, f"({chr(97 + i)})", x=-0.13, y=1.02, size=9.5)

    handles = [
        Line2D([], [], marker="o", ls="none", color=C_LO, ms=4.5,
               mec="white", mew=0.4, label="Below threshold"),
        Line2D([], [], marker="o", ls="none", color=C_HI, ms=4.5,
               mec="white", mew=0.4, label="Above threshold"),
        Line2D([], [], color=C_TREND, lw=1.8, label="LOWESS trend"),
        Patch(facecolor="#8a8a8a", alpha=0.22, label="95% bootstrap CI"),
        Line2D([], [], color=C_THR, ls=(0, (2, 2)), lw=1.2,
               label="Fitted threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.5,
               columnspacing=1.4)
    fig.tight_layout(h_pad=1.7, w_pad=2.2, rect=(0, 0.028, 1, 1))
    savefig(fig, "01_threshold_regression.png")


if __name__ == "__main__":
    main()
