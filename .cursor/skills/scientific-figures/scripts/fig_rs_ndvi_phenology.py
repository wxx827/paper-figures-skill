"""NDVI time-series reconstruction and land-surface phenology retrieval.

Panel (a) one full year of 8-day NDVI composites with cloud-contaminated
outliers, an iterative Savitzky-Golay upper-envelope reconstruction and a
double-logistic model fit, annotated with SOS / POS / EOS / LOS.
Panel (b) fitted phenology curves for four crop types.
Panels (c)-(d) inter-annual phenology anomalies and growing-season length.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label


# --------------------------------------------------------------------------
def dbl_logistic(t, vmin, amp, m1, s1, m2, s2):
    """Classic Zhang et al. (2003) double-logistic phenology model."""
    return vmin + amp * (1.0 / (1.0 + np.exp(-m1 * (t - s1)))
                         - 1.0 / (1.0 + np.exp(-m2 * (t - s2))))


def sg_upper_envelope(y, window=9, poly=3, n_iter=4):
    """Chen et al. (2004) iterative SG filter: keep the upper envelope so that
    cloud-driven negative outliers are progressively rejected."""
    z = y.copy()
    for _ in range(n_iter):
        s = savgol_filter(z, window, poly, mode="nearest")
        z = np.maximum(z, s)
    return savgol_filter(z, window, poly, mode="nearest")


def phenology_metrics(t, curve, frac=0.2):
    """20 % amplitude thresholds on the rising / falling limbs."""
    base, peak = curve.min(), curve.max()
    thr = base + frac * (peak - base)
    ipos = int(np.argmax(curve))
    rise = np.where(curve[:ipos] < thr)[0]
    fall = np.where(curve[ipos:] < thr)[0]
    sos = t[rise[-1] + 1] if rise.size else t[0]
    eos = t[ipos + fall[0]] if fall.size else t[-1]
    return sos, t[ipos], eos


# --------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240183)

    green = "#2e7d5b"
    dark = "#0b3d2e"
    rust = "#c0603f"
    blue = "#3b5b92"

    # ---- (a) one-year observation series ---------------------------------
    doy = 1 + 8 * np.arange(46)
    truth = dbl_logistic(doy.astype(float), 0.16, 0.66, 0.115, 122.0, 0.095, 281.0)
    obs = truth + rng.normal(0, 0.018, doy.size)
    cloud = rng.random(doy.size) < 0.24
    cloud[:3] = False
    obs[cloud] -= rng.uniform(0.09, 0.42, cloud.sum())
    obs = np.clip(obs, 0.02, 0.95)

    sg = sg_upper_envelope(obs, window=9, poly=3, n_iter=5)
    p0 = [0.15, 0.65, 0.1, 120.0, 0.1, 280.0]
    popt, _ = curve_fit(dbl_logistic, doy.astype(float), sg, p0=p0, maxfev=40000)
    tg = np.linspace(1, 365, 730)
    fit = dbl_logistic(tg, *popt)
    sos, pos, eos = phenology_metrics(tg, fit)
    los = eos - sos

    fig = plt.figure(figsize=(7.2, 5.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.55, 1.0],
                  width_ratios=[1.0, 1.0, 0.92],
                  hspace=0.52, wspace=0.34)

    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor("#fcfcfa")
    ax.axhspan(0.0, 1.0, color="none")

    # growing-season shading
    ax.axvspan(sos, eos, color=green, alpha=0.07, lw=0)

    ax.scatter(doy[~cloud], obs[~cloud], s=15, facecolor="white",
               edgecolor=dark, linewidth=0.7, zorder=3,
               label="Clear-sky observation")
    ax.scatter(doy[cloud], obs[cloud], s=16, marker="v", color=blue,
               alpha=0.75, linewidth=0, zorder=3,
               label="Cloud / snow contaminated")
    ax.plot(doy, sg, color=rust, lw=1.3, ls="--", zorder=4,
            label="S-G upper envelope")
    ax.plot(tg, fit, color=dark, lw=1.9, zorder=5, label="Double-logistic fit")

    ymax = 1.06
    ax.set_ylim(0.0, ymax)
    ax.set_xlim(0, 366)
    for x, lab, col in ((sos, "SOS", green), (pos, "POS", "#b8860b"),
                        (eos, "EOS", rust)):
        ax.axvline(x, color=col, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
        ax.text(x, 0.965 * ymax, f" {lab}\n DOY {x:.0f}", color=col,
                fontsize=7, ha="left" if lab != "EOS" else "right",
                va="top", fontweight="bold", linespacing=1.15)

    yarr = 0.115
    ax.annotate("", xy=(sos, yarr), xytext=(eos, yarr),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=0.9))
    ax.text(0.5 * (sos + eos), yarr + 0.032,
            f"LOS = {los:.0f} d", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color="#1a1a1a")

    ax.set_xlabel("Day of year (DOY)")
    ax.set_ylabel("NDVI (dimensionless)")
    ax.set_xticks(np.arange(0, 366, 60))
    ax.set_title("Cropland pixel, 8-day MODIS-like composites (2023)",
                 fontsize=8.5)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.235), ncol=4,
              fontsize=6.6, frameon=False, handletextpad=0.45,
              columnspacing=1.4, borderaxespad=0.0)
    add_panel_label(ax, "(a)", x=-0.085, y=1.03, size=10)

    # ---- (b) crop-type comparison ----------------------------------------
    axb = fig.add_subplot(gs[0, 2])
    crops = [
        ("Winter wheat", 0.18, 0.60, 0.055, 72.0, 0.070, 168.0, "#3b5b92"),
        ("Maize", 0.14, 0.72, 0.075, 158.0, 0.085, 262.0, "#2e7d5b"),
        ("Soybean", 0.15, 0.61, 0.065, 172.0, 0.075, 270.0, "#c9a227"),
        ("Paddy rice", 0.12, 0.68, 0.070, 148.0, 0.065, 276.0, "#8c2f2f"),
    ]
    for name, vmin, amp, m1, s1, m2, s2, col in crops:
        yc = dbl_logistic(tg, vmin, amp, m1, s1, m2, s2)
        axb.plot(tg, yc, color=col, lw=1.35, label=name)
        ip = int(np.argmax(yc))
        axb.plot(tg[ip], yc[ip], marker="o", ms=3.0, color=col,
                 mec="white", mew=0.6, zorder=5)
    axb.set_xlim(0, 366)
    axb.set_ylim(0.0, 1.02)
    axb.set_xticks(np.arange(0, 366, 90))
    axb.set_xlabel("Day of year (DOY)")
    axb.set_ylabel("NDVI (dimensionless)")
    axb.set_title("Crop-type phenology", fontsize=8.5)
    style_axes(axb)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.235), ncol=2,
               fontsize=6.4, frameon=False, handlelength=1.2,
               labelspacing=0.25, columnspacing=0.9, borderaxespad=0.0)
    add_panel_label(axb, "(b)", x=-0.24, y=1.03, size=10)

    # ---- (c)-(d) inter-annual trends -------------------------------------
    years = np.arange(2004, 2024)
    n = years.size
    sos_y = 119.0 - 0.42 * (years - 2004) + rng.normal(0, 3.0, n)
    eos_y = 287.0 + 0.26 * (years - 2004) + rng.normal(0, 3.4, n)
    los_y = eos_y - sos_y

    axc = fig.add_subplot(gs[1, :2])
    axc.set_facecolor("#f7f7f4")
    w = 0.4
    a_sos = sos_y - sos_y.mean()
    a_eos = eos_y - eos_y.mean()
    axc.bar(years - w / 2, a_sos, width=w, color=green, alpha=0.85,
            edgecolor="white", linewidth=0.4, label="SOS anomaly")
    axc.bar(years + w / 2, a_eos, width=w, color=rust, alpha=0.85,
            edgecolor="white", linewidth=0.4, label="EOS anomaly")
    for arr, col, off, lab in ((a_sos, dark, -w / 2, "SOS"),
                               (a_eos, "#7a3218", w / 2, "EOS")):
        k, b = np.polyfit(years, arr, 1)
        axc.plot(years + off, k * years + b, color=col, lw=1.3, zorder=5)
    axc.axhline(0, color="#555555", lw=0.7)
    axc.set_xlim(2003.2, 2023.8)
    axc.set_xlabel("Year")
    axc.set_ylabel("Phenology anomaly (d)")
    axc.set_xticks(np.arange(2004, 2024, 4))
    style_axes(axc)
    axc.text(0.015, 0.06,
             r"$\bf{SOS}$: $-$0.42 d yr$^{-1}$   "
             r"$\bf{EOS}$: $+$0.26 d yr$^{-1}$  ($p<0.01$)",
             transform=axc.transAxes, fontsize=6.8, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                       lw=0.5, alpha=0.95))
    axc.legend(loc="upper right", fontsize=6.5, ncol=2, frameon=False,
               handlelength=1.1, columnspacing=0.9, borderaxespad=0.2)
    add_panel_label(axc, "(c)", x=-0.085, y=1.04, size=10)

    axd = fig.add_subplot(gs[1, 2])
    axd.set_facecolor("#f7f7f4")
    a_los = los_y - los_y.mean()
    axd.bar(years, a_los, width=0.68, color="#6bbf8a", edgecolor="white",
            linewidth=0.4)
    k, b = np.polyfit(years, a_los, 1)
    axd.plot(years, k * years + b, color=dark, lw=1.3)
    axd.axhline(0, color="#555555", lw=0.7)
    lim = np.abs(a_los).max() * 1.45
    axd.set_ylim(-lim, lim)
    axd.set_xlim(2003.2, 2023.8)
    axd.set_xticks(np.arange(2004, 2024, 6))
    axd.set_xlabel("Year")
    axd.set_ylabel("LOS anomaly (d)")
    axd.set_title("Season length", fontsize=8.5)
    style_axes(axd)
    axd.text(0.03, 0.965,
             f"mean {los_y.mean():.0f} d\ntrend $+${k:.2f} d yr$^{{-1}}$",
             transform=axd.transAxes, fontsize=6.6, va="top",
             fontweight="bold", color=dark, linespacing=1.2)
    add_panel_label(axd, "(d)", x=-0.24, y=1.03, size=10)

    fig.subplots_adjust(top=0.90, bottom=0.085, left=0.075, right=0.985)
    savefig(fig, "183_rs_ndvi_phenology.png")


if __name__ == "__main__":
    main()
