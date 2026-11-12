"""Cross-calibration of three optical sensors over a common test site.

Panel (a): red-band surface reflectance time series from Landsat-8 OLI
(655 nm), Sentinel-2 MSI (665 nm) and MODIS/Terra (645 nm) sampled at their
own revisit intervals over three years.
Panels (b)-(d): pairwise same-day regressions with slope, intercept, R2 and
RMSD.
Panel (e): Bland-Altman style drift of the same-day differences, with the
pooled +/-1.96 sigma limits of agreement.

MODIS carries a slow negative radiometric drift, which shows up as a
statistically significant trend in the difference series but leaves the
pairwise regressions almost unchanged.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

T0, T1 = 2019.0, 2022.0
SENSORS = {
    "Landsat-8 OLI": dict(rev=16 / 365.0, gain=1.000, off=0.000, sd=0.0055,
                          drift=0.0, col="#3b5b92", mk="o", wl=655),
    "Sentinel-2 MSI": dict(rev=5 / 365.0, gain=1.021, off=-0.0015, sd=0.0048,
                           drift=0.0, col="#2e7d5b", mk="s", wl=665),
    "MODIS/Terra": dict(rev=8 / 365.0, gain=0.968, off=0.0052, sd=0.0065,
                        drift=-0.0030, col="#c0603f", mk="^", wl=645),
}
PAIRS = [("Sentinel-2 MSI", "Landsat-8 OLI"),
         ("MODIS/Terra", "Landsat-8 OLI"),
         ("MODIS/Terra", "Sentinel-2 MSI")]
SHORT = {"Landsat-8 OLI": "L8", "Sentinel-2 MSI": "S2", "MODIS/Terra": "MODIS"}


def true_signal(t):
    """Seasonal red-band reflectance of a semi-arid test site."""
    return (0.148
            - 0.042 * np.sin(2 * np.pi * (t - 0.10))
            + 0.010 * np.sin(4 * np.pi * (t - 0.32))
            + 0.004 * (t - T0))


def sample_sensor(rng, spec):
    t = np.arange(T0, T1, spec["rev"])
    t = t + rng.normal(0, spec["rev"] * 0.12, t.size)
    t = t[(t >= T0) & (t < T1)]
    keep = rng.random(t.size) > 0.18            # cloud screening
    t = t[keep]
    rho = (spec["gain"] * true_signal(t) + spec["off"]
           + spec["drift"] * (t - T0)
           + rng.normal(0, spec["sd"], t.size))
    return t, rho


def match_pairs(t1, y1, t2, y2, tol=3.0 / 365.0):
    """Same-day matchups between two irregular time series."""
    idx = np.abs(t1[:, None] - t2[None, :]).argmin(axis=1)
    dt = np.abs(t1 - t2[idx])
    ok = dt <= tol
    return t1[ok], y1[ok], y2[idx][ok]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240190)

    obs = {name: sample_sensor(rng, spec) for name, spec in SENSORS.items()}

    fig = plt.figure(figsize=(7.4, 6.1))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.95, 1.15, 0.85],
                  hspace=0.62, wspace=0.34)

    # ---- (a) overlaid time series ----------------------------------------
    axa = fig.add_subplot(gs[0, :])
    tg = np.linspace(T0, T1, 800)
    axa.plot(tg, true_signal(tg), color="#b0b0b0", lw=1.0, zorder=2,
             label="reference signal")
    for name, spec in SENSORS.items():
        t, y = obs[name]
        axa.plot(t, y, color=spec["col"], lw=0.5, alpha=0.45, zorder=3)
        axa.plot(t, y, ls="none", marker=spec["mk"], ms=2.6,
                 mfc=spec["col"], mec="white", mew=0.25, alpha=0.95,
                 zorder=4, label=f"{name} ({spec['wl']} nm), n = {t.size}")
    axa.set_xlim(T0, T1)
    axa.set_xticks(np.arange(T0, T1 + 0.01, 0.5))
    axa.set_xticklabels(["2019.0", "2019.5", "2020.0", "2020.5", "2021.0",
                         "2021.5", "2022.0"], fontsize=6.8)
    axa.set_xlabel("Year", labelpad=1.5)
    axa.set_ylabel("Red-band reflectance (\u2013)")
    axa.set_title("Co-located reflectance time series over the "
                  "pseudo-invariant test site", fontsize=8.4)
    style_axes(axa)
    axa.grid(axis="y", lw=0.45, alpha=0.5)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
               fontsize=6.3, frameon=False, handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0.0)
    add_panel_label(axa, "(a)", x=-0.055, y=1.03, size=10)

    # ---- (b)-(d) pairwise regressions ------------------------------------
    lim = (0.075, 0.225)
    stats = {}
    for i, (a, b) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[1, i])
        ta, ya = obs[a]
        tb, yb = obs[b]
        tm, y_a, y_b = match_pairs(ta, ya, tb, yb)
        k, c = np.polyfit(y_b, y_a, 1)
        r2 = np.corrcoef(y_b, y_a)[0, 1] ** 2
        rmsd = float(np.sqrt(np.mean((y_a - y_b) ** 2)))
        stats[(a, b)] = (tm, y_a - y_b, k, c, r2, rmsd)

        ax.plot(lim, lim, color="#555555", lw=0.85, ls=(0, (4, 3)), zorder=2)
        ax.scatter(y_b, y_a, s=9, facecolor=SENSORS[a]["col"],
                   edgecolor="white", linewidth=0.25, alpha=0.85, zorder=3)
        xg = np.linspace(*lim, 20)
        ax.plot(xg, k * xg + c, color="#b23a48", lw=1.3, zorder=4)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xticks([0.10, 0.15, 0.20])
        ax.set_yticks([0.10, 0.15, 0.20])
        ax.tick_params(labelsize=6.6)
        ax.set_xlabel(f"{SHORT[b]} reflectance (\u2013)", labelpad=1.5)
        ax.set_ylabel(f"{SHORT[a]} reflectance (\u2013)")
        style_axes(ax)
        ax.grid(lw=0.45, alpha=0.45)
        ax.set_title(f"{SHORT[a]} vs {SHORT[b]}", fontsize=8.0)
        ax.text(0.045, 0.955,
                (f"slope {k:.3f}\nintercept {c:+.4f}\n").replace(
                    "-", "\u2212")
                + f"$R^2$ {r2:.3f}\nRMSD {rmsd:.4f}\n$n$ = {tm.size}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
                linespacing=1.30,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#cccccc",
                          lw=0.5, alpha=0.94))
        add_panel_label(ax, f"({'bcd'[i]})", x=-0.26, y=1.03, size=10)

    # ---- (e) Bland-Altman drift ------------------------------------------
    axe = fig.add_subplot(gs[2, :])
    axe.set_facecolor("#fbfbf9")
    pooled = np.concatenate([stats[p][1] for p in PAIRS])
    loa = 1.96 * pooled.std()
    axe.axhspan(-loa, loa, color="#c8c8c8", alpha=0.28, lw=0, zorder=1)
    axe.axhline(0, color="#555555", lw=0.9, zorder=3)
    for lv, tag, va in ((loa, "+1.96$\\sigma$", "bottom"),
                        (-loa, "\u22121.96$\\sigma$", "top")):
        axe.axhline(lv, color="#888888", lw=0.7, ls=(0, (4, 3)), zorder=3)
        axe.text(T1 - 0.02, lv, f"{tag} = {lv * 1000:+.1f}".replace(
            "-", "\u2212") + r" $\times10^{-3}$",
            fontsize=6.0, color="#666666", va=va, ha="right", zorder=5)
    for (a, b) in PAIRS:
        tm, d, *_ = stats[(a, b)]
        col = SENSORS[a]["col"] if a != "MODIS/Terra" or b != "Sentinel-2 MSI" \
            else "#8c4a12"
        axe.scatter(tm, d, s=6, color=col, alpha=0.35, linewidth=0, zorder=4)
        kk, bb = np.polyfit(tm, d, 1)
        tg2 = np.linspace(tm.min(), tm.max(), 20)
        axe.plot(tg2, kk * tg2 + bb, color=col, lw=1.6, zorder=6,
                 label=f"{SHORT[a]} \u2212 {SHORT[b]}:  "
                       f"{kk * 1000:+.2f}".replace("-", "\u2212")
                       + r" $\times10^{-3}$ yr$^{-1}$")
    axe.set_xlim(T0, T1)
    axe.set_xticks(np.arange(T0, T1 + 0.01, 0.5))
    axe.set_xticklabels(["2019.0", "2019.5", "2020.0", "2020.5", "2021.0",
                         "2021.5", "2022.0"], fontsize=6.8)
    axe.set_xlabel("Year", labelpad=1.5)
    axe.set_ylabel("Reflectance difference (\u2013)")
    axe.set_title("Inter-sensor difference drift and limits of agreement",
                  fontsize=8.4)
    style_axes(axe)
    axe.legend(loc="upper left", fontsize=6.2, ncol=3, frameon=True,
               framealpha=0.92, edgecolor="#cccccc", handlelength=1.3,
               columnspacing=1.0, borderpad=0.3, borderaxespad=0.3)
    add_panel_label(axe, "(e)", x=-0.055, y=1.04, size=10)

    fig.subplots_adjust(top=0.915, bottom=0.075, left=0.075, right=0.985)
    savefig(fig, "190_rs_sensor_crosscal.png")


if __name__ == "__main__":
    main()
