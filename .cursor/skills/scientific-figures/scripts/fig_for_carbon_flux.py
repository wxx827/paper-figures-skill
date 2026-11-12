"""Eddy-covariance carbon budget: NEE as the small residual of two big fluxes.

Everything is generated once at half-hourly resolution and then aggregated,
so the panels cannot disagree with each other.  A solar-geometry PPFD driver
and a seasonal temperature course feed a rectangular-hyperbola light response
for GPP and a Q10 function for ecosystem respiration; NEE is their difference
at every half hour.  The daily panels are the half-hourly series integrated
with the 12.011 g/mol carbon mass, the diurnal panel is the same series
averaged by hour, and the light-response panel is the raw half hours the
daily numbers were built from.

Sign convention is micrometeorological throughout: negative NEE is uptake by
the ecosystem (a sink), positive NEE is release to the atmosphere (a source).
The two daily panels share one y range on purpose - it is the point of the
figure that a net sink of a few grams a day is the residual of gross fluxes
several times larger.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
SINK = "#2e7d5b"        # house green: net uptake
SOURCE = "#b23a48"      # house red: net release
CUM = "#8e6bbf"
TEMP = "#c9a227"
WATER = "#3b5b92"

SEASONS = [("DJF", WATER), ("MAM", SINK), ("JJA", SOURCE), ("SON", TEMP)]

FW, FH = 10.4, 6.4
LEFT_X, LEFT_W = 0.86, 5.42
RIGHT_X, RIGHT_W = 7.32, 2.74
ROW_H, ROW_GAP, ROW_Y0 = 1.17, 0.28, 1.05

N_DAY, N_HALF = 365, 48
LAT = np.deg2rad(51.0)
C_MOL = 12.011e-6        # g C per micromole of CO2
DT = 1800.0              # s per half hour

ALPHA, AMAX = 0.048, 27.0     # quantum yield, light-saturated GPP
R10, Q10 = 3.05, 1.95         # base respiration at 10 C, temperature factor


def solar_ppfd(doy, hour):
    """Clear-sky photon flux density from declination and hour angle."""
    dec = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (doy - 81.0) / 365.0)
    ha = np.deg2rad(15.0 * (hour - 12.0))
    sin_el = (np.sin(LAT) * np.sin(dec) +
              np.cos(LAT) * np.cos(dec) * np.cos(ha))
    return 2050.0 * np.clip(sin_el, 0.0, None)


def leaf_area(doy):
    """Phenology factor in 0..1: leaf-out in May, senescence in October."""
    out = 1.0 / (1.0 + np.exp(-(doy - 128.0) / 7.0))
    down = 1.0 / (1.0 + np.exp((doy - 288.0) / 9.0))
    return 0.10 + 0.90 * out * down


def build_series(rng):
    doy = np.arange(1, N_DAY + 1, dtype=float)
    hour = np.arange(N_HALF) * 0.5
    D, H = np.meshgrid(doy, hour, indexing="ij")

    cloud = np.clip(rng.beta(4.2, 1.7, N_DAY), 0.18, 1.0)[:, None]
    ppfd = solar_ppfd(D, H) * cloud

    # weather noise is a 6-day moving average of white noise, otherwise the
    # driver panel reads as instrument scatter rather than synoptic weather
    wx = np.convolve(rng.normal(0.0, 4.6, N_DAY + 10),
                     np.ones(6) / 6.0, mode="same")[:N_DAY]
    ta = (8.4 + 10.6 * np.sin(2.0 * np.pi * (D - 112.0) / 365.0)
          + 4.6 * np.sin(2.0 * np.pi * (H - 9.5) / 24.0)
          + rng.normal(0.0, 1.15, D.shape).cumsum(axis=1) * 0.03)
    ta += wx[:, None]

    lai = leaf_area(D)
    gpp = lai * (ALPHA * ppfd * AMAX) / (ALPHA * ppfd + AMAX + 1e-9)
    reco = R10 * Q10 ** ((ta - 10.0) / 10.0) * (0.48 + 0.52 * lai)
    nee = reco - gpp                      # positive = release to atmosphere

    to_daily = DT * C_MOL                 # umol m-2 s-1 -> g C m-2 per step
    daily = dict(
        doy=doy,
        nee=nee.sum(axis=1) * to_daily,
        gpp=gpp.sum(axis=1) * to_daily,
        reco=reco.sum(axis=1) * to_daily,
        ta=ta.mean(axis=1),
        lai=lai[:, 0])
    daily["cum"] = np.cumsum(daily["nee"]) * 0.01      # g C m-2 -> t C ha-1

    rain = (rng.random(N_DAY) < 0.16) * rng.exponential(6.4, N_DAY)
    kern = np.exp(-np.arange(45) / 9.0)
    base = 27.5 - 7.5 * np.sin(2.0 * np.pi * (doy - 100.0) / 365.0)
    daily["swc"] = np.clip(base - 3.0 +
                           0.28 * np.convolve(rain, kern)[:N_DAY], 9.0, 39.0)
    return doy, hour, ppfd, ta, nee, daily


def season_mask(doy):
    out = {}
    out["DJF"] = (doy <= 59) | (doy >= 335)
    out["MAM"] = (doy >= 60) & (doy <= 151)
    out["JJA"] = (doy >= 152) & (doy <= 243)
    out["SON"] = (doy >= 244) & (doy <= 334)
    return out


def draw_nee(ax, d, ylim):
    ax.axhline(0.0, color="#808080", lw=0.8, zorder=3)
    ax.fill_between(d["doy"], d["nee"], 0.0, where=d["nee"] <= 0.0,
                    color=SINK, alpha=0.75, linewidth=0.0, zorder=4,
                    interpolate=True)
    ax.fill_between(d["doy"], d["nee"], 0.0, where=d["nee"] > 0.0,
                    color=SOURCE, alpha=0.75, linewidth=0.0, zorder=4,
                    interpolate=True)
    ax.set_ylim(*ylim)
    ax.set_ylabel("NEE\n(g C $m^{-2}$ $d^{-1}$)", fontsize=7.6)
    # anchored to the zero line, which is where the sign convention lives
    ax.annotate("source  +", xy=(6.0, 0.0), xytext=(0, 2),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=6.2, color=SOURCE)
    ax.annotate("sink  $-$", xy=(6.0, 0.0), xytext=(0, -3),
                textcoords="offset points", ha="left", va="top",
                fontsize=6.2, color=SINK)

    ax2 = ax.twinx()
    ax2.plot(d["doy"], d["cum"], color=CUM, lw=1.5, zorder=6)
    ax2.set_ylabel("cumulative NEE\n(t C $ha^{-1}$)", fontsize=7.2,
                   color=CUM)
    ax2.tick_params(labelsize=6.6, colors=CUM)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(CUM)
    ax2.spines["top"].set_visible(False)
    ax2.annotate(f"{d['cum'][-1]:+.2f} t C $ha^{{-1}}$ $yr^{{-1}}$",
                 xy=(d["doy"][-1], d["cum"][-1]), xytext=(-6, 4),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.6, color=CUM)
    ax.set_title("Daily net ecosystem exchange with the running annual total",
                 fontsize=8.5, fontweight="bold")
    return ax2


def draw_partition(ax, d, ylim):
    ax.axhline(0.0, color="#808080", lw=0.8, zorder=3)
    ax.fill_between(d["doy"], -d["gpp"], 0.0, color=SINK, alpha=0.35,
                    linewidth=0.0, zorder=2)
    ax.fill_between(d["doy"], d["reco"], 0.0, color=SOURCE, alpha=0.35,
                    linewidth=0.0, zorder=2)
    ax.plot(d["doy"], -d["gpp"], color=SINK, lw=1.1, zorder=4)
    ax.plot(d["doy"], d["reco"], color=SOURCE, lw=1.1, zorder=4)
    ax.plot(d["doy"], d["nee"], color=RULE, lw=1.0, zorder=5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("flux\n(g C $m^{-2}$ $d^{-1}$)", fontsize=7.6)
    ax.set_title("Gross fluxes: GPP plotted as uptake, $R_{eco}$ as release",
                 fontsize=8.5, fontweight="bold")
    handles = [Line2D([0], [0], color=SINK, lw=1.4, label="GPP (uptake)"),
               Line2D([0], [0], color=SOURCE, lw=1.4, label="$R_{eco}$"),
               Line2D([0], [0], color=RULE, lw=1.2, label="NEE")]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, ncol=3,
              loc="lower left", bbox_to_anchor=(0.005, 0.005),
              handlelength=1.4, columnspacing=1.2)


def draw_drivers(ax, d):
    ax.plot(d["doy"], d["ta"], color=TEMP, lw=1.2, zorder=4)
    ax.set_ylabel("air temp.\n($^\\circ$C)", fontsize=7.6, color=TEMP)
    ax.tick_params(axis="y", labelsize=6.8, colors=TEMP)
    ax.spines["left"].set_color(TEMP)
    ax.axhline(0.0, color="#cccccc", lw=0.7, zorder=2)

    ax2 = ax.twinx()
    ax2.plot(d["doy"], d["swc"], color=WATER, lw=1.1, zorder=4)
    ax2.set_ylabel("soil water\n(%)", fontsize=7.2, color=WATER)
    ax2.tick_params(labelsize=6.6, colors=WATER)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(WATER)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("day of year (-)", fontsize=8.0)
    ax.set_title("Environmental drivers", fontsize=8.5, fontweight="bold")
    return ax2


def draw_diurnal(ax, doy, hour, nee, ylim):
    masks = season_mask(doy)
    curves = {}
    for name, col in SEASONS:
        y = nee[masks[name]].mean(axis=0)
        curves[name] = y
        ax.plot(hour, y, color=col, lw=1.35, zorder=4, label=name)
    ax.axhline(0.0, color="#808080", lw=0.8, zorder=3)
    ax.set_xlim(0.0, 23.5)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(*ylim)
    ax.set_xlabel("hour of day (h)", fontsize=7.8)
    ax.set_ylabel("NEE ($\\mu mol$ $m^{-2}$ $s^{-1}$)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.legend(frameon=False, fontsize=6.4, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.2,
              columnspacing=1.0)
    ax.set_title("Mean diurnal cycle by season", fontsize=8.5,
                 fontweight="bold")
    return curves


def light_response(q, alpha, amax, rd):
    return -(alpha * q * amax) / (alpha * q + amax) + rd


def draw_light(ax, rng, ppfd, nee, lai, ylim):
    grow = (lai > 0.65)[:, None] & (ppfd > 15.0)
    q, y = ppfd[grow], nee[grow]
    par, _ = curve_fit(light_response, q, y, p0=[0.05, 25.0, 3.0],
                       maxfev=20000)
    pick = rng.choice(q.size, size=min(900, q.size), replace=False)
    ax.scatter(q[pick], y[pick], s=3.0, color="#9cc3dd", alpha=0.55,
               linewidths=0.0, zorder=3)
    gx = np.linspace(0.0, q.max(), 200)
    ax.plot(gx, light_response(gx, *par), color=RULE, lw=1.4, zorder=5)
    ax.axhline(0.0, color="#808080", lw=0.8, zorder=4)
    lcp = par[2] * par[1] / (par[0] * (par[1] - par[2]))
    ax.plot([lcp], [0.0], marker="o", ms=4.0, mfc=SOURCE, mec="white",
            mew=0.6, zorder=6)
    ax.set_xlim(0.0, q.max() * 1.02)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PPFD ($\\mu mol$ $m^{-2}$ $s^{-1}$)", fontsize=7.8)
    ax.set_ylabel("NEE ($\\mu mol$ $m^{-2}$ $s^{-1}$)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Growing-season light response", fontsize=8.5,
                 fontweight="bold")
    ax.text(0.975, 0.935,
            f"$\\alpha$ = {par[0]:.3f},  $A_{{max}}$ = {par[1]:.1f},\n"
            f"$R_d$ = {par[2]:.2f},  LCP = {lcp:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
            color=SUB, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))
    return par, lcp


def main():
    set_pub_style()
    rng = np.random.default_rng(686)
    doy, hour, ppfd, ta, nee, d = build_series(rng)

    fig = plt.figure(figsize=(FW, FH))
    rows = [axes_inches(fig, (LEFT_X, ROW_Y0 + k * (ROW_H + ROW_GAP),
                              LEFT_W, ROW_H)) for k in (2, 1, 0)]
    ax_n, ax_p, ax_e = rows
    ax_d = axes_inches(fig, (RIGHT_X, 3.32, RIGHT_W, 1.80))
    ax_l = axes_inches(fig, (RIGHT_X, 1.05, RIGHT_W, 1.58))

    # one y range for both daily-flux panels: NEE is a residual, and the
    # figure only says so if the reader can compare the three curves directly
    day_lim = shared_limits([d["nee"], -d["gpp"], d["reco"]], pad=0.09)
    # one y range for both half-hourly panels: same quantity, same units
    masks = season_mask(doy)
    diurnal = [nee[masks[s]].mean(axis=0) for s, _ in SEASONS]
    grow = (d["lai"] > 0.65)[:, None] & (ppfd > 15.0)
    half_lim = shared_limits(diurnal + [np.percentile(nee[grow], [0.4, 99.6])],
                             pad=0.10)

    draw_nee(ax_n, d, day_lim)
    draw_partition(ax_p, d, day_lim)
    draw_drivers(ax_e, d)
    draw_diurnal(ax_d, doy, hour, nee, half_lim)
    par, lcp = draw_light(ax_l, rng, ppfd, nee, d["lai"], half_lim)

    for ax in rows:
        ax.set_xlim(1, N_DAY)
        ax.set_xticks([1, 60, 121, 182, 244, 305, 365])
        ax.tick_params(labelsize=6.8)
    for ax in (ax_n, ax_p):
        ax.tick_params(labelbottom=False)

    panel_labels_inches(fig, [ax_n, ax_p, ax_e, ax_d, ax_l], dx=-0.68,
                        dy=0.05)

    sink_days = int(np.sum(d["nee"] < 0.0))
    peak = d["nee"].min()

    print(f"annual={d['cum'][-1]:.3f} tC/ha sink_days={sink_days} "
          f"peak={peak:.2f} gpp={d['gpp'].sum() * 0.01:.2f} "
          f"reco={d['reco'].sum() * 0.01:.2f} alpha={par[0]:.4f} "
          f"amax={par[1]:.2f} lcp={lcp:.1f}")
    savefig(fig, "686_for_carbon_flux.png")


if __name__ == "__main__":
    main()
