"""Bayesian age-depth model for a lake core, with one date rejected.

The main panel is the model: depth positive downward, calibrated age
increasing to the right, the ensemble median drawn through a 95 % credible
ribbon that pinches at every dated level and swells between them.  Each
radiocarbon determination is shown as its own calibrated probability
distribution laid on its side, which is the only honest way to draw a
calibrated date - it is not a symmetric error bar.  One determination sits
far off the trend and is marked as excluded; the model is built without it.

The two smaller panels are derived from the same ensemble, so they cannot
disagree with the main panel: sedimentation rate against depth, and the
cumulative dry mass the core has accumulated.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"      # the age model
GOLD = "#c9a227"      # accepted radiocarbon dates
RED = "#b23a48"       # the excluded date, the one accent
GREEN = "#2e7d5b"     # sedimentation rate
OCHRE = "#c0603f"     # cumulative dry mass
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
SOLID = "#3a3a3a"

DEPTH_MAX = 600.0
N_ENSEMBLE = 400
DATE_DEPTHS = [25.0, 70.0, 120.0, 175.0, 230.0, 285.0,
               340.0, 400.0, 455.0, 510.0, 560.0, 595.0]
OUTLIER = 4               # index into DATE_DEPTHS
OUTLIER_OFFSET = 1900.0   # yr, how far the rejected date plots off the trend


def core(seed=61):
    """True accumulation history, the dates drawn from it, and bulk density."""
    rng = np.random.default_rng(seed)
    d = np.linspace(0.0, DEPTH_MAX, 601)

    # Sedimentation rate in mm per ka, with a slow interval near 400 cm.
    rate = (620.0 + 300.0 * np.sin(2 * np.pi * d / 430.0 + 0.6)
            - 210.0 * np.exp(-0.5 * ((d - 400.0) / 55.0) ** 2))
    age_ka = 10.0 * cumulative_trapezoid(1.0 / rate, d, initial=0.0)
    age_yr = 1000.0 * age_ka + 45.0

    dates = []
    for i, dz in enumerate(DATE_DEPTHS):
        mu = float(np.interp(dz, d, age_yr))
        sd = float(rng.uniform(85.0, 205.0))
        mu = mu + rng.normal(0.0, 0.45 * sd)
        if i == OUTLIER:
            mu = mu + OUTLIER_OFFSET
        dates.append(dict(depth=dz, mu=mu, sd=sd, rejected=(i == OUTLIER),
                          skew=float(rng.uniform(-0.9, 0.9)),
                          split=float(rng.uniform(0.30, 0.62))))

    rho = (0.17 + 0.56 * (1.0 - np.exp(-d / 230.0))
           + 0.035 * np.sin(2 * np.pi * d / 175.0))
    mass = cumulative_trapezoid(rho, d, initial=0.0)
    return d, rate, age_yr, dates, rho, mass


def calibrated(date, n=240):
    """Two-component, mildly skewed density standing in for a calibration."""
    mu, sd, k, w = date["mu"], date["sd"], date["skew"], date["split"]
    g = np.linspace(mu - 4.2 * sd, mu + 4.2 * sd, n)
    a = np.exp(-0.5 * ((g - (mu - 0.55 * sd * k)) / (0.72 * sd)) ** 2)
    b = np.exp(-0.5 * ((g - (mu + 0.95 * sd * k)) / (1.05 * sd)) ** 2)
    dens = w * a + (1.0 - w) * b
    return g, dens / dens.max()


def ensemble(d, dates, seed=9):
    """Monotone age-depth realisations through the accepted dates only."""
    rng = np.random.default_rng(seed)
    used = [dt for dt in dates if not dt["rejected"]]
    zs = np.array([0.0] + [dt["depth"] for dt in used])
    mus = np.array([0.0] + [dt["mu"] for dt in used])
    sds = np.array([30.0] + [dt["sd"] for dt in used])

    # A plain running maximum would flatten reversed draws into exact ties,
    # and a zero slope turns into an infinite sedimentation rate downstream.
    # Reversals are instead pushed up to a minimum share of the prior gap.
    floor = 0.30 * np.diff(mus)

    ages = np.empty((N_ENSEMBLE, d.size))
    rates = np.empty_like(ages)
    for k in range(N_ENSEMBLE):
        y = mus + rng.normal(0.0, 1.0, mus.size) * sds
        for i in range(1, y.size):
            y[i] = max(y[i], y[i - 1] + floor[i - 1])
        ages[k] = PchipInterpolator(zs, y, extrapolate=True)(d)
        # Rate is read between dated levels, not from the pointwise gradient
        # of the interpolant: a spline can flatten locally, and 10 / (nearly
        # zero) is a spike that would dominate the percentile band.
        seg = 10.0 * np.diff(zs) / (np.diff(y) / 1000.0)   # mm per ka
        idx = np.clip(np.searchsorted(zs, d, side="right") - 1, 0,
                      seg.size - 1)
        rates[k] = seg[idx]
    return ages, rates


def band(arr, lo=2.5, hi=97.5):
    return (np.percentile(arr, lo, axis=0), np.percentile(arr, 50.0, axis=0),
            np.percentile(arr, hi, axis=0))


def panel_model(ax, d, dates, age_lo, age_md, age_hi):
    ax.fill_betweenx(d, age_lo, age_hi, color=BLUE, alpha=0.20,
                     linewidth=0.0, zorder=2, label="95 % credible interval")
    ax.plot(age_md, d, color=BLUE, lw=1.6, zorder=5, label="ensemble median")

    scale = 24.0
    for dt in dates:
        g, dens = calibrated(dt)
        col = RED if dt["rejected"] else GOLD
        ax.fill_between(g, dt["depth"] - scale * dens,
                        dt["depth"] + scale * dens, color=col, alpha=0.80,
                        linewidth=0.4, edgecolor=SOLID, zorder=6)
    ax.fill_between([], [], [], color=GOLD, alpha=0.80,
                    label="calibrated date")
    ax.fill_between([], [], [], color=RED, alpha=0.80,
                    label="excluded date")

    bad = dates[OUTLIER]
    off = bad["mu"] - float(np.interp(bad["depth"], d, age_md))
    ax.annotate("excluded, %+.0f yr off the model" % off,
                xy=(bad["mu"] + 1.9 * bad["sd"], bad["depth"]),
                xytext=(bad["mu"] + 1500.0, bad["depth"] - 78.0),
                ha="left", va="center", fontsize=7.0, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9))

    ax.set_xlabel("calibrated age (cal yr BP), increasing to the right")
    ax.set_ylabel("depth below sediment surface (cm)")
    ax.set_xlim(-450.0, max(age_hi.max(), bad["mu"] + 5200.0))
    ax.set_ylim(DEPTH_MAX, 0.0)
    ax.set_yticks(np.arange(0, 601, 100))
    ax.grid(lw=0.5, alpha=0.6)
    ax.legend(fontsize=7.0, frameon=False, loc="lower left",
              borderaxespad=0.35, handlelength=1.5)
    ax.text(0.985, 0.965,
            "%d of %d determinations accepted"
            % (len(dates) - 1, len(dates)), transform=ax.transAxes,
            ha="right", va="top", fontsize=7.0, color=SUB)
    ax.set_title("One rejected date, eleven that agree", fontsize=9.0,
                 fontweight="bold", color=INK)
    return off


def panel_rate(ax, d, rate_lo, rate_md, rate_hi):
    ax.fill_betweenx(d, rate_lo, rate_hi, color=GREEN, alpha=0.20,
                     linewidth=0.0, zorder=2)
    ax.plot(rate_md, d, color=GREEN, lw=1.2, zorder=3)
    k = int(np.argmin(rate_md))
    ax.plot([rate_md[k]], [d[k]], marker="o", ms=4.0, color=GREEN,
            mfc="white", mew=1.2, zorder=5)
    ax.annotate("slowest interval\n%.0f mm ka$^{-1}$ at %.0f cm"
                % (rate_md[k], d[k]),
                xy=(rate_md[k], d[k]), xytext=(rate_md[k] + 210.0, d[k] + 96.0),
                ha="left", va="center", fontsize=6.4, color=GREEN,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))
    ax.set_ylim(DEPTH_MAX, 0.0)
    ax.set_xlim(0.0, rate_hi.max() * 1.06)
    ax.set_yticks(np.arange(0, 601, 200))
    ax.set_xlabel("sedimentation rate (mm ka$^{-1}$)", fontsize=7.6)
    ax.set_ylabel("depth (cm)", fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_title("Accumulation rate with depth", fontsize=8.5,
                 fontweight="bold", color=INK)
    return rate_md[k], d[k]


def panel_mass(ax, d, mass, rho):
    ax.fill_betweenx(d, mass * 0.92, mass * 1.08, color=OCHRE, alpha=0.20,
                     linewidth=0.0, zorder=2)
    ax.plot(mass, d, color=OCHRE, lw=1.2, zorder=3)
    ax.set_ylim(DEPTH_MAX, 0.0)
    ax.set_xlim(0.0, mass.max() * 1.08)
    ax.set_yticks(np.arange(0, 601, 200))
    ax.set_xlabel("cumulative dry mass (g cm$^{-2}$)", fontsize=7.6,
                  labelpad=1.5)
    ax.set_ylabel("depth (cm)", fontsize=7.6)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.text(0.97, 0.06,
            "dry bulk density %.2f - %.2f g cm$^{-3}$" % (rho.min(),
                                                          rho.max()),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            color=SUB)
    ax.set_title("Cumulative accumulation", fontsize=8.5, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    d, _, _, dates, rho, mass = core()
    ages, rates = ensemble(d, dates)
    age_lo, age_md, age_hi = band(ages)
    rate_lo, rate_md, rate_hi = band(rates)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_mod = axes_inches(fig, (0.95, 0.82, 5.02, 4.62))
    ax_rate = axes_inches(fig, (7.10, 3.28, 2.85, 2.16))
    ax_mass = axes_inches(fig, (7.10, 0.86, 2.85, 1.72))

    off = panel_model(ax_mod, d, dates, age_lo, age_md, age_hi)
    slow, slow_z = panel_rate(ax_rate, d, rate_lo, rate_md, rate_hi)
    panel_mass(ax_mass, d, mass, rho)

    width = age_hi - age_lo

    panel_labels_inches(fig, [ax_mod, ax_rate, ax_mass], dx=-0.64, dy=0.10)
    savefig(fig, "755_iso_age_depth_model.png")


if __name__ == "__main__":
    main()
