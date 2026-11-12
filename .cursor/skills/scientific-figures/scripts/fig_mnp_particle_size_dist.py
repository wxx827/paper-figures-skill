"""Particle size distributions around a hydrocyclone classification stage.

The main panel is the working chart of a grinding circuit: frequency of the
three streams against a logarithmic sieve size, with the cumulative passing
curves on the right axis and the percentile sizes read off those curves by
interpolation rather than assumed.  The two products are produced from the
feed by a partition function, so the three streams really do balance, and
the two smaller panels are the analyses that follow: the Rosin-Rammler plot,
whose least-squares slope *is* the uniformity index, and the Tromp curve of
the classifier, refitted from the scattered size-by-size mass balance to
recover the corrected cut size and the imperfection.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"        # cyclone underflow, the coarse stream
BLUE = "#3b5b92"       # cyclone overflow, the fine product
GREEN = "#2e7d5b"      # cyclone feed
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"

D_LO, D_HI = 1.0, 3000.0        # sieve size axis, micrometres
SEED = 20260813
FEED_D63, FEED_N = 210.0, 1.05  # Rosin-Rammler parameters of the feed
D50C_TRUE, SHARP_TRUE = 105.0, 2.40
BYPASS_TRUE = 0.14
LN2 = np.log(2.0)

ORDER = ["cyclone feed", "cyclone overflow", "cyclone underflow"]
COLOURS = {"cyclone feed": GREEN, "cyclone overflow": BLUE,
           "cyclone underflow": RED}


def rosin_rammler(d, d63, n):
    return 1.0 - np.exp(-(np.asarray(d, float) / d63) ** n)


def plitt(d, bypass, d50c, sharp):
    """Fraction of each size that reports to the underflow."""
    core = 1.0 - np.exp(-LN2 * (np.asarray(d, float) / d50c) ** sharp)
    return bypass + (1.0 - bypass) * core


def streams(n_bin=44):
    """Feed sampled from Rosin-Rammler, then split by the partition curve."""
    edges = np.geomspace(D_LO, D_HI, n_bin + 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    rng = np.random.default_rng(SEED)

    mass_f = np.diff(rosin_rammler(edges, FEED_D63, FEED_N))
    mass_f = np.clip(mass_f * (1.0 + rng.normal(0.0, 0.05, mass_f.size)),
                     1e-9, None)
    mass_f /= mass_f.sum()

    part = plitt(mid, BYPASS_TRUE, D50C_TRUE, SHARP_TRUE)
    split_uf = float((part * mass_f).sum())
    mass_u = np.clip(part * mass_f * (1.0 + rng.normal(0.0, 0.07, mid.size)),
                     1e-12, None)
    mass_o = np.clip((1.0 - part) * mass_f
                     * (1.0 + rng.normal(0.0, 0.07, mid.size)), 1e-12, None)
    mass_u /= mass_u.sum()
    mass_o /= mass_o.sum()

    data = {}
    for name, mass in zip(ORDER, [mass_f, mass_o, mass_u]):
        cum = np.concatenate([[0.0], np.cumsum(mass)])
        data[name] = dict(color=COLOURS[name], edges=edges, mid=mid,
                          mass=mass, cum=cum,
                          freq=mass / np.diff(np.log10(edges)))
    data["split_uf"] = split_uf
    return data


def percentile_size(edges, cum, frac):
    """Size at a given cumulative passing, interpolated in log space."""
    return float(10.0 ** np.interp(frac, cum, np.log10(edges)))


def fit_rosin_rammler(edges, cum):
    """Least-squares slope of ln(-ln(1-F)) against ln(d); returns (n, d63)."""
    ok = (cum > 0.02) & (cum < 0.985)
    x = np.log(edges[ok])
    y = np.log(-np.log(1.0 - cum[ok]))
    n, c = np.polyfit(x, y, 1)
    return float(n), float(np.exp(-c / n))


def fit_partition(data):
    """Refit the Plitt curve to the measured size-by-size mass balance."""
    mid = data["cyclone feed"]["mid"]
    meas = (data["split_uf"] * data["cyclone underflow"]["mass"]
            / np.maximum(data["cyclone feed"]["mass"], 1e-12))
    ok = np.isfinite(meas) & (mid > 2.5) & (meas < 1.35)
    p, _ = curve_fit(plitt, mid[ok], np.clip(meas[ok], 0.0, 1.0),
                     p0=[0.12, 100.0, 2.0],
                     bounds=([0.0, 10.0, 0.6], [0.6, 1000.0, 8.0]))
    bypass, d50c, sharp = (float(v) for v in p)
    d25c = d50c * (np.log(1.0 / 0.75) / LN2) ** (1.0 / sharp)
    d75c = d50c * 2.0 ** (1.0 / sharp)
    pc50 = (0.5 - bypass) / (1.0 - bypass)
    d50 = d50c * (np.log(1.0 / (1.0 - pc50)) / LN2) ** (1.0 / sharp)
    return dict(mid=mid[ok], meas=np.clip(meas[ok], 0.0, 1.0), bypass=bypass,
                d50c=d50c, sharp=sharp, d50=d50,
                imperfection=(d75c - d25c) / (2.0 * d50c))


def main_panel(ax, data):
    ax_c = ax.twinx()
    for name in ORDER:
        d = data[name]
        ax.step(d["edges"], np.concatenate([d["freq"][:1], d["freq"]]),
                where="pre", color=d["color"], lw=1.3, alpha=0.9, zorder=3)
        ax.fill_between(d["mid"], 0.0, d["freq"],
                        color=lighten(d["color"], 0.78), alpha=0.60,
                        linewidth=0.0, zorder=2)
        ax_c.plot(d["edges"], d["cum"] * 100.0, color=d["color"], lw=1.9,
                  ls=(0, (5.5, 2.2)), zorder=5, label=name)

    ref = data["cyclone feed"]
    for frac, tag, dy in [(0.10, "$d_{10}$", 9.0), (0.50, "$d_{50}$", 9.0),
                          (0.80, "$d_{80}$", 8.0), (0.90, "$d_{90}$", 6.0)]:
        d_x = percentile_size(ref["edges"], ref["cum"], frac)
        ax_c.plot([d_x], [frac * 100.0], marker="o", ms=4.2,
                  color=ref["color"], mfc="white", mew=1.2, zorder=8)
        ax_c.annotate(f"{tag} = {d_x:.0f} " + r"$\mu m$",
                      xy=(d_x, frac * 100.0),
                      xytext=(d_x * 0.16, frac * 100.0 + dy), ha="right",
                      va="center", fontsize=6.4, color=ref["color"],
                      bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                edgecolor="none", alpha=0.90),
                      arrowprops=dict(arrowstyle="-", color=ref["color"],
                                      lw=0.7), zorder=9)

    of = data["cyclone overflow"]
    d80_of = percentile_size(of["edges"], of["cum"], 0.80)
    ax_c.axhline(80.0, color=GREY, lw=0.8, ls=(0, (2, 2.2)), zorder=3)
    ax_c.plot([d80_of], [80.0], marker="s", ms=4.4, color=BLUE, mfc="white",
              mew=1.3, zorder=8)
    ax_c.text(d80_of * 1.9, 66.0,
              "product " + r"$d_{80}$ = " + f"{d80_of:.0f} " + r"$\mu m$",
              ha="left", va="top", fontsize=6.8, color=BLUE, zorder=9,
              bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                        edgecolor="none", alpha=0.90))

    ax.set_xscale("log")
    ax.set_xlim(D_LO, D_HI)
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"particle size ($\mu m$, log scale)")
    ax.set_ylabel("frequency (mass fraction per decade of size)")
    ax.tick_params(labelsize=7.2)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)

    ax_c.set_ylim(0.0, 104.0)
    ax_c.set_ylabel("cumulative passing (%)", color=SUB)
    ax_c.tick_params(labelsize=7.2, colors=SUB)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_color(SUB)
    leg = ax_c.legend(fontsize=6.9, frameon=True, framealpha=0.93,
                      loc="upper left", handlelength=2.0, borderaxespad=0.4,
                      edgecolor="#cccccc",
                      title="dashed lines: cumulative passing (right axis)")
    leg.get_title().set_fontsize(6.3)
    leg.get_title().set_color(SUB)
    ax.set_title("Filled histograms on the left axis, cumulative curves on "
                 "the right", fontsize=9.0, fontweight="bold", color=INK)
    return d80_of


def rr_panel(ax, data):
    for name in ORDER:
        d = data[name]
        n_fit, d63_fit = fit_rosin_rammler(d["edges"], d["cum"])
        ok = (d["cum"] > 0.02) & (d["cum"] < 0.985)
        ax.plot(d["edges"][ok], np.log(-np.log(1.0 - d["cum"][ok])),
                marker="o", ms=2.4, ls="none", color=d["color"], alpha=0.72,
                zorder=4)
        xx = np.geomspace(d["edges"][ok][0], d["edges"][ok][-1], 40)
        ax.plot(xx, n_fit * np.log(xx / d63_fit), color=d["color"], lw=1.3,
                zorder=5,
                label=f"{name.split()[-1]}:  n = {n_fit:.2f},  "
                      + r"$d_{63}$ = " + f"{d63_fit:.0f} " + r"$\mu m$")

    ax.set_xscale("log")
    ax.set_xlim(D_LO * 2.0, D_HI)
    ax.set_xlabel(r"particle size ($\mu m$, log scale)", fontsize=7.4)
    ax.set_ylabel(r"$\ln[-\ln(1-F)]$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.55)
    ax.legend(fontsize=6.1, frameon=True, framealpha=0.92, edgecolor="none",
              loc="upper left", handlelength=1.4, borderaxespad=0.25,
              labelspacing=0.32)
    ax.set_title("Rosin-Rammler fit: slope is the index n", fontsize=8.2,
                 fontweight="bold", color=INK)


def tromp_panel(ax, fit):
    xs = np.geomspace(D_LO * 2.0, D_HI, 300)
    total = plitt(xs, fit["bypass"], fit["d50c"], fit["sharp"])
    corrected = 1.0 - np.exp(-LN2 * (xs / fit["d50c"]) ** fit["sharp"])

    ax.plot(fit["mid"], fit["meas"] * 100.0, marker="o", ms=2.8, ls="none",
            color=PURPLE, alpha=0.62, zorder=4)
    ax.plot(xs, total * 100.0, color=PURPLE, lw=1.7, zorder=5,
            label="actual partition")
    ax.plot(xs, corrected * 100.0, color=GOLD, lw=1.4, ls=(0, (5, 2.2)),
            zorder=5, label="corrected for bypass")
    ax.axhline(fit["bypass"] * 100.0, color=GREY, lw=0.8, ls=(0, (2, 2.2)),
               zorder=3)
    ax.text(3.0, fit["bypass"] * 100.0 + 4.0,
            f"bypass {fit['bypass'] * 100:.0f} %", ha="left", va="bottom",
            fontsize=6.2, color=GREY)
    ax.axhline(50.0, color=GREY, lw=0.7, ls=(0, (1.6, 2.0)), zorder=3)
    ax.plot([fit["d50"], fit["d50c"]], [50.0, 50.0], marker="o", ms=4.2,
            ls="none", color=PURPLE, mfc="white", mew=1.2, zorder=8)
    ax.text(4.0, 92.0,
            r"$d_{50}$ = " + f"{fit['d50']:.0f} " + r"$\mu m$" + "\n"
            + r"$d_{50c}$ = " + f"{fit['d50c']:.0f} " + r"$\mu m$" + "\n"
            + f"imperfection I = {fit['imperfection']:.2f}", ha="left",
            va="top", fontsize=6.3, color=INK, zorder=9,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    ax.set_xscale("log")
    ax.set_xlim(D_LO * 2.0, D_HI)
    ax.set_ylim(0.0, 104.0)
    ax.set_xlabel(r"particle size ($\mu m$, log scale)", fontsize=7.4)
    ax.set_ylabel("reporting to underflow (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.55)
    ax.legend(fontsize=6.2, frameon=False, loc="lower right",
              handlelength=1.5, borderaxespad=0.3)
    ax.set_title("Tromp partition curve of the classifier", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    data = streams()
    fit = fit_partition(data)

    fig = plt.figure(figsize=(10.6, 6.6))
    ax_main = axes_inches(fig, (0.86, 0.94, 5.36, 4.66))
    ax_rr = axes_inches(fig, (7.34, 3.64, 2.82, 1.96))
    ax_tr = axes_inches(fig, (7.34, 0.94, 2.82, 1.96))

    d80_of = main_panel(ax_main, data)
    rr_panel(ax_rr, data)
    tromp_panel(ax_tr, fit)

    panel_labels_inches(fig, [ax_main, ax_rr, ax_tr], dx=-0.54, dy=0.10)
    savefig(fig, "831_mnp_particle_size_dist.png")


if __name__ == "__main__":
    main()
