"""Tafel analysis of four oxygen-evolution catalysts.

(a) Main panel: measured current density on a logarithmic axis against
    overpotential, so a Tafel branch is a straight line.  Each data set is
    fitted over the same 1 to 50 mA/cm2 window; the fit is extrapolated as a
    dashed line and its slope quoted in mV/dec.  The benchmark overpotential
    is read where each fit crosses 10 mA/cm2.
(b) Auxiliary panel: Tafel slope bars with the benchmark overpotential of the
    same fit on a second axis, so kinetics and activity can be compared side
    by side.
(c) Auxiliary panel: chronoamperometric hold at a fixed overpotential, with
    the retained current after the hold.

Currents are anodic, so a positive sign is implied throughout.  All slopes,
benchmark overpotentials and retentions come from the plotted arrays.
Illustrative values.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

RNG = np.random.default_rng(76501)

J_BENCH = 10.0                # benchmark current density, mA cm-2
J_LIM = 320.0                 # transport-limited plateau, mA cm-2
FIT_LO, FIT_HI = 1.0, 50.0    # Tafel fitting window, mA cm-2
ETA_HOLD = 300.0              # overpotential of the stability hold, mV
HOURS = 12.0                  # duration of the hold, h

BENCH = "#b23a48"             # house red: the single benchmark line

CATS = [dict(name="NiFe LDH", b=41.0, eta10=248.0, fade=0.06, col="#3b5b92"),
        dict(name="CoP", b=63.0, eta10=286.0, fade=0.11, col="#2e7d5b"),
        dict(name="MnO$_2$", b=89.0, eta10=345.0, fade=0.19, col="#c9a227"),
        dict(name="Ni foam", b=132.0, eta10=402.0, fade=0.31, col="#8e6bbf")]


def polarisation(eta, b, j0):
    """Butler-Volmer branch in decade form, closed by a transport plateau."""
    j_kin = j0 * (10.0 ** (eta / b) - 10.0 ** (-eta / b))
    j_kin = np.clip(j_kin, 1e-9, None)
    return j_kin * J_LIM / (j_kin + J_LIM)


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.90, 1.02, 5.30, 4.62))
    axs = axes_inches(fig, (7.36, 3.74, 2.72, 1.90))
    axb = axes_inches(fig, (7.36, 1.02, 2.72, 1.90))

    eta = np.linspace(60.0, 470.0, 42)
    for c in CATS:
        c["j0"] = J_BENCH * 10.0 ** (-c["eta10"] / c["b"])
        j = polarisation(eta, c["b"], c["j0"])
        c["j"] = j * (1.0 + RNG.normal(0.0, 0.045, eta.size))
        w = (c["j"] >= FIT_LO) & (c["j"] <= FIT_HI)
        slope, icept = np.polyfit(np.log10(c["j"][w]), eta[w], 1)
        c["w"] = w
        c["slope"] = float(slope)                 # mV per decade
        c["eta_bench"] = float(slope * np.log10(J_BENCH) + icept)
        c["icept"] = float(icept)

    # ---------------- (a) Tafel plot --------------------------------------
    ax.axhline(J_BENCH, color=BENCH, lw=0.9, ls=(0, (3.0, 2.2)), zorder=2)
    ax.text(24.0, J_BENCH * 1.18, "benchmark 10 mA cm$^{-2}$", fontsize=6.6,
            color=BENCH, ha="left", va="bottom")
    ax.axhspan(FIT_HI, 1e3, color="#f2f2f2", zorder=0)
    ax.text(466.0, 6.0e2, "transport limited,\nexcluded from the fit",
            fontsize=6.6, color="#777777", ha="right", va="top",
            linespacing=1.40)

    for c in CATS:
        j_line = np.logspace(-2.0, np.log10(J_LIM * 0.9), 80)
        ax.plot(c["slope"] * np.log10(j_line) + c["icept"], j_line,
                color=c["col"], lw=0.8, ls=(0, (3.0, 2.2)), alpha=0.85,
                zorder=3)
        j_fit = np.logspace(np.log10(FIT_LO), np.log10(FIT_HI), 30)
        ax.plot(c["slope"] * np.log10(j_fit) + c["icept"], j_fit,
                color=c["col"], lw=2.0, zorder=5)
        ax.plot(eta, c["j"], ls="none", marker="o", ms=3.4, mfc="white",
                mec=c["col"], mew=0.8, zorder=6,
                label="%s,  $b$ = %.0f mV dec$^{-1}$" % (c["name"],
                                                         c["slope"]))
        ax.plot([c["eta_bench"]], [J_BENCH], marker="o", ms=5.0, mfc="white",
                mec=BENCH, mew=1.1, zorder=8)

    best = int(np.argmin([c["eta_bench"] for c in CATS]))
    worst = int(np.argmax([c["eta_bench"] for c in CATS]))
    for c in CATS:
        ax.plot([c["eta_bench"], c["eta_bench"]], [2.2e-2, J_BENCH],
                color=c["col"], lw=0.7, ls=(0, (2.0, 2.0)), alpha=0.8,
                zorder=2)
        ax.text(c["eta_bench"] - 5.0, 2.6e-2, "$\\eta_{10}$ = %.0f mV"
                % c["eta_bench"], fontsize=6.5, color=c["col"], rotation=90,
                ha="right", va="bottom")
    ax.text(474.0, 1.06e-2, "%.0f mV separates %s from %s at the benchmark"
            % (CATS[worst]["eta_bench"] - CATS[best]["eta_bench"],
               CATS[best]["name"], CATS[worst]["name"]), fontsize=6.6,
            color="#555555", ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(45.0, 480.0)
    ax.set_ylim(1.0e-2, 1.0e3)
    ax.set_xlabel("Overpotential $\\eta$ (mV)")
    ax.set_ylabel("Current density $|j|$ (mA cm$^{-2}$)")
    ax.tick_params(labelsize=7.2, which="both")
    ax.grid(axis="y", color="#f4f4f4", lw=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=7.0, handlelength=1.2,
              labelspacing=0.32, borderaxespad=0.4)
    ax.text(0.025, 0.545, "solid: least-squares fit over 1 to 50 mA cm$^{-2}$\n"
            "dashed: the same fit extrapolated", transform=ax.transAxes,
            fontsize=6.5, color="#777777", ha="left", va="top",
            linespacing=1.45)

    # ---------------- (b) slope and benchmark -----------------------------
    slopes = np.array([c["slope"] for c in CATS])
    bench = np.array([c["eta_bench"] for c in CATS])
    xs = np.arange(len(CATS))
    axs.bar(xs, slopes, width=0.56, color=[c["col"] for c in CATS],
            edgecolor="white", linewidth=0.6, zorder=3)
    for x, s in zip(xs, slopes):
        axs.text(x, s + 3.0, "%.0f" % s, fontsize=6.3, color="#555555",
                 ha="center", va="bottom", zorder=4)
    axs.set_xticks(xs)
    axs.set_xticklabels([c["name"] for c in CATS], fontsize=6.5)
    axs.set_ylim(0.0, slopes.max() * 1.42)
    axs.set_xlabel("Kinetics against activity", fontsize=8, labelpad=1.5)
    axs.set_ylabel("Tafel slope $b$ (mV dec$^{-1}$)", fontsize=8, labelpad=1.5)
    axs.tick_params(axis="y", labelsize=6.5)
    axs.tick_params(axis="x", length=0)
    axs.grid(axis="y", color="#ededed", lw=0.5, zorder=0)

    axt = axs.twinx()
    axt.spines["top"].set_visible(False)
    axt.plot(xs, bench, ls="none", marker="o", ms=5.0, mfc="white",
             mec=BENCH, mew=1.2, zorder=6)
    axt.set_ylim(0.0, bench.max() * 1.42)
    axt.set_ylabel("$\\eta_{10}$ (mV)", fontsize=8, labelpad=2.0, color=BENCH)
    axt.tick_params(axis="y", labelsize=6.5, colors=BENCH)
    axs.legend(handles=[Line2D([], [], ls="none", marker="o", ms=5.0,
                               mfc="white", mec=BENCH, mew=1.2,
                               label="$\\eta$ at 10 mA cm$^{-2}$")],
               loc="upper left", fontsize=6.3, handlelength=1.0,
               borderaxespad=0.25)

    # ---------------- (c) stability hold ----------------------------------
    hrs = np.linspace(0.0, HOURS, 240)
    keep = []
    for c in CATS:
        j_start = polarisation(np.array([ETA_HOLD]), c["b"], c["j0"])[0]
        trace = j_start * (1.0 - c["fade"] * (hrs / HOURS) ** 0.65)
        trace = trace * (1.0 + RNG.normal(0.0, 0.006, hrs.size))
        c["trace"] = trace
        keep.append(trace[-1] / trace[0])
        axb.plot(hrs, trace, color=c["col"], lw=1.15, zorder=4)
    for k in (best, worst):
        axb.text(HOURS * 0.985, CATS[k]["trace"][-1],
                 "%s  %.0f %% kept" % (CATS[k]["name"], 100.0 * keep[k]),
                 fontsize=6.3, color=CATS[k]["col"], ha="right",
                 va="bottom" if k == best else "top")
    axb.set_yscale("log")
    axb.set_xlim(-0.3, HOURS + 0.3)
    all_j = np.concatenate([c["trace"] for c in CATS])
    axb.set_ylim(all_j.min() * 0.75, all_j.max() * 2.6)
    axb.set_xlabel("Hold at $\\eta$ = %.0f mV (h)" % ETA_HOLD, fontsize=8,
                   labelpad=1.5)
    axb.set_ylabel("$|j|$ (mA cm$^{-2}$)", fontsize=8, labelpad=1.5)
    axb.tick_params(labelsize=6.5, which="both")
    axb.grid(color="#ededed", lw=0.5, zorder=0)
    axb.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    axb.text(0.025, 0.955, "colours as in (a)", transform=axb.transAxes,
             fontsize=6.2, color="#777777", ha="left", va="top")

    panel_labels_inches(fig, [ax, axs, axb], dx=-0.40, dy=0.10)

    savefig(fig, "765_ec_tafel_analysis.png")


if __name__ == "__main__":
    main()
