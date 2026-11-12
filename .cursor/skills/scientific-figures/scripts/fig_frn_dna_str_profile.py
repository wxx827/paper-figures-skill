"""Autosomal STR electropherogram with an imbalanced locus and a stutter peak.

One synthetic peak list drives everything on the page: the trace is the sum of
Gaussian peaks built from that list, and every annotated number - peak heights
in RFU, the peak height ratio of the imbalanced locus, the stutter percentage -
is read back out of the same arrays rather than typed into the caption.  The
main panel is the profile; the two smaller panels on the right carry the
supporting statistics at 58 % of the main panel width.

Synthetic illustration for a visual reference library; not a case record.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

ORANGE = "#c0603f"          # questioned / evidence sample
BLUE = "#3b5b92"            # reference statistics
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
BAND = "#f0f0f0"

FW, FH = 10.4, 6.4
MAIN = (0.78, 0.88, 5.12, 4.46)
AUX_T = (6.88, 3.59, 2.96, 1.75)
AUX_B = (6.88, 0.88, 2.96, 1.75)

ANALYTICAL = 150.0          # analytical threshold, RFU
PHR_LIMIT = 0.60            # heterozygote balance limit (-)
SIGMA = 1.15                # electrophoretic peak width, bp

LOCI = [
    ("D3S1358", (96.0, 132.0), [(15, 106.0), (17, 122.0)]),
    ("vWA", (136.0, 178.0), [(16, 146.0), (19, 168.0)]),
    ("D16S539", (182.0, 214.0), [(11, 192.0), (12, 204.0)]),
    ("D2S1338", (218.0, 262.0), [(20, 228.0), (23, 252.0)]),
    ("D8S1179", (266.0, 300.0), [(13, 280.0)]),
    ("D21S11", (304.0, 342.0), [(29, 314.0), (31, 332.0)]),
    ("D18S51", (346.0, 388.0), [(14, 356.0), (18, 380.0)]),
    ("FGA", (392.0, 438.0), [(22, 402.0), (25, 428.0)]),
]
IMBALANCED = "D2S1338"
STUTTER_SHOWN = "D21S11"


def build_peaks(rng):
    """Allele and stutter peaks; heights follow a degradation slope in bp."""
    peaks = []
    for name, _span, alleles in LOCI:
        base = 2550.0 * np.exp(-(alleles[0][1] - 100.0) / 520.0)
        if len(alleles) == 1:
            heights = [base * 1.92 * rng.uniform(0.94, 1.06)]
        else:
            tall = base * rng.uniform(0.90, 1.10)
            ratio = 0.42 if name == IMBALANCED else rng.uniform(0.86, 0.99)
            heights = [tall, tall * ratio]
        for (allele, size), h in zip(alleles, heights):
            peaks.append(dict(locus=name, allele=allele, size=size,
                              rfu=h, kind="allele"))
            frac = (0.082 if name == STUTTER_SHOWN
                    else rng.uniform(0.040, 0.070))
            peaks.append(dict(locus=name, allele=allele - 1, size=size - 4.0,
                              rfu=h * frac, kind="stutter"))
    return peaks


def draw_profile(ax, grid, trace, peaks, ymax, phr, stut):
    for i, (name, (lo, hi), _al) in enumerate(LOCI):
        ax.axvspan(lo, hi, facecolor=BAND if i % 2 == 0 else "#fafafa",
                   edgecolor="none", zorder=0)
        ax.text(0.5 * (lo + hi), -0.150 * ymax, name, ha="center",
                va="center", fontsize=6.4, color=INK)

    ax.fill_between(grid, 0.0, trace, color=ORANGE, alpha=0.17, lw=0.0,
                    zorder=2)
    ax.plot(grid, trace, color=ORANGE, lw=0.95, zorder=3)
    ax.axhline(0.0, color=HAIR, lw=0.7, zorder=1)
    ax.axhline(ANALYTICAL, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=4)
    ax.text(441.0, ANALYTICAL + 0.012 * ymax,
            f"analytical threshold {ANALYTICAL:.0f} RFU", ha="right",
            va="bottom", fontsize=6.2, color=SUB, zorder=5)

    for p in peaks:
        if p["kind"] != "allele":
            continue
        ax.text(p["size"], p["rfu"] + 0.018 * ymax, f"{p['rfu']:.0f}",
                rotation=90, ha="center", va="bottom", fontsize=6.0,
                color=INK, zorder=6)
        ax.text(p["size"], -0.058 * ymax, str(p["allele"]), ha="center",
                va="center", fontsize=6.0, color=SUB, zorder=6)

    st, parent, st_ratio = stut
    ax.annotate(f"stutter n-4\n{st_ratio * 100:.1f} % of {parent['allele']}",
                xy=(st["size"], st["rfu"] + 0.012 * ymax),
                xytext=(316.0, 0.60 * ymax), ha="center", va="bottom",
                fontsize=6.4, color=INK, linespacing=1.35, zorder=7,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=SUB,
                                shrinkA=1.0, shrinkB=1.0))

    small = min((p for p in peaks
                 if p["locus"] == IMBALANCED and p["kind"] == "allele"),
                key=lambda p: p["rfu"])
    ax.annotate(f"allele imbalance\nPHR = {phr[IMBALANCED]:.2f}",
                xy=(small["size"], small["rfu"] + 0.020 * ymax),
                xytext=(232.0, 0.86 * ymax), ha="center", va="bottom",
                fontsize=6.4, color=INK, linespacing=1.35, zorder=7,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ORANGE,
                                shrinkA=1.0, shrinkB=1.0))

    ax.set_xlim(92.0, 444.0)
    ax.set_ylim(-0.205 * ymax, 1.13 * ymax)
    ax.set_yticks(np.arange(0.0, 1.05 * ymax, 500.0))
    ax.set_xticks(np.arange(100.0, 441.0, 40.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("fragment length (bp)", fontsize=8.0)
    ax.set_ylabel("signal (RFU)", fontsize=8.0)
    ax.set_title("Electropherogram of the questioned sample",
                 fontsize=9.0, fontweight="bold", color=INK)


def draw_phr(ax, phr_pop, phr, frac_low):
    edges = np.linspace(0.10, 1.00, 25)
    ax.hist(phr_pop, bins=edges, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(PHR_LIMIT, color=ORANGE, lw=1.0, ls=(0, (3.0, 2.0)))
    ax.text(PHR_LIMIT - 0.02, ax.get_ylim()[1] * 0.96,
            f"balance limit {PHR_LIMIT:.2f}", rotation=90, ha="right",
            va="top", fontsize=6.2, color=ORANGE)
    ax.plot([phr[IMBALANCED]], [ax.get_ylim()[1] * 0.045], marker="v",
            ms=4.2, color=ORANGE, mec="white", mew=0.5, clip_on=False)
    ax.text(phr[IMBALANCED] + 0.015, ax.get_ylim()[1] * 0.17,
            f"{IMBALANCED}\n{phr[IMBALANCED]:.2f}", ha="left", va="bottom",
            fontsize=6.0, color=INK, linespacing=1.3)
    ax.text(0.03, 0.95, f"{frac_low * 100:.1f} % below limit\n"
                        f"n = {phr_pop.size} replicate loci",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
            color=SUB, linespacing=1.35)
    ax.set_xlim(0.10, 1.02)
    ax.tick_params(labelsize=6.4)
    ax.set_xlabel("peak height ratio min/max (-)", fontsize=7.0)
    ax.set_ylabel("loci (count)", fontsize=7.0)
    ax.set_title("Heterozygote balance", fontsize=8.5, fontweight="bold",
                 color=INK)


def draw_het(ax, het):
    names = [n for n, _s, _a in LOCI]
    pos = np.arange(len(names))
    cols = [ORANGE if n == IMBALANCED else BLUE for n in names]
    ax.bar(pos, het * 100.0, width=0.68, color=cols, edgecolor="white",
           linewidth=0.5)
    mean = float(np.mean(het)) * 100.0
    ax.axhline(mean, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)))
    ax.text(-0.42, mean + 1.4, f"mean {mean:.1f} %", ha="left", va="bottom",
            fontsize=6.2, color=SUB)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=6.0)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("observed heterozygosity (%)", fontsize=7.0)
    ax.set_title("Locus informativeness", fontsize=8.5, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(4137)

    peaks = build_peaks(rng)
    grid = np.linspace(90.0, 446.0, 5200)
    trace = 12.0 + rng.normal(0.0, 7.5, grid.size)
    for p in peaks:
        trace += p["rfu"] * np.exp(-0.5 * ((grid - p["size"]) / SIGMA) ** 2)
    trace = np.clip(trace, 0.0, None)
    ymax = float(trace.max())

    phr = {}
    for name, _span, _al in LOCI:
        hs = [p["rfu"] for p in peaks
              if p["locus"] == name and p["kind"] == "allele"]
        if len(hs) == 2:
            phr[name] = min(hs) / max(hs)

    st = max((p for p in peaks if p["locus"] == STUTTER_SHOWN
              and p["kind"] == "stutter"), key=lambda p: p["rfu"])
    parent = min((p for p in peaks if p["locus"] == STUTTER_SHOWN
                  and p["kind"] == "allele"),
                 key=lambda p: abs(p["size"] - (st["size"] + 4.0)))
    stut = (st, parent, st["rfu"] / parent["rfu"])

    n_pop = 320
    core = rng.normal(0.905, 0.070, n_pop)
    low = rng.normal(0.52, 0.14, n_pop)
    phr_pop = np.clip(np.where(rng.random(n_pop) < 0.12, low, core),
                      0.10, 1.0)
    frac_low = float(np.mean(phr_pop < PHR_LIMIT))
    het = rng.uniform(0.70, 0.885, len(LOCI))

    fig = plt.figure(figsize=(FW, FH))
    ax_main = axes_inches(fig, MAIN)
    ax_phr = axes_inches(fig, AUX_T)
    ax_het = axes_inches(fig, AUX_B)

    draw_profile(ax_main, grid, trace, peaks, ymax, phr, stut)
    draw_phr(ax_phr, phr_pop, phr, frac_low)
    draw_het(ax_het, het)

    n_allele = sum(1 for p in peaks if p["kind"] == "allele")

    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.62, dy=0.14)
    panel_labels_inches(fig, [ax_phr, ax_het], labels=["(b)", "(c)"],
                        dx=-0.54, dy=0.14)
    savefig(fig, "792_frn_dna_str_profile.png")


if __name__ == "__main__":
    main()
