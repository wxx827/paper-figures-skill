"""A marine food web drawn on a real trophic-level axis, with the fishery on it.

Nodes are not placed on tidy integer rows.  Each group's vertical position is
its *realised* trophic level, solved from the diet matrix as TL = 1 + sum of
diet fractions times prey trophic level, so omnivores such as sardine sit
visibly between the levels their neighbours occupy.  Edge width is the energy
flow B x Q/B x diet fraction, node area is keyed to biomass, and the red
arrows are the fishery: it removes biomass from five groups at once, and the
arrow widths are the catches those removals imply.

The side panels carry the consequences.  The biomass pyramid is the same
biomass used to size the nodes, plotted on a log axis with the transfer
efficiency between successive levels computed from it, and the mean trophic
level of the catch is anchored on the value implied by the catch vector in the
main panel - fishing down the food web, ending exactly where the network says
it should.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig, get_palette, annotate_box
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
ACCENT = "#b23a48"
EDGE_COL = "#8fb3cc"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.06, 4.78, 4.10)
AUX1 = (6.62, 3.10, 2.76, 2.06)
AUX2 = (6.62, 1.06, 2.76, 1.44)

# group -> (level index, x position, biomass t/km2, Q/B per year, F per year)
GROUPS = {
    "diatoms":      (0, 0.13, 820.0, 120.0, 0.0),
    "flagellates":  (0, 0.42, 310.0, 145.0, 0.0),
    "picoplankton": (0, 0.71, 190.0, 180.0, 0.0),
    "copepods":     (1, 0.19, 96.0, 32.0, 0.0),
    "krill":        (1, 0.48, 42.0, 26.0, 0.010),
    "gelatinous":   (1, 0.77, 21.0, 20.0, 0.0),
    "sardine":      (2, 0.14, 14.5, 7.8, 0.170),
    "anchovy":      (2, 0.45, 9.2, 8.4, 0.205),
    "squid":        (2, 0.75, 4.1, 9.0, 0.120),
    "hake":         (3, 0.30, 3.4, 3.6, 0.235),
    "mackerel":     (3, 0.62, 2.6, 4.1, 0.260),
    "tuna":         (4, 0.30, 0.62, 2.2, 0.185),
    "seals":        (4, 0.64, 0.29, 9.5, 0.0),
}
DIET = {
    "copepods":   {"diatoms": 0.62, "flagellates": 0.30, "picoplankton": 0.08},
    "krill":      {"diatoms": 0.72, "flagellates": 0.20, "copepods": 0.08},
    "gelatinous": {"flagellates": 0.34, "copepods": 0.52, "krill": 0.14},
    "sardine":    {"diatoms": 0.18, "copepods": 0.58, "krill": 0.24},
    "anchovy":    {"copepods": 0.66, "krill": 0.26, "gelatinous": 0.08},
    "squid":      {"copepods": 0.22, "anchovy": 0.46, "sardine": 0.32},
    "hake":       {"anchovy": 0.38, "sardine": 0.30, "squid": 0.20,
                   "krill": 0.12},
    "mackerel":   {"copepods": 0.30, "krill": 0.22, "anchovy": 0.34,
                   "sardine": 0.14},
    "tuna":       {"sardine": 0.20, "anchovy": 0.18, "squid": 0.26,
                   "mackerel": 0.24, "hake": 0.12},
    "seals":      {"hake": 0.34, "mackerel": 0.28, "squid": 0.22,
                   "sardine": 0.16},
}
LEVEL_NAMES = ["primary producers", "zooplankton", "small pelagics",
               "large fish", "apex predators"]
Y0, Y1 = 1960, 2024


def solve_trophic():
    tl = {k: 1.0 for k in GROUPS}
    for _ in range(60):
        for name, diet in DIET.items():
            tl[name] = 1.0 + sum(f * tl[p] for p, f in diet.items())
    return tl


def flows():
    out = {}
    for pred, diet in DIET.items():
        _, _, b, qb, _ = GROUPS[pred]
        for prey, f in diet.items():
            out[(prey, pred)] = b * qb * f
    return out


def catches():
    return {k: v[2] * v[4] for k, v in GROUPS.items() if v[4] > 0.0}


def draw_web(ax, tl, flow, catch):
    cols = get_palette("cb_sequential_blue", 6)[1:]
    fmax = max(flow.values())
    for (prey, pred), f in sorted(flow.items(), key=lambda kv: kv[1]):
        x0, y0 = GROUPS[prey][1], tl[prey]
        x1, y1 = GROUPS[pred][1], tl[pred]
        ax.plot([x0, x1], [y0, y1], color=EDGE_COL, lw=0.35 + 3.4 * f / fmax,
                alpha=0.55, solid_capstyle="round", zorder=2)

    bio = np.array([v[2] for v in GROUPS.values()])
    lo, hi = np.log10(bio.min()), np.log10(bio.max())
    for name, (lev, x, b, _, _) in GROUPS.items():
        s = 34.0 + 300.0 * (np.log10(b) - lo) / (hi - lo)
        ax.scatter([x], [tl[name]], s=s, color=cols[lev], edgecolors="white",
                   linewidths=0.8, zorder=5)
        ax.text(x, tl[name] - 0.055 - 0.0012 * s, name, ha="center",
                va="top", fontsize=6.4, color=INK, zorder=6)

    fx, fy = 1.03, 3.55
    ax.scatter([fx], [fy], s=115.0, marker="s", color=ACCENT,
               edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(fx, fy + 0.15, "fishery", ha="center", va="bottom", fontsize=6.8,
            color=ACCENT, fontweight="bold", zorder=6)
    cmax = max(catch.values())
    for name, c in catch.items():
        ax.add_patch(FancyArrowPatch(
            (GROUPS[name][1], tl[name]), (fx, fy), arrowstyle="-|>",
            mutation_scale=8.0, linewidth=0.5 + 2.6 * c / cmax,
            color=ACCENT, alpha=0.80, shrinkA=6.0, shrinkB=7.0,
            connectionstyle="arc3,rad=-0.16", zorder=4))

    for lev, name in enumerate(LEVEL_NAMES):
        ax.axhline(lev + 1.0, color="#e4e4e4", lw=0.7, zorder=0)
        ax.text(-0.045, lev + 1.0, name, ha="right", va="center",
                fontsize=6.2, color=MUTED, zorder=3)

    keys = [1.0, 30.0, 800.0]
    handles = [Line2D([0], [0], marker="o", ls="none", mfc="#9cc3dd",
                      mec="white", mew=0.7,
                      ms=np.sqrt(34.0 + 300.0 * (np.log10(k) - lo)
                                 / (hi - lo)),
                      label="{:g} t/km$^2$".format(k)) for k in keys]
    leg = ax.legend(handles=handles, frameon=False, fontsize=6.4,
                    loc="lower right", labelspacing=0.9, handletextpad=0.9,
                    borderpad=0.2, title="biomass", title_fontsize=6.4)
    leg._legend_box.align = "left"

    ax.set_xlim(-0.30, 1.16)
    ax.set_ylim(0.62, 5.35)
    ax.set_xticks([])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("realised trophic level (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Energy flow and fishery removals by trophic level",
                 fontsize=8.5, fontweight="bold")


def draw_pyramid(ax, tl):
    cols = get_palette("cb_sequential_blue", 6)[1:]
    tot = np.zeros(5)
    for name, (lev, _, b, _, _) in GROUPS.items():
        tot[lev] += b
    y = np.arange(5)
    ax.barh(y, tot, height=0.66, color=cols, edgecolor="white",
            linewidth=0.7, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.12, tot.max() * 9.0)
    for k in range(5):
        lab = "{:.0f}".format(tot[k]) if tot[k] >= 10 else \
            "{:.1f}".format(tot[k])
        txt = lab + " t/km$^2$"
        if k > 0:
            txt = txt + "   TE {:.0%}".format(tot[k] / tot[k - 1])
        ax.text(tot[k] * 1.35, k, txt, ha="left", va="center", fontsize=6.2,
                color=SUB, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(["TL {:d}".format(k + 1) for k in range(5)],
                       fontsize=6.8)
    ax.set_ylim(-0.62, 4.62)
    ax.set_xlabel("standing biomass (t/km$^2$, log scale)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Biomass pyramid and transfer efficiency", fontsize=8.5,
                 fontweight="bold")
    return tot


def draw_mtl(ax, tl, catch, rng):
    years = np.arange(Y0, Y1 + 1)
    mtl_now = (sum(c * tl[k] for k, c in catch.items())
               / sum(catch.values()))
    drift = 0.0132
    series = mtl_now + drift * (Y1 - years)
    series = series + rng.normal(0.0, 0.030, years.size)
    series[-1] = mtl_now
    slope, inter = np.polyfit(years, series, 1)
    ax.plot(years, series, color="#8fb3cc", lw=0.9, marker="o", ms=2.0,
            mfc="#3b5b92", mec="none", zorder=3)
    ax.plot(years, inter + slope * years, color=ACCENT, lw=1.4, zorder=5)
    ax.plot([Y1], [mtl_now], marker="o", ms=4.6, mfc="white", mec=ACCENT,
            mew=1.3, zorder=6)
    annotate_box(ax, "{:+.2f} TL per decade\n{:.2f} in {:d}".format(
        slope * 10.0, mtl_now, Y1), loc="lower left", fontsize=6.4,
        margin=0.03)
    ax.set_xlim(Y0 - 1, Y1 + 1)
    ax.set_xlabel("year", fontsize=7.8)
    ax.set_ylabel("mean trophic level\nof the catch (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Fishing down the food web", fontsize=8.5,
                 fontweight="bold")
    return mtl_now, slope


def main():
    set_pub_style()
    rng = np.random.default_rng(694)
    tl = solve_trophic()
    flow = flows()
    catch = catches()

    fig = plt.figure(figsize=(FW, FH))
    ax_w = axes_inches(fig, MAIN)
    ax_p = axes_inches(fig, AUX1)
    ax_m = axes_inches(fig, AUX2)

    draw_web(ax_w, tl, flow, catch)
    tot = draw_pyramid(ax_p, tl)
    mtl_now, slope = draw_mtl(ax_m, tl, catch, rng)

    panel_labels_inches(fig, [ax_w, ax_p, ax_m], dx=-0.52, dy=0.20)

    top_prey = max(catch.items(), key=lambda kv: kv[1])

    print("TL: " + ", ".join("{:s}={:.2f}".format(k, tl[k])
                             for k in ["sardine", "hake", "tuna", "seals"]))
    print("mtl={:.3f} slope={:+.4f}/yr catch={:.3f}".format(
        mtl_now, slope, sum(catch.values())))
    savefig(fig, "694_mar_food_web_trophic.png")


if __name__ == "__main__":
    main()
