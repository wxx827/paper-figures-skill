"""Which precedents the case law actually leans on: a citation network.

Main panel: eighty reported decisions placed by year of judgment on the
horizontal axis and by court tier on the vertical, with an arc for every
citation.  Because citations can only run backwards in time, every arc
travels right to left and the direction needs no arrowhead; node area is
proportional to times cited, and the four most-cited decisions are ringed
and named.  The densest eight-year window of intra-window citation is shaded
and bracketed - that is where the doctrine consolidated.

Auxiliary (both at 58 % of the main panel width): the complementary
cumulative distribution of times cited on log-log axes with a fitted slope,
which is what makes the "a few cases carry the doctrine" claim quantitative;
and the citation age profile, the share of citations still being made to a
precedent of a given age, split by the tier of the cited court, with the
half-life read off each curve.

Nothing is hard-coded: the citation edges are drawn by preferential
attachment with a recency penalty, in-degrees are counted from those edges,
the power-law slope comes from a least-squares fit in log space, and both
half-lives are interpolated off the plotted survival curves.  Fictional
jurisdiction, fictional case names, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches, halo_text

BLUE = "#3b5b92"
DEEP = "#2b4470"
TEAL = "#2e7d5b"
GOLD = "#c9a227"
ACCENT = "#b23a48"
RUST = "#c0603f"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
WIRE = "#c2c2c2"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.92, 0.86, 5.50, 4.62)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

N_CASES = 80
YEAR0, YEAR1 = 1972, 2020
TIERS = ["Trial division", "Court of appeal", "Supreme court"]
TIER_P = [0.475, 0.375, 0.150]
TIER_PULL = [1.0, 1.9, 4.2]     # how much more a higher court gets cited
TIER_COL = [lighten(BLUE, 0.45), BLUE, DEEP]
WINDOW = 8

CASE_NAMES = ["Halloran v. Veyne Board", "In re Corbray Trust",
              "Estate of Pell v. Ashgrove", "Marrow County v. Delane"]


def build_network(seed=926):
    """Years, tiers and citation edges from preferential attachment."""
    rng = np.random.default_rng(seed)
    u = rng.beta(1.5, 1.15, N_CASES)
    year = np.sort(np.round(YEAR0 + u * (YEAR1 - YEAR0)).astype(int))
    tier = rng.choice(3, N_CASES, p=TIER_P)

    indeg = np.zeros(N_CASES)
    edges = []
    for i in range(N_CASES):
        older = np.flatnonzero(year < year[i])
        if older.size < 3:
            continue
        age = year[i] - year[older]
        w = ((1.0 + indeg[older]) ** 1.15
             * np.exp(-age / 13.0)
             * np.array([TIER_PULL[t] for t in tier[older]]))
        w = w / w.sum()
        k = min(int(rng.poisson(3.1)), older.size)
        if k == 0:
            continue
        for j in rng.choice(older, size=k, replace=False, p=w):
            edges.append((i, int(j)))
            indeg[j] += 1.0
    return year, tier, np.array(edges), indeg, rng


def ccdf(counts):
    """(k, P[K >= k]) over the observed support, positive counts only."""
    c = np.sort(counts[counts > 0])
    k = np.unique(c)
    p = np.array([(c >= v).mean() for v in k])
    return k.astype(float), p


def survival(lags, grid):
    return np.array([(lags >= t).mean() for t in grid])


def main():
    set_pub_style(base_font=8)
    year, tier, edges, indeg, rng = build_network()

    jit = rng.uniform(-0.27, 0.27, N_CASES)
    ypos = tier.astype(float) + jit

    milestones = np.argsort(indeg)[::-1][:4]
    # only the two heaviest get coloured in-edges; colouring all four turns
    # the left half of the panel into a solid wash
    into_milestone = {int(m) for m in milestones[:2]}

    # densest WINDOW-year block of citations that stay inside the window
    starts = np.arange(YEAR0, YEAR1 - WINDOW + 2)
    dens = []
    for s in starts:
        inside = ((year[edges[:, 0]] >= s) & (year[edges[:, 0]] < s + WINDOW)
                  & (year[edges[:, 1]] >= s) & (year[edges[:, 1]] < s + WINDOW))
        dens.append(int(inside.sum()))
    best = int(starts[int(np.argmax(dens))])
    n_dense = int(max(dens))

    lags = (year[edges[:, 0]] - year[edges[:, 1]]).astype(float)
    grid = np.linspace(0.0, 46.0, 300)
    is_top = tier[edges[:, 1]] == 2
    s_all = survival(lags, grid)
    s_top = survival(lags[is_top], grid)
    s_low = survival(lags[~is_top], grid)
    hl = {k: float(np.interp(0.5, v[::-1], grid[::-1]))
          for k, v in (("all", s_all), ("top", s_top), ("low", s_low))}

    kk, pp = ccdf(indeg)
    fit = np.polyfit(np.log10(kk[kk >= 2.0]), np.log10(pp[kk >= 2.0]), 1)
    slope = float(fit[0])
    top10 = float(np.sort(indeg)[::-1][:8].sum() / indeg.sum()) * 100.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_p = axes_inches(fig, AUX_T_RECT)
    ax_h = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_p, ax_h):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the network --------------------------------------------------
    ax.add_patch(Rectangle((best - 0.5, -0.60), WINDOW, 3.28,
                           facecolor="#eef1f5", edgecolor="none", zorder=0))
    for t in (0.5, 1.5):
        ax.axhline(t, color="#e2e2e2", lw=0.7, zorder=1)

    for i, j in edges:
        hot = j in into_milestone
        rad = 0.14 if ypos[i] >= ypos[j] else -0.14
        ax.add_patch(FancyArrowPatch(
            (year[i], ypos[i]), (year[j], ypos[j]), arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}", shrinkA=1.2, shrinkB=1.6,
            color=GOLD if hot else WIRE, alpha=0.48 if hot else 0.32,
            linewidth=0.55 if hot else 0.35, zorder=3 if hot else 2))

    size = 2.0 + 2.55 * np.sqrt(indeg)
    for t in range(3):
        m = tier == t
        ax.scatter(year[m], ypos[m], s=size[m] ** 2, facecolor=TIER_COL[t],
                   edgecolor="white", linewidth=0.5, zorder=6)

    for rank, m in enumerate(milestones):
        ax.plot([year[m]], [ypos[m]], marker="o", ms=size[m] + 3.6,
                mfc="none", mec=ACCENT, mew=1.15, zorder=8)
        halo_text(ax, year[m] + 0.9, ypos[m] + 0.17, str(rank + 1),
                  color=ACCENT, halo="white", halo_lw=2.2, fontsize=7.2,
                  fontweight="bold", ha="left", va="bottom", zorder=9)

    ax.plot([best - 0.5, best - 0.5, best + WINDOW - 0.5, best + WINDOW - 0.5],
            [2.52, 2.66, 2.66, 2.52], lw=0.8, color=SUB, zorder=7,
            solid_joinstyle="miter")
    ax.text(best - 0.5 + WINDOW / 2.0, 2.72,
            f"densest doctrinal cluster, {best}-{best + WINDOW - 1}:\n"
            f"{n_dense} citations stay inside the window",
            ha="center", va="bottom", fontsize=6.5, color=SUB,
            linespacing=1.4)

    ax.set_yticks(range(3))
    ax.set_yticklabels(TIERS, fontsize=7.6)
    ax.set_ylim(-2.05, 3.42)
    ax.set_xlim(YEAR0 - 2, YEAR1 + 3)
    ax.set_xticks(np.arange(1975, 2021, 5))
    ax.set_xlabel("year of judgment", fontsize=8.5)
    ax.set_title(f"Reported decisions of the Veyne jurisdiction, "
                 f"n={N_CASES} cases and {len(edges)} citations",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.axhline(-0.62, color="#c9c9c9", lw=0.7, zorder=2)

    # ranked milestone list in the reserved strip under the network
    ax.text(YEAR0 - 1.4, -0.80, "Most-cited decisions (ringed above)",
            ha="left", va="top", fontsize=6.6, fontweight="bold", color=SUB)
    for rank, m in enumerate(milestones):
        col, row = divmod(rank, 2)
        ax.text(YEAR0 - 1.4 + col * 25.5, -1.16 - row * 0.33,
                f"{rank + 1}   {CASE_NAMES[rank]} ({year[m]}), "
                f"cited {int(indeg[m])} times",
                ha="left", va="top", fontsize=6.3, color=ACCENT)
    ax.annotate("", xy=(YEAR0 + 1.0, -1.90), xytext=(YEAR0 + 9.0, -1.90),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=SUB,
                                connectionstyle="arc3,rad=0.16"))
    ax.text(YEAR0 + 10.0, -1.92,
            "each arc is one citation, always running from the later case "
            "back to the earlier one",
            ha="left", va="center", fontsize=6.5, color=SUB)

    handles = [Line2D([], [], marker="o", ls="none", ms=np.sqrt(s),
                      mfc=lighten(BLUE, 0.30), mec="white", mew=0.5,
                      label=f"cited {c} times")
               for s, c in ((14, 1), (60, 10), (150, 25))]
    leg = ax.legend(handles=handles, loc="upper right", frameon=False,
                    fontsize=6.7, labelspacing=0.60, handletextpad=0.9,
                    borderpad=0.2, bbox_to_anchor=(1.006, 1.006),
                    title="node area = times cited")
    leg.get_title().set_fontsize(6.7)
    leg.get_title().set_color(SUB)

    # ---- aux (b): heavy tail of citation counts -----------------------------
    ax_p.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_p.loglog(kk, pp, ls="none", marker="o", ms=3.4, mfc=lighten(BLUE, 0.30),
                mec=BLUE, mew=0.7, zorder=5)
    xf = np.logspace(np.log10(2.0), np.log10(kk.max() * 1.25), 40)
    ax_p.loglog(xf, 10.0 ** (fit[1] + slope * np.log10(xf)), color=ACCENT,
                lw=1.3, ls=(0, (3.6, 2.0)), zorder=6)
    ax_p.set_xlim(0.8, kk.max() * 2.2)
    ax_p.set_ylim(pp.min() * 0.55, 1.6)
    ax_p.tick_params(labelsize=6.6)
    ax_p.set_xlabel("times cited (log scale)", fontsize=7.6, labelpad=1.5)
    ax_p.set_ylabel("P[times cited >= k]", fontsize=7.6)
    ax_p.set_title("A few precedents carry the doctrine", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_p.text(0.030, 0.045,
              f"log-log slope {slope:.2f} (degree exponent "
              f"{1.0 - slope:.2f})\ntop 10 % of cases take {top10:.0f} % of "
              f"all {int(indeg.sum())} citations",
              transform=ax_p.transAxes, ha="left", va="bottom", fontsize=6.1,
              color=SUB, linespacing=1.5)

    # ---- aux (c): citation half-life ----------------------------------------
    ax_h.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_h.axhline(0.5, color=FAINT, lw=0.8, ls=(0, (2.6, 2.0)), zorder=2)
    ax_h.plot(grid, s_top, color=DEEP, lw=1.8, zorder=6)
    ax_h.plot(grid, s_low, color=RUST, lw=1.5, zorder=5)
    ax_h.plot(grid, s_all, color=FAINT, lw=1.0, ls=(0, (3.4, 2.0)), zorder=4)
    for key, col in (("top", DEEP), ("low", RUST)):
        ax_h.plot([hl[key], hl[key]], [0.0, 0.5], lw=0.85, color=col,
                  ls=(0, (2.2, 1.8)), zorder=5)
        ax_h.plot([hl[key]], [0.5], marker="o", ms=3.6, mfc="white", mec=col,
                  mew=1.2, zorder=8)
    ax_h.set_xlim(0.0, 46.0)
    ax_h.set_ylim(0.0, 1.06)
    ax_h.set_xticks([0, 10, 20, 30, 40])
    ax_h.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_h.tick_params(labelsize=6.6)
    ax_h.set_xlabel("age of the cited precedent (years)", fontsize=7.6,
                    labelpad=1.5)
    ax_h.set_ylabel("share of citations at that age or older", fontsize=7.6)
    ax_h.set_title("Higher courts age more slowly", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_h.text(0.975, 0.955,
              f"half-life {hl['top']:.1f} yr, supreme court\n"
              f"half-life {hl['low']:.1f} yr, lower courts\n"
              f"all citations combined {hl['all']:.1f} yr",
              transform=ax_h.transAxes, ha="right", va="top", fontsize=6.1,
              color=SUB, linespacing=1.55)
    hh = [Line2D([], [], color=DEEP, lw=1.8, label="supreme court cited"),
          Line2D([], [], color=RUST, lw=1.5, label="lower courts cited")]
    ax_h.legend(handles=hh, loc="lower left", frameon=False, fontsize=6.2,
                handlelength=1.9, labelspacing=0.32, borderpad=0.15,
                bbox_to_anchor=(-0.006, -0.012))

    panel_labels_inches(fig, [ax, ax_p, ax_h], dx=-0.64, dy=0.12)
    savefig(fig, "926_law_citation_network.png")


if __name__ == "__main__":
    main()
