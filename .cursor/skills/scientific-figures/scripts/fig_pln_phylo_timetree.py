"""Time-calibrated phylogeny with node-age posteriors and a lineage-through-time plot.

One tree object drives all three panels.  Node ages are sampled once as
posteriors, the 95 % HPD bars in the main panel are percentiles of those same
samples, the ridgeline densities on the right are the samples themselves, and
the lineage-through-time curve is swept out of the same node and tip ages, so
no panel can quietly disagree with another about when a clade split.

Time runs from old on the left to young on the right, matching the other
palaeobiology figures in this series.  The label strip to the right of the
present is deliberately outside the period band, so a reader never mistakes a
taxon name for something that happened after 0 Ma.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GOLD = "#c9a227"

FIG_W, FIG_H = 10.4, 6.4
SEED = 7
N_POST = 4000
T_OLD, T_LABEL = 298.9, -98.0

PERIODS = [("P", 298.9, 251.9), ("Tr", 251.9, 201.4), ("J", 201.4, 145.0),
           ("K", 145.0, 66.0), ("Pg", 66.0, 23.0), ("Ng", 23.0, 0.0)]
BAND = ("#f0f0f0", "#e0e0e0")


def leaf(name, end=0.0):
    return {"name": name, "end": end, "children": []}


def node(age, *children, cal=None):
    return {"age": float(age), "children": list(children), "cal": cal}


def build_tree():
    return node(
        262.0,
        node(214.0,
             node(163.0, leaf("Taxon A"),
                  node(88.0, leaf("Taxon B"), leaf("Taxon M", 34.0))),
             node(178.0, leaf("Taxon C", 47.0),
                  node(101.0, leaf("Taxon D"), leaf("Taxon E"))),
             cal="cal. 1: 208 Ma min"),
        node(238.0,
             node(186.0, leaf("Taxon F", 121.0),
                  node(74.0, leaf("Taxon G"), leaf("Taxon N"))),
             node(221.0,
                  node(128.0, leaf("Taxon H", 96.0), leaf("Taxon I")),
                  node(158.0, leaf("Taxon J", 66.0),
                       node(92.0, leaf("Taxon K"), leaf("Taxon L"))),
                  cal="cal. 2: 216 Ma min"),
             cal="cal. 3: 231 Ma min"),
    )


def layout(nd, counter):
    """Depth-first tip ordering; internal y is the mean of its children."""
    if not nd["children"]:
        nd["y"] = float(next(counter))
        return nd["y"]
    ys = [layout(c, counter) for c in nd["children"]]
    nd["y"] = float(np.mean(ys))
    return nd["y"]


def walk(nd, out_int, out_tip):
    if nd["children"]:
        out_int.append(nd)
        for c in nd["children"]:
            walk(c, out_int, out_tip)
    else:
        out_tip.append(nd)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    root = build_tree()
    layout(root, iter(range(1000)))
    internals, tips = [], []
    walk(root, internals, tips)
    n_tip = len(tips)

    # One posterior per divergence; the HPD bars are read back off these.
    for nd in internals:
        s = rng.normal(nd["age"], nd["age"] * 0.055, N_POST)
        nd["post"] = s
        nd["hpd"] = np.percentile(s, [2.5, 97.5])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, (0.95, 1.44, 5.30, 4.03))
    ax_b = axes_inches(fig, (0.95, 1.12, 5.30, 0.28))
    ax_p = axes_inches(fig, (7.20, 3.62, 2.90, 1.85))
    ax_l = axes_inches(fig, (7.20, 0.92, 2.90, 1.85))

    # ---- (a) the tree -----------------------------------------------------
    ax.set_xlim(T_OLD, T_LABEL)
    ax.set_ylim(n_tip - 0.4, -0.9)
    for k, (ab, base, tp) in enumerate(PERIODS):
        ax.axvspan(min(base, T_OLD), tp,
                   facecolor=("#fafafa", "#f2f2f2")[k % 2], zorder=0)
    ax.axvline(0.0, color=HAIR, lw=0.8, zorder=1)
    ax.text(0.0, -0.55, "present", fontsize=6.2, color=MUTED, ha="center",
            va="center", zorder=4)

    for nd in internals:
        ys = [c["y"] for c in nd["children"]]
        ax.plot([nd["age"]] * 2, [min(ys), max(ys)], color=INK, lw=1.1,
                zorder=4, solid_capstyle="round")
        for c in nd["children"]:
            x_end = c["age"] if c["children"] else c["end"]
            ax.plot([nd["age"], x_end], [c["y"]] * 2, color=INK, lw=1.1,
                    zorder=4, solid_capstyle="round")
        ax.plot(nd["hpd"], [nd["y"]] * 2, color=BLUE, lw=4.2, alpha=0.30,
                solid_capstyle="butt", zorder=3)
        ax.plot([nd["age"]], [nd["y"]], marker="o", ms=2.6, color=BLUE,
                zorder=5)
        if nd["cal"]:
            ax.plot([nd["age"]], [nd["y"]], marker="D", ms=4.6, color=GOLD,
                    mec="white", mew=0.6, zorder=6)
            ax.text(nd["age"] - 4.0, nd["y"] - 0.34, nd["cal"], fontsize=6.2,
                    color="#8a6f1a", ha="left", va="center", zorder=6)

    for tp in tips:
        fossil = tp["end"] > 0.0
        if fossil:
            ax.plot([tp["end"]], [tp["y"]], marker="x", ms=4.4, mew=1.2,
                    color=RED, zorder=6)
        ax.plot([tp["end"], -4.0], [tp["y"]] * 2, color=HAIR, lw=0.6,
                ls=(0, (1.2, 1.4)), zorder=2)
        ax.text(-6.0, tp["y"],
                (f"{tp['name']}  (LAD {tp['end']:.0f} Ma)" if fossil
                 else tp["name"]),
                fontsize=6.6, color=RED if fossil else INK, ha="left",
                va="center", style="italic", zorder=6)

    n_fossil = sum(1 for t in tips if t["end"] > 0.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, length=0)
    for s in ("left", "right", "bottom", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Time-calibrated phylogeny of {n_tip} taxa, "
                 f"{n_tip - n_fossil} extant and {n_fossil} extinct",
                 fontsize=9.0, fontweight="bold", color=INK)
    handles = [Line2D([], [], color=BLUE, lw=4.2, alpha=0.30,
                      label="95 % HPD on node age"),
               Line2D([], [], marker="D", ls="none", ms=4.6, color=GOLD,
                      label="fossil-calibrated node"),
               Line2D([], [], marker="x", ls="none", ms=4.4, mew=1.2,
                      color=RED, label="lineage terminates (extinct)")]
    ax.legend(handles=handles, fontsize=6.6, frameon=False, loc="lower left",
              bbox_to_anchor=(0.005, 0.005), handlelength=1.6,
              labelspacing=0.30)

    # ---- period band ------------------------------------------------------
    ax_b.set_xlim(T_OLD, T_LABEL)
    ax_b.set_ylim(0.0, 1.0)
    for k, (ab, base, tp) in enumerate(PERIODS):
        ax_b.axvspan(min(base, T_OLD), tp, facecolor=BAND[k % 2],
                     edgecolor="white", linewidth=0.6, zorder=1)
        ax_b.text(0.5 * (min(base, T_OLD) + tp), 0.42, ab, fontsize=6.6,
                  color=INK, ha="center", va="center", zorder=6)
    ax_b.set_xticks([250, 200, 150, 100, 50, 0])
    ax_b.set_yticks([])
    ax_b.set_xlabel("divergence time (Ma before present) -- older to the left",
                    fontsize=8.0)
    ax_b.tick_params(axis="x", labelsize=6.8)
    for s in ax_b.spines.values():
        s.set_visible(False)

    # ---- (b) node-age posteriors -----------------------------------------
    show = sorted(internals, key=lambda d: -d["age"])[:4]
    grid = np.linspace(150.0, 300.0, 400)
    pitch, dens_h = 1.42, 0.72
    ax_p.set_xlim(300.0, 150.0)
    for i, nd in enumerate(show):
        base = i * pitch
        dens = gaussian_kde(nd["post"])(grid)
        dens = dens_h * dens / dens.max()
        ax_p.fill_between(grid, base, base + dens, color=BLUE, alpha=0.28,
                          linewidth=0, zorder=3)
        ax_p.plot(grid, base + dens, color=BLUE, lw=0.9, zorder=4)
        ax_p.plot(nd["hpd"], [base - 0.15] * 2, color=INK, lw=1.3, zorder=5,
                  solid_capstyle="butt")
        ax_p.plot([nd["age"]], [base - 0.15], marker="o", ms=2.6, color=RED,
                  zorder=6)
        ax_p.text(nd["age"], base + dens_h + 0.05,
                  f"node {nd['age']:.0f} Ma  "
                  f"[{nd['hpd'][0]:.0f}, {nd['hpd'][1]:.0f}]",
                  fontsize=6.4, color=SUB, ha="center", va="bottom")
    ax_p.set_ylim(-0.55, (len(show) - 1) * pitch + dens_h + 0.40)
    ax_p.set_yticks([])
    ax_p.spines["left"].set_visible(False)
    ax_p.set_xlabel("node age (Ma)", fontsize=7.4)
    ax_p.tick_params(labelsize=6.6)
    ax_p.set_title("Posterior age of the four deepest nodes", fontsize=8.5,
                   fontweight="bold", color=INK)

    # ---- (c) lineage-through-time ----------------------------------------
    events = [(nd["age"], len(nd["children"]) - 1) for nd in internals]
    events += [(tp["end"], -1) for tp in tips if tp["end"] > 0.0]
    events.sort(key=lambda e: -e[0])
    t_ltt, n_ltt = [root["age"]], [1]
    for age, delta in events:
        if age == root["age"]:
            n_ltt[-1] += delta
            continue
        t_ltt += [age, age]
        n_ltt += [n_ltt[-1], n_ltt[-1] + delta]
    t_ltt.append(0.0)
    n_ltt.append(n_ltt[-1])

    ax_l.step(t_ltt, n_ltt, where="post", color=BLUE, lw=1.4, zorder=4)
    ax_l.set_yscale("log")
    ax_l.set_xlim(T_OLD, 0.0)
    ax_l.set_ylim(1.6, 22.0)
    ax_l.set_yticks([2, 3, 5, 10, 20])
    ax_l.set_yticklabels(["2", "3", "5", "10", "20"])
    ax_l.minorticks_off()
    for k, (ab, base, tp) in enumerate(PERIODS):
        ax_l.axvspan(min(base, T_OLD), tp,
                     facecolor=("#fafafa", "#f2f2f2")[k % 2], zorder=0)
    ax_l.set_xlabel("geological time (Ma) -- older to the left", fontsize=7.4)
    ax_l.set_ylabel("lineages (n, log axis)", fontsize=7.4)
    ax_l.tick_params(labelsize=6.6)
    ax_l.text(0.97, 0.06,
              f"{max(n_ltt)} lineages at peak, "
              f"{sum(1 for t in tips if t['end'] == 0.0)} survive to 0 Ma",
              transform=ax_l.transAxes, ha="right", va="bottom", fontsize=6.4,
              color=SUB)
    ax_l.set_title("Lineage accumulation flattens after the Jurassic",
                   fontsize=8.5, fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.30, dy=0.10)
    panel_labels_inches(fig, [ax_p], labels=["(b)"], dx=-0.40, dy=0.10)
    panel_labels_inches(fig, [ax_l], labels=["(c)"], dx=-0.52, dy=0.10)
    savefig(fig, "827_pln_phylo_timetree.png")


if __name__ == "__main__":
    main()
