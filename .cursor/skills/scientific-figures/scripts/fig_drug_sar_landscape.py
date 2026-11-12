"""Structure-activity landscape of a four-series lead-optimisation campaign:
a 2D chemical-space projection coloured by potency, with activity cliffs drawn
between near-identical analogues, plus the potency distribution and the
steepest SALI pairs."""
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, lighten
from figkit import panel_labels_inches, safe_colorbar, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
BASE = "#808080"

# series name, centre (x, y), spread, potency centre (pIC50), n compounds
SERIES = [("Pyrazole series", (-1.55, 1.05), 0.62, 7.55, 62),
          ("Quinazoline series", (1.35, 1.30), 0.55, 8.35, 54),
          ("Sulfonamide series", (1.05, -1.35), 0.70, 6.60, 58),
          ("Fragment growth", (-1.30, -1.25), 0.80, 5.85, 46)]

SIM_SCALE = 1.90        # 2D distance that maps to a Tanimoto-like similarity
SIM_LO, SIM_HI = 0.75, 0.93   # below: not analogues; above: near-replicates
MARKERS = ["o", "s", "^", "D"]


def make_library(rng):
    """Synthetic chemical space: coordinates, potency, series index, ids."""
    xy, pic, grp = [], [], []
    for i, (_n, c, s, p0, n) in enumerate(SERIES):
        pts = rng.normal(c, s, size=(n, 2))
        # potency drifts along a series-specific direction plus analogue noise
        ang = np.deg2rad(35.0 + 70.0 * i)
        proj = (pts - c) @ np.array([np.cos(ang), np.sin(ang)])
        xy.append(pts)
        pic.append(p0 + 0.55 * proj + rng.normal(0.0, 0.42, n))
        grp.append(np.full(n, i))
    xy = np.vstack(xy)
    pic = np.clip(np.concatenate(pic), 4.2, 9.9)
    return xy, pic, np.concatenate(grp)


def cliff_pairs(xy, pic, k=7, n_keep=9):
    """Nearest-neighbour pairs ranked by SALI = d(pIC50) / (1 - similarity)."""
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    sim = np.exp(-d / SIM_SCALE)
    out = {}
    for i in range(xy.shape[0]):
        for j in np.argsort(d[i])[:k]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            s = sim[a, b]
            if not (SIM_LO <= s <= SIM_HI):
                continue
            out[(a, b)] = abs(pic[a] - pic[b]) / max(1.0 - s, 1e-3)
    pairs = sorted(out.items(), key=lambda kv: -kv[1])[:n_keep]
    return [(a, b, v, sim[a, b]) for (a, b), v in pairs]


def landscape_panel(fig, ax, xy, pic, grp, cliffs, marked):
    cmap = get_cmap("cb_sequential_blue")
    lo, hi = shared_limits([pic], pad=0.02)
    for a, b, _v, _s in cliffs:
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=RED, lw=1.3, alpha=0.9,
                zorder=4, solid_capstyle="round")
    for i, mk in enumerate(MARKERS):
        m = grp == i
        sc = ax.scatter(xy[m, 0], xy[m, 1], c=pic[m], cmap=cmap, vmin=lo,
                        vmax=hi, s=27, marker=mk, linewidths=0.45,
                        edgecolors="white", zorder=5)
    cliff_idx = sorted({i for a, b, _v, _s in cliffs for i in (a, b)})
    ax.scatter(xy[cliff_idx, 0], xy[cliff_idx, 1], s=62, facecolors="none",
               edgecolors=RED, linewidths=0.8, zorder=6)

    # leader-line callouts: each analogue takes the nearest free corner slot,
    # which is what keeps the leader lines from crossing one another
    slots = [(0.035, 0.845), (0.035, 0.045), (0.965, 0.845), (0.965, 0.045),
             (0.500, 0.028)]
    x0, x1 = -3.4, 3.4
    y0, y1 = -3.6, 4.5
    frac = [((xy[i, 0] - x0) / (x1 - x0), (xy[i, 1] - y0) / (y1 - y0))
            for i, _t in marked]
    perms = list(permutations(range(len(slots))))
    best_p = min(perms, key=lambda p: sum(
        (slots[p[i]][0] - frac[i][0]) ** 2 + (slots[p[i]][1] - frac[i][1]) ** 2
        for i in range(len(marked))))
    order = [(marked[i], slots[best_p[i]]) for i in range(len(marked))]

    for (idx, tag), (fx, fy) in order:
        ax.annotate(f"{tag}\npIC$_{{50}}$ = {pic[idx]:.2f}",
                    xy=(xy[idx, 0], xy[idx, 1]), xycoords="data",
                    xytext=(fx, fy), textcoords="axes fraction",
                    ha="left" if fx < 0.5 else ("right" if fx > 0.5
                                                else "center"),
                    va="top" if fy > 0.5 else "bottom",
                    fontsize=6.4, color=INK, zorder=8, linespacing=1.30,
                    arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                    shrinkA=2, shrinkB=3))

    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.6, 4.5)
    ax.set_xlabel("Chemical-space projection, component 1 (-)", fontsize=8.0)
    ax.set_ylabel("Chemical-space projection, component 2 (-)", fontsize=8.0)
    ax.set_title("Activity landscape, 220 analogues in four chemical series",
                 fontsize=9.0, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, color="#eceff2", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    safe_colorbar(fig, sc, ax, label="potency  pIC$_{50}$ (-)", width=0.12,
                  pad=0.09)

    for (name, _c, _s, _p, _n), mk in zip(SERIES, MARKERS):
        ax.plot([], [], ls="none", marker=mk, ms=3.6, mfc="none", mec=SUB,
                mew=0.8, label=name)
    ax.plot([], [], color=RED, lw=1.3, label="activity cliff (top 9 SALI)")
    ax.legend(loc="upper center", fontsize=6.4, frameon=False, ncol=3,
              handlelength=1.4, columnspacing=1.1, labelspacing=0.30,
              borderaxespad=0.25, bbox_to_anchor=(0.5, 1.005))


def hist_panel(ax, pic, cliffs):
    lo, hi = shared_limits([pic], pad=0.02)
    bins = np.linspace(lo, hi, 22)
    ax.hist(pic, bins=bins, color=lighten(BLUE, 0.45), edgecolor="white",
            linewidth=0.5, zorder=3)
    cliff_idx = sorted({i for a, b, _v, _s in cliffs for i in (a, b)})
    ax.hist(pic[cliff_idx], bins=bins, color=RED, edgecolor="white",
            linewidth=0.5, zorder=4, label="cliff members")
    med = float(np.median(pic))
    ax.axvline(med, color=BASE, lw=0.9, ls=(0, (4, 2.5)), zorder=5)
    ax.text(med + 0.08, ax.get_ylim()[1] * 0.94, f"median {med:.2f}",
            fontsize=6.2, color=SUB, ha="left", va="top")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Potency  pIC$_{50}$ (-)", fontsize=7.5)
    ax.set_ylabel("Compounds (n)", fontsize=7.5)
    ax.set_title(f"Potency distribution, n = {pic.size}", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.3, frameon=False, handlelength=1.2,
              borderaxespad=0.25)
    style_axes(ax)


def sali_panel(ax, pic, cliffs):
    vals = [v for _a, _b, v, _s in cliffs]
    labs = [f"C{a:03d} / C{b:03d}" for a, b, _v, _s in cliffs]
    dpic = [abs(pic[a] - pic[b]) for a, b, _v, _s in cliffs]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.62, color=RED, edgecolor="white", linewidth=0.5,
            zorder=3)
    for yi, v, d in zip(y, vals, dpic):
        ax.text(v + max(vals) * 0.018, yi,
                f"$\\Delta$pIC$_{{50}}$ = {d:.2f}", ha="left", va="center",
                fontsize=6.0, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6.2)
    ax.set_xlim(0, max(vals) * 1.52)
    ax.set_xlabel("SALI  $\\Delta$pIC$_{50}$ / (1 $-$ similarity) (-)",
                  fontsize=7.5)
    ax.set_title("Steepest activity cliffs", fontsize=8.5, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(3517)
    xy, pic, grp = make_library(rng)
    cliffs = cliff_pairs(xy, pic)

    top = int(np.argmax(pic))
    worst = int(np.argmin(pic))
    steep_a, steep_b = cliffs[0][0], cliffs[0][1]
    lo_of_pair = steep_a if pic[steep_a] < pic[steep_b] else steep_b
    hi_of_pair = steep_b if lo_of_pair == steep_a else steep_a
    marked = [(top, f"C{top:03d}  best in class"),
              (worst, f"C{worst:03d}  fragment start"),
              (hi_of_pair, f"C{hi_of_pair:03d}  cliff, potent side"),
              (lo_of_pair, f"C{lo_of_pair:03d}  cliff, weak side"),
              (int(np.argsort(pic)[-6]), f"C{int(np.argsort(pic)[-6]):03d}  "
                                         "backup lead")]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.52, hspace=0.62,
                          left=0.062, right=0.965, top=0.885, bottom=0.105)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    landscape_panel(fig, ax_m, xy, pic, grp, cliffs, marked)
    hist_panel(ax_h, pic, cliffs)
    sali_panel(ax_s, pic, cliffs)

    d_top = abs(pic[steep_a] - pic[steep_b])

    panel_labels_inches(fig, [ax_m, ax_h, ax_s], dx=-0.40, dy=0.08)
    savefig(fig, "517_drug_sar_landscape.png")


if __name__ == "__main__":
    main()
