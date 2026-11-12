"""Figure 1203 - Where in the state space the safety constraints break.

One of the two figures in this set allowed to spend red, and it spends it on
exactly one thing: the safety boundary and the regions that sit on the wrong
side of it.  Everything else - density, counts, cards - is blue or grey.

A single population of rolled-out states drives all three views.  States are
sampled once; a state is a *violation* when its gap falls under

    d_safe(v) = TAU * v + v ** 2 / (2 * A_BRAKE)

and only violations are histogrammed, boxed, zoomed and counted.  Each
violation is then given exactly one primary constraint label by the priority
ladder in ``classify``, so the bars on the right are a partition of the same
points the heat map is built from: they add up to the population size by
construction, and the footer prints that sum rather than asserting it in prose.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import get_cmap, savefig, set_pub_style, style_axes

SEED = 1608  # original id; figure number is 1203
V_MAX, D_MAX = 32.0, 90.0
TAU, A_BRAKE = 0.90, 10.0          # headway time and comfortable deceleration
V_LIMIT, D_MIN, TTC_MIN, ALAT_MAX = 27.8, 3.0, 1.5, 3.0

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
CARD, CARD_EDGE, GRID = "#e2e8ee", "#c3cdd6", "#ededed"
DEEP, MID, SOFT, PALE = "#0b3d63", "#2e6f9e", "#5f9dc6", "#9cc3dd"
RED = "#b23a48"                    # boundary and violating region, nothing else
SAFEBG = "#eef4f9"

# (label, box in state space) - the two clusters called out on the map
BOX_A = dict(name="A", v=(25.2, 31.6), d=(18.0, 56.0),
             note="highway tailgating")
BOX_B = dict(name="B", v=(0.4, 9.2), d=(0.2, 9.4),
             note="stop-and-go creep")


def d_safe(v):
    return TAU * v + v ** 2 / (2.0 * A_BRAKE)


def population():
    """One sampled log; the violations are the subset under the boundary."""
    rng = np.random.default_rng(SEED)
    v = np.concatenate([
        np.clip(rng.normal(20.5, 6.2, 6000), 0.4, V_MAX),
        np.clip(rng.normal(4.6, 2.2, 900), 0.4, V_MAX),
        np.clip(rng.normal(28.6, 1.7, 700), 0.4, V_MAX)])
    d = np.concatenate([
        d_safe(v[:6000]) * np.exp(rng.normal(0.30, 0.42, 6000))
        + rng.normal(0.0, 1.4, 6000),
        np.abs(rng.normal(4.2, 2.6, 900)) + 0.3,
        np.clip(rng.normal(38.0, 8.5, 700), 3.5, None)])
    d = np.clip(d, 0.2, D_MAX)
    hit = d < d_safe(v)
    v, d = v[hit], d[hit]
    dv = np.abs(rng.normal(1.6, 2.6, v.size)) + 0.05      # closing speed
    return dict(v=v, d=d, ttc=d / dv,
                alat=np.abs(rng.normal(1.05, 1.35, v.size)))


def classify(p):
    """One primary label per violation, by a fixed severity ladder."""
    v, d, ttc, alat = p["v"], p["d"], p["ttc"], p["alat"]
    lab = np.full(v.size, 4, dtype=int)
    lab = np.where(alat > ALAT_MAX, 3, lab)
    lab = np.where(ttc < TTC_MIN, 2, lab)
    lab = np.where(v > V_LIMIT, 1, lab)
    lab = np.where(d < D_MIN, 0, lab)
    return lab


TYPES = [(f"absolute gap < {D_MIN:.1f} m", DEEP, None),
         (f"speed over {V_LIMIT:.1f} m/s limit", MID, None),
         (f"time to collision < {TTC_MIN:.1f} s", SOFT, "///"),
         (f"lateral accel > {ALAT_MAX:.1f} m/s$^2$", PALE, "..."),
         ("gap under $d_{safe}(v)$ only", "#cfe1ee", "xxx")]


def inside(p, box):
    return ((p["v"] >= box["v"][0]) & (p["v"] <= box["v"][1])
            & (p["d"] >= box["d"][0]) & (p["d"] <= box["d"][1]))


def draw_box(ax, box, lw=1.1, label=True):
    x0, x1 = box["v"]
    y0, y1 = box["d"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor="white", lw=lw + 1.2, zorder=6))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=RED, lw=lw, zorder=7))
    if label:
        ax.text(x0 + 0.25, y1 - 1.0, box["name"], ha="left", va="top",
                fontsize=8.6, fontweight="bold", color=RED, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.94))


def main():
    set_pub_style(base_font=8.5)
    p = population()
    lab = classify(p)
    n = int(p["v"].size)
    counts = [int((lab == k).sum()) for k in range(len(TYPES))]
    assert sum(counts) == n, "the constraint labels do not partition the set"

    grid_v = np.linspace(0.0, V_MAX, 129)
    grid_d = np.linspace(0.0, D_MAX, 121)
    H, _, _ = np.histogram2d(p["v"], p["d"], bins=[grid_v, grid_d])
    H = gaussian_filter(H.T, 1.9)
    vmax = float(H.max())

    in_a, in_b = inside(p, BOX_A), inside(p, BOX_B)
    na, nb = int(in_a.sum()), int(in_b.sum())
    area = ((BOX_A["v"][1] - BOX_A["v"][0]) * (BOX_A["d"][1] - BOX_A["d"][0])
            + (BOX_B["v"][1] - BOX_B["v"][0])
            * (BOX_B["d"][1] - BOX_B["d"][0]))
    area_pc = 100.0 * area / (V_MAX * D_MAX)
    hit_pc = 100.0 * (na + nb) / n

    fig = plt.figure(figsize=(10.6, 6.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.44], left=0.058,
                          right=0.988, top=0.828, bottom=0.128, wspace=0.235)
    ax = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])

    # ---- (a) violation density over the state space ------------------------
    style_axes(ax)
    ax.set_xlim(0, V_MAX)
    ax.set_ylim(0, D_MAX)
    vs = np.linspace(0, V_MAX, 400)
    ax.fill_between(vs, d_safe(vs), D_MAX, color=SAFEBG, zorder=0.5)
    im = ax.imshow(H, origin="lower", extent=[0, V_MAX, 0, D_MAX],
                   aspect="auto", cmap=get_cmap("cb_sequential_blue"),
                   vmin=0.0, vmax=vmax, interpolation="bilinear", zorder=1)
    ax.plot(vs, d_safe(vs), color="white", lw=3.0, zorder=3)
    ax.plot(vs, d_safe(vs), color=RED, lw=1.7, zorder=4,
            label="safety boundary")
    ax.plot([0, V_MAX], [D_MIN, D_MIN], color=RED, lw=1.0,
            ls=(0, (4.0, 2.6)), zorder=4)
    ax.plot([V_LIMIT, V_LIMIT], [0, D_MAX], color=RED, lw=1.0,
            ls=(0, (4.0, 2.6)), zorder=4)
    ax.text(19.0, D_MAX - 2.0, "compliant region\n$d \\geq d_{safe}(v)$",
            ha="center", va="top", fontsize=7.6, color=SUB, linespacing=1.5,
            zorder=5)
    ax.text(10.6, D_MIN + 1.2, f"absolute gap floor {D_MIN:.1f} m", ha="left",
            va="bottom", fontsize=6.9, color=RED, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                      alpha=0.93))
    ax.text(V_LIMIT - 0.7, 68.0, f"speed limit {V_LIMIT:.1f} m/s", ha="right",
            va="bottom", fontsize=6.9, color=RED, rotation=90.0, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                      alpha=0.93))
    ax.text(15.5, 21.0, "$d_{safe}(v) = 0.90\\,v + v^{2}/20$", ha="center",
            va="center", fontsize=8.0, color=INK, zorder=6, rotation=27.0,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=CARD_EDGE,
                      lw=0.5, alpha=0.95))
    draw_box(ax, BOX_A)
    draw_box(ax, BOX_B, label=False)
    ax.text(BOX_B["v"][1] + 0.8, BOX_B["d"][1] - 0.6, "B", ha="left",
            va="top", fontsize=8.6, fontweight="bold", color=RED, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                      alpha=0.94))
    ax.text(BOX_A["v"][0] - 0.6, BOX_A["d"][1] + 2.4,
            f"{BOX_A['note']}\n{na} violations", ha="right", va="bottom",
            fontsize=7.0, color=INK, linespacing=1.45, zorder=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                      alpha=0.93))
    ax.set_xlabel("ego speed  $v$  (m/s)", fontsize=8.4)
    ax.set_ylabel("gap to the lead vehicle  $d$  (m)", fontsize=8.4)
    ax.tick_params(labelsize=7.2, length=2.4)
    ax.set_title(f"(a)  {n} constraint violations from 240 rollouts",
                 fontsize=9.2, fontweight="bold", pad=4.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.031, pad=0.012)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.set_ticks([0.0, 0.5 * vmax, vmax])
    cb.set_ticklabels(["0", f"{0.5 * vmax:.1f}", f"{vmax:.1f}"])
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2)
    cb.set_label("violations per cell, smoothed", fontsize=7.2, color=SUB)

    # ---- 2x zoom on cluster B ----------------------------------------------
    axi = ax.inset_axes([0.048, 0.560, 0.300, 0.372])
    axi.set_zorder(9)
    axi.patch.set_visible(True)
    axi.imshow(H, origin="lower", extent=[0, V_MAX, 0, D_MAX], aspect="auto",
               cmap=get_cmap("cb_sequential_blue"), vmin=0.0, vmax=vmax,
               interpolation="bilinear", zorder=1)
    axi.scatter(p["v"][in_b], p["d"][in_b], s=2.6, color=DEEP, alpha=0.55,
                linewidths=0.0, zorder=3)
    axi.plot(vs, d_safe(vs), color="white", lw=2.6, zorder=4)
    axi.plot(vs, d_safe(vs), color=RED, lw=1.5, zorder=5)
    axi.plot([0, V_MAX], [D_MIN, D_MIN], color=RED, lw=0.9,
             ls=(0, (4.0, 2.6)), zorder=5)
    axi.set_xlim(*BOX_B["v"])
    axi.set_ylim(*BOX_B["d"])
    axi.tick_params(labelsize=6.4, length=1.8, pad=1.4)
    axi.set_xticks([2, 5, 8])
    axi.set_yticks([2, 5, 8])
    for sp in axi.spines.values():
        sp.set_edgecolor(RED)
        sp.set_linewidth(1.0)
    axi.set_title(f"B  {BOX_B['note']}  -  {nb} violations, "
                  f"{100.0 * nb / n:.1f} %", fontsize=6.9, color=INK, pad=2.4)
    # The box and the zoom sit at opposite corners, so the default connector
    # pair would draw two long red diagonals across the whole map - more red
    # than this figure is allowed and no clearer than naming the box "B".
    _, links = ax.indicate_inset(
        [BOX_B["v"][0], BOX_B["d"][0], BOX_B["v"][1] - BOX_B["v"][0],
         BOX_B["d"][1] - BOX_B["d"][0]], inset_ax=axi, edgecolor=RED,
        alpha=1.0, linewidth=0.9, zorder=8)
    for ln in links:
        ln.set_visible(False)

    # ---- (b) counts per constraint type ------------------------------------
    style_axes(axc)
    order = np.argsort(counts)[::-1]
    ys = np.arange(len(TYPES))
    axc.set_axisbelow(True)
    axc.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for row, k in enumerate(order):
        name, col, hat = TYPES[k]
        axc.barh(ys[row] + 0.16, counts[k], height=0.46, color=col,
                 edgecolor=MID if hat else "white", linewidth=0.7,
                 hatch=hat, zorder=3)
        axc.text(counts[k] + 0.014 * max(counts), ys[row] + 0.16,
                 f"{counts[k]}   {100.0 * counts[k] / n:.1f} %", ha="left",
                 va="center", fontsize=7.2, color=INK, fontweight="bold",
                 zorder=4)
        # names ride above their own bar: a left tick column would collide
        # with the colour bar of panel (a)
        axc.text(0.010 * max(counts), ys[row] - 0.16, name, ha="left",
                 va="bottom", fontsize=7.4, color=INK, zorder=4)
    axc.set_yticks([])
    axc.set_ylim(len(TYPES) + 0.12, -0.62)     # a free row for the sum card
    axc.spines["left"].set_visible(False)
    axc.set_xlim(0, max(counts) * 1.40)
    axc.set_xlabel("violations  (primary constraint)", fontsize=8.4)
    axc.tick_params(labelsize=7.2, length=2.4)
    axc.set_title("(b)  One primary label per violation",
                  fontsize=9.2, fontweight="bold", pad=4.0)
    axc.text(0.985, 0.035, f"{'  +  '.join(str(c) for c in counts)}\n"
             f"=  {sum(counts)}  =  all violations", transform=axc.transAxes,
             ha="right", va="bottom", fontsize=6.9, color=SUB,
             linespacing=1.5, zorder=6,
             bbox=dict(boxstyle="round,pad=0.32", fc=CARD, ec=CARD_EDGE,
                       lw=0.6))

    # ---- headline + footer --------------------------------------------------
    top = int(np.argmax(counts))
    savefig(fig, "1203_wm_safe_constraint_map.png")


if __name__ == "__main__":
    main()
