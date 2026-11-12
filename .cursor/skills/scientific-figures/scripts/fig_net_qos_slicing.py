"""Figure 647 - Three network slices sharing one carrier for a day.

The stacked area in (a) is the radio-resource split; the lines on the right
axis are each slice's SLA *margin* rather than its raw attainment, because
attainment targets of 98 %, 99.9 % and 95 % cannot honestly share one axis -
plotted raw, the URLLC line would be a flat trace pinned to the top of the
frame and its breach would be invisible.  Expressed as distance to its own
target, every slice becomes comparable and zero is the line that matters.
Panels (b) and (c) are reductions of the same run: (b) regresses each slice's
margin on every slice's occupancy, (c) places the measured operating points
against the box each slice was sold.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import panel_label_inches, safe_colorbar, shared_norm

ACCENT = "#b23a48"
GUARD = "#d9d9d9"
SLICES = [("eMBB", "#3b5b92", 98.00), ("URLLC", "#2e7d5b", 99.90),
          ("mMTC", "#c9a227", 95.00)]
# per slice: median latency (ms), median outage probability, target box
TARGET = {"eMBB": (10.0, 60.0, 1e-3, 3e-2),
          "URLLC": (0.30, 2.0, 1e-6, 1e-5),
          "mMTC": (90.0, 600.0, 8e-3, 1e-1)}
OPPOINT = {"eMBB": (26.0, 0.55, 6.0e-3, 0.75),
           "URLLC": (1.05, 0.42, 6.0e-6, 0.85),
           "mMTC": (240.0, 0.40, 2.4e-2, 0.60)}


def day(rng, n=289):
    t = np.linspace(0.0, 24.0, n)

    def wobble(scale):
        w = np.cumsum(rng.normal(0.0, 1.0, n))
        return scale * (w - w.mean()) / (np.abs(w).max() + 1e-9)

    embb = (28.0 + 50.0 * np.exp(-0.5 * ((t - 20.8) / 3.1) ** 2) +
            18.0 * np.exp(-0.5 * ((t - 12.2) / 2.6) ** 2) + wobble(3.0))
    urllc = 13.0 + 7.0 * np.exp(-0.5 * ((t - 9.4) / 3.8) ** 2) + wobble(1.4)
    mmtc = 9.0 + 3.4 * np.sin(2 * np.pi * (t - 4.0) / 24.0) + wobble(1.0)
    prb = np.vstack([np.clip(embb, 4.0, None), np.clip(urllc, 4.0, None),
                     np.clip(mmtc, 3.0, None)])
    prb = prb / np.clip(prb.sum(axis=0) / 98.0, 1.0, None)

    press = np.clip(prb.sum(axis=0) - 90.0, 0.0, None)
    att = np.vstack([
        99.40 - 0.55 * press + rng.normal(0.0, 0.045, n),
        99.995 - 0.020 * press ** 1.4 + rng.normal(0.0, 0.010, n),
        98.50 - 0.200 * press + rng.normal(0.0, 0.060, n)])
    margin = att - np.array([s[2] for s in SLICES])[:, None]
    return t, prb, margin


def main():
    set_pub_style()
    rng = np.random.default_rng(647)
    t, prb, margin = day(rng)
    guard = 100.0 - prb.sum(axis=0)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.058, right=0.905, top=0.830, bottom=0.150,
                          wspace=0.52, hspace=0.62)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) occupancy and SLA margin -------------------------------------
    ax_a.stackplot(t, np.vstack([prb, guard]),
                   colors=[s[1] for s in SLICES] + [GUARD],
                   edgecolor="white", linewidth=0.4, alpha=0.88, zorder=2)
    ax_a.set_xlim(0.0, 24.0)
    ax_a.set_ylim(0.0, 100.0)
    ax_a.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("time of day (h)", fontsize=8.0)
    ax_a.set_ylabel("physical resource blocks allocated (%)", fontsize=8.0)
    ax_a.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)

    ax_a2 = ax_a.twinx()
    for k, (name, col, tgt) in enumerate(SLICES):
        ax_a2.plot(t, margin[k], lw=1.25, color=col, ls=(0, (3.2, 1.5)),
                   zorder=6)
        bad = margin[k] < 0.0
        ax_a2.scatter(t[bad][::3], margin[k][bad][::3], s=9, c=ACCENT,
                      edgecolors="white", linewidths=0.3, zorder=7)
        ax_a2.text(1.4, margin[k][int(0.06 * t.size)] + 0.12, name,
                   ha="left", va="bottom", fontsize=6.6, color=col,
                   fontweight="bold", zorder=8)
    ax_a2.axhline(0.0, color=ACCENT, lw=0.9, zorder=5)
    ax_a2.set_ylim(-3.4, 4.6)
    ax_a2.set_ylabel("SLA margin over own target (pp, 0 = target)",
                     fontsize=8.0)
    ax_a2.tick_params(labelsize=7.0)
    ax_a2.spines["top"].set_visible(False)

    breach = margin < 0.0
    step = (t[1] - t[0]) * 60.0
    worst = int(np.argmax(breach.sum(axis=1)))
    ax_a.set_title("Occupancy against SLA margin: %s is under target for "
                   "%.0f min" % (SLICES[worst][0],
                                 breach[worst].sum() * step),
                   fontsize=8.8, fontweight="bold", pad=4.0)

    # ---- (b) who degrades whom -------------------------------------------
    iso = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            iso[i, j] = -np.polyfit(prb[j], margin[i], 1)[0] * 10.0
    cmap = get_cmap("cb_diverging")
    norm = shared_norm([iso], symmetric=True)
    im = ax_b.imshow(iso, cmap=cmap, norm=norm, interpolation="nearest")
    ax_b.set_aspect("equal")
    names = [s[0] for s in SLICES]
    ax_b.set_xticks(np.arange(3))
    ax_b.set_yticks(np.arange(3))
    ax_b.set_xticklabels(names, fontsize=6.6)
    ax_b.set_yticklabels(names, fontsize=6.6)
    for lab, (_, col, _) in zip(ax_b.get_xticklabels(), SLICES):
        lab.set_color(col)
    for lab, (_, col, _) in zip(ax_b.get_yticklabels(), SLICES):
        lab.set_color(col)
    ax_b.tick_params(length=2.0)
    for i in range(3):
        for j in range(3):
            v = iso[i, j]
            ax_b.text(j, i, "%+.2f" % v, ha="center", va="center",
                      fontsize=6.4, color="white"
                      if abs(v) > 0.62 * norm.vmax else "#1a1a1a")
    ax_b.set_xlabel("occupancy of slice", fontsize=7.4)
    ax_b.set_ylabel("margin lost by slice", fontsize=7.4)
    for s in ax_b.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    safe_colorbar(fig, im, ax_b, label="margin lost per 10 pp of PRB (pp)",
                  width=0.10, pad=0.06, labelsize=6.2)
    ax_b.set_title("Cross-slice isolation", fontsize=8.5, fontweight="bold",
                   pad=3.5)

    # ---- (c) where each slice actually operates ---------------------------
    out_total = 0
    for name, col, _ in SLICES:
        x0, x1, y0, y1 = TARGET[name]
        lat_m, lat_s, out_m, out_s = OPPOINT[name]
        lat = rng.lognormal(np.log(lat_m), lat_s, 70)
        out = rng.lognormal(np.log(out_m), out_s, 70)
        inside = (lat >= x0) & (lat <= x1) & (out >= y0) & (out <= y1)
        out_total += int((~inside).sum())
        ax_c.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=col,
                                 alpha=0.10, edgecolor=col, linewidth=0.9,
                                 linestyle=(0, (3.0, 1.6)), zorder=2))
        ax_c.scatter(lat[inside], out[inside], s=8, c=col, alpha=0.75,
                     edgecolors="white", linewidths=0.25, zorder=4)
        ax_c.scatter(lat[~inside], out[~inside], s=13, c=ACCENT, alpha=0.85,
                     edgecolors="white", linewidths=0.3, zorder=5)
        ax_c.text(np.sqrt(x0 * x1), y1 * 1.9, name, ha="center", va="bottom",
                  fontsize=6.6, color=col, fontweight="bold", zorder=6)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(0.08, 3000.0)
    ax_c.set_ylim(2e-7, 1.2)
    ax_c.set_xlabel("one-way user-plane latency (ms)", fontsize=7.4)
    ax_c.set_ylabel("outage probability (-)", fontsize=7.4)
    ax_c.tick_params(labelsize=6.5)
    ax_c.grid(lw=0.4, alpha=0.4)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.set_title("%d of %d samples outside their box"
                   % (out_total, 3 * 70), fontsize=8.5, fontweight="bold",
                   pad=3.5)

    handles = ([Patch(facecolor=c, edgecolor="white", label="%s occupancy" % n)
                for n, c, _ in SLICES] +
               [Patch(facecolor=GUARD, edgecolor="white",
                      label="unallocated guard"),
                Line2D([], [], color=COLORS["sub"], lw=1.25,
                       ls=(0, (3.2, 1.5)), label="SLA margin (right axis)"),
                Line2D([], [], color=ACCENT, lw=0.0, marker="o", ms=3.4,
                       label="below target")])

    for a, lab, dx in ((ax_a, "(a)", -0.46), (ax_b, "(b)", -0.44),
                       (ax_c, "(c)", -0.44)):
        panel_label_inches(fig, a, lab, dx=dx, dy=0.08)

    occ = prb.sum(axis=0)
    peak = occ.max()
    onset = occ[breach[1]].min() if breach[1].any() else np.nan

    savefig(fig, "647_net_qos_slicing.png")


if __name__ == "__main__":
    main()
