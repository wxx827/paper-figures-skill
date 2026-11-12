"""Monthly water supply against demand, and what a dry year would do to it.

The main panel stacks the four supply sources and draws total demand over the
top of them; wherever the demand line rises above the stack the gap is filled
in red, so the shortage months are found by the arithmetic rather than
asserted.  Demand is assembled from four sector profiles, which means the
donut on the right is the annual integral of the very line plotted on the
left.  The risk bars re-run the same balance with the surface-water resource
scaled to four hydrological frequencies, so the sensitivity panel and the
main panel share one model.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, annotate_box, lighten, darken
from figkit import panel_labels_inches, place_outside_labels

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
SHORT = "#b23a48"
MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
         "Oct", "Nov", "Dec"]

# annual sector demand (10^8 m3) and its monthly shape
SECTORS = [
    ("Agriculture", 18.5, "#2e7d5b",
     [0.20, 0.25, 0.70, 1.55, 1.85, 1.70, 1.25, 1.00, 0.75, 0.45, 0.20,
      0.15]),
    ("Industry", 7.2, "#3d6b8e",
     [1.00, 0.92, 1.02, 1.05, 1.06, 1.04, 1.02, 1.00, 1.03, 1.02, 0.95,
      0.98]),
    ("Domestic", 5.4, "#c9a227",
     [0.92, 0.90, 0.96, 1.00, 1.05, 1.12, 1.18, 1.16, 1.04, 0.96, 0.90,
      0.90]),
    ("Ecological", 2.3, "#6bbf8a",
     [0.85, 0.85, 0.95, 1.05, 1.15, 1.20, 1.15, 1.10, 1.05, 0.95, 0.85,
      0.85]),
]

# annual availability (10^8 m3) and monthly shape
SOURCES = [
    ("Surface water", 15.4, "#3d6b8e",
     [0.35, 0.35, 0.55, 0.85, 1.20, 1.60, 2.05, 1.95, 1.35, 0.85, 0.50,
      0.40]),
    ("Groundwater", 8.6, "#2a7f7a",
     [0.95, 0.95, 1.00, 1.08, 1.12, 1.10, 1.02, 0.98, 0.96, 0.96, 0.94,
      0.94]),
    ("Inter-basin transfer", 4.8, "#8fb0c2",
     [1.05, 1.05, 1.02, 1.00, 0.98, 0.94, 0.92, 0.94, 0.98, 1.02, 1.05,
      1.05]),
    ("Reclaimed water", 2.1, "#c9a227",
     [0.88, 0.90, 0.94, 0.98, 1.02, 1.06, 1.12, 1.14, 1.08, 1.02, 0.94,
      0.92]),
]

# hydrological frequency label and the factor applied to the surface resource
FREQ = [("P = 50%", 1.12, "normal year"), ("P = 75%", 1.00, "dry year"),
        ("P = 90%", 0.80, "very dry year"),
        ("P = 95%", 0.68, "extreme dry year")]


def monthly(annual, shape):
    s = np.asarray(shape, dtype=float)
    return annual * s / s.sum()


def balance(surface_factor=1.0):
    """Monthly supply stack and demand; returns (stack, demand, shortage)."""
    stack = []
    for name, tot, _, shape in SOURCES:
        f = surface_factor if name == "Surface water" else 1.0
        stack.append(monthly(tot * f, shape))
    demand = np.sum([monthly(t, s) for _, t, _, s in SECTORS], axis=0)
    supply = np.sum(stack, axis=0)
    return np.array(stack), demand, np.clip(demand - supply, 0.0, None)


def main():
    set_pub_style()

    stack, demand, short = balance(1.0)
    supply = stack.sum(axis=0)
    rate = 100.0 * short.sum() / demand.sum()
    worst = int(np.argmax(short))

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86],
                          left=0.055, right=0.965, top=0.850, bottom=0.150,
                          wspace=0.24, hspace=0.52)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    # ---- monthly balance ---------------------------------------------------
    x = np.arange(12)
    ax_m.stackplot(x, stack, colors=[c for _, _, c, _ in SOURCES],
                   labels=[n for n, _, _, _ in SOURCES], lw=0.4,
                   edgecolor="white", zorder=2)
    ax_m.plot(x, demand, color=INK, lw=1.7, marker="o", ms=3.4,
              mfc="white", mew=1.0, zorder=6, label="total demand")
    ax_m.fill_between(x, supply, demand, where=demand > supply,
                      color=SHORT, alpha=0.38, lw=0, zorder=4,
                      interpolate=True, label="shortage")
    for i in np.flatnonzero(short > 0.02):
        ax_m.plot([i, i], [supply[i], demand[i]], color=SHORT, lw=1.0,
                  zorder=5)
    ax_m.annotate(f"peak deficit {short[worst]:.2f} \u00d7 10$^{8}$ m$^{3}$\n"
                  f"in {MONTH[worst]} ({100 * short[worst] / demand[worst]:.0f}% "
                  "of that month's demand)",
                  xy=(worst, 0.5 * (supply[worst] + demand[worst])),
                  xytext=(worst + 1.4, demand.max() * 1.02), fontsize=6.6,
                  color=SHORT, ha="left", va="top",
                  arrowprops=dict(arrowstyle="->", color=SHORT, lw=0.8,
                                  connectionstyle="arc3,rad=0.22"))
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(MONTH, fontsize=6.8)
    ax_m.set_xlim(-0.35, 11.35)
    ax_m.set_ylim(0, demand.max() * 1.22)
    ax_m.set_ylabel("volume (10$^{8}$ m$^{3}$ per month)", fontsize=8.0)
    ax_m.tick_params(labelsize=6.8)
    ax_m.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_m.set_title("Monthly supply by source against total demand",
                   fontsize=9.0, fontweight="bold")
    annotate_box(ax_m,
                 f"annual demand   {demand.sum():.1f} \u00d7 10$^{8}$ "
                 f"m$^{3}$\nannual supply   {supply.sum():.1f} \u00d7 "
                 f"10$^{8}$ m$^{3}$\nannual shortage rate   {rate:.1f}%",
                 loc="upper left", fontsize=6.8, margin=0.020)

    # ---- demand by sector --------------------------------------------------
    tot = sum(t for _, t, _, _ in SECTORS)
    frac = [t / tot for _, t, _, _ in SECTORS]
    ax_d.pie(frac, colors=[c for _, _, c, _ in SECTORS], startangle=90,
             counterclock=False, radius=1.0,
             wedgeprops=dict(width=0.40, edgecolor="white", linewidth=0.9))
    ang, items = 90.0, []
    for (name, t, col, _), f in zip(SECTORS, frac):
        mid = ang - 180.0 * f
        items.append((mid, name, f"{t:.1f}  ({100 * f:.0f}%)", col))
        ang -= 360.0 * f
    place_outside_labels(ax_d, items, r_edge=1.0, r_stub=1.10, x_knee=1.20,
                         x_text=1.28, min_gap=0.30, y_span=1.30,
                         name_size=6.8, stat_size=6.4)
    ax_d.text(0.0, 0.06, f"{tot:.1f}", ha="center", va="center",
              fontsize=10.5, fontweight="bold", color=INK)
    ax_d.text(0.0, -0.14, "10$^{8}$ m$^{3}$ a$^{-1}$", ha="center",
              va="center", fontsize=6.4, color=MUTED)
    ax_d.set_xlim(-1.72, 1.72)
    ax_d.set_ylim(-1.42, 1.42)
    ax_d.set_aspect("equal")
    ax_d.set_title("Demand by sector", fontsize=8.5, fontweight="bold",
                   pad=2)

    # ---- shortage risk by hydrological frequency ---------------------------
    rates, labels = [], []
    for name, fac, note in FREQ:
        _, dem_f, short_f = balance(fac)
        rates.append(100.0 * short_f.sum() / dem_f.sum())
        labels.append(f"{name}\n{note}")
    cols = [lighten(SHORT, a) for a in (0.62, 0.44, 0.22, 0.0)]
    bars = ax_r.bar(np.arange(len(FREQ)), rates, width=0.62, color=cols,
                    edgecolor=[darken(c, 0.25) for c in cols], lw=0.6,
                    zorder=3)
    for b, v in zip(bars, rates):
        ax_r.text(b.get_x() + b.get_width() / 2.0, v + 0.4, f"{v:.1f}%",
                  ha="center", va="bottom", fontsize=6.4, color=SUB)
    ax_r.set_xticks(np.arange(len(FREQ)))
    ax_r.set_xticklabels(labels, fontsize=6.3, linespacing=1.3)
    ax_r.set_ylim(0, max(rates) * 1.24)
    ax_r.set_ylabel("annual shortage rate (%)", fontsize=7.4)
    ax_r.tick_params(axis="y", labelsize=6.5)
    ax_r.tick_params(axis="x", length=0)
    ax_r.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_r.set_title("Shortage risk by design frequency", fontsize=8.5,
                   fontweight="bold")

    h, l = ax_m.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.082),
               ncol=6, frameon=False, fontsize=7.0, handlelength=1.6,
               columnspacing=1.6)

    panel_labels_inches(fig, [ax_m, ax_d, ax_r], dx=-0.34, dy=0.12)

    n_short = int((short > 0.02).sum())

    print(f"demand={demand.sum():.2f}  supply={supply.sum():.2f}  "
          f"shortage={short.sum():.2f} ({rate:.2f}%)  months={n_short}")
    print("risk by frequency:",
          {f[0]: round(r, 2) for f, r in zip(FREQ, rates)})
    savefig(fig, "605_wat_supply_demand.png")


if __name__ == "__main__":
    main()
