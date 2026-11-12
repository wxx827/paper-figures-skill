"""Biomagnification of a lipophilic contaminant along a lake food web.

The main panel plots the measured burden of every compartment on a log axis -
the only scale on which a three-order-of-magnitude food chain can be compared
honestly - and puts the step-by-step biomagnification factor on the arrow
between neighbouring levels.  The upper auxiliary panel collapses those steps
into a single regression against trophic position, whose slope is the trophic
magnification factor; the lower one shows where the burden ends up inside the
top predator.

Synthetic data.  Every factor, slope and percentage printed on the figure is
recomputed from the plotted arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, get_palette, darken
from figkit import panel_labels_inches, place_outside_labels

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
ACCENT = "#b23a48"

# compartment, trophic position (from d15N), median burden (mg/kg lipid)
WEB = [
    ("Phytoplankton",   1.00, 0.045),
    ("Zooplankton",     2.05, 0.185),
    ("Mysid shrimp",    2.60, 0.510),
    ("Forage fish",     3.35, 1.95),
    ("Predatory fish",  4.10, 7.40),
    ("Piscivorous bird", 4.65, 21.0),
]
# tissue, share of whole-body burden in the top fish
TISSUE = [("Lipid / adipose", 0.52), ("Liver", 0.24),
          ("Muscle", 0.16), ("Kidney", 0.08)]
BURDEN = 7.40 * 1.0          # mg/kg lipid, top predatory fish


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(625)

    names = [w[0] for w in WEB]
    tl = np.array([w[1] for w in WEB])
    conc = np.array([w[2] for w in WEB])
    cols = get_palette("cb_sequential_green", 9)[2:8]

    # replicate individuals, log-normally scattered about the median
    reps = [c * 10.0 ** rng.normal(0.0, 0.16, 9) for c in conc]

    bmf = conc[1:] / conc[:-1]
    slope, intercept = np.polyfit(tl, np.log10(conc), 1)
    tmf = 10.0 ** slope
    pred = intercept + slope * tl
    ss_res = np.sum((np.log10(conc) - pred) ** 2)
    ss_tot = np.sum((np.log10(conc) - np.log10(conc).mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0],
                          left=0.062, right=0.975, top=0.865, bottom=0.170,
                          wspace=0.26, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    ax_reg = fig.add_subplot(gs[0, 1])
    ax_pie = fig.add_subplot(gs[1, 1])

    # ---- (a) burden per compartment ---------------------------------------
    base = 0.010
    x = np.arange(len(WEB))
    ax.bar(x, conc - base, bottom=base, width=0.60, color=cols,
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, r in enumerate(reps):
        ax.plot(i + rng.uniform(-0.16, 0.16, r.size), r, ls="none",
                marker="o", ms=2.6, mfc="white", mec=darken(cols[i], 0.25),
                mew=0.7, alpha=0.9, zorder=5)
        ax.text(i, conc[i] * 1.08, f"{conc[i]:.3g}", ha="center", va="bottom",
                fontsize=6.6, color=darken(cols[i], 0.35), zorder=6)

    # biomagnification arrows ride above the bars on their own log-space rail
    rail = conc.max() * 3.2
    for i, f in enumerate(bmf):
        y = rail
        ax.annotate("", xy=(i + 1, y), xytext=(i, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=RULE,
                                    shrinkA=1.0, shrinkB=1.0,
                                    mutation_scale=8))
        ax.plot([i, i], [conc[i] * 1.55, y], lw=0.5, color="#cccccc", zorder=1)
        ax.plot([i + 1, i + 1], [conc[i + 1] * 1.55, y], lw=0.5,
                color="#cccccc", zorder=1)
        ax.text(i + 0.5, y * 1.10, f"BMF {f:.1f}$\\times$", ha="center",
                va="bottom", fontsize=6.6, color=RULE, zorder=6)

    ax.set_yscale("log")
    ax.set_ylim(base, conc.max() * 16.0)
    ax.set_xlim(-0.62, len(WEB) - 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7.0,
                       linespacing=1.3)
    ax.set_ylabel("Contaminant burden (mg/kg lipid weight)")
    ax.set_xlabel("Food-web compartment, increasing trophic position")
    ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax)
    ax.set_title("Burden rises by three orders of magnitude from algae to bird",
                 fontsize=9.0, fontweight="bold", loc="left", pad=5)
    ax.text(0.015, 0.720,
            f"overall magnification {conc[-1] / conc[0]:,.0f}$\\times$\n"
            f"median of n = {reps[0].size} individuals per compartment",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB, linespacing=1.5)

    # ---- (b) trophic magnification factor ---------------------------------
    xs = np.linspace(tl.min() - 0.25, tl.max() + 0.25, 100)
    ax_reg.plot(xs, 10.0 ** (intercept + slope * xs), lw=1.4, color="#2e7d5b",
                zorder=4)
    for i in range(len(WEB)):
        ax_reg.plot(tl[i] + rng.uniform(-0.05, 0.05, reps[i].size), reps[i],
                    ls="none", marker="o", ms=2.4, mfc="white",
                    mec="#9a9a9a", mew=0.6, zorder=3)
        ax_reg.plot(tl[i], conc[i], marker="o", ms=4.2, mfc=cols[i],
                    mec="white", mew=0.8, zorder=5)
    ax_reg.set_yscale("log")
    ax_reg.set_xlim(tl.min() - 0.35, tl.max() + 0.35)
    ax_reg.set_ylim(base * 1.4, conc.max() * 4.5)
    ax_reg.set_xlabel("Trophic position from $\\delta^{15}$N (-)", fontsize=8.0)
    ax_reg.set_ylabel("Burden (mg/kg lipid)", fontsize=8.0)
    ax_reg.tick_params(labelsize=6.8)
    ax_reg.grid(color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_reg)
    ax_reg.text(0.035, 0.955,
                f"$\\log_{{10}}C$ = {slope:.2f} TL {intercept:+.2f}\n"
                f"TMF = {tmf:.1f}$\\times$ per level    $R^2$ = {r2:.3f}",
                transform=ax_reg.transAxes, ha="left", va="top",
                fontsize=6.6, color=INK, linespacing=1.5)
    ax_reg.set_title("Trophic magnification factor", fontsize=8.5,
                     fontweight="bold", loc="left", pad=4)

    # ---- (c) tissue distribution ------------------------------------------
    shares = np.array([t[1] for t in TISSUE])
    shares = shares / shares.sum()
    tcols = get_palette("cb_sequential_green", 7)[2:6]
    ax_pie.pie(shares, radius=1.0, startangle=90, counterclock=True,
               colors=tcols, labels=None,
               wedgeprops=dict(width=0.42, edgecolor="white", linewidth=0.8))
    items, a0 = [], 90.0
    for (name, _), s, c in zip(TISSUE, shares, tcols):
        a1 = a0 + 360.0 * s
        items.append((0.5 * (a0 + a1), name,
                      f"{100.0 * s:.0f}%  ({s * BURDEN:.2f} mg/kg)", c))
        a0 = a1
    place_outside_labels(ax_pie, items, r_edge=1.0, r_stub=1.08, x_knee=1.18,
                         x_text=1.26, min_gap=0.46, y_span=1.05,
                         name_size=6.8, stat_size=6.2)
    ax_pie.text(0.0, 0.06, f"{BURDEN:.1f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=INK)
    ax_pie.text(0.0, -0.16, "mg/kg lipid", ha="center", va="center",
                fontsize=6.4, color=MUTED)
    ax_pie.set_xlim(-2.35, 2.35)
    ax_pie.set_ylim(-1.18, 1.18)
    ax_pie.set_aspect("equal")
    ax_pie.axis("off")
    ax_pie.set_title("Where the burden sits in the predatory fish",
                     fontsize=8.5, fontweight="bold", loc="left", pad=4,
                     x=0.02)

    panel_labels_inches(fig, [ax, ax_reg, ax_pie], dx=-0.44, dy=0.09)


    print(f"TMF={tmf:.3f}  slope={slope:.3f}  R2={r2:.4f}  "
          f"BMF={np.round(bmf, 2)}")
    savefig(fig, "625_tox_bioaccumulation.png")


if __name__ == "__main__":
    main()
