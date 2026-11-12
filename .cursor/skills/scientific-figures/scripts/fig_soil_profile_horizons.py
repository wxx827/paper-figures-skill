"""Soil profile: genetic horizons plus co-registered physico-chemical curves.

The pedon column carries the message and is read the way a soil scientist
reads one: every genetic horizon gets its own hatch ornament as well as an
approximate Munsell face colour, because the pattern survives photocopying
and thumbnailing while the colour does not.  Depth increases downward on a
single shared axis, so the four property curves to the right (organic matter,
pH, bulk density, clay) line up horizon for horizon with the column, and the
horizon contacts are repeated as dashed rules across every panel.  Horizon
descriptions sit furthest right, level with the horizon they describe.

Values are synthetic but internally consistent: the clay bulge in the B
horizon, the ratio quoted in the footnote and the arrow annotation all come
from the same smoothed depth series.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, darken
from figkit import halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"

# name, top (cm), base (cm), approximate Munsell face colour, hatch, note.
# The hatch is the domain convention (DESIGN_SYSTEM section 2.2); two-character
# densities keep the ornament legible without turning a thick horizon grey.
HORIZONS = [
    ("O", 0.0, 6.0, "#4a3a26", "//",
     "Leaf litter and well humified organic mat"),
    ("A", 6.0, 28.0, "#6d5334", "..",
     "Dark mineral topsoil, granular peds, root dense"),
    ("E", 28.0, 46.0, "#c3b291", "--",
     "Pale eluvial horizon, clay and iron depleted"),
    ("B", 46.0, 96.0, "#9c5f34", "\\\\",
     "Reddish illuvial horizon, clay skins on ped faces"),
    ("C", 96.0, 132.0, "#bda079", "xx",
     "Weathered parent material, faint mottles, few roots"),
    ("R", 132.0, 150.0, "#8d949b", "++",
     "Fractured sandstone bedrock, root limiting layer"),
]

# per-horizon anchor values for the four property curves
ANCHOR = {
    "om":   {"O": 34.0, "A": 4.6, "E": 1.1, "B": 0.62, "C": 0.30, "R": 0.10},
    "ph":   {"O": 5.05, "A": 5.50, "E": 5.85, "B": 6.35, "C": 7.15,
             "R": 7.45},
    "bd":   {"O": 0.32, "A": 1.14, "E": 1.38, "B": 1.52, "C": 1.64,
             "R": 1.98},
    "clay": {"O": 4.0, "A": 12.5, "E": 8.0, "B": 33.0, "C": 17.5, "R": 6.0},
}
NOISE = {"om": 0.055, "ph": 0.045, "bd": 0.022, "clay": 0.060}

SOIL_BROWN = "#7a5230"
CURVES = [
    ("om", "Organic matter", "%", SOIL_BROWN, True),
    ("ph", "pH", "(-)", "#3b5b92", False),
    ("bd", "Bulk density", "g cm$^{-3}$", "#555555", False),
    ("clay", "Clay content", "%", "#b23a48", False),
]

ZMAX = 150.0


def build_profile(rng):
    """Smoothed depth series for each property, plus the depth vector."""
    z = np.linspace(0.0, ZMAX, 601)
    out = {}
    for key, anchors in ANCHOR.items():
        step = np.zeros_like(z)
        for name, top, base, *_ in HORIZONS:
            step[(z >= top) & (z <= base)] = anchors[name]
        # relative noise so the log-scaled organic-matter curve stays positive
        wobble = gaussian_filter1d(rng.normal(0.0, 1.0, z.size), 6.0)
        wobble /= np.abs(wobble).max()
        prof = step * (1.0 + NOISE[key] * wobble) if key != "ph" \
            else step + NOISE[key] * 4.0 * wobble
        out[key] = gaussian_filter1d(prof, 9.0)
    return z, out


def horizon_mean(z, values, name):
    top, base = [(t, b) for n, t, b, *_ in HORIZONS if n == name][0]
    m = (z >= top) & (z <= base)
    return float(values[m].mean())


def spread(vals, gap, lo, hi):
    """Push label anchors apart while preserving order, so leaders cannot cross."""
    y = [float(v) for v in vals]
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    if y[-1] > hi:
        y[-1] = hi
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - gap)
    y[0] = max(y[0], lo)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    return y


def main():
    set_pub_style(**{"hatch.linewidth": 0.42})
    rng = np.random.default_rng(600)
    z, prof = build_profile(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[0.86, 0.50, 0.50, 0.50, 0.50,
                                              1.30],
                          left=0.062, right=0.985, top=0.855, bottom=0.180,
                          wspace=0.30)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_c = [fig.add_subplot(gs[0, i + 1], sharey=ax_p) for i in range(4)]
    ax_d = fig.add_subplot(gs[0, 5], sharey=ax_p)

    # ---- pedon column ------------------------------------------------------
    for name, top, base, fc, hatch, _ in HORIZONS:
        ec = darken(fc, 0.40)
        ax_p.add_patch(Rectangle((0.0, top), 1.0, base - top, facecolor=fc,
                                 edgecolor=ec, linewidth=0.7, hatch=hatch,
                                 zorder=2))
        halo_text(ax_p, 0.5, 0.5 * (top + base), name, ha="center",
                  va="center", fontsize=9.0, fontweight="bold",
                  color="white", halo="#1a1a1a", halo_lw=1.9, zorder=6)

    ax_p.set_xlim(0.0, 1.0)
    ax_p.set_ylim(ZMAX, 0.0)
    ax_p.set_yticks(np.arange(0, ZMAX + 1, 20))
    ax_p.set_ylabel("Depth below surface (cm)", fontsize=8.0)
    ax_p.set_xticks([])
    ax_p.set_title("Pedon", fontsize=8.5, fontweight="bold")
    for s in ax_p.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)

    # ---- property curves ---------------------------------------------------
    contacts = [b for _, _, b, *_ in HORIZONS[:-1]]
    for ax, (key, name, unit, col, logx) in zip(ax_c, CURVES):
        v = prof[key]
        if logx:
            ax.set_xscale("log")
            lo, hi = 0.07, 60.0
            ax.set_xticks([0.1, 1.0, 10.0])
            ax.set_xticklabels(["0.1", "1", "10"])
            ax.fill_betweenx(z, lo, v, color=col, alpha=0.16, lw=0)
        else:
            span = v.max() - v.min()
            lo, hi = v.min() - 0.18 * span, v.max() + 0.18 * span
            ax.fill_betweenx(z, lo, v, color=col, alpha=0.16, lw=0)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.plot(v, z, color=col, lw=1.4, zorder=4)
        ax.set_xlim(lo, hi)
        for cz in contacts:
            ax.axhline(cz, color="#cccccc", lw=0.55, ls=(0, (3, 2)),
                       zorder=1)
        ax.set_title(name, fontsize=7.8, fontweight="bold")
        ax.set_xlabel(unit, fontsize=7.2, color=SUB, labelpad=1.5)
        ax.tick_params(labelleft=False, labelsize=6.6)
        ax.tick_params(axis="y", length=2.0)

    # accent: the argillic clay maximum, measured off the curve that is drawn
    clay = prof["clay"]
    i_max = int(np.argmax(np.where(z <= 132.0, clay, -np.inf)))
    ax_c[3].annotate(f"clay max\n{clay[i_max]:.0f}% at {z[i_max]:.0f} cm",
                     xy=(clay[i_max], z[i_max]),
                     xytext=(0.02, 0.30), textcoords="axes fraction",
                     fontsize=6.4, color="#b23a48", ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color="#b23a48",
                                     lw=0.7,
                                     connectionstyle="arc3,rad=-0.28"))

    # ---- horizon descriptions ---------------------------------------------
    ax_d.set_xlim(0.0, 1.0)
    ax_d.axis("off")
    mids = [0.5 * (t + b) for _, t, b, *_ in HORIZONS]
    ys = spread(mids, gap=22.0, lo=13.0, hi=ZMAX - 8.0)
    for (name, top, base, fc, hatch, note), zm, y in zip(HORIZONS, mids, ys):
        ax_d.add_patch(Rectangle((0.0, top), 0.070, base - top, facecolor=fc,
                                 edgecolor=darken(fc, 0.40), linewidth=0.6,
                                 hatch=hatch, zorder=2))
        ax_d.plot([0.070, 0.105, 0.125], [zm, y, y], color="#cccccc", lw=0.6,
                  zorder=1)
        ax_d.text(0.138, y - 7.0,
                  f"{name} horizon   {top:.0f}-{base:.0f} cm", ha="left",
                  va="center", fontsize=7.0, color=INK, fontweight="bold")
        ax_d.text(0.138, y + 6.0, textwrap.fill(note, 36), ha="left",
                  va="center", fontsize=6.6, color=SUB, linespacing=1.35)
    ax_d.set_title("Horizon description", fontsize=8.5, fontweight="bold",
                   loc="left", x=0.0)

    # ---- shared ornament key ----------------------------------------------
    handles = [Rectangle((0, 0), 1, 1, facecolor=fc, hatch=h,
                         edgecolor=darken(fc, 0.40), linewidth=0.6,
                         label=f"{n} horizon")
               for n, _, _, fc, h, _ in HORIZONS]
    handles.append(Line2D([0], [0], color="#cccccc", lw=0.8, ls=(0, (3, 2)),
                          label="horizon contact"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.098), ncol=7, frameon=False,
               fontsize=7.0, handlelength=1.5, handleheight=1.15,
               columnspacing=1.5)

    panel_labels_inches(fig, [ax_p] + ax_c, dx=-0.28, dy=0.20)

    clay_b = horizon_mean(z, clay, "B")
    clay_a = horizon_mean(z, clay, "A")
    om_a = horizon_mean(z, prof["om"], "A")
    solum = [b for n, _, b, *_ in HORIZONS if n == "B"][0]


    print(f"clay B={clay_b:.2f}%  A={clay_a:.2f}%  "
          f"ratio={clay_b / clay_a:.2f}")
    print(f"clay max {clay[i_max]:.1f}% at {z[i_max]:.0f} cm; "
          f"A-horizon OM {om_a:.2f}%")
    savefig(fig, "600_soil_profile_horizons.png")


if __name__ == "__main__":
    main()
