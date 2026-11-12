"""Population pyramids for two demographic regimes, with a 1995 overlay.

Each panel shows males (left) and females (right) as a share of the total
population in five-year age groups, with the 1995 age structure drawn as a
dashed outline for comparison and the age-group labels running down the
middle.  A strip of indicator cards under each pyramid reports median age
and the dependency ratios implied by the same distribution.

Cohort sizes come from a births x survivorship model, so the shapes are
demographically consistent (an expansive pyramid vs a constrictive one).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

MALE, FEMALE = "#4c78a8", "#c07f9a"
OUTLINE = "#4d4d4d"
GAP = 0.62          # half-width of the central label gutter, in % units
STARTS = np.arange(0, 90, 5)
LABELS = [f"{a}-{a + 4}" for a in STARTS[:-1]] + ["85+"]


def births_index(year, regime):
    """Relative birth-cohort size for a given calendar year."""
    y = np.asarray(year, dtype=float)
    if regime == "post":
        boom = 0.42 * np.exp(-0.5 * ((y - 1949) / 3.0) ** 2)
        echo = 0.24 * np.exp(-0.5 * ((y - 1973) / 4.0) ** 2)
        decline = 0.45 + 0.55 / (1.0 + np.exp((y - 1988) / 9.0))
        return (1.0 + boom + echo) * decline
    rise = 0.34 + 0.66 / (1.0 + np.exp(-(y - 1972) / 12.0))
    late = 1.0 - 0.16 / (1.0 + np.exp(-(y - 2008) / 6.0))
    return rise * late


def survivorship(age, scale, shape):
    return np.exp(-((age / scale) ** shape))


def structure(ref_year, regime, scale, shape):
    """Return male and female shares (% of total) by five-year age group."""
    mid = STARTS + 2.5
    mid[-1] = 89.0                       # open-ended 85+ group
    b = births_index(ref_year - mid, regime)
    m = b * 0.5122 * survivorship(mid, scale * 0.955, shape)
    f = b * 0.4878 * survivorship(mid, scale * 1.045, shape)
    tot = m.sum() + f.sum()
    return 100.0 * m / tot, 100.0 * f / tot


def indicators(m, f):
    tot = m + f
    child = tot[:3].sum()                # 0-14
    old = tot[13:].sum()                 # 65+
    work = tot[3:13].sum()               # 15-64
    cum = np.cumsum(tot)
    k = int(np.searchsorted(cum, 50.0))
    prev = cum[k - 1] if k else 0.0
    median = STARTS[k] + 5.0 * (50.0 - prev) / tot[k]
    return {
        "Median age (yr)": f"{median:.1f}",
        "Total dep. (%)": f"{100 * (child + old) / work:.0f}",
        "Old-age dep. (%)": f"{100 * old / work:.0f}",
        "Aged 65+ (%)": f"{old:.1f}",
    }


def outline_path(ax, vals, sign, color):
    xs, ys = [0.0], [-0.5]
    for i, v in enumerate(vals):
        xs += [sign * (GAP + v), sign * (GAP + v)]
        ys += [i - 0.5, i + 0.5]
    xs.append(0.0)
    ys.append(len(vals) - 0.5)
    ax.plot(xs, ys, color=color, lw=0.95, ls=(0, (3.5, 2.2)), zorder=6,
            solid_capstyle="butt")


def draw_pyramid(ax, m, f, m_old, f_old, title, xmax):
    y = np.arange(len(STARTS))
    ax.barh(y, -m, left=-GAP, height=0.80, color=MALE, edgecolor="white",
            lw=0.5, zorder=3)
    ax.barh(y, f, left=GAP, height=0.80, color=FEMALE, edgecolor="white",
            lw=0.5, zorder=3)
    outline_path(ax, m_old, -1, OUTLINE)
    outline_path(ax, f_old, +1, OUTLINE)

    halo = [pe.withStroke(linewidth=1.7, foreground="white")]
    for i in y:
        ax.text(-(GAP + m[i] + 0.10), i, f"{m[i]:.1f}", ha="right",
                va="center", fontsize=6.0, color="#3a3a3a",
                path_effects=halo, zorder=7)
        ax.text(GAP + f[i] + 0.10, i, f"{f[i]:.1f}", ha="left",
                va="center", fontsize=6.0, color="#3a3a3a",
                path_effects=halo, zorder=7)
        ax.text(0, i, LABELS[i], ha="center", va="center", fontsize=6.0,
                color="#333333", zorder=7)

    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=2)
    ticks = np.array([0.0, 2.0, 4.0, 6.0])
    ticks = ticks[ticks <= xmax - GAP]
    ax.set_xticks(np.concatenate([-(GAP + ticks[::-1]), GAP + ticks]))
    ax.set_xticklabels([f"{t:.0f}" for t in ticks[::-1]] +
                       [f"{t:.0f}" for t in ticks], fontsize=7)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-0.75, len(STARTS) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("Share of total population (%)")
    ax.set_title(title, fontsize=8.8, loc="center", pad=14, weight="bold")
    ax.text(-xmax * 0.72, len(STARTS) - 0.6, "MALE", fontsize=7.2,
            color=MALE, fontweight="bold", ha="center", va="center")
    ax.text(xmax * 0.72, len(STARTS) - 0.6, "FEMALE", fontsize=7.2,
            color=FEMALE, fontweight="bold", ha="center", va="center")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)


def draw_cards(ax, metrics, accent):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(metrics)
    w = 1.0 / n
    for j, (lab, val) in enumerate(metrics.items()):
        x0 = j * w
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.014, 0.10), w - 0.028, 0.80,
            boxstyle="round,pad=0.004,rounding_size=0.03",
            facecolor="#f4f5f6", edgecolor="#dcdcdc", lw=0.6, zorder=1))
        ax.text(x0 + w / 2, 0.63, val, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=accent, zorder=2)
        ax.text(x0 + w / 2, 0.27, lab, ha="center", va="center",
                fontsize=6.2, color="#5a5a5a", zorder=2)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(7.4, 5.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.155],
                  hspace=0.24, wspace=0.20, left=0.055, right=0.975,
                  top=0.865, bottom=0.058)

    cases = [
        ("Region A - post-transition (2025)", "post", 88.0, 6.2, "#2f4f6f"),
        ("Region B - late transition (2025)", "late", 74.0, 4.6, "#7a4356"),
    ]

    xmax = 0.0
    packed = []
    for _, regime, scale, shape, _accent in cases:
        m, f = structure(2025, regime, scale, shape)
        mo, fo = structure(1995, regime, scale * 0.965, shape)
        packed.append((m, f, mo, fo))
        xmax = max(xmax, m.max(), f.max(), mo.max(), fo.max())
    xmax = GAP + xmax + 0.95

    axes = []
    for col, ((title, _, _, _, accent), (m, f, mo, fo)) in enumerate(
            zip(cases, packed)):
        ax = fig.add_subplot(gs[0, col])
        draw_pyramid(ax, m, f, mo, fo, title, xmax)
        axc = fig.add_subplot(gs[1, col])
        draw_cards(axc, indicators(m, f), accent)
        axes.append(ax)

    add_panel_label(axes[0], "(a)", x=-0.045, y=1.045)
    add_panel_label(axes[1], "(b)", x=-0.045, y=1.045)

    handles = [
        Rectangle((0, 0), 1, 1, fc=MALE, ec="white", lw=0.5, label="Male, 2025"),
        Rectangle((0, 0), 1, 1, fc=FEMALE, ec="white", lw=0.5,
                  label="Female, 2025"),
        Line2D([], [], color=OUTLINE, lw=1.0, ls=(0, (3.5, 2.2)),
               label="1995 age structure"),
    ]

    savefig(fig, "106_population_pyramid.png")


if __name__ == "__main__":
    main()
