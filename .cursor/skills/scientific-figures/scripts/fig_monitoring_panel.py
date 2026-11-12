"""Long-term biodiversity monitoring dashboard.

Top: three KPI cards, each pairing a headline statistic with a sparkline and a
signed change against the previous decade.  Middle: a species x year abundance
heatmap (rows ordered by trend) beside the Shannon diversity trajectory with a
detected changepoint.  Bottom: a full-width phenology strip comparing the
1995-2004 and 2015-2024 timing of six seasonal events.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes


DARK = "#1b5e4b"
GREEN = "#2e7d5b"
OCHRE = "#c08f3e"
TEAL = "#3d8b93"
RUST = "#a2593c"
INK = "#1a1a1a"
MUTED = "#82827a"

ABUND_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "abund", ["#f7f5ec", "#e2ead6", "#bcd8bf", "#8bbf9c", "#54a07a",
              "#2a7757", "#0d4a35"])

SPECIES = [
    ("Willow tit", -1.00), ("Spotted flycatcher", -0.86),
    ("Tree sparrow", -0.70), ("Wood warbler", -0.55),
    ("Song thrush", -0.34), ("Chiffchaff", -0.16),
    ("Wren", -0.05), ("Robin", 0.06), ("Chaffinch", 0.18),
    ("Great tit", 0.34), ("Blue tit", 0.48), ("Nuthatch", 0.64),
    ("Blackcap", 0.82), ("Goldfinch", 1.00),
]

#     event                  hist start/end   recent start/end   colour
PHENO = [
    ("Oak bud burst", 108, 124, 96, 113, GREEN),
    ("Understorey flowering", 128, 168, 118, 159, GREEN),
    ("Insect emergence peak", 134, 182, 122, 172, OCHRE),
    ("Bird first egg", 116, 148, 107, 141, TEAL),
    ("Bird fledging", 152, 186, 143, 178, TEAL),
    ("Leaf senescence", 274, 306, 286, 319, RUST),
]

MONTHS = [(60, "Mar"), (91, "Apr"), (121, "May"), (152, "Jun"),
          (182, "Jul"), (213, "Aug"), (244, "Sep"), (274, "Oct"),
          (305, "Nov")]


def panel_tag(fig, ax, label, dx=-0.050, dy=0.018, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def kpi_card(ax, title, value, unit, delta, good, spark, spark_col):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.012, 0.03), 0.976, 0.94,
        boxstyle="round,pad=0,rounding_size=0.055", facecolor="#f5f6f2",
        edgecolor="#e0e2db", linewidth=0.8, clip_on=False, zorder=1))
    ax.text(0.058, 0.82, title, fontsize=6.8, va="center", color=MUTED,
            zorder=3)
    ax.text(0.058, 0.47, value, fontsize=15.5, va="center", color=INK,
            fontweight="bold", zorder=3)
    ax.text(0.058, 0.125, unit, fontsize=6.2, va="center", color=MUTED,
            zorder=3)
    dcol = GREEN if good else RUST
    ax.text(0.955, 0.83, delta, fontsize=8.0, va="center", ha="right",
            color=dcol, fontweight="bold", zorder=3)
    ax.text(0.955, 0.66, "vs 2014", fontsize=6.0, va="center", ha="right",
            color=MUTED, zorder=3)

    sx = np.linspace(0.52, 0.945, spark.size)
    lo, hi = spark.min(), spark.max()
    sy = 0.205 + 0.29 * (spark - lo) / (hi - lo)
    ax.plot(sx, sy, color=spark_col, lw=1.1, zorder=3)
    ax.fill_between(sx, 0.195, sy, color=spark_col, alpha=0.16, lw=0,
                    zorder=2)
    ax.scatter([sx[-1]], [sy[-1]], s=11, color=spark_col, edgecolor="white",
               linewidth=0.7, zorder=4)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260232)

    years = np.arange(2001, 2025)
    ny = years.size

    fig = plt.figure(figsize=(7.2, 6.1))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.40, 1.0, 0.68],
                     hspace=0.44)
    kpi_gs = outer[0].subgridspec(1, 3, wspace=0.10)
    mid_gs = outer[1].subgridspec(1, 2, width_ratios=[1.52, 1.0], wspace=0.46)
    kpis = [fig.add_subplot(kpi_gs[i]) for i in range(3)]
    ax_hm = fig.add_subplot(mid_gs[0])
    ax_div = fig.add_subplot(mid_gs[1])
    ax_ph = fig.add_subplot(outer[2])

    # ---- (a) KPI cards ----------------------------------------------------
    sp_rich = 112 + np.cumsum(rng.normal(0.7, 1.6, ny))
    tot_ab = 5200 * np.exp(np.cumsum(rng.normal(-0.008, 0.020, ny)))
    shannon = 3.05 + np.cumsum(rng.normal(0.016, 0.030, ny))

    kpi_card(kpis[0], "Species richness", f"{sp_rich[-1]:.0f}",
             "species recorded", "+6", True,
             gaussian_filter1d(sp_rich, 1.2), DARK)
    kpi_card(kpis[1], "Total abundance", f"{tot_ab[-1]:,.0f}".replace(",", " "),
             "individuals / 100 ha", "$-$12 %", False,
             gaussian_filter1d(tot_ab, 1.2), RUST)
    kpi_card(kpis[2], "Shannon diversity", f"{shannon[-1]:.2f}",
             "H' on log$_e$ scale", "+0.18", True,
             gaussian_filter1d(shannon, 1.2), TEAL)

    # ---- (b) species x year heatmap ---------------------------------------
    names = [s[0] for s in SPECIES]
    trends = np.array([s[1] for s in SPECIES])
    t = (years - years[0]) / (ny - 1.0)
    base = rng.uniform(44, 62, len(SPECIES))[:, None]
    mat = base * (1 + 0.78 * trends[:, None] * (t[None, :] - 0.5) * 2)
    mat = mat * np.exp(gaussian_filter1d(rng.normal(0, 0.16, mat.shape),
                                         1.6, axis=1))
    mat = np.clip(mat, 2, None)

    im = ax_hm.pcolormesh(years, np.arange(len(SPECIES)), mat,
                          cmap=ABUND_CMAP, shading="nearest",
                          rasterized=True)
    ax_hm.set_yticks(np.arange(len(SPECIES)))
    ax_hm.set_yticklabels(names, fontsize=6.0)
    ax_hm.set_xticks(np.arange(2002, 2025, 4))
    ax_hm.tick_params(labelsize=6.3, length=2.0)
    ax_hm.set_xlabel("Year", fontsize=7.4, labelpad=2)
    ax_hm.set_ylim(len(SPECIES) - 0.5, -0.5)
    for sp in ax_hm.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_edgecolor("#b9b9b1")
    ax_hm.set_title("Annual abundance index by species (rows ordered by "
                    "trend)", fontsize=7.6, loc="left", pad=4)

    cax = ax_hm.inset_axes([1.022, 0.0, 0.020, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Individuals per 100 ha", fontsize=6.4, labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#b9b9b1")
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.0, pad=1.5)

    # ---- (c) diversity trend with changepoint -----------------------------
    cp = 2011
    h = np.where(years <= cp,
                 3.16 + 0.004 * (years - 2001),
                 3.19 + 0.031 * (years - cp))
    h = h + rng.normal(0, 0.035, ny)
    se = 0.055 + 0.012 * np.abs(np.linspace(-1, 1, ny))

    ax_div.set_facecolor("#fbfbf8")
    ax_div.fill_between(years, h - 1.96 * se, h + 1.96 * se, color=GREEN,
                        alpha=0.16, lw=0, zorder=2)
    ax_div.plot(years, h, color=DARK, lw=1.5, zorder=4)
    ax_div.scatter(years, h, s=8, color=DARK, zorder=5)

    pre, post = years <= cp, years >= cp
    ax_div.plot(years[pre], np.full(pre.sum(), h[pre].mean()), color=MUTED,
                lw=0.9, ls=(0, (3.5, 2.0)), zorder=6)
    ax_div.plot(years[post], np.full(post.sum(), h[post].mean()), color=RUST,
                lw=0.9, ls=(0, (3.5, 2.0)), zorder=6)
    ax_div.axvline(cp, color=RUST, lw=0.95, ls=(0, (2.5, 2.0)), zorder=3)
    ax_div.text(cp - 0.7, 3.655, f"changepoint\n{cp}", fontsize=6.3,
                color=RUST, ha="right", va="top", linespacing=1.3,
                fontweight="bold", zorder=7)

    ax_div.set_xlim(2000.4, 2024.6)
    ax_div.set_ylim(3.02, 3.68)
    ax_div.set_xticks(np.arange(2002, 2025, 5))
    ax_div.tick_params(labelsize=6.3, length=2.0)
    ax_div.set_xlabel("Year", fontsize=7.4, labelpad=2)
    ax_div.set_ylabel("Shannon diversity H'", fontsize=7.4, labelpad=2)
    ax_div.grid(axis="y", lw=0.45, color="#ebebe5")
    style_axes(ax_div)
    ax_div.set_title("Diversity trajectory", fontsize=7.6, loc="left", pad=4)
    ax_div.text(0.965, 0.048,
                "pre  0.004 H' yr$^{-1}$\npost $\\bf{0.031}$ H' yr$^{-1}$",
                transform=ax_div.transAxes, fontsize=6.2, va="bottom",
                ha="right", color="#4a4a44", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#dededa", linewidth=0.55, alpha=0.93))

    # ---- (d) phenology strip ----------------------------------------------
    ax_ph.set_facecolor("#fbfbf8")
    for i, (name, h0, h1, r0, r1, col) in enumerate(PHENO):
        y = len(PHENO) - 1 - i
        ax_ph.add_patch(Rectangle((h0, y + 0.06), h1 - h0, 0.28,
                                  facecolor=col, alpha=0.28,
                                  edgecolor="white", linewidth=0.5, zorder=3))
        ax_ph.add_patch(Rectangle((r0, y - 0.34), r1 - r0, 0.28,
                                  facecolor=col, alpha=0.92,
                                  edgecolor="white", linewidth=0.5, zorder=3))
        shift = r0 - h0
        ax_ph.annotate("", xy=(r0, y - 0.04), xytext=(h0, y + 0.02),
                       arrowprops=dict(arrowstyle="-|>", color="#6f6f66",
                                       lw=0.7, shrinkA=0.5, shrinkB=0.5),
                       zorder=5)
        scol = GREEN if shift < 0 else RUST
        ax_ph.text(max(h1, r1) + 5, y - 0.14, f"{shift:+d} d", fontsize=6.3,
                   color=scol, va="center", ha="left", fontweight="bold",
                   zorder=5)

    ax_ph.set_yticks(np.arange(len(PHENO)))
    ax_ph.set_yticklabels([p[0] for p in PHENO][::-1], fontsize=6.5)
    ax_ph.set_xticks([m[0] for m in MONTHS])
    ax_ph.set_xticklabels([m[1] for m in MONTHS], fontsize=6.3)
    ax_ph.set_xlim(82, 348)
    ax_ph.set_ylim(-0.72, len(PHENO) - 0.28)
    ax_ph.tick_params(length=2.0)
    ax_ph.set_xlabel("Day of year", fontsize=7.4, labelpad=2)
    ax_ph.grid(axis="x", lw=0.45, color="#eaeae4")
    style_axes(ax_ph)
    ax_ph.set_title("Shift in seasonal event timing", fontsize=7.6,
                    loc="left", pad=4)

    ph_key = [
        Line2D([], [], marker="s", ls="", ms=5.2, mfc="#b9cfc0",
               mec="white", mew=0.5, label="1995-2004"),
        Line2D([], [], marker="s", ls="", ms=5.2, mfc=GREEN, mec="white",
               mew=0.5, label="2015-2024"),
    ]
    lp = ax_ph.legend(handles=ph_key, loc="lower right", ncol=2,
                      bbox_to_anchor=(1.002, 1.002), fontsize=6.4,
                      frameon=False, handletextpad=0.35, columnspacing=1.1,
                      borderaxespad=0.0)
    lp.set_clip_on(False)

    fig.subplots_adjust(left=0.034, right=0.965, top=0.912, bottom=0.062)

    # The heatmap and the phenology strip carry long row labels; inset only
    # those two so the KPI row and the strip still share one content block.
    for ax in (ax_hm, ax_ph):
        bb = ax.get_position()
        ax.set_position([bb.x0 + 0.098, bb.y0, bb.width - 0.098, bb.height])

    fig.canvas.draw()
    # every tag hugs the left edge of its own panel block by the same margin
    panel_tag(fig, kpis[0], "(a)", dx=-0.022, dy=0.014)
    panel_tag(fig, ax_hm, "(b)", dx=-0.120, dy=0.026)
    panel_tag(fig, ax_div, "(c)", dx=-0.058, dy=0.026)
    panel_tag(fig, ax_ph, "(d)", dx=-0.120, dy=0.026)

    savefig(fig, "232_monitoring_panel.png")


if __name__ == "__main__":
    main()
