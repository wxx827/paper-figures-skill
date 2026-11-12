"""Survey venue split: ranked publication counts beside a venue-by-year grid.

Left, one horizontal bar per venue, sorted by total and shaded down a single
blue ladder so that rank reads as tone; right, the same venues in the same row
order as an eight-year count grid on one shared blue scale.  The bar totals are
the row sums of the grid, so the two panels can be checked against each other
by eye, and every printed number is formatted straight from the matrix.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from pubstyle import COLORS, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
BLUE_CMAP = LinearSegmentedColormap.from_list("wm_blue", BLUE[::-1])

# (venue, kind); kind "C" = conference / workshop, "J" = journal
VENUES = [("CVPR", "C"), ("NeurIPS", "C"), ("ICLR", "C"), ("ICRA", "C"),
          ("ICCV / ECCV", "C"), ("CoRL", "C"), ("ICML", "C"), ("IROS", "C"),
          ("AAAI", "C"), ("T-RO", "J"), ("RA-L", "J"), ("TPAMI", "J"),
          ("IJCV", "J"), ("T-ITS", "J")]
YEARS = np.arange(2019, 2027)


def gen_data(seed=1553):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_survey_venue_split.csv", index_col=0)
        return [(v, k) for v, k in zip(df.index, df["kind"])], \
               df.columns.to_numpy(), df.iloc[:, 1:].to_numpy()
    """
    rng = np.random.default_rng(seed)
    n, m = len(VENUES), YEARS.size
    scale = np.array([34, 30, 27, 24, 23, 17, 16, 15, 12,
                      9, 9, 8, 6, 6], dtype=float)
    growth = np.linspace(0.30, 1.55, m) ** 1.7
    # Journals ramp later than conferences.
    lag = np.array([1.0 if k == "C" else 0.62 for _, k in VENUES])
    base = scale[:, None] * growth[None, :] * lag[:, None]
    base *= rng.uniform(0.72, 1.30, base.shape)
    counts = np.maximum(np.rint(base).astype(int), 0)
    return VENUES, YEARS, counts


def text_colour(rgba):
    """White on dark fills, ink on light ones (relative luminance < 0.55)."""
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


def main():
    set_pub_style(base_font=8)
    venues, years, counts = gen_data()

    order = np.argsort(-counts.sum(axis=1), kind="stable")
    venues = [venues[i] for i in order]
    counts = counts[order]
    totals = counts.sum(axis=1)
    grand = int(counts.sum())
    n, m = counts.shape

    kinds = np.array([k for _, k in venues])
    n_conf = int((kinds == "C").sum())
    conf_papers = int(totals[kinds == "C"].sum())
    jour_papers = int(totals[kinds == "J"].sum())

    # One shared cell scale for the whole grid, fixed before anything is drawn.
    norm = Normalize(vmin=int(counts.min()), vmax=int(counts.max()))
    # Rank tone: darkest rung for the top venue, lightest for the last.
    bar_tone = [BLUE_CMAP(0.94 - 0.66 * i / (n - 1)) for i in range(n)]

    fig = plt.figure(figsize=(10.4, 6.4))
    axb = fig.add_axes([0.150, 0.152, 0.278, 0.688])
    axh = fig.add_axes([0.545, 0.152, 0.345, 0.688])

    # ---- left: ranked totals ------------------------------------------------
    y = np.arange(n)
    axb.barh(y, totals, height=0.66, color=bar_tone, edgecolor="white",
             linewidth=0.5, zorder=3)
    for i, tot in enumerate(totals):
        axb.text(tot + totals.max() * 0.022, i,
                 f"{tot:d}  ({tot / grand:.1%})", ha="left", va="center",
                 fontsize=6.8, color=INK, zorder=4)
    axb.set_xlim(0, totals.max() * 1.34)
    axb.set_ylim(n - 0.5, -0.5)
    axb.set_yticks(y)
    axb.set_yticklabels([f"{v}" for v, _ in venues], fontsize=7.4, color=INK)
    axb.set_xlabel(f"Papers {years[0]}-{years[-1]}", fontsize=8.0, color=INK)
    axb.tick_params(axis="y", length=0, pad=17)
    axb.tick_params(axis="x", labelsize=7.0)
    axb.grid(axis="x", color="#e3e7ea", lw=0.6)
    axb.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axb.spines[side].set_visible(False)

    # Conference / journal tag column, just left of the bars.
    for i, (_, kind) in enumerate(venues):
        axb.text(-0.043, i, kind, transform=axb.get_yaxis_transform(),
                 ha="center", va="center", fontsize=6.4, color="white",
                 zorder=5,
                 bbox=dict(boxstyle="square,pad=0.24",
                           facecolor=BLUE[1] if kind == "C" else BLUE[3],
                           edgecolor="none"))

    # ---- right: venue x year grid, same row order ---------------------------
    axh.imshow(counts, cmap=BLUE_CMAP, norm=norm, aspect="auto",
               interpolation="nearest",
               extent=[-0.5, m - 0.5, n - 0.5, -0.5])
    for i in range(n):
        for j in range(m):
            v = int(counts[i, j])
            axh.text(j, i, f"{v:d}", ha="center", va="center", fontsize=6.2,
                     color=text_colour(BLUE_CMAP(norm(v))))
    axh.set_xticks(range(m))
    axh.set_xticklabels([str(v) for v in years], fontsize=7.0, color=INK,
                        rotation=45, ha="right", rotation_mode="anchor")
    axh.set_yticks([])
    axh.set_xticks(np.arange(-0.5, m, 1), minor=True)
    axh.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axh.grid(which="minor", color="white", lw=0.7)
    axh.tick_params(which="minor", length=0)
    axh.tick_params(length=0, pad=2.5)
    for s in axh.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor("#8d979f")

    # Warm accent: the single busiest venue-year cell, outlined and named.
    bi, bj = np.unravel_index(int(np.argmax(counts)), counts.shape)
    axh.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, facecolor="none",
                            edgecolor=ACCENT, linewidth=1.6, zorder=5))

    cax = fig.add_axes([0.905, 0.152, 0.014, 0.225])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUE_CMAP),
                      cax=cax, ticks=[int(counts.min()), int(counts.max())])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.6, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_title("Papers\nper cell", fontsize=6.8, color=INK, pad=4,
                    linespacing=1.4)

    axh.set_title(f"Same {n:d} venues, same row order, by year",
                  fontsize=8.6, fontweight="bold", color=INK, pad=8)
    axb.set_title("Ranked by total, shaded by rank", fontsize=8.6,
                  fontweight="bold", color=INK, pad=8)


    savefig(fig, "1148_wm_survey_venue_split.png")


if __name__ == "__main__":
    main()
