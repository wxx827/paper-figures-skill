"""Leaderboard table: 12 world models x 5 metrics as a shaded score table.

The five metrics carry different units and two different directions, so the
shading is not the raw value but the per-column normalised score (0 = worst
entry in that column, 1 = best), which puts every column on one shared blue
scale.  The printed value in each cell is still the raw metric.  The bold
outline in each column is placed by ``argmin`` / ``argmax`` under that column's
direction, and the mean-rank bar on the right is the row mean of the same
per-column ranks -- neither is written down by hand.

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

METHODS = [("DreamerV3", 2023), ("TD-MPC2", 2024), ("IRIS", 2022),
           ("GAIA-1", 2023), ("Genie", 2024), ("Genie-2", 2025),
           ("Sora-WM", 2025), ("RoboDreamer", 2024), ("EmbodiedWM", 2025),
           ("NavGen", 2026), ("WM-Agent", 2026), ("VideoGPT", 2021)]
# name, unit, direction (+1 = higher is better), value format
METRICS = [("FVD", "", -1, "{:.0f}"),
           # Four decimals: at three, two entries collide and the printed
           # table can no longer reproduce the mean ranks beside it.
           ("LPIPS", "", -1, "{:.4f}"),
           ("PSNR", "dB", +1, "{:.1f}"),
           ("Task success", "%", +1, "{:.1f}"),
           ("Rollout latency", "ms", -1, "{:.0f}")]


def gen_data(seed=1556):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_survey_leaderboard.csv", index_col=0)
        return list(zip(df.index, df["year"])), METRICS, df.iloc[:, 1:].to_numpy()
    """
    rng = np.random.default_rng(seed)
    n = len(METHODS)
    q = rng.normal(0.0, 1.0, n)                    # latent quality per method
    noise = rng.normal(0.0, 0.45, (n, len(METRICS)))
    z = q[:, None] + noise
    vals = np.column_stack([
        420.0 - 78.0 * z[:, 0],
        0.176 - 0.030 * z[:, 1],
        24.6 + 2.10 * z[:, 2],
        61.0 + 9.4 * z[:, 3],
        188.0 - 34.0 * z[:, 4],
    ])
    return METHODS, METRICS, vals


def text_colour(rgba):
    """White on dark fills, ink on light ones (relative luminance < 0.55)."""
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


def main():
    set_pub_style(base_font=8)
    methods, metrics, vals = gen_data()
    dirs = np.array([d for _, _, d, _ in metrics], dtype=float)

    # Per-column rank (1 = best under that column's direction).
    signed = vals * -dirs[None, :]                  # smaller is always better
    ranks = np.argsort(np.argsort(signed, axis=0), axis=0) + 1
    mean_rank = ranks.mean(axis=1)

    order = np.argsort(mean_rank, kind="stable")
    methods = [methods[i] for i in order]
    vals, ranks, signed = vals[order], ranks[order], signed[order]
    mean_rank = mean_rank[order]
    n, m = vals.shape

    # One shared 0-1 goodness scale for every column.
    best_v, worst_v = signed.min(axis=0), signed.max(axis=0)
    good = (worst_v[None, :] - signed) / (worst_v - best_v)[None, :]
    norm = Normalize(vmin=0.0, vmax=1.0)
    best_i = np.argmin(signed, axis=0)

    fig = plt.figure(figsize=(9.0, 6.8))
    axm = fig.add_axes([0.238, 0.212, 0.480, 0.588])
    axr = fig.add_axes([0.760, 0.212, 0.150, 0.588])

    axm.imshow(good, cmap=BLUE_CMAP, norm=norm, aspect="auto",
               interpolation="nearest",
               extent=[-0.5, m - 0.5, n - 0.5, -0.5])
    for i in range(n):
        for j in range(m):
            axm.text(j, i, metrics[j][3].format(vals[i, j]), ha="center",
                     va="center", fontsize=6.6,
                     color=text_colour(BLUE_CMAP(norm(good[i, j]))))
    for j in range(m):
        axm.add_patch(Rectangle((j - 0.5, best_i[j] - 0.5), 1.0, 1.0,
                                facecolor="none", edgecolor=ACCENT,
                                linewidth=1.7, zorder=5))

    axm.set_xlim(-0.5, m - 0.5)
    axm.set_ylim(n - 0.5, -0.5)
    axm.set_xticks(range(m))
    axm.set_xticklabels([f"{nm}\n({u})" if u else nm
                         for nm, u, _, _ in metrics], fontsize=7.4, color=INK,
                        linespacing=1.4)
    axm.xaxis.set_ticks_position("top")
    axm.set_yticks(range(n))
    axm.set_yticklabels([f"{nm}  {yr}" for nm, yr in methods], fontsize=7.4,
                        color=INK)
    axm.tick_params(length=0, pad=4)
    axm.set_xticks(np.arange(-0.5, m, 1), minor=True)
    axm.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axm.grid(which="minor", color="white", lw=0.8)
    axm.tick_params(which="minor", length=0)
    for s in axm.spines.values():
        s.set_visible(False)

    # Direction strip under the table.
    for j, (_, _, d, _) in enumerate(metrics):
        axm.text(j, n - 0.34,
                 "$\\uparrow$ higher\nis better" if d > 0
                 else "$\\downarrow$ lower\nis better",
                 ha="center", va="top", fontsize=6.4, color=COLORS["sub"],
                 linespacing=1.45, clip_on=False)

    # ---- mean-rank bars -----------------------------------------------------
    y = np.arange(n)
    bar_col = [BLUE[1] if i == 0 else BLUE[3] for i in range(n)]
    axr.barh(y, mean_rank, height=0.58, color=bar_col, edgecolor="white",
             linewidth=0.4, zorder=3)
    for i in range(n):
        axr.text(mean_rank[i] + 0.28, i, f"{mean_rank[i]:.1f}", ha="left",
                 va="center", fontsize=6.8, zorder=4, color=INK)
    axr.set_xlim(0, n + 1.4)
    axr.set_ylim(n - 0.5, -0.5)
    axr.set_yticks([])
    axr.set_xticks([1, n // 2, n])
    axr.tick_params(axis="x", labelsize=6.8, length=2.5)
    axr.set_xlabel(f"mean rank over {m:d} metrics\n(shorter is better, "
                   f"1 = best of {n:d})",
                   fontsize=7.0, color=COLORS["sub"], labelpad=2.0,
                   linespacing=1.45)
    axr.grid(axis="x", color="#e6eaee", lw=0.6)
    axr.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axr.spines[side].set_visible(False)
    axr.set_title("Mean rank", fontsize=7.6, fontweight="bold", color=INK,
                  pad=22)

    cax = fig.add_axes([0.252, 0.086, 0.190, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUE_CMAP),
                      cax=cax, orientation="horizontal", ticks=[0.0, 1.0])
    cb.ax.set_xticklabels(["worst in column", "best in column"])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.6, color=INK,
                      labelcolor=INK)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)

    top_name, top_year = methods[0]
    wins = int((best_i == 0).sum())
    fig.text(0.520, 0.086,
             f"Warm outline = best entry in that column, placed by argmin / "
             f"argmax under\nthe column's direction. Rows sorted by mean rank, "
             f"so {top_name} ({top_year}) is first.",
             ha="left", va="bottom", fontsize=6.5, color=COLORS["muted"],
             linespacing=1.6)

    savefig(fig, "1151_wm_survey_leaderboard.png")


if __name__ == "__main__":
    main()
