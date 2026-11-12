"""Survey growth chart: world-model papers per year, split by subarea.

Eleven years of publication counts stacked into six subareas, one rung of the
house blue ladder each, with the running total drawn against a right-hand axis
in the single warm accent.  Every printed number -- the per-year totals above
the bars, the share in the legend, the growth call-outs -- is formatted from the
same integer array that the bars are drawn from, so the stack always sums to the
label on top of it.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import COLORS, savefig, set_pub_style

# BRIEF_WM_BLUE.md section 2 ladder; six subareas take six consecutive rungs.
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

SUBAREAS = ["Latent dynamics models", "Video prediction / generation",
            "Model-based RL agents", "Driving world models",
            "Embodied / robot world models", "Evaluation & benchmarks"]
YEARS = np.arange(2016, 2027)


def gen_data(seed=1552):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_survey_papers_by_year.csv", index_col=0)
        return df.index.to_numpy(), df.columns.tolist(), df.to_numpy().T
    """
    rng = np.random.default_rng(seed)
    # Logistic take-off per subarea: onset year, steepness, saturation level.
    onset = np.array([2017.0, 2019.6, 2017.8, 2020.4, 2021.3, 2022.1])
    steep = np.array([0.46, 0.72, 0.40, 0.62, 0.80, 0.86])
    ceiling = np.array([96.0, 214.0, 88.0, 132.0, 158.0, 74.0])
    t = YEARS[None, :].astype(float)
    curve = ceiling[:, None] / (1.0 + np.exp(-steep[:, None] *
                                             (t - onset[:, None])))
    curve *= rng.uniform(0.88, 1.12, curve.shape)
    counts = np.maximum(np.rint(curve).astype(int), 0)
    return YEARS, SUBAREAS, counts


def main():
    set_pub_style(base_font=8)
    years, subareas, counts = gen_data()
    totals = counts.sum(axis=0)
    cumulative = np.cumsum(totals)
    grand = int(cumulative[-1])

    # Inflection years, all derived from the arrays that get drawn.
    ratio = totals[1:] / np.maximum(totals[:-1], 1)
    jump_i = int(np.argmax(ratio)) + 1
    half_i = int(np.argmax(cumulative >= grand / 2.0))
    peak_i = int(np.argmax(totals))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = fig.add_axes([0.062, 0.115, 0.615, 0.735])
    axc = ax.twinx()

    bottom = np.zeros_like(totals, dtype=float)
    for k, (name, colour) in enumerate(zip(subareas, BLUE[:len(subareas)])):
        ax.bar(years, counts[k], bottom=bottom, width=0.70, color=colour,
               edgecolor="white", linewidth=0.55, zorder=3,
               label=f"{name}  ({counts[k].sum() / grand:.0%})")
        bottom += counts[k]

    # Totals live on the twin axes (drawn last) so the cumulative line cannot
    # cross them, but keep ax's data coordinates for placement.
    for x, tot in zip(years, totals):
        axc.text(x, tot + totals.max() * 0.022, f"{tot:d}", ha="center",
                 va="bottom", fontsize=6.6, color=INK, zorder=8,
                 transform=ax.transData,
                 bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                           edgecolor="none", alpha=0.9))

    axc.plot(years, cumulative, color=ACCENT, lw=1.7, zorder=5,
             marker="o", markersize=3.2, markerfacecolor="white",
             markeredgewidth=1.1, label="Cumulative total (right axis)")

    ax.set_ylim(0, totals.max() * 1.28)
    axc.set_ylim(0, cumulative.max() * 1.06)
    ax.set_xlim(years[0] - 0.72, years[-1] + 0.72)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=7.2, color=INK)
    ax.set_ylabel("Papers published per year", fontsize=8.0, color=INK)
    axc.set_ylabel(f"Cumulative papers (total {grand:d})", fontsize=8.0,
                   color=ACCENT)
    axc.tick_params(axis="y", colors=ACCENT, labelsize=7.2)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="y", color="#e3e7ea", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    axc.spines["top"].set_visible(False)
    axc.spines["right"].set_color(ACCENT)
    axc.spines["right"].set_visible(True)
    axc.spines["left"].set_visible(False)

    # ---- call-outs, every number read back off the arrays -------------------
    def flag(idx, text, tx, ty):
        ax.annotate(text, xy=(years[idx], totals[idx]), xytext=(tx, ty),
                    ha="center", va="center", fontsize=6.8, color=INK,
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="#c3ccd4", linewidth=0.6),
                    arrowprops=dict(arrowstyle="-", color="#8d979f", lw=0.7,
                                    shrinkA=2.0, shrinkB=3.0))

    top = totals.max()
    flag(jump_i, f"{years[jump_i]}: steepest jump\n"
                 f"{totals[jump_i - 1]:d} $\\rightarrow$ {totals[jump_i]:d} "
                 f"({ratio[jump_i - 1]:.2f}$\\times$)",
         tx=2018.7, ty=top * 0.55)
    flag(half_i, f"{years[half_i]}: half of the corpus\n"
                 f"({cumulative[half_i]:d} of {grand:d} papers)",
         tx=2020.5, ty=top * 0.955)
    flag(peak_i, f"{years[peak_i]}: peak year, {totals[peak_i]:d} papers",
         tx=2024.0, ty=top * 1.185)

    # ---- legend as a right-hand key ----------------------------------------
    h_bar, l_bar = ax.get_legend_handles_labels()
    h_cum, l_cum = axc.get_legend_handles_labels()
    leg = fig.legend(h_bar[::-1] + h_cum, l_bar[::-1] + l_cum,
                     loc="upper left", bbox_to_anchor=(0.700, 0.815),
                     fontsize=7.4, frameon=False, labelspacing=0.72,
                     handlelength=1.5, handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.text(0.700, 0.225,
             f"Corpus: {grand:d} papers, {years[0]}-{years[-1]}\n"
             f"Median year {int(np.median(np.repeat(years, totals))):d}\n"
             f"Last three years hold "
             f"{totals[-3:].sum() / grand:.0%} of the corpus",
             ha="left", va="top", fontsize=7.0, color=COLORS["sub"],
             linespacing=1.65)

    # Where the last strong year-on-year jump lands is where the plateau
    # starts; both halves of the headline are read back off the bars.
    plateau_i = int(np.nonzero(ratio >= 1.15)[0][-1]) + 1
    lead_k = int(np.argmax(counts.sum(axis=1)))
    lead_share = counts[lead_k].sum() / grand

    savefig(fig, "1147_wm_survey_papers_by_year.png")


if __name__ == "__main__":
    main()
