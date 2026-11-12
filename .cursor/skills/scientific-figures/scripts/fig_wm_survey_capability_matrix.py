"""Capability coverage matrix: 12 world-model benchmarks x 9 capabilities.

The survey-table idiom: a solid dot means the capability is fully evaluated, a
half dot means it is only partly covered, an open dot means it is absent.  The
right-hand column stacks the two supported counts into one bar per benchmark,
and the strip under the grid counts full support per capability -- both are
read straight off the same integer matrix, so the bar length and every printed
count are the matrix, not a second hand-written table.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle

from pubstyle import COLORS, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

FULL, PART, NONE = 2, 1, 0
BENCHMARKS = ["AgentWorld", "CityDrive-B", "DreamBench", "EgoSim-1.0",
              "GenieEval", "NavSuite-X", "OmniWorld", "PhyRe-Bench",
              "RoboDream", "TabletopWM", "VideoPred-B", "WM-Arena"]
CAPABILITIES = ["Static geometry", "Object dynamics", "Contact physics",
                "Long-horizon rollout", "Action conditioning",
                "Multi-agent interaction", "Language grounding",
                "Egocentric viewpoint", "Sim-to-real transfer"]


def gen_data(seed=1554):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_survey_capability_matrix.csv", index_col=0)
        return df.index.tolist(), df.columns.tolist(), df.to_numpy()
    """
    rng = np.random.default_rng(seed)
    n, m = len(BENCHMARKS), len(CAPABILITIES)
    # Broad benchmarks cover more; easy capabilities are covered by more suites.
    breadth = rng.uniform(-0.55, 0.95, n)[:, None]
    ease = np.linspace(0.95, -0.75, m)[None, :]
    score = breadth + ease + rng.normal(0.0, 0.55, (n, m))
    support = np.where(score > 0.72, FULL, np.where(score > -0.05, PART, NONE))
    return BENCHMARKS, CAPABILITIES, support


def main():
    set_pub_style(base_font=8)
    benches, caps, sup = gen_data()

    n_full = (sup == FULL).sum(axis=1)
    n_part = (sup == PART).sum(axis=1)
    order = np.lexsort((n_part, n_full))[::-1]
    benches = [benches[i] for i in order]
    sup = sup[order]
    n_full, n_part = n_full[order], n_part[order]
    n, m = sup.shape

    col_full = (sup == FULL).sum(axis=0)
    col_part = (sup == PART).sum(axis=0)
    # Support points per column: a solid dot counts 2, a half dot counts 1.
    col_points = 2 * col_full + col_part
    weak_j = int(np.argmin(col_points))

    fig = plt.figure(figsize=(9.0, 6.8))
    axm = fig.add_axes([0.222, 0.150, 0.520, 0.640])
    axb = fig.add_axes([0.775, 0.150, 0.165, 0.640])

    # ---- the dot grid -------------------------------------------------------
    for i in range(n):
        if i % 2 == 0:
            axm.add_patch(Rectangle((-0.5, i - 0.5), m, 1.0,
                                    facecolor="#f4f7fa", edgecolor="none",
                                    zorder=0))
    for j in range(m):
        axm.plot([j, j], [-0.5, n - 0.5], color="#e6eaee", lw=0.6, zorder=1)
    axm.add_patch(Rectangle((weak_j - 0.5, -0.5), 1.0, n, facecolor="none",
                            edgecolor=ACCENT, linewidth=1.0, zorder=4))

    glyph = {
        FULL: dict(marker="o", mfc=BLUE[1], mec=BLUE[0], mew=0.7),
        PART: dict(marker=MarkerStyle("o", fillstyle="left"), mfc=BLUE[3],
                   mfcalt="white", mec=BLUE[1], mew=0.7),
        NONE: dict(marker="o", mfc="white", mec="#b7c1c9", mew=0.7),
    }
    for i in range(n):
        for j in range(m):
            axm.plot([j], [i], linestyle="none", markersize=8.4, zorder=3,
                     **glyph[int(sup[i, j])])

    axm.set_xlim(-0.5, m - 0.5)
    axm.set_ylim(n - 0.5, -0.5)
    axm.set_xticks(range(m))
    axm.set_xticklabels(caps, rotation=45, ha="left", rotation_mode="anchor",
                        fontsize=7.2, color=INK)
    axm.xaxis.set_ticks_position("top")
    axm.set_yticks(range(n))
    axm.set_yticklabels(benches, fontsize=7.6, color=INK)
    axm.tick_params(length=0, pad=3.5)
    axm.get_xticklabels()[weak_j].set_color(ACCENT)
    for s in axm.spines.values():
        s.set_visible(False)

    # Per-capability tally strip under the grid.
    for j in range(m):
        axm.text(j, n - 0.18,
                 f"{int(col_full[j]):d}+{int(col_part[j]):d}", ha="center",
                 va="top", fontsize=6.6, fontweight="bold", clip_on=False,
                 color=ACCENT if j == weak_j else BLUE[1])
    axm.text(-0.72, n - 0.18, f"full+partial (of {n:d}):", ha="right",
             va="top", fontsize=6.6, color=COLORS["sub"], clip_on=False)

    # ---- coverage bars, stacked full + partial ------------------------------
    y = np.arange(n)
    axb.barh(y, n_full, height=0.52, color=BLUE[1], edgecolor="white",
             linewidth=0.4, zorder=3)
    axb.barh(y, n_part, left=n_full, height=0.52, color=BLUE[4],
             edgecolor="white", linewidth=0.4, zorder=3)
    for i in range(n):
        axb.text(n_full[i] + n_part[i] + 0.22, i,
                 f"{int(n_full[i]):d}+{int(n_part[i]):d}", ha="left",
                 va="center", fontsize=6.8, color=INK, zorder=4)
    axb.set_xlim(0, m)
    axb.set_ylim(n - 0.5, -0.5)
    axb.set_yticks([])
    axb.set_xticks([0, m // 2, m])
    axb.tick_params(axis="x", labelsize=6.8, length=2.5)
    axb.xaxis.set_ticks_position("bottom")
    axb.set_xlabel(f"capabilities covered (of {m:d})", fontsize=7.0,
                   color=COLORS["sub"], labelpad=2.0)
    axb.grid(axis="x", color="#e6eaee", lw=0.6)
    axb.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axb.spines[side].set_visible(False)
    axb.set_title("Coverage", fontsize=7.6, fontweight="bold", color=INK,
                  pad=5)

    # ---- key ----------------------------------------------------------------
    keys = [Line2D([], [], linestyle="none", markersize=8.4,
                   label="full support", **glyph[FULL]),
            Line2D([], [], linestyle="none", markersize=8.4,
                   label="partial support", **glyph[PART]),
            Line2D([], [], linestyle="none", markersize=8.4,
                   label="not supported", **glyph[NONE])]
    leg = fig.legend(handles=keys, loc="lower left",
                     bbox_to_anchor=(0.222, 0.038), ncol=3, fontsize=7.4,
                     frameon=False, handletextpad=0.55, columnspacing=2.0)
    for txt in leg.get_texts():
        txt.set_color(INK)

    totals = int(n_full.sum() + n_part.sum())
    best = int((n_full + n_part).max())

    savefig(fig, "1149_wm_survey_capability_matrix.png")


if __name__ == "__main__":
    main()
