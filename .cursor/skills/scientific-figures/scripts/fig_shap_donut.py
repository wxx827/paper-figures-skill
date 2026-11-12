"""SHAP contribution figure: importance bars + nested donut + beeswarm.

Left = horizontal %contribution bars per feature with value labels; centre = a
two-ring nested donut that aggregates the same contributions into three feature
categories; right = a SHAP beeswarm whose row order and spread match the bars,
coloured by feature value with a shared Low->High colorbar. Two model rows
(temperature / humidity). All three panels of a row are driven by one set of
numbers, so the bars, the donut and the beeswarm tell the same story.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes

FEATS = ["NDVI", "SVF", "SDH", "SCD", "FD", "ISF", "MD", "BD", "FAR",
         "RG", "AWH", "MBS", "DEI"]
# category of each feature: 0=Landscape, 1=2-D morphology, 2=3-D morphology
GROUP = [0, 2, 2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0]
GROUP_NAMES = ["Landscape ecological indicators",
               "Two-dimensional morphology",
               "Three-dimensional morphology"]
GROUP_COLORS = ["#4a9d6f", "#d16b7c", "#5b8fc9"]
SHAPCMAP = LinearSegmentedColormap.from_list(
    "gr_pk", ["#3d6b3d", "#8bbf6b", "#e0d090", "#d98ba0", "#a83a5b"])


def draw_bars(ax, order, pct, title):
    y = np.arange(len(order))[::-1]
    cols = [GROUP_COLORS[GROUP[i]] for i in order]
    ax.barh(y, pct, color=cols, edgecolor="white", height=0.74, lw=0.6,
            zorder=3)
    for yi, v in zip(y, pct):
        ax.text(v + pct.max() * 0.02, yi, f"{v:.1f}%", va="center",
                fontsize=6.8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([FEATS[i] for i in order], fontsize=7.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("Mean |SHAP| contribution (%)", fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlim(0, pct.max() * 1.22)
    ax.grid(axis="x", lw=0.5, alpha=0.4)
    style_axes(ax)


def draw_donut(ax, order, pct, center_label):
    ocols = [GROUP_COLORS[GROUP[i]] for i in order]
    ax.pie(pct, radius=1.0, colors=ocols, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.30, edgecolor="white", linewidth=0.7))
    gsum = [float(pct[[GROUP[i] == g for i in order]].sum()) for g in range(3)]
    wedges, _ = ax.pie(gsum, radius=0.66, colors=GROUP_COLORS, startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.30, edgecolor="white",
                                       linewidth=0.9))
    for w, v in zip(wedges, gsum):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(0.51 * np.cos(ang), 0.51 * np.sin(ang), f"{v:.0f}%",
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")
    ax.text(0, 0, center_label, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="#1a1a1a")
    ax.text(0, -1.24, "outer ring: features   inner ring: category",
            ha="center", va="center", fontsize=6.8, color="#666666")
    ax.set_aspect("equal")


def draw_beeswarm(ax, rng, order, pct):
    scale = pct / pct.max()
    for row, (i, sc) in enumerate(zip(order, scale)):
        n = 200
        val = rng.uniform(0, 1, n)                    # normalised feature value
        sign = 1 if rng.random() < 0.6 else -1
        sh = sign * 0.30 * sc * (val - 0.5) + rng.normal(0, 0.06 * sc + 0.006, n)
        # density-aware jitter -> the classic beeswarm silhouette
        cnt, edges = np.histogram(sh, bins=32)
        k = np.clip(np.digitize(sh, edges[1:-1]), 0, 31)
        spread = 0.40 * np.sqrt(cnt[k] / max(cnt.max(), 1))
        yj = (len(order) - 1 - row) + rng.uniform(-1, 1, n) * spread
        o = np.argsort(np.abs(sh))
        ax.scatter(sh[o], yj[o], c=val[o], cmap=SHAPCMAP, s=5.5, alpha=0.85,
                   edgecolors="none", vmin=0, vmax=1, zorder=3)
    ax.axvline(0, color="#999999", lw=0.7, zorder=2)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels([FEATS[i] for i in order], fontsize=7.5)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=7.5)
    ax.grid(axis="x", lw=0.5, alpha=0.35)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(261)
    fig = plt.figure(figsize=(12.2, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 0.88, 1.45],
                          left=0.065, right=0.895, top=0.94, bottom=0.10,
                          wspace=0.30, hspace=0.40)

    rows = [("(a)  SHAP contributions - land-surface temperature model",
             "Temperature\nmodel"),
            ("(b)  SHAP contributions - relative humidity model",
             "Humidity\nmodel")]
    for r, (title, center) in enumerate(rows):
        raw = rng.gamma(2.4, 1.0, len(FEATS)) + 0.25
        pct = np.sort(raw)[::-1] / raw.sum() * 100
        order = list(np.argsort(-raw))
        draw_bars(fig.add_subplot(gs[r, 0]), order, pct, title)
        draw_donut(fig.add_subplot(gs[r, 1]), order, pct, center)
        draw_beeswarm(fig.add_subplot(gs[r, 2]), rng, order, pct)

    cax = fig.add_axes([0.915, 0.30, 0.013, 0.40])
    sm = plt.cm.ScalarMappable(cmap=SHAPCMAP, norm=Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.ax.tick_params(labelsize=7.5, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.set_label("Feature value", fontsize=8)

    handles = [Patch(facecolor=GROUP_COLORS[g], edgecolor="white",
                     label=GROUP_NAMES[g]) for g in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.025))
    savefig(fig, "33_shap_donut.png")


if __name__ == "__main__":
    main()
