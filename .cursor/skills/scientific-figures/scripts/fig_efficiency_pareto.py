"""Accuracy against inference latency: the efficiency Pareto front.

Bubble area encodes parameter count, hue encodes the model family and the
dashed staircase is the non-dominated set for (minimise latency, maximise
accuracy), computed from the data rather than drawn by hand.  Labels are
attached with thin leader lines so they never sit on a marker.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

OURS = "#3b5b92"
GREY = "#8c8c8c"
GREEN = "#2e7d5b"
ORANGE = "#c9a227"
PURPLE = "#8e6bbf"
GRID = "#eeeeee"

FCOL = {"CNN": GREY, "Transformer": GREEN, "Hybrid": ORANGE,
        "Efficient / NAS": PURPLE, "Ours": OURS}
FAMILIES = ["CNN", "Transformer", "Hybrid", "Efficient / NAS", "Ours"]

# name, family, latency (ms), top-1 (%), params (M), label offset (pts)
MODELS = [
    ("ResNet-18", "CNN", 4.2, 71.5, 11.7, None),
    ("ResNet-50", "CNN", 9.8, 77.2, 25.6, (-2, -26)),
    ("ResNet-101", "CNN", 17.5, 78.6, 44.5, None),
    ("RegNetY-4G", "CNN", 11.2, 79.4, 20.6, None),
    ("DeiT-Ti", "Transformer", 5.1, 72.2, 5.7, None),
    ("ViT-S/16", "Transformer", 12.4, 79.9, 22.1, None),
    ("ViT-B/16", "Transformer", 32.0, 81.8, 86.6, (16, -18)),
    ("Swin-T", "Transformer", 14.8, 81.3, 28.3, None),
    ("Swin-B", "Transformer", 38.5, 83.5, 88.0, (10, 20)),
    ("ConvNeXt-T", "Hybrid", 13.6, 82.1, 28.6, None),
    ("CoAtNet-0", "Hybrid", 16.2, 81.6, 25.0, None),
    ("MaxViT-T", "Hybrid", 24.0, 83.6, 31.0, (-4, -25)),
    ("MobileNetV3", "Efficient / NAS", 1.9, 75.2, 5.4, (14, -20)),
    ("EfficientNet-B0", "Efficient / NAS", 3.6, 77.1, 5.3, (18, -16)),
    ("EfficientNet-B4", "Efficient / NAS", 18.0, 82.9, 19.0, None),
    ("Ours-S", "Ours", 4.6, 79.8, 8.9, (-4, 22)),
    ("Ours-B", "Ours", 11.0, 83.2, 24.0, (-8, 22)),
    ("Ours-L", "Ours", 26.0, 85.4, 62.0, (-6, 20)),
]


def pareto(lat, acc):
    """Non-dominated set for (min latency, max accuracy)."""
    order = np.argsort(lat)
    keep, best = [], -np.inf
    for i in order:
        if acc[i] > best + 1e-9:
            keep.append(i)
            best = acc[i]
    return keep


def bubble(p):
    return 14.0 + 3.4 * p


def main():
    set_pub_style(base_font=8)

    lat = np.array([m[2] for m in MODELS])
    acc = np.array([m[3] for m in MODELS])
    par = np.array([m[4] for m in MODELS])
    front = pareto(lat, acc)
    print("  Pareto front:",
          ", ".join(f"{MODELS[i][0]} ({lat[i]:.1f} ms, {acc[i]:.1f}%)"
                    for i in front))

    fig = plt.figure(figsize=(7.0, 4.4))
    ax = fig.add_axes([0.083, 0.115, 0.895, 0.80])
    ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.grid(which="major", color=GRID, lw=0.7, zorder=0)
    ax.grid(which="minor", color="#f7f7f7", lw=0.5, zorder=0)

    xlo, xhi, ylo, yhi = 1.35, 62.0, 68.6, 87.5

    # ---- Pareto staircase + dominated region ------------------------------
    fx = np.array([lat[i] for i in front])
    fy = np.array([acc[i] for i in front])
    sx = np.r_[fx, xhi]
    sy = np.r_[fy, fy[-1]]
    ax.fill_between(sx, ylo, sy, step="post", color="#8fa8c0", alpha=0.085,
                    lw=0, zorder=1)
    ax.step(sx, sy, where="post", color=OURS, lw=1.3, ls=(0, (4.5, 2.0)),
            zorder=6)
    ax.text(0.985, 0.335, "dominated region", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color="#95a5b5",
            style="italic")

    # ---- bubbles -----------------------------------------------------------
    for k, (name, fam, x, y, p, off) in enumerate(MODELS):
        hot = fam == "Ours"
        onf = k in front
        ax.scatter(x, y, s=bubble(p), facecolor=FCOL[fam],
                   alpha=0.30 if not hot else 0.42,
                   edgecolor=FCOL[fam], linewidth=1.5 if hot else 0.9,
                   zorder=8 if hot else 5)
        if onf:
            ax.scatter(x, y, s=9, facecolor=FCOL[fam], edgecolor="white",
                       linewidth=0.6, zorder=9)
        if off is not None:
            ax.annotate(name, xy=(x, y), xytext=off,
                        textcoords="offset points", ha="center",
                        va="center", fontsize=6.6 if not hot else 7.2,
                        color=FCOL[fam],
                        fontweight="bold" if hot else "normal", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", lw=0.55,
                                        color=FCOL[fam], alpha=0.8,
                                        shrinkA=0, shrinkB=2,
                                        connectionstyle="arc3,rad=0.0"))

    # ---- "better" direction ------------------------------------------------
    ax.annotate("", xy=(0.055, 0.955), xytext=(0.20, 0.80),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#5b7f9e",
                                mutation_scale=13))
    ax.text(0.215, 0.795, "better\n(faster, more accurate)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            color="#5b7f9e", linespacing=1.3, style="italic")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                           numticks=40))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(which="minor", length=1.6)
    ax.set_yticks(np.arange(70, 88, 2))
    ax.set_xlabel("Inference latency on one A100, batch 1 (ms, log scale)",
                  fontsize=8.2)
    ax.set_ylabel("ImageNet top-1 accuracy (%)", fontsize=8.2)

    fam_h = [Line2D([], [], ls="none", marker="o", ms=5.6,
                    mfc=FCOL[f] + "66", mec=FCOL[f],
                    mew=1.5 if f == "Ours" else 0.9, label=f)
             for f in FAMILIES]
    leg1 = ax.legend(handles=fam_h, loc="lower right",
                     bbox_to_anchor=(1.0, 0.015), frameon=False, fontsize=7.2,
                     handletextpad=0.85, labelspacing=0.42, borderpad=0.2)
    leg1.get_texts()[-1].set_fontweight("bold")
    leg1.set_zorder(-5)
    ax.add_artist(leg1)
    # add_artist() makes the legend inherit the axes clip path, which chops the
    # bold "Ours" row off at the right spine
    leg1.set_clip_on(False)
    leg1.set_clip_path(None)

    size_h = [Line2D([], [], ls="none", marker="o", mfc="#d8d8d8",
                     mec="#8c8c8c", mew=0.8,
                     ms=np.sqrt(bubble(p)), label=f"{p:g} M")
              for p in (5, 25, 90)]
    leg2 = ax.legend(handles=size_h, loc="lower left",
                     bbox_to_anchor=(0.02, 0.015), frameon=False, ncol=3,
                     fontsize=6.8, columnspacing=1.1, handletextpad=0.55,
                     handlelength=2.9, borderpad=0.2, title="parameters",
                     title_fontsize=6.8)
    leg2._legend_box.align = "left"
    leg2.set_zorder(-5)
    leg2.set_clip_on(False)
    leg2.set_clip_path(None)

    style_axes(ax)

    savefig(fig, "308_efficiency_pareto.png")


if __name__ == "__main__":
    main()
