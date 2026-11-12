"""One-figure evaluation summary for 3-D perception methods.

(a) per-category radar of mAP, (b) accuracy against object distance,
(c) class-versus-method AP heat map, (d) throughput against accuracy with the
Pareto front and bubble area encoding model size.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, add_colorbar

SEED = 1234

METHODS = ["PointPillars", "SECOND", "CenterPoint", "PV-RCNN", "VoxelNeXt",
           "BEVFusion", "Ours"]
MCOL = ["#b0b7bf", "#8c9aa8", "#6f8fb0", "#c9701f", "#2e8b57", "#8e4bbf",
        "#b23a48"]
AXES = ["Car", "Pedestrian", "Cyclist", "Truck", "Barrier", "Trailer"]

AP = np.array([
    [80.1, 43.2, 55.4, 48.6, 52.1, 31.4],     # PointPillars
    [83.4, 47.8, 60.2, 52.0, 55.6, 34.8],     # SECOND
    [86.9, 58.4, 68.1, 58.7, 62.4, 40.2],     # CenterPoint
    [88.2, 60.7, 71.5, 61.3, 64.0, 42.9],     # PV-RCNN
    [89.4, 63.5, 73.8, 63.9, 66.8, 45.6],     # VoxelNeXt
    [90.6, 67.2, 76.4, 66.1, 69.3, 48.1],     # BEVFusion
    [92.3, 72.8, 81.0, 70.4, 73.6, 53.7],     # Ours
])
FPS = np.array([62.0, 41.0, 27.5, 12.4, 22.8, 8.6, 19.6])
PARAMS = np.array([4.8, 5.3, 9.1, 13.9, 7.4, 41.2, 15.6])
DIST = np.array([5, 15, 25, 35, 45])
DIST_AP = np.array([
    [86.0, 74.5, 58.2, 40.1, 24.6],
    [88.2, 78.0, 63.4, 45.8, 29.3],
    [91.0, 83.1, 71.0, 55.2, 37.8],
    [92.1, 85.0, 74.2, 58.9, 41.0],
    [92.8, 86.4, 76.5, 62.1, 44.7],
    [93.5, 88.0, 79.1, 65.6, 48.5],
    [95.1, 91.2, 84.0, 72.8, 57.9],
])


def radar(ax):
    n = len(AXES)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    thc = np.r_[th, th[:1]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(20, 100)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_yticklabels(["40", "60", "80", "100"], fontsize=6.0,
                       color="#777777")
    ax.set_xticks(th)
    ax.set_xticklabels(AXES, fontsize=7.0)
    ax.tick_params(axis="x", pad=-1.0)
    ax.grid(color="#dcdcdc", lw=0.5)
    ax.spines["polar"].set_linewidth(0.6)
    ax.spines["polar"].set_color("#bbbbbb")
    for i, m in enumerate(METHODS):
        show = m in ("PointPillars", "CenterPoint", "VoxelNeXt", "BEVFusion",
                     "Ours")
        if not show:
            continue
        v = np.r_[AP[i], AP[i, :1]]
        hot = m == "Ours"
        ax.plot(thc, v, color=MCOL[i], lw=1.9 if hot else 1.0,
                zorder=6 if hot else 3, label=m)
        if hot:
            ax.fill(thc, v, color=MCOL[i], alpha=0.13, zorder=2)
    ax.set_title("Per-category AP (%)", fontsize=8.4, pad=3.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=6.2, frameon=False, handlelength=1.4,
              columnspacing=0.9, handletextpad=0.4, labelspacing=0.25)


def dist_curves(ax):
    for i, m in enumerate(METHODS):
        hot = m == "Ours"
        if m in ("SECOND", "PV-RCNN"):
            continue
        ax.plot(DIST, DIST_AP[i], "-o", color=MCOL[i],
                lw=2.0 if hot else 1.1, ms=4.0 if hot else 2.8,
                mec="white", mew=0.5, zorder=6 if hot else 3, label=m)
    ax.fill_between(DIST, DIST_AP[0], DIST_AP[6], color="#b23a48", alpha=0.07,
                    zorder=1)
    ax.annotate("+33.3 pp at 40-50 m",
                xy=(45, 0.5 * (DIST_AP[0, -1] + DIST_AP[6, -1])),
                xytext=(31.5, 30.0), fontsize=6.3, color="#8c2f2f",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, mutation_scale=6,
                                color="#8c2f2f", shrinkA=1, shrinkB=1))
    ax.set_xlabel("object distance (m)", fontsize=7.4, labelpad=1.5)
    ax.set_ylabel("AP (%)", fontsize=7.4, labelpad=2.0)
    ax.set_xticks(DIST)
    ax.set_xticklabels(["0-10", "10-20", "20-30", "30-40", "40-50"],
                       fontsize=6.4)
    ax.set_ylim(15, 100)
    ax.tick_params(labelsize=6.6, length=2.4)
    ax.grid(lw=0.45, color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.set_title("Accuracy versus range", fontsize=8.4, pad=3.5)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left",
              handlelength=1.5, handletextpad=0.4, labelspacing=0.28,
              borderaxespad=0.3)


def heat(fig, ax):
    cmap = LinearSegmentedColormap.from_list(
        "ap", ["#f2f7fb", "#cfe1ee", "#9cc3dd", "#5f9dc6", "#2e74a8",
               "#134a76"])
    im = ax.imshow(AP, cmap=cmap, vmin=28, vmax=95, aspect="auto")
    ax.set_xticks(np.arange(len(AXES)))
    ax.set_xticklabels(AXES, fontsize=6.4, rotation=28, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=6.4)
    for lb in ax.get_yticklabels():
        if lb.get_text() == "Ours":
            lb.set_fontweight("bold")
            lb.set_color("#8c2f2f")
    for i in range(AP.shape[0]):
        for j in range(AP.shape[1]):
            v = AP[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.0,
                    color="white" if v > 68 else "#333333",
                    fontweight="bold" if i == 6 else "normal")
    ax.set_xticks(np.arange(-0.5, len(AXES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Class-wise AP heat map (%)", fontsize=8.4, pad=3.5)
    add_colorbar(fig, im, ax, label="AP (%)", fraction=0.035, pad=0.02,
                 shrink=0.90, aspect=18, ticks=[30, 60, 90])


def pareto(ax):
    mAP = AP.mean(axis=1)
    idx = np.argsort(-FPS)
    front, best = [], -np.inf
    for i in idx:
        if mAP[i] > best:
            front.append(i)
            best = mAP[i]
    front = front[::-1]
    ax.plot(FPS[front], mAP[front], color="#b23a48", lw=1.0,
            ls=(0, (4, 2)), zorder=2, label="Pareto front")
    ax.scatter(FPS, mAP, s=PARAMS * 15.0, c=MCOL, edgecolors="#3a3a3a",
               linewidths=0.6, alpha=0.92, zorder=5)
    off = {"PointPillars": (-3.0, 4.6), "SECOND": (0.0, 4.6),
           "CenterPoint": (10.0, -0.6), "PV-RCNN": (0.5, -5.6),
           "VoxelNeXt": (10.5, 1.2), "BEVFusion": (1.5, 7.4),
           "Ours": (0.5, 4.8)}
    for i, m in enumerate(METHODS):
        dx, dy = off[m]
        ax.annotate(m, (FPS[i] + dx, mAP[i] + dy), fontsize=6.3,
                    ha="center", va="center",
                    color="#8c2f2f" if m == "Ours" else "#333333",
                    fontweight="bold" if m == "Ours" else "normal")
    ax.axvline(10.0, color="#999999", lw=0.7, ls=(0, (2.6, 2.0)), zorder=1)
    ax.text(10.6, 45.6, "real time\n10 FPS", fontsize=6.0, color="#777777",
            ha="left", va="bottom", linespacing=1.3)
    ax.set_xlabel("inference speed (FPS)", fontsize=7.4, labelpad=1.5)
    ax.set_ylabel("mAP (%)", fontsize=7.4, labelpad=2.0)
    ax.set_xlim(2, 74)
    ax.set_ylim(44, 89)
    ax.tick_params(labelsize=6.6, length=2.4)
    ax.grid(lw=0.45, color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.set_title("Throughput versus accuracy", fontsize=8.4, pad=3.5)
    sizes = [5, 20, 40]
    handles = [Line2D([], [], marker="o", ls="none", mfc="#d8d8d8",
                      mec="#3a3a3a", mew=0.5,
                      ms=np.sqrt(s * 15.0) * 0.62, label=f"{s} M")
               for s in sizes]
    handles.append(Line2D([], [], color="#b23a48", lw=1.0, ls=(0, (4, 2)),
                          label="Pareto front"))
    lg = ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=6.0,
                   frameon=False, columnspacing=1.1, handletextpad=0.8,
                   borderaxespad=0.25, title="model parameters",
                   title_fontsize=6.2)
    lg._legend_box.align = "left"


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(7.4, 5.75))

    axr = fig.add_axes([0.055, 0.572, 0.335, 0.318], projection="polar")
    radar(axr)
    axd = fig.add_axes([0.585, 0.588, 0.375, 0.325])
    dist_curves(axd)
    axh = fig.add_axes([0.130, 0.088, 0.290, 0.335])
    heat(fig, axh)
    axp = fig.add_axes([0.612, 0.088, 0.348, 0.335])
    pareto(axp)

    for lab, x, y in [("(a)", 0.020, 0.945), ("(b)", 0.512, 0.945),
                      ("(c)", 0.020, 0.462), ("(d)", 0.512, 0.462)]:
        fig.text(x, y, lab, fontsize=9.8, fontweight="bold", ha="left",
                 va="center")


    savefig(fig, "292_3d_benchmark_panel.png")


if __name__ == "__main__":
    main()
