"""Taylor diagram: model skill (correlation + standard deviation).

Standard orientation: radius = standard deviation normalised by the
observations, azimuth = arccos(correlation), so a perfect model sits on the
horizontal axis at radius 1.  Dashed grey arcs are centred-RMS-difference
contours around the reference point and are labelled on the radial axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig

RMAX = 1.65
STD_REF = 1.0
CORR_TICKS = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
RMS_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.25]

# model, correlation, normalised standard deviation
MODELS = [
    ("MLR", 0.79, 0.72, "#3d6b8e"),
    ("SVR", 0.83, 1.24, "#8c2f2f"),
    ("MLP", 0.88, 0.91, "#d98c5f"),
    ("RF", 0.93, 0.88, "#2e7d5b"),
    ("XGB", 0.95, 0.97, "#c9a227"),
    ("LGBM", 0.96, 1.03, "#6b5b95"),
    ("CatBoost", 0.97, 0.99, "#2a8f86"),
]


def main():
    set_pub_style()
    fig = plt.figure(figsize=(7.4, 6.6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_direction(1)
    ax.set_theta_offset(0)
    ax.set_ylim(0, RMAX)

    ax.set_xticks(np.arccos(CORR_TICKS))
    ax.set_xticklabels([f"{c:g}" for c in CORR_TICKS], fontsize=7.5)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                  labels=["0.25", "0.50", "0.75", "1.00", "1.25", "1.50"],
                  angle=90, fontsize=7)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.spines["polar"].set_color("#666666")
    ax.spines["polar"].set_linewidth(0.9)
    ax.tick_params(pad=2)

    # centred-RMS-difference arcs around the reference point
    circ = np.linspace(0, 2 * np.pi, 400)
    for rms in RMS_LEVELS:
        xs = STD_REF + rms * np.cos(circ)
        ys = rms * np.sin(circ)
        r = np.hypot(xs, ys)
        t = np.arctan2(ys, xs)
        m = (t >= 0) & (t <= np.pi / 2) & (r <= RMAX)
        ax.plot(t[m], r[m], color="#c0c0c0", ls=(0, (3, 2)), lw=0.6,
                zorder=1)
        if STD_REF - rms > 0.06:
            ax.text(np.deg2rad(7), STD_REF - rms, f"{rms:.2f}", ha="center",
                    va="bottom", fontsize=6.2, color="#9a9a9a", zorder=4)

    # unit standard-deviation arc
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(th, np.ones_like(th) * STD_REF, color="#9a9a9a", lw=0.8,
            ls=(0, (1, 2)), zorder=1)

    ax.plot([0], [STD_REF], marker="*", ms=15, color="#1a1a1a", zorder=6,
            mec="white", mew=0.6)
    ax.text(0.045, STD_REF + 0.06, "Observations", fontsize=7.5,
            color="#1a1a1a", ha="left", va="bottom")

    handles = []
    for name, corr, std, col in MODELS:
        ax.plot([np.arccos(corr)], [std], marker="o", ms=8, color=col,
                mec="white", mew=0.8, zorder=5)
        rmsd = np.sqrt(std ** 2 + STD_REF ** 2 - 2 * std * STD_REF * corr)
        handles.append(Line2D([], [], marker="o", ls="none", color=col, ms=6,
                              mec="white", mew=0.6,
                              label=f"{name}  (E' = {rmsd:.2f})"))

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.99),
              fontsize=7.5, frameon=False, labelspacing=0.5,
              handletextpad=0.4, title="Model (centred RMS difference)",
              title_fontsize=8, alignment="left")

    ax.text(np.deg2rad(45), RMAX * 1.10, "Correlation coefficient",
            ha="center", va="center", fontsize=8.5, rotation=-45,
            color="#333333")
    ax.text(np.deg2rad(90), RMAX * 0.55, "Normalised standard deviation",
            ha="center", va="bottom", fontsize=8.5, rotation=90,
            color="#333333")
    fig.tight_layout(rect=(0.02, 0.0, 0.98, 0.94))
    savefig(fig, "21_taylor.png")


if __name__ == "__main__":
    main()
