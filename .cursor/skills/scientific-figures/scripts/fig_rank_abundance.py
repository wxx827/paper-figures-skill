"""Rank-abundance (Whittaker) plots for six survey plots.

Every panel shares one globally computed rank axis and one log abundance axis
so that curve steepness is directly comparable across plots; the shared range
is derived by sweeping all six datasets before any axes are drawn.  Two
candidate species-abundance models (log-normal, geometric series) are overlaid
on the observed ranks, and each panel is annotated with its dominant species
and diversity statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

from pubstyle import set_pub_style, savefig, style_axes


OBS = "#3f4a45"
LOGN = "#2e7d5b"
GEOM = "#c08f3e"
DOM = "#a2593c"
INK = "#1a1a1a"

#        plot label            S   sigma  a_max  dominant species
SITES = [
    ("Old-growth forest", 38, 1.15, 620.0, "Fagus sylvatica"),
    ("Secondary forest", 32, 1.05, 480.0, "Betula pendula"),
    ("Riparian corridor", 27, 1.00, 350.0, "Salix alba"),
    ("Grassland mosaic", 21, 1.35, 900.0, "Festuca rubra"),
    ("Degraded scrub", 14, 1.55, 1100.0, "Rubus fruticosus"),
    ("Restored plantation", 24, 1.10, 400.0, "Pinus sylvestris"),
]


def panel_tag(fig, ax, label, dx=-0.045, dy=0.014, size=9.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def lognormal_ranks(s, sigma, a_max):
    """Expected abundance of each rank under a log-normal SAD."""
    i = np.arange(1, s + 1)
    z = norm.ppf(1.0 - (i - 0.5) / s)
    a = np.exp(sigma * z)
    return i, a * (a_max / a[0])


def shannon(counts):
    p = counts / counts.sum()
    h = -(p * np.log(p)).sum()
    return h, h / np.log(counts.size)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260226)

    # ---- generate every dataset first, then derive shared limits ---------
    data = []
    for name, s, sigma, a_max, dominant in SITES:
        rank, expected = lognormal_ranks(s, sigma, a_max)
        obs = np.maximum(1.0, expected * np.exp(rng.normal(0, 0.20, s)))
        obs = np.sort(obs)[::-1]
        # geometric series = straight line in log space
        b1, b0 = np.polyfit(rank, np.log(obs), 1)
        geom = np.exp(b0 + b1 * rank)
        h, j = shannon(obs)
        data.append(dict(name=name, rank=rank, obs=obs, exp=expected,
                         geom=geom, dominant=dominant, s=s, h=h, j=j))

    rank_max = max(d["rank"].max() for d in data)
    obs_lo = min(d["obs"].min() for d in data)
    obs_hi = max(d["obs"].max() for d in data)
    xlim = (0.0, rank_max + 2.0)
    ylim = (obs_lo * 0.42, obs_hi * 3.0)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True,
                             sharey=True)
    axes = axes.ravel()

    for ax, d in zip(axes, data):
        ax.set_facecolor("#fbfbf9")
        ax.plot(d["rank"], d["geom"], color=GEOM, lw=1.15,
                ls=(0, (4.5, 2.0)), zorder=3)
        ax.plot(d["rank"], d["exp"], color=LOGN, lw=1.5, zorder=4)
        ax.plot(d["rank"], d["obs"], color=OBS, lw=0.6, alpha=0.55, zorder=5)
        ax.scatter(d["rank"], d["obs"], s=9, facecolor="white",
                   edgecolor=OBS, linewidth=0.65, zorder=6)
        ax.scatter([1], [d["obs"][0]], s=26, facecolor=DOM,
                   edgecolor="white", linewidth=0.8, zorder=7)

        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_yticks([10, 100, 1000])
        ax.grid(lw=0.5, color="#eaeae5", zorder=0)
        style_axes(ax)
        ax.set_title(d["name"], fontsize=8.0, loc="left", pad=4)

        ax.text(0.965, 0.955, d["dominant"], transform=ax.transAxes,
                ha="right", va="top", fontsize=6.4, fontstyle="italic",
                color=DOM)
        ax.text(0.035, 0.055,
                f"S = $\\bf{{{d['s']}}}$\n"
                f"H' = {d['h']:.2f}\nJ' = {d['j']:.2f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.3, color="#4a4a44", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#dedeD8", linewidth=0.55, alpha=0.92))

    for ax in axes[3:]:
        ax.set_xlabel("Species rank")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Abundance (individuals)")

    handles = [
        Line2D([], [], color=OBS, lw=0.9, marker="o", ms=3.4, mfc="white",
               mew=0.65, label="Observed rank abundance"),
        Line2D([], [], color=LOGN, lw=1.5, label="Log-normal SAD"),
        Line2D([], [], color=GEOM, lw=1.15, ls=(0, (4.5, 2.0)),
               label="Geometric series"),
        Line2D([], [], color="none", marker="o", ms=4.4, mfc=DOM,
               mec="white", mew=0.8, label="Dominant species"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                     fontsize=7, frameon=False, handlelength=1.7,
                     columnspacing=1.9, handletextpad=0.55,
                     bbox_to_anchor=(0.53, 0.005))
    leg.set_clip_on(False)

    fig.subplots_adjust(left=0.082, right=0.985, top=0.865, bottom=0.155,
                        hspace=0.40, wspace=0.12)

    for ax, lab in zip(axes, "abcdef"):
        panel_tag(fig, ax, f"({lab})")

    savefig(fig, "226_rank_abundance.png")


if __name__ == "__main__":
    main()
