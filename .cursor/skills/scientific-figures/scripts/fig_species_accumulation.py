"""Sample-based species accumulation (rarefaction / extrapolation) curves.

Main panel: individual-based rarefaction for four habitats, drawn solid up to
the reference sample size and dashed beyond it, each with a bootstrap
confidence band and a dotted asymptotic-richness estimate.
Accessory panel (right, ~40 % width): observed vs. estimated richness bars,
subordinate in size so the curve panel stays the visual anchor.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes


# Shared ecology palette -- habitat colours are reused across the whole set.
HABITATS = [
    ("Old-growth forest", "Old-growth", "#1b5e4b", 96.0, 215.0),
    ("Secondary forest", "Secondary", "#5aa17f", 74.0, 175.0),
    ("Riparian corridor", "Riparian", "#3d8b93", 61.0, 132.0),
    ("Grassland mosaic", "Grassland", "#c08f3e", 43.0, 98.0),
]
GREY = "#8a8a85"
INK = "#1a1a1a"


def panel_tag(fig, ax, label, dx=-0.055, dy=0.018, size=10):
    """Stamp a panel label in *figure* coordinates so the visual offset is
    identical for panels of different widths."""
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def accumulation(n, s_inf, half):
    """Michaelis-Menten style accumulation curve."""
    return s_inf * n / (half + n)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20260223)

    n_ref = 450          # reference sample size (individuals)
    n_max = 900          # extrapolation limit
    n = np.linspace(1, n_max, 500)

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.42],
                  height_ratios=[0.42, 1.0], hspace=0.42, wspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])

    obs_rich, est_rich = [], []

    # soft wash marking the extrapolated domain
    ax.axvspan(n_ref, n_max * 1.01, color="#f4f4f0", lw=0, zorder=0)

    for name, short, col, s_inf, half in HABITATS:
        curve = accumulation(n, s_inf, half)
        # band widens once we leave the observed range
        rel = np.where(n <= n_ref, 0.042,
                       0.042 + 0.085 * (n - n_ref) / (n_max - n_ref))
        se = curve * rel
        inside = n <= n_ref
        outside = n >= n_ref

        ax.fill_between(n[inside], (curve - 1.96 * se)[inside],
                        (curve + 1.96 * se)[inside], color=col, alpha=0.17,
                        lw=0, zorder=2)
        ax.fill_between(n[outside], (curve - 1.96 * se)[outside],
                        (curve + 1.96 * se)[outside], color=col, alpha=0.09,
                        lw=0, zorder=2)
        ax.plot(n[inside], curve[inside], color=col, lw=1.7, zorder=4)
        ax.plot(n[outside], curve[outside], color=col, lw=1.4,
                ls=(0, (4.5, 2.2)), zorder=4)

        s_obs = accumulation(n_ref, s_inf, half)
        ax.plot([n_ref], [s_obs], marker="o", ms=4.2, color=col,
                mec="white", mew=0.8, zorder=6)

        # asymptote
        ax.plot([n_ref, n_max * 1.005], [s_inf, s_inf], color=col, lw=0.7,
                ls=(0, (1.4, 1.8)), alpha=0.85, zorder=3)
        ax.text(n_ref + 16, s_inf + 1.2, f"$\\hat{{S}}$ = {s_inf:.0f}",
                color=col, fontsize=6.6, ha="left", va="bottom",
                fontweight="bold")

        obs_rich.append(s_obs)
        est_rich.append(s_inf)

    ax.axvline(n_ref, color=GREY, lw=0.8, ls=(0, (3, 2.5)), zorder=1)
    ax.text(n_ref - 12, 3.0, "reference sample\n$n$ = 450 individuals",
            fontsize=6.6, color="#5c5c58", ha="right", va="bottom",
            linespacing=1.25)
    ax.text(n_max * 0.995, 105.5, "extrapolated domain", fontsize=6.6,
            color="#77776f", ha="right", va="top", style="italic")

    ax.set_xlim(0, n_max * 1.01)
    ax.set_ylim(0, 108)
    ax.set_xticks(np.arange(0, 901, 150))
    ax.set_yticks(np.arange(0, 109, 20))
    ax.set_xlabel("Individuals sampled (n)")
    ax.set_ylabel("Cumulative species richness (species)")
    ax.set_title("Individual-based rarefaction and extrapolation, "
                 "2019-2024 plot network", fontsize=8.5, loc="left", pad=6)
    ax.grid(axis="y", lw=0.5, color="#e6e6e2")
    style_axes(ax)

    # ---- accessory legend block ------------------------------------------
    ax_leg.axis("off")
    handles = [Line2D([], [], color=c, lw=1.8, label=full)
               for full, _s, c, _a, _b in HABITATS]
    handles += [
        Line2D([], [], color=GREY, lw=1.5, label="Interpolated"),
        Line2D([], [], color=GREY, lw=1.3, ls=(0, (4.5, 2.2)),
               label="Extrapolated"),
    ]
    leg = ax_leg.legend(handles=handles, loc="center left", fontsize=6.8,
                        frameon=False, handlelength=1.5, labelspacing=0.42,
                        handletextpad=0.6, borderaxespad=0.0,
                        bbox_to_anchor=(-0.02, 0.46))
    leg.set_clip_path(None)
    leg.set_clip_on(False)

    # ---- accessory bar panel ---------------------------------------------
    ax_bar.set_facecolor("#f6f6f2")
    ypos = np.arange(len(HABITATS))[::-1]
    h = 0.34
    for i, (full, short, col, s_inf, half) in enumerate(HABITATS):
        y = ypos[i]
        ax_bar.barh(y + h / 2 + 0.02, est_rich[i], height=h, color=col,
                    alpha=0.30, edgecolor=col, linewidth=0.6, zorder=3)
        ax_bar.barh(y - h / 2 - 0.02, obs_rich[i], height=h, color=col,
                    edgecolor="white", linewidth=0.5, zorder=4)
        ax_bar.text(est_rich[i] + 2.5, y + h / 2 + 0.02, f"{est_rich[i]:.0f}",
                    va="center", ha="left", fontsize=6.3, color=col)
        ax_bar.text(obs_rich[i] + 2.5, y - h / 2 - 0.02, f"{obs_rich[i]:.0f}",
                    va="center", ha="left", fontsize=6.3, color="#4a4a46")

    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels([s for _f, s, _c, _a, _b in HABITATS], fontsize=6.8)
    ax_bar.set_xlim(0, 122)
    ax_bar.set_ylim(-0.62, len(HABITATS) - 0.38)
    ax_bar.set_xticks([0, 40, 80, 120])
    ax_bar.set_xlabel("Species richness", fontsize=7)
    ax_bar.tick_params(labelsize=6.6)
    ax_bar.set_title("Observed (n = 450) vs.\nestimated asymptote",
                     fontsize=7.2, loc="left", pad=4, linespacing=1.3)
    ax_bar.grid(axis="x", lw=0.5, color="#e2e2dd")
    style_axes(ax_bar)

    bar_key = [
        Line2D([], [], marker="s", ls="", ms=4.6, mfc="#6f6f6a",
               mec="white", mew=0.5, label="observed"),
        Line2D([], [], marker="s", ls="", ms=4.6, mfc="#d5d5cf",
               mec="#9a9a94", mew=0.6, label="estimated"),
    ]
    lb = ax_bar.legend(handles=bar_key, loc="lower right", fontsize=6.2,
                       frameon=False, handletextpad=0.35, labelspacing=0.25,
                       borderaxespad=0.25)
    lb.set_clip_on(False)

    fig.subplots_adjust(left=0.072, right=0.988, top=0.865, bottom=0.115)

    panel_tag(fig, ax, "(a)", dx=-0.052)
    panel_tag(fig, ax_bar, "(b)", dx=-0.052, dy=0.052)

    savefig(fig, "223_species_accumulation.png")


if __name__ == "__main__":
    main()
