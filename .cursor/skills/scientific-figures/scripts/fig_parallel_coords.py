"""Parallel coordinates: multi-variable profiles for grouped sites.

Each vertical axis carries its own physical range (printed at the top and
bottom of the axis) and the polylines are min-max normalised per axis. The four
protection zones have genuinely different multivariate signatures, so the
profiles cross rather than sitting as four flat, parallel bands.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, PALETTES, savefig

# (label, unit, axis min, axis max)
AXES = [
    ("Water yield", "mm yr$^{-1}$", 150, 540),
    ("Carbon stock", "Mg C ha$^{-1}$", 25, 100),
    ("Habitat quality", "index", 0.20, 1.00),
    ("Soil loss", "t ha$^{-1}$ yr$^{-1}$", 0, 38),
    ("NPP", "g C m$^{-2}$ yr$^{-1}$", 350, 900),
    ("PM$_{2.5}$", "$\\mu$g m$^{-3}$", 8, 62),
    ("Recreation", "visits ha$^{-1}$", 0, 130),
]
GROUPS = ["Reserve core", "Buffer zone", "Transition", "Outside"]
COLORS = PALETTES["cb_safe"][:4]
# normalised group centre on each axis (0 = axis min, 1 = axis max)
PROFILES = {
    "Reserve core":  [0.72, 0.86, 0.90, 0.12, 0.83, 0.18, 0.10],
    "Buffer zone":   [0.63, 0.70, 0.72, 0.26, 0.71, 0.31, 0.34],
    "Transition":    [0.48, 0.47, 0.45, 0.52, 0.52, 0.55, 0.68],
    "Outside":       [0.35, 0.24, 0.22, 0.79, 0.34, 0.80, 0.88],
}
N_SITES = 30


def main():
    set_pub_style()
    rng = np.random.default_rng(371)
    n_ax = len(AXES)
    x = np.arange(n_ax)
    fig, ax = plt.subplots(figsize=(10.4, 5.8))

    for j in range(n_ax):
        ax.axvline(j, color="#c9c9c9", lw=0.9, zorder=1)

    for name, col in zip(GROUPS, COLORS):
        centre = np.array(PROFILES[name])
        for _ in range(N_SITES):
            prof = np.clip(centre + rng.normal(0, 0.075, n_ax), 0.01, 0.99)
            ax.plot(x, prof, color=col, alpha=0.20, lw=0.8, zorder=2)
        ax.plot(x, centre, color=col, lw=2.4, label=name, zorder=5,
                solid_capstyle="round")
        ax.scatter(x, centre, color=col, s=34, zorder=6, edgecolors="white",
                   linewidths=0.9)

    for j, (label, unit, lo, hi) in enumerate(AXES):
        ax.text(j, 1.035, f"{hi:g}", ha="center", va="bottom", fontsize=6.8,
                color="#666666")
        ax.text(j, -0.035, f"{lo:g}", ha="center", va="top", fontsize=6.8,
                color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a[0]}\n({a[1]})" for a in AXES], fontsize=8,
                       linespacing=1.5)
    ax.tick_params(axis="x", length=0, pad=16)
    ax.set_yticks([])
    ax.set_ylim(-0.10, 1.12)
    ax.set_xlim(-0.35, n_ax - 0.65)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Axis-wise min-max normalised value", fontsize=8.5)
    ax.set_title("Multivariate ecosystem-service signatures by protection zone",
                 fontsize=10.5, fontweight="bold", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              fontsize=8.5, frameon=False, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    savefig(fig, "41_parallel_coords.png")


if __name__ == "__main__":
    main()
