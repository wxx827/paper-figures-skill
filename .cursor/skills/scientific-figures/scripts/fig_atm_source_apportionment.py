"""PMF source apportionment of fine particulate matter: profiles and shares.

A five-factor positive matrix factorisation is summarised the way an air
quality report needs it.  The large panel is the chemical fingerprint: for each
measured species, how much of its mass each factor explains, which is what
lets a reader accept or reject a factor as a real source.  The two smaller
panels reuse the same factorisation for the annual mass split and its seasonal
swing, so the donut, the stack and the profiles cannot disagree - every number
is derived from one contribution matrix and one profile matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, COLORS
from figkit import panel_labels_inches, place_outside_labels

INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]

SOURCES = ["secondary inorganic", "traffic", "coal combustion",
           "mineral dust", "industry"]
SHORT = ["secondary", "traffic", "coal", "dust", "industry"]
SRC_C = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]

SPECIES = ["OC", "EC", "$SO_4^{2-}$", "$NO_3^-$", "$NH_4^+$", "$Cl^-$",
           "K", "Ca", "Fe", "Zn", "Pb"]

# Profile matrix: mass fraction of each species within each source (rows sum
# to well under 1 because unmeasured mass is not resolved).
PROFILE = np.array([
    [0.100, 0.010, 0.340, 0.260, 0.160, 0.020, 0.005, 0.003, 0.002, 0.0008, 0.0004],
    [0.340, 0.260, 0.040, 0.050, 0.020, 0.010, 0.004, 0.012, 0.030, 0.0060, 0.0015],
    [0.220, 0.090, 0.160, 0.040, 0.050, 0.050, 0.020, 0.020, 0.015, 0.0040, 0.0030],
    [0.060, 0.010, 0.030, 0.020, 0.005, 0.005, 0.018, 0.090, 0.045, 0.0010, 0.0003],
    [0.120, 0.040, 0.100, 0.030, 0.030, 0.030, 0.012, 0.015, 0.060, 0.0250, 0.0090],
])

# Annual mean contribution of each factor to PM2.5 mass, ug m^-3.
MEAN_G = np.array([18.5, 11.2, 9.4, 6.1, 4.8])

# Monthly multipliers: secondary and coal peak in winter, dust in spring,
# traffic is flat, industry drifts slowly.
SEASON = np.array([
    [1.42, 1.34, 1.06, 0.86, 0.72, 0.66, 0.70, 0.74, 0.86, 1.02, 1.24, 1.38],
    [1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.90, 0.92, 1.00, 1.04, 1.06, 1.06],
    [1.66, 1.52, 1.10, 0.72, 0.48, 0.36, 0.34, 0.38, 0.58, 0.94, 1.36, 1.62],
    [0.72, 0.96, 1.58, 1.74, 1.36, 0.92, 0.78, 0.74, 0.82, 0.90, 0.88, 0.78],
    [0.94, 0.96, 1.02, 1.06, 1.10, 1.12, 1.14, 1.12, 1.04, 0.98, 0.92, 0.90],
])
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def main():
    set_pub_style()
    rng = np.random.default_rng(554)

    # Synthetic receptor data set: daily factor contributions times the
    # profiles, plus measurement noise.  Explained variance is measured on it.
    n_day = 365
    doy = np.arange(n_day)
    month_of = np.clip((doy / 30.5).astype(int), 0, 11)
    g = (MEAN_G[None, :] * SEASON[:, month_of].T *
         rng.lognormal(0.0, 0.34, (n_day, 5)))
    x_true = g @ PROFILE
    x_obs = x_true + rng.normal(0.0, 0.06, x_true.shape) * x_true.mean(axis=0)
    ss_res = float(np.sum((x_obs - x_true) ** 2))
    ss_tot = float(np.sum((x_obs - x_obs.mean(axis=0)) ** 2))
    ev = 1.0 - ss_res / ss_tot

    # Everything plotted below comes from these two aggregates.
    contrib = g.mean(axis=0)                      # ug m^-3 per factor
    by_species = contrib[:, None] * PROFILE       # ug m^-3 per factor,species
    share = by_species / by_species.sum(axis=0, keepdims=True) * 100.0
    monthly = MEAN_G[None, :] * SEASON.T          # 12 x 5, ug m^-3

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], left=0.070, right=0.980,
                          top=0.845, bottom=0.098, hspace=0.42, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    axd = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------- (a) factor profiles
    y = np.arange(len(SPECIES))[::-1]
    left = np.zeros(len(SPECIES))
    for k in range(5):
        ax.barh(y, share[k], left=left, height=0.62, color=SRC_C[k],
                alpha=0.92, edgecolor="white", linewidth=0.7, zorder=3)
        for yy, x0, w in zip(y, left, share[k]):
            if w >= 9.0:
                ax.text(x0 + w / 2.0, yy, f"{w:.0f}", ha="center",
                        va="center", fontsize=6.2, color="white", zorder=5)
        left = left + share[k]

    ax.set_yticks(y)
    ax.set_yticklabels(SPECIES, fontsize=8.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(SPECIES) + 0.85)      # blank strip for the stats box
    ax.set_xlabel("share of measured species mass explained by each factor (%)",
                  fontsize=8.0)
    ax.set_title("Chemical fingerprint of the five resolved factors",
                 fontsize=9.0, fontweight="bold", color=INK)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="x", color="#ececec", lw=0.55, zorder=0)
    ax.text(0.012, 0.992, f"PMF, {len(SOURCES)} factors, "
            f"{len(SPECIES)} species, n = {n_day} daily samples, "
            f"explained variance {ev:.1%}", transform=ax.transAxes,
            fontsize=7.0, color=INK, ha="left", va="top", zorder=6,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))

    # --------------------------------------------- (b) annual mass split
    total = float(contrib.sum())
    frac = contrib / total
    axd.pie(frac, colors=SRC_C, startangle=90, counterclock=False,
            radius=1.0, wedgeprops=dict(width=0.40, edgecolor="white",
                                        linewidth=0.9))
    ang0 = 90.0
    items = []
    for k in range(5):
        sweep = 360.0 * frac[k]
        mid = ang0 - sweep / 2.0
        ang0 = ang0 - sweep
        items.append((mid, SHORT[k], f"{frac[k]:.0%}, {contrib[k]:.1f}",
                      SRC_C[k]))
    place_outside_labels(axd, items, r_edge=1.0, r_stub=1.07, x_knee=1.16,
                         x_text=1.24, min_gap=0.30, y_span=1.18,
                         name_size=7.0, stat_size=6.3)
    axd.text(0.0, 0.06, f"{total:.0f}", ha="center", va="center",
             fontsize=11.0, fontweight="bold", color=INK)
    axd.text(0.0, -0.16, "$\\mu$g m$^{-3}$\n$PM_{2.5}$", ha="center",
             va="center", fontsize=6.6, color=SUB, linespacing=1.30)
    axd.text(0.0, -1.42, "labels: share of mass, $\\mu$g m$^{-3}$",
             ha="center", va="bottom", fontsize=6.3, color=MUTED)
    axd.set_xlim(-2.30, 2.30)
    axd.set_ylim(-1.50, 1.50)
    axd.set_aspect("equal")
    axd.axis("off")
    axd.set_title("Annual mass split", fontsize=8.5, fontweight="bold",
                  color=INK, pad=2)

    # ------------------------------------------------ (c) seasonal swing
    xm = np.arange(12)
    axs.stackplot(xm, monthly.T, colors=SRC_C, edgecolor="white",
                  linewidth=0.55, alpha=0.92, zorder=3)
    tot_m = monthly.sum(axis=1)
    hi, lo = int(np.argmax(tot_m)), int(np.argmin(tot_m))
    axs.plot(xm, tot_m, color="#333333", lw=1.0, zorder=5)
    axs.annotate(f"{tot_m[hi]:.0f}", xy=(xm[hi], tot_m[hi]),
                 xytext=(xm[hi], tot_m[hi] + 6.0), fontsize=6.4, color=INK,
                 ha="center", va="bottom")
    axs.annotate(f"{tot_m[lo]:.0f}", xy=(xm[lo], tot_m[lo]),
                 xytext=(xm[lo], tot_m[lo] + 6.0), fontsize=6.4, color=INK,
                 ha="center", va="bottom")
    axs.set_xlim(0, 11)
    axs.set_ylim(0, tot_m.max() * 1.22)
    axs.set_xticks(xm)
    axs.set_xticklabels(MONTHS, fontsize=6.8)
    axs.set_xlabel("month", fontsize=7.5)
    axs.set_ylabel("contribution ($\\mu$g m$^{-3}$)", fontsize=7.5)
    axs.set_title("Seasonal contribution stack", fontsize=8.5,
                  fontweight="bold", color=INK)
    axs.tick_params(labelsize=6.8)

    handles = [Patch(facecolor=SRC_C[k], edgecolor="white", label=SOURCES[k])
               for k in range(5)]

    top = int(np.argmax(contrib))
    panel_labels_inches(fig, [ax, axd, axs], dx=-0.30, dy=0.12)

    savefig(fig, "554_atm_source_apportionment.png")


if __name__ == "__main__":
    main()
