"""Three centuries of succession: who holds the site, and how diverse it is.

One importance-value matrix drives all three panels.  Each species gets a
response along the chronosequence - log-normal pulses for the pioneers and
mid-successional trees, a logistic rise for the shade-tolerant climax pair -
which is then renormalised so the five values sum to 100 % at every age.
That normalisation is what makes the main panel a partition rather than five
unrelated curves, and it is what lets the side panels be derived rather than
invented: Shannon entropy and evenness in (b) are computed from the very
proportions stacked in (a), and the functional-group panel (c) is the same
matrix summed into three shade-tolerance classes.

Species colours are the shared forest set used in 684 and 685.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"

SPECIES = ["Betula pendula", "Pinus sylvestris", "Quercus robur",
           "Picea abies", "Fagus sylvatica"]
SPECIES_COLOR = {
    "Picea abies": "#2e7d5b",
    "Pinus sylvestris": "#b23a48",
    "Betula pendula": "#c9a227",
    "Quercus robur": "#3b5b92",
    "Fagus sylvatica": "#8e6bbf",
}
# (amplitude, colonisation age, colonisation rate, senescence age,
# senescence rate).  A None senescence age means the species never gives the
# site back, which is what "climax" means.
SPECIES_NICHE = {
    "Betula pendula": (1.00, 2.0, 1.5, 45.0, 12.0),
    "Pinus sylvestris": (1.00, 12.0, 4.0, 120.0, 30.0),
    "Quercus robur": (0.80, 45.0, 12.0, 240.0, 45.0),
    "Picea abies": (0.95, 85.0, 25.0, None, 0.0),
    "Fagus sylvatica": (1.15, 130.0, 32.0, None, 0.0),
}
GROUPS = ["shade intolerant", "mid tolerant", "shade tolerant"]
GROUP_OF = {"Betula pendula": "shade intolerant",
            "Pinus sylvestris": "shade intolerant",
            "Quercus robur": "mid tolerant",
            "Picea abies": "shade tolerant",
            "Fagus sylvatica": "shade tolerant"}
GROUP_COLOR = dict(zip(GROUPS, get_palette("cb_sequential_green", 5)[1:4]))

STAGES = [(0.0, 25.0, "initiation"), (25.0, 80.0, "stem exclusion"),
          (80.0, 160.0, "understorey reinit."), (160.0, 300.0, "old growth")]

FW, FH = 10.4, 6.4
MAIN_RECT = (0.74, 1.04, 5.30, 4.02)
AGE_LO, AGE_HI, N_AGE = 1.0, 300.0, 600
# one label slot per species, in successional order (fraction of the x range)
LABEL_AT = [0.09, 0.23, 0.43, 0.68, 0.86]


def build_importance(rng):
    t = np.linspace(AGE_LO, AGE_HI, N_AGE)
    w = {}
    for s in SPECIES:
        amp, t_on, k_on, t_off, k_off = SPECIES_NICHE[s]
        base = amp / (1.0 + np.exp(-(t - t_on) / k_on))
        if t_off is not None:
            base = base / (1.0 + np.exp((t - t_off) / k_off))
        wig = gaussian_filter1d(rng.normal(0.0, 1.0, t.size), 34.0)
        w[s] = np.clip(base * (1.0 + 0.30 * wig / np.abs(wig).max()), 1e-5,
                       None)
    tot = sum(w.values())
    iv = np.array([100.0 * w[s] / tot for s in SPECIES])
    return t, iv


def draw_main(ax, t, iv):
    ax.stackplot(t, iv, colors=[SPECIES_COLOR[s] for s in SPECIES],
                 edgecolor="white", linewidth=0.55, zorder=3)
    for x0, x1, name in STAGES[1:]:
        ax.axvline(x0, color="white", lw=1.1, ls=(0, (3.4, 2.2)), zorder=5)
    tb = blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, name in STAGES:
        ax.text(0.5 * (x0 + x1), 1.012, name, transform=tb, ha="center",
                va="bottom", fontsize=6.3, color=SUB)
    ax.set_xlim(AGE_LO, AGE_HI)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("stand age since stand-replacing disturbance (years)",
                  fontsize=8.0)
    ax.set_ylabel("importance value (%)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Importance value along the successional sequence",
                 fontsize=8.5, fontweight="bold", pad=13.0)

    # label each species inside its own band
    for k, s in enumerate(SPECIES):
        # anchor each label at its own slot along the sequence, so the five
        # names cannot collide however the bands happen to fall
        x = LABEL_AT[k] * AGE_HI
        i = int(np.argmin(np.abs(t - x)))
        if iv[k, i] < 8.0:
            continue
        y = iv[:k, i].sum() + 0.5 * iv[k, i]
        ax.text(x, y, s, ha="center", va="center", fontsize=6.8,
                color="white", zorder=6)


def draw_diversity(ax, t, iv):
    p = np.clip(iv / 100.0, 1e-9, None)
    shannon = -(p * np.log(p)).sum(axis=0)
    simpson = 1.0 - (p ** 2).sum(axis=0)
    ax.plot(t, shannon, color="#3b5b92", lw=1.5, zorder=4)
    ax.set_ylabel("Shannon $H'$ (nats)", fontsize=7.8, color="#3b5b92")
    ax.tick_params(axis="y", labelsize=6.8, colors="#3b5b92")
    ax.spines["left"].set_color("#3b5b92")
    ax.set_ylim(0.0, np.log(len(SPECIES)) * 1.06)
    ax.axhline(np.log(len(SPECIES)), color="#808080", lw=0.8,
               ls=(0, (4, 2.4)), zorder=2)
    ax.text(AGE_HI, np.log(len(SPECIES)) - 0.03,
            f"maximum for {len(SPECIES)} species", ha="right", va="top",
            fontsize=6.1, color=MUTED)

    ax2 = ax.twinx()
    ax2.plot(t, simpson, color="#c9a227", lw=1.3, ls=(0, (4.0, 1.8)),
             zorder=4)
    ax2.set_ylabel("Simpson $1-D$ (-)", fontsize=7.4, color="#c9a227")
    ax2.tick_params(labelsize=6.6, colors="#c9a227")
    ax2.set_ylim(0.0, 1.06)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#c9a227")
    ax2.spines["top"].set_visible(False)

    ax.set_xlim(AGE_LO, AGE_HI)
    ax.set_xlabel("stand age (years)", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_title("Diversity peaks in mid-succession", fontsize=8.5,
                 fontweight="bold")
    return shannon, simpson


def draw_groups(ax, t, iv):
    grp = np.array([sum(iv[k] for k, s in enumerate(SPECIES)
                        if GROUP_OF[s] == g) for g in GROUPS])
    ax.stackplot(t, grp, colors=[GROUP_COLOR[g] for g in GROUPS],
                 edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(AGE_LO, AGE_HI)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("stand age (years)", fontsize=7.8)
    ax.set_ylabel("share of importance (%)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Shade tolerance replaces light demand", fontsize=8.5,
                 fontweight="bold")
    cross = np.argmax(grp[2] > grp[0])
    ax.axvline(t[cross], color=RULE, lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.text(t[cross] + 5.0, 6.0, f"crossover  {t[cross]:.0f} yr", ha="left",
            va="bottom", fontsize=6.3, color=INK, zorder=6)
    return grp, float(t[cross])


def main():
    set_pub_style()
    rng = np.random.default_rng(688)
    t, iv = build_importance(rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN_RECT)
    ax_d = axes_inches(fig, (7.22, 3.22, 2.82, 1.84))
    ax_g = axes_inches(fig, (7.22, 1.04, 2.82, 1.52))

    draw_main(ax_m, t, iv)
    shannon, simpson = draw_diversity(ax_d, t, iv)
    grp, cross = draw_groups(ax_g, t, iv)

    handles = [Line2D([0], [0], marker="s", ls="none", ms=5.0,
                      mfc=SPECIES_COLOR[s], mec="white", mew=0.5, label=s)
               for s in SPECIES]
    handles += [Line2D([0], [0], marker="s", ls="none", ms=5.0,
                       mfc=GROUP_COLOR[g], mec="white", mew=0.5, label=g)
                for g in GROUPS]

    panel_labels_inches(fig, [ax_m, ax_d, ax_g], dx=-0.40, dy=0.30)

    peak_age = float(t[int(np.argmax(shannon))])
    pio_peak = float(t[int(np.argmax(iv[0]))])

    print(f"H'max={shannon.max():.3f} at {peak_age:.0f} yr, birch peak "
          f"{iv[0].max():.1f}% at {pio_peak:.0f} yr, beech end "
          f"{iv[4, -1]:.1f}%, crossover {cross:.0f} yr")
    savefig(fig, "688_for_species_succession.png")


if __name__ == "__main__":
    main()
