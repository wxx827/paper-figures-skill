"""Figure 395 - Sigmoid growth trajectories of six wheat varieties.

Plant height, above-ground biomass and leaf area index differ by three orders
of magnitude in their native units, so every trait is drawn relative to its own
genotype maximum (the absolute maxima are printed in each panel).  That is what
makes one shared y axis across all six panels honest, and it is the only way a
reader can see that the late varieties simply shift the whole S curve right.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, letter_panels
from figkit import shared_limits

SEED = 2027

# Three traits, house categorical positions 3 / 2 / 4.
TRAITS = [
    ("Plant height", "#3b5b92", "o", "m", "H"),
    ("Above-ground biomass", "#c9a227", "s", "t/ha", "B"),
    ("Leaf area index", "#2e7d5b", "^", "m$^2$/m$^2$", "LAI"),
]

# name, maturity class, thermal-time midpoint shift, steepness, trait maxima
VARIETIES = [
    ("Aurora",   "early", -175.0, 1.06, (0.92, 17.6, 4.9)),
    ("Borealis", "early", -120.0, 1.00, (1.03, 19.2, 5.4)),
    ("Cascade",  "mid",    -15.0, 0.94, (1.11, 21.8, 6.1)),
    ("Delta",    "mid",     40.0, 0.90, (1.06, 20.9, 5.8)),
    ("Everest",  "late",   130.0, 0.84, (1.18, 23.4, 6.6)),
    ("Fjord",    "late",   195.0, 0.79, (1.14, 22.1, 6.3)),
]

# Trait-specific timing: height finishes first, LAI peaks last.
TRAIT_T0 = [860.0, 1010.0, 940.0]
TRAIT_K = [0.0062, 0.0051, 0.0058]

TT = np.linspace(0.0, 2200.0, 400)          # thermal time, degree-days


def logistic(t, t0, k):
    return 1.0 / (1.0 + np.exp(-k * (t - t0)))


def variety_curves(shift, steep):
    """Relative curves plus the observation triplets for one variety."""
    curves, obs = [], []
    for j in range(3):
        t0 = TRAIT_T0[j] + shift
        k = TRAIT_K[j] * steep
        y = logistic(TT, t0, k)
        y = y / y[-1]                        # relative to genotype maximum
        curves.append((t0, k, y))
    return curves, obs


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    t_obs = np.array([280.0, 520.0, 760.0, 1000.0, 1250.0, 1520.0,
                      1820.0, 2120.0])

    data = []
    for shift, steep in [(v[2], v[3]) for v in VARIETIES]:
        curves, _ = variety_curves(shift, steep)
        rows = []
        for t0, k, y in curves:
            mu = logistic(t_obs, t0, k) / logistic(TT[-1], t0, k)
            se = 0.021 + 0.052 * np.exp(-((t_obs - t0) / 460.0) ** 2)
            val = np.clip(mu + rng.normal(0.0, se * 0.80), 0.0, None)
            rows.append((t0, k, y, val, se))
        data.append(rows)

    pool = [r[2] for v in data for r in v]
    pool += [r[3] + r[4] for v in data for r in v]
    lo, hi = shared_limits(pool, pad=0.06)
    lo = min(lo, -0.02)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4), sharex=True,
                             sharey=True)

    for i, (ax, (name, cls, shift, steep)) in enumerate(
            zip(axes.ravel(), [(v[0], v[1], v[2], v[3]) for v in VARIETIES])):
        maxima = VARIETIES[i][4]
        rows = data[i]

        # Inflection of the LAI curve drives the panel annotation.
        note_max = []
        for j, (t0, k, y) in enumerate([(r[0], r[1], r[2]) for r in rows]):
            col = TRAITS[j][1]
            ax.plot(TT, y, color=col, lw=1.5, zorder=3)
            ax.errorbar(t_obs, rows[j][3], yerr=rows[j][4], fmt=TRAITS[j][2],
                        ms=3.0, mfc="white", mec=col, mew=0.9, ecolor=col,
                        elinewidth=0.7, capsize=1.8, ls="none", zorder=4)
            ax.plot([t0], [np.interp(t0, TT, y)], marker="*", ms=8.0,
                    mfc=col, mec="white", mew=0.7, zorder=6)
            note_max.append(f"{TRAITS[j][4]}$_{{max}}$ = "
                            f"{maxima[j]:.1f} {TRAITS[j][3]}")

        t0_lai, k_lai, y_lai = rows[2][0], rows[2][1], rows[2][2]
        rate = np.gradient(y_lai, TT)
        r_max = rate.max() * 100.0                 # relative units / 100 dd
        t_infl = TT[int(np.argmax(rate))]
        ax.axvline(t_infl, color="#808080", lw=0.7, ls=(0, (3.0, 2.0)),
                   zorder=1)

        ax.set_title(f"{name}  ({cls}-maturing)", fontsize=8.5,
                     fontweight="bold")
        ax.text(0.035, 0.965,
                f"LAI inflection  {t_infl:.0f} \u00b0C\u00b7d\n"
                f"max rate  {r_max:.2f} rel. / 100 \u00b0C\u00b7d",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                color="#555555", linespacing=1.4)
        ax.text(0.975, 0.045, "\n".join(note_max), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.4, color="#333333",
                linespacing=1.5)

        ax.set_xlim(0.0, 2200.0)
        ax.set_ylim(lo, hi)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.tick_params(labelsize=7.0)
        ax.grid(axis="y", lw=0.5, alpha=0.55)

    for ax in axes[:, 0]:
        ax.set_ylabel("trait, relative to genotype maximum (-)", fontsize=8.0)
    for ax in axes[1, :]:
        ax.set_xlabel("thermal time since sowing (\u00b0C\u00b7d, base 0 "
                      "\u00b0C)", fontsize=8.0)

    letter_panels(axes, x=-0.10, y=1.04)

    handles = [Line2D([0], [0], color=c, lw=1.5, marker=m, ms=3.4,
                      mfc="white", mec=c, label=f"{n} ({u})")
               for n, c, m, u, _ in TRAITS]
    handles.append(Line2D([0], [0], color="#555555", lw=0, marker="*", ms=7.0,
                          mfc="#555555", label="logistic inflection"))
    handles.append(Line2D([0], [0], color="#808080", lw=0.7,
                          ls=(0, (3.0, 2.0)), label="LAI inflection"))

    fig.tight_layout(rect=[0, 0.028, 1, 0.905], h_pad=2.6, w_pad=2.2)

    savefig(fig, "395_agr_growth_curves.png")


if __name__ == "__main__":
    main()
