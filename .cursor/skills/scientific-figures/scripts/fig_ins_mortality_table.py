"""Life-table mortality: the three-phase shape of qx and what it implies.

Main panel: annual death probability q_x on a log axis for two sexes and two
table vintages, so the three classic features are all visible at once -- the
infant shoulder at age 0, the young-adult accident hump, and the near
log-linear senescent rise.  The two narrow right-hand panels carry the derived
quantities that an actuary actually prices with: the survivorship curve l_x and
the curtate expectation of life e_x, both computed from the very same q_x
series so the three panels cannot disagree.

Mortality is generated from a Heligman-Pollard law with era- and sex-specific
parameters; replace ``build_tables`` with a real table loader and nothing about
the styling has to change.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches

BLUE = "#3b5b92"      # male
GOLD = "#c9a227"      # female
RED = "#b23a48"        # single accent: the accident hump
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ededed"

AGE_MAX = 100
AGES_FULL = np.arange(0, 121)
RADIX = 1000.0

# Heligman-Pollard: q/(1-q) = A^((x+B)^C) + D exp(-E ln(x/F)^2) + G H^x
PARAMS = {
    ("male", 2020): dict(A=5.4e-4, B=0.017, C=0.101, D=3.2e-4, E=12.0,
                         F=19.5, G=1.05e-5, H=1.1105),
    ("female", 2020): dict(A=4.6e-4, B=0.017, C=0.104, D=1.3e-4, E=13.0,
                           F=21.0, G=8.0e-6, H=1.1080),
    ("male", 1980): dict(A=2.2e-3, B=0.019, C=0.098, D=4.6e-4, E=11.0,
                         F=20.5, G=2.6e-5, H=1.1035),
    ("female", 1980): dict(A=1.9e-3, B=0.019, C=0.100, D=1.9e-4, E=12.0,
                           F=22.0, G=1.9e-5, H=1.1010),
}
SERIES = [("male", 2020), ("female", 2020), ("male", 1980), ("female", 1980)]
COL = {"male": BLUE, "female": GOLD}
STYLE = {2020: "-", 1980: (0, (3.4, 2.0))}
LW = {2020: 1.9, 1980: 1.2}


def hp_qx(x, A, B, C, D, E, F, G, H, hump=True):
    """Heligman-Pollard annual death probability on an age vector."""
    xs = np.maximum(np.asarray(x, dtype=float), 1e-9)
    t1 = A ** ((xs + B) ** C)
    t2 = (D * np.exp(-E * np.log(xs / F) ** 2)) if hump else 0.0
    t3 = G * H ** xs
    r = t1 + t2 + t3
    return r / (1.0 + r)


def life_table(q):
    """Survivors l_x (radix RADIX) and curtate life expectancy e_x."""
    lx = np.empty(q.size + 1)
    lx[0] = RADIX
    for i, qi in enumerate(q):
        lx[i + 1] = lx[i] * (1.0 - qi)
    big_l = 0.5 * (lx[:-1] + lx[1:])
    tx = np.cumsum(big_l[::-1])[::-1]
    return lx[:-1], tx / np.maximum(lx[:-1], 1e-9)


def build_tables(seed=636):
    """{(sex, era): dict of q_x, l_x, e_x} on the plotted age grid."""
    rng = np.random.default_rng(seed)
    out = {}
    for key in SERIES:
        p = PARAMS[key]
        q = hp_qx(AGES_FULL, **p)
        # graduation noise: real published tables are smoothed, not exact
        q = np.clip(q * np.exp(rng.normal(0.0, 0.015, q.size)), 1e-6, 0.9)
        lx, ex = life_table(q)
        n = AGE_MAX + 1
        out[key] = dict(q=q[:n], lx=lx[:n], ex=ex[:n], q_full=q)
    return out


def hump_age(seed=636):
    """Age at which the accident term contributes most, from the law itself."""
    p = PARAMS[("male", 2020)]
    lo, hi = 12, 45
    x = AGES_FULL[lo:hi]
    gap = hp_qx(x, **p) - hp_qx(x, **dict(p, hump=False))
    return int(x[int(np.argmax(gap))])


def main():
    set_pub_style(base_font=8)
    tab = build_tables()
    ages = np.arange(0, AGE_MAX + 1)

    x_hump = hump_age()
    q_m20 = tab[("male", 2020)]["q"]
    q_m80 = tab[("male", 1980)]["q"]
    q_f20 = tab[("female", 2020)]["q"]
    # every annotated number below is read back off the plotted arrays
    improve = float(np.median(q_m80 / q_m20))
    decades = np.log10(q_m20.max() / q_m20.min())
    dip = int(20 + np.argmin(q_m20[20:45]))
    hump_ratio = float(q_m20[x_hump] / q_m20[dip])
    e0_m = tab[("male", 2020)]["ex"][0]
    e0_f = tab[("female", 2020)]["ex"][0]
    e0_m80 = tab[("male", 1980)]["ex"][0]
    e65_m = tab[("male", 2020)]["ex"][65]
    e65_f = tab[("female", 2020)]["ex"][65]
    sex_gap = float(q_m20[x_hump] / q_f20[x_hump])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          left=0.062, right=0.978, top=0.882, bottom=0.128,
                          hspace=0.46, wspace=0.205)
    ax = fig.add_subplot(gs[:, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])

    for a in (ax, ax_l, ax_e):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.7, zorder=0)
        style_axes(a)

    # ---- main: q_x on a log axis -------------------------------------------
    ax.set_yscale("log")
    for sex, era in SERIES:
        ax.plot(ages, tab[(sex, era)]["q"], color=COL[sex], lw=LW[era],
                ls=STYLE[era], zorder=5 if era == 2020 else 3,
                solid_capstyle="round")

    # the three phases, shaded and named
    bands = [(0, 4, "infant"), (x_hump - 7, x_hump + 9, "accident hump"),
             (60, AGE_MAX, "senescent rise")]
    for x0, x1, name in bands:
        ax.axvspan(x0, x1, facecolor="#f2f4f7", lw=0, zorder=1)
    ax.axvspan(x_hump - 7, x_hump + 9, facecolor="#f7eef0", lw=0, zorder=1)

    ax.set_xlim(-1.5, AGE_MAX + 1.5)
    ax.set_ylim(3e-5, 1.0)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Attained age  $x$ (years)", fontsize=9.0)
    ax.set_ylabel("$q_x$, annual probability of death (log scale, $-$)",
                  fontsize=9.0)
    ax.set_title("Three phases of the mortality curve", fontsize=9.0,
                 fontweight="bold", pad=5.0)

    ax.text(2.2, 0.030, "infant mortality", ha="left", va="bottom",
            fontsize=7.0, color=SUB)
    ax.annotate("", xy=(0.4, q_m20[0] * 1.8), xytext=(2.6, 0.026),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUTED,
                                shrinkA=0, shrinkB=1), zorder=8)
    ax.text(x_hump + 15.0, 0.036,
            f"accident hump: $q_{{{x_hump}}}$ is {hump_ratio:.2f}$\\times$\n"
            f"the age-{dip} trough", ha="left", va="bottom", fontsize=7.0,
            color=RED, linespacing=1.35)
    ax.annotate("", xy=(x_hump + 0.6, q_m20[x_hump] * 1.10),
                xytext=(x_hump + 14.0, 0.030),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=RED,
                                shrinkA=0, shrinkB=1), zorder=8)
    ax.text(AGE_MAX - 2, 3.0e-4, "senescent rise\n(log-linear in age)",
            ha="right", va="bottom", fontsize=7.0, color=SUB,
            linespacing=1.35)
    ax.text(56, 5.2e-5,
            f"1980 tables sit {improve:.2f}$\\times$ above 2020 at the\n"
            f"median age; $q_x$ spans {decades:.1f} decades",
            ha="left", va="bottom", fontsize=6.8, color=MUTED,
            linespacing=1.4)

    handles = [Line2D([], [], color=COL[s], lw=LW[e], ls=STYLE[e],
                      label=f"{s.capitalize()}, {e} table")
               for s, e in SERIES]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.4,
              handlelength=2.4, labelspacing=0.42, borderpad=0.35,
              bbox_to_anchor=(0.005, 0.995))

    # ---- aux (b): survivorship ---------------------------------------------
    for sex, era in SERIES:
        ax_l.plot(ages, tab[(sex, era)]["lx"], color=COL[sex],
                  lw=1.5 if era == 2020 else 1.0, ls=STYLE[era],
                  zorder=5 if era == 2020 else 3)
    ax_l.set_xlim(-1.5, AGE_MAX + 1.5)
    ax_l.set_ylim(0, RADIX * 1.04)
    ax_l.set_xticks([0, 20, 40, 60, 80, 100])
    ax_l.set_yticks([0, 250, 500, 750, 1000])
    ax_l.tick_params(labelsize=6.8)
    ax_l.set_xlabel("age  $x$ (years)", fontsize=8.0, labelpad=1.5)
    ax_l.set_ylabel("$l_x$ (survivors per 1,000)", fontsize=8.0)
    ax_l.set_title("Survivorship rectangularises", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    # median survival age, read off the plotted curve
    med = {}
    for sex in ("male", "female"):
        lx = tab[(sex, 2020)]["lx"]
        med[sex] = int(ages[np.argmin(np.abs(lx - RADIX / 2.0))])
        ax_l.plot([med[sex]], [lx[med[sex]]], marker="o", ms=3.6, mfc="white",
                  mec=COL[sex], mew=1.2, zorder=7)
    ax_l.axhline(RADIX / 2.0, color="#9a9a9a", lw=0.7, ls=(0, (2.4, 2.0)),
                 zorder=2)
    ax_l.text(0.035, 0.055,
              f"half the cohort is gone by\nage {med['male']} yr (M), "
              f"age {med['female']} yr (F)", transform=ax_l.transAxes,
              ha="left", va="bottom", fontsize=6.5, color=SUB,
              linespacing=1.45)

    # ---- aux (c): expectation of life --------------------------------------
    for sex, era in SERIES:
        ax_e.plot(ages, tab[(sex, era)]["ex"], color=COL[sex],
                  lw=1.5 if era == 2020 else 1.0, ls=STYLE[era],
                  zorder=5 if era == 2020 else 3)
    ax_e.set_xlim(-1.5, AGE_MAX + 1.5)
    ax_e.set_ylim(0, 92)
    ax_e.set_xticks([0, 20, 40, 60, 80, 100])
    ax_e.tick_params(labelsize=6.8)
    ax_e.set_xlabel("age  $x$ (years)", fontsize=8.0, labelpad=1.5)
    ax_e.set_ylabel("$e_x$ (expected years left)", fontsize=8.0)
    ax_e.set_title("Expectation of life", fontsize=8.5, fontweight="bold",
                   pad=4.0)
    ax_e.axvline(65, color="#9a9a9a", lw=0.7, ls=(0, (2.4, 2.0)), zorder=2)
    ax_e.text(0.965, 0.955,
              f"$e_0$ = {e0_f:.1f} F / {e0_m:.1f} M yr (2020)\n"
              f"$e_0$ = {e0_m80:.1f} yr M (1980)\n"
              f"$e_{{65}}$ = {e65_f:.1f} F / {e65_m:.1f} M yr",
              transform=ax_e.transAxes, ha="right", va="top", fontsize=6.5,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_l, ax_e], dx=-0.42, dy=0.06)

    savefig(fig, "636_ins_mortality_table.png")


if __name__ == "__main__":
    main()
