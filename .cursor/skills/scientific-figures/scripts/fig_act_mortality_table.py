"""Life table: the mortality curve read in per mille, phase by phase.

Main panel: the annual death probability q_x on a semi-log axis for two sexes
and two table vintages, with the four textbook segments bracketed along the
top -- the infant shoulder, the juvenile trough, the young-adult accident
hump and the log-linear Gompertz ascent.  The two narrow right-hand panels
carry what the table is actually used for: the survivorship column l_x with
its median attained age, and the curtate expectation of life e_x.

Mortality is generated as a force of mortality
mu(x) = A exp(-Bx) + C + D exp(-E ln(x/F)^2) + G exp(Hx),
converted to q_x = 1 - exp(-mu); l_x and e_x are recursed from that same
q_x vector, so no two panels can disagree.  Values are illustrative.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches

MALE = "#3b5b92"
FEMALE = "#2e7d5b"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.80, 0.74, 5.30, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

AGE_MAX = 100
RADIX = 100000.0

# A: infant decline, C: Makeham floor, D/E/F: accident hump, G/H: Gompertz
PARAMS = {
    ("Male", 2000): dict(A=7.5e-3, B=1.35, C=1.6e-4, D=5.5e-4, E=9.0,
                         F=21.0, G=1.90e-5, H=0.0935),
    ("Female", 2000): dict(A=6.2e-3, B=1.35, C=1.2e-4, D=2.2e-4, E=10.5,
                           F=23.0, G=1.35e-5, H=0.0945),
    ("Male", 1960): dict(A=1.9e-2, B=1.20, C=4.2e-4, D=8.2e-4, E=8.5,
                         F=20.0, G=3.60e-5, H=0.0905),
    ("Female", 1960): dict(A=1.6e-2, B=1.20, C=3.0e-4, D=3.5e-4, E=10.0,
                           F=22.0, G=2.60e-5, H=0.0915),
}
SERIES = [("Male", 2000), ("Female", 2000), ("Male", 1960), ("Female", 1960)]
COL = {"Male": MALE, "Female": FEMALE}
DASH = {2000: "-", 1960: (0, (3.4, 2.0))}
LW = {2000: 1.9, 1960: 1.15}


def force_of_mortality(x, A, B, C, D, E, F, G, H):
    """Four-component force of mortality on an age vector."""
    xs = np.maximum(np.asarray(x, dtype=float), 1e-6)
    infant = A * np.exp(-B * xs)
    hump = D * np.exp(-E * np.log(xs / F) ** 2)
    senescent = G * np.exp(H * xs)
    return infant + C + hump + senescent


def life_table(q):
    """Survivors l_x on radix RADIX and curtate expectation of life e_x."""
    lx = np.empty(q.size + 1)
    lx[0] = RADIX
    for i, qi in enumerate(q):
        lx[i + 1] = lx[i] * (1.0 - qi)
    big_l = 0.5 * (lx[:-1] + lx[1:])
    tx = np.cumsum(big_l[::-1])[::-1]
    return lx[:-1], tx / np.maximum(lx[:-1], 1e-9)


def build_tables(seed=882):
    """{(sex, vintage): q_x in per mille, l_x, e_x} on ages 0..AGE_MAX."""
    rng = np.random.default_rng(seed)
    ages_full = np.arange(0, 121)
    out = {}
    for key in SERIES:
        mu = force_of_mortality(ages_full, **PARAMS[key])
        q = 1.0 - np.exp(-mu)
        # graduation scatter: published tables are smoothed, never exact
        q = np.clip(q * np.exp(rng.normal(0.0, 0.014, q.size)), 1e-6, 0.92)
        lx, ex = life_table(q)
        n = AGE_MAX + 1
        out[key] = dict(q=q[:n] * 1000.0, lx=lx[:n], ex=ex[:n])
    return out


def phase_bracket(ax, x0, x1, y, label, color=SUB, drop=0.78):
    """Square bracket spanning an age segment, label centred above it."""
    ax.plot([x0, x0, x1, x1], [y * drop, y, y, y * drop], lw=0.8,
            color=color, zorder=6, solid_joinstyle="miter")
    ax.text(0.5 * (x0 + x1), y * 1.22, label, ha="center", va="bottom",
            fontsize=6.8, color=color, zorder=6)


def main():
    set_pub_style(base_font=8)
    tab = build_tables()
    ages = np.arange(0, AGE_MAX + 1)

    q_m00 = tab[("Male", 2000)]["q"]
    q_f00 = tab[("Female", 2000)]["q"]
    q_m60 = tab[("Male", 1960)]["q"]

    # every annotated number below is read back off the plotted arrays
    trough = int(6 + np.argmin(q_m00[6:18]))
    hump = int(16 + np.argmax(q_m00[16:34]))
    hump_ratio = float(q_m00[hump] / q_m00[trough])
    infant_ratio = float(q_m00[0] / q_m00[trough])
    gomp = np.polyfit(ages[60:], np.log(q_m00[60:]), 1)
    doubling = float(np.log(2.0) / gomp[0])
    improve = float(np.median(q_m60[1:] / q_m00[1:]))
    sex_gap = float(q_m00[hump] / q_f00[hump])
    decades = float(np.log10(q_m00.max() / q_m00.min()))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_l = axes_inches(fig, AUX_T_RECT)
    ax_e = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_l, ax_e):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: q_x per mille on a log axis ---------------------------------
    ax.set_yscale("log")
    for sex, era in SERIES:
        ax.plot(ages, tab[(sex, era)]["q"], color=COL[sex], lw=LW[era],
                ls=DASH[era], zorder=5 if era == 2000 else 3,
                solid_capstyle="round")
    ax.set_xlim(-2.0, AGE_MAX + 2.0)
    ax.set_ylim(0.12, 1000.0)
    ax.set_xticks(np.arange(0, AGE_MAX + 1, 10))
    ax.set_xlabel("attained age $x$ (years)", fontsize=8.5)
    ax.set_ylabel("$q_x$, probability of death within one year "
                  "(per mille, log scale)", fontsize=8.5)
    ax.set_title("Four segments of the mortality curve", fontsize=9.0,
                 fontweight="bold", pad=5.0)

    phase_bracket(ax, 0, 4, 330.0, "infant")
    phase_bracket(ax, 7, 15, 330.0, "trough")
    phase_bracket(ax, 17, 32, 330.0, "accident hump", color=ACCENT)
    phase_bracket(ax, 55, AGE_MAX, 330.0, "Gompertz log-linear rise")

    ax.plot([trough], [q_m00[trough]], marker="o", ms=3.8, mfc="white",
            mec=MALE, mew=1.2, zorder=8)
    ax.plot([hump], [q_m00[hump]], marker="o", ms=3.8, mfc="white",
            mec=ACCENT, mew=1.2, zorder=8)
    ax.annotate(f"male hump at age {hump} yr: {q_m00[hump]:.2f} per mille,\n"
                f"{hump_ratio:.2f}x the age-{trough} trough "
                f"(M/F ratio {sex_gap:.2f}x)",
                xy=(hump + 0.5, q_m00[hump] * 1.12), xytext=(24.0, 22.0),
                fontsize=6.9, color=ACCENT, linespacing=1.45,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ACCENT,
                                shrinkA=2, shrinkB=1))
    ax.text(1.0, 88.0,
            f"infant rate is {infant_ratio:.0f}x the juvenile trough; "
            f"$q_x$ spans {decades:.1f} decades of risk\n"
            f"after age 60 the curve doubles every {doubling:.1f} yr, and the "
            f"1960 table sits {improve:.1f}x above 2000",
            ha="left", va="bottom", fontsize=6.8, color=MUTED,
            linespacing=1.5)

    handles = [Line2D([], [], color=COL[s], lw=LW[e], ls=DASH[e],
                      label=f"{s}, {e} table") for s, e in SERIES]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.3,
              handlelength=2.5, labelspacing=0.42, borderpad=0.30,
              bbox_to_anchor=(0.998, 0.005))

    # ---- aux (b): survivorship ---------------------------------------------
    for sex, era in SERIES:
        ax_l.plot(ages, tab[(sex, era)]["lx"] / 1000.0, color=COL[sex],
                  lw=1.5 if era == 2000 else 1.0, ls=DASH[era],
                  zorder=5 if era == 2000 else 3)
    ax_l.axhline(RADIX / 2000.0, color=FAINT, lw=0.7, ls=(0, (2.4, 2.0)),
                 zorder=2)
    med = {}
    for sex in ("Male", "Female"):
        lx = tab[(sex, 2000)]["lx"]
        med[sex] = int(ages[int(np.argmin(np.abs(lx - RADIX / 2.0)))])
        ax_l.plot([med[sex]], [lx[med[sex]] / 1000.0], marker="o", ms=3.6,
                  mfc="white", mec=COL[sex], mew=1.2, zorder=7)
    ax_l.set_xlim(-2.0, AGE_MAX + 2.0)
    ax_l.set_ylim(0, 106)
    ax_l.set_xticks([0, 20, 40, 60, 80, 100])
    ax_l.set_yticks([0, 25, 50, 75, 100])
    ax_l.tick_params(labelsize=6.8)
    ax_l.set_xlabel("age $x$ (years)", fontsize=7.8, labelpad=1.5)
    ax_l.set_ylabel("$l_x$ (thousand survivors)", fontsize=7.8)
    ax_l.set_title("Survivorship rectangularises", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_l.text(0.035, 0.06,
              f"radix {RADIX / 1000.0:.0f}k lives; half the 2000 cohort is\n"
              f"gone by age {med['Male']} yr (M) and {med['Female']} yr (F)",
              transform=ax_l.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): expectation of life --------------------------------------
    for sex, era in SERIES:
        ax_e.plot(ages, tab[(sex, era)]["ex"], color=COL[sex],
                  lw=1.5 if era == 2000 else 1.0, ls=DASH[era],
                  zorder=5 if era == 2000 else 3)
    ax_e.axvline(65, color=FAINT, lw=0.7, ls=(0, (2.4, 2.0)), zorder=2)
    ax_e.set_xlim(-2.0, AGE_MAX + 2.0)
    ax_e.set_ylim(0, 92)
    ax_e.set_xticks([0, 20, 40, 60, 80, 100])
    ax_e.tick_params(labelsize=6.8)
    ax_e.set_xlabel("age $x$ (years)", fontsize=7.8, labelpad=1.5)
    ax_e.set_ylabel("$e_x$ (further years expected)", fontsize=7.8)
    ax_e.set_title("Expectation of life", fontsize=8.5, fontweight="bold",
                   pad=4.0)
    e0f = tab[("Female", 2000)]["ex"][0]
    e0m = tab[("Male", 2000)]["ex"][0]
    e0m60 = tab[("Male", 1960)]["ex"][0]
    e65f = tab[("Female", 2000)]["ex"][65]
    e65m = tab[("Male", 2000)]["ex"][65]
    ax_e.text(0.965, 0.955,
              f"$e_0$ = {e0f:.1f} F / {e0m:.1f} M yr (2000)\n"
              f"$e_0$ = {e0m60:.1f} yr, male 1960 table\n"
              f"$e_{{65}}$ = {e65f:.1f} F / {e65m:.1f} M yr",
              transform=ax_e.transAxes, ha="right", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_l, ax_e], dx=-0.56, dy=0.10)
    savefig(fig, "882_act_mortality_table.png")


if __name__ == "__main__":
    main()
