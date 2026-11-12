"""Two midfielders compared on nine percentile axes, with the league behind it.

A reference population of 240 same-position players is drawn once, each of the
nine metrics from its own distribution (several of them deliberately skewed).
The two players on the radar are *members* of that population, so their radial
coordinate is literally their rank inside it: the radar cannot drift away from
the violins on the right, because the violins are the population the ranks
were counted in.  The similarity ranking is the Euclidean distance to player A
in the same nine-dimensional percentile space.

Percentiles rather than raw values are what makes the nine axes comparable at
all - "6.4 progressive passes" and "58% duels won" have no common scale - and
the raw value is printed under each axis label so the reader can still see
what the rank was earned with.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import savefig, set_pub_style
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
A_COL = "#3b5b92"         # house blue - player A
B_COL = "#b23a48"         # house red - player B
POP = "#c7d3e4"           # league population bodies
GRID = "#d9d9d9"

N_POP = 240
N_SHOW = 6

# short label, unit, distribution, location, scale, print format
METRICS = [
    ("Progressive\npasses", "/90", "gamma", 5.6, 1.05, "{:.1f}"),
    ("Pass\ncompletion", "%", "norm", 82.0, 5.2, "{:.0f}"),
    ("Key\npasses", "/90", "gamma", 2.4, 0.62, "{:.1f}"),
    ("xG\nassisted", "/90", "gamma", 0.17, 0.055, "{:.2f}"),
    ("Dribbles\ncompleted", "/90", "gamma", 1.5, 0.55, "{:.1f}"),
    ("Duels\nwon", "%", "norm", 51.0, 6.4, "{:.0f}"),
    ("Ball\nrecoveries", "/90", "norm", 7.4, 1.9, "{:.1f}"),
    ("Pressures\napplied", "/90", "norm", 21.0, 4.6, "{:.0f}"),
    ("Distance\ncovered", "km", "norm", 10.6, 0.72, "{:.1f}"),
]
FLAT = [m[0].replace("\n", " ") for m in METRICS]

SURNAMES = [
    "Nakamura", "Bergstrom", "Delacroix", "Okonkwo", "Marchetti", "Vasquez",
    "Halvorsen", "Ferreira", "Kowalski", "Ibarra", "Sorensen", "Andrade",
    "Novak", "Bouchard", "Vandermeer", "Kiryu", "Serrano", "Blomqvist",
    "Achebe", "Tanaka", "Petrov", "Lindqvist", "Moreau", "Duarte",
    "Haraldsen", "Oyelaran", "Castellanos", "Wiegand", "Fontaine", "Rahman",
    "Solberg", "Mendoza", "Kovacic", "Lindberg", "Adeyemi", "Sato",
    "Rossini", "Bakker", "Grimaldi", "Larsen",
]


def draw_population(rng):
    """One (N_POP, 9) table of raw metric values; several axes are skewed."""
    cols = []
    for _lab, _u, kind, loc, scale, _f in METRICS:
        if kind == "gamma":
            k = (loc / scale) ** 2
            cols.append(rng.gamma(k, loc / k, N_POP))
        else:
            cols.append(rng.normal(loc, scale, N_POP))
    return np.column_stack(cols)


def main():
    set_pub_style()
    rng = np.random.default_rng(868)

    raw = draw_population(rng)
    # percentile = share of the population this player actually beats
    pct = 100.0 * (raw[:, None, :] > raw[None, :, :]).mean(axis=1)

    names, seen = [], set()
    while len(names) < N_POP:
        nm = (f"{chr(65 + int(rng.integers(0, 26)))}. "
              f"{SURNAMES[int(rng.integers(0, len(SURNAMES)))]}")
        if nm not in seen:
            seen.add(nm)
            names.append(nm)

    creative = np.array([1.1, 0.5, 1.3, 1.2, 1.0, 0.2, 0.2, 0.3, 0.4])
    engine = np.array([0.6, 1.0, 0.2, 0.1, 0.2, 1.1, 1.3, 1.2, 1.0])
    # deliberately contrasting picks, so the two polygons are worth overlaying
    ia = int(np.argmax(pct @ creative - 0.75 * (pct @ engine)))
    ib = int(np.argmax(pct @ engine - 0.75 * (pct @ creative)))
    name_a, name_b = names[ia], names[ib]
    pa, pb = pct[ia], pct[ib]
    va, vb = raw[ia], raw[ib]
    rate_a, rate_b = pa.mean(), pb.mean()

    # ---- similarity in the same percentile space ---------------------------
    dd = np.sqrt(((pct - pa) ** 2).sum(axis=1))
    sim = 100.0 * (1.0 - dd / dd.max())
    near = [i for i in np.argsort(dd) if i != ia][:N_SHOW][::-1]

    # ---- standardised population, for the violins --------------------------
    z = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    za, zb = z[ia], z[ib]

    # ---- layout -------------------------------------------------------------
    FW, FH = 10.4, 6.8
    fig = plt.figure(figsize=(FW, FH))
    ax_r = axes_inches(fig, (0.95, 1.02, 3.90, 4.10), projection="polar")
    ax_v = axes_inches(fig, (6.62, 3.20, 3.38, 2.20))
    ax_s = axes_inches(fig, (6.62, 1.05, 3.38, 1.50))

    # ---- radar ---------------------------------------------------------------
    k = len(METRICS)
    ang = 2.0 * np.pi * np.arange(k) / k
    loop = np.append(ang, ang[0])
    ax_r.set_theta_zero_location("N")
    ax_r.set_theta_direction(-1)
    ax_r.set_ylim(0.0, 100.0)
    ax_r.set_yticks([20, 40, 60, 80, 100])
    ax_r.set_yticklabels(["20th", "40th", "60th", "80th", "100th"],
                         fontsize=6.0, color=MUTED)
    ax_r.set_rlabel_position(np.degrees(ang[1]) * 0.5)
    ax_r.set_xticks(ang)
    ax_r.set_xticklabels([])
    ax_r.grid(color=GRID, lw=0.6)
    ax_r.spines["polar"].set_color(GRID)
    ax_r.spines["polar"].set_linewidth(0.8)
    ax_r.set_facecolor("#fbfbfb")

    for pp, col, nm in ((pa, A_COL, name_a), (pb, B_COL, name_b)):
        rr = np.append(pp, pp[0])
        ax_r.fill(loop, rr, color=col, alpha=0.20, lw=0.0, zorder=3)
        ax_r.plot(loop, rr, color=col, lw=1.6, zorder=4, label=nm)
        ax_r.plot(ang, pp, marker="o", ls="none", ms=3.4, mfc="white",
                  mec=col, mew=1.1, zorder=5)

    for i, (lab, unit, _k, _l, _s, fmt) in enumerate(METRICS):
        a = ang[i]
        sx, sy = np.sin(a), np.cos(a)
        ha = "left" if sx > 0.20 else ("right" if sx < -0.20 else "center")
        vaa = "bottom" if sy > 0.20 else ("top" if sy < -0.20 else "center")
        txt = (f"{lab}\n{fmt.format(va[i])} / {fmt.format(vb[i])} {unit}")
        ax_r.text(a, 108.0, txt, ha=ha, va=vaa, fontsize=6.4, color=INK,
                  linespacing=1.20, zorder=6)


    # ---- league distributions -------------------------------------------------
    yy = np.arange(k)[::-1]
    parts = ax_v.violinplot([z[:, i] for i in range(k)], positions=yy,
                            vert=False, widths=0.86, showextrema=False,
                            showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor(POP)
        body.set_edgecolor("#9fb3cc")
        body.set_linewidth(0.5)
        body.set_alpha(0.95)
    ax_v.axvline(0.0, color="#808080", lw=0.8, ls="--", zorder=2)
    ax_v.plot(za, yy, marker="o", ls="none", ms=4.6, mfc=A_COL, mec="white",
              mew=0.9, zorder=6)
    ax_v.plot(zb, yy, marker="s", ls="none", ms=4.2, mfc=B_COL, mec="white",
              mew=0.9, zorder=6)
    ax_v.set_yticks(yy)
    ax_v.set_yticklabels(FLAT, fontsize=6.2)
    ax_v.set_xlim(-3.4, 3.4)
    ax_v.set_ylim(-0.75, k - 0.25)
    ax_v.set_xlabel("standardised score (SD from the league mean)",
                    fontsize=8.0)
    ax_v.tick_params(axis="x", labelsize=7.0)
    ax_v.grid(axis="x", lw=0.5, color="#f0f0f0", zorder=0)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    ax_v.set_title(f"Where they sit in a {N_POP}-player league",
                   fontsize=8.5, fontweight="bold")

    # ---- similarity ranking ---------------------------------------------------
    yy2 = np.arange(len(near))
    ax_s.hlines(yy2, 0.0, sim[near], color="#9cc3dd", lw=1.4, zorder=3)
    ax_s.plot(sim[near], yy2, marker="o", ls="none", ms=5.0, mfc=A_COL,
              mec="white", mew=0.8, zorder=4)
    for y, i in zip(yy2, near):
        ax_s.text(sim[i] + 0.4, y, f"{sim[i]:.1f}", ha="left", va="center",
                  fontsize=6.2, color=SUB)
    ax_s.set_yticks(yy2)
    ax_s.set_yticklabels([names[i] for i in near], fontsize=6.4)
    ax_s.set_xlim(0.0, sim[near].max() * 1.16)
    ax_s.set_ylim(-0.7, len(near) - 0.35)
    ax_s.set_xlabel(f"similarity to {name_a} (100 = identical profile)",
                    fontsize=8.0)
    ax_s.tick_params(axis="x", labelsize=7.0)
    ax_s.grid(axis="x", lw=0.5, color="#ececec", zorder=0)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.set_title("Closest profiles in the same position", fontsize=8.5,
                   fontweight="bold")

    fig.text(2.90 / FW, 5.62 / FH, "Nine-axis percentile profile, same "
             "position", ha="center", va="bottom", fontsize=9.0,
             fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax_v, ax_s], labels=["(b)", "(c)"], dx=-0.30,
                        dy=0.12)
    fig.text(0.30 / FW, 5.62 / FH, "(a)", fontsize=11, fontweight="bold",
             color=INK, ha="left", va="bottom")

    gap = pa - pb
    big = int(np.argmax(np.abs(gap)))

    print(f"A {name_a} overall {rate_a:.1f}  B {name_b} overall {rate_b:.1f}")
    print("A percentiles:", pa.round(0).tolist())
    print("B percentiles:", pb.round(0).tolist())
    print("A raw:", np.round(va, 2).tolist())
    print("closest:", [(names[i], round(sim[i], 1)) for i in near[::-1]])
    savefig(fig, "868_tac_player_radar.png")


if __name__ == "__main__":
    main()
