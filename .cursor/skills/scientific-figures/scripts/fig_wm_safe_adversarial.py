"""Figure 1204 - Task success under an L-infinity observation attack.

The curve panel and the example row are one object.  ``success`` is a closed
form per defence; the four tiles are rendered at the four budgets marked on
the curve, the caption under each tile reads its number off the *Ours* curve,
and how many vehicles the tile shows as detected is
``round(3 * success_ours(eps))`` - so a tile can never claim a detection rate
the curve above it disagrees with.

Change ``DEFENCES`` and the tiles follow; do not hand-type a tile caption.

All four tiles are the same frame under the same adversarial pattern at four
amplitudes, which is the one case where re-using a scene is the point rather
than a shortcut (BRIEF_WM section 1.6, step five).
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _wm_metrics import closed_ratio
from _wm_scenes import build, octave_noise
from pubstyle import savefig, set_pub_style, style_axes

SEED = 1609  # original id; figure number is 1204
EPS_MAX, N_EP = 0.16, 200            # attack budget and episodes per point
SHOTS = (0.000, 0.050, 0.100, 0.150)  # the four budgets rendered below
TILE = (150, 232)
N_VEH = 3

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
CARD, CARD_EDGE, GRID = "#e2e8ee", "#c3cdd6", "#ededed"
DEEP, MID, SOFT, PALE = "#0b3d63", "#2e6f9e", "#5f9dc6", "#9cc3dd"
MISS = "#8a97a3"

# (name, clean success, half-point budget, width, colour, dash, marker)
DEFENCES = [
    ("no defence", 0.940, 0.038, 0.0125, PALE, (0, (2.0, 2.0)), "s"),
    ("input smoothing", 0.921, 0.063, 0.0165, SOFT, (0, (5.0, 2.4)), "^"),
    ("adversarial training", 0.908, 0.090, 0.0205, MID, (0, (7.0, 2.2, 1.4,
                                                             2.2)), "D"),
    ("Ours: consistency + smoothing", 0.931, 0.121, 0.0245, DEEP, "solid",
     "o"),
]


def success(eps, s0, e50, w):
    """A logistic collapse pinned so that ``success(0) == s0`` exactly."""
    return s0 * (1.0 + np.exp(-e50 / w)) / (1.0 + np.exp((eps - e50) / w))


def breakdown(s0, e50, w, level=0.50):
    """The budget at which the curve first drops under ``level``."""
    g = np.linspace(0.0, EPS_MAX, 20001)
    y = success(g, s0, e50, w)
    k = int(np.argmax(y < level))
    return float(g[k]) if bool((y < level).any()) else float("nan")


def stderr(s):
    """Binomial standard error over N_EP episodes - printed in the caption."""
    return np.sqrt(np.clip(s, 0, 1) * (1.0 - np.clip(s, 0, 1)) / N_EP)


def tiles():
    """One frame, one adversarial pattern, four amplitudes."""
    sc = build("road", seed=SEED, shape=TILE, t=0.55)
    rng = np.random.default_rng(SEED + 3)
    # high ``base`` keeps the octaves small: an attack pattern is fine grain,
    # not the low-frequency cloud a default noise field produces
    f = octave_noise(rng, TILE, base=80.0, octaves=3)
    f = np.tanh(2.4 * f / (np.abs(f).max() + 1e-9))
    veh = [a for a in sc["anchors"] if a["name"] == "vehicle"]
    veh = sorted(veh, key=lambda a: -a["rx"])[:N_VEH]
    out = []
    for e in SHOTS:
        img = np.clip(sc["rgb"] + 2.1 * e * f[..., None], 0.0, 1.0)
        out.append(img)
    return out, veh


def main():
    set_pub_style(base_font=8.5)
    grid = np.linspace(0.0, EPS_MAX, 321)
    curves = [dict(name=nm, s0=s0, e50=e50, w=w, col=c, ls=ls, mk=mk,
                   y=success(grid, s0, e50, w),
                   bd=breakdown(s0, e50, w))
              for (nm, s0, e50, w, c, ls, mk) in DEFENCES]
    ours = curves[-1]
    base = curves[0]
    assert ours["bd"] > base["bd"], "the defence did not move the breakdown"

    shot_y = [float(success(e, ours["s0"], ours["e50"], ours["w"]))
              for e in SHOTS]
    det = [int(round(N_VEH * y)) for y in shot_y]
    assert det[0] == N_VEH and det[-1] < N_VEH, "the tiles show no degradation"
    imgs, veh = tiles()

    ns, ds, q, _ = closed_ratio(ours["bd"], base["bd"])

    fig = plt.figure(figsize=(10.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.60], left=0.062,
                          right=0.988, top=0.828, bottom=0.108, hspace=0.335)
    ax = fig.add_subplot(gs[0])
    gsb = gs[1].subgridspec(1, 4, wspace=0.055)
    axs = [fig.add_subplot(gsb[0, k]) for k in range(4)]

    # ---- (a) success against the attack budget -----------------------------
    style_axes(ax)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_xlim(0, EPS_MAX)
    ax.set_ylim(0, 1.06)
    ax.plot([0, EPS_MAX], [0.5, 0.5], color=MUTED, lw=0.9,
            ls=(0, (4.0, 3.0)), zorder=2)
    ax.text(EPS_MAX - 0.001, 0.505, "breakdown level 0.50", ha="right",
            va="bottom", fontsize=7.0, color=SUB, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                      alpha=0.93))
    for c in curves:
        se = stderr(c["y"])
        ax.fill_between(grid, c["y"] - se, c["y"] + se, color=c["col"],
                        alpha=0.26, linewidth=0.0, zorder=2.4)
        ax.plot(grid, c["y"], color=c["col"], lw=2.1 if c is ours else 1.5,
                ls=c["ls"], zorder=4 if c is ours else 3)
    for i, c in enumerate(curves):
        ax.plot([c["bd"]], [0.5], marker="o", ms=5.4, mfc="white",
                mec=c["col"], mew=1.5, zorder=6)
        ax.plot([c["bd"], c["bd"]], [0.5, 0.14 + 0.075 * i], color=c["col"],
                lw=0.8, ls=(0, (2.0, 2.0)), zorder=5)
        # four decimals because the ratio in the footer is only reproducible
        # from these numbers at four (see ``closed_ratio``)
        ax.text(c["bd"], 0.128 + 0.075 * i, f"{c['name']}   "
                f"$\\epsilon^{{*}}$ = {c['bd']:.4f}", ha="center", va="top",
                fontsize=7.1, color=INK, zorder=7,
                bbox=dict(boxstyle="round,pad=0.24", fc="white",
                          ec=c["col"], lw=0.6, alpha=0.96))
    for k, (e, y) in enumerate(zip(SHOTS, shot_y)):
        ax.plot([e], [y], marker=ours["mk"], ms=5.4, mfc="white", mec=DEEP,
                mew=1.5, zorder=7)
        side = 1 if k < 3 else -1
        ax.text(e + 0.0035 * side, y + 0.030,
                f"({'i' * (k + 1) if k < 3 else 'iv'})",
                ha="left" if side > 0 else "right", va="bottom",
                fontsize=7.6, fontweight="bold", color=DEEP, zorder=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.93))
    ax.set_xlabel("attack budget  $\\epsilon$  (L$_\\infty$ on the observation)",
                  fontsize=8.4)
    ax.set_ylabel("task success rate", fontsize=8.4)
    ax.tick_params(labelsize=7.2, length=2.4)
    ax.set_title("(a)  Success collapses at a budget that depends on the "
                 "defence", fontsize=9.2, fontweight="bold", pad=10.0)
    handles = [Line2D([], [], color=c["col"], lw=2.1 if c is ours else 1.5,
                      ls=c["ls"], label=c["name"]) for c in curves]
    handles.append(Line2D([], [], color=MUTED, lw=0.0, marker="o", ms=5.4,
                          mfc="white", mec=MUTED, mew=1.5,
                          label="$\\epsilon^{*}$, first budget under 0.50"))
    leg = ax.legend(handles=handles, loc="upper right",
                    bbox_to_anchor=(0.998, 0.995), frameon=True,
                    facecolor="white", edgecolor=HAIR, framealpha=0.95,
                    fontsize=7.2, handlelength=2.4, labelspacing=0.34,
                    borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(9)

    # ---- (b) the same four budgets, rendered -------------------------------
    h, w = TILE
    for k, (axk, e, img, y, nd) in enumerate(zip(axs, SHOTS, imgs, shot_y,
                                                 det)):
        axk.imshow(img, extent=[0, w, h, 0], interpolation="bilinear",
                   zorder=1)
        axk.set_xlim(0, w)
        axk.set_ylim(h, 0)
        axk.set_xticks([])
        axk.set_yticks([])
        for sp in axk.spines.values():
            sp.set_edgecolor(DEEP if k == 0 else CARD_EDGE)
            sp.set_linewidth(1.1 if k == 0 else 0.8)
        order = sorted(range(N_VEH), key=lambda i: -veh[i]["rx"])
        for rank, i in enumerate(order):
            # ``rx`` / ``ry`` in the road scene are 1.4x and 2.2x the body
            # half-size, so a tight detection box has to divide them back out
            a = veh[i]
            x0, y0 = a["cx"] - 0.74 * a["rx"], a["cy"] - 1.06 * a["ry"]
            bw, bh = 1.48 * a["rx"], 1.62 * a["ry"]
            found = rank < nd
            axk.add_patch(Rectangle((x0, y0), bw, bh, facecolor="none",
                                    edgecolor="white", lw=2.0, zorder=3,
                                    alpha=0.85))
            axk.add_patch(Rectangle(
                (x0, y0), bw, bh, facecolor="none",
                edgecolor=SOFT if found else MISS, lw=1.2,
                ls="solid" if found else (0, (2.6, 2.0)), zorder=4))
            if not found:
                axk.plot([x0, x0 + bw], [y0, y0 + bh], color=MISS, lw=0.9,
                         zorder=4)
                axk.plot([x0, x0 + bw], [y0 + bh, y0], color=MISS, lw=0.9,
                         zorder=4)
        axk.text(4.0, 4.0, f"({'i' * (k + 1) if k < 3 else 'iv'})  "
                 f"$\\epsilon$ = {e:.3f}", ha="left", va="top", fontsize=7.4,
                 fontweight="bold", color=INK, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec=CARD_EDGE, lw=0.5, alpha=0.94))
        axk.set_xlabel(f"{nd} of {N_VEH} vehicles kept   |   Ours "
                       f"{y:.2f} $\\pm$ {stderr(y):.2f}", fontsize=7.2,
                       color=INK, labelpad=3.0)
    axs[0].text(0.0, 1.055, "(b)  One frame, one adversarial pattern, four "
                "amplitudes - detections shown are round(3 x Ours success)",
                transform=axs[0].transAxes, ha="left", va="bottom",
                fontsize=9.2, fontweight="bold", color=INK)
    key = [Line2D([], [], color=SOFT, lw=1.2, label="kept detection"),
           Line2D([], [], color=MISS, lw=1.2, ls=(0, (2.6, 2.0)),
                  label="lost detection")]
    kleg = axs[3].legend(handles=key, loc="lower right",
                         bbox_to_anchor=(1.0, 1.015), ncol=2, frameon=False,
                         fontsize=7.0, handlelength=1.9, columnspacing=1.3,
                         handletextpad=0.5)
    kleg.set_zorder(9)

    # ---- headline + footer --------------------------------------------------
    savefig(fig, "1204_wm_safe_adversarial.png")


if __name__ == "__main__":
    main()
