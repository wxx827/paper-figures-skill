"""Reward-shaping ablation read left to right: which terms, what curves, what score.

The three blocks are one object.  Each row is a reward configuration; the tick
matrix says which terms are on, the middle panel shows the curves those terms
produce, and the bar on the right is the *endpoint of that same curve* rather
than a separately typed number.  The last row keeps the shaping terms but drops
the task reward, which is how a shaped agent ends up optimising the proxy.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, get_palette, lighten
from figkit import panel_label_inches

TERMS = ["$r_{\\mathrm{task}}$", "$r_{\\mathrm{dist}}$",
         "$r_{\\mathrm{energy}}$", "$r_{\\mathrm{smooth}}$",
         "$r_{\\mathrm{curio}}$"]
TERM_LONG = ["task success", "distance shaping", "energy penalty",
             "action smoothness", "curiosity bonus"]
# term -> (asymptote gain, learning-rate gain, seed-noise factor)
EFFECT = {0: (0.34, 0.30, 1.00), 1: (0.26, 0.85, 0.80), 2: (0.07, 0.05, 0.90),
          3: (0.06, 0.10, 0.75), 4: (0.15, 0.45, 0.85)}
ROWS = [
    ("Sparse task only", (1, 0, 0, 0, 0)),
    ("+ distance", (1, 1, 0, 0, 0)),
    ("+ energy", (1, 1, 1, 0, 0)),
    ("+ smoothness", (1, 1, 1, 1, 0)),
    ("Full shaping (Ours)", (1, 1, 1, 1, 1)),
    ("Shaping without task reward", (0, 1, 1, 1, 1)),
]
NR = len(ROWS)
N_SEED, N_PT = 5, 200
STEPS = np.linspace(0, 4.0, N_PT)          # millions of env steps
OURS = "#3b5b92"
WARN = "#b23a48"
GRID = "#eeeeee"


def curves(rng):
    """One family of seed curves per configuration."""
    ramp = get_palette("cb_sequential_blue", NR + 4)[4:]
    out, cols = [], []
    for r, (name, flags) in enumerate(ROWS):
        asym = sum(EFFECT[i][0] for i, on in enumerate(flags) if on)
        rate = 0.55 + sum(EFFECT[i][1] for i, on in enumerate(flags) if on)
        noise = np.prod([EFFECT[i][2] for i, on in enumerate(flags) if on])
        if not flags[0]:
            # proxy-only agent: it optimises the shaped terms, not the task
            asym *= 0.42
            noise *= 1.5
        runs = np.empty((N_SEED, N_PT))
        for s in range(N_SEED):
            a = asym * rng.normal(1.0, 0.045)
            k = rate * rng.normal(1.0, 0.15)
            mean = a * (1 - np.exp(-k * STEPS))
            e = np.zeros(N_PT)
            for i in range(1, N_PT):
                e[i] = 0.90 * e[i - 1] + rng.normal(0, 1)
            e /= max(np.abs(e).max(), 1e-9)
            runs[s] = mean + e * 0.055 * noise * (1.2 - 0.5 * STEPS / STEPS[-1])
        out.append(runs)
        cols.append(WARN if not flags[0] else
                    (OURS if "Ours" in name else ramp[r]))
    return out, cols


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4400)
    runs, cols = curves(rng)
    final = np.array([np.median(r[:, -8:]) for r in runs])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.0, 0.52],
                          left=0.155, right=0.975, top=0.815, bottom=0.115,
                          wspace=0.09)
    axm = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[0, 2])

    # ---- (a) configuration matrix -----------------------------------------
    axm.set_xlim(-0.75, len(TERMS) - 0.25)
    axm.set_ylim(NR - 0.4, -0.9)
    axm.axis("off")
    for r, (name, flags) in enumerate(ROWS):
        hot = "Ours" in name
        bad = not flags[0]
        axm.add_patch(Rectangle((-4.4, r - 0.40), 4.4 + len(TERMS) - 0.35, 0.80,
                                facecolor="#eef3f8" if hot else
                                ("#fbf1f2" if bad else
                                 ("#fafafa" if r % 2 else "white")),
                                edgecolor="none", zorder=0, clip_on=False))
        axm.text(-0.95, r, name, ha="right", va="center",
                 fontsize=7.8 if (hot or bad) else 7.5,
                 color=OURS if hot else (WARN if bad else "#1a1a1a"),
                 fontweight="bold" if hot else "normal", zorder=3,
                 clip_on=False)
        for c, on in enumerate(flags):
            if on:
                axm.plot([c], [r], marker="o", ms=10.0,
                         mfc=lighten(cols[r], 0.20), mec=cols[r], mew=0.8,
                         zorder=3)
                axm.plot([c - 0.115, c - 0.03, c + 0.135],
                         [r - 0.015, r + 0.10, r - 0.13], color="white",
                         lw=1.35, solid_capstyle="round",
                         solid_joinstyle="miter", zorder=4)
            else:
                axm.plot([c], [r], marker="o", ms=10.0, mfc="white",
                         mec="#dcdcdc", mew=0.8, zorder=3)
                for sgn in (1, -1):
                    axm.plot([c - 0.10, c + 0.10],
                             [r - sgn * 0.09, r + sgn * 0.09], color="#c9c9c9",
                             lw=1.0, zorder=4)
    for c, t in enumerate(TERMS):
        axm.text(c, -0.62, t, ha="center", va="center", fontsize=7.6,
                 fontweight="bold", color="#333333")
    axm.set_title("Reward terms enabled", fontsize=8.6, fontweight="bold",
                  pad=16)

    # ---- (b) the curves those configurations produce -----------------------
    for r, (name, flags) in enumerate(ROWS):
        hot = "Ours" in name
        med = np.median(runs[r], axis=0)
        for s in range(N_SEED):
            axc.plot(STEPS, runs[r][s], color=cols[r], lw=0.6, alpha=0.28,
                     zorder=2)
        if hot:
            q1, q3 = np.percentile(runs[r], [25, 75], axis=0)
            axc.fill_between(STEPS, q1, q3, color=cols[r], alpha=0.20, lw=0,
                             zorder=3)
        axc.plot(STEPS, med, color=cols[r], lw=2.0 if hot else 1.3,
                 zorder=7 if hot else 5, solid_capstyle="round")
        lab = axc.text(STEPS[-1] * 0.985, med[-1] + 0.012, name, ha="right",
                       va="bottom", fontsize=6.6, color=cols[r], zorder=8,
                       fontweight="bold" if hot else "normal")
        lab.set_path_effects([pe.withStroke(linewidth=2.0,
                                            foreground="white")])
    axc.set_xlim(0, STEPS[-1])
    axc.set_ylim(-0.03, max(final) * 1.18)
    axc.set_xlabel("Environment steps ($\\times 10^6$)", fontsize=8)
    axc.set_ylabel("Task return (normalised, 1.0 $=$ expert)", fontsize=8)
    axc.set_title("Return curves, 5 seeds each", fontsize=8.6,
                  fontweight="bold", pad=16)
    axc.grid(color=GRID, lw=0.7)
    axc.set_axisbelow(True)
    axc.tick_params(labelsize=7.0)
    style_axes(axc)

    # ---- (c) endpoint of each curve ---------------------------------------
    axb.set_ylim(NR - 0.4, -0.9)
    base = final[0]
    for r in range(NR):
        hot = "Ours" in ROWS[r][0]
        spread = np.percentile(runs[r][:, -8:], [25, 75])
        axb.barh(r, final[r], height=0.50, color=cols[r],
                 edgecolor="white", lw=0.6, zorder=3)
        axb.plot([spread[0], spread[1]], [r, r], color="#333333", lw=0.9,
                 zorder=5, solid_capstyle="butt")
        axb.text(final[r] + 0.022, r, f"{final[r]:.2f}", ha="left",
                 va="center", fontsize=7.0,
                 color=cols[r] if hot else "#4a4a4a",
                 fontweight="bold" if hot else "normal", zorder=5)
    axb.axvline(base, color="#808080", lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    axb.text(base, -0.62, "sparse\nbaseline", ha="center", va="center",
             fontsize=6.2, color="#777777")
    axb.set_yticks([])
    axb.set_xlim(0, max(final) * 1.30)
    axb.set_xlabel("final return (-)", fontsize=8)
    axb.set_title("Score at 4M steps", fontsize=8.6, fontweight="bold", pad=16)
    axb.grid(axis="x", color=GRID, lw=0.7)
    axb.set_axisbelow(True)
    axb.tick_params(labelsize=7.0)
    style_axes(axb)

    for ax, lab in ((axm, "(a)"), (axc, "(b)"), (axb, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.10, dy=0.32)

    savefig(fig, "440_rl_reward_shaping.png")


if __name__ == "__main__":
    main()
