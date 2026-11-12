"""Descriptive sensory profile of three formulations, over time and preference.

The main panel is the quantitative descriptive analysis radar: six taste and
mouthfeel attributes scored on the same 0-10 line scale, three formulations
laid over one another, and the reference standard drawn underneath as a pale
grey polygon so that every deviation is read against a common baseline rather
than against the frame.

The subordinate panels answer the two questions a radar cannot: when the
sensation arrives and how long it lasts (time-intensity traces on one shared
intensity axis), and whether the differences the panel resolves are the ones
consumers care about (a preference map with the liking vectors of four
consumer segments projected into the same space).  Scores, maxima, areas and
loadings are all computed from the generated panel data.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"
HAIR = "#cccccc"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

ATTRS = ["sweet", "sour", "bitter", "salty", "umami", "astringent"]
FOCUS = ["formulation A", "formulation B", "formulation C"]
FOCUS_COL = dict(zip(FOCUS, [RED, BLUE, GREEN]))
SEGMENTS = ["savoury seekers", "sweet tooth", "clean finish",
            "bold flavour"]
R_MAX = 10.0


def sensory_panel(seed=23):
    """Nine formulations scored on six attributes, plus a reference standard."""
    rng = np.random.default_rng(seed)
    ref = np.array([5.0, 4.2, 3.4, 4.6, 4.0, 3.0])
    design = np.array([
        [7.6, 3.1, 2.2, 3.9, 3.4, 2.4],       # A - sweet led
        [4.1, 6.4, 4.9, 5.2, 3.1, 5.6],       # B - sour and astringent
        [4.6, 3.6, 3.0, 6.5, 7.8, 3.2],       # C - salty and umami
        [6.4, 4.4, 2.6, 4.2, 4.6, 2.9],
        [3.4, 5.6, 6.2, 4.8, 3.6, 6.1],
        [5.2, 4.0, 3.6, 5.9, 6.4, 3.4],
        [6.9, 5.1, 4.1, 3.4, 3.0, 4.2],
        [4.0, 3.2, 5.4, 6.1, 5.2, 4.6],
        [5.6, 6.0, 3.2, 4.4, 4.8, 3.8]])
    scores = design + rng.normal(0.0, 0.16, design.shape)
    return np.clip(scores, 0.0, R_MAX), ref


def ti_curves(peak, seed=31):
    t = np.linspace(0.0, 60.0, 400)
    rng = np.random.default_rng(seed)
    par = {"formulation A": (1.9, 9.0, 8.5),
           "formulation B": (3.4, 16.0, 21.0),
           "formulation C": (2.6, 13.0, 14.0)}
    out = {}
    knots = np.linspace(0.0, 60.0, 16)
    for name, (tr, tp, td) in par.items():
        i = (1.0 - np.exp(-t / tr)) * np.exp(-np.clip(t - tp, 0.0, None) / td)
        i = peak[name] * i / i.max()
        i = i + 0.06 * np.interp(t, knots, rng.normal(size=knots.size))
        out[name] = np.clip(i, 0.0, None)
    return t, out, par


def preference(scores, seed=41):
    rng = np.random.default_rng(seed)
    xs = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    pc = u * s
    explained = s ** 2 / np.sum(s ** 2)
    weights = np.array([
        [-0.2, 0.1, -0.3, 0.7, 0.9, -0.2],    # savoury seekers
        [0.9, -0.4, -0.6, 0.0, 0.1, -0.3],    # sweet tooth
        [0.2, 0.2, -0.8, 0.1, 0.2, -0.9],     # clean finish
        [0.1, 0.7, 0.5, 0.3, 0.4, 0.6]])      # bold flavour
    arrows = []
    for w in weights:
        liking = xs @ w + rng.normal(0.0, 0.25, xs.shape[0])
        arrows.append([np.corrcoef(liking, pc[:, k])[0, 1] for k in (0, 1)])
    return pc, np.array(arrows), explained


def radar_panel(ax, scores, ref):
    ang = np.linspace(0.0, 2.0 * np.pi, len(ATTRS), endpoint=False)
    close = np.append(ang, ang[0])

    r = np.append(ref, ref[0])
    ax.fill(close, r, color="#d5d5d5", alpha=0.55, zorder=2, linewidth=0.0)
    ax.plot(close, r, color=GREY, lw=1.0, ls=(0, (3.2, 2.2)), zorder=3)

    for i, name in enumerate(FOCUS):
        v = np.append(scores[i], scores[i][0])
        col = FOCUS_COL[name]
        ax.fill(close, v, color=col, alpha=0.07, zorder=4, linewidth=0.0)
        ax.plot(close, v, color=col, lw=1.7, marker="o", ms=3.6, mfc="white",
                mew=1.0, zorder=5, label=name)

    dev = scores[:3] - ref[None, :]
    i_s, i_a = np.unravel_index(np.argmax(np.abs(dev)), dev.shape)
    ax.annotate(f"{dev[i_s, i_a]:+.1f} vs reference",
                xy=(ang[i_a], scores[i_s, i_a]),
                xytext=(ang[i_a] + 0.30, R_MAX * 1.02),
                fontsize=6.8, color=FOCUS_COL[FOCUS[i_s]],
                arrowprops=dict(arrowstyle="-", lw=0.7,
                                color=FOCUS_COL[FOCUS[i_s]]))

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ang), ATTRS)
    ax.tick_params(axis="x", pad=2.0, labelsize=8.0)
    ax.set_rlim(0.0, R_MAX)
    ax.set_rgrids([2.0, 4.0, 6.0, 8.0, 10.0],
                  labels=["2", "4", "6", "8", "10"], angle=16.0,
                  fontsize=6.5, color=SUB)
    ax.set_rlabel_position(16.0)
    ax.grid(color=HAIR, lw=0.6)
    ax.spines["polar"].set_color(HAIR)
    ax.set_title("Intensity on a 0-10 line scale, reference standard in grey",
                 fontsize=9.0, fontweight="bold", color=INK, pad=10.0)
    ax.legend(fontsize=7.0, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.14), ncol=3, handlelength=1.5,
              columnspacing=1.4)
    return dev


def ti_panel(ax, t, curves, par):
    stats = {}
    for name in FOCUS:
        y = curves[name]
        col = FOCUS_COL[name]
        ax.plot(t, y, color=col, lw=1.5)
        k = int(np.argmax(y))
        stats[name] = (float(y[k]), float(t[k]), float(np.trapezoid(y, t)))
        ax.plot([t[k]], [y[k]], marker="o", ms=3.0, color=col, mfc="white",
                mew=1.0)
    tr, tp, td = par["formulation B"]
    for x, lab in [(tr, "onset"), (tp, "plateau"), (tp + td, "decay")]:
        ax.axvline(x, color=HAIR, lw=0.6, ls=(0, (2.0, 2.0)), zorder=1)
        ax.text(x + 0.8, 9.4, lab, ha="left", va="top", fontsize=6.2,
                color=MUTED)

    top = max(stats, key=lambda k: stats[k][2])
    ax.text(0.97, 0.60,
            f"{top.split()[-1]} carries the longest\ncarry-over: "
            f"{stats[top][2]:.0f} unit s", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color=SUB, linespacing=1.3)
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel("time after ingestion (s)", fontsize=7.4)
    ax.set_ylabel("perceived intensity (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Time-intensity traces", fontsize=8.2, fontweight="bold",
                 color=INK)
    return stats


def pref_panel(ax, pc, arrows, explained):
    ax.axhline(0.0, color=GREY, lw=0.6, ls=(0, (1.6, 2.0)), zorder=1)
    ax.axvline(0.0, color=GREY, lw=0.6, ls=(0, (1.6, 2.0)), zorder=1)
    ax.plot(pc[3:, 0], pc[3:, 1], marker="o", ms=3.4, ls="none",
            color="#9a9a9a", mec="white", mew=0.5, zorder=4)
    for i, name in enumerate(FOCUS):
        col = FOCUS_COL[name]
        ax.plot([pc[i, 0]], [pc[i, 1]], marker="o", ms=5.6, color=col,
                mec="white", mew=0.8, zorder=6)
        ax.text(pc[i, 0], pc[i, 1] - 0.24, name.split()[-1], ha="center",
                va="top", fontsize=6.6, color=col, zorder=7)

    lim = 1.62 * np.abs(pc[:, :2]).max()
    k = 0.50 * lim / np.abs(arrows).max()
    for (dx, dy), name in zip(arrows * k, SEGMENTS):
        ax.annotate("", xy=(dx, dy), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="#3a3a3a",
                                    lw=0.9, shrinkA=0.0, shrinkB=0.0),
                    zorder=5)
        ax.text(dx * 1.08, dy * 1.08, name, ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top", fontsize=6.3,
                color="#3a3a3a", zorder=7,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.78))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.86, lim * 0.86)
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.0f} %) (-)", fontsize=7.4)
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.0f} %) (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Preference map with segment liking vectors", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    scores, ref = sensory_panel()
    peak = {n: float(scores[i].max()) for i, n in enumerate(FOCUS)}
    t, curves, par = ti_curves(peak)
    pc, arrows, explained = preference(scores)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.72)
    ax_main = fig.add_subplot(gs[:, 0], projection="polar")
    ax_ti = fig.add_subplot(gs[0, 1])
    ax_pm = fig.add_subplot(gs[1, 1])

    dev = radar_panel(ax_main, scores, ref)
    stats = ti_panel(ax_ti, t, curves, par)
    pref_panel(ax_pm, pc, arrows, explained)

    spread = float(np.abs(dev).max())
    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_ti, ax_pm], dx=-0.46, dy=0.12)
    savefig(fig, "730_sen_taste_profile.png")


if __name__ == "__main__":
    main()
