"""Bayesian knowledge tracing for one learner across a 20-step practice run.

A single BKT simulation drives all three panels: the mastery posterior after
every opportunity (main heat map), the state the learner ends in (bar panel),
and the prerequisite graph the transfer term follows (thumbnail).  The
correct / incorrect glyphs are drawn as line segments rather than font marks
so their on-paper size stays square even though the heat-map cells are not,
and the bars and graph nodes reuse the heat map's colour scale so the three
panels cannot disagree about what "mastered" means.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import arrow, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
ACCENT = "#b23a48"

SKILLS = ["KC1 place value", "KC2 add / sub", "KC3 multiply",
          "KC4 fractions", "KC5 decimals", "KC6 ratio", "KC7 percent"]
SHORT = ["KC1", "KC2", "KC3", "KC4", "KC5", "KC6", "KC7"]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
NODE_XY = [(0.0, 1.00), (1.05, 1.62), (1.05, 0.38), (2.10, 1.62),
           (2.10, 0.38), (3.15, 1.00), (4.20, 1.00)]
SEQUENCE = [0, 0, 1, 2, 0, 1, 3, 2, 4, 1, 3, 5, 4, 2, 5, 6, 3, 5, 6, 6]

P_INIT = [0.35, 0.28, 0.25, 0.18, 0.16, 0.12, 0.10]
P_TRANSIT = 0.34
P_SLIP = 0.10
P_GUESS = 0.22
TRANSFER = 0.25
MASTERY_CUT = 0.90


def trace(seed=3):
    """Run BKT; returns the posterior history and the response record."""
    rng = np.random.default_rng(seed)
    n_s, n_t = len(SKILLS), len(SEQUENCE)
    p = np.array(P_INIT, dtype=float)
    hist = np.zeros((n_s, n_t))
    responses = np.zeros(n_t, dtype=bool)
    for t, s in enumerate(SEQUENCE):
        prior = p[s]
        p_corr = prior * (1.0 - P_SLIP) + (1.0 - prior) * P_GUESS
        ok = bool(rng.random() < p_corr)
        if ok:
            post = prior * (1.0 - P_SLIP) / p_corr
        else:
            post = prior * P_SLIP / (1.0 - p_corr)
        new = post + (1.0 - post) * P_TRANSIT
        gain = new - prior
        p[s] = new
        if gain > 0.0:                       # partial credit to dependants
            for a, b in EDGES:
                if a == s:
                    p[b] = min(p[b] + TRANSFER * gain, 0.995)
        hist[:, t] = p
        responses[t] = ok
    return hist, responses


def glyph(ax, x, y, ok, gw, gh):
    """Check / cross drawn from line segments, sized square on paper."""
    col = "#f2f7fb" if ok else ACCENT
    halo = "#1a1a1a" if ok else "white"
    if ok:
        px = [-0.85, -0.25, 0.85]
        py = [0.00, 0.66, -0.78]
        segs = [(px, py)]
    else:
        segs = [([-0.68, 0.68], [-0.68, 0.68]), ([-0.68, 0.68], [0.68, -0.68])]
    for sx, sy in segs:
        ln, = ax.plot([x + v * gw for v in sx], [y + v * gh for v in sy],
                      color=col, lw=1.35, solid_capstyle="round",
                      solid_joinstyle="round", zorder=7)
        ln.set_path_effects([pe.withStroke(linewidth=2.5, foreground=halo)])


def main():
    set_pub_style()
    hist, responses = trace()
    n_s, n_t = hist.shape
    cmap = get_cmap("cb_sequential_blue")

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.40,
                          hspace=0.46)
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_dag = fig.add_subplot(gs[1, 1])
    # Row-spanning panels inside a gridspec with unequal height_ratios make
    # tight_layout refuse to run, so margins are set explicitly - and set
    # first, because the glyph aspect correction below reads the panel box.
    fig.subplots_adjust(left=0.105, right=0.955, bottom=0.150, top=0.885)

    # ---- main: posterior mastery per skill x opportunity -----------------
    im = ax_heat.imshow(hist, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
                        interpolation="nearest",
                        extent=(-0.5, n_t - 0.5, n_s - 0.5, -0.5))
    # Blank strip above the image for the glyph key; the extent above keeps
    # the rows themselves on integer centres.
    ax_heat.set_ylim(n_s - 0.5, -1.85)
    ax_heat.set_yticks(np.arange(n_s))
    ax_heat.set_yticklabels(SKILLS, fontsize=6.8)
    ax_heat.set_xticks(np.arange(0, n_t, 2))
    ax_heat.set_xticklabels(np.arange(1, n_t + 1, 2), fontsize=7.0)
    ax_heat.set_xlabel("practice opportunity $t$ (steps)", fontsize=8.0)
    ax_heat.set_ylabel("knowledge component", fontsize=8.0)
    ax_heat.tick_params(length=2.2)
    for s in ax_heat.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")

    # Cell paper aspect -> glyph half-height, so checks stay square.
    bb = ax_heat.get_position()
    fw, fh = fig.get_size_inches()
    col_in = bb.width * fw / n_t
    row_in = bb.height * fh / (n_s + 1.35)
    gw = 0.30
    gh = gw * col_in / row_in

    for t, s in enumerate(SEQUENCE):
        glyph(ax_heat, t, s, bool(responses[t]), gw, gh)

    ax_heat.axhline(-0.5, color="#8a8a8a", lw=0.6)
    for x0, ok, lab in [(0.1, True, "correct response"),
                        (6.1, False, "incorrect response")]:
        glyph(ax_heat, x0, -1.15, ok, gw, gh)
        ax_heat.text(x0 + 0.55, -1.15, lab, ha="left", va="center",
                     fontsize=6.8, color=SUB)
    ax_heat.text(n_t - 0.7, -1.15,
                 f"observed accuracy {responses.mean() * 100:.0f} %",
                 ha="right", va="center", fontsize=6.8, color=MUTED)
    ax_heat.set_title("Posterior mastery updates only where the learner "
                      "practises", fontsize=9.0, fontweight="bold", color=INK)
    safe_colorbar(fig, im, ax_heat, label=r"$P(\mathrm{mastery})$ (-)",
                  width=0.10, pad=0.07, labelsize=6.8)

    # ---- aux 1: end state -------------------------------------------------
    final = hist[:, -1]
    ypos = np.arange(n_s)[::-1]
    ax_bar.barh(ypos, final, height=0.66, color=[cmap(v) for v in final],
                edgecolor="white", linewidth=0.6)
    ax_bar.axvline(MASTERY_CUT, color=GREY, lw=0.9, ls=(0, (3.4, 2.0)))
    ax_bar.text(MASTERY_CUT, -0.95, f"mastery cut {MASTERY_CUT:.2f}",
                ha="center", va="center", fontsize=6.3, color=SUB)
    for y, v in zip(ypos, final):
        ax_bar.text(v + 0.02, y, f"{v:.2f}", ha="left", va="center",
                    fontsize=6.3, color=SUB)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(SHORT, fontsize=6.6)
    ax_bar.set_ylim(-1.35, n_s - 0.4)
    ax_bar.set_xlim(0.0, 1.18)
    ax_bar.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.tick_params(axis="x", labelsize=6.6)
    ax_bar.set_xlabel(r"final $P(\mathrm{mastery})$ (-)", fontsize=7.4)
    ax_bar.grid(axis="x", lw=0.5, color="#e4e4e4")
    ax_bar.set_title("State after 20 opportunities", fontsize=8.5,
                     fontweight="bold", color=INK)

    # ---- aux 2: prerequisite graph ---------------------------------------
    for a, b in EDGES:
        arrow(ax_dag, NODE_XY[a], NODE_XY[b], color="#b3b3b3", lw=0.8,
              ms=7.0, gap=0.30, zorder=2)
    for (x, y), v, lab in zip(NODE_XY, final, SHORT):
        ax_dag.plot([x], [y], marker="o", ms=13.0, mfc=cmap(v),
                    mec="white", mew=1.0, zorder=4)
        ax_dag.text(x, y - 0.40, lab, ha="center", va="top", fontsize=6.4,
                    color=SUB, zorder=5)
    ax_dag.set_xlim(-0.45, 4.65)
    ax_dag.set_ylim(-0.25, 2.15)
    ax_dag.set_aspect("equal")
    ax_dag.axis("off")
    ax_dag.set_title("Prerequisite graph", fontsize=8.5, fontweight="bold",
                     color=INK)


    note = (f"Two-parameter BKT with $P(L_0)$ = {min(P_INIT):.2f}-"
            f"{max(P_INIT):.2f} per skill, $P(T)$ = {P_TRANSIT:.2f}, "
            f"$P(\\mathrm{{slip}})$ = {P_SLIP:.2f}, "
            f"$P(\\mathrm{{guess}})$ = {P_GUESS:.2f}; a mastery gain also "
            f"passes {TRANSFER * 100:.0f} % of itself to each dependant along "
            f"the prerequisite graph. One simulated learner, {n_t} "
            f"opportunities, {int(responses.sum())} of {n_t} correct; "
            f"{int((final >= MASTERY_CUT).sum())} of {n_s} components end "
            f"above the {MASTERY_CUT:.2f} cut.")

    panel_labels_inches(fig, [ax_heat, ax_bar, ax_dag],
                        labels=["(a)", "(b)", "(c)"], dx=-0.44, dy=0.10)
    savefig(fig, "595_edu_knowledge_tracing.png")


if __name__ == "__main__":
    main()
