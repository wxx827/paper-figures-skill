"""Stroop interference and the sequential conflict adaptation that follows it.

Main panel: per-participant response-time distributions for congruent,
neutral and incongruent colour-word trials drawn as half violins over the raw
subject means, with the interference and facilitation contrasts bracketed and
a paired bootstrap interval on each.  The error-rate strip underneath is built
from the same simulated trial table.

Auxiliary (both at 58 % of the main panel width): the Gratton effect, where
the congruency cost on trial n is measured separately after a congruent and
after an incongruent trial; and the delta plot, which asks whether the
interference effect grows or shrinks across the response-time distribution.

Fictional 24-participant experiment, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
TEAL = "#2e7d5b"
RUST = "#c0603f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 1.62, 5.54, 3.86)
STRIP_RECT = (0.86, 0.86, 5.54, 0.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

N_SUBJ = 24
N_TRIALS = 96          # trials per condition per participant
CONDS = [("congruent", 0.0, BLUE), ("neutral", 34.0, FAINT),
         ("incongruent", 92.0, ORANGE)]
BASE_RT = 545.0        # grand mean of the congruent condition, ms
ERR_BASE = 0.021       # congruent error rate
ERR_SLOPE = 0.00048    # extra error probability per ms of conflict cost
PREV_COST = 44.0       # conflict cost removed after an incongruent trial, ms


def ex_gaussian(rng, mu, sigma, tau, size):
    """Response times as a Gaussian plus an exponential tail."""
    return rng.normal(mu, sigma, size) + rng.exponential(tau, size)


def paired_ci(rng, diff, n_boot=4000):
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(956)

    subj_offset = rng.normal(0.0, 46.0, N_SUBJ)
    subj_gain = rng.normal(1.0, 0.28, N_SUBJ)          # conflict sensitivity

    rt_trials, err_trials = {}, {}
    for name, cost, _ in CONDS:
        rows_rt, rows_err = [], []
        for s in range(N_SUBJ):
            c = cost * subj_gain[s]
            mu = BASE_RT + subj_offset[s] + 0.62 * c
            r = ex_gaussian(rng, mu, 42.0, 78.0 + 0.38 * c, N_TRIALS)
            p_err = np.clip(ERR_BASE + ERR_SLOPE * c, 0.004, 0.35)
            rows_rt.append(r)
            rows_err.append(rng.random(N_TRIALS) < p_err)
        rt_trials[name] = np.array(rows_rt)
        err_trials[name] = np.array(rows_err)

    subj_mean = {k: v.mean(axis=1) for k, v in rt_trials.items()}
    subj_err = {k: 100.0 * v.mean(axis=1) for k, v in err_trials.items()}

    d_interf = subj_mean["incongruent"] - subj_mean["congruent"]
    d_facil = subj_mean["neutral"] - subj_mean["congruent"]
    ci_interf = paired_ci(rng, d_interf)
    ci_facil = paired_ci(rng, d_facil)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_s = axes_inches(fig, STRIP_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_s, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: distributions and subject means -----------------------------
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    pos = np.arange(len(CONDS))
    for i, (name, _, col) in enumerate(CONDS):
        pooled = rt_trials[name].ravel()
        parts = ax.violinplot([pooled], positions=[i], widths=0.78,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.clip(verts[:, 0], i, i + 0.42)
            body.set_facecolor(lighten(col, 0.62))
            body.set_edgecolor(col)
            body.set_linewidth(0.8)
            body.set_alpha(1.0)
            body.set_zorder(2)
        jit = (rng.random(N_SUBJ) - 0.5) * 0.30
        ax.plot(i - 0.20 + jit, subj_mean[name], marker="o", ls="none",
                ms=3.3, mfc="white", mec=col, mew=0.8, alpha=0.95, zorder=4)
        m = float(subj_mean[name].mean())
        sem = float(subj_mean[name].std(ddof=1) / np.sqrt(N_SUBJ))
        ax.plot([i - 0.40, i - 0.02], [m, m], color=col, lw=2.0, zorder=6)
        ax.plot([i - 0.21, i - 0.21], [m - 1.96 * sem, m + 1.96 * sem],
                color=col, lw=1.2, zorder=6)
        ax.text(i - 0.43, m, f"{m:.0f} ms", ha="right", va="center",
                fontsize=6.8, color=col)

    for s in range(N_SUBJ):
        ax.plot(pos - 0.20, [subj_mean[n][s] for n, _, _ in CONDS],
                color="#c9d0d8", lw=0.4, zorder=1)

    y_top = 830.0
    for (a_i, b_i), diff, ci, lab, col, y in (
            ((0, 2), d_interf, ci_interf, "interference", ACCENT, y_top),
            ((0, 1), d_facil, ci_facil, "conflict-free cost", SUB,
             y_top - 62.0)):
        ax.plot([a_i - 0.21, a_i - 0.21, b_i - 0.21, b_i - 0.21],
                [y - 14, y, y, y - 14], color=col, lw=0.9, zorder=6)
        ax.text((a_i + b_i) / 2.0 - 0.21, y + 5.0,
                f"{lab} = {diff.mean():+.0f} ms "
                f"[{ci[0]:+.0f}, {ci[1]:+.0f}] 95 % CI",
                ha="center", va="bottom", fontsize=6.9, color=col)

    ax.set_xticks(pos)
    ax.set_xticklabels([])
    ax.set_xlim(-0.62, len(CONDS) - 0.30)
    ax.set_ylim(400.0, 900.0)
    ax.set_yticks(np.arange(450, 851, 100))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("response time (ms)", fontsize=8.5)
    ax.set_title("Naming the ink colour is slower when the word disagrees",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.text(-0.55, 412.0,
            f"n={N_SUBJ} participants, {N_TRIALS} trials per condition per "
            "participant", ha="left", va="bottom", fontsize=6.6, color=MUTED)
    handles = [
        Line2D([], [], color=FAINT, lw=0, marker="o", ms=3.6, mfc="white",
               mec=SUB, mew=0.8, label="participant mean"),
        Line2D([], [], color=SUB, lw=2.0, label="condition mean, 95 % CI"),
        Line2D([], [], color=lighten(SUB, 0.55), lw=5,
               label="pooled trial distribution"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.8,
              handlelength=1.5, labelspacing=0.30, borderpad=0.12,
              bbox_to_anchor=(1.006, -0.010))

    # ---- error-rate strip on the same x axis --------------------------------
    for i, (name, _, col) in enumerate(CONDS):
        e = float(subj_err[name].mean())
        sem = float(subj_err[name].std(ddof=1) / np.sqrt(N_SUBJ))
        ax_s.barh([i], [e], height=0.55, color=lighten(col, 0.35),
                  edgecolor=col, linewidth=0.7, zorder=3)
        ax_s.plot([e - 1.96 * sem, e + 1.96 * sem], [i, i], color=col,
                  lw=1.0, zorder=5)
        ax_s.text(e + 1.96 * sem + 0.22, i, f"{e:.1f} %", ha="left",
                  va="center", fontsize=6.6, color=col)
    ax_s.set_yticks(pos)
    ax_s.set_yticklabels([f"{n} trials" for n, _, _ in CONDS], fontsize=7.4)
    ax_s.set_ylim(-0.62, len(CONDS) - 0.30)
    ax_s.set_xlim(0.0, 9.5)
    ax_s.invert_yaxis()
    ax_s.grid(axis="x", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_s.tick_params(axis="x", labelsize=6.8)
    ax_s.set_xlabel("error rate (%)", fontsize=8.0, labelpad=1.5)

    # ---- aux (b): the Gratton effect ---------------------------------------
    prev = ["after congruent", "after incongruent"]
    cost_prev = np.zeros((len(prev), N_SUBJ))
    for j, _ in enumerate(prev):
        relief = 0.0 if j == 0 else PREV_COST
        for s in range(N_SUBJ):
            c_inc = max(92.0 * subj_gain[s] - relief * subj_gain[s], 0.0)
            mu_i = BASE_RT + subj_offset[s] + 0.62 * c_inc
            mu_c = BASE_RT + subj_offset[s] + (12.0 if j == 1 else 0.0)
            ri = ex_gaussian(rng, mu_i, 42.0, 78.0 + 0.38 * c_inc, 48)
            rc = ex_gaussian(rng, mu_c, 42.0, 78.0, 48)
            cost_prev[j, s] = ri.mean() - rc.mean()

    ax_t.grid(axis="y", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_t.axhline(0.0, color=FAINT, lw=0.8, zorder=1)
    for j in range(len(prev)):
        col = ORANGE if j == 0 else TEAL
        jit = (rng.random(N_SUBJ) - 0.5) * 0.26
        ax_t.plot(j + jit, cost_prev[j], marker="o", ls="none", ms=2.8,
                  mfc="white", mec=col, mew=0.7, alpha=0.9, zorder=4)
        m = float(cost_prev[j].mean())
        sem = float(cost_prev[j].std(ddof=1) / np.sqrt(N_SUBJ))
        ax_t.plot([j - 0.24, j + 0.24], [m, m], color=col, lw=2.0, zorder=6)
        ax_t.plot([j, j], [m - 1.96 * sem, m + 1.96 * sem], color=col,
                  lw=1.1, zorder=6)
        ax_t.text(j, m + 1.96 * sem + 5.0, f"{m:.0f} ms", ha="center",
                  va="bottom", fontsize=6.6, color=col)
    grat = float(cost_prev[0].mean() - cost_prev[1].mean())
    ax_t.annotate("", xy=(1.0, float(cost_prev[1].mean())),
                  xytext=(0.0, float(cost_prev[0].mean())),
                  arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.1,
                                  ls=(0, (3.0, 1.8)), shrinkA=4, shrinkB=4,
                                  mutation_scale=9))
    ax_t.text(0.5, 12.0, f"Gratton effect\n{grat:.0f} ms", ha="center",
              va="bottom", fontsize=6.5, color=ACCENT, linespacing=1.35)
    ax_t.set_xticks(range(len(prev)))
    ax_t.set_xticklabels(prev, fontsize=6.8)
    ax_t.set_xlim(-0.58, len(prev) - 0.42)
    ax_t.set_ylim(-25.0, 132.0)
    ax_t.tick_params(axis="y", labelsize=6.6)
    ax_t.set_ylabel("congruency cost (ms)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Conflict on the previous trial shrinks the cost",
                   fontsize=8.4, fontweight="bold", pad=4.0)

    # ---- aux (c): the delta plot --------------------------------------------
    q_levels = np.array([10, 30, 50, 70, 90])
    q_con = np.array([np.percentile(rt_trials["congruent"][s], q_levels)
                      for s in range(N_SUBJ)]).mean(axis=0)
    q_inc = np.array([np.percentile(rt_trials["incongruent"][s], q_levels)
                      for s in range(N_SUBJ)]).mean(axis=0)
    delta = q_inc - q_con
    mid = 0.5 * (q_inc + q_con)
    slope = float(np.polyfit(mid, delta, 1)[0])

    ax_b.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_b.plot(mid, delta, marker="o", ms=4.6, color=ACCENT, lw=1.5,
              mfc="white", mec=ACCENT, mew=1.1, zorder=5)
    fitx = np.linspace(mid.min() - 12, mid.max() + 12, 40)
    ax_b.plot(fitx, np.polyval(np.polyfit(mid, delta, 1), fitx), color=FAINT,
              lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    for x, y, q in zip(mid, delta, q_levels):
        ax_b.text(x, y + 4.0, f"{q}", ha="center", va="bottom", fontsize=6.2,
                  color=SUB)
    ax_b.text(mid[0] - 6.0, delta.max() + 12.0,
              f"delta slope = {slope:+.3f} ms per ms\n"
              f"quantile labels are % of the RT distribution",
              ha="left", va="top", fontsize=6.3, color=SUB, linespacing=1.45)
    ax_b.set_xlim(mid.min() - 34, mid.max() + 34)
    ax_b.set_ylim(delta.min() - 22, delta.max() + 44)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("mean of the two conditions at that quantile (ms)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("interference (ms)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Interference grows in the slow tail", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "956_cog_stroop_conflict.png")


if __name__ == "__main__":
    main()
