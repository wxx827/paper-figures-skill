"""Drift diffusion model of a two-choice decision, simulated path by path.

Main panel: sample evidence trajectories between an upper (correct) and a
lower (error) absorbing boundary, with the non-decision period shaded before
the accumulation starts.  The defective first-passage-time distributions are
drawn back to back above and below the boundaries, so their relative areas are
the accuracy of the run; every printed mean, accuracy and quantile is computed
from the same first-passage array.

Auxiliary (both at 58 % of the main panel width): accuracy and mean response
time across four drift rates, which is the speed-accuracy signature of the
model; and an EZ-diffusion recovery check that puts the drift rate estimated
from accuracy and response-time moments against the value that generated the
trials.

Fictional two-choice task, illustrative parameters only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
TEAL = "#2e7d5b"
RUST = "#c0603f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

BOUND = 1.00          # boundary separation a, in evidence units
START = 0.50          # relative start point z / a (unbiased)
TER_S = 0.260         # non-decision time, seconds
DRIFT = 1.80          # drift rate of the main condition, evidence units / s
DT = 0.001            # integration step, seconds
N_TRIALS = 5000
DRIFT_SET = [0.80, 1.40, 2.00, 2.60]


def simulate(rng, drift, n, a=BOUND, z=START, ter=TER_S, t_max=3.2):
    """Euler-Maruyama first passage; returns response time (s) and choice."""
    x = np.full(n, z * a)
    rt = np.full(n, np.nan)
    upper = np.zeros(n, dtype=bool)
    live = np.ones(n, dtype=bool)
    root = np.sqrt(DT)
    for step in range(int(t_max / DT)):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        x[idx] += drift * DT + root * rng.standard_normal(idx.size)
        t = (step + 1) * DT + ter
        hi = idx[x[idx] >= a]
        lo = idx[x[idx] <= 0.0]
        rt[hi] = t
        upper[hi] = True
        rt[lo] = t
        live[hi] = False
        live[lo] = False
    ok = np.isfinite(rt)
    return rt[ok], upper[ok]


def sample_paths(rng, drift, n, a=BOUND, z=START, t_max=2.0):
    """A handful of full trajectories, for drawing."""
    out = []
    root = np.sqrt(DT)
    for _ in range(n):
        x, path = z * a, [z * a]
        for _ in range(int(t_max / DT)):
            x += drift * DT + root * rng.standard_normal()
            path.append(x)
            if x >= a or x <= 0.0:
                break
        out.append(np.array(path))
    return out


def ez_diffusion(pc, mrt, vrt, s=1.0):
    """Wagenmakers EZ estimates of drift, boundary and non-decision time."""
    pc = float(np.clip(pc, 1e-4, 1.0 - 1e-4))
    logit = np.log(pc / (1.0 - pc))
    num = logit * (logit * pc ** 2 - logit * pc + pc - 0.5)
    v = np.sign(pc - 0.5) * s * (num / vrt) ** 0.25
    a = s ** 2 * logit / v
    y = -v * a / s ** 2
    mdt = (a / (2.0 * v)) * (1.0 - np.exp(y)) / (1.0 + np.exp(y))
    return float(v), float(a), float(mrt - mdt)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(955)

    rt, up = simulate(rng, DRIFT, N_TRIALS)
    rt_ms = rt * 1000.0
    acc = float(up.mean())
    rt_c, rt_e = rt_ms[up], rt_ms[~up]
    q_levels = [10, 30, 50, 70, 90]
    q_c = np.percentile(rt_c, q_levels)
    q_e = np.percentile(rt_e, q_levels)

    paths = sample_paths(rng, DRIFT, 14)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: trajectories between two boundaries -------------------------
    t_lo, t_hi = 0.0, 1250.0
    ax.axvspan(t_lo, TER_S * 1000.0, facecolor="#eef1f4", edgecolor="none",
               zorder=0)
    ax.text(TER_S * 1000.0 / 2.0, 1.60, "non-decision\ntime Ter",
            ha="center", va="top", fontsize=6.5, color=MUTED, linespacing=1.3)

    ax.axhline(BOUND, color=TEAL, lw=1.6, zorder=6)
    ax.axhline(0.0, color=RUST, lw=1.6, zorder=6)
    ax.axhline(START * BOUND, color=FAINT, lw=0.9, ls=(0, (3.0, 2.4)),
               zorder=3)

    for p in paths:
        tt = TER_S * 1000.0 + np.arange(p.size) * DT * 1000.0
        hit_up = p[-1] >= BOUND
        col = TEAL if hit_up else RUST
        ax.plot(tt, np.clip(p, -0.02, BOUND + 0.02), color=lighten(col, 0.52),
                lw=0.6, zorder=4, solid_joinstyle="round")

    # mean drift path, drawn from the same generating rate and stopped at a
    t_end = TER_S * 1000.0 + 1000.0 * (1.0 - START) * BOUND / DRIFT
    t_drift = np.linspace(TER_S * 1000.0, t_end, 80)
    ax.plot(t_drift, START * BOUND + DRIFT * (t_drift - TER_S * 1000.0) / 1000.0,
            color=INK, lw=1.7, zorder=7)
    ax.annotate("", xy=(t_end, BOUND),
                xytext=(t_end - 70, BOUND - 70 * DRIFT / 1000.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.7,
                                shrinkA=0, shrinkB=0, mutation_scale=10),
                zorder=7)
    lab = ax.text(292, 0.585, f"v = {DRIFT:.2f} evidence units / s",
                  rotation=36.0, rotation_mode="anchor", ha="left",
                  va="bottom", fontsize=6.8, color=INK, zorder=8)
    lab.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    # back-to-back defective first-passage distributions
    edges = np.arange(200.0, 1260.0, 25.0)
    h_c, _ = np.histogram(rt_c, bins=edges)
    h_e, _ = np.histogram(rt_e, bins=edges)
    scale = 0.60 / max(h_c.max(), 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(ctr, h_c * scale, width=23.0, bottom=BOUND + 0.03,
           color=lighten(TEAL, 0.30), edgecolor="none", zorder=5)
    ax.bar(ctr, -h_e * scale, width=23.0, bottom=-0.03,
           color=lighten(RUST, 0.30), edgecolor="none", zorder=5)
    top = BOUND + 0.03 + h_c.max() * scale
    ax.plot(q_c, np.full(len(q_c), top + 0.035), marker="v", ls="none",
            ms=3.4, mfc=TEAL, mec="white", mew=0.6, zorder=8)
    ax.text(q_c[2], top + 0.075, "10 / 30 / 50 / 70 / 90 % quantiles",
            ha="center", va="bottom", fontsize=6.2, color=SUB)

    ax.text(1245, BOUND + 0.42,
            f"correct  {100 * acc:.1f} % of trials\n"
            f"mean RT {rt_c.mean():.0f} ms, median {q_c[2]:.0f} ms",
            ha="right", va="center", fontsize=6.9, color=TEAL,
            linespacing=1.5)
    ax.text(1245, -0.44,
            f"error  {100 * (1 - acc):.1f} % of trials\n"
            f"mean RT {rt_e.mean():.0f} ms, median {q_e[2]:.0f} ms",
            ha="right", va="center", fontsize=6.9, color=RUST,
            linespacing=1.5)

    ax.text(1245, BOUND + 0.035, "a = correct boundary", ha="right",
            va="bottom", fontsize=6.8, color=TEAL)
    ax.text(1245, -0.035, "0 = error boundary", ha="right", va="top",
            fontsize=6.8, color=RUST)
    ax.text(250, START * BOUND + 0.030, f"z = {START:.2f}a", ha="right",
            va="bottom", fontsize=6.6, color=MUTED)

    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-0.80, 1.80)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])
    ax.set_xticks(np.arange(0, 1251, 250))
    ax.set_xlabel("time from stimulus onset (ms)", fontsize=8.5)
    ax.set_ylabel("accumulated evidence (evidence units)", fontsize=8.5)
    ax.set_title(f"One accumulator, two boundaries: n={N_TRIALS} simulated "
                 "trials", fontsize=9.0, fontweight="bold", pad=5.0)
    handles = [
        Line2D([], [], color=TEAL, lw=1.6, label="correct-terminating paths"),
        Line2D([], [], color=RUST, lw=1.6, label="error-terminating paths"),
        Line2D([], [], color=INK, lw=1.7, label="mean drift"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.8,
              handlelength=1.5, labelspacing=0.30, borderpad=0.12,
              bbox_to_anchor=(1.006, -0.012))

    # ---- aux (b): speed-accuracy across drift rates -------------------------
    accs, mrts, ez_v, ez_a, ez_t = [], [], [], [], []
    for v in DRIFT_SET:
        r, u = simulate(rng, v, 3000)
        rc = r[u]
        accs.append(float(u.mean()))
        mrts.append(float(r.mean() * 1000.0))
        vh, ah, th = ez_diffusion(u.mean(), rc.mean(), rc.var(ddof=1))
        ez_v.append(vh)
        ez_a.append(ah)
        ez_t.append(th * 1000.0)
    accs = np.array(accs)
    mrts = np.array(mrts)

    ax_t.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_t.plot(DRIFT_SET, 100 * accs, marker="o", ms=4.2, color=TEAL, lw=1.5,
              mfc="white", mec=TEAL, mew=1.1, zorder=5)
    ax_t.set_ylim(60.0, 104.0)
    ax_t.set_xlim(0.55, 2.85)
    ax_t.set_ylabel("accuracy (%)", fontsize=7.6, labelpad=2.0, color=TEAL)
    ax_t.tick_params(labelsize=6.6)
    ax_t.tick_params(axis="y", colors=TEAL)
    ax_t.set_xlabel("drift rate v (evidence units / s)", fontsize=7.6,
                    labelpad=1.5)

    ax_r = ax_t.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.plot(DRIFT_SET, mrts, marker="s", ms=4.0, color=BLUE, lw=1.5,
              ls=(0, (3.4, 2.0)), mfc="white", mec=BLUE, mew=1.1, zorder=5)
    ax_r.set_ylim(330.0, 720.0)
    ax_r.set_ylabel("mean RT (ms)", fontsize=7.6, labelpad=2.0, color=BLUE)
    ax_r.tick_params(labelsize=6.6, colors=BLUE)
    ax_t.set_title("Stronger evidence is faster and more accurate",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_t.text(0.62, 63.0, "n=3000 trials per drift level", fontsize=6.2,
              color=MUTED, ha="left", va="bottom")

    # ---- aux (c): EZ-diffusion parameter recovery ---------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    lim = (0.55, 2.95)
    ax_b.plot(lim, lim, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=2)
    ax_b.plot(DRIFT_SET, ez_v, marker="o", ls="none", ms=5.4, mfc=ACCENT,
              mec="white", mew=0.8, zorder=5)
    for v0, vh in zip(DRIFT_SET, ez_v):
        ax_b.text(v0, vh - 0.10, f"{vh:.2f}", ha="center", va="top",
                  fontsize=6.2, color=ACCENT)
    err = np.max(np.abs(np.array(ez_v) - np.array(DRIFT_SET)))
    ax_b.text(0.62, 2.82,
              f"boundary a: generating {BOUND:.2f} -> recovered "
              f"{np.mean(ez_a):.2f}\n"
              f"non-decision Ter: generating {1000 * TER_S:.0f} ms -> "
              f"recovered {np.mean(ez_t):.0f} ms\n"
              f"largest drift error {err:.2f} evidence units / s",
              ha="left", va="top", fontsize=6.3, color=SUB, linespacing=1.5)
    ax_b.set_xlim(*lim)
    ax_b.set_ylim(*lim)
    ax_b.set_xticks([1.0, 1.5, 2.0, 2.5])
    ax_b.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("generating drift rate (evidence units / s)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("EZ estimate", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("EZ recovery sits on the identity line", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "955_cog_drift_diffusion.png")


if __name__ == "__main__":
    main()
