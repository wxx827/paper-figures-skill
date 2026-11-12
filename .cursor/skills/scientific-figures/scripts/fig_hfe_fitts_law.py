"""Fitts' law: movement time against index of difficulty for a pointing task.

Main panel: per-participant condition means for sixteen target conditions
(four amplitudes x four widths) plotted against ID = log2(2A / W), coloured by
target width, with the ordinary-least-squares fit, its 95 % confidence band
and the resulting throughput IP = 1 / b.  The intercept, slope, coefficient of
determination and throughput are all computed from the points that are drawn.

Auxiliary (both at 58 % of the main panel width): throughput for three input
devices, each from its own regression over its own simulated trials; and the
error rate against index of difficulty for the same three devices, counted
from simulated Bernoulli hits.

Fictional laboratory data, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, lighten, annotate_box
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
TEAL = "#2e7d5b"
GOLD = "#c9a227"
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

AMPS = [60.0, 120.0, 240.0, 480.0]        # movement amplitude, mm
WIDTHS = [48.0, 24.0, 12.0, 6.0]          # target width, mm
W_COLORS = ["#9cc3dd", "#5f9dc6", "#2e74a8", "#134a76"]
N_SUBJ = 12

# intercept (ms) and slope (ms/bit) used to synthesise each device
DEVICES = [("mouse", 168.0, 148.0, 0.019, BLUE),
           ("touchscreen", 122.0, 196.0, 0.034, TEAL),
           ("gamepad", 235.0, 268.0, 0.047, GOLD)]
N_TRIALS = 20                              # trials per participant-condition


def conditions():
    """ID and geometry for every amplitude-width combination."""
    rows = []
    for a in AMPS:
        for w in WIDTHS:
            rows.append((a, w, float(np.log2(2.0 * a / w))))
    return rows


def simulate(rng, intercept, slope, err_scale):
    """Condition means (ms) per participant, plus per-condition error counts."""
    conds = conditions()
    ids = np.array([c[2] for c in conds])
    subj_gain = rng.normal(1.0, 0.085, N_SUBJ)
    subj_off = rng.normal(0.0, 26.0, N_SUBJ)
    mt = np.empty((N_SUBJ, len(conds)))
    for s in range(N_SUBJ):
        mu = subj_off[s] + subj_gain[s] * (intercept + slope * ids)
        mt[s] = mu + rng.normal(0.0, 22.0, ids.size)
    p_err = np.clip(err_scale * np.exp(0.285 * (ids - 3.0)), 0.002, 0.55)
    n_tot = N_SUBJ * N_TRIALS
    misses = rng.binomial(n_tot, p_err)
    return ids, mt, misses, n_tot


def ols_line(x, y):
    """Slope, intercept, R^2 and standard errors from the normal equations."""
    X = np.column_stack([np.ones(x.size), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = x.size - 2
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(xtx_inv) * s2)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot
    return beta, se, r2, s2, dof, xtx_inv


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(949)

    conds = conditions()
    dev_fit = {}
    for name, a0, b0, esc, col in DEVICES:
        ids, mt, misses, n_tot = simulate(rng, a0, b0, esc)
        x = np.repeat(ids[None, :], N_SUBJ, axis=0).ravel()
        y = mt.ravel()
        beta, se, r2, s2, dof, xtx_inv = ols_line(x, y)
        dev_fit[name] = dict(ids=ids, mt=mt, x=x, y=y, beta=beta, se=se,
                             r2=r2, s2=s2, dof=dof, xtx_inv=xtx_inv,
                             misses=misses, n_tot=n_tot, color=col)

    f = dev_fit["mouse"]
    a_hat, b_hat = f["beta"]
    ip = 1000.0 / b_hat                       # bit/s, since b is ms per bit
    se_ip = 1000.0 * f["se"][1] / b_hat ** 2

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: movement time against index of difficulty -------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    jit = rng.normal(0.0, 0.022, f["x"].size)
    for k, w in enumerate(WIDTHS):
        sel = np.array([c[1] == w for c in conds])
        m = np.repeat(sel[None, :], N_SUBJ, axis=0).ravel()
        ax.plot(f["x"][m] + jit[m], f["y"][m], ls="none", marker="o", ms=3.1,
                mfc=W_COLORS[k], mec="white", mew=0.35, alpha=0.9, zorder=4,
                label=f"W = {w:.0f} mm")

    xg = np.linspace(1.20, 7.45, 200)
    yg = a_hat + b_hat * xg
    Xg = np.column_stack([np.ones(xg.size), xg])
    se_fit = np.sqrt(np.einsum("ij,jk,ik->i", Xg, f["xtx_inv"], Xg) * f["s2"])
    tcrit = float(stats.t.ppf(0.975, f["dof"]))
    ax.fill_between(xg, yg - tcrit * se_fit, yg + tcrit * se_fit,
                    facecolor=lighten(ACCENT, 0.78), edgecolor="none",
                    zorder=5)
    ax.plot(xg, yg, color=ACCENT, lw=1.7, zorder=6)

    # condition means as open squares, so the regression target is visible
    cond_mean = f["mt"].mean(axis=0)
    ax.plot(f["ids"], cond_mean, ls="none", marker="s", ms=4.6, mfc="none",
            mec=INK, mew=0.85, zorder=7)

    ax.set_xlim(1.0, 7.7)
    ax.set_ylim(200.0, 1560.0)
    ax.set_xticks(np.arange(1.0, 7.6, 0.5))
    ax.set_xlabel("index of difficulty  ID = log2(2A / W)  (bit)",
                  fontsize=8.5)
    ax.set_ylabel("movement time (ms)", fontsize=8.5)
    ax.set_title("Discrete pointing task, mouse: 16 conditions, "
                 f"n={N_SUBJ} participants", fontsize=9.0, fontweight="bold",
                 pad=5.0)

    annotate_box(
        ax,
        f"MT = {a_hat:.0f} + {b_hat:.0f} x ID   (ms)\n"
        f"a = {a_hat:.0f} +/- {f['se'][0]:.0f} ms,  "
        f"b = {b_hat:.0f} +/- {f['se'][1]:.0f} ms/bit\n"
        f"R2 = {f['r2']:.3f},  n = {f['x'].size} condition means\n"
        f"throughput IP = 1/b = {ip:.2f} +/- {se_ip:.2f} bit/s",
        loc="upper left", fontsize=7.0, color=INK, margin=0.028,
        linespacing=1.55)

    handles = [Line2D([], [], marker="o", ls="none", ms=3.6, mfc=c,
                      mec="white", mew=0.35, label=f"W = {w:.0f} mm")
               for w, c in zip(WIDTHS, W_COLORS)]
    handles += [
        Line2D([], [], marker="s", ls="none", ms=4.4, mfc="none", mec=INK,
               mew=0.85, label="condition mean"),
        Line2D([], [], color=ACCENT, lw=1.7, label="least-squares fit"),
        Line2D([], [], color=lighten(ACCENT, 0.60), lw=4.0,
               label="95 % confidence band"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.9,
              handlelength=1.5, labelspacing=0.32, borderpad=0.15, ncol=2,
              columnspacing=1.0, bbox_to_anchor=(1.004, -0.006))
    ax.text(7.62, 1545.0,
            f"amplitudes A = {', '.join(f'{a:.0f}' for a in AMPS)} mm\n"
            f"{N_TRIALS} trials per participant per condition",
            ha="right", va="top", fontsize=6.6, color=MUTED,
            linespacing=1.5)

    # ---- aux (b): throughput by device -------------------------------------
    names = [d[0] for d in DEVICES]
    ips = np.array([1000.0 / dev_fit[n]["beta"][1] for n in names])
    ip_se = np.array([1000.0 * dev_fit[n]["se"][1] /
                      dev_fit[n]["beta"][1] ** 2 for n in names])
    pos = np.arange(len(names))
    ax_t.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_t.bar(pos, ips, width=0.56, color=[dev_fit[n]["color"] for n in names],
             edgecolor="white", linewidth=0.7, zorder=3)
    ax_t.errorbar(pos, ips, yerr=1.96 * ip_se, ls="none", ecolor=INK,
                  elinewidth=0.9, capsize=2.2, capthick=0.8, zorder=5)
    for k, n in enumerate(names):
        ax_t.text(k, ips[k] + 0.30, f"{ips[k]:.2f}", ha="center", va="bottom",
                  fontsize=6.6, color=INK)
        ax_t.text(k, 0.22, f"{dev_fit[n]['beta'][1]:.0f} ms/bit",
                  ha="center", va="bottom", fontsize=6.0, color="white")
    ax_t.set_xticks(pos)
    ax_t.set_xticklabels(names, fontsize=6.8)
    ax_t.set_xlim(-0.62, len(names) - 0.38)
    ax_t.set_ylim(0.0, max(ips) * 1.34)
    ax_t.tick_params(axis="y", labelsize=6.6)
    ax_t.set_ylabel("throughput IP (bit/s)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Throughput = 1 / regression slope", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(len(names) - 0.42, max(ips) * 1.31,
              "bars +/- 95 % CI (delta method)", ha="right", va="top",
              fontsize=6.1, color=MUTED)

    # ---- aux (c): error rate against difficulty ----------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    for name, *_ , col in DEVICES:
        d = dev_fit[name]
        rate = 100.0 * d["misses"] / d["n_tot"]
        # one point per unique ID: average the duplicated ID conditions
        uid, inv = np.unique(np.round(d["ids"], 4), return_inverse=True)
        ym = np.array([rate[inv == i].mean() for i in range(uid.size)])
        ax_b.plot(uid, ym, color=col, lw=1.4, marker="o", ms=3.0,
                  mfc=col, mec="white", mew=0.4, zorder=4, label=name)
    ax_b.axhline(4.0, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    ax_b.text(1.12, 4.5, "4 % nominal error target", ha="left", va="bottom",
              fontsize=6.1, color=MUTED)
    ax_b.set_xlim(1.0, 7.7)
    ax_b.set_ylim(0.0, 26.0)
    ax_b.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("index of difficulty (bit)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("mis-hits (%)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Accuracy cost of small, distant targets", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.3,
                handlelength=1.3, labelspacing=0.28, borderpad=0.12,
                bbox_to_anchor=(-0.012, 1.03))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "949_hfe_fitts_law.png")


if __name__ == "__main__":
    main()
