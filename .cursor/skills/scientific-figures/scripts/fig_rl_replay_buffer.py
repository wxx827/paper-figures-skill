"""What prioritised experience replay actually does to the buffer.

One simulated buffer of 200k transitions drives all three panels: the TD-error
ridgeline is the error distribution at eight checkpoints, the middle panel is
the sampling probability those same errors induce for four exponents alpha, and
the right panel is the age distribution the chosen alpha produces - uniform
replay would be flat there, and is drawn for contrast.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from pubstyle import (set_pub_style, savefig, get_palette, style_axes,
                      annotate_box)
from figkit import panel_label_inches

CKPTS = [0.2, 0.5, 0.9, 1.4, 2.0, 2.7, 3.5, 4.4]      # millions of env steps
SIGMA = np.array([0.92, 0.74, 0.58, 0.46, 0.38, 0.32, 0.28, 0.25])
N_DRAW = 6000
BUF, N_SAMPLE = 200_000, 60_000
ALPHAS = [0.0, 0.4, 0.7, 1.0]
ALPHA_C = {0.0: "#808080", 0.4: "#c9a227", 0.7: "#3b5b92", 1.0: "#8e6bbf"}
ALPHA_USED = 0.7
GREY = "#8c8c8c"
GRID = "#eeeeee"


def td_errors(rng):
    """Heavy-tailed signed TD errors per checkpoint, shrinking with training."""
    out = []
    for s in SIGMA:
        d = rng.standard_t(df=3.2, size=N_DRAW) * s / np.sqrt(3.2 / 1.2)
        out.append(d + rng.normal(0.02 * s, 0.03 * s, N_DRAW))
    return out


def buffer_sim(rng):
    """FIFO buffer: age 0 = newest.  Fresh transitions carry larger errors."""
    age = np.arange(BUF)
    scale = 0.18 + 1.30 * np.exp(-age / (0.16 * BUF))
    delta = np.abs(rng.standard_t(df=2.0, size=BUF)) * scale
    return age, delta


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4380)
    errs = td_errors(rng)
    age, delta = buffer_sim(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62],
                          left=0.075, right=0.975, top=0.885, bottom=0.095,
                          hspace=0.42, wspace=0.24)
    axr = fig.add_subplot(gs[:, 0])
    axp = fig.add_subplot(gs[0, 1])
    axa = fig.add_subplot(gs[1, 1])

    # ---- (a) ridgeline of TD-error distributions --------------------------
    xg = np.linspace(-2.6, 2.6, 600)
    dens = [gaussian_kde(e, bw_method=0.28)(xg) for e in errs]
    peak = max(d.max() for d in dens)
    spacing = 1.0
    height = 2.05 * spacing / peak            # one shared density scale
    cols = get_palette("cb_sequential_blue", len(CKPTS) + 1)[1:]
    for k in range(len(CKPTS) - 1, -1, -1):
        base = k * spacing
        y = base + dens[k] * height
        axr.fill_between(xg, base, y, color=cols[k], alpha=0.88, lw=0,
                         zorder=3 + k)
        axr.plot(xg, y, color="white", lw=0.7, zorder=3 + k)
        axr.plot(xg, y, color="#1a1a1a", lw=0.45, alpha=0.55, zorder=3 + k)
        axr.plot([np.median(errs[k])] * 2, [base, base + 0.30 * spacing],
                 color="#1a1a1a", lw=0.8, zorder=3 + k + 1)
        iqr = float(np.subtract(*np.percentile(errs[k], [75, 25])))
        axr.text(2.52, base + 0.10, f"IQR $= {iqr:.2f}$", fontsize=6.4,
                 color="#555555", ha="right", va="bottom", zorder=40)
    axr.axvline(0.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    axr.set_yticks([k * spacing for k in range(len(CKPTS))])
    axr.set_yticklabels([f"{c:.1f}M" for c in CKPTS], fontsize=7.0)
    axr.set_ylim(-0.25, (len(CKPTS) - 1) * spacing + 2.5)
    axr.set_xlim(-2.6, 2.6)
    axr.set_xlabel("TD error $\\delta = r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)$"
                   "  (return units)", fontsize=8)
    axr.set_ylabel("Environment steps at checkpoint", fontsize=8)
    axr.set_title("TD-error distribution narrows as training proceeds",
                  fontsize=8.8, fontweight="bold", pad=4)
    axr.grid(axis="x", color=GRID, lw=0.7)
    axr.set_axisbelow(True)
    style_axes(axr)

    # ---- (b) priority weights ---------------------------------------------
    order = np.argsort(-delta)
    rank = np.arange(1, BUF + 1) / BUF * 100.0
    sd = delta[order]
    for a in ALPHAS:
        p = sd ** a
        p = p / p.sum()
        lab = "uniform ($\\alpha=0$)" if a == 0 else f"$\\alpha={a}$"
        axp.plot(rank, p * BUF, color=ALPHA_C[a],
                 lw=2.0 if a == ALPHA_USED else 1.3,
                 ls=(0, (3.2, 1.6)) if a == 0 else "-", label=lab,
                 zorder=6 if a == ALPHA_USED else 4)
    pu = sd ** ALPHA_USED
    pu = pu / pu.sum()
    top1 = pu[:BUF // 100].sum() * 100.0
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.set_xlim(0.01, 100)
    axp.set_ylim(2e-2, 3e2)
    axp.set_xlabel("transition rank by $|\\delta|$ (% of buffer)", fontsize=8)
    axp.set_ylabel("relative draw rate ($\\times$ uniform)", fontsize=8)
    axp.set_title("Priority exponent decides how skewed replay is",
                  fontsize=8.4, fontweight="bold", pad=4)
    axp.grid(color=GRID, lw=0.7)
    axp.set_axisbelow(True)
    axp.legend(frameon=False, fontsize=6.9, loc="lower left", ncol=2,
               columnspacing=1.0, handlelength=1.7)
    annotate_box(axp, f"at $\\alpha={ALPHA_USED}$ the top 1% by $|\\delta|$\n"
                 f"take {top1:.0f}% of all draws "
                 f"({top1:.0f}$\\times$ uniform)",
                 loc="upper right", fontsize=6.6)
    axp.tick_params(labelsize=7.0)
    style_axes(axp)

    # ---- (c) age of what gets replayed ------------------------------------
    prob = delta ** ALPHA_USED
    prob = prob / prob.sum()
    drawn = rng.choice(BUF, size=N_SAMPLE, p=prob)
    bins = np.linspace(0, BUF, 41) / 1000.0
    axa.hist(age[drawn] / 1000.0, bins=bins, density=True, histtype="stepfilled",
             color="#3b5b92", alpha=0.30, lw=0, zorder=3)
    axa.hist(age[drawn] / 1000.0, bins=bins, density=True, histtype="step",
             color="#3b5b92", lw=1.8, zorder=5,
             label=f"drawn by PER ($\\alpha={ALPHA_USED}$)")
    axa.axhline(1.0 / (BUF / 1000.0), color=GREY, lw=1.3,
                ls=(0, (3.2, 1.6)), zorder=4, label="uniform replay")
    mean_per = age[drawn].mean() / 1000.0
    axa.axvline(mean_per, color="#b23a48", lw=1.1, zorder=6)
    axa.annotate(f"mean age {mean_per:.0f}k\n(uniform: {BUF / 2000:.0f}k)",
                 xy=(mean_per, 0.75), xycoords=("data", "axes fraction"),
                 xytext=(14, 0), textcoords="offset points", fontsize=6.6,
                 color="#b23a48", va="center",
                 arrowprops=dict(arrowstyle="-", color="#b23a48", lw=0.7))
    axa.set_xlim(0, BUF / 1000.0)
    axa.set_xlabel("transition age in buffer (k steps, 0 = newest)", fontsize=8)
    axa.set_ylabel("density (1/k steps)", fontsize=8)
    axa.set_title("Replay skews young; the buffer tail is rarely revisited",
                  fontsize=8.4, fontweight="bold", pad=4)
    axa.grid(axis="y", color=GRID, lw=0.7)
    axa.set_axisbelow(True)
    axa.legend(frameon=False, fontsize=6.9, loc="upper right")
    axa.tick_params(labelsize=7.0)
    style_axes(axa)

    for ax, lab in ((axr, "(a)"), (axp, "(b)"), (axa, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.42, dy=0.08)

    savefig(fig, "438_rl_replay_buffer.png")


if __name__ == "__main__":
    main()
