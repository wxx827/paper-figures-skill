"""Bland-Altman agreement analysis paired with a method-comparison scatter."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

PT = "#3b5b92"
BIAS_C = "#b23a48"
LOA_C = "#5f6b73"


def main():
    set_pub_style()
    rng = np.random.default_rng(4)
    n = 124

    truth = rng.normal(132.0, 16.0, n)
    diff = 1.2 - 0.09 * (truth - 132.0) + rng.normal(0.0, 5.2, n)
    dev = truth + diff / 2.0          # oscillometric device
    ref = truth - diff / 2.0          # intra-arterial reference
    avg = (dev + ref) / 2.0

    bias = diff.mean()
    sd = diff.std(ddof=1)
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    ci_bias = 1.96 * sd / np.sqrt(n)
    ci_loa = 1.96 * sd * np.sqrt(3.0 / n)
    reg = stats.linregress(avg, diff)
    outside = int(np.sum((diff < loa_lo) | (diff > loa_hi)))

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(7.4, 3.5),
        gridspec_kw={"width_ratios": [1.42, 1.0], "wspace": 0.30})
    fig.subplots_adjust(left=0.082, right=0.985, top=0.815, bottom=0.145)

    # ---- panel a: Bland-Altman --------------------------------------------
    x0, x1 = avg.min() - 6, avg.max() + 6
    span = np.array([x0, x1])
    for lvl, ci, ls, col, lab in (
            (bias, ci_bias, "-", BIAS_C, "Mean difference (bias)"),
            (loa_hi, ci_loa, (0, (5, 3)), LOA_C, "Upper limit (+1.96 SD)"),
            (loa_lo, ci_loa, (0, (5, 3)), LOA_C, "Lower limit (\u22121.96 SD)")):
        ax.fill_between(span, lvl - ci, lvl + ci, color=col, alpha=0.13,
                        linewidth=0, zorder=1)
        ax.plot(span, [lvl, lvl], ls=ls, lw=1.1, color=col, zorder=3,
                label=lab)

    ax.axhline(0, color="#c4c4c4", lw=0.7, zorder=1)
    inside = (diff >= loa_lo) & (diff <= loa_hi)
    ax.scatter(avg[inside], diff[inside], s=13, facecolor=PT, edgecolor="white",
               linewidth=0.35, alpha=0.85, zorder=4)
    ax.scatter(avg[~inside], diff[~inside], s=20, facecolor="none",
               edgecolor=BIAS_C, linewidth=0.9, zorder=5)
    ax.plot(span, reg.intercept + reg.slope * span, ls=(0, (1.5, 1.5)),
            lw=1.1, color="#1a1a1a", zorder=6,
            label="Regression of difference on mean")

    for lvl, txt, side in ((bias, f"bias {bias:+.2f}", "right"),
                           (loa_hi, f"+1.96 SD  {loa_hi:+.1f}", "left"),
                           (loa_lo, f"\u22121.96 SD  {loa_lo:+.1f}", "right")):
        xa = x0 + 0.8 if side == "left" else x1 - 0.6
        ax.text(xa, lvl + 0.55, txt, ha=side, va="bottom",
                fontsize=6.3, fontweight="bold", zorder=7, path_effects=HALO,
                color=BIAS_C if txt.startswith("bias") else LOA_C)

    ax.set_xlim(x0, x1)
    # headroom at the top keeps the statistics call-out clear of the data
    ax.set_ylim(bias - 4.4 * sd, bias + 5.8 * sd)
    ax.set_xlabel("Mean of the two methods (mmHg)")
    ax.set_ylabel("Difference, device \u2212 reference (mmHg)")
    ax.grid(color="#f0f0f0", lw=0.6, zorder=0)
    style_axes(ax)
    leg = ax.legend(loc="lower left", frameon=True, fontsize=6.2,
                    handlelength=1.8, labelspacing=0.32, borderpad=0.45,
                    framealpha=0.94, edgecolor="#d5d5d5")
    leg._legend_box.align = "left"
    ax.set_title("Bland\u2013Altman agreement", fontsize=8.4, loc="left", pad=4)
    add_panel_label(ax, "(a)", x=-0.105, y=1.02, size=10)

    ax.text(0.985, 0.975,
            f"Bias {bias:+.2f} mmHg (95% CI {bias - ci_bias:+.2f} to "
            f"{bias + ci_bias:+.2f})\n"
            f"Limits of agreement {loa_lo:+.1f} to {loa_hi:+.1f} mmHg\n"
            f"Proportional bias: slope {reg.slope:+.3f}, "
            f"$P$ = {reg.pvalue:.3f}\n"
            f"{outside}/{n} points ({100 * outside / n:.1f}%) outside limits",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.3,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="#d5d5d5", linewidth=0.6, alpha=0.95))

    # ---- panel b: method comparison ---------------------------------------
    lo = min(ref.min(), dev.min()) - 5
    hi = max(ref.max(), dev.max()) + 5
    fit = stats.linregress(ref, dev)
    r = np.corrcoef(ref, dev)[0, 1]
    ccc = (2 * np.cov(ref, dev, ddof=1)[0, 1] /
           (ref.var(ddof=1) + dev.var(ddof=1) + (ref.mean() - dev.mean()) ** 2))

    axr.plot([lo, hi], [lo, hi], ls=(0, (5, 3)), lw=1.0, color="#9a9a9a",
             zorder=2, label="Line of identity")
    axr.scatter(ref, dev, s=13, facecolor=PT, edgecolor="white",
                linewidth=0.35, alpha=0.85, zorder=3)
    xs = np.array([lo, hi])
    axr.plot(xs, fit.intercept + fit.slope * xs, lw=1.3, color=BIAS_C,
             zorder=4, label="Ordinary least squares")

    axr.set_xlim(lo, hi)
    axr.set_ylim(lo, hi)
    axr.set_aspect("equal", adjustable="box")
    axr.set_xlabel("Intra-arterial reference (mmHg)")
    axr.set_ylabel("Oscillometric device (mmHg)")
    axr.grid(color="#f0f0f0", lw=0.6, zorder=0)
    style_axes(axr)
    axr.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.8,
               labelspacing=0.3)
    axr.set_title("Method comparison", fontsize=8.4, loc="left", pad=4)
    add_panel_label(axr, "(b)", x=-0.185, y=1.02, size=10)

    axr.text(0.975, 0.035,
             f"Pearson $r$ = {r:.3f}\n"
             f"Lin's CCC = {ccc:.3f}\n"
             f"slope = {fit.slope:.3f}\n"
             f"intercept = {fit.intercept:+.1f} mmHg",
             transform=axr.transAxes, ha="right", va="bottom", fontsize=6.3,
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                       edgecolor="#d5d5d5", linewidth=0.6, alpha=0.95))


    savefig(fig, "116_bland_altman.png")


if __name__ == "__main__":
    main()
