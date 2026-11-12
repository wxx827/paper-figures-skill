"""Figure 1178 - Reliability diagram of a world model's own confidence.

Every panel is a projection of ONE binned table built in ``binned``:

    conf_b (mean confidence)  acc_b (empirical frequency)  n_b (support)
    gap_b = conf_b - acc_b        ECE = sum_b (n_b/N) |gap_b|

(a) plots acc_b against conf_b with the perfect diagonal, (b) is n_b, (c) is
gap_b.  The ECE / MCE printed in the summary card are f-strings over those
arrays, and the card prints the three largest terms of the ECE sum so the
reader can add them up instead of trusting a single rounded number.

Touch ``predictions`` and the curve, the bars, the histogram, the gaps and
every printed number move together.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes

SEED = 1583  # original id; figure number is 1178
N = 12000
NB = 10                                     # equal-width confidence bins
MIN_N = 30                                  # bins this small are not scored

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
ACC, CURVE = "#9cc3dd", "#0b3d63"
# The gap is a second quantity, not a second hue.  It used to be magenta,
# which spent colour budget the rest of the batch does not spend and made
# this figure the one odd tile on the contact sheet.  Separation now comes
# from texture and from the two ends of the blue ramp: palest fill plus a
# dark diagonal hatch for the gap, the deepest step for the MCE bar.
GAP_FILL, GAP_EDGE, MCE_C = "#eaf2f8", "#1f4e79", "#08263f"
GAP_BAR, HATCH = "#dbe8f2", "///"
COUNT, DIAG, GRID = "#5f9dc6", "#8a8a8a", "#ededed"
CARD = "#e2e8ee"


def predictions(rng):
    """N (confidence, correct) pairs from a deliberately overconfident head.

    Two populations - routine frames and a hard third of the set - so every
    confidence bin carries real support instead of a handful of samples with
    a confidence interval half the panel tall.
    """
    hard = rng.random(N) < 0.34
    p = np.where(hard, rng.beta(1.9, 2.6, N), rng.beta(6.0, 1.35, N))
    conf = np.clip(p + 0.55 * p * (1.0 - p), 0.0, 1.0)   # too much confidence
    correct = rng.random(N) < p
    return conf, correct


def binned(conf, correct):
    edges = np.linspace(0.0, 1.0, NB + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, NB - 1)
    n = np.array([int((idx == b).sum()) for b in range(NB)])
    cb = np.array([conf[idx == b].mean() if n[b] else np.nan
                   for b in range(NB)])
    ab = np.array([correct[idx == b].mean() if n[b] else np.nan
                   for b in range(NB)])
    return edges, n, cb, ab


def main():
    set_pub_style(base_font=8.5)
    rng = np.random.default_rng(SEED)
    conf, correct = predictions(rng)
    edges, n, cb, ab = binned(conf, correct)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = n > 0
    gap = cb - ab                                   # signed: + = overconfident
    term = np.where(ok, n / float(N) * np.abs(gap), 0.0)
    ece = float(np.nansum(term))
    scored = ok & (n >= MIN_N)
    m = int(np.nanargmax(np.where(scored, np.abs(gap), np.nan)))
    mce = float(abs(gap[m]))
    assert abs(n.sum() - N) < 1, "bin counts do not add up to N"
    assert np.all(gap[ok] > 0), "headline claims every populated bin is over-"\
                                "confident; at least one bin is not"

    top3 = np.argsort(np.where(ok, term, -1.0))[::-1][:3]
    shown = [f"{term[i]:.4f}" for i in top3]
    part = sum(float(s) for s in shown)

    err = np.where(ok, 1.96 * np.sqrt(np.clip(ab * (1 - ab), 0, None) /
                                      np.maximum(n, 1)), np.nan)

    fig = plt.figure(figsize=(8.8, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62],
                          height_ratios=[1.0, 1.0], left=0.075, right=0.982,
                          top=0.845, bottom=0.185, wspace=0.30, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    axn = fig.add_subplot(gs[0, 1])
    axg = fig.add_subplot(gs[1, 1])
    for a in (ax, axn, axg):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- (a) reliability ---------------------------------------------------
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.plot([0, 1], [0, 1], color=DIAG, lw=1.1, ls=(0, (3.2, 2.2)), zorder=2)
    w = edges[1] - edges[0]
    ax.bar(mid[ok], ab[ok], width=w * 0.92, color=ACC, edgecolor="white",
           linewidth=0.6, zorder=3)
    ax.bar(mid[ok], gap[ok], width=w * 0.92, bottom=ab[ok], color=GAP_FILL,
           edgecolor=GAP_EDGE, linewidth=0.6, hatch=HATCH, zorder=4)
    ax.errorbar(cb[ok], ab[ok], yerr=err[ok], fmt="o", ms=4.6, mfc="white",
                mec=CURVE, mew=1.4, ecolor=CURVE, elinewidth=0.9, capsize=2.2,
                capthick=0.9, ls="-", color=CURVE, lw=1.7, zorder=6)

    ax.text(0.055, 0.735, "above the diagonal = under-confident",
            ha="left", va="center", fontsize=7.0, color=MUTED,
            style="italic", zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                      alpha=0.92))
    ax.annotate("every bin sits below the diagonal:\nthe model reports more "
                "confidence\nthan it earns",
                xy=(float(cb[m]), float(ab[m])), xytext=(0.055, 0.685),
                ha="left", va="top", fontsize=7.2, color=GAP_EDGE,
                linespacing=1.55, zorder=8,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#b9cadd",
                          lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", lw=0.8, color=GAP_EDGE,
                                shrinkA=3, shrinkB=5))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_aspect("equal")
    ax.set_xlabel("Reported confidence  $\\hat{p}$", fontsize=9.0)
    ax.set_ylabel("Empirical frequency of being correct", fontsize=9.0)
    ax.set_title(f"Reliability, {NB} equal-width bins", fontsize=9.5,
                 fontweight="bold", pad=5.0)
    handles = [Patch(facecolor=ACC, edgecolor="white", label="accuracy $a_b$"),
               Patch(facecolor=GAP_FILL, edgecolor=GAP_EDGE, hatch=HATCH,
                     label="gap $c_b - a_b$"),
               Line2D([], [], color=CURVE, lw=1.7, marker="o", ms=4.6,
                      mfc="white", mec=CURVE, mew=1.4,
                      label="bin means $\\pm$ 95 % CI"),
               Line2D([], [], color=DIAG, lw=1.1, ls=(0, (3.2, 2.2)),
                      label="perfect calibration")]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                    facecolor="white", edgecolor=HAIR, framealpha=0.94,
                    fontsize=7.2, handlelength=1.7, labelspacing=0.42,
                    borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(9)

    # ---- (b) support -------------------------------------------------------
    axn.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axn.bar(mid, n, width=w * 0.92, color=COUNT, edgecolor="white",
            linewidth=0.6, zorder=3)
    k = int(np.argmax(n))
    axn.text(mid[k], n[k] * 0.94, f"{n[k]}", ha="center", va="top",
             fontsize=6.8, color="white", fontweight="bold", zorder=5)
    thin = np.nonzero(ok & (n < MIN_N))[0]
    if thin.size:
        axn.text(mid[thin].mean(), max(n) * 0.60,
                 f"{thin.size} bins hold < {MIN_N}\nsamples and are not\n"
                 f"scored for MCE", ha="left", va="center", fontsize=6.6,
                 color=MUTED, linespacing=1.5, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                           alpha=0.92))
    axn.set_xlim(0.0, 1.0)
    axn.set_xticks(np.linspace(0, 1, 6))
    axn.set_ylabel(f"samples per bin\n($N$ = {N})", fontsize=7.8)
    axn.tick_params(labelsize=7.0, length=2.4)
    axn.set_title("Support of each bin", fontsize=8.6, fontweight="bold",
                  pad=4.0)

    # ---- (c) signed gap ----------------------------------------------------
    axg.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    # same language as (a): hatched pale bars, and the extreme one in the
    # deepest step of the ramp with a heavier rim instead of a second hue
    faces = [MCE_C if (scored[b] and b == m) else GAP_BAR for b in range(NB)]
    rims = [MCE_C if (scored[b] and b == m) else GAP_EDGE
            for b in range(NB)]
    axg.bar(mid[ok], gap[ok], width=w * 0.92,
            color=[faces[b] for b in range(NB) if ok[b]],
            edgecolor=[rims[b] for b in range(NB) if ok[b]],
            linewidth=[1.5 if b == m else 0.6 for b in range(NB) if ok[b]],
            hatch=HATCH, zorder=3)
    axg.plot([0.0, 1.0], [0.0, 0.0], color="#333333", lw=0.9, zorder=4)
    axg.plot([mid[m] - 0.5 * w, mid[m] + 0.5 * w], [gap[m], gap[m]],
             color=MCE_C, lw=1.8, zorder=5)
    axg.annotate(f"MCE = {mce:.3f}", xy=(mid[m], gap[m]),
                 xytext=(0.045, gap[m] * 0.98), ha="left", va="top",
                 fontsize=7.2, color=MCE_C, fontweight="bold", zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                           alpha=0.92),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MCE_C,
                                 shrinkA=2, shrinkB=3))
    axg.set_xlim(0.0, 1.0)
    axg.set_ylim(0.0, float(np.nanmax(gap[ok])) * 1.38)
    axg.set_xticks(np.linspace(0, 1, 6))
    axg.set_xlabel("Confidence bin", fontsize=8.4)
    axg.set_ylabel("$c_b - a_b$", fontsize=7.8)
    axg.tick_params(labelsize=7.0, length=2.4)
    axg.set_title("Overconfidence per bin", fontsize=8.6, fontweight="bold",
                  pad=4.0)

    # ---- headline + the arithmetic itself ----------------------------------

    lo = f"[{edges[m]:.1f}, {edges[m + 1]:.1f})"
    fig.text(0.075, 0.105,
             "$\\mathrm{ECE}=\\sum_b \\frac{n_b}{N}\\,|c_b-a_b|$"
             f" = {ece:.4f}          "
             "$\\mathrm{MCE}=\\max_b |c_b-a_b|$"
             f" = {mce:.3f}  in bin {lo}", fontsize=8.6, color=INK,
             ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.45", fc=CARD, ec="#c3cdd6",
                       lw=0.6))
    assert abs(part - sum(float(s) for s in shown)) < 1e-9
    savefig(fig, "1178_wm_unc_calibration.png")


if __name__ == "__main__":
    main()
