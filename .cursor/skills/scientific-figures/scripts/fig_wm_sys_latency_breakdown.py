"""Figure 1189 - Inference latency broken down by stage, three serving configs.

One table of stage timings drives the whole page:

    STAGES x CONFIGS  ->  (a) horizontal stacked bars, absolute milliseconds
                      ->  (b) the same rows renormalised to per-stage share
                      ->  footer, the four printed segment labels added up

The total printed at the end of every bar is the sum of the four *printed*
segment labels, not a separately rounded number, so a reader who adds the
labels lands exactly on it; an ``assert`` re-checks that on every run.  The
share panel picks the fewest decimals at which its own four numbers still add
to 100.0 for all three rows.

Editing the timings changes (a), (b), the footer arithmetic and the speed-up
together - there is no second copy of the numbers anywhere.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from pubstyle import set_pub_style, savefig, style_axes
from _wm_metrics import closed_ratio

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
GRID, CARD = "#ededed", "#e2e8ee"          # card grey 0.906, clear of 0.94

# One blue ladder; the two pale steps carry a texture so they stay separable
# in greyscale and at contact-sheet size, which is cheaper than a second hue.
STAGES = [
    ("Vision encoder", "#9cc3dd", "///", INK),
    ("Latent dynamics", "#2e74a8", "", "white"),
    ("Pixel decoder", "#134a76", "", "white"),
    ("Post-process", "#cfe1ee", "...", INK),
]
BH = 0.62                                  # bar height, both panels
CONFIGS = [
    ("A   dense FP16, no cache", [18.4, 42.7, 21.3, 5.6]),
    ("B   + KV cache reuse", [18.4, 24.9, 21.3, 5.6]),
    ("C   + fused decode kernel", [11.7, 24.9, 12.8, 4.1]),
]


def share_decimals(rows, lo=1, hi=3):
    """Fewest decimals at which every row of shares still adds to 100.0."""
    for dp in range(lo, hi + 1):
        if all(abs(sum(round(v, dp) for v in r) - 100.0) < 1e-9 for r in rows):
            return dp
    return hi


def main():
    set_pub_style(base_font=8.5)

    names = [c[0] for c in CONFIGS]
    vals = np.array([c[1] for c in CONFIGS], dtype=float)
    # every printed number below comes from these two arrays
    seg = np.round(vals, 1)                      # what the segments say
    tot = seg.sum(axis=1)                        # what the bar ends say
    assert np.allclose(tot, vals.sum(axis=1), atol=0.05), \
        "segment rounding moved a total by more than half a display digit"
    shares = 100.0 * seg / tot[:, None]
    dp = share_decimals(shares)
    shown_share = np.round(shares, dp)
    assert np.allclose(shown_share.sum(axis=1), 100.0, atol=1e-9), \
        "printed stage shares do not add to 100 %"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.52], left=0.145,
                          right=0.985, top=0.800, bottom=0.250, wspace=0.30)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    ypos = np.arange(len(CONFIGS))[::-1].astype(float)
    xmax = float(tot.max()) * 1.30

    # ---- (a) absolute milliseconds -----------------------------------------
    style_axes(axa)
    axa.set_axisbelow(True)
    axa.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for i in range(len(CONFIGS)):
        left = 0.0
        for j, (lab, col, hatch, tc) in enumerate(STAGES):
            v = seg[i, j]
            axa.barh(ypos[i], v, left=left, height=BH, color=col,
                     edgecolor="white", linewidth=0.8, hatch=hatch,
                     zorder=3)
            if v >= 7.5:
                axa.text(left + v / 2.0, ypos[i], f"{v:.1f}", ha="center",
                         va="center", fontsize=7.4, color=tc,
                         fontweight="bold", zorder=5)
            else:
                axa.plot([left + v / 2.0, left + v / 2.0 + 2.4],
                         [ypos[i] + BH / 2 + 0.02, ypos[i] + BH / 2 + 0.16],
                         color="#9a9a9a", lw=0.7, zorder=4)
                axa.text(left + v / 2.0 + 2.6, ypos[i] + BH / 2 + 0.18,
                         f"{v:.1f}",
                         ha="left", va="center", fontsize=7.0, color=SUB,
                         zorder=5)
            left += v
        axa.text(left + 1.6, ypos[i], f"= {tot[i]:.1f} ms", ha="left",
                 va="center", fontsize=8.8, color=INK, fontweight="bold",
                 zorder=6)
    axa.set_ylim(-0.62, len(CONFIGS) - 0.38)
    axa.set_xlim(0.0, xmax)
    axa.set_yticks(ypos)
    axa.set_yticklabels(names, fontsize=8.2)
    axa.get_yticklabels()[-1].set_fontweight("bold")
    axa.set_xlabel("end-to-end latency per frame  (ms)", fontsize=8.6)
    axa.tick_params(labelsize=7.4, length=2.4)
    axa.set_title("(a)  Where the milliseconds go", fontsize=9.4,
                  fontweight="bold", pad=5.0, loc="left")

    # speed-up quoted from the two printed totals, at a precision that closes
    ns, ds, q, _ = closed_ratio(float(tot[0]), float(tot[-1]))
    axa.text(0.985, 0.055, f"end-to-end speed-up  {ns} / {ds} = {q}$\\times$",
             transform=axa.transAxes, ha="right", va="bottom", fontsize=7.8,
             color=INK, zorder=7,
             bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=HAIR,
                       lw=0.6, alpha=0.95))

    # ---- (b) the same rows, renormalised ------------------------------------
    style_axes(axb)
    axb.set_axisbelow(True)
    axb.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for i in range(len(CONFIGS)):
        left = 0.0
        for j, (lab, col, hatch, tc) in enumerate(STAGES):
            v = shown_share[i, j]
            axb.barh(ypos[i], v, left=left, height=BH, color=col,
                     edgecolor="white", linewidth=0.8, hatch=hatch, zorder=3)
            if v >= 11.0:
                axb.text(left + v / 2.0, ypos[i], f"{v:.{dp}f}", ha="center",
                         va="center", fontsize=7.2, color=tc,
                         fontweight="bold", zorder=5)
            else:
                axb.plot([left + v / 2.0, left + v / 2.0 + 2.2],
                         [ypos[i] + BH / 2 + 0.02, ypos[i] + BH / 2 + 0.16],
                         color="#9a9a9a", lw=0.7, zorder=4)
                axb.text(left + v / 2.0 + 2.4, ypos[i] + BH / 2 + 0.18,
                         f"{v:.{dp}f}", ha="left", va="center", fontsize=7.0,
                         color=SUB, zorder=5)
            left += v
    axb.set_ylim(-0.62, len(CONFIGS) - 0.38)
    axb.set_xlim(0.0, 103.0)
    axb.set_yticks(ypos)
    axb.set_yticklabels(["A", "B", "C"], fontsize=8.2)
    axb.get_yticklabels()[-1].set_fontweight("bold")
    axb.set_xticks([0, 25, 50, 75, 100])
    axb.set_xlabel("share of that row's total  (%)", fontsize=8.6)
    axb.tick_params(labelsize=7.4, length=2.4)
    axb.set_title("(b)  Same three rows, renormalised", fontsize=9.4,
                  fontweight="bold", pad=5.0, loc="left")

    # ---- legend, one entry per stage actually drawn -------------------------
    handles = [Patch(facecolor=col, edgecolor="white", hatch=hatch, label=lab)
               for (lab, col, hatch, _) in STAGES]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.53, 0.135), ncol=4, frameon=True,
                     facecolor="white", edgecolor=HAIR, framealpha=0.95,
                     fontsize=8.0, handlelength=1.5, columnspacing=1.6,
                     borderpad=0.45)
    leg.get_frame().set_linewidth(0.5)

    # ---- headline and the addition itself -----------------------------------
    save_cache, save_fuse = tot[0] - tot[1], tot[1] - tot[2]
    assert save_cache > save_fuse, "the headline claims the wrong winner"
    # wrapped, or the one-line version widens the exported canvas by 23 %

    lines = "     ".join(
        f"{n.split()[0]}:  " + " + ".join(f"{v:.1f}" for v in seg[i])
        + f" = {tot[i]:.1f} ms" for i, n in enumerate(names))
    pct = "     ".join(
        f"{n.split()[0]}:  " + " + ".join(f"{v:.{dp}f}" for v in shown_share[i])
        + " = 100.0" for i, n in enumerate(names))
    band = FancyBboxPatch((0.022, 0.012), 0.956, 0.086,
                          boxstyle="round,pad=0.004,rounding_size=0.008",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#c3cdd6", linewidth=0.6, zorder=-5)
    fig.add_artist(band)
    band.set_in_layout(False)
    fig.text(0.036, 0.082, f"milliseconds        {lines}", fontsize=7.4,
             color=SUB, ha="left", va="top")
    savefig(fig, "1189_wm_sys_latency_breakdown.png")


if __name__ == "__main__":
    main()
