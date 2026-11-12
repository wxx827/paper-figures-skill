"""Figure 1190 - Device memory against context length, and where each config dies.

One memory model, three configurations, two panels:

    parts(L, cfg) = parameters + KV cache + activations        (GB)
    L_max(cfg)    = the L at which those three add up to CAP    (exact root)

(a) stacks the three terms for the baseline and draws the other two as total
curves; (b) is the same function evaluated at each configuration's own
``L_max``, so every bar in (b) necessarily adds up to the cap line in (a).
Nothing in (b) is typed in by hand - change a coefficient and both panels,
the footer arithmetic and the quoted context lengths move together.

The cap is drawn with explicit end points rather than ``axhline`` so it cannot
strike through the y-axis labels.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from pubstyle import set_pub_style, savefig, style_axes

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
GRID, CARD = "#ededed", "#e2e8ee"
C_PAR, C_KV, C_ACT = "#134a76", "#2e74a8", "#9cc3dd"
H_PAR, H_KV, H_ACT = "", "", "///"
CAP = 80.0                      # GB of HBM on the device
PARAMS = 13.4                   # GB, identical for the three configs
LMAX_X = 34.0                   # right edge, k tokens

# name -> (KV GB per k tokens, activation offset, linear, quadratic)
CONFIGS = [
    ("A   dense KV, no checkpointing", 3.100, 2.80, 0.620, 0.01200, "-"),
    ("B   + paged KV (block 16)", 1.798, 2.80, 0.620, 0.01200, (0, (4.2, 1.8))),
    ("C   + paged KV + act. checkpointing", 1.798, 2.80, 0.217, 0.00300,
     (0, (1.2, 1.5))),
]


def parts(L, cfg):
    """(parameters, KV cache, activations) in GB at context length L."""
    _, kv, a0, a1, a2, _ = cfg
    L = np.asarray(L, dtype=float)
    return (np.full_like(L, PARAMS), kv * L, a0 + a1 * L + a2 * L ** 2)


def total(L, cfg):
    return sum(parts(L, cfg))


def l_max(cfg):
    """Exact root of total(L) = CAP for the quadratic memory model."""
    _, kv, a0, a1, a2, _ = cfg
    c = PARAMS + a0 - CAP
    return float((-(kv + a1) + np.sqrt((kv + a1) ** 2 - 4 * a2 * c))
                 / (2 * a2))


def parts_decimals(rows, target, lo=1, hi=3):
    """Fewest decimals at which every row of parts still adds to the cap."""
    for dp in range(lo, hi + 1):
        if all(abs(sum(round(v, dp) for v in r) - round(target, dp)) < 1e-9
               for r in rows):
            return dp
    return hi


def main():
    set_pub_style(base_font=8.5)

    L = np.linspace(0.0, LMAX_X, 400)
    lm = [l_max(c) for c in CONFIGS]
    peak = [[float(p) for p in parts(lm[i], c)] for i, c in enumerate(CONFIGS)]
    dp = parts_decimals(peak, CAP)
    shown = [[round(v, dp) for v in r] for r in peak]
    for r in shown:
        assert abs(sum(r) - CAP) < 1e-9, \
            "a printed peak breakdown does not add up to the cap"
    assert lm[0] < lm[1] < lm[2], "the configs are no longer ordered"

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], left=0.070,
                          right=0.988, top=0.800, bottom=0.235, wspace=0.24)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    ytop = 104.0

    # ---- (a) stacked footprint of the baseline ------------------------------
    style_axes(axa)
    axa.set_axisbelow(True)
    axa.grid(color=GRID, lw=0.7, zorder=0)
    axa.fill_between([0.0, LMAX_X], [CAP, CAP], [ytop, ytop],
                     facecolor="#f7f7f7", edgecolor="#dcdcdc", hatch="//",
                     lw=0.0, zorder=1)
    axa.text(LMAX_X - 0.4, ytop - 2.0, "out of memory", ha="right", va="top",
             fontsize=8.6, color="#8a8a8a", style="italic", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.6,
                       alpha=0.92))

    # the stack is only drawn where configuration A actually fits
    Lf = np.append(L[L < lm[0]], lm[0])
    pp, kv, ac = parts(Lf, CONFIGS[0])
    axa.fill_between(Lf, 0.0, pp, facecolor=C_PAR, edgecolor="white", lw=0.5,
                     hatch=H_PAR, zorder=2)
    axa.fill_between(Lf, pp, pp + kv, facecolor=C_KV, edgecolor="white",
                     lw=0.5, hatch=H_KV, zorder=2)
    axa.fill_between(Lf, pp + kv, pp + kv + ac, facecolor=C_ACT,
                     edgecolor="white", lw=0.5, hatch=H_ACT, zorder=2)
    axa.plot(L, total(L, CONFIGS[0]), color="#0b3d63", lw=1.6, zorder=6)

    # in-band component labels, parked where the bands are widest
    axa.text(6.0, PARAMS / 2.0, f"parameters  {PARAMS:.1f} GB",
             ha="center", va="center", fontsize=7.6, color="white",
             fontweight="bold", zorder=5)
    axa.text(8.4, PARAMS + 3.10 * 8.4 / 2.0, "KV cache", ha="center",
             va="center", fontsize=7.8, color="white", fontweight="bold",
             zorder=5)
    axa.text(13.6, float(PARAMS + 3.10 * 13.6 + 0.5
                         * (2.80 + 0.620 * 13.6 + 0.012 * 13.6 ** 2)),
             "activations", ha="center", va="center", fontsize=7.6,
             color=INK, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0,
                       alpha=0.88))

    # the other two configurations, totals only
    for i, cfg in enumerate(CONFIGS):
        if i == 0:
            continue
        axa.plot(L, total(L, cfg), color="#0b3d63", lw=1.5, ls=cfg[5],
                 zorder=6)

    # cap: explicit end points, never axhline
    axa.plot([0.0, LMAX_X], [CAP, CAP], color="#3a3a3a", lw=1.5, zorder=7)
    axa.text(0.35, CAP + 1.4, f"{CAP:.0f} GB HBM cap", ha="left", va="bottom",
             fontsize=8.0, color=INK, fontweight="bold", zorder=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4,
                       alpha=0.94))

    for i, cfg in enumerate(CONFIGS):
        axa.plot([lm[i], lm[i]], [0.0, CAP], color="#4d4d4d", lw=0.8,
                 ls=(0, (2.4, 2.0)), zorder=6)
        axa.plot([lm[i]], [CAP], marker="o", ms=5.2, mfc="white",
                 mec="#0b3d63", mew=1.5, zorder=9)
        axa.text(lm[i], CAP + 5.2 + 6.4 * (i % 2),
                 f"{cfg[0].split()[0]}   {lm[i]:.1f}k", ha="center",
                 va="bottom", fontsize=8.0, color="#0b3d63",
                 fontweight="bold", zorder=9,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=HAIR,
                           lw=0.5, alpha=0.96))

    axa.set_xlim(0.0, LMAX_X)
    axa.set_ylim(0.0, ytop)
    axa.set_xlabel("context length  (k tokens)", fontsize=8.6)
    axa.set_ylabel("device memory  (GB)", fontsize=8.6)
    axa.tick_params(labelsize=7.4, length=2.4)
    axa.set_title("(a)  Footprint of configuration A, with the two "
                  "optimised totals", fontsize=9.4, fontweight="bold",
                  pad=5.0, loc="left")

    pat = [Patch(facecolor=C_PAR, edgecolor="white", hatch=H_PAR,
                 label="parameters"),
           Patch(facecolor=C_KV, edgecolor="white", hatch=H_KV,
                 label="KV cache"),
           Patch(facecolor=C_ACT, edgecolor="white", hatch=H_ACT,
                 label="activations")]
    leg1 = axa.legend(handles=pat, loc="upper left",
                      bbox_to_anchor=(0.012, 0.985), frameon=True,
                      facecolor="white", edgecolor=HAIR, framealpha=0.95,
                      fontsize=7.8, handlelength=1.4, labelspacing=0.30,
                      borderpad=0.42)
    leg1.get_frame().set_linewidth(0.5)
    leg1.set_zorder(10)

    # ---- (b) the same model, evaluated at each L_max ------------------------
    style_axes(axb)
    axb.set_axisbelow(True)
    axb.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    xs = np.arange(len(CONFIGS), dtype=float)
    bottom = np.zeros(len(CONFIGS))
    for j, (col, hat) in enumerate(((C_PAR, H_PAR), (C_KV, H_KV),
                                    (C_ACT, H_ACT))):
        vals = np.array([shown[i][j] for i in range(len(CONFIGS))])
        axb.bar(xs, vals, bottom=bottom, width=0.56, color=col,
                edgecolor="white", linewidth=0.8, hatch=hat, zorder=3)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            axb.text(xs[i], b + v / 2.0, f"{v:.{dp}f}", ha="center",
                     va="center", fontsize=7.4,
                     color="white" if j < 2 else INK, fontweight="bold",
                     zorder=5)
        bottom = bottom + vals
    axb.plot([-0.62, len(CONFIGS) - 0.38], [CAP, CAP], color="#3a3a3a",
             lw=1.5, zorder=6)
    axb.set_xlim(-0.62, len(CONFIGS) - 0.38)
    axb.set_ylim(0.0, ytop)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"{c[0].split()[0]}\n{lm[i]:.1f}k tokens"
                         for i, c in enumerate(CONFIGS)], fontsize=8.2,
                        linespacing=1.6)
    axb.set_ylabel("peak memory at $L_{max}$  (GB)", fontsize=8.6)
    axb.tick_params(labelsize=7.4, length=2.4)
    axb.set_title("(b)  Peak mix at each $L_{max}$", fontsize=9.4,
                  fontweight="bold", pad=5.0, loc="left")
    axb.text(0.5, 0.955, "every bar is the same cap,\nsplit differently",
             transform=axb.transAxes, ha="center", va="top", fontsize=7.4,
             color=SUB, linespacing=1.5, zorder=7,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.95))

    # ---- line legend, one entry per curve actually drawn --------------------
    lines = [Line2D([], [], color="#0b3d63", lw=1.6, ls=c[5],
                    label=f"total, {c[0].split(maxsplit=1)[0]}"
                          f"  ({c[0].split(maxsplit=1)[1]})")
             for c in CONFIGS]
    lines.append(Line2D([], [], color="#3a3a3a", lw=1.5,
                        label=f"{CAP:.0f} GB cap"))
    leg2 = fig.legend(handles=lines, loc="lower center",
                      bbox_to_anchor=(0.5, 0.128), ncol=4, frameon=True,
                      facecolor="white", edgecolor=HAIR, framealpha=0.95,
                      fontsize=7.8, handlelength=2.2, columnspacing=1.5,
                      borderpad=0.42)
    leg2.get_frame().set_linewidth(0.5)

    # ---- headline and the arithmetic ---------------------------------------
    gain = lm[2] / lm[0]
    # wrapped, or the one-line version widens the exported canvas by 10 %

    band = FancyBboxPatch((0.022, 0.020), 0.956, 0.078,
                          boxstyle="round,pad=0.004,rounding_size=0.008",
                          transform=fig.transFigure, facecolor=CARD,
                          edgecolor="#c3cdd6", linewidth=0.6, zorder=-5)
    fig.add_artist(band)
    band.set_in_layout(False)
    txt = "     ".join(
        f"{c[0].split()[0]} at {lm[i]:.1f}k:  "
        + " + ".join(f"{v:.{dp}f}" for v in shown[i])
        + f" = {sum(shown[i]):.{dp}f} GB"
        for i, c in enumerate(CONFIGS))
    fig.text(0.036, 0.081, f"parameters + KV + activations at $L_{{max}}$"
             f"        {txt}", fontsize=7.4, color=SUB, ha="left", va="top")
    savefig(fig, "1190_wm_sys_memory_footprint.png")


if __name__ == "__main__":
    main()
