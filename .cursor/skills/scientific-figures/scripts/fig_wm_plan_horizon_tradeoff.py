"""Why a learned world model has a best planning horizon, and where it moves.

The main panel plots task return against the number of planned steps for four
model accuracies.  Return is built from exactly two terms,

    R(H) = gain(H) - cost(H, eps),   gain(H) = 1 - g^H,
                                     cost(H, eps) = eps * H(H+1)/2

so it is concave and has one integer maximiser.  The narrow right-hand panel
shows the two *marginal* terms behind that maximiser: planning one step further
buys a shrinking increment of value while it pays a growing increment of
compounded model error, and the optimal horizon is the last step at which the
first still exceeds the second.  The crossings in the right panel are therefore
the ringed optima in the left panel -- same arrays, one derivation, and the
script asserts that argmax R equals the last H with marginal gain >= marginal
cost before it draws anything.

Single-hue blue ladder (BRIEF_WM_BLUE.md section 2): the four accuracies are
separated by ladder depth *and* dash pattern.  The one warm accent marks the
optima and the trend line through them.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
CARD = "#dfe7ef"            # luminance 0.90: data panels carry the card
LIGHT = "#f4f8fb"           # luminance 0.97: statistics panels stay pale
HAIR = "#aab7c4"

FW, FH = 10.4, 6.1
GAMMA = 0.93                # value discount: how fast lookahead saturates
HMAX = 30

# one-step model error, its ladder rung and its dash pattern
MODELS = [
    (0.0015, BLUE[0], "-", "high-fidelity model"),
    (0.0035, BLUE[2], (0, (5.0, 1.9)), "well-trained model"),
    (0.0080, BLUE[3], (0, (3.0, 1.6, 0.9, 1.6)), "partially trained"),
    (0.0180, BLUE[4], (0, (1.4, 1.5)), "coarse model"),
]


def curves(eps):
    """Return, marginal gain and marginal cost over H = 1..HMAX."""
    H = np.arange(1, HMAX + 1, dtype=float)
    gain = 1.0 - GAMMA ** H
    cost = eps * H * (H + 1.0) / 2.0
    ret = gain - cost
    d_gain = np.diff(np.concatenate([[0.0], gain]))
    d_cost = np.diff(np.concatenate([[0.0], cost]))
    return H, ret, d_gain, d_cost


def main():
    set_pub_style(base_font=8)

    H, _, D_GAIN, _ = curves(MODELS[0][0])
    data = []
    for eps, col, ls, name in MODELS:
        _, ret, d_gain, d_cost = curves(eps)
        ok = np.nonzero(d_gain >= d_cost)[0]
        k = int(ok[-1])                       # last step still worth planning
        assert k == int(np.argmax(ret)), "marginal rule and argmax disagree"
        data.append(dict(eps=eps, col=col, ls=ls, name=name, ret=ret,
                         d_cost=d_cost, k=k, hstar=int(H[k]),
                         rstar=float(ret[k])))

    ref = data[1]
    k = ref["k"]
    shown = ["%.4f" % v for v in (D_GAIN[k], ref["d_cost"][k],
                                  D_GAIN[k + 1], ref["d_cost"][k + 1])]
    assert float(shown[0]) >= float(shown[1]) and \
        float(shown[2]) < float(shown[3]), "printed marginals do not close"

    fig = plt.figure(figsize=(FW, FH))
    axL = fig.add_axes([0.058, 0.112, 0.566, 0.726])
    axR = fig.add_axes([0.727, 0.112, 0.246, 0.726])

    # ---- (a) return against planning horizon ------------------------------
    axL.set_facecolor(CARD)
    for d in data:
        axL.plot(H, d["ret"], color=d["col"], lw=1.9, ls=d["ls"], zorder=4,
                 solid_capstyle="round")
    axL.plot([H[0], H[-1]], [0.0, 0.0], color="#7f8c99", lw=0.8,
             ls=(0, (1.2, 2.2)), zorder=3)
    axL.text(H[-1], 0.012, "zero return", ha="right", va="bottom",
             fontsize=6.6, color="#5d6b78", zorder=6,
             bbox=dict(boxstyle="round,pad=0.20", fc=CARD, ec="none",
                       alpha=0.92))

    kx = [d["hstar"] for d in data]
    ky = [d["rstar"] for d in data]
    axL.plot(kx, ky, color=ACCENT, lw=1.0, ls=(0, (2.4, 1.8)), zorder=6)
    axL.plot(kx, ky, ls="none", marker="o", ms=6.6, mfc="white", mec=ACCENT,
             mew=1.5, zorder=7)
    for d, dy in zip(data, (0.052, 0.052, -0.060, -0.060)):
        axL.text(d["hstar"], d["rstar"] + dy,
                 "$H^{*}$ = %d\nR = %.3f" % (d["hstar"], d["rstar"]),
                 ha="center", va="bottom" if dy > 0 else "top", fontsize=6.9,
                 color=INK, fontweight="bold", zorder=8, linespacing=1.35,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white",
                           ec="#c3ccd4", lw=0.6, alpha=0.96))
    axL.annotate("the optimum walks left\nas the model degrades",
                 xy=(9.2, 0.296), xytext=(11.6, -0.045), fontsize=7.2,
                 color=ACCENT, fontweight="bold", ha="center", va="center",
                 zorder=9, linespacing=1.4,
                 bbox=dict(boxstyle="round,pad=0.28", fc="white",
                           ec="#d9c4c1", lw=0.7, alpha=0.96),
                 arrowprops=dict(arrowstyle="-", color="#8a97a6", lw=0.8,
                                 shrinkA=3.0, shrinkB=3.0))

    axL.set_xlim(0.4, HMAX + 0.6)
    axL.set_ylim(-0.40, 0.62)
    axL.set_xlabel("planning horizon $H$ (steps unrolled in the world model)",
                   fontsize=8.0, labelpad=2.0)
    axL.set_ylabel("normalised task return $R(H)$", fontsize=8.0, labelpad=2.4)
    axL.tick_params(labelsize=7.2, length=2.4, pad=1.6)
    axL.grid(True, lw=0.45, color="#ccd7e2")
    axL.set_axisbelow(True)
    for side in ("left", "bottom"):
        axL.spines[side].set_linewidth(0.8)
        axL.spines[side].set_color("#7f8c99")
    axL.set_title("Return peaks at a finite horizon, and the peak depends on "
                  "model accuracy", fontsize=8.8, fontweight="bold",
                  color=INK, pad=4.0)

    handles = [Line2D([], [], color=d["col"], lw=1.9, ls=d["ls"],
                      label="%s,  $\\epsilon$ = %.4f" % (d["name"], d["eps"]))
               for d in data]
    handles.append(Line2D([], [], color=ACCENT, lw=1.0, ls=(0, (2.4, 1.8)),
                          marker="o", ms=5.6, mfc="white", mec=ACCENT,
                          mew=1.4, label="optimal horizon $H^{*}$"))
    leg = axL.legend(handles=handles, loc="upper left", fontsize=7.0,
                     frameon=True, handlelength=2.4, labelspacing=0.42,
                     borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#c3ccd4")
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_alpha(0.95)
    leg.set_zorder(9)

    # ---- (b) the two marginal terms ---------------------------------------
    axR.set_facecolor(LIGHT)
    axR.plot(H, D_GAIN, color=INK, lw=2.0, zorder=6, solid_capstyle="round")
    for d in data:
        axR.plot(H, d["d_cost"], color=d["col"], lw=1.5, ls=d["ls"], zorder=4)
        axR.plot([d["hstar"]], [D_GAIN[d["k"]]], ls="none", marker="o",
                 ms=5.4, mfc="white", mec=ACCENT, mew=1.4, zorder=7)
    axR.set_yscale("log")
    axR.set_xlim(0.4, HMAX + 0.6)
    axR.set_ylim(1.6e-3, 1.6)
    axR.set_xlabel("planning horizon $H$", fontsize=7.6, labelpad=2.0)
    axR.set_ylabel("change per extra planned step", fontsize=7.6, labelpad=2.0)
    axR.tick_params(labelsize=7.0, length=2.4, pad=1.6)
    axR.grid(True, lw=0.4, color="#dde5ee")
    axR.set_axisbelow(True)
    for side in ("left", "bottom"):
        axR.spines[side].set_linewidth(0.8)
        axR.spines[side].set_color("#7f8c99")
    axR.set_title("Marginal value bought\nversus model error paid",
                  fontsize=8.4, fontweight="bold", color=INK, pad=4.0,
                  linespacing=1.35)

    axR.text(0.045, 0.955, "each crossing is the ringed\noptimum in panel (a)",
             transform=axR.transAxes, ha="left", va="top", fontsize=6.8,
             color=ACCENT, fontweight="bold", zorder=9, linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#d9c4c1",
                       lw=0.7, alpha=0.96))
    axR.text(25.5, 0.0053, "$\\Delta$gain($H$)", ha="center", va="top",
             fontsize=6.9, color=INK, fontweight="bold", zorder=9,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.94))
    axR.text(26.0, 0.62, "$\\Delta$cost($H,\\epsilon$)", ha="center",
             va="center", fontsize=6.9, color=BLUE[1], zorder=9,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.94))

    savefig(fig, "1208_wm_plan_horizon_tradeoff.png")


if __name__ == "__main__":
    main()
