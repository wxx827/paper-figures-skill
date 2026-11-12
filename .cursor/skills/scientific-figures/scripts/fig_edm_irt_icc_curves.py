"""Three-parameter IRT: what a, b and c each do to an item characteristic curve.

Six items are defined once by a single (a, b, c) table and every curve,
marker and number on the page comes out of it.  The main panel puts each
item's leader line at P = 0.5 + c/2 rather than at 0.5, because that is where
theta = b actually sits once a lower asymptote is allowed - drawing it at 0.5
is the standard way this figure lies.  One deliberately flat item (a = 0.45)
is dashed so the cost of low discrimination is visible as a shape, not only
as a number in the legend.  The parameter map and the item information curves
re-use the same table, so a point in the quality box and a peak in the
information panel can never disagree.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
ACCENT = "#b23a48"

FIGW, FIGH = 10.4, 6.4
MAIN = (0.78, 1.05, 5.45, 4.35)
AUX_T = (7.05, 3.62, 3.00, 1.78)
AUX_B = (7.05, 1.05, 3.00, 1.78)

THETA_LO, THETA_HI = -4.0, 4.0

# The six exemplar items: discrimination a, difficulty b (logits), guessing c.
A_PAR = np.array([1.65, 1.10, 1.85, 0.45, 1.30, 0.95])
B_PAR = np.array([-1.90, -0.85, 0.15, 0.70, 1.55, 2.40])
C_PAR = np.array([0.18, 0.05, 0.22, 0.20, 0.12, 0.25])
FLAT = 3                      # the low-discrimination item, drawn dashed
N_POOL = 42                   # the bank the six exemplars are drawn from
A_OK = (0.80, 2.50)           # usable discrimination window
B_OK = (-2.50, 2.50)          # usable difficulty window (logits)


def icc(theta, a, b, c):
    """3PL probability of a correct response."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def item_info(theta, a, b, c):
    """3PL item information; reduces to a^2 P Q when c = 0."""
    p = icc(theta, a, b, c)
    q = 1.0 - p
    return a ** 2 * (q / p) * ((p - c) / (1.0 - c)) ** 2


def bank(seed=894):
    """The rest of the item bank behind the six exemplars."""
    rng = np.random.default_rng(seed)
    n = N_POOL - A_PAR.size
    b = np.clip(rng.normal(0.05, 1.20, n), -3.2, 3.2)
    a = np.clip(rng.gamma(4.5, 0.27, n), 0.22, 2.65)
    c = np.clip(rng.beta(2.4, 12.0, n), 0.0, 0.34)
    return a, b, c


def main():
    set_pub_style()
    theta = np.linspace(THETA_LO, THETA_HI, 801)
    probs = np.array([icc(theta, a, b, c)
                      for a, b, c in zip(A_PAR, B_PAR, C_PAR)])
    infos = np.array([item_info(theta, a, b, c)
                      for a, b, c in zip(A_PAR, B_PAR, C_PAR)])
    p_at_b = 0.5 + C_PAR / 2.0

    a_pool, b_pool, c_pool = bank()
    a_all = np.concatenate([A_PAR, a_pool])
    b_all = np.concatenate([B_PAR, b_pool])
    inside = ((a_all >= A_OK[0]) & (a_all <= A_OK[1]) &
              (b_all >= B_OK[0]) & (b_all <= B_OK[1]))
    n_in = int(inside.sum())

    # Dark blue = hardest item; the flat item keeps its ramp colour but is
    # dashed, so line style carries discrimination and hue carries difficulty.
    ramp = get_palette("cb_sequential_blue", 8)
    order = np.argsort(B_PAR)
    col = {int(k): ramp[i + 2] for i, k in enumerate(order)}

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax_icc = axes_inches(fig, MAIN)
    ax_par = axes_inches(fig, AUX_T)
    ax_inf = axes_inches(fig, AUX_B)

    # ---- main: the six characteristic curves -----------------------------
    for i in range(A_PAR.size):
        c_i = col[i]
        flat = (i == FLAT)
        ax_icc.plot(theta, probs[i], color=c_i, lw=2.1 if flat else 1.8,
                    ls=(0, (4.2, 2.0)) if flat else "-", zorder=5 if flat else 4,
                    label=(f"item {i + 1}   a = {A_PAR[i]:.2f},  "
                           f"b = {B_PAR[i]:+.2f},  c = {C_PAR[i]:.2f}"))
        ax_icc.plot([B_PAR[i], B_PAR[i]], [C_PAR[i], p_at_b[i]], color=c_i,
                    lw=0.8, ls=(0, (2.0, 1.7)), alpha=0.85, zorder=2)
        ax_icc.plot([B_PAR[i]], [p_at_b[i]], marker="o", ms=4.6, mfc="white",
                    mec=c_i, mew=1.25, zorder=6)
        ax_icc.plot([B_PAR[i]], [0.0], marker="^", ms=3.6, color=c_i,
                    clip_on=False, zorder=6)
        ax_icc.plot([THETA_LO, B_PAR[i]], [C_PAR[i], C_PAR[i]], color=c_i,
                    lw=0.6, ls=(0, (1.2, 1.9)), alpha=0.55, zorder=1)

    k = 2
    ax_icc.annotate(
        f"theta = b at P = 0.5 + c/2 = {p_at_b[k]:.3f}",
        xy=(B_PAR[k], p_at_b[k]), xytext=(B_PAR[k] + 0.75, p_at_b[k] - 0.20),
        fontsize=6.9, color=INK, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=col[k], lw=0.8,
                        shrinkA=1.0, shrinkB=3.0))
    ax_icc.annotate(
        "lower asymptote c: the floor a\nnon-zero guessing rate leaves",
        xy=(THETA_LO + 0.10, C_PAR[5]), xytext=(-3.86, 0.44),
        fontsize=6.7, color=SUB, ha="left", va="bottom", linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                        shrinkA=2.0, shrinkB=1.0))
    # Width of the 10 % -> 90 % rise, straight from the logistic: 2 ln(9) / a.
    rise = 2.0 * np.log(9.0) / A_PAR
    ax_icc.text(3.86, 0.05,
                f"item {FLAT + 1} is dashed: a = {A_PAR[FLAT]:.2f} stretches the\n"
                f"10-90 % rise over {rise[FLAT]:.1f} logits, against\n"
                f"{rise[2]:.1f} logits for item 3 (a = {A_PAR[2]:.2f})",
                ha="right", va="bottom", fontsize=6.7, color=ACCENT,
                linespacing=1.35)

    ax_icc.set_xlim(THETA_LO, THETA_HI)
    ax_icc.set_ylim(0.0, 1.02)
    ax_icc.set_xticks(np.arange(-4, 5))
    ax_icc.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax_icc.set_xlabel("latent ability theta (logits, SD units)", fontsize=8.0)
    ax_icc.set_ylabel("probability of a correct response P (-)", fontsize=8.0)
    ax_icc.tick_params(labelsize=7.2)
    ax_icc.grid(axis="y", lw=0.5, color="#e6e6e6")
    ax_icc.set_title("Item characteristic curves: b slides the curve, a steepens "
                     "it, c lifts its floor",
                     fontsize=9.0, fontweight="bold", color=INK, pad=10.0)
    ax_icc.legend(loc="upper left", frameon=False, fontsize=6.8,
                  handlelength=1.8, labelspacing=0.32, borderaxespad=0.5)

    # ---- aux (b): the parameter map --------------------------------------
    ax_par.add_patch(Rectangle((B_OK[0], A_OK[0]), B_OK[1] - B_OK[0],
                               A_OK[1] - A_OK[0], facecolor="#eef4f9",
                               edgecolor=HAIR, lw=0.7, zorder=0))
    ax_par.scatter(b_pool, a_pool, s=11, facecolor="#c8d4e2",
                   edgecolor="#8fa4bd", linewidth=0.4, zorder=2,
                   label=f"rest of the bank (n = {b_pool.size} items)")
    for i in range(A_PAR.size):
        ax_par.scatter([B_PAR[i]], [A_PAR[i]], s=34, facecolor=col[i],
                       edgecolor="white", linewidth=0.7, zorder=4)
        ax_par.annotate(f"{i + 1}", xy=(B_PAR[i], A_PAR[i]),
                        xytext=(0.0, 6.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.2, color=col[i])
    ax_par.scatter([B_PAR[FLAT]], [A_PAR[FLAT]], s=86, facecolor="none",
                   edgecolor=ACCENT, linewidth=1.0, zorder=5)
    ax_par.axhline(A_OK[0], color=GREY, lw=0.8, ls=(0, (3.2, 2.0)), zorder=3)
    ax_par.text(-3.85, A_OK[0] - 0.10, f"a = {A_OK[0]:.2f} screening floor",
                ha="left", va="top", fontsize=6.2, color=GREY)
    ax_par.text(0.0, A_OK[1] - 0.13,
                f"usable window: {n_in} of {a_all.size} items",
                ha="center", va="top", fontsize=6.4, color=SUB)
    ax_par.set_xlim(THETA_LO, THETA_HI)
    ax_par.set_ylim(0.0, 2.85)
    ax_par.set_xticks(np.arange(-4, 5, 2))
    ax_par.set_xlabel("difficulty b (logits)", fontsize=7.4)
    ax_par.set_ylabel("discrimination a (-)", fontsize=7.4)
    ax_par.tick_params(labelsize=6.6)
    ax_par.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_par.set_title("Parameter map of the item bank", fontsize=8.5,
                     fontweight="bold", color=INK)
    ax_par.legend(loc="lower left", frameon=False, fontsize=6.2,
                  handlelength=1.0, borderaxespad=0.35)

    # ---- aux (c): item information ---------------------------------------
    for i in range(A_PAR.size):
        ax_inf.plot(theta, infos[i], color=col[i], lw=1.5,
                    ls=(0, (4.2, 2.0)) if i == FLAT else "-")
        j = int(np.argmax(infos[i]))
        ax_inf.plot([theta[j]], [infos[i][j]], marker="o", ms=3.0,
                    mfc="white", mec=col[i], mew=0.9)
    best = int(np.argmax(infos.max(axis=1)))
    ax_inf.annotate(f"item {best + 1}: peak I = {infos[best].max():.2f}",
                    xy=(theta[int(np.argmax(infos[best]))],
                        infos[best].max()),
                    xytext=(6.0, 1.0), textcoords="offset points",
                    fontsize=6.4, color=col[best], ha="left", va="bottom")
    ax_inf.set_xlim(THETA_LO, THETA_HI)
    ax_inf.set_ylim(0.0, float(infos.max()) * 1.30)
    ax_inf.set_xticks(np.arange(-4, 5, 2))
    ax_inf.set_xlabel("theta (logits)", fontsize=7.4)
    ax_inf.set_ylabel("item information I (-)", fontsize=7.4)
    ax_inf.tick_params(labelsize=6.6)
    ax_inf.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_inf.set_title("Each item informs only near its own b", fontsize=8.5,
                     fontweight="bold", color=INK)
    ax_inf.text(0.985, 0.93, "peak height rises with a, falls with c",
                transform=ax_inf.transAxes, ha="right", va="top",
                fontsize=6.3, color=MUTED)


    note = (
        "Three-parameter logistic model P(theta) = c + (1 - c) / "
        "(1 + exp(-a (theta - b))), plotted on 801 grid points over theta = "
        f"{THETA_LO:+.0f} to {THETA_HI:+.0f} logits. The six exemplar items "
        f"carry a = " + " / ".join(f"{v:.2f}" for v in A_PAR) +
        ", b = " + " / ".join(f"{v:+.2f}" for v in B_PAR) +
        " logits and c = " + " / ".join(f"{v:.2f}" for v in C_PAR) +
        f"; the open circle on each curve is theta = b, where P = 0.5 + c/2 "
        f"(here {p_at_b.min():.3f} to {p_at_b.max():.3f}), not 0.5. Panel (b) "
        f"places them in a bank of {a_all.size} items and panel (c) shows "
        "item information I(theta) = a^2 (Q/P) [(P - c)/(1 - c)]^2 computed "
        "from the same table; item "
        f"{FLAT + 1} contributes a peak of only {infos[FLAT].max():.2f} "
        f"against {infos[best].max():.2f} for item {best + 1}.")
    fig.text(0.5, 0.16 / FIGH, textwrap.fill(note, 178), ha="center",
             va="bottom", fontsize=6.5, color=MUTED, linespacing=1.5)

    panel_labels_inches(fig, [ax_icc, ax_par, ax_inf],
                        labels=["(a)", "(b)", "(c)"], dx=-0.56, dy=0.12)
    savefig(fig, "894_edm_irt_icc_curves.png")


if __name__ == "__main__":
    main()
