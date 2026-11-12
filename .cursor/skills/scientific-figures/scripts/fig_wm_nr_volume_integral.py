"""Figure 1155 - The volume rendering integral, one term at a time.

Every panel is a projection of the SAME per-sample table built in ``samples``:

    sigma_i (density)  ->  alpha_i = 1 - exp(-sigma_i * delta)
                       ->  T_i     = prod_{j<i} (1 - alpha_j)
                       ->  w_i     = T_i * alpha_i

(a) draws c_i as swatches on the ray, (b) draws alpha_i, (c) draws w_i with
T_i over it, and (d) stacks segments of height w_i tinted c_i - so the stack in
(d) IS the bar chart in (c), and the composite swatch at its foot is literally
``sum_i w_i c_i + T_last * c_bg``, recomputed from the same arrays.

Touch DENSITY or COL_* and the swatches, the bars, the stack, the composite
and every printed number move together; nothing here is a hand-typed constant.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from pubstyle import set_pub_style, savefig, add_panel_label
from figkit import background_card

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# schematics like this one run sparse, so the backing cards use the deeper of
# the two house card greys (DESIGN_SYSTEM 2.1) to carry the empty corners
HAIR, CARD = "#d5d5d5", "#f0f0f0"
ALPHA_C, WEIGHT_C, TRANS_C = "#5f9dc6", "#2e74a8", "#134a76"
SEED = 1560  # original id; figure number is 1155

N = 14
COL_FAR = np.array([0.760, 0.812, 0.868])      # pale wall / sky
COL_OBJ = np.array([0.176, 0.318, 0.451])      # steel-blue subject
COL_HAZE = np.array([0.404, 0.541, 0.588])     # teal participating medium
COL_BG = np.array([0.898, 0.925, 0.949])       # what is left behind the ray


def samples():
    rng = np.random.default_rng(SEED)
    t = np.linspace(0.06, 0.98, N)
    delta = float(t[1] - t[0])

    sigma = (0.45
             + 22.0 * np.exp(-((t - 0.520) / 0.082) ** 2)
             + 6.5 * np.exp(-((t - 0.930) / 0.045) ** 2)
             + 1.9 * np.exp(-((t - 0.255) / 0.060) ** 2))
    alpha = 1.0 - np.exp(-sigma * delta)
    trans = np.concatenate([[1.0], np.cumprod(1.0 - alpha)])
    w = trans[:-1] * alpha

    obj = np.exp(-((t - 0.520) / 0.115) ** 2)
    haze = np.exp(-((t - 0.255) / 0.150) ** 2)
    col = (COL_FAR[None, :] * (1.0 - obj - 0.55 * haze)[:, None]
           + COL_OBJ[None, :] * obj[:, None]
           + COL_HAZE[None, :] * (0.55 * haze)[:, None])
    col = np.clip(col + rng.normal(0.0, 0.016, col.shape), 0.0, 1.0)
    return t, delta, sigma, alpha, trans, w, col


def main():
    set_pub_style(base_font=8.5)
    t, delta, sigma, alpha, trans, w, col = samples()
    idx = np.arange(1, N + 1)
    k = int(np.argmax(w))
    acc = float(w.sum())
    resid = float(trans[-1])
    pix = (w[:, None] * col).sum(axis=0) + resid * COL_BG
    top3 = np.argsort(w)[::-1][:3]

    # Calculator closure.  Only four weights are legible on the page: the
    # three labelled bars and the total.  Percentages are therefore taken
    # from the 2-dp values a reader can actually read off, over the printed
    # total - so 0.36 + 0.19 + 0.12 gives the 67 % in the header, and
    # 0.99 + 0.01 gives the 1.00 those percentages are fractions of.
    # (Summing all fourteen rounded w_i instead would lose 0.02 to rounding.)
    w2 = np.round(w, 2)
    acc2, resid2 = round(acc, 2), round(resid, 2)
    total2 = acc2 + resid2

    fig = plt.figure(figsize=(10.4, 6.3))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.28, 1.0],
                          height_ratios=[0.60, 1.0, 1.0], left=0.062,
                          right=0.972, top=0.842, bottom=0.212, wspace=0.20,
                          hspace=0.30)
    axs = fig.add_subplot(gs[0, 0])
    axa = fig.add_subplot(gs[1, 0])
    axw = fig.add_subplot(gs[2, 0])
    axc = fig.add_subplot(gs[:, 1])

    # ---- (a) the ray and its sample colours -------------------------------
    axs.set_facecolor(CARD)
    axs.plot([0.30, N + 0.8], [0.0, 0.0], color="#20374f", lw=1.0, zorder=2)
    for i in range(N):
        axs.add_patch(Rectangle((idx[i] - 0.34, -0.34), 0.68, 0.68,
                                facecolor=col[i], edgecolor="#3a3a3a",
                                lw=0.55, zorder=4))
    axs.annotate("", xy=(N + 1.15, 0.0), xytext=(N + 0.75, 0.0), zorder=5,
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#20374f",
                                 mutation_scale=9))
    axs.text(-0.85, 0.0, "camera", ha="left", va="center", fontsize=7.0,
             color=SUB, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.92))
    axs.text(N + 1.30, 0.0, "$t_f$", ha="left", va="center", fontsize=7.4,
             color="#20374f", zorder=5)
    axs.set_xlim(-0.9, N + 1.9)
    axs.set_ylim(-0.62, 0.62)
    axs.set_yticks([])
    axs.set_xticks([])
    for s in axs.spines.values():
        s.set_visible(False)
    axs.set_ylabel("$\\mathbf{c}_i$", fontsize=9.0, rotation=0, labelpad=10,
                   va="center")
    axs.set_title(f"{N} samples on one ray: colour, opacity, weight",
                  fontsize=9.0, fontweight="bold", pad=4.0)
    add_panel_label(axs, "(a)", x=-0.052, y=0.90)

    # ---- (b) opacity ------------------------------------------------------
    axa.bar(idx, alpha, width=0.68, color=ALPHA_C, edgecolor="white",
            linewidth=0.5, zorder=3)
    for i in top3:
        axa.text(idx[i], alpha[i] + 0.035, f"{alpha[i]:.2f}", ha="center",
                 va="bottom", fontsize=6.6, color=TRANS_C, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.92))
    axa.set_xlim(-0.9, N + 1.9)
    axa.set_ylim(0.0, 1.06)
    axa.set_xticks([])
    axa.set_yticks([0.0, 0.5, 1.0])
    axa.set_ylabel("$\\alpha_i = 1-e^{-\\sigma_i\\delta}$", fontsize=8.0)
    axa.tick_params(labelsize=7.0, length=2.4)
    axa.grid(True, axis="y", lw=0.45, color="#ececec")
    axa.set_axisbelow(True)
    axa.text(0.985, 0.93, f"$\\delta$ = {delta:.3f} along a ray of length "
             f"{t[-1] - t[0] + delta:.2f}", transform=axa.transAxes,
             ha="right", va="top", fontsize=6.8, color=SUB, zorder=6,
             bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.92))
    add_panel_label(axa, "(b)", x=-0.052, y=1.02)

    # ---- (c) weights and transmittance ------------------------------------
    axw.step(np.concatenate([idx - 0.5, [N + 0.5]]),
             np.concatenate([trans[:-1], trans[-1:]]), where="post",
             color=TRANS_C, lw=1.4, zorder=4, label="$T_i$")
    axw.bar(idx, w, width=0.68, color=WEIGHT_C, edgecolor="white",
            linewidth=0.5, zorder=5, label="$w_i = T_i\\,\\alpha_i$")
    axw.bar(idx[k], w[k], width=0.68, color="#0b3d63", edgecolor="white",
            linewidth=0.5, zorder=6)
    axw.annotate(f"surface: sample {idx[k]}, $w$ = {w[k]:.2f}\n"
                 f"({100 * w2[k] / total2:.0f} % of the pixel)",
                 xy=(idx[k], w[k]), xytext=(idx[k] + 1.05, 0.86), fontsize=6.9,
                 color="#0b3d63", ha="left", va="center", linespacing=1.5,
                 zorder=8,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white",
                           ec="#b9c9d8", lw=0.6, alpha=0.94),
                 arrowprops=dict(arrowstyle="-", lw=0.8, color="#0b3d63",
                                 shrinkA=2, shrinkB=4))
    axw.set_xlim(-0.9, N + 1.9)
    axw.set_ylim(0.0, 1.06)
    axw.set_xticks(idx)
    axw.set_xticklabels([str(v) for v in idx], fontsize=7.0)
    axw.set_yticks([0.0, 0.5, 1.0])
    axw.set_xlabel("sample index $i$ along the ray", fontsize=8.0)
    axw.set_ylabel("$w_i$   and   $T_i$", fontsize=8.0)
    axw.tick_params(labelsize=7.0, length=2.4)
    axw.grid(True, axis="y", lw=0.45, color="#ececec")
    axw.set_axisbelow(True)
    leg = axw.legend(loc="upper left", bbox_to_anchor=(0.015, 0.86),
                     fontsize=7.0, frameon=True, facecolor="white",
                     edgecolor=HAIR, handlelength=1.5, borderpad=0.35,
                     labelspacing=0.3)
    leg.get_frame().set_linewidth(0.5)
    add_panel_label(axw, "(c)", x=-0.052, y=1.02)

    # ---- (d) the accumulation ---------------------------------------------
    axc.set_facecolor(CARD)
    x0, wid = 0.135, 0.30
    base = 0.0
    for i in range(N):
        axc.add_patch(Rectangle((x0, base), wid, w[i], facecolor=col[i],
                                edgecolor="white", lw=0.45, zorder=4))
        base += w[i]
    axc.add_patch(Rectangle((x0, base), wid, resid, facecolor=COL_BG,
                            edgecolor="white", lw=0.45, hatch="////",
                            zorder=4))
    axc.add_patch(Rectangle((x0, 0.0), wid, 1.0, facecolor="none",
                            edgecolor="#3a3a3a", lw=0.8, zorder=6))
    for i in top3:
        yy = float(w[:i].sum() + 0.5 * w[i])
        axc.annotate(f"$w_{{{idx[i]}}}$ = {w[i]:.2f}", xy=(x0 + wid, yy),
                     xytext=(x0 + wid + 0.10, yy), fontsize=6.8, color=SUB,
                     ha="left", va="center", zorder=7,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                               alpha=0.92),
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#9aa7b2",
                                     shrinkA=1, shrinkB=1))
    axc.annotate(f"$T_{{{N + 1}}}$ = {resid:.2f}, background",
                 xy=(x0 + wid, base + 0.5 * resid),
                 xytext=(x0 + wid + 0.10, 0.945), ha="left", va="center",
                 fontsize=6.8, color=SUB, zorder=7,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.92),
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#9aa7b2",
                                 shrinkA=1, shrinkB=1))
    axc.text(x0 - 0.050, 0.5, "accumulated opacity", ha="right", va="center",
             fontsize=7.4, color=SUB, rotation=90, zorder=7)

    sw = 0.30
    axc.add_patch(Rectangle((x0, -0.335), sw, 0.185, facecolor=pix,
                            edgecolor="#3a3a3a", lw=0.8, zorder=5))
    axc.add_patch(FancyArrowPatch((x0 + sw / 2, -0.035),
                                  (x0 + sw / 2, -0.125), arrowstyle="-|>",
                                  mutation_scale=10, lw=1.0, color="#20374f",
                                  zorder=6))
    axc.text(x0 + sw + 0.055, -0.243, "rendered pixel\n"
             f"({pix[0]:.2f}, {pix[1]:.2f}, {pix[2]:.2f})", ha="left",
             va="center", fontsize=7.0, color=INK, linespacing=1.5, zorder=7)
    axc.set_xlim(0.0, 1.0)
    axc.set_ylim(-0.42, 1.02)
    axc.set_xticks([])
    axc.set_yticks([])
    for s in axc.spines.values():
        s.set_visible(False)
    axc.set_title(f"{100 * acc2:.0f} % of the pixel is opaque",
                  fontsize=9.0, fontweight="bold", pad=4.0)
    add_panel_label(axc, "(d)", x=-0.045, y=1.005)

    # ---- headline + closing equation --------------------------------------

    background_card(fig, (0.062, 0.032, 0.910, 0.118), facecolor=CARD,
                    edgecolor=HAIR, zorder=-5, radius=0.010)
    fig.text(0.092, 0.112, r"$\hat{C}(\mathbf{r})=\sum_{i=1}^{" + str(N) +
             r"}$", fontsize=11.0, color=INK, ha="left", va="center")
    fig.text(0.168, 0.112, r"$T_i$", fontsize=11.0, color=TRANS_C, ha="left",
             va="center")
    fig.text(0.194, 0.112, r"$\alpha_i$", fontsize=11.0, color=ALPHA_C,
             ha="left", va="center")
    fig.text(0.220, 0.112, r"$\mathbf{c}_i\;+\;$", fontsize=11.0, color=INK,
             ha="left", va="center")
    fig.text(0.278, 0.112, r"$T_{" + str(N + 1) + r"}\,\mathbf{c}_{bg}$",
             fontsize=11.0, color=TRANS_C, ha="left", va="center")
    fig.text(0.372, 0.112, f"$=\\;({pix[0]:.2f},\\,{pix[1]:.2f},\\,"
             f"{pix[2]:.2f})$", fontsize=11.0, color=INK, ha="left",
             va="center")
    savefig(fig, "1155_wm_nr_volume_integral.png")


if __name__ == "__main__":
    main()
