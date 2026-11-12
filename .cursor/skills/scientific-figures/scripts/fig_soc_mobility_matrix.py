"""Intergenerational class mobility as an outflow matrix with its margins.

The main panel is the five-by-five origin-by-destination table, row-normalised
so each row reads as "of fathers in this class, where did the sons end up".
The diagonal carries the inheritance the table exists to show, so it is left to
speak for itself rather than being annotated.  The two marginal strips are the
origin distribution the table was weighted with and the destination
distribution obtained by multiplying it through the matrix.  Everything in the
supporting panels - upward, downward and immobile shares, and the total
mobility rate per birth cohort - is a weighted sum over those same matrices.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
BAR = "#5f9dc6"
SEQ = get_cmap("cb_sequential_blue")

FW, FH = 10.4, 6.4
HEAT = (1.12, 1.02, 3.72, 3.72)
TOPS = (1.12, 4.86, 3.72, 0.40)
RGHT = (4.94, 1.02, 0.40, 3.72)
AUX_X, AUX_W, AUX_H = 7.12, 2.90, 1.98
AUX_TOP = (AUX_X, 3.42, AUX_W, AUX_H)
AUX_BOT = (AUX_X, 0.84, AUX_W, AUX_H)

CLASSES = ["I", "II", "III", "IV", "V"]
ORIGIN = np.array([9.0, 17.0, 28.0, 27.0, 19.0])     # % of fathers
TAU = 1.05          # smaller = stickier class structure
STICKY = np.array([2.30, 1.35, 1.15, 1.25, 1.75])    # per-class inheritance
DRIFT = 0.24        # net upward drift from occupational upgrading
COHORTS = np.array([1935, 1945, 1955, 1965, 1975, 1985])


def outflow(tau=TAU, drift=DRIFT, sticky=STICKY):
    i = np.arange(5)[:, None]
    j = np.arange(5)[None, :]
    raw = np.exp(-np.abs(i - j) / tau) * np.exp(drift * (i - j) / 4.0)
    raw = raw * np.where(i == j, sticky[:, None], 1.0)
    return 100.0 * raw / raw.sum(axis=1, keepdims=True)


def shares(m, origin=ORIGIN):
    w = origin / origin.sum()
    i = np.arange(5)[:, None]
    j = np.arange(5)[None, :]
    up = float((w[:, None] * m * (j < i)).sum()) / 100.0
    down = float((w[:, None] * m * (j > i)).sum()) / 100.0
    same = float((w * np.diag(m)).sum()) / 100.0
    return 100.0 * np.array([up, same, down])


def cohort_series():
    taus = np.array([0.86, 0.97, 1.12, 1.27, 1.36, 1.40])
    mats = [outflow(tau=t, sticky=STICKY * np.linspace(1.10, 0.86,
                                                       COHORTS.size)[k])
            for k, t in enumerate(taus)]
    return mats, np.array([100.0 - shares(m)[1] for m in mats])


def heat_panel(fig, ax, m):
    im = ax.imshow(m, cmap=SEQ, vmin=0.0, vmax=float(m.max()))
    ax.set_aspect("equal")
    for i in range(5):
        for j in range(5):
            v = m[i, j]
            ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=7.6,
                    color="white" if v > 0.55 * m.max() else INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(CLASSES, fontsize=7.4)
    ax.set_yticklabels(CLASSES, fontsize=7.4)
    ax.set_xlabel("son's class (destination)")
    ax.set_ylabel("father's class (origin)")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#cccccc")
    ax.tick_params(length=0)
    safe_colorbar(fig, im, ax, label="outflow within row (%)", width=0.12,
                  pad=0.66, labelsize=6.8, ticks=[0, 20, 40, 60, 80])
    return im


def margin_panels(ax_top, ax_right, m, origin=ORIGIN):
    dest = origin @ (m / 100.0)
    ax_top.bar(np.arange(5), dest, width=0.72, color=BAR, alpha=0.95,
               edgecolor="white", linewidth=0.7)
    for j, v in enumerate(dest):
        ax_top.text(j, v + 0.06 * dest.max(), "%.0f" % v, ha="center",
                    va="bottom", fontsize=6.3, color=SUB)
    ax_top.set_xlim(-0.5, 4.5)
    ax_top.set_ylim(0.0, dest.max() * 1.55)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)
    ax_top.text(-0.62, dest.max() * 0.7, "destination\nshare (%)", ha="right",
                va="center", fontsize=6.3, color=MUTED, linespacing=1.3)

    ax_right.barh(np.arange(5), origin, height=0.72, color=BAR, alpha=0.95,
                  edgecolor="white", linewidth=0.7)
    for i, v in enumerate(origin):
        ax_right.text(v + 0.06 * origin.max(), i, "%.0f" % v, ha="left",
                      va="center", fontsize=6.3, color=SUB)
    ax_right.set_ylim(4.5, -0.5)
    ax_right.set_xlim(0.0, origin.max() * 1.62)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for s in ax_right.spines.values():
        s.set_visible(False)
    ax_right.text(origin.max() * 0.1, 4.78, "origin\nshare (%)", ha="left",
                  va="top", fontsize=6.3, color=MUTED, linespacing=1.3)
    return dest


def flow_panel(ax, m):
    vals = shares(m)
    names = ["upward", "immobile", "downward"]
    cols = [BLUE, GREY, RED]
    ax.bar(np.arange(3), vals, width=0.56, color=cols, alpha=0.92,
           edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, "%.1f %%" % v, ha="center", va="bottom",
                fontsize=7.0, color=INK)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylim(0.0, vals.max() * 1.30)
    ax.set_ylabel("share of all sons (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Where sons end up relative to their father", fontsize=8.2,
                 fontweight="bold", color=INK)
    return vals


def index_panel(ax, idx):
    ax.plot(COHORTS, idx, marker="o", ms=4.2, lw=1.7, color=BLUE,
            mfc="white", mew=1.2, zorder=3)
    b, a = np.polyfit(COHORTS, idx, 1)
    ax.plot(COHORTS, a + b * COHORTS, color=GREY, lw=1.0, ls=(0, (4.0, 2.4)),
            zorder=2)
    ax.text(0.04, 0.94, "%+.1f points per decade" % (b * 10.0),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB)
    ax.set_xticks(COHORTS)
    ax.set_xlabel("birth cohort (year)", fontsize=7.4)
    ax.set_ylabel("total mobility rate (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Fluidity by birth cohort", fontsize=8.2, fontweight="bold",
                 color=INK)
    return b


def main():
    set_pub_style()
    m = outflow()
    mats, idx = cohort_series()

    fig = plt.figure(figsize=(FW, FH))
    ax_heat = fig.add_axes([HEAT[0] / FW, HEAT[1] / FH,
                            HEAT[2] / FW, HEAT[3] / FH])
    ax_top = fig.add_axes([TOPS[0] / FW, TOPS[1] / FH,
                           TOPS[2] / FW, TOPS[3] / FH])
    ax_rgt = fig.add_axes([RGHT[0] / FW, RGHT[1] / FH,
                           RGHT[2] / FW, RGHT[3] / FH])
    ax_flow = fig.add_axes([AUX_TOP[0] / FW, AUX_TOP[1] / FH,
                            AUX_TOP[2] / FW, AUX_TOP[3] / FH])
    ax_idx = fig.add_axes([AUX_BOT[0] / FW, AUX_BOT[1] / FH,
                           AUX_BOT[2] / FW, AUX_BOT[3] / FH])

    heat_panel(fig, ax_heat, m)
    margin_panels(ax_top, ax_rgt, m)
    vals = flow_panel(ax_flow, m)
    index_panel(ax_idx, idx)

    ax_top.set_title("Outflow table: the diagonal is class inheritance",
                     fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    panel_labels_inches(fig, [ax_top, ax_flow, ax_idx], dx=-0.70, dy=0.10)
    savefig(fig, "808_soc_mobility_matrix.png")


if __name__ == "__main__":
    main()
