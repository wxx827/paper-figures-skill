"""Diverging stacked bars for a nine-item Likert battery.

One simulated respondent table drives all three panels: a latent attitude per
person and item is cut at fixed thresholds into the five response categories,
and item non-response is drawn separately.  The main panel stacks those
categories so that the neutral segment straddles zero - the left edge of each
bar is minus the two negative shares minus half of neutral, which is the only
alignment that lets a reader compare disagreement across items by eye.  Items
are ordered by net agreement, and the net column on the right is recomputed
from the same stacked segments.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"

FW, FH = 10.4, 6.4
MAIN = (1.74, 0.84, 4.06, 4.30)
AUX_X, AUX_W, AUX_H = 7.12, 2.90, 1.90
AUX_TOP = (AUX_X, 3.24, AUX_W, AUX_H)
AUX_BOT = (AUX_X, 0.84, AUX_W, AUX_H)

CATS = ["strongly disagree", "disagree", "neutral", "agree",
        "strongly agree"]
COLORS = [darken(RED, 0.22), lighten(RED, 0.34), "#cccccc",
          lighten(BLUE, 0.34), darken(BLUE, 0.16)]
CUTS = np.array([-1.15, -0.38, 0.38, 1.15])

ITEMS = [("Q1", "government listens", -0.55, 0.075),
         ("Q2", "transport meets my needs", -0.10, 0.032),
         ("Q3", "safe after dark", 0.28, 0.024),
         ("Q4", "housing costs manageable", -0.92, 0.061),
         ("Q5", "schools prepare children", 0.34, 0.086),
         ("Q6", "health services reachable", 0.12, 0.028),
         ("Q7", "enough green space", 0.66, 0.019),
         ("Q8", "recycling is easy", 0.48, 0.022),
         ("Q9", "would recommend living here", 0.80, 0.030)]
N_RESP = 1200
GROUPS = ["age 18-34", "age 35-59", "age 60+"]
G_COLORS = ["#8e6bbf", "#c9a227", "#2e7d5b"]
G_EFFECT = np.array([0.42, 0.0, -0.38])
G_SHARE = [0.34, 0.36, 0.30]


def survey(seed=809):
    """Respondent-level simulation; returns per-item shares, nets and gaps."""
    rng = np.random.default_rng(seed)
    grp = rng.choice(3, size=N_RESP, p=G_SHARE)
    shares, gnets, nores = [], [], []
    for _, _, mu, nr in ITEMS:
        z = mu + G_EFFECT[grp] + rng.normal(0.0, 1.0, N_RESP)
        cat = np.digitize(z, CUTS)
        miss = rng.random(N_RESP) < nr
        ok = ~miss
        sh = np.bincount(cat[ok], minlength=5) / ok.sum() * 100.0
        shares.append(sh)
        gn = []
        for g in range(3):
            m = ok & (grp == g)
            s = np.bincount(cat[m], minlength=5) / m.sum() * 100.0
            gn.append(s[3] + s[4] - s[0] - s[1])
        gnets.append(gn)
        nores.append(miss.mean() * 100.0)
    return (np.asarray(shares), np.asarray(gnets), np.asarray(nores),
            np.bincount(grp, minlength=3))


def diverging_panel(ax, shares, order, labels):
    p = shares[order]
    n = p.shape[0]
    y = np.arange(n, dtype=float)
    left0 = -(p[:, 0] + p[:, 1] + p[:, 2] / 2.0)
    lefts = left0[:, None] + np.concatenate(
        [np.zeros((n, 1)), np.cumsum(p, axis=1)[:, :-1]], axis=1)

    for k in range(5):
        ax.barh(y, p[:, k], left=lefts[:, k], height=0.64, color=COLORS[k],
                edgecolor="white", linewidth=0.7, label=CATS[k], zorder=3)
        for i in range(n):
            if p[i, k] < 7.0:
                continue
            t = ax.text(lefts[i, k] + p[i, k] / 2.0, y[i], "%.0f" % p[i, k],
                        ha="center", va="center", fontsize=6.2, zorder=5,
                        color="white" if k in (0, 4) else INK)
            if k == 2:
                # the neutral figure sits on the zero rule; halo it in the
                # segment's own colour so the digits stay readable
                t.set_path_effects([pe.withStroke(linewidth=2.0,
                                                  foreground=COLORS[2])])

    ax.axvline(0.0, color="#333333", lw=1.0, ls=(0, (3.0, 2.2)), zorder=4)
    net = p[:, 3] + p[:, 4] - p[:, 0] - p[:, 1]
    # the net column lives just outside the data area, like a table column
    for i, v in enumerate(net):
        ax.text(93.0, y[i], "%+.0f" % v, ha="right", va="center",
                fontsize=7.0, color=BLUE if v > 0 else RED, clip_on=False)
    ax.text(93.0, n - 0.32, "net (pp)", ha="right", va="bottom",
            fontsize=6.5, color=MUTED, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(-0.72, n - 0.28)
    ax.set_xlim(-84.0, 84.0)
    ax.set_xticks(np.arange(-80, 81, 20))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % abs(v)))
    ax.set_xlabel("share of valid responses (%),  disagree left / agree right")
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.008), ncol=5,
              frameon=False, fontsize=6.5, handlelength=1.4,
              columnspacing=1.0, handletextpad=0.45, borderaxespad=0.0)
    ax.set_title("Neutral is centred on zero, so the two wings are "
                 "comparable", fontsize=9.0, fontweight="bold", color=INK,
                 pad=15.0)
    return net


def group_panel(ax, gnets, order, codes):
    g = gnets[order]
    y = np.arange(g.shape[0], dtype=float)
    for i in range(g.shape[0]):
        ax.plot([g[i].min(), g[i].max()], [y[i], y[i]], color="#cccccc",
                lw=1.0, zorder=2)
    for k in range(3):
        ax.scatter(g[:, k], y, s=17.0, color=G_COLORS[k], edgecolor="white",
                   linewidths=0.6, zorder=3, label=GROUPS[k])
    ax.axvline(0.0, color="#333333", lw=0.9, ls=(0, (3.0, 2.2)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(codes, fontsize=6.5)
    ax.set_ylim(-0.7, g.shape[0] - 0.3)
    ax.set_xlabel("net agreement (pp)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=6.3,
              handlelength=1.0, borderaxespad=0.25, labelspacing=0.3)
    ax.set_title("Net agreement by age band", fontsize=8.2,
                 fontweight="bold", color=INK)
    return float(np.abs(g[:, 0] - g[:, 2]).max())


def nores_panel(ax, nores, order, codes):
    v = nores[order]
    y = np.arange(v.size, dtype=float)
    ax.barh(y, v, height=0.62, color="#8a8a8a", alpha=0.9, edgecolor="white",
            linewidth=0.7)
    for i, val in enumerate(v):
        ax.text(val + 0.16, y[i], "%.1f" % val, ha="left", va="center",
                fontsize=6.3, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels(codes, fontsize=6.5)
    ax.set_ylim(-0.7, v.size - 0.3)
    ax.set_xlim(0.0, v.max() * 1.32)
    ax.set_xlabel("item non-response (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Who skipped the item", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    shares, gnets, nores, gcount = survey()
    net_all = shares[:, 3] + shares[:, 4] - shares[:, 0] - shares[:, 1]
    order = np.argsort(net_all)
    labels = ["%s  %s" % (ITEMS[i][0], ITEMS[i][1]) for i in order]
    codes = [ITEMS[i][0] for i in order]

    fig = plt.figure(figsize=(FW, FH))
    ax_main = fig.add_axes([MAIN[0] / FW, MAIN[1] / FH,
                            MAIN[2] / FW, MAIN[3] / FH])
    ax_grp = fig.add_axes([AUX_TOP[0] / FW, AUX_TOP[1] / FH,
                           AUX_TOP[2] / FW, AUX_TOP[3] / FH])
    ax_nr = fig.add_axes([AUX_BOT[0] / FW, AUX_BOT[1] / FH,
                          AUX_BOT[2] / FW, AUX_BOT[3] / FH])

    net = diverging_panel(ax_main, shares, order, labels)
    gap = group_panel(ax_grp, gnets, order, codes)
    nores_panel(ax_nr, nores, order, codes)

    panel_labels_inches(fig, [ax_main, ax_grp, ax_nr], dx=-0.62, dy=0.28)
    savefig(fig, "809_soc_survey_likert.png")


if __name__ == "__main__":
    main()
