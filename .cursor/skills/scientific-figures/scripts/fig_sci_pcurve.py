"""The p-curve: what the shape of the significant p values gives away.

Main panel: the distribution of significant p values (0 < p <= 0.05) in
0.01-wide bins for three simulated literatures - one where the effect is
real, one where nothing is going on, and one where nothing is going on but
each study quietly reports its best of five analyses.  A real effect leans
left (right-skewed: most of the significant results are far below 0.05);
selective analysis leans right and piles up just under the threshold; a pure
null is flat at 20 % per bin, which is drawn as the reference line rather
than assumed.

Auxiliary (both at 58 % of the main panel width): the full p-value
distribution over the whole unit interval on a logarithmic count axis, where
the spike at zero and the step at 0.05 are both visible at once; and the
share of significant results below 0.025 as a function of true statistical
power, simulated on a grid, which turns the shape of the main panel back
into an implied power for each literature.

Every p value comes from a simulated two-sample t test; the bin heights, the
binomial split and the implied powers are counted from those p values.
Illustrative simulation, no real literature is depicted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
RUST = "#c0603f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (1.00, 0.90, 5.42, 4.58)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

N_PER_GROUP = 26
N_STUDIES = 24000
N_LOOK0, N_LOOK1, N_STEP = 8, 44, 3     # optional-stopping schedule
ALPHA = 0.05
EDGES = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def t_test_p(rng, n_sim, n, d):
    """Two-sided p values of ``n_sim`` independent two-sample t tests."""
    a = rng.normal(0.0, 1.0, (n_sim, n))
    b = rng.normal(d, 1.0, (n_sim, n))
    va = a.var(axis=1, ddof=1)
    vb = b.var(axis=1, ddof=1)
    sp = np.sqrt(0.5 * (va + vb))
    t = (b.mean(axis=1) - a.mean(axis=1)) / (sp * np.sqrt(2.0 / n))
    return 2.0 * stats.t.sf(np.abs(t), 2 * n - 2)


def stopped_p(rng, n_sim, n_start, n_max, step, d):
    """Optional stopping: test after every few subjects, report on crossing.

    This is the p-hacking that bends a p-curve, and it has to be simulated
    rather than asserted: a study that stops the moment it crosses 0.05 ends
    up reporting a p value that sits just under the threshold.
    """
    a = rng.normal(0.0, 1.0, (n_sim, n_max))
    b = rng.normal(d, 1.0, (n_sim, n_max))
    ca, ca2 = np.cumsum(a, axis=1), np.cumsum(a ** 2, axis=1)
    cb, cb2 = np.cumsum(b, axis=1), np.cumsum(b ** 2, axis=1)
    out = np.ones(n_sim)
    done = np.zeros(n_sim, dtype=bool)
    for n in range(n_start, n_max + 1, step):
        ma, mb = ca[:, n - 1] / n, cb[:, n - 1] / n
        va = (ca2[:, n - 1] - n * ma ** 2) / (n - 1)
        vb = (cb2[:, n - 1] - n * mb ** 2) / (n - 1)
        sp = np.sqrt(0.5 * (va + vb))
        t = (mb - ma) / (sp * np.sqrt(2.0 / n))
        p = 2.0 * stats.t.sf(np.abs(t), 2 * n - 2)
        hit = (~done) & (p <= ALPHA)
        out[hit] = p[hit]
        done |= hit
        last = p
    out[~done] = last[~done]
    return out


def pcurve(p):
    """Percentage of the significant p values falling in each 0.01 bin."""
    sig = p[(p > 0) & (p <= ALPHA)]
    counts, _ = np.histogram(sig, bins=EDGES)
    return 100.0 * counts / counts.sum(), sig


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(993)

    p_real = t_test_p(rng, N_STUDIES, N_PER_GROUP, 0.62)
    p_null = t_test_p(rng, N_STUDIES, N_PER_GROUP, 0.0)
    p_hack = stopped_p(rng, N_STUDIES, N_LOOK0, N_LOOK1, N_STEP, 0.0)

    lits = [
        ("Real effect, d = 0.62", p_real, DEEP, 2.0, "o"),
        ("No effect, one analysis each", p_null, lighten(BLUE, 0.35), 1.5,
         "s"),
        ("No effect, tested after every 3 subjects", p_hack, ACCENT, 2.0,
         "D"),
    ]
    rows = []
    for name, p, col, lw, mk in lits:
        h, sig = pcurve(p)
        low = float((sig < 0.025).mean()) * 100.0
        rows.append(dict(name=name, p=p, col=col, lw=lw, mk=mk, h=h,
                         n_sig=sig.size, low=low,
                         rate=100.0 * sig.size / p.size))

    # power grid: the same test at a range of true effect sizes
    ds = np.linspace(0.0, 1.25, 26)
    pw, idx = [], []
    for d in ds:
        p = t_test_p(rng, 6000, N_PER_GROUP, d)
        sig = p[p <= ALPHA]
        pw.append(100.0 * sig.size / p.size)
        idx.append(100.0 * (sig < 0.025).mean() if sig.size else np.nan)
    pw = np.array(pw)
    idx = np.array(idx)
    order = np.argsort(pw)
    for r in rows:
        r["implied"] = float(np.interp(r["low"], idx[order], pw[order]))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the p-curve --------------------------------------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.55, zorder=0)
    ax.bar(CENTRES, rows[0]["h"], width=0.0082,
           facecolor=lighten(BLUE, 0.62), edgecolor="none", zorder=2)
    ax.axhline(20.0, color=FAINT, lw=1.0, ls=(0, (3.2, 2.2)), zorder=3)
    ax.text(0.0506, 14.0, "20 % per bin:\nwhat a true null gives",
            ha="right", va="center", fontsize=6.4, color=SUB,
            linespacing=1.5)

    for r in rows:
        ax.plot(CENTRES, r["h"], color=r["col"], lw=r["lw"], marker=r["mk"],
                ms=5.0, mfc="white", mew=1.3, zorder=6)
        for x, y in zip(CENTRES, r["h"]):
            ax.text(x, y + 1.6, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=6.2, color=r["col"])

    ax.annotate("right-skewed:\nevidential value",
                xy=(CENTRES[0], rows[0]["h"][0]),
                xytext=(0.0142, 62.0), fontsize=6.8,
                color=DEEP, ha="left", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                connectionstyle="arc3,rad=0.20"))
    ax.annotate("left-skewed: results bunched\nagainst the 0.05 threshold",
                xy=(CENTRES[-1], rows[2]["h"][-1]),
                xytext=(0.0362, 62.0), fontsize=6.8,
                color=ACCENT, ha="center", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                connectionstyle="arc3,rad=-0.20"))

    ax.set_xlim(0.004, 0.0512)
    ax.set_ylim(0, 86)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_xticks(CENTRES)
    ax.set_xticklabels(["0.00-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04",
                        "0.04-0.05"], fontsize=7.4)
    ax.set_xlabel("p value of the reported test (p, significant results only)",
                  fontsize=8.5)
    ax.set_ylabel("share of that literature's significant results (%)",
                  fontsize=8.5)
    ax.set_title(f"p-curves of three simulated literatures, "
                 f"{N_STUDIES} two-sample t tests each",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    hl = [Line2D([], [], color=r["col"], lw=r["lw"], marker=r["mk"], ms=5.0,
                 mfc="white", mew=1.3,
                 label=f"{r['name']}: {r['n_sig']} significant "
                       f"({r['rate']:.0f} % of studies)") for r in rows]
    leg = ax.legend(handles=hl, loc="upper center", frameon=False,
                    fontsize=6.7, labelspacing=0.45, handlelength=2.2,
                    borderpad=0.2, bbox_to_anchor=(0.52, 1.010))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): the whole p distribution ----------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    bins = np.linspace(0.0, 1.0, 41)
    for r in rows:
        ax_t.hist(r["p"], bins=bins, histtype="step", color=r["col"],
                  lw=1.2, zorder=5)
    ax_t.axvline(ALPHA, color=SUB, lw=0.9, ls=(0, (2.6, 2.0)), zorder=6)
    ax_t.set_yscale("log")
    ax_t.set_xlim(0.0, 1.0)
    ax_t.set_ylim(30.0, 4.0e4)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("p value (p, all studies)", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("studies per bin (log scale)", fontsize=7.6)
    ax_t.set_title("The same literatures, unfiltered", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(0.062, 0.94, "p = 0.05", transform=ax_t.transAxes, ha="left",
              va="top", fontsize=6.1, color=SUB)
    ax_t.text(0.975, 0.955,
              "counts are on a log axis: the significant\ntail is two orders "
              "of magnitude below the bulk",
              transform=ax_t.transAxes, ha="right", va="top", fontsize=6.1,
              color=SUB, linespacing=1.5)

    # ---- aux (c): shape against true power ----------------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_b.plot(pw, idx, color=BLUE, lw=1.6, zorder=5)
    ax_b.axhline(50.0, color=FAINT, lw=0.9, ls=(0, (2.6, 2.0)), zorder=3)
    for r, dy in zip(rows, (-4.0, 7.0, 0.0)):
        ax_b.plot([r["implied"]], [r["low"]], marker=r["mk"], ms=5.4,
                  mfc="white", mec=r["col"], mew=1.4, zorder=8)
        ax_b.plot([r["implied"], r["implied"]], [0.0, r["low"]],
                  color=r["col"], lw=0.8, ls=(0, (2.2, 1.8)), zorder=6)
        ax_b.text(r["implied"] + 3.0, r["low"] + dy,
                  f"implied power {r['implied']:.0f} %", ha="left",
                  va="center", fontsize=6.1, color=r["col"])
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("true power of the test (1 - beta, %)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("significant results below p = 0.025 (%)", fontsize=7.6)
    ax_b.set_title("Shape reads back as power", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.text(0.030, 0.055,
              "curve: 6 000 simulated studies at each of 26 effect sizes",
              transform=ax_b.transAxes, ha="left", va="bottom", fontsize=6.1,
              color=MUTED)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.72, dy=0.12)
    savefig(fig, "993_sci_pcurve.png")


if __name__ == "__main__":
    main()
