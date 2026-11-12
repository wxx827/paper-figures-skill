"""Sleeping beauties: papers whose citations arrive a decade late.

Main panel: annual citations against years since publication for a cohort of
sixty fictional papers, drawn as a faint bundle, with four archetypes lifted
out of it - an immediate hit, a slow-burning classic, a flash in the pan and
a sleeping beauty.  The beauty carries the geometric construction of the
beauty coefficient B: the straight line from the paper's first year to its
peak year, and the vertical gap between that line and the curve at the
awakening year, which is where the gap is widest.  B is the accumulated
gap, computed from the plotted counts, not asserted.

Auxiliary (both at 58 % of the main panel width): the distribution of B over
the whole cohort on a logarithmic axis, because B spans four orders of
magnitude and a linear axis would show a single spike at zero; and the share
of each paper's citations collected in its first five years against its
career total, where the beauties sit alone in the bottom-right corner.

Fictional papers, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
RUST = "#c0603f"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
WIRE = "#c8c8c8"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.98, 0.90, 5.44, 4.58)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

T_MAX = 30
YEARS = np.arange(T_MAX + 1)
EARLY = 5          # "first five years" window used in aux (c)


def hit_curve(rng, amp, peak, alpha):
    """Rise-and-decay citation history: a smooth shape plus Poisson noise."""
    t = YEARS + 0.5
    shape = (t / peak) ** alpha * np.exp(alpha * (1.0 - t / peak))
    return rng.poisson(np.clip(amp * shape, 0.02, None))


def beauty_curve(rng, base, amp, wake, ramp, decay):
    """Flat for years, then a logistic awakening and a slow decline."""
    t = YEARS.astype(float)
    burst = amp / (1.0 + np.exp(-(t - wake) / ramp))
    burst *= np.exp(-np.clip(t - wake - 5.0, 0.0, None) / decay)
    return rng.poisson(base + burst)


def beauty_coefficient(c):
    """B of Ke et al.: accumulated gap between the curve and its 0-to-peak chord."""
    tm = int(np.argmax(c))
    if tm == 0:
        return 0.0, 0, 0.0
    t = np.arange(tm + 1, dtype=float)
    chord = (c[tm] - c[0]) / tm * t + c[0]
    gap = (chord - c[:tm + 1]) / np.maximum(c[:tm + 1], 1.0)
    ta = int(np.argmax(chord - c[:tm + 1]))
    return float(gap.sum()), ta, float(chord[ta])


def early_share(c):
    """Percentage of a paper's citations collected in its first five years."""
    tot = c.sum()
    return 100.0 * c[:EARLY + 1].sum() / tot if tot else 0.0


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(992)

    cohort = []
    for _ in range(40):
        if rng.random() < 0.14:
            c = beauty_curve(rng, rng.uniform(0.2, 1.1),
                             rng.uniform(14.0, 46.0), rng.uniform(9.0, 19.0),
                             rng.uniform(1.1, 2.2), rng.uniform(7.0, 14.0))
        else:
            c = hit_curve(rng, rng.uniform(4.0, 34.0), rng.uniform(2.0, 7.5),
                          rng.uniform(1.3, 3.2))
        cohort.append(c)

    marks = [
        ("Immediate hit", hit_curve(rng, 46.0, 2.6, 2.4), DEEP, 1.9),
        ("Slow-burning classic", hit_curve(rng, 26.0, 11.0, 1.5),
         lighten(BLUE, 0.30), 1.6),
        ("Flash in the pan", hit_curve(rng, 30.0, 1.6, 4.2),
         lighten(RUST, 0.35), 1.6),
        ("Sleeping beauty", beauty_curve(rng, 0.7, 52.0, 17.0, 1.4, 11.0),
         ACCENT, 2.1),
    ]
    named = []
    for name, c, col, lw in marks:
        b, ta, chord_ta = beauty_coefficient(c)
        named.append(dict(name=name, c=c, col=col, lw=lw, b=b, ta=ta,
                          chord=chord_ta, tm=int(np.argmax(c)),
                          total=int(c.sum()), early=early_share(c)))
    bea = named[-1]

    coh_b = np.array([beauty_coefficient(c)[0] for c in cohort])
    coh_tot = np.array([c.sum() for c in cohort], dtype=float)
    coh_early = np.array([early_share(c) for c in cohort])
    b_all = np.concatenate([coh_b, [d["b"] for d in named]])
    thr = float(np.quantile(b_all, 0.90))
    n_over = int((b_all >= thr).sum())

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: citation histories -------------------------------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.55, zorder=0)
    for c in cohort:
        ax.plot(YEARS, c, color=WIRE, lw=0.55, alpha=0.6, zorder=2)
    for d in named:
        ax.plot(YEARS, d["c"], color=d["col"], lw=d["lw"],
                zorder=6 if d is bea else 5)

    tm, ta = bea["tm"], bea["ta"]
    ax.plot([0, tm], [bea["c"][0], bea["c"][tm]], color=GOLD, lw=1.2,
            ls=(0, (3.4, 2.0)), zorder=7)
    ax.plot([ta, ta], [bea["c"][ta], bea["chord"]], color=ACCENT, lw=1.4,
            zorder=8)
    ax.plot([ta], [bea["chord"]], marker="o", ms=3.8, mfc="white", mec=ACCENT,
            mew=1.2, zorder=9)
    ax.plot([ta], [bea["c"][ta]], marker="o", ms=3.8, mfc="white", mec=ACCENT,
            mew=1.2, zorder=9)
    ax.annotate(f"awakening year t_a = {ta} years after publication:\n"
                f"widest gap between the curve and the gold chord,\n"
                f"which runs from the first year to the peak year",
                xy=(ta, 0.5 * (bea["c"][ta] + bea["chord"])),
                xytext=(ta - 1.5, 41.0), fontsize=6.5,
                color=ACCENT, ha="center", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(tm + 0.6, bea["c"][tm],
            f"peak year t_m = {tm}\nB = {bea['b']:.1f} (dimensionless)",
            ha="left", va="center", fontsize=6.6, color=ACCENT,
            linespacing=1.5)

    ax.set_xlim(-0.6, T_MAX + 0.6)
    ax.set_ylim(0, 84)
    ax.set_yticks(np.arange(0, 71, 10))
    ax.set_xticks(np.arange(0, T_MAX + 1, 5))
    ax.set_xlabel("years since publication (years)", fontsize=8.5)
    ax.set_ylabel("citations received in that year (citations/year)",
                  fontsize=8.5)
    ax.set_title(f"Annual citation histories of {len(cohort) + len(named)} "
                 f"fictional papers, four archetypes highlighted",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    hl = [Line2D([], [], color=d["col"], lw=d["lw"],
                 label=f"{d['name']}: {d['total']} citations, "
                       f"B = {d['b']:.1f}") for d in named]
    hl.append(Line2D([], [], color=WIRE, lw=0.8,
                     label=f"{len(cohort)} other papers of the cohort"))
    leg = ax.legend(handles=hl, loc="upper right", frameon=False,
                    fontsize=6.6, labelspacing=0.45, handlelength=2.0,
                    borderpad=0.2, bbox_to_anchor=(1.008, 1.008))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): distribution of B -----------------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    pos = b_all[b_all > 0.02]
    bins = np.logspace(np.log10(0.02), np.log10(max(pos.max(), 10.0) * 1.4), 26)
    ax_t.hist(pos, bins=bins, facecolor=lighten(BLUE, 0.42), edgecolor=BLUE,
              linewidth=0.5, zorder=4)
    ax_t.set_xscale("log")
    ax_t.axvline(thr, color=SUB, lw=0.9, ls=(0, (2.6, 2.0)), zorder=6)
    for d in named:
        if d["b"] > 0.02:
            ax_t.plot([d["b"]], [0.6], marker="v", ms=4.4, mfc=d["col"],
                      mec="white", mew=0.6, clip_on=False, zorder=8)
    ax_t.set_xlim(bins[0], bins[-1])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("beauty coefficient B (dimensionless, log scale)",
                    fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("papers", fontsize=7.6)
    ax_t.set_title("B spans four orders of magnitude", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(0.975, 0.955,
              f"90th percentile B = {thr:.1f}\n{n_over} of "
              f"{b_all.size} papers at or above it\ntriangles mark the "
              f"archetypes with B > 0",
              transform=ax_t.transAxes, ha="right", va="top", fontsize=6.1,
              color=SUB, linespacing=1.5)

    # ---- aux (c): early share against career total --------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_b.scatter(coh_tot, coh_early, s=9.0, facecolor=lighten(BLUE, 0.45),
                 edgecolor=BLUE, linewidth=0.35, alpha=0.85, zorder=4)
    for d in named:
        ax_b.plot([d["total"]], [d["early"]], marker="o", ms=5.0,
                  mfc=d["col"], mec="white", mew=0.8, zorder=7)
    ax_b.axhline(float(np.median(np.concatenate(
        [coh_early, [d["early"] for d in named]]))), color=SUB, lw=0.9,
        ls=(0, (2.6, 2.0)), zorder=5)
    ax_b.set_xscale("log")
    ax_b.set_xlim(8.0, 1400.0)
    ax_b.set_ylim(-3.0, 100.0)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("career citations (citations, log scale)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("share collected in years 0-5 (%)", fontsize=7.6)
    ax_b.set_title("Beauties are cited late, not little", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.annotate("sleeping beauty", xy=(bea["total"], bea["early"]),
                  xytext=(bea["total"] * 0.30, bea["early"] + 26.0),
                  fontsize=6.2, color=ACCENT, ha="center", va="bottom",
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                  connectionstyle="arc3,rad=0.20"))
    ax_b.text(0.030, 0.955, "dashed line: cohort median early share",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=6.1,
              color=SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.70, dy=0.12)
    savefig(fig, "992_sci_citation_age.png")


if __name__ == "__main__":
    main()
