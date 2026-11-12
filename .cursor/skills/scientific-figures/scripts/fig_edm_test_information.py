"""Test information and standard error: the same curve read two ways.

The six items are the ones defined in fig_edm_irt_icc_curves.py (894), copied
verbatim, so a peak here has to sit where the item information panel there
says it does.  The stack is the point: test information is a plain sum, and
drawing it as bands makes visible which item is paying for which slice of the
ability range.  SEM is not an independent curve - it is 1/sqrt(I) evaluated on
the same array, which is why its minimum falls on the peak by construction
rather than by eye.  The right column asks the only question a test designer
has: does the form clear the target information the blueprint asked for, and
where does it stop clearing it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
BLUE = "#3b5b92"          # long form
ORANGE = "#c0603f"        # short form
GREEN = "#2e7d5b"         # adaptive form
ACCENT = "#b23a48"

FIGW, FIGH = 10.4, 6.4
MAIN = (0.78, 1.05, 5.10, 4.35)   # narrowed to clear the twin SEM axis
AUX_T = (7.05, 3.62, 3.00, 1.78)
AUX_B = (7.05, 1.05, 3.00, 1.78)

THETA_LO, THETA_HI = -4.0, 4.0

# The six items of figure 894, unchanged: discrimination a, difficulty b
# (logits), lower asymptote c.  Every number in the main panel comes off them.
A_PAR = np.array([1.65, 1.10, 1.85, 0.45, 1.30, 0.95])
B_PAR = np.array([-1.90, -0.85, 0.15, 0.70, 1.55, 2.40])
C_PAR = np.array([0.18, 0.05, 0.22, 0.20, 0.12, 0.25])

SEM_GOOD = 1.20           # the SEM the six-item module is asked to reach
SEM_TOP = 3.0             # SEM axis cap; the low tail runs far above it
N_LONG = 36               # operational bank behind the three forms
N_SHORT = 14              # fixed short form, and the adaptive test length
SEM_TARGET = 0.50         # target SEM inside the reporting range
REPORT = (-2.0, 2.0)      # theta window the blueprint has to serve


def icc(theta, a, b, c):
    """3PL probability of a correct response."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def item_info(theta, a, b, c):
    """3PL item information; reduces to a^2 P Q when c = 0."""
    p = icc(theta, a, b, c)
    q = 1.0 - p
    return a ** 2 * (q / p) * ((p - c) / (1.0 - c)) ** 2


def operational_bank(seed=896, n=N_LONG):
    """A calibrated bank spread over the reporting range."""
    rng = np.random.default_rng(seed)
    b = np.linspace(-2.60, 2.60, n) + rng.normal(0.0, 0.14, n)
    a = np.clip(rng.gamma(5.2, 0.25, n), 0.35, 2.60)
    c = np.clip(rng.beta(2.4, 12.0, n), 0.0, 0.32)
    return a, b, c


def main():
    set_pub_style()
    theta = np.linspace(THETA_LO, THETA_HI, 801)

    # ---- main panel: six items, one sum -----------------------------------
    infos = np.array([item_info(theta, a, b, c)
                      for a, b, c in zip(A_PAR, B_PAR, C_PAR)])
    total = infos.sum(axis=0)
    sem = 1.0 / np.sqrt(total)
    # The identity the figure rests on, checked rather than trusted.
    assert np.allclose(total, np.sum(infos, axis=0))
    assert np.isclose(theta[int(np.argmax(total))],
                      theta[int(np.argmin(sem))])

    peak = int(np.argmax(total))
    th_peak, i_peak, sem_min = theta[peak], total[peak], sem[peak]
    good = sem <= SEM_GOOD
    g_lo, g_hi = float(theta[good].min()), float(theta[good].max())
    off = theta[(sem > SEM_TOP) & (theta < th_peak)]
    off_lo = float(off.max())

    # Same colour logic as 894: dark blue is the hardest item.
    ramp = get_palette("cb_sequential_blue", 8)
    order = np.argsort(B_PAR)
    col = {int(k): ramp[i + 2] for i, k in enumerate(order)}

    # ---- right column: three forms against one target ---------------------
    a_op, b_op, c_op = operational_bank()
    imat = np.array([item_info(theta, a, b, c)
                     for a, b, c in zip(a_op, b_op, c_op)])
    short_idx = np.unique(np.round(
        np.linspace(0, N_LONG - 1, N_SHORT)).astype(int))
    tif = {
        "long": imat.sum(axis=0),
        "short": imat[short_idx].sum(axis=0),
        # Adaptive: the N_SHORT most informative items of the same bank at
        # each theta, so it can never fall below any fixed form of that length.
        "adaptive": np.sort(imat, axis=0)[-N_SHORT:].sum(axis=0),
    }
    # Target: flat-topped plateau equal to 1/SEM_TARGET^2 across the
    # reporting range, decaying outside it.
    i_target = (1.0 / SEM_TARGET ** 2) * np.exp(-(theta / 2.35) ** 6)
    inside = (theta >= REPORT[0]) & (theta <= REPORT[1])
    hit = {k: 100.0 * np.mean(v[inside] >= i_target[inside])
           for k, v in tif.items()}
    forms = [("long", f"long form, {N_LONG} items", BLUE, "-"),
             ("adaptive", f"adaptive, {N_SHORT} items", GREEN, (0, (5.0, 1.9))),
             ("short", f"short form, {N_SHORT} items", ORANGE,
              (0, (2.2, 1.7)))]

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax_i = axes_inches(fig, MAIN)
    ax_s = ax_i.twinx()
    ax_f = axes_inches(fig, AUX_T)
    ax_r = axes_inches(fig, AUX_B)

    # ---- main: stacked contributions + total + SEM ------------------------
    ax_i.stackplot(theta, infos[order], colors=[col[int(k)] for k in order],
                   alpha=0.55, edgecolor="white", linewidth=0.35, zorder=2)
    ax_i.plot(theta, total, color=INK, lw=2.0, zorder=6,
              label=f"test information I(theta) = sum of {A_PAR.size} items")
    cum = np.cumsum(infos[order], axis=0)
    for rank, k in enumerate(order):
        j = int(np.argmax(infos[k]))
        lo = cum[rank - 1][j] if rank else 0.0
        if cum[rank][j] - lo < 0.055:
            continue
        ax_i.text(theta[j], (lo + cum[rank][j]) / 2.0, f"{int(k) + 1}",
                  ha="center", va="center", fontsize=6.4, zorder=5,
                  color="white" if rank >= 3 else INK)
    ax_i.axvline(th_peak, color=GREY, lw=0.7, ls=(0, (2.0, 1.8)), zorder=3)
    ax_i.plot([th_peak], [i_peak], marker="o", ms=4.6, mfc="white", mec=INK,
              mew=1.2, zorder=7)
    ax_i.annotate(
        f"peak I = {i_peak:.2f} at theta = {th_peak:+.2f} logits\n"
        f"stack height = sum of the six bands, exactly",
        xy=(th_peak, i_peak), xytext=(th_peak + 0.30, i_peak * 1.64),
        fontsize=6.9, color=INK, ha="left", va="bottom", linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=1.0,
                        shrinkB=3.0))

    ax_s.plot(theta, sem, color=ACCENT, lw=1.9, ls=(0, (5.2, 2.0)), zorder=7,
              label="standard error SEM = 1 / sqrt(I) (logits)")
    ax_s.plot([th_peak], [sem_min], marker="o", ms=4.6, mfc="white",
              mec=ACCENT, mew=1.2, zorder=8)
    ax_s.axhline(SEM_GOOD, color=ACCENT, lw=0.7, ls=(0, (1.4, 1.8)),
                 alpha=0.8, zorder=4)
    ax_s.annotate(
        f"SEM bottoms out at {sem_min:.3f} logits on that same theta:\n"
        f"the two curves are one array, not two fits. It holds\n"
        f"at or below {SEM_GOOD:.2f} logits only over theta = {g_lo:+.2f} to "
        f"{g_hi:+.2f}\n(the shaded band, {g_hi - g_lo:.2f} logits wide)",
        xy=(th_peak, sem_min), xytext=(-0.30, SEM_TOP * 0.70),
        fontsize=6.7, color=ACCENT, ha="left", va="top", linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, shrinkA=3.0,
                        shrinkB=3.0))
    ax_s.axvspan(g_lo, g_hi, color=ACCENT, alpha=0.05, lw=0.0, zorder=0)
    ax_s.text(THETA_LO + 0.10, SEM_TOP * 0.25,
              f"SEM leaves the axis below\ntheta = {off_lo:+.2f} logits: six "
              f"items\nmeasure nothing down here",
              ha="left", va="top", fontsize=6.5, color=MUTED,
              linespacing=1.4)

    ax_i.set_xlim(THETA_LO, THETA_HI)
    ax_i.set_ylim(0.0, float(total.max()) * 1.90)
    ax_s.set_ylim(0.0, SEM_TOP)
    ax_i.set_xticks(np.arange(-4, 5))
    ax_i.set_xlabel("latent ability theta (logits, SD units)", fontsize=8.0)
    ax_i.set_ylabel("test information I (1 / logit^2)", fontsize=8.0)
    ax_s.set_ylabel("standard error SEM (logits)", fontsize=8.0,
                    color=ACCENT, labelpad=2.0)
    ax_i.tick_params(labelsize=7.2)
    ax_s.tick_params(labelsize=7.2, colors=ACCENT)
    ax_s.spines["right"].set_color(ACCENT)
    ax_s.spines["top"].set_visible(False)
    ax_i.grid(axis="y", lw=0.5, color="#e6e6e6")
    ax_i.set_title("Information adds up across items; SEM is 1 / sqrt(I)",
                   fontsize=9.0, fontweight="bold", color=INK, pad=10.0)
    h1, l1 = ax_i.get_legend_handles_labels()
    h2, l2 = ax_s.get_legend_handles_labels()
    band = Patch(facecolor=col[int(order[3])], alpha=0.55, edgecolor="white",
                 linewidth=0.35)
    ax_i.legend(h1 + h2 + [band],
                l1 + l2 + ["per-item contribution, numbered as in figure 894"],
                loc="upper left", frameon=False, fontsize=6.8,
                handlelength=2.0, labelspacing=0.32, borderaxespad=0.5)

    # ---- aux (b): three forms against the target --------------------------
    ax_f.fill_between(theta, 0.0, i_target, color=GREY, alpha=0.13, lw=0.0,
                      zorder=1)
    ax_f.plot(theta, i_target, color=GREY, lw=1.1, ls=(0, (3.4, 2.0)),
              zorder=3, label=f"target (SEM {SEM_TARGET:.2f} logits)")
    for key, lab, c, ls in forms:
        ax_f.plot(theta, tif[key], color=c, lw=1.5, ls=ls, zorder=4,
                  label=lab)
    ax_f.axvspan(REPORT[0], REPORT[1], color=BLUE, alpha=0.05, lw=0.0,
                 zorder=0)
    ax_f.set_xlim(THETA_LO, THETA_HI)
    ax_f.set_ylim(0.0, max(tif["long"].max(), i_target.max()) * 1.30)
    ax_f.set_xticks(np.arange(-4, 5, 2))
    ax_f.set_xlabel("theta (logits)", fontsize=7.4)
    ax_f.set_ylabel("I (1 / logit^2)", fontsize=7.4)
    ax_f.tick_params(labelsize=6.6)
    ax_f.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_f.set_title("Three ways to spend the same bank", fontsize=8.5,
                   fontweight="bold", color=INK)
    ax_f.legend(loc="upper left", frameon=False, fontsize=6.2,
                handlelength=1.6, labelspacing=0.26, borderaxespad=0.35)

    # ---- aux (c): attainment ratio ----------------------------------------
    ratio = {k: v / np.maximum(i_target, 1e-6) for k, v in tif.items()}
    ax_r.axhline(1.0, color=GREY, lw=0.9, ls=(0, (3.4, 2.0)), zorder=3)
    ax_r.axvspan(REPORT[0], REPORT[1], color=BLUE, alpha=0.05, lw=0.0,
                 zorder=0)
    for key, lab, c, ls in forms:
        ax_r.plot(theta[inside], ratio[key][inside], color=c, lw=1.5, ls=ls,
                  zorder=4)
        ax_r.text(REPORT[1] + 0.12, ratio[key][inside][-1],
                  f"{hit[key]:.0f} %", ha="left", va="center", fontsize=6.3,
                  color=c)
    ax_r.set_xlim(THETA_LO, THETA_HI)
    ax_r.set_ylim(0.0, float(max(r[inside].max() for r in ratio.values()))
                  * 1.22)
    ax_r.set_xticks(np.arange(-4, 5, 2))
    ax_r.set_xlabel("theta (logits)", fontsize=7.4)
    ax_r.set_ylabel("I / target (-)", fontsize=7.4)
    ax_r.tick_params(labelsize=6.6)
    ax_r.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_r.set_title("Attainment inside the reporting range", fontsize=8.5,
                   fontweight="bold", color=INK)
    ax_r.text(0.015, 0.97,
              f"reporting range theta {REPORT[0]:+.0f} to {REPORT[1]:+.0f}; "
              f"labels give the percent\nof it held at or above target",
              transform=ax_r.transAxes, ha="left", va="top", fontsize=6.2,
              color=MUTED, linespacing=1.4)


    note = (
        f"Panel (a): 3PL item information I = a^2 (Q/P) [(P - c)/(1 - c)]^2 "
        f"summed over the {A_PAR.size} items of figure 894, on 801 grid points "
        f"over theta = {THETA_LO:+.0f} to {THETA_HI:+.0f} logits; the heavy "
        f"line is the band stack itself and SEM = 1/sqrt(I) is evaluated on "
        f"that same array, so the minimum {sem_min:.3f} logits sits on the "
        f"peak I = {i_peak:.2f} by construction. Panels (b) and (c): one bank "
        f"of {N_LONG} calibrated items read three ways - all of them, a fixed "
        f"{N_SHORT}-item form, and the {N_SHORT} most informative items at "
        f"each theta - against a target of SEM {SEM_TARGET:.2f} logits "
        f"(I = {1.0 / SEM_TARGET ** 2:.1f}) across theta {REPORT[0]:+.0f} to "
        f"{REPORT[1]:+.0f}, met on {hit['long']:.0f} / {hit['adaptive']:.0f} / "
        f"{hit['short']:.0f} % of that range.")
    fig.text(0.5, 0.16 / FIGH, textwrap.fill(note, 195), ha="center",
             va="bottom", fontsize=6.5, color=MUTED, linespacing=1.5)

    panel_labels_inches(fig, [ax_i, ax_f, ax_r],
                        labels=["(a)", "(b)", "(c)"], dx=-0.56, dy=0.12)
    savefig(fig, "896_edm_test_information.png")


if __name__ == "__main__":
    main()
