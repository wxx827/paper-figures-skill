"""Radiocarbon calibration drawn as the projection it actually is.

The three panels of the main block are one construction, not three charts: the
laboratory determination is a normal density on the vertical radiocarbon axis,
the calibration curve carries it across, and what comes out at the bottom is
the calendar density.  Because the synthetic curve carries a Hallstatt-style
plateau, a single-peaked measurement projects into a wide, three-humped
calendar distribution - that asymmetry is the whole reason archaeologists
calibrate rather than quote BP ages, so the wiggles here are load bearing.

Every number is measured off the arrays.  The highest posterior density
intervals are found by ranking grid cells by probability until 68.2 % and
95.4 % of the mass is enclosed, so a multimodal date honestly reports several
disjoint ranges.  The right-hand panels are supporting evidence: the interval
bars show that eight determinations in stratigraphic order overlap heavily,
and the sequence model - a Gibbs sampler that only ever accepts orderings
consistent with the section - converts that ordering into narrower intervals.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, shared_limits

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"
CURVE = "#3b5b92"          # house blue: the calibration curve
POST = "#b23a48"           # house red: the calibrated (calendar) result
MEAS = "#555555"           # neutral: the laboratory determination

FW, FH = 10.4, 6.4         # DESIGN_SYSTEM section 1.1, "wide" tier

# Calendar grid.  Negative values are BC; the axis formatter prints them as
# positive numbers under a "cal BC" label, which is how the period is read.
CAL = np.arange(-1150.0, -280.0 + 0.5, 1.0)

# One determination per context, deepest first, so index order is also
# stratigraphic order (older -> younger).
SAMPLES = [
    ("SU 118", "OxA-31207", 2812.0, 30.0),
    ("SU 116", "OxA-31212", 2764.0, 28.0),
    ("SU 114", "OxA-31219", 2731.0, 32.0),
    ("SU 112", "OxA-31226", 2708.0, 26.0),
    ("SU 110", "OxA-31233", 2690.0, 30.0),
    ("SU 108", "OxA-31241", 2668.0, 34.0),
    ("SU 106", "OxA-31248", 2611.0, 29.0),
    ("SU 104", "OxA-31255", 2549.0, 27.0),
]
MAIN_IDX = 4               # the determination taken apart in panel (a)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def curve_mu(cal):
    """Synthetic IntCal-like curve: linear ideal + plateau + three wiggles.

    The plateau term adds back almost exactly the 380 yr the ideal line loses
    over the same span, so between roughly 800 and 420 cal BC the curve is
    flat and the wiggles dominate.  A determination that lands there has
    several equally good calendar solutions.
    """
    ideal = 1950.0 - cal
    plateau = 344.0 * smoothstep((cal + 800.0) / 380.0)
    wig = (54.0 * np.sin(2 * np.pi * (cal + 1000.0) / 205.0)
           + 19.0 * np.sin(2 * np.pi * (cal + 120.0) / 93.0)
           + 5.0 * np.sin(2 * np.pi * (cal + 30.0) / 44.0))
    return ideal + plateau + wig


def curve_sigma(cal):
    """1 sigma envelope of the curve itself; wider where the record is thin."""
    return 15.0 + 9.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (cal + 430.0) / 320.0))


def calibrate(r, sr, cal=CAL):
    """Posterior over calendar years for a determination ``r +/- sr`` BP."""
    mu, sg = curve_mu(cal), curve_sigma(cal)
    s2 = sr ** 2 + sg ** 2
    p = np.exp(-0.5 * (r - mu) ** 2 / s2) / np.sqrt(s2)
    return p / np.trapezoid(p, cal)


def hpd_segments(x, p, mass, gap=20.0, min_len=10.0):
    """Highest posterior density region as a list of disjoint ``(lo, hi)``.

    Cells are ranked by density and taken until the requested mass is
    enclosed; the surviving cells are then run-length encoded.  Slivers
    shorter than ``min_len`` are dropped and neighbours closer than ``gap``
    merged, otherwise a ragged density produces a dozen unreadable stubs.
    """
    dx = float(x[1] - x[0])
    order = np.argsort(p)[::-1]
    csum = np.cumsum(p[order]) * dx
    k = int(np.searchsorted(csum, mass)) + 1
    keep = np.zeros(x.size, dtype=bool)
    keep[order[:min(k, x.size)]] = True

    segs, i = [], 0
    while i < x.size:
        if keep[i]:
            j = i
            while j + 1 < x.size and keep[j + 1]:
                j += 1
            segs.append([float(x[i]), float(x[j])])
            i = j + 1
        else:
            i += 1
    merged = []
    for lo, hi in segs:
        if merged and lo - merged[-1][1] <= gap:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    out = [tuple(m) for m in merged if m[1] - m[0] >= min_len]
    return out or [tuple(m) for m in merged]


def total_width(segs):
    return float(sum(hi - lo for lo, hi in segs))


def span_text(segs, sep="; "):
    """BC ranges are written oldest first, e.g. ``800-787``."""
    return sep.join(f"{-lo:.0f}-{-hi:.0f}" for lo, hi in segs)


def gibbs_sequence(dens, n_iter=6000, burn=1200, seed=7):
    """Order-constrained sequence model: Gibbs sampling under t0 < t1 < ...

    Each date is redrawn from its own calibrated density truncated to the
    window left by its neighbours, which is exactly the stratigraphic prior
    "this context is older than the one above it".  Nothing else is imposed.
    """
    rng = np.random.default_rng(seed)
    cdfs = []
    for p in dens:
        c = np.cumsum(p + 1e-12)
        cdfs.append(c / c[-1])
    t = np.array([float(CAL[np.searchsorted(c, 0.5)]) for c in cdfs])
    t = np.sort(t)
    n = len(dens)
    keep = np.empty((n_iter - burn, n))
    for it in range(n_iter):
        for i in range(n):
            lo = t[i - 1] if i > 0 else CAL[0]
            hi = t[i + 1] if i < n - 1 else CAL[-1]
            c = cdfs[i]
            a, b = np.interp(lo, CAL, c), np.interp(hi, CAL, c)
            if b - a < 1e-9:
                t[i] = 0.5 * (lo + hi)
                continue
            t[i] = float(np.interp(a + (b - a) * rng.random(), c, CAL))
        if it >= burn:
            keep[it - burn] = t
    return keep


def bc(v):
    return f"{-v:.0f} cal BC"


def main():
    set_pub_style()

    dens = [calibrate(r, s) for _, _, r, s in SAMPLES]
    hpd68 = [hpd_segments(CAL, p, 0.682) for p in dens]
    hpd95 = [hpd_segments(CAL, p, 0.954) for p in dens]

    chain = gibbs_sequence(dens)
    edges = np.arange(CAL[0], CAL[-1] + 4.0, 4.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mod95 = []
    for i in range(len(SAMPLES)):
        h, _ = np.histogram(chain[:, i], bins=edges, density=True)
        mod95.append(hpd_segments(centers, h, 0.954, gap=24.0, min_len=10.0))

    w_raw = np.array([total_width(s) for s in hpd95])
    w_mod = np.array([total_width(s) for s in mod95])
    cut = 1.0 - w_mod / w_raw

    name, lab, R, SR = SAMPLES[MAIN_IDX]
    post = dens[MAIN_IDX]
    n_ranges = len(hpd95[MAIN_IDX])

    # ---- one calendar ruler for every panel that shows calendar years -----
    all_cal = [np.array([lo, hi]) for segs in hpd95 + mod95 for lo, hi in segs]
    xlo, xhi = shared_limits(all_cal + [np.array([-980.0, -380.0])], pad=0.06)

    fig = plt.figure(figsize=(FW, FH))

    # Main block: measurement (left), curve (centre), calendar (bottom).
    ax_m = axes_inches(fig, (0.70, 2.16, 0.74, 3.10))
    ax_c = axes_inches(fig, (1.50, 2.16, 4.55, 3.10))
    ax_p = axes_inches(fig, (1.50, 1.15, 4.55, 0.95))
    # Supporting panels, 58 % of the main panel width.
    ax_b = axes_inches(fig, (7.00, 3.45, 2.65, 1.80))
    ax_w = axes_inches(fig, (7.00, 1.15, 2.65, 1.55))

    # ---- (a) centre: the calibration curve -------------------------------
    m = (CAL >= xlo) & (CAL <= xhi)
    mu, sg = curve_mu(CAL), curve_sigma(CAL)
    ax_c.fill_between(CAL[m], (mu - sg)[m], (mu + sg)[m], color=CURVE,
                      alpha=0.24, lw=0, zorder=2)
    ax_c.plot(CAL[m], mu[m], color=CURVE, lw=1.3, zorder=3)

    for lo, hi in hpd95[MAIN_IDX]:
        ax_c.axvspan(lo, hi, color=POST, alpha=0.10, lw=0, zorder=1)
        ax_p.axvspan(lo, hi, color=POST, alpha=0.10, lw=0, zorder=1)
    ax_c.axhspan(R - SR, R + SR, color=MEAS, alpha=0.12, lw=0, zorder=1)
    ax_c.axhline(R, color=MEAS, lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax_m.axhline(R, color=MEAS, lw=0.9, ls=(0, (4, 2)), zorder=4)

    ylo, yhi = shared_limits([mu[m] - sg[m], mu[m] + sg[m],
                              np.array([R - 4 * SR, R + 4 * SR])], pad=0.04)
    ax_c.set_xlim(xlo, xhi)
    ax_c.set_ylim(ylo, yhi)
    ax_c.set_xticklabels([])
    ax_c.set_yticklabels([])
    ax_c.tick_params(axis="y", length=2.2)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    x_arrow = xlo + 0.055 * (xhi - xlo)
    ax_c.annotate("", xy=(x_arrow + 0.10 * (xhi - xlo), R), xytext=(xlo, R),
                  arrowprops=dict(arrowstyle="-|>", color=MEAS, lw=0.9,
                                  shrinkA=0, shrinkB=0), zorder=5)
    seg0 = hpd95[MAIN_IDX][0]
    ax_c.annotate("", xy=(0.5 * (seg0[0] + seg0[1]), ylo),
                  xytext=(0.5 * (seg0[0] + seg0[1]), R - 0.9 * SR),
                  arrowprops=dict(arrowstyle="-|>", color=POST, lw=0.9,
                                  shrinkA=0, shrinkB=0), zorder=5)
    ax_c.text(xhi - 0.02 * (xhi - xlo), yhi - 0.05 * (yhi - ylo),
              "IntCal-style curve, $\\pm$1$\\sigma$ envelope", ha="right",
              va="top", fontsize=7.0, color=CURVE)
    ax_c.text(xlo + 0.02 * (xhi - xlo), R + 0.035 * (yhi - ylo),
              f"{lab}  {R:.0f} $\\pm$ {SR:.0f} BP", ha="left", va="bottom",
              fontsize=7.0, color=MEAS)
    wide = max(hpd95[MAIN_IDX], key=lambda s: s[1] - s[0])
    ax_c.text(0.5 * (wide[0] + wide[1]), ylo + 0.035 * (yhi - ylo),
              "plateau: one BP age,\nseveral calendar solutions", ha="center",
              va="bottom", fontsize=6.6, color=POST, linespacing=1.25)

    # ---- (a) left: the laboratory determination --------------------------
    bp = np.linspace(ylo, yhi, 400)
    gm = np.exp(-0.5 * ((bp - R) / SR) ** 2) / (SR * np.sqrt(2 * np.pi))
    ax_m.fill_betweenx(bp, 0.0, gm, color=MEAS, alpha=0.22, lw=0)
    ax_m.plot(gm, bp, color=MEAS, lw=1.1)
    ax_m.set_ylim(ylo, yhi)
    ax_m.set_xlim(0.0, gm.max() * 1.18)
    ax_m.invert_xaxis()
    ax_m.set_xticks([])
    ax_m.set_ylabel("radiocarbon age (BP)", fontsize=8.0)
    ax_m.tick_params(axis="y", labelsize=7.0, length=2.2)
    for s in ("top", "right", "bottom"):
        ax_m.spines[s].set_visible(False)
    ax_m.text(0.06, 0.985, "measured\ndetermination", transform=ax_m.transAxes,
              ha="left", va="top", fontsize=6.6, color=MEAS, linespacing=1.25)

    # ---- (a) bottom: the calibrated calendar density ---------------------
    pk = post.max()
    ax_p.fill_between(CAL[m], 0.0, post[m] / pk, color=POST, alpha=0.28, lw=0,
                      zorder=2)
    ax_p.plot(CAL[m], post[m] / pk, color=POST, lw=1.1, zorder=3)
    for lo, hi in hpd95[MAIN_IDX]:
        ax_p.plot([lo, hi], [-0.30, -0.30], color=POST, lw=2.6,
                  solid_capstyle="butt", alpha=0.45, zorder=4)
    for lo, hi in hpd68[MAIN_IDX]:
        ax_p.plot([lo, hi], [-0.30, -0.30], color=POST, lw=2.6,
                  solid_capstyle="butt", zorder=5)
    ax_p.set_xlim(xlo, xhi)
    ax_p.set_ylim(-0.52, 1.16)
    ax_p.set_yticks([])
    ax_p.set_ylabel("rel. prob.", fontsize=8.0)
    ax_p.set_xlabel("calibrated calendar age (cal BC)", fontsize=8.0)
    ax_p.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{-v:.0f}"))
    ax_p.tick_params(axis="x", labelsize=7.0, length=2.6)
    for s in ("top", "right", "left"):
        ax_p.spines[s].set_visible(False)
    ax_p.text(0.012, 0.97,
              f"68.2 % HPD  {span_text(hpd68[MAIN_IDX])} cal BC\n"
              f"95.4 % HPD  {span_text(hpd95[MAIN_IDX])} cal BC",
              transform=ax_p.transAxes, ha="left", va="top", fontsize=6.6,
              color=INK, linespacing=1.35)

    # ---- (b) calibrated intervals in stratigraphic order -----------------
    ylab = [s[0] for s in SAMPLES]
    for i in range(len(SAMPLES)):
        col = POST if i == MAIN_IDX else RULE
        for lo, hi in hpd95[i]:
            ax_b.plot([lo, hi], [i, i], color=col, lw=4.2, alpha=0.32,
                      solid_capstyle="butt", zorder=2)
        for lo, hi in hpd68[i]:
            ax_b.plot([lo, hi], [i, i], color=col, lw=4.2, alpha=0.95,
                      solid_capstyle="butt", zorder=3)
        med = float(np.median(chain[:, i]))
        ax_b.plot([med], [i], marker="|", ms=5.0, mew=1.0, color="white",
                  zorder=4)
    ax_b.set_xlim(xlo, xhi)
    # Extra room at the bottom so the legend sits under the bars, not on them.
    ax_b.set_ylim(-2.5, len(SAMPLES) - 0.4)
    ax_b.set_yticks(range(len(SAMPLES)))
    ax_b.set_yticklabels(ylab, fontsize=7.0)
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{-v:.0f}"))
    ax_b.tick_params(labelsize=7.0, length=2.4)
    ax_b.set_xlabel("calibrated calendar age (cal BC)", fontsize=8.0)
    ax_b.set_title("Eight dates, deepest context at the bottom",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_b.grid(axis="x", lw=0.5, color=HAIR, alpha=0.7)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.legend(handles=[
        Line2D([0], [0], color=RULE, lw=4.2, alpha=0.95, label="68.2 % HPD"),
        Line2D([0], [0], color=RULE, lw=4.2, alpha=0.32, label="95.4 % HPD"),
        Line2D([0], [0], color=POST, lw=4.2, alpha=0.95,
               label=f"{name}, panel (a)")],
        loc="lower left", fontsize=6.6, frameon=False, handlelength=1.2,
        borderpad=0.1, labelspacing=0.22, ncol=3, columnspacing=0.9,
        handletextpad=0.4)

    # ---- (c) what the sequence model buys --------------------------------
    for i in range(len(SAMPLES)):
        ax_w.plot([w_mod[i], w_raw[i]], [i, i], color=HAIR, lw=1.4, zorder=1)
        ax_w.plot([w_raw[i]], [i], marker="o", ms=4.0, color="white",
                  mec=MUTED, mew=1.0, zorder=3)
        ax_w.plot([w_mod[i]], [i], marker="o", ms=4.0, color=POST, mec="white",
                  mew=0.6, zorder=4)
        ax_w.text(w_mod[i] - 6.0, i, f"-{cut[i]:.0%}", ha="right", va="center",
                  fontsize=6.2, color=SUB)
    ax_w.set_xlim(shared_limits([w_mod - 60.0, w_raw], pad=0.08, lo=0.0))
    ax_w.set_ylim(-0.8, len(SAMPLES) - 0.2)
    ax_w.set_yticks(range(len(SAMPLES)))
    ax_w.set_yticklabels(ylab, fontsize=7.0)
    ax_w.tick_params(labelsize=7.0, length=2.4)
    ax_w.set_xlabel("width of the 95.4 % interval (yr)", fontsize=8.0)
    ax_w.set_title("Stratigraphic order narrows every date",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_w.grid(axis="x", lw=0.5, color=HAIR, alpha=0.7)
    ax_w.set_axisbelow(True)
    for s in ("top", "right"):
        ax_w.spines[s].set_visible(False)
    ax_w.legend(handles=[
        Line2D([0], [0], marker="o", ls="none", ms=4.0, color="white",
               mec=MUTED, label="unmodelled"),
        Line2D([0], [0], marker="o", ls="none", ms=4.0, color=POST,
               label="sequence model")],
        loc="lower right", fontsize=6.6, frameon=False, handletextpad=0.4,
        borderpad=0.15, labelspacing=0.28)

    panel_labels_inches(fig, [ax_m, ax_b, ax_w], dx=-0.34, dy=0.14)


    print(f"main sample {lab} ranges={n_ranges} "
          f"95.4% = {span_text(hpd95[MAIN_IDX])} cal BC "
          f"({total_width(hpd95[MAIN_IDX]):.0f} yr)")
    for i, (nm, lb, r, sr) in enumerate(SAMPLES):
        print(f"  {nm} {lb} {r:.0f}+-{sr:.0f} BP  raw {w_raw[i]:6.1f} yr  "
              f"modelled {w_mod[i]:6.1f} yr  -{cut[i]:.0%}")
    savefig(fig, "888_arq_c14_calibration.png")


if __name__ == "__main__":
    main()
