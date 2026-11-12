"""Fatigue assessment of a welded bridge detail from a monitored strain record.

Main panel: the rainflow spectrum of one representative half-hour record
scaled to a year, with the damage each stress-range bin contributes read on
the right-hand axis, so the reader can see that the bins carrying the cycles
and the bins carrying the damage are not the same ones.  Auxiliary (both at
58 % of the main panel width): a minute of the stress history with its
turning points and the largest counted cycle marked, and the detail category
curve with the annual spectrum laid against it.

The counting is a real four-point rainflow over the turning points of the
plotted signal, not a histogram invented to look plausible.  Allowable lives
come from the EC3-style curve with slope three above the constant-amplitude
limit and five below it, damage is the Palmgren-Miner sum of n / N over the
counted bins, the equivalent stress range is the cubic mean of the same
spectrum, and the remaining life is what is left of unity at that rate.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"

E_STEEL = 210000.0        # MPa, for the microstrain axis
DET_CAT = 71.0            # MPa at 2e6 cycles
N_C, N_D, N_L = 2.0e6, 5.0e6, 1.0e8
SIG_D = DET_CAT * (N_C / N_D) ** (1.0 / 3.0)
SIG_L = SIG_D * (N_D / N_L) ** (1.0 / 5.0)

FS, DUR = 10.0, 1800.0    # sampling rate (Hz) and record length (s)
RATE = 0.050              # truck arrivals per second
AGE = 18.0                # years already in service
GATE = 4.0                # MPa, rainflow filter for gauge noise
BINS = np.arange(GATE, 108.0, 4.0)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def stress_record(rng):
    """Half an hour of stress at the weld toe, in MPa."""
    t = np.arange(0.0, DUR, 1.0 / FS)
    s = 1.6 * np.sin(2.0 * np.pi * t / 900.0)          # slow thermal drift
    n_ev = rng.poisson(RATE * DUR)
    t0 = rng.uniform(2.0, DUR - 8.0, n_ev)
    amp = rng.lognormal(np.log(13.0), 0.50, n_ev)
    wid = rng.uniform(0.9, 2.2, n_ev)
    for a, c, w in zip(amp, t0, wid):
        env = np.exp(-0.5 * ((t - c) / w) ** 2)
        axle = 1.0 + 0.30 * np.sin(2.0 * np.pi * (t - c) / (0.55 * w))
        s = s + a * env * axle
    s = s + rng.normal(0.0, 0.55, t.size)
    return t, s


def turning_points(y):
    """Indices of the peaks and valleys of a series, ends included."""
    d = np.diff(y)
    d[d == 0.0] = 1e-12
    idx = np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0] + 1
    return np.concatenate([[0], idx, [y.size - 1]])


def rainflow(points):
    """Four-point rainflow counting; returns (range, count) pairs."""
    stack, cycles = [], []
    for p in points:
        stack.append(float(p))
        while len(stack) >= 4:
            s1, s2, s3, s4 = stack[-4:]
            r1, r2, r3 = abs(s2 - s1), abs(s3 - s2), abs(s4 - s3)
            if r2 <= r1 and r2 <= r3:
                cycles.append((r2, 1.0))
                del stack[-3:-1]
            else:
                break
    for a, b in zip(stack[:-1], stack[1:]):
        cycles.append((abs(b - a), 0.5))
    return cycles


def allowable(sig):
    """Cycles to failure at a stress range, on the two-slope design curve."""
    sig = np.asarray(sig, dtype=float)
    n = np.where(sig >= SIG_D,
                 N_C * (DET_CAT / np.maximum(sig, 1e-9)) ** 3,
                 N_D * (SIG_D / np.maximum(sig, 1e-9)) ** 5)
    return np.where(sig < SIG_L, np.inf, n)


def spectrum(cycles, scale):
    """Annual cycle count and Miner damage per stress-range bin.

    Ranges under the gate are discarded first: at 10 Hz the gauge noise alone
    produces tens of millions of sub-MPa reversals a year, which carry no
    damage at all but would otherwise swamp both the histogram and any mean
    taken over it.
    """
    rng_arr = np.array([c[0] for c in cycles if c[0] >= GATE])
    cnt_arr = np.array([c[1] for c in cycles if c[0] >= GATE]) * scale
    mid = 0.5 * (BINS[:-1] + BINS[1:])
    n_year = np.array([cnt_arr[(rng_arr >= a) & (rng_arr < b)].sum()
                       for a, b in zip(BINS[:-1], BINS[1:])])
    dmg = n_year / allowable(mid)
    return mid, n_year, dmg


def main_panel(ax, mid, n_year, dmg, sig_eq):
    w = 0.86 * (BINS[1] - BINS[0])
    keep = n_year > 0.0
    ax.bar(mid[keep], n_year[keep], width=w, bottom=1.0,
           facecolor=lighten(BLUE, 0.42), edgecolor=BLUE, linewidth=0.6,
           zorder=3, label="counted cycles per year")
    ax.set_yscale("log")
    ax.set_ylim(1.0, max(n_year.max() * 12.0, 10.0))
    ax.set_xlim(0.0, mid[keep].max() + 8.0)
    ax.set_xlabel("stress range (MPa)")
    ax.set_ylabel("cycles per year (log scale)")
    ax.grid(axis="y", lw=0.5, alpha=0.30)
    for side in ("top",):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    share = 100.0 * dmg / dmg.sum()
    ax2.fill_between(mid, 0.0, share, step="mid", facecolor=ORANGE,
                     alpha=0.30, lw=0.0, zorder=4)
    ax2.step(mid, share, where="mid", color=ORANGE, lw=1.6, zorder=5,
             label="share of the annual damage")
    ax2.set_ylim(0.0, share.max() * 1.45)
    ax2.set_ylabel("share of annual Miner damage (%)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)

    # Three vertical rules that would otherwise stack their captions; each
    # caption is placed at its own height in axes fraction.
    marks = ((sig_eq, "equivalent range", TEAL, 0.30),
             (SIG_L, "cut-off limit", MUTED, 0.20),
             (SIG_D, "constant-amplitude limit", RED, 0.10))
    for val, lab, col, yf in marks:
        ax.axvline(val, color=col, lw=1.0, ls=(0, (3.0, 2.0)), zorder=6)
        ax.text(val + 1.0, yf, "{:s} {:.1f} MPa".format(lab, val),
                transform=ax.get_xaxis_transform(), fontsize=6.2, color=col,
                ha="left", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.88))

    peak = int(np.argmax(share))
    left = mid[peak] > 0.55 * ax.get_xlim()[1]
    ax2.annotate("{:.0f} - {:.0f} MPa: {:.0f} % of the damage\nfrom {:.2f} % "
                 "of the cycles"
                 .format(BINS[peak], BINS[peak + 1], share[peak],
                         100.0 * n_year[peak] / n_year.sum()),
                 xy=(mid[peak], share[peak]),
                 xytext=(mid[peak] + (-6.0 if left else 6.0),
                         share[peak] * 1.20), fontsize=6.6, color=ORANGE,
                 ha="right" if left else "left", va="bottom", zorder=9,
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8,
                                 shrinkA=2, shrinkB=3))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.4,
              frameon=True, facecolor="white", edgecolor="#dddddd",
              framealpha=0.94, handlelength=1.6, borderaxespad=0.4)
    ax.set_title("Rainflow spectrum against the damage it produces",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def trace_panel(ax, t, s, tp, t0, t1):
    m = (t >= t0) & (t <= t1)
    ax.plot(t[m], s[m], color=BLUE, lw=0.8, zorder=4)
    keep = tp[(t[tp] >= t0) & (t[tp] <= t1)]
    big = keep[np.abs(s[keep] - np.median(s[keep])) > 2.0]
    ax.plot(t[big], s[big], lw=0.0, marker="o", ms=2.0, mfc=RED, mec="none",
            zorder=6)

    if keep.size > 4:
        # The tallest event in the window, measured from its peak down to the
        # deepest turning point beside it, which is the cycle rainflow pairs.
        k = int(np.argmax(s[keep]))
        near = keep[max(k - 4, 0):k + 5]
        top, bot = s[keep[k]], float(np.min(s[near]))
        xp = t[keep[k]]
        ax.annotate("", xy=(xp, top), xytext=(xp, bot), zorder=7,
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8,
                                    mutation_scale=6))
        ax.text(xp + 1.4, 0.5 * (top + bot),
                "largest cycle in this minute\n{:.0f} MPa".format(top - bot),
                fontsize=6.0, color=INK, ha="left", va="center",
                linespacing=1.3, zorder=8)

    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (s)", fontsize=6.8)
    ax.set_ylabel("stress at the weld toe (MPa)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top",):
        ax.spines[side].set_visible(False)
    lo, hi = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim(1e6 * lo / E_STEEL, 1e6 * hi / E_STEEL)
    ax2.set_ylabel(r"gauge strain ($\mu\varepsilon$)", fontsize=6.8,
                   color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=6)
    ax2.spines["top"].set_visible(False)
    ax.plot([], [], lw=0.0, marker="o", ms=2.0, mfc=RED, mec="none",
            label="turning point")
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.0, borderaxespad=0.3)
    ax.set_title("One minute of the monitored record", fontsize=8.0,
                 fontweight="bold", color=INK)


def sn_panel(ax, mid, n_year, sig_eq, d_year, life):
    ng = np.logspace(4.0, 8.3, 300)
    sg = np.where(ng <= N_D, DET_CAT * (N_C / ng) ** (1.0 / 3.0),
                  SIG_D * (N_D / ng) ** (1.0 / 5.0))
    sg = np.where(ng > N_L, SIG_L, sg)
    ax.plot(ng, sg, color=INK, lw=1.6, zorder=5,
            label="detail category {:.0f}".format(DET_CAT))
    ax.axhline(SIG_D, color=RED, lw=0.8, ls=(0, (3.0, 2.0)), zorder=4)
    ax.axhline(SIG_L, color=MUTED, lw=0.8, ls=(0, (3.0, 2.0)), zorder=4)

    keep = n_year > 0.5
    ax.plot(n_year[keep] * life, mid[keep], color=ORANGE, lw=1.4,
            marker="o", ms=2.4, mfc=ORANGE, mec="none", zorder=6,
            label="spectrum over {:.0f} years".format(life))
    # Drawn as a segment rather than an axhline so it leaves the lower left
    # corner free for the damage caption.
    ax.plot([1.1e6, 2.6e8], [sig_eq, sig_eq], color=TEAL, lw=1.0,
            ls=(0, (3.5, 2.0)), zorder=7)
    ax.text(2.4e8, sig_eq * 1.06, "equivalent range {:.1f} MPa"
            .format(sig_eq), fontsize=6.0, color=TEAL, ha="right",
            va="bottom", zorder=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 3e8)
    ax.set_ylim(14.0, 160.0)
    ax.set_yticks([20, 30, 50, 70, 100, 150])
    ax.set_yticklabels(["20", "30", "50", "70", "100", "150"])
    ax.set_xlabel("number of cycles", fontsize=6.8)
    ax.set_ylabel("stress range (MPa)", fontsize=6.8)
    ax.grid(which="both", lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(1.6e4, 15.0, "Miner damage {:.4f} per year\nremaining life "
            "{:.0f} years after {:.0f} in service"
            .format(d_year, life, AGE), fontsize=6.2, color=INK, ha="left",
            va="bottom", linespacing=1.35, zorder=9)
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.24)
    ax.set_title("Spectrum laid against the design curve", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(2026)

    t, s = stress_record(rng)
    tp = turning_points(s)
    cycles = rainflow(s[tp])
    scale = 365.0 * 24.0 * 3600.0 / DUR
    mid, n_year, dmg = spectrum(cycles, scale)

    d_year = float(dmg.sum())
    life = (1.0 - d_year * AGE) / d_year
    w = n_year / n_year.sum()
    sig_eq = float((np.sum(w * mid ** 3)) ** (1.0 / 3.0))

    print("cycles/yr {:.3g}  damage/yr {:.4f}  life {:.0f} yr  eq {:.1f} MPa"
          .format(n_year.sum(), d_year, life, sig_eq))
    assert life > 0.0, "the detail must have life left"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_tr = axes_inches(fig, AUX_T)
    ax_sn = axes_inches(fig, AUX_B)

    main_panel(ax_main, mid, n_year, dmg, sig_eq)
    trace_panel(ax_tr, t, s, tp, 300.0, 360.0)
    sn_panel(ax_sn, mid, n_year, sig_eq, d_year, life)

    # How small is the tail that does the harm?  Take bins from the largest
    # range downward until four fifths of the damage is accounted for.
    order = np.argsort(mid)[::-1]
    cum = np.cumsum(dmg[order]) / dmg.sum()
    kk = order[:int(np.searchsorted(cum, 0.80) + 1)]


    panel_labels_inches(fig, [ax_main, ax_tr, ax_sn], dx=-0.62, dy=0.14)
    savefig(fig, "988_shm_fatigue_rainflow.png")


if __name__ == "__main__":
    main()
