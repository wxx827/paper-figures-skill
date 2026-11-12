"""Pure-tone audiogram in the clinical format, with speech and case-mix panels.

The main panel keeps the two conventions an audiologist reads without
thinking: octave frequencies on a logarithmic abscissa, and hearing level
running *downwards*, so that a lower trace always means a worse ear.  Shaded
bands behind the traces carry the severity grades, and the four standard
transducer symbols (right circle, left cross, bone-conduction carets)
separate the ears without a colour-only cue.

The two subordinate panels underneath the same colour code supply the
evidence: how much speech each ear still resolves as the presentation level
rises, and how the grades are distributed over the clinic population.  Every
number printed on the figure - the pure-tone averages, the air-bone gap, the
maximum word scores, the wedge percentages - is measured from the arrays that
are plotted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten, get_palette
from figkit import panel_labels_inches

RED = "#b23a48"       # right ear
BLUE = "#3b5b92"      # left ear
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

FREQS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
FTICKS = ["125", "250", "500", "1k", "2k", "4k", "8k"]
PTA_IDX = [2, 3, 4]                       # 500 / 1000 / 2000 Hz

# Severity grades in dB HL (WHO-style cuts) with a monotone pale-blue ramp.
GRADES = [(-10.0, 20.0, "normal"),
          (20.0, 40.0, "mild"),
          (40.0, 55.0, "moderate"),
          (55.0, 70.0, "moderately severe"),
          (70.0, 90.0, "severe"),
          (90.0, 120.0, "profound")]
BAND_FC = [lighten(BLUE, a) for a in (0.975, 0.935, 0.895, 0.855, 0.815, 0.775)]


def round5(a):
    """Audiometers step in 5 dB; keep the synthetic data on the same grid."""
    return np.round(np.asarray(a, dtype=float) / 5.0) * 5.0


def ears(seed=7):
    rng = np.random.default_rng(seed)
    ac_r = round5(np.array([12, 16, 21, 30, 43, 58, 66], float)
                  + rng.normal(0.0, 2.4, FREQS.size))
    ac_l = round5(np.array([19, 23, 29, 39, 53, 68, 75], float)
                  + rng.normal(0.0, 2.4, FREQS.size))
    gap_r = np.array([16, 17, 18, 15, 11, 7], float)
    gap_l = np.array([4, 4, 5, 4, 3, 2], float)
    bc_r = round5(ac_r[:6] - gap_r - rng.normal(0.0, 1.4, 6))
    bc_l = round5(ac_l[:6] - gap_l - rng.normal(0.0, 1.4, 6))
    return ac_r, ac_l, np.maximum(bc_r, -10.0), np.maximum(bc_l, -10.0)


def speech(srt_r, srt_l):
    lev = np.linspace(0.0, 110.0, 220)

    def curve(srt, pbmax, slope=6.5, rollover=0.0):
        s = pbmax / (1.0 + np.exp(-(lev - srt) / slope))
        s = s - rollover * np.clip(lev - (srt + 28.0), 0.0, None)
        return np.clip(s, 0.0, 100.0)

    return (lev, curve(20.0, 100.0), curve(srt_r, 92.0),
            curve(srt_l, 80.0, rollover=0.26))


def case_mix():
    return (np.array([412.0, 268.0, 151.0, 86.0, 39.0]),
            ["normal", "mild", "moderate", "severe", "profound"])


def main_panel(ax, ac_r, ac_l, bc_r, bc_l):
    for (lo, hi, name), fc in zip(GRADES, BAND_FC):
        ax.axhspan(lo, hi, facecolor=fc, edgecolor="none", zorder=0)
        ax.text(8900.0, (lo + hi) / 2.0, name, ha="left", va="center",
                fontsize=6.4, color=SUB, clip_on=False)

    ax.plot(FREQS, ac_r, color=RED, lw=1.4, marker="o", ms=6.0, mfc="white",
            mew=1.3, mec=RED, zorder=5)
    ax.plot(FREQS, ac_l, color=BLUE, lw=1.4, ls=(0, (5, 2.2)), marker="x",
            ms=6.2, mew=1.4, mec=BLUE, zorder=5)
    ax.plot(FREQS[:6], bc_r, color=RED, lw=0.0, marker="<", ms=6.0,
            mfc="none", mew=1.2, mec=RED, zorder=5)
    ax.plot(FREQS[:6], bc_l, color=BLUE, lw=0.0, marker=">", ms=6.0,
            mfc="none", mew=1.2, mec=BLUE, zorder=5)

    pta_r = float(ac_r[PTA_IDX].mean())
    pta_l = float(ac_l[PTA_IDX].mean())
    abg_r = float(np.mean(ac_r[:6] - bc_r))
    abg_l = float(np.mean(ac_l[:6] - bc_l))

    k = 4                                   # 2 kHz: widest air-bone gap shown
    ax.annotate("", xy=(FREQS[k], ac_r[k]), xytext=(FREQS[k], bc_r[k]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=0.8,
                                shrinkA=3.0, shrinkB=3.0), zorder=6)
    ax.text(FREQS[k] * 1.08, (ac_r[k] + bc_r[k]) / 2.0,
            f"air-bone gap\n{ac_r[k] - bc_r[k]:.0f} dB", ha="left",
            va="center", fontsize=6.6, color=RED, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(105.0, 9600.0)
    ax.set_xticks(FREQS)
    ax.set_xticklabels(FTICKS)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(120.0, -10.0)               # clinical convention: down is worse
    ax.set_yticks(np.arange(-10.0, 121.0, 10.0))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("hearing level (dB HL)")
    ax.grid(color="#cccccc", lw=0.5, alpha=0.8, zorder=1)
    ax.set_axisbelow(False)
    for line in ax.get_ygridlines() + ax.get_xgridlines():
        line.set_zorder(1)

    handles = [
        Line2D([], [], color=RED, lw=1.4, marker="o", ms=5.4, mfc="white",
               mew=1.2, label="right air conduction"),
        Line2D([], [], color=BLUE, lw=1.4, ls=(0, (5, 2.2)), marker="x",
               ms=5.6, mew=1.3, label="left air conduction"),
        Line2D([], [], color=RED, lw=0.0, marker="<", ms=5.4, mfc="none",
               mew=1.1, label="right bone conduction"),
        Line2D([], [], color=BLUE, lw=0.0, marker=">", ms=5.4, mfc="none",
               mew=1.1, label="left bone conduction"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.0,
                    frameon=True, framealpha=0.92, facecolor="white",
                    edgecolor="#cccccc", borderpad=0.4, handlelength=1.8,
                    labelspacing=0.32)
    leg.set_zorder(8)

    ax.set_title(f"Right PTA {pta_r:.0f} dB HL (conductive component), "
                 f"left PTA {pta_l:.0f} dB HL", fontsize=9.0,
                 fontweight="bold", color=INK)
    return pta_r, pta_l, abg_r, abg_l


def speech_panel(ax, lev, ref, sr_r, sr_l):
    ax.plot(lev, ref, color=GREY, lw=1.0, ls=(0, (4, 2.4)),
            label="normal reference")
    ax.plot(lev, sr_r, color=RED, lw=1.5, label="right ear")
    ax.plot(lev, sr_l, color=BLUE, lw=1.5, ls=(0, (5, 2.2)), label="left ear")

    i_l = int(np.argmax(sr_l))
    ax.plot([lev[i_l]], [sr_l[i_l]], marker="o", ms=3.4, color=BLUE,
            mfc="white", mew=1.0)
    ax.plot([lev[i_l], lev[i_l]], [sr_l[i_l] - 6.0, 36.0], color=BLUE,
            lw=0.7)
    ax.text(108.0, 32.0, f"PB max {sr_l[i_l]:.0f} %\nwith rollover",
            ha="right", va="top", fontsize=6.5, color=BLUE, linespacing=1.3)

    ax.set_xlim(0.0, 110.0)
    ax.set_ylim(0.0, 134.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("presentation level (dB HL)", fontsize=7.4)
    ax.set_ylabel("word recognition (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              handlelength=1.5, borderaxespad=0.2, labelspacing=0.28)
    ax.set_title("Speech recognition against level", fontsize=8.2,
                 fontweight="bold", color=INK)


def mix_panel(ax, counts, names):
    cols = get_palette("cb_sequential_blue", 7)[1:6]
    total = counts.sum()
    wedges, _, autotexts = ax.pie(
        counts, startangle=90.0, counterclock=False, colors=cols,
        radius=1.0, autopct=lambda p: f"{p:.0f}%", pctdistance=0.76,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=0.8),
        textprops=dict(fontsize=6.4, color=INK))
    for t, c in zip(autotexts, cols):
        t.set_color("white" if c in cols[2:] else INK)
    ax.text(0.0, 0.0, f"n = {total:.0f}\nears", ha="center", va="center",
            fontsize=7.0, color=SUB, linespacing=1.35)
    ax.set_aspect("equal")
    ax.legend(wedges, [f"{n}  {c / total * 100:.0f} %"
                       for n, c in zip(names, counts)],
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.5,
              frameon=False, handlelength=1.1, labelspacing=0.40,
              borderaxespad=0.0)
    ax.set_title("Grade mix in the clinic cohort", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    ac_r, ac_l, bc_r, bc_l = ears()
    counts, names = case_mix()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.46, hspace=0.66)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_sp = fig.add_subplot(gs[0, 1])
    ax_mix = fig.add_subplot(gs[1, 1])

    pta_r, pta_l, abg_r, abg_l = main_panel(ax_main, ac_r, ac_l, bc_r, bc_l)
    lev, ref, sr_r, sr_l = speech(pta_r + 18.0, pta_l + 24.0)
    speech_panel(ax_sp, lev, ref, sr_r, sr_l)
    mix_panel(ax_mix, counts, names)

    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_sp, ax_mix], dx=-0.46, dy=0.12)
    savefig(fig, "726_sen_audiogram.png")


if __name__ == "__main__":
    main()
