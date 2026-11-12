"""Beat-by-beat arrhythmia detection: signal, tachogram and classifier output.

Three panels stacked on one shared 20 s time axis, so a reader can drop a
vertical line anywhere and read the same instant three ways: the raw waveform,
the RR interval that beat terminated, and what the detector called it.  The
point of the layout is that the evidence for "this beat is ectopic" is visible
in two independent channels at once - a premature ventricular complex is both
a wide bizarre QRS *and* a short RR followed by a compensatory pause.

Ectopic beats are generated first as a schedule of (time, class) pairs and the
waveform is synthesised from that schedule, so the coloured markers, the
tachogram, the confusion counts and the class proportions all derive from one
ground truth.  The detector output is the same schedule with two deliberate
errors, and the reported accuracy is counted from the comparison rather than
quoted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TRACE = "#1f2933"
BAND = "#cfe1ee"
GRIDC = "#e2e2e2"

# House order: normal = blue, the emphasis class (ventricular) = red, atrial
# = ochre.  Three classes, so no grouping into an "other" bucket is needed.
CLASS = {
    "N": ("Sinus (N)", "#3b5b92"),
    "V": ("Ventricular ectopic (V)", "#b23a48"),
    "A": ("Atrial ectopic (A)", "#c0603f"),
}

FS = 500.0
DUR = 20.0
RR0 = 60.0 / 74.0

# Gaussian wave sets per beat class: (amplitude mV, centre s, width s).
MORPH = {
    "N": [(0.14, -0.186, 0.021), (-0.09, -0.023, 0.0090),
          (1.15, 0.0, 0.0102), (-0.28, 0.029, 0.0120),
          (0.33, 0.205, 0.047)],
    "A": [(0.09, -0.135, 0.017), (-0.07, -0.023, 0.0090),
          (1.02, 0.0, 0.0104), (-0.24, 0.029, 0.0122),
          (0.29, 0.198, 0.046)],
    # No P wave, broad notched QRS, T discordant with the main deflection.
    "V": [(1.42, 0.0, 0.0300), (-0.95, 0.062, 0.0330),
          (-0.52, 0.215, 0.0620)],
}

# Sinus-beat indices at which an ectopic interrupts the rhythm.
PLAN = {5: "V", 11: "A", 15: "V", 19: "V", 22: "A"}


def beat_schedule(rng):
    """(times, classes): sinus train interrupted by scheduled ectopics."""
    times, kinds, t, k = [], [], 0.55, 0
    while t < DUR - 0.25:
        kind = PLAN.get(k, "N")
        if kind == "V":
            # Premature, then a full compensatory pause: the sinus node keeps
            # its own clock, so the *next* sinus beat lands back on schedule.
            times.append(t - 0.42 * RR0)
            kinds.append("V")
            t += RR0 * (1.0 + rng.normal(0.0, 0.018))
        elif kind == "A":
            # Premature and it resets the sinus node -> incomplete pause.
            tp = t - 0.30 * RR0
            times.append(tp)
            kinds.append("A")
            t = tp + RR0 * 1.05
        else:
            times.append(t)
            kinds.append("N")
            t += RR0 * (1.0 + rng.normal(0.0, 0.020))
        k += 1
    return np.asarray(times), kinds


def synth(t, times, kinds, rng):
    y = np.zeros_like(t)
    for bt, kind in zip(times, kinds):
        for amp, mu, sd in MORPH[kind]:
            c = bt + mu
            m = np.abs(t - c) < 6.0 * sd
            y[m] += amp * np.exp(-0.5 * ((t[m] - c) / sd) ** 2)
    y += 0.028 * np.sin(2 * np.pi * 0.19 * t + rng.uniform(0, 6.28))
    y += 0.0055 * np.convolve(rng.standard_normal(t.size),
                              np.ones(5) / 5.0, mode="same")
    return y


def main():
    set_pub_style()
    rng = np.random.default_rng(5471)

    t = np.arange(0.0, DUR, 1.0 / FS)
    times, kinds = beat_schedule(rng)
    ecg = synth(t, times, kinds, rng)

    # Detector output = truth with two errors, so accuracy can be counted.
    pred = list(kinds)
    pred[9] = "A"                                   # false atrial alarm
    idx_v = [i for i, k in enumerate(kinds) if k == "V"]
    pred[idx_v[-1]] = "N"                           # one missed ventricular
    acc = float(np.mean([a == b for a, b in zip(kinds, pred)]))

    rr = np.diff(times) * 1000.0
    rr_t = times[1:]
    rr_k = kinds[1:]
    # A true N-N interval needs *both* ends sinus: the pause that follows a
    # ventricular beat also terminates on a sinus beat, and letting it into the
    # reference band would widen it until nothing looked abnormal.
    nn = rr[[kinds[i] == "N" and kinds[i + 1] == "N" for i in range(rr.size)]]
    lo, hi = nn.mean() - 2 * nn.std(), nn.mean() + 2 * nn.std()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.66, 0.30],
                          hspace=0.34, wspace=0.20)
    ax_ecg = fig.add_subplot(gs[0, 0])
    ax_rr = fig.add_subplot(gs[1, 0], sharex=ax_ecg)
    ax_bar = fig.add_subplot(gs[2, 0], sharex=ax_ecg)
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_prop = fig.add_subplot(gs[1:, 1])

    # ---- (a) raw ECG with ectopic beats marked -----------------------------
    ylo, yhi = shared_limits([ecg], pad=0.16)
    for bt, kind in zip(times, kinds):
        if kind == "N":
            continue
        col = CLASS[kind][1]
        ax_ecg.axvspan(bt - 0.16, bt + 0.20, color=col, alpha=0.11, lw=0,
                       zorder=0)
        ax_ecg.plot([bt], [yhi * 0.93], marker="v", ms=4.2, color=col,
                    zorder=5)
    ax_ecg.plot(t, ecg, color=TRACE, lw=0.75, zorder=3)
    ax_ecg.set_ylim(ylo, yhi)
    ax_ecg.set_ylabel("lead II (mV)", fontsize=8.0)
    ax_ecg.set_title("Single-lead recording with annotated ectopic beats",
                     fontsize=8.8, fontweight="bold", loc="left")
    ax_ecg.tick_params(labelbottom=False, labelsize=7.0)
    ax_ecg.grid(axis="y", color=GRIDC, lw=0.5)

    handles = [Line2D([], [], color=c, lw=2.0, label=n)
               for n, c in CLASS.values()]
    ax_ecg.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                  fontsize=7.0, bbox_to_anchor=(0.5, 1.30),
                  handlelength=1.4, columnspacing=1.6)

    # ---- (b) RR tachogram --------------------------------------------------
    ax_rr.axhspan(lo, hi, color=BAND, alpha=0.55, lw=0, zorder=0)
    ax_rr.plot(rr_t, rr, color="#9a9a9a", lw=0.7, zorder=2)
    for kind in ("N", "A", "V"):
        m = np.array([k == kind for k in rr_k])
        if m.any():
            ax_rr.plot(rr_t[m], rr[m], ls="none", marker="o", ms=4.4,
                       mfc=CLASS[kind][1], mec="white", mew=0.6, zorder=4)
    ax_rr.text(0.30, hi + 22, f"normal N\u2013N band  {lo:.0f}\u2013{hi:.0f} ms "
               r"(mean $\pm$ 2 SD)", fontsize=6.6, color=SUB, va="bottom")
    ax_rr.set_ylabel("RR interval (ms)", fontsize=8.0)
    ax_rr.set_ylim(min(rr.min() - 90, lo - 90), max(rr.max() + 110, hi + 110))
    ax_rr.tick_params(labelbottom=False, labelsize=7.0)
    ax_rr.grid(axis="y", color=GRIDC, lw=0.5)
    ax_rr.set_title("Interval sequence: prematurity and the pause that "
                    "follows it", fontsize=8.8, fontweight="bold", loc="left")

    # ---- (c) classification tracks ----------------------------------------
    for row, seq in enumerate([pred, kinds]):
        y = row * 1.0
        for i, (bt, kind) in enumerate(zip(times, seq)):
            bad = kind != kinds[i]
            w = 0.30
            ax_bar.add_patch(plt.Rectangle(
                (bt - w / 2, y + 0.08), w, 0.80, facecolor=CLASS[kind][1],
                edgecolor=INK if bad else "white", lw=1.1 if bad else 0.4,
                zorder=4 if bad else 3))
            if bad:
                ax_bar.plot([bt], [2.06], marker="v", ms=4.0, color=INK,
                            zorder=5)
    ax_bar.set_yticks([0.48, 1.48])
    ax_bar.set_yticklabels(["detector", "reference"], fontsize=7.0)
    ax_bar.set_ylim(-0.05, 2.60)
    ax_bar.set_xlim(0, DUR)
    ax_bar.set_xlabel("time (s)", fontsize=8.0)
    ax_bar.tick_params(labelsize=7.0)
    for s in ("left",):
        ax_bar.spines[s].set_visible(False)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.text(0.05, 2.30, f"beat-level agreement {acc * 100:.1f}%  "
                f"\u2014  {sum(a != b for a, b in zip(kinds, pred))} of "
                f"{len(kinds)} beats mislabelled (outlined, \u25bc)",
                ha="left", va="bottom", fontsize=6.6, color=SUB)

    # ---- (d) RR histogram (auxiliary) --------------------------------------
    bins = np.arange(300, 1400, 55)
    stack = [rr[[k == c for k in rr_k]] for c in ("N", "A", "V")]
    ax_hist.hist(stack, bins=bins, stacked=True,
                 color=[CLASS[c][1] for c in ("N", "A", "V")],
                 edgecolor="white", linewidth=0.5)
    ax_hist.axvspan(lo, hi, color=BAND, alpha=0.45, lw=0, zorder=0)
    ax_hist.set_xlabel("RR interval (ms)", fontsize=7.4)
    ax_hist.set_ylabel("beats (n)", fontsize=7.4)
    ax_hist.set_title(f"RR distribution (n = {rr.size} intervals)",
                      fontsize=8.5, fontweight="bold", loc="left")
    ax_hist.tick_params(labelsize=6.5)
    ax_hist.grid(axis="y", color=GRIDC, lw=0.5)
    ax_hist.text(0.97, 0.92,
                 f"pNN50 = {np.mean(np.abs(np.diff(rr)) > 50) * 100:.0f}%\n"
                 f"RMSSD = {np.sqrt(np.mean(np.diff(rr) ** 2)):.0f} ms",
                 transform=ax_hist.transAxes, ha="right", va="top",
                 fontsize=6.5, color=SUB, linespacing=1.4)

    # ---- (e) class mix (auxiliary) -----------------------------------------
    order = ["N", "V", "A"]
    counts = [sum(1 for k in kinds if k == c) for c in order]
    frac = np.array(counts, dtype=float) / len(kinds) * 100.0
    ypos = np.arange(len(order))[::-1]
    ax_prop.barh(ypos, frac, height=0.52,
                 color=[CLASS[c][1] for c in order], edgecolor="white",
                 linewidth=0.6)
    for y, f, n in zip(ypos, frac, counts):
        ax_prop.text(f + 2.0, y, f"{f:.1f}%   (n = {n})", va="center",
                     fontsize=6.6, color=INK)
    ax_prop.set_yticks(ypos)
    ax_prop.set_yticklabels(order, fontsize=6.8)
    ax_prop.set_xlim(0, 108)
    ax_prop.set_xlabel("share of detected beats (%)", fontsize=7.4)
    ax_prop.set_title("Rhythm composition over the 20 s window",
                      fontsize=8.5, fontweight="bold", loc="left")
    ax_prop.tick_params(labelsize=6.5)
    ax_prop.tick_params(axis="y", length=0)
    ax_prop.spines["left"].set_visible(False)
    ax_prop.grid(axis="x", color=GRIDC, lw=0.5)
    ax_prop.text(0.985, 0.06, "class codes as in the legend above",
                 transform=ax_prop.transAxes, ha="right", va="bottom",
                 fontsize=6.4, color=MUTED)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94], h_pad=2.0, w_pad=2.6)
    panel_labels_inches(fig, [ax_ecg, ax_rr, ax_bar, ax_hist, ax_prop],
                        dx=-0.44, dy=0.06)
    savefig(fig, "547_cvd_arrhythmia_detect.png")


if __name__ == "__main__":
    main()
