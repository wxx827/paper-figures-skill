"""MIDI piano roll with a drawn keyboard axis, chord band and note statistics.

The roll is generated from an eight-chord progression: a bass line, a block
chord part and a melody are voiced from the same chord tables, so every note
rectangle on the roll is a real event in a note list.  Both side panels are
derived from that list rather than being drawn independently - the pitch-class
histogram counts the notes actually plotted and the velocity panel bins the
same velocity values that set each rectangle's opacity, so changing the seed
moves the roll and both statistics together.

The keyboard on the left is the y tick: white and black keys are placed from
``pitch % 12``, which is what puts a black key between C and D but none
between E and F or between B and C.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#cccccc"
HAIR = "#d5d5d5"
BAND = "#f0f0f0"

VOICES = ["Melody", "Chords", "Bass"]
VOICE_COLOR = {"Melody": "#b23a48", "Chords": "#3b5b92", "Bass": "#2e7d5b"}

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_PC = (1, 3, 6, 8, 10)
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)

# name, root pitch class, chord tones - two bars each, sixteen bars total.
PROGRESSION = [
    ("Cmaj7", 0, (0, 4, 7, 11)),
    ("Am7", 9, (9, 0, 4, 7)),
    ("Dm7", 2, (2, 5, 9, 0)),
    ("G7", 7, (7, 11, 2, 5)),
    ("Em7", 4, (4, 7, 11, 2)),
    ("Am7", 9, (9, 0, 4, 7)),
    ("Dm7", 2, (2, 5, 9, 0)),
    ("G7", 7, (7, 11, 2, 5)),
]

BPM = 112.0
BEATS_PER_BAR = 4
BARS_PER_CHORD = 2
PITCH_LO, PITCH_HI = 36, 86          # C2 to D6


def pitch_at_or_above(pc, lo):
    """Lowest MIDI number >= ``lo`` whose pitch class is ``pc``."""
    return lo + (pc - lo) % 12


def build_notes(rng):
    """Bass, block chords and melody voiced from PROGRESSION.

    Returns a list of ``(start_beat, duration_beats, pitch, velocity, voice)``.
    """
    notes = []
    prev = 79
    for ci, (_, root, tones) in enumerate(PROGRESSION):
        t0 = ci * BARS_PER_CHORD * BEATS_PER_BAR

        for bar in range(BARS_PER_CHORD):
            b0 = t0 + bar * BEATS_PER_BAR
            # bass: root - fifth - root, accented on the downbeat
            fifth = (root + 7) % 12
            for beat, pc, dur in ((0.0, root, 1.75), (2.0, fifth, 0.85),
                                  (3.0, root, 0.85)):
                vel = int(rng.integers(66, 92) + (10 if beat == 0.0 else 0))
                notes.append((b0 + beat, dur, pitch_at_or_above(pc, 36),
                              vel, "Bass"))
            # block chords: a sustained voicing plus an off-beat stab
            for beat, dur in ((0.0, 1.8), (2.5, 1.2)):
                base = 55
                for k, pc in enumerate(tones):
                    p = pitch_at_or_above(pc, base)
                    base = p + 1
                    vel = int(rng.integers(50, 74) + (8 if beat == 0.0 else 0)
                              - 3 * k)
                    notes.append((b0 + beat, dur, p, vel, "Chords"))

        # melody: stepwise contour over the two bars, chord tones on strong beats
        t = 0.0
        span = BARS_PER_CHORD * BEATS_PER_BAR
        while t < span - 1e-9:
            dur = float(rng.choice([0.5, 0.5, 0.75, 1.0, 1.0, 1.5, 2.0]))
            dur = min(dur, span - t)
            if rng.random() < 0.16 and t > 0.0:
                t += dur
                continue
            strong = abs(t % 2.0) < 1e-9
            pool = tones if (strong or rng.random() < 0.55) else MAJOR_SCALE
            cand = [p for p in range(70, 87) if p % 12 in pool]
            cost = [abs(p - prev) + 3.0 * rng.random() for p in cand]
            p = cand[int(np.argmin(cost))]
            vel = int(rng.integers(80, 112) + (8 if strong else 0))
            notes.append((t0 + t, dur * 0.92, p, vel, "Melody"))
            prev = p
            t += dur
    return notes


def draw_keyboard(ax, lo, hi):
    """Piano keys as the y axis: white rows full width, black rows narrower."""
    for p in range(lo, hi + 1):
        if p % 12 not in BLACK_PC:
            ax.add_patch(Rectangle((0.0, p - 0.5), 1.0, 1.0, facecolor="white",
                                   edgecolor=HAIR, linewidth=0.4, zorder=2))
    for p in range(lo, hi + 1):
        if p % 12 in BLACK_PC:
            ax.add_patch(Rectangle((0.0, p - 0.42), 0.62, 0.84,
                                   facecolor="#3a3a3a", edgecolor="#3a3a3a",
                                   linewidth=0.3, zorder=3))
    for p in range(lo, hi + 1):
        if p % 12 == 0:
            ax.text(0.96, p, "C%d" % (p // 12 - 1), ha="right", va="center",
                    fontsize=6.0, color=RULE, zorder=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(RULE)


def main():
    set_pub_style()
    rng = np.random.default_rng(846)
    notes = build_notes(rng)

    sec_per_bar = 60.0 / BPM * BEATS_PER_BAR
    n_bars = len(PROGRESSION) * BARS_PER_CHORD
    vel_all = np.array([n[3] for n in notes], dtype=float)
    v_lo, v_hi = vel_all.min(), vel_all.max()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_key = axes_inches(fig, (0.72, 0.92, 0.40, 4.38))
    ax_roll = axes_inches(fig, (1.12, 0.92, 4.90, 4.38))
    ax_chord = axes_inches(fig, (1.12, 5.28, 4.90, 0.30))
    ax_pc = axes_inches(fig, (7.05, 3.60, 3.05, 2.00))
    ax_vel = axes_inches(fig, (7.05, 0.92, 3.05, 2.00))

    # ---- chord band ------------------------------------------------------
    for ci, (name, _, _) in enumerate(PROGRESSION):
        x0 = ci * BARS_PER_CHORD
        ax_chord.add_patch(Rectangle((x0, 0.0), BARS_PER_CHORD, 1.0,
                                     facecolor=BAND if ci % 2 == 0 else "#f7f7f7",
                                     edgecolor="white", linewidth=0.8))
        ax_chord.text(x0 + BARS_PER_CHORD / 2.0, 0.46, name, ha="center",
                      va="center", fontsize=6.8, color=INK)
    ax_chord.set_xlim(0, n_bars)
    ax_chord.set_ylim(0, 1)
    ax_chord.set_xticks([])
    ax_chord.set_yticks([])
    for s in ax_chord.spines.values():
        s.set_visible(False)
    ax_chord.set_title("Piano roll with chord band, %d notes over %d bars "
                       "(%.1f s)" % (len(notes), n_bars,
                                     n_bars * sec_per_bar),
                       fontsize=9.0, fontweight="bold", pad=4.0)

    # ---- piano roll ------------------------------------------------------
    for p in range(PITCH_LO, PITCH_HI + 1):
        if p % 12 in BLACK_PC:
            ax_roll.add_patch(Rectangle((0, p - 0.5), n_bars, 1.0,
                                        facecolor="#f5f5f5", edgecolor="none",
                                        zorder=0))
    for b in range(n_bars + 1):
        heavy = b % 4 == 0
        ax_roll.axvline(b, color=GRID if not heavy else "#b3b3b3",
                        lw=0.9 if heavy else 0.5, zorder=1)

    for start, dur, pitch, vel, voice in notes:
        a = 0.40 + 0.60 * (vel - v_lo) / (v_hi - v_lo)
        ax_roll.add_patch(Rectangle(
            (start / BEATS_PER_BAR, pitch - 0.40), dur / BEATS_PER_BAR, 0.80,
            facecolor=VOICE_COLOR[voice], edgecolor="white", linewidth=0.25,
            alpha=a, zorder=4))

    ax_roll.set_xlim(0, n_bars)
    ax_roll.set_ylim(PITCH_LO - 0.5, PITCH_HI + 0.5)
    ax_roll.set_yticks([])
    ax_roll.set_xticks(np.arange(0, n_bars + 1, 2))
    ax_roll.set_xlabel("time (bars; 4/4 at %.0f BPM, 1 bar = %.2f s)"
                       % (BPM, sec_per_bar), fontsize=8.0)
    ax_roll.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        ax_roll.spines[s].set_visible(False)

    draw_keyboard(ax_key, PITCH_LO, PITCH_HI)
    ax_key.set_ylabel("pitch (MIDI note number, semitones)", fontsize=8.0,
                      labelpad=2.0)

    # ---- pitch-class histogram, stacked by voice -------------------------
    counts = {v: np.zeros(12) for v in VOICES}
    for _, _, pitch, _, voice in notes:
        counts[voice][pitch % 12] += 1.0
    bottom = np.zeros(12)
    xs = np.arange(12)
    for v in VOICES:
        ax_pc.bar(xs, counts[v], bottom=bottom, width=0.74,
                  color=VOICE_COLOR[v], edgecolor="white", linewidth=0.5,
                  alpha=0.88)
        bottom += counts[v]
    top_pc = int(np.argmax(bottom))
    ax_pc.annotate("%s, n=%d" % (PC_NAMES[top_pc], int(bottom[top_pc])),
                   xy=(top_pc, bottom[top_pc]),
                   xytext=(top_pc + 0.6, bottom[top_pc] * 1.02),
                   fontsize=6.5, color=SUB,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))
    ax_pc.set_xticks(xs)
    ax_pc.set_xticklabels(PC_NAMES, fontsize=6.5)
    ax_pc.set_xlim(-0.7, 11.7)
    ax_pc.set_ylim(0, bottom.max() * 1.22)
    ax_pc.set_ylabel("note count (n)", fontsize=7.5)
    ax_pc.set_xlabel("pitch class (semitone, 12-TET)", fontsize=7.5)
    ax_pc.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_pc.spines[s].set_visible(False)
    ax_pc.set_title("Pitch-class profile", fontsize=8.5, fontweight="bold",
                    pad=3.0)

    # ---- velocity distribution -------------------------------------------
    bins = np.arange(45, 126, 5)
    for v in VOICES:
        vv = np.array([n[3] for n in notes if n[4] == v], dtype=float)
        ax_vel.hist(vv, bins=bins, histtype="step", lw=1.2,
                    color=VOICE_COLOR[v],
                    label="%s (n=%d, med %.0f)" % (v, vv.size, np.median(vv)))
    ax_vel.set_xlim(bins[0], bins[-1])
    ax_vel.set_xlabel("MIDI velocity (a.u., 0-127)", fontsize=7.5)
    ax_vel.set_ylabel("note count (n)", fontsize=7.5)
    ax_vel.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_vel.spines[s].set_visible(False)
    ax_vel.legend(frameon=False, fontsize=6.6, loc="upper right",
                  handlelength=1.5, labelspacing=0.35)
    ax_vel.set_title("Velocity distribution by voice", fontsize=8.5,
                     fontweight="bold", pad=3.0)

    panel_labels_inches(fig, [ax_chord, ax_pc, ax_vel], dx=-0.40, dy=0.16)


    savefig(fig, "846_mir_piano_roll.png")


if __name__ == "__main__":
    main()
