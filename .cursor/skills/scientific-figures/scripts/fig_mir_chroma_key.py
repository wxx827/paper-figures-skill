"""Chromagram with chord segmentation, key correlation profile and fifths map.

A chord script is rendered into a 12-bin chroma matrix by summing weighted
chord tones plus their fifth and third partials, a moving melody voice and
frame noise, then peak-normalising each frame.  Everything downstream is
measured off that matrix: the key panel correlates the time-averaged chroma
with all 24 rotated Krumhansl-Schmuckler profiles, and the circle-of-fifths
map colours each node by the mean energy of that pitch class, so the two side
panels cannot disagree with the image they describe.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#cccccc"
ACCENT = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

# Krumhansl-Kessler key profiles (probe-tone ratings, a.u.).
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# name, chord tones (pitch classes), duration in seconds
SCRIPT = [
    ("C", (0, 4, 7), 4.0),
    ("Am", (9, 0, 4), 3.5),
    ("F", (5, 9, 0), 4.0),
    ("G", (7, 11, 2), 3.5),
    ("Em", (4, 7, 11), 3.0),
    ("Am", (9, 0, 4), 3.0),
    ("Dm", (2, 5, 9), 3.0),
    ("G7", (7, 11, 2, 5), 3.0),
    ("C", (0, 4, 7), 3.0),
]
HOP = 0.05
TONE_W = (1.00, 0.78, 0.86, 0.58)      # root, third, fifth, seventh


def synth_chroma(rng):
    """Frame-wise 12-bin chroma from the chord script; returns (chroma, edges)."""
    total = sum(d for _, _, d in SCRIPT)
    n = int(round(total / HOP))
    chroma = np.zeros((12, n))
    t = np.arange(n) * HOP
    bounds = np.cumsum([0.0] + [d for _, _, d in SCRIPT])

    for si, (_, tones, _) in enumerate(SCRIPT):
        m = (t >= bounds[si]) & (t < bounds[si + 1])
        for k, pc in enumerate(tones):
            w = TONE_W[min(k, len(TONE_W) - 1)]
            chroma[pc, m] += w
            chroma[(pc + 7) % 12, m] += 0.22 * w      # fifth partial
            chroma[(pc + 4) % 12, m] += 0.13 * w      # third partial
        # onset transient: short broadband smear right after the chord change
        onset = m & (t < bounds[si] + 0.35)
        chroma[:, onset] += 0.28 * np.exp(
            -(t[onset] - bounds[si]) / 0.12)[None, :]

    # melody voice: one diatonic pitch class every half second
    scale = [0, 2, 4, 5, 7, 9, 11]
    for i in range(int(total / 0.5)):
        m = (t >= i * 0.5) & (t < (i + 1) * 0.5)
        chroma[int(rng.choice(scale)), m] += 0.55

    chroma += 0.07 * rng.random(chroma.shape)
    # short temporal smoothing, as a real chroma feature would have
    k = np.ones(5) / 5.0
    chroma = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"),
                                 1, chroma)
    chroma /= chroma.max(axis=0, keepdims=True)
    edges = np.arange(n + 1) * HOP
    return chroma, edges, bounds


def key_correlations(mean_chroma):
    """Pearson r of the mean chroma against all 24 rotated KS profiles."""
    maj = np.array([np.corrcoef(mean_chroma, np.roll(KS_MAJOR, k))[0, 1]
                    for k in range(12)])
    minr = np.array([np.corrcoef(mean_chroma, np.roll(KS_MINOR, k))[0, 1]
                     for k in range(12)])
    return maj, minr


def main():
    set_pub_style()
    rng = np.random.default_rng(847)
    chroma, edges, bounds = synth_chroma(rng)
    mean_chroma = chroma.mean(axis=1)
    maj, minr = key_correlations(mean_chroma)

    if maj.max() >= minr.max():
        key_pc, key_mode, key_r = int(np.argmax(maj)), "major", maj.max()
    else:
        key_pc, key_mode, key_r = int(np.argmax(minr)), "minor", minr.max()
    key_name = "%s %s" % (PC_NAMES[key_pc], key_mode)

    cmap = get_cmap("cb_sequential_blue")
    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.78, 0.92, 5.24, 4.42))
    ax_band = axes_inches(fig, (0.78, 5.34, 5.24, 0.26))
    ax_key = axes_inches(fig, (7.15, 3.45, 2.30, 2.15), projection="polar")
    ax_cof = axes_inches(fig, (7.15, 0.75, 2.30, 2.15))

    # ---- chromagram -------------------------------------------------------
    mesh = ax.pcolormesh(edges, np.arange(13), chroma, cmap=cmap,
                         vmin=0.0, vmax=1.0, shading="flat", rasterized=True)
    for b in bounds[1:-1]:
        ax.axvline(b, color="white", lw=1.0, alpha=0.85, zorder=3)
        ax.axvline(b, color=RULE, lw=0.5, ls=(0, (3, 2)), zorder=4)
    for si, (name, _, _) in enumerate(SCRIPT):
        ax_band.axvspan(bounds[si], bounds[si + 1],
                        facecolor="#f0f0f0" if si % 2 == 0 else "#f7f7f7",
                        edgecolor="white", lw=0.8)
        ax_band.text(0.5 * (bounds[si] + bounds[si + 1]), 0.46, name,
                     ha="center", va="center", fontsize=6.8, color=INK)
    ax_band.set_xlim(bounds[0], bounds[-1])
    ax_band.set_ylim(0, 1)
    ax_band.set_xticks([])
    ax_band.set_yticks([])
    for s in ax_band.spines.values():
        s.set_visible(False)
    ax_band.set_title("Chromagram with detected chord segments", fontsize=9.0,
                      fontweight="bold", pad=4.0)

    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels(PC_NAMES, fontsize=7.0)
    ax.set_ylim(0, 12)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("time (s)", fontsize=8.0)
    ax.set_ylabel("pitch class (semitone, 12-TET)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    safe_colorbar(fig, mesh, ax, label="normalised chroma energy (a.u.)",
                  width=0.11, pad=0.09, labelsize=7.0)

    # ---- key correlation profile -----------------------------------------
    ang = np.arange(12) * 2.0 * np.pi / 12.0
    ax_key.set_theta_zero_location("N")
    ax_key.set_theta_direction(-1)
    for vals, col, lab in ((maj, BLUE, "major"), (minr, GREEN, "minor")):
        ax_key.plot(np.append(ang, ang[0]), np.append(vals, vals[0]),
                    color=col, lw=1.1, marker="o", ms=2.6, label=lab)
        ax_key.fill(np.append(ang, ang[0]), np.append(vals, vals[0]),
                    color=col, alpha=0.10)
    ax_key.plot([ang[key_pc]], [key_r], marker="o", ms=6.0, mfc="none",
                mec=ACCENT, mew=1.4, zorder=6)
    ax_key.set_rlim(-0.85, 1.05)
    ax_key.set_rticks([0.0, 0.5, 1.0])
    ax_key.set_yticklabels(["0", "0.5", "1"], fontsize=6.0, color=MUTED)
    ax_key.set_rlabel_position(105)
    ax_key.set_xticks(ang)
    ax_key.set_xticklabels(PC_NAMES, fontsize=6.4)
    ax_key.tick_params(pad=0.6)
    ax_key.grid(color=GRID, lw=0.5)
    ax_key.set_title("Key correlation r (-), 24 profiles", fontsize=8.5,
                     fontweight="bold", pad=7.0)
    ax_key.legend(frameon=False, fontsize=6.4, loc="lower center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.4,
                  columnspacing=1.0)

    # ---- circle of fifths -------------------------------------------------
    ax_cof.set_aspect("equal")
    ax_cof.set_xlim(-1.62, 1.62)
    ax_cof.set_ylim(-1.62, 1.62)
    ax_cof.axis("off")
    ring = np.linspace(0, 2 * np.pi, 361)
    ax_cof.plot(np.cos(ring), np.sin(ring), color=GRID, lw=0.7, zorder=1)
    e_lo, e_hi = mean_chroma.min(), mean_chroma.max()
    for i, pc in enumerate(FIFTHS):
        a = np.pi / 2.0 - i * 2.0 * np.pi / 12.0
        x, y = np.cos(a), np.sin(a)
        frac = (mean_chroma[pc] - e_lo) / (e_hi - e_lo)
        tonic = pc == key_pc
        ax_cof.add_patch(Circle((x, y), 0.185, facecolor=cmap(0.12 + 0.8 * frac),
                                edgecolor=ACCENT if tonic else "#8a8a8a",
                                lw=1.6 if tonic else 0.6, zorder=3))
        ax_cof.text(x, y, PC_NAMES[pc], ha="center", va="center",
                    fontsize=6.2, color="white" if frac > 0.55 else INK,
                    zorder=4)
        ax_cof.text(1.36 * x, 1.36 * y, "%.2f" % mean_chroma[pc],
                    ha="center", va="center", fontsize=6.0, color=MUTED,
                    zorder=4)
    ax_cof.set_title("Mean chroma on the circle of fifths", fontsize=8.5,
                     fontweight="bold", pad=4.0)
    ax_cof.text(0.0, -0.03, "estimated key\n%s\nr = %.2f"
                % (key_name, key_r), ha="center", va="center", fontsize=6.6,
                color=INK, linespacing=1.5, zorder=5)

    panel_labels_inches(fig, [ax_band, ax_key, ax_cof], dx=-0.42, dy=0.16)


    savefig(fig, "847_mir_chroma_key.png")


if __name__ == "__main__":
    main()
