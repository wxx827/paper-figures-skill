"""462 - Structure analysis of a synthetic pop song from chroma features.

The self-similarity matrix, the annotation ribbon, the novelty curve and the
chromagram are all drawn on the same time grid: the ribbon and the novelty
curve share the horizontal axis of the matrix, the chromagram shares its
vertical axis, so a block in the matrix lines up with the section that produced
it.  Boundaries are not annotated by hand - they are picked from a checkerboard
novelty curve computed on the matrix, and the caption reports how many of the
scored boundaries that recovers.  The chromagram and the per-section templates
share one colour scale, and the matrix and the section-similarity summary share
another.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from pubstyle import set_pub_style, savefig, style_axes, lighten, get_cmap
from figkit import axes_inches, shared_norm, safe_colorbar, text_card

INK, RULE, SUB = "#1a1a1a", "#333333", "#555555"
MUTED, HAIR = "#777777", "#cccccc"
DET = "#b23a48"

PITCH = ["C", "C\u266f", "D", "E\u266d", "E", "F", "F\u266f", "G",
         "A\u266d", "A", "B\u266d", "B"]
TYPES = ["intro", "verse", "chorus", "bridge"]
TYPE_COL = {"intro": "#8a8a8a", "verse": "#3b5b92", "chorus": "#b23a48",
            "bridge": "#c9a227"}
TYPE_LAB = {"intro": "Intro / Outro", "verse": "Verse", "chorus": "Chorus",
            "bridge": "Bridge"}

# root pitch class and quality, one chord per bar; the bridge modulates, which
# is what makes its block stand out from everything else
PROG = {
    "intro":  [(0, "maj"), (7, "maj")],
    "verse":  [(2, "min"), (10, "maj"), (5, "maj"), (0, "maj")],
    "chorus": [(5, "maj"), (0, "maj"), (7, "maj"), (9, "min")],
    "bridge": [(4, "maj"), (1, "min"), (9, "maj"), (11, "maj")],
}
LEVEL = {"intro": 0.52, "verse": 0.80, "chorus": 1.00, "bridge": 0.68}
# melody pitch class emphasised in each section type
MELODY = {"intro": 7, "verse": 2, "chorus": 9, "bridge": 4}

# (display label, type, start s, end s)
SECTIONS = [("Intro", "intro", 0, 24), ("Verse 1", "verse", 24, 56),
            ("Chorus 1", "chorus", 56, 88), ("Verse 2", "verse", 88, 120),
            ("Chorus 2", "chorus", 120, 152), ("Bridge", "bridge", 152, 176),
            ("Chorus 3", "chorus", 176, 208), ("Outro", "intro", 208, 232)]
T_END = SECTIONS[-1][3]
HOP = 0.5                    # s per chroma frame
BAR = 2.0                    # s per chord
SMOOTH = 2.0                 # s, chroma smoothing before the matrix
KERNEL = 12.0                # s, half-width of the checkerboard kernel
MIN_SEG = 14.0               # s, minimum spacing between detected boundaries
TOL = 3.0                    # s, boundary hit tolerance

W, H = 10.4, 6.4             # canvas, inches


def chord_vector(root, qual):
    v = np.zeros(12)
    third = 4 if qual == "maj" else 3
    for pc, w in ((root, 1.00), ((root + third) % 12, 0.82),
                  ((root + 7) % 12, 0.90)):
        v[pc] += w
        v[(pc + 1) % 12] += 0.10          # harmonic leakage to neighbours
        v[(pc - 1) % 12] += 0.10
    v[(root + 10) % 12] += 0.18           # dominant seventh colour
    return v


def synth_chroma(rng):
    """(12, n_frame) chroma with repeated sections that are alike, not equal."""
    t = np.arange(0.0, T_END, HOP)
    X = np.zeros((12, t.size))
    for _, typ, t0, t1 in SECTIONS:
        prog = PROG[typ]
        tilt = rng.normal(1.0, 0.06, 12)          # per-instance timbre drift
        m = (t >= t0) & (t < t1)
        idx = np.nonzero(m)[0]
        for k in idx:
            bar = int((t[k] - t0) // BAR) % len(prog)
            root, qual = prog[bar]
            v = chord_vector(root, qual) * tilt
            v[MELODY[typ]] += 0.45 * (0.6 + 0.4 * np.sin(2 * np.pi * t[k] / 8.0))
            X[:, k] = LEVEL[typ] * v
    X += rng.normal(0.0, 0.045, X.shape) ** 2 + 0.02
    X = gaussian_filter1d(np.clip(X, 0.0, None), 1.1, axis=1, mode="nearest")
    return t, X


def enhance(S, L=6):
    """Average the matrix along the diagonal so repeated passages form paths."""
    out = np.zeros_like(S)
    n = S.shape[0]
    for l in range(L):
        out[:n - l, :n - l] += S[l:, l:]
        out[l:, l:] += S[:n - l, :n - l]
    return out / (2.0 * L)


def novelty(S, M=int(KERNEL / HOP)):
    """Checkerboard-kernel novelty along the main diagonal."""
    g = np.exp(-0.5 * (np.linspace(-2.0, 2.0, 2 * M) ** 2))
    K = np.outer(g, g) * np.outer(np.r_[-np.ones(M), np.ones(M)],
                                  np.r_[-np.ones(M), np.ones(M)])
    n = S.shape[0]
    nov = np.zeros(n)
    P = np.pad(S, M, mode="edge")
    for i in range(n):
        nov[i] = float((P[i:i + 2 * M, i:i + 2 * M] * K).sum())
    nov = gaussian_filter1d(nov, 1.4)
    nov -= nov.min()
    return nov / (nov.max() + 1e-12)


def fits(label, width_in, fs=6.0):
    return len(label) * fs * 0.0085 < width_in - 0.04


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(462)

    t, X = synth_chroma(rng)
    # the matrix is built on chroma smoothed over one chord cycle, otherwise
    # the bar-rate chord changes fill it with fine stripes and bury the blocks
    Xs = gaussian_filter1d(X, SMOOTH / HOP, axis=1, mode="nearest")
    Xn = Xs / (np.linalg.norm(Xs, axis=0, keepdims=True) + 1e-12)
    S = enhance(np.clip(Xn.T @ Xn, 0.0, 1.0), L=4)
    nov = novelty(S)

    true_b = np.array([s[2] for s in SECTIONS[1:]], dtype=float)
    pk, _ = find_peaks(nov, distance=int(MIN_SEG / HOP), prominence=0.10)
    det = t[pk]
    hit = sum(bool(np.any(np.abs(det - b) <= TOL)) for b in true_b)
    err = [float(np.min(np.abs(det - b))) for b in true_b
           if np.any(np.abs(det - b) <= TOL)]

    # per-type mean chroma and per-type mean similarity, from the same arrays
    frame_type = np.empty(t.size, dtype=object)
    for _, typ, t0, t1 in SECTIONS:
        frame_type[(t >= t0) & (t < t1)] = typ
    tmpl = np.stack([X[:, frame_type == ty].mean(axis=1) for ty in TYPES])
    far = np.abs(np.arange(t.size)[:, None] - np.arange(t.size)[None, :]) > 6
    tsim = np.zeros((len(TYPES), len(TYPES)))
    for a, ta in enumerate(TYPES):
        for b, tb in enumerate(TYPES):
            m = np.outer(frame_type == ta, frame_type == tb) & far
            tsim[a, b] = float(S[m].mean()) if m.any() else np.nan

    n_chroma = shared_norm([X, tmpl], robust=(4, 99))
    n_sim = shared_norm([S], robust=(2, 98))
    cmap_c, cmap_s = "magma", get_cmap("cb_sequential_blue")

    # ---- inch-grid layout: the matrix must stay square ---------------------
    SQ = 3.72
    fig = plt.figure(figsize=(W, H))
    ax_s = axes_inches(fig, (0.62, 1.56, SQ, SQ))
    ax_rb = axes_inches(fig, (0.62, 1.20, SQ, 0.30))
    ax_nv = axes_inches(fig, (0.62, 0.48, SQ, 0.66))
    ax_ch = axes_inches(fig, (4.46, 1.56, 1.24, SQ))
    ax_tm = axes_inches(fig, (6.32, 4.42, 2.84, 0.84))
    ax_ts = axes_inches(fig, (6.38, 1.98, 1.45, 1.45))
    ax_rd = axes_inches(fig, (8.15, 1.66, 1.90, 1.78))

    # ---- (a) self-similarity matrix ----------------------------------------
    im_s = ax_s.imshow(S, cmap=cmap_s, norm=n_sim, origin="lower",
                       extent=[0, T_END, 0, T_END], interpolation="nearest",
                       rasterized=True)
    for _, _, t0, _ in SECTIONS[1:]:
        ax_s.axvline(t0, color="white", lw=0.55, alpha=0.45)
        ax_s.axhline(t0, color="white", lw=0.55, alpha=0.45)
    ax_s.set_xlim(0, T_END)
    ax_s.set_ylim(0, T_END)
    ax_s.set_xticks(np.arange(0, T_END + 1, 40))
    ax_s.set_yticks(np.arange(0, T_END + 1, 40))
    ax_s.tick_params(labelbottom=False, labelsize=6.8, length=2.4)
    ax_s.set_ylabel("Time (s)", fontsize=8.0)
    for s in ax_s.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax_s.set_title("Chroma self-similarity matrix", fontsize=8.8,
                   fontweight="bold", pad=4.0, loc="left")
    ax_s.annotate("chorus 1 repeats\nas chorus 3", xy=(72, 192),
                  xytext=(16, 216), fontsize=6.6, color=INK, ha="left",
                  va="center", linespacing=1.2,
                  bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none",
                            alpha=0.88),
                  arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                                  connectionstyle="arc3,rad=-0.2"))

    # ---- (b) annotation ribbon ---------------------------------------------
    for lab, typ, t0, t1 in SECTIONS:
        ax_rb.add_patch(Rectangle((t0, 0.0), t1 - t0, 1.0,
                                  facecolor=lighten(TYPE_COL[typ], 0.35),
                                  edgecolor="white", lw=0.8, zorder=2))
        w_in = SQ * (t1 - t0) / T_END
        short = {"Chorus 1": "Ch 1", "Chorus 2": "Ch 2", "Chorus 3": "Ch 3",
                 "Verse 1": "V 1", "Verse 2": "V 2", "Bridge": "Brg",
                 "Intro": "Int", "Outro": "Out"}
        txt = lab if fits(lab, w_in) else short[lab]
        ax_rb.text(0.5 * (t0 + t1), 0.5, txt, ha="center", va="center",
                   fontsize=6.2, color="white", fontweight="bold", zorder=3)
    ax_rb.set_xlim(0, T_END)
    ax_rb.set_ylim(0, 1)
    ax_rb.set_xticks(np.arange(0, T_END + 1, 40))
    ax_rb.set_yticks([])
    ax_rb.tick_params(labelbottom=False, length=0)
    for s in ax_rb.spines.values():
        s.set_visible(False)
    ax_rb.text(-0.008, 0.5, "annotation", transform=ax_rb.transAxes,
               ha="right", va="center", fontsize=6.4, color=SUB)

    # ---- (c) novelty curve and detected boundaries -------------------------
    ax_nv.fill_between(t, 0.0, nov, color=lighten("#3b5b92", 0.72), lw=0,
                       zorder=1)
    ax_nv.plot(t, nov, color="#3b5b92", lw=0.9, zorder=3)
    for b in true_b:
        ax_nv.axvline(b, color=MUTED, lw=0.6, ls=(0, (2.5, 1.8)), zorder=2)
    ax_nv.plot(det, nov[pk], ls="none", marker="v", ms=4.0, mfc=DET,
               mec="white", mew=0.6, zorder=5)
    for b in det:
        ax_nv.axvline(b, color=DET, lw=0.7, alpha=0.55, zorder=2)
        ax_rb.axvline(b, color=DET, lw=0.7, alpha=0.85, zorder=4)
        ax_s.axvline(b, color=DET, lw=0.5, alpha=0.35, zorder=4)
    ax_nv.set_xlim(0, T_END)
    ax_nv.set_ylim(0, 1.18)
    ax_nv.set_xticks(np.arange(0, T_END + 1, 40))
    ax_nv.set_yticks([0, 1])
    ax_nv.tick_params(labelsize=6.8, length=2.4)
    ax_nv.set_xlabel("Time (s)", fontsize=8.0)
    ax_nv.set_ylabel("novelty\n(norm.)", fontsize=6.8, linespacing=1.15)
    style_axes(ax_nv)
    ax_nv.text(0.004, 0.97, f"checkerboard novelty, {2 * KERNEL:.0f} s kernel",
               transform=ax_nv.transAxes, ha="left", va="top", fontsize=6.2,
               color=SUB)

    # ---- (d) chromagram, sharing the vertical time axis --------------------
    im_c = ax_ch.imshow(X.T, cmap=cmap_c, norm=n_chroma, origin="lower",
                        extent=[0, 12, 0, T_END], aspect="auto",
                        interpolation="nearest", rasterized=True)
    ax_ch.set_ylim(0, T_END)
    ax_ch.set_xticks(np.arange(12) + 0.5)
    ax_ch.set_xticklabels(PITCH, fontsize=6.0, rotation=90)
    ax_ch.xaxis.set_ticks_position("top")
    ax_ch.set_yticks(np.arange(0, T_END + 1, 40))
    ax_ch.tick_params(labelleft=False, length=2.0, pad=1.5)
    for s in ax_ch.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    for _, _, t0, _ in SECTIONS[1:]:
        ax_ch.axhline(t0, color="white", lw=0.5, alpha=0.45)
    ax_ch.set_title("Chromagram", fontsize=8.0, fontweight="bold", pad=16.0)

    # ---- (e) per-section-type chroma templates -----------------------------
    ax_tm.imshow(tmpl, cmap=cmap_c, norm=n_chroma, aspect="auto",
                 extent=[0, 12, len(TYPES), 0], interpolation="nearest")
    ax_tm.set_xticks(np.arange(12) + 0.5)
    ax_tm.set_xticklabels(PITCH, fontsize=6)
    ax_tm.xaxis.set_ticks_position("top")
    ax_tm.set_yticks(np.arange(len(TYPES)) + 0.5)
    ax_tm.set_yticklabels([TYPE_LAB[ty] for ty in TYPES], fontsize=6.6)
    for lab, ty in zip(ax_tm.get_yticklabels(), TYPES):
        lab.set_color(TYPE_COL[ty])
        lab.set_fontweight("bold")
    ax_tm.tick_params(length=0, pad=2.0)
    for s in ax_tm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    for i in range(1, len(TYPES)):
        ax_tm.axhline(i, color="white", lw=0.6)
    ax_tm.set_title("Mean chroma per section type", fontsize=8.0,
                    fontweight="bold", pad=13.0, loc="left")
    cb_c = safe_colorbar(fig, im_c, ax_tm,
                         label="chroma energy (norm.), shared scale",
                         side="bottom", width=0.10, pad=0.09, shrink=0.72,
                         labelsize=6.3)
    cb_c.ax.tick_params(labelsize=6.2)

    # ---- (f) mean similarity between section types -------------------------
    ax_ts.imshow(tsim, cmap=cmap_s, norm=n_sim, interpolation="nearest")
    ax_ts.set_xticks(range(len(TYPES)))
    ax_ts.set_yticks(range(len(TYPES)))
    lab_s = ["Int", "Vrs", "Cho", "Brg"]
    ax_ts.set_xticklabels(lab_s, fontsize=6.4)
    ax_ts.set_yticklabels(lab_s, fontsize=6.4)
    ax_ts.xaxis.set_ticks_position("top")
    for labs, keys in ((ax_ts.get_xticklabels(), TYPES),
                       (ax_ts.get_yticklabels(), TYPES)):
        for lab, ty in zip(labs, keys):
            lab.set_color(TYPE_COL[ty])
            lab.set_fontweight("bold")
    ax_ts.tick_params(length=0, pad=2.0)
    for i in range(len(TYPES)):
        for j in range(len(TYPES)):
            v = tsim[i, j]
            ax_ts.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=6.4,
                       color="white" if n_sim(v) > 0.62 else INK)
    for s in ax_ts.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax_ts.set_title("Mean similarity between types", fontsize=8.0,
                    fontweight="bold", pad=13.0, loc="left")
    cb_s = safe_colorbar(fig, im_s, ax_ts,
                         label="cosine similarity (-), shared scale",
                         side="bottom", width=0.10, pad=0.09, shrink=1.0,
                         labelsize=6.3)
    cb_s.ax.tick_params(labelsize=6.2)

    # ---- (g) read-out --------------------------------------------------------
    ax_rd.set_xlim(0, 1)
    ax_rd.set_ylim(0, 1)
    ax_rd.axis("off")
    i_c, i_v = TYPES.index("chorus"), TYPES.index("verse")
    text_card(
        ax_rd, 0.0, 0.0, 1.0, 1.0,
        f"\u2022 {len(SECTIONS)} sections over {T_END} s, "
        f"{len(true_b)} scored boundaries\n"
        f"\u2022 novelty peaks recover {hit} of {len(true_b)} within "
        f"\u00b1{TOL:.0f} s\n"
        f"\u2022 median hit error {np.median(err) if err else np.nan:.1f} s "
        f"({len(det)} peaks picked)\n"
        f"\u2022 chorus-to-chorus similarity {tsim[i_c, i_c]:.2f} against "
        f"{tsim[i_c, i_v]:.2f} chorus-to-verse\n"
        f"\u2022 off-diagonal blocks are the repeats, not artefacts",
        fs=6.4, va="top", ha="left", pad=0.055, rounding=0.05,
        title="Read-out", title_fs=7.4, color=SUB, title_color=INK)
    ax_rd.texts[-1].set_y(0.86)

    savefig(fig, "462_spx_music_structure.png")


if __name__ == "__main__":
    main()
