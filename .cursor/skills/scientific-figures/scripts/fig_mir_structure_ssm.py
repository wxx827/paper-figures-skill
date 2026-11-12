"""Self-similarity matrix of a song form, with novelty-based boundary picking.

The feature sequence is built first and the matrix second: every section type
owns one time-varying 16-dimensional template, and each occurrence of that
section replays its template with fresh noise.  The off-diagonal stripes are
therefore produced by genuinely repeated material rather than drawn by hand,
and the novelty curve is a Foote checkerboard kernel correlated along the
diagonal of that same matrix.  Detected boundaries in panel (c) are the peaks
of panel (b), matched against the true section starts at run time.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#cccccc"
ACCENT = "#b23a48"

SECTION_COLOR = {
    "Intro": "#c9a227",
    "Verse": "#3b5b92",
    "Chorus": "#b23a48",
    "Bridge": "#8e6bbf",
    "Outro": "#2e7d5b",
}
FORM = ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Bridge",
        "Chorus", "Outro"]
LENGTH_S = {"Intro": 10.0, "Verse": 18.0, "Chorus": 16.0,
            "Bridge": 12.0, "Outro": 10.0}
HOP = 0.5           # s per feature frame
N_DIM = 16
KERNEL_HALF = 16    # frames, i.e. 8 s on either side of the candidate


def section_templates(rng):
    """One time-varying feature trajectory per section type."""
    templates = {}
    for name, dur in LENGTH_S.items():
        n = int(round(dur / HOP))
        base = rng.random(N_DIM) * 0.9 + 0.15
        base[rng.integers(0, N_DIM, size=4)] += 0.9
        phase = rng.random(N_DIM) * 2.0 * np.pi
        rate = 0.6 + 1.8 * rng.random(N_DIM)
        u = np.linspace(0.0, 1.0, n)
        traj = (base[None, :]
                + 0.42 * np.sin(2.0 * np.pi * rate[None, :] * u[:, None]
                                + phase[None, :]))
        templates[name] = np.clip(traj, 0.02, None)
    return templates


def build_features(rng):
    """Concatenate the form, returning features and (name, t0, t1) sections."""
    templates = section_templates(rng)
    rows, sections, t = [], [], 0.0
    for name in FORM:
        tpl = templates[name]
        occ = tpl * (1.0 + 0.07 * rng.standard_normal(tpl.shape))
        occ += 0.10 * rng.standard_normal(tpl.shape)
        rows.append(np.clip(occ, 0.01, None))
        sections.append((name, t, t + tpl.shape[0] * HOP))
        t += tpl.shape[0] * HOP
    feats = np.vstack(rows)
    # slow production drift: nothing repeats *exactly* later in a track
    drift = 1.0 + 0.09 * np.linspace(-1.0, 1.0, feats.shape[0])[:, None]
    return feats * drift, sections


def self_similarity(feats):
    """Cosine similarity after removing the track-average feature vector.

    Raw chroma-like features all point into the positive orthant, so plain
    cosine similarity floors around 0.7 and the repeats sit inside a narrow,
    low-contrast band.  Subtracting the mean frame first re-centres the cloud
    and lets the repeated material separate visibly from the background.
    """
    f = feats - feats.mean(axis=0, keepdims=True)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    return f @ f.T


def novelty(ssm, half=KERNEL_HALF):
    """Foote checkerboard novelty along the diagonal, scaled to 0-1 a.u."""
    u = np.linspace(-2.0, 2.0, 2 * half)
    g = np.exp(-0.5 * u ** 2)
    kern = np.ones((2 * half, 2 * half))
    kern[:half, half:] = -1.0
    kern[half:, :half] = -1.0
    kern *= np.outer(g, g)
    n = ssm.shape[0]
    out = np.zeros(n)
    for i in range(half, n - half):
        out[i] = float((ssm[i - half:i + half, i - half:i + half]
                        * kern).sum())
    out[:half] = out[half]
    out[n - half:] = out[n - half - 1]
    return (out - out.min()) / (out.max() - out.min())


def main():
    set_pub_style()
    rng = np.random.default_rng(848)
    feats, sections = build_features(rng)
    ssm = self_similarity(feats)
    nov = novelty(ssm)
    n = ssm.shape[0]
    total = n * HOP
    t_frames = np.arange(n) * HOP

    peaks, _ = find_peaks(nov, distance=14, prominence=0.11)
    det = np.r_[0.0, t_frames[peaks], total]
    truth = np.array([s[1] for s in sections[1:]])
    tol = 2.0
    hits = int(sum(np.min(np.abs(t_frames[peaks] - b)) <= tol for b in truth)) \
        if peaks.size else 0

    cmap = get_cmap("cb_sequential_blue")
    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.80, 0.92, 4.68, 4.68))
    ax_nov = axes_inches(fig, (6.75, 3.30, 2.80, 2.30))
    ax_rib = axes_inches(fig, (6.75, 0.92, 2.80, 1.90))

    # ---- self-similarity matrix ------------------------------------------
    vmin = float(np.percentile(ssm, 2))
    im = ax.imshow(ssm, cmap=cmap, vmin=vmin, vmax=1.0, origin="upper",
                   extent=[0.0, total, total, 0.0], interpolation="nearest")
    ax.set_aspect("equal")
    for name, t0, t1 in sections:
        ax.add_patch(Rectangle((t0, t0), t1 - t0, t1 - t0, facecolor="none",
                               edgecolor=SECTION_COLOR[name], lw=1.2,
                               zorder=5))
        halo_text(ax, 0.5 * (t0 + t1), 0.5 * (t0 + t1), name, ha="center",
                  va="center", fontsize=6.4, color="white", halo=INK,
                  halo_lw=1.8, zorder=7)
    for b in t_frames[peaks]:
        ax.plot([b, b], [total, total - 3.0], color=ACCENT, lw=1.2, zorder=6)
        ax.plot([0.0, 3.0], [b, b], color=ACCENT, lw=1.2, zorder=6)
    ax.set_xlabel("time (s)", fontsize=8.0)
    ax.set_ylabel("time (s)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Self-similarity matrix, %.0f s at %.1f s per frame"
                 % (total, HOP), fontsize=9.0, fontweight="bold", pad=5.0)
    safe_colorbar(fig, im, ax, label="mean-removed cosine similarity (-)",
                  width=0.11, pad=0.09, labelsize=7.0)

    # ---- novelty curve ----------------------------------------------------
    ax_nov.plot(t_frames, nov, color=RULE, lw=1.0)
    ax_nov.fill_between(t_frames, 0.0, nov, color="#3b5b92", alpha=0.14)
    for b in truth:
        ax_nov.axvline(b, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax_nov.plot(t_frames[peaks], nov[peaks], ls="none", marker="v", ms=4.0,
                color=ACCENT, zorder=4)
    ax_nov.set_xlim(0.0, total)
    ax_nov.set_ylim(0.0, 1.12)
    ax_nov.set_xlabel("time (s)", fontsize=7.5)
    ax_nov.set_ylabel("checkerboard novelty (a.u.)", fontsize=7.5)
    ax_nov.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_nov.spines[s].set_visible(False)
    ax_nov.set_title("Novelty peaks: %d of %d hits within %.0f s"
                     % (hits, truth.size, tol),
                     fontsize=8.5, fontweight="bold", pad=3.0)

    # ---- structure ribbons -------------------------------------------------
    for name, t0, t1 in sections:
        ax_rib.add_patch(Rectangle((t0, 0.66), t1 - t0, 0.26,
                                   facecolor=SECTION_COLOR[name],
                                   edgecolor="white", lw=0.7, alpha=0.92))
    for i in range(det.size - 1):
        ax_rib.add_patch(Rectangle((det[i], 0.34), det[i + 1] - det[i], 0.26,
                                   facecolor="#b3b3b3" if i % 2 else "#808080",
                                   edgecolor="white", lw=0.7))
    ax_rib.text(-1.0, 0.79, "annotated", ha="right", va="center",
                fontsize=6.4, color=SUB)
    ax_rib.text(-1.0, 0.47, "detected", ha="right", va="center",
                fontsize=6.4, color=SUB)
    handles = [Line2D([], [], marker="s", ls="none", ms=4.6,
                      color=SECTION_COLOR[k], label=k)
               for k in ("Intro", "Verse", "Chorus", "Bridge", "Outro")]
    ax_rib.legend(handles=handles, frameon=False, fontsize=6.4, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, -0.03),
                  handlelength=1.0, columnspacing=1.1, handletextpad=0.4)
    ax_rib.set_xlim(0.0, total)
    ax_rib.set_ylim(0.0, 1.0)
    ax_rib.set_yticks([])
    ax_rib.set_xlabel("time (s)", fontsize=7.5)
    ax_rib.tick_params(labelsize=6.5)
    for s in ("top", "right", "left"):
        ax_rib.spines[s].set_visible(False)
    ax_rib.set_title("Section sequence, %d segments detected"
                     % (det.size - 1), fontsize=8.5, fontweight="bold",
                     pad=3.0)

    panel_labels_inches(fig, [ax, ax_nov, ax_rib], dx=-0.46, dy=0.14)

    idx = [(int(t0 / HOP), int(t1 / HOP)) for _, t0, t1 in sections]
    same, diff = [], []
    for i, (a0, a1) in enumerate(idx):
        for j, (b0, b1) in enumerate(idx):
            block = ssm[a0:a1, b0:b1].mean()
            (same if FORM[i] == FORM[j] else diff).append(block)

    savefig(fig, "848_mir_structure_ssm.png")


if __name__ == "__main__":
    main()
