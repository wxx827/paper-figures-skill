"""Neural decoding panel: time course, temporal generalisation, confusion, ROIs."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

TIME = np.arange(-200, 801, 10.0)        # ms relative to stimulus onset
N_SUB = 22
CLASSES = ["Face", "House", "Tool", "Body", "Scene"]
CHANCE = 1.0 / len(CLASSES)

ROIS = [("EVC (V1\u2013V3)", 0.415, 0.020, "#4c78a8"),
        ("LOC", 0.512, 0.024, "#4c78a8"),
        ("FFA", 0.583, 0.027, "#b1493f"),
        ("PPA", 0.498, 0.023, "#b1493f"),
        ("IPS", 0.352, 0.019, "#5aa17f"),
        ("dlPFC", 0.286, 0.021, "#5aa17f"),
        ("A1 (control)", 0.208, 0.017, "#9a9a9a")]

ACC = "#22557f"


def accuracy_profile(t):
    """Two-stage decoding profile: perceptual peak then a maintained plateau.

    Everything above chance is gated by an afferent-delay sigmoid, so the
    decoder carries no information before the response reaches cortex.
    """
    a = 0.40 * np.exp(-0.5 * ((t - 145) / 62) ** 2)
    a += 0.14 * np.exp(-0.5 * ((t - 330) / 130) ** 2)
    a += 0.085 / (1.0 + np.exp(-(t - 250) / 70)) \
        * np.exp(-np.clip(t - 250, 0, None) / 900)
    gate = 1.0 / (1.0 + np.exp(-(t - 58) / 11.0))
    return CHANCE + gate * a


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(31)

    # ---- subject-level decoding time courses -------------------------------
    prof = accuracy_profile(TIME)
    subs = np.empty((N_SUB, TIME.size))
    for i in range(N_SUB):
        gain = 1.0 + rng.normal(0, 0.22)
        shift = rng.normal(0, 14.0)
        s = CHANCE + (accuracy_profile(TIME - shift) - CHANCE) * gain
        subs[i] = s + gaussian_filter1d(rng.normal(0, 0.030, TIME.size), 1.6)
    mean = subs.mean(axis=0)
    sem = subs.std(axis=0, ddof=1) / np.sqrt(N_SUB)
    tval = (mean - CHANCE) / np.maximum(sem, 1e-9)
    sig = tval > 3.2

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.32, 1.0, 0.055],
                  height_ratios=[1.0, 1.0], wspace=0.36, hspace=0.34,
                  left=0.082, right=0.930, top=0.915, bottom=0.100)

    # ---- (a) decoding accuracy over time -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(TIME, mean - sem, mean + sem, color=ACC, alpha=0.22, lw=0)
    ax.plot(TIME, mean, color=ACC, lw=1.6, zorder=3)
    ax.axhline(CHANCE, color="#8a8a8a", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, color="#333333", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(TIME[0], TIME[-1])
    ax.set_ylim(0.12, 0.70)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.text(TIME[-1] - 8, CHANCE + 0.006, "chance (20%)", ha="right",
            va="bottom", fontsize=6.6, color="#7a7a7a")

    # keep only sustained clusters, so single noisy bins cannot set the onset
    ybar = 0.155
    d = np.diff(np.concatenate([[0], sig.view(np.int8), [0]]))
    clusters = [(lo, hi) for lo, hi in zip(np.flatnonzero(d == 1),
                                           np.flatnonzero(d == -1))
                if hi - lo >= 4]
    for lo, hi in clusters:
        ax.plot([TIME[lo], TIME[hi - 1]], [ybar, ybar], lw=3.0, color=ACC,
                solid_capstyle="butt", zorder=4)
    onset = TIME[max(clusters, key=lambda c: c[1] - c[0])[0]]
    ax.text(TIME[-1] - 8, ybar + 0.008, "$p$ < .001, cluster-corrected",
            ha="right", va="bottom", fontsize=6.4, color=ACC)

    pk = int(np.argmax(mean))
    ax.annotate(f"peak {mean[pk] * 100:.0f}%\nat {TIME[pk]:.0f} ms",
                xy=(TIME[pk], mean[pk]), xytext=(30, 6),
                textcoords="offset points", fontsize=6.8, fontweight="bold",
                color="#1a1a1a", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", lw=0.55, color="#9a9a9a"))
    ax.annotate(f"onset {onset:.0f} ms", xy=(onset, ybar), xytext=(-7, 15),
                textcoords="offset points", fontsize=6.6, color=ACC,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.55, color=ACC))
    style_axes(ax)
    add_panel_label(ax, "(a)", x=-0.125, y=1.03, size=10)

    # ---- (b) temporal generalisation matrix --------------------------------
    axg = fig.add_subplot(gs[0, 1])
    p = accuracy_profile(TIME) - CHANCE
    diag = np.outer(p, p) / max(p.max() ** 2, 1e-9)
    dt = np.abs(TIME[:, None] - TIME[None, :])
    G = CHANCE + p.max() * (0.86 * diag * np.exp(-(dt / 105.0) ** 1.5)
                            + 0.30 * diag)
    ramp = 1.0 / (1.0 + np.exp(-(TIME - 240) / 55.0))
    G += 0.16 * np.outer(ramp, ramp) * np.sqrt(np.abs(diag))
    G = gaussian_filter(G, 1.1)
    G += rng.normal(0, 0.006, G.shape)

    im = axg.imshow(G, origin="lower", cmap="magma",
                    extent=(TIME[0], TIME[-1], TIME[0], TIME[-1]),
                    vmin=CHANCE - 0.03, vmax=0.62, interpolation="bilinear",
                    aspect="equal")
    axg.contour(TIME, TIME, gaussian_filter(G, 1.4), levels=[0.30],
                colors="white", linewidths=0.7, alpha=0.9)
    axg.plot([TIME[0], TIME[-1]], [TIME[0], TIME[-1]], color="#dddddd",
             lw=0.6, ls=(0, (4, 3)))
    for v in (0,):
        axg.axhline(v, color="#dddddd", lw=0.6)
        axg.axvline(v, color="#dddddd", lw=0.6)
    axg.set_xlabel("Test time (ms)")
    axg.set_ylabel("Train time (ms)")
    axg.set_xticks([-200, 0, 400, 800])
    axg.set_yticks([-200, 0, 400, 800])
    for s in axg.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")
    add_panel_label(axg, "(b)", x=-0.20, y=1.03, size=10)
    axg.text(0.5, 1.02, "temporal generalisation", transform=axg.transAxes,
             ha="center", va="bottom", fontsize=7.4, color="#444444")

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0.2, 0.3, 0.4, 0.5, 0.6])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.2, color="#333333")
    cb.set_label("Accuracy", fontsize=7, labelpad=3)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + pos.height * 0.13, pos.width,
                      pos.height * 0.70])

    # ---- (c) confusion matrix ----------------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    n = len(CLASSES)
    conf = np.full((n, n), 0.10) + rng.normal(0, 0.018, (n, n))
    np.fill_diagonal(conf, [0.71, 0.64, 0.58, 0.52, 0.61])
    conf[0, 3] += 0.09        # faces <-> bodies
    conf[3, 0] += 0.11
    conf[1, 4] += 0.13        # houses <-> scenes
    conf[4, 1] += 0.12
    conf = np.clip(conf, 0.01, None)
    conf /= conf.sum(axis=1, keepdims=True)

    imc = axc.imshow(conf, cmap="Blues", vmin=0, vmax=0.72,
                     interpolation="nearest")
    for i in range(n):
        for j in range(n):
            axc.text(j, i, f"{conf[i, j] * 100:.0f}", ha="center", va="center",
                     fontsize=6.6, color="white" if conf[i, j] > 0.40
                     else "#333333",
                     fontweight="bold" if i == j else "normal")
    axc.set_xticks(range(n), CLASSES, fontsize=6.8, rotation=32, ha="right")
    axc.set_yticks(range(n), CLASSES, fontsize=6.8)
    axc.set_xlabel("Predicted class", labelpad=1)
    axc.set_ylabel("True class")
    axc.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axc.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axc.grid(which="minor", color="white", lw=1.0)
    axc.tick_params(which="minor", length=0)
    for s in axc.spines.values():
        s.set_visible(False)
    add_panel_label(axc, "(c)", x=-0.125, y=1.03, size=10)
    axc.text(1.0, 1.02, "row percentages, 300\u2013500 ms",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=6.6,
             color="#777777")

    # ---- (d) per-ROI decoding ----------------------------------------------
    axr = fig.add_subplot(gs[1, 1:])
    names = [r[0] for r in ROIS]
    vals = np.array([r[1] for r in ROIS])
    errs = np.array([r[2] for r in ROIS])
    cols = [r[3] for r in ROIS]
    y = np.arange(len(ROIS))[::-1]

    axr.barh(y, vals - CHANCE, left=CHANCE, height=0.62, color=cols,
             edgecolor="white", lw=0.5, zorder=3)
    axr.errorbar(vals, y, xerr=errs, fmt="none", ecolor="#333333",
                 elinewidth=0.8, capsize=2.0, zorder=4)
    axr.axvline(CHANCE, color="#8a8a8a", lw=0.9, ls=(0, (3, 2)), zorder=2)
    for yi, v, e in zip(y, vals, errs):
        stars = "***" if v - CHANCE > 6 * e else "**" if v - CHANCE > 3 * e \
            else "n.s."
        axr.text(v + e + 0.012, yi, stars, va="center", ha="left",
                 fontsize=6.6, fontweight="bold",
                 color="#333333" if stars != "n.s." else "#9a9a9a")
    axr.set_yticks(y, names, fontsize=7)
    axr.set_xlim(0.15, 0.68)
    axr.set_ylim(-0.7, len(ROIS) - 0.3)
    axr.set_xlabel("Decoding accuracy (300\u2013500 ms)")
    axr.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
    axr.text(CHANCE, len(ROIS) - 0.45, "chance", ha="center", va="bottom",
             fontsize=6.4, color="#7a7a7a")
    axr.grid(axis="x", lw=0.4, alpha=0.55)
    style_axes(axr)
    add_panel_label(axr, "(d)", x=-0.185, y=1.03, size=10)


    savefig(fig, "142_neural_decoding.png")


if __name__ == "__main__":
    main()
