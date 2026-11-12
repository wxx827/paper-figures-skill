"""Stacked multichannel EEG traces with L-shaped scale bar (no y axis)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from pubstyle import set_pub_style, savefig

FS = 250.0
DUR = 6.5
SPACING = 120.0            # microvolts between channel baselines

CHANNELS = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
            "PO3", "PO4", "O1", "Oz", "O2"]

FOCUS = {"F7": 1.00, "T7": 1.00, "P7": 0.78}          # left temporal focus
SPREAD = {"F3": 0.30, "C3": 0.34, "CP5": 0.42, "P3": 0.26, "FC5": 0.55}
POSTERIOR = ["PO3", "PO4", "O1", "Oz", "O2"]

C_BASE = "#39434d"
C_IED = "#b23a48"
C_ALPHA = "#1f7a63"

EVENTS = [(1.20, "eyes closed"), (3.10, "IED"), (5.60, "eyes open")]
IED_TIMES = [3.10, 4.02, 5.05]
BLINKS = [2.05, 6.05]


def pink(rng, n, exponent=1.4):
    W = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    x = np.fft.irfft(W * f ** (-exponent / 2.0), n)
    return x / x.std()


def build(rng):
    n = int(DUR * FS)
    t = np.arange(n) / FS
    nch = len(CHANNELS)

    # spatially smooth background from a handful of shared latent sources
    sources = np.array([pink(rng, n) for _ in range(7)])
    mix = rng.normal(0, 1, (nch, 7)) * np.array([1.0, .9, .8, .7, .6, .5, .5])
    X = (mix @ sources) * 6.2
    X += np.array([pink(rng, n) for _ in range(nch)]) * 8.0

    # posterior alpha, present only while the eyes are closed
    env = ((t > 1.20) & (t < 5.60)).astype(float)
    w = np.hanning(int(0.5 * FS) | 1)
    env = np.convolve(env, w / w.sum(), mode="same")
    for name in POSTERIOR:
        i = CHANNELS.index(name)
        amp = 26.0 if name in ("O1", "Oz", "O2") else 16.0
        X[i] += amp * env * np.sin(2 * np.pi * 9.8 * t + rng.uniform(0, 6.3))
        X[i] += 0.35 * amp * env * np.sin(2 * np.pi * 11.3 * t
                                          + rng.uniform(0, 6.3))

    # blink artefact, frontally dominant
    front = {"Fp1": 1.0, "Fp2": 0.96, "AF3": 0.72, "AF4": 0.70,
             "F7": 0.34, "F8": 0.32, "F3": 0.28, "Fz": 0.26, "F4": 0.28}
    for bt in BLINKS:
        shape = np.exp(-0.5 * ((t - bt) / 0.075) ** 2) \
            - 0.28 * np.exp(-0.5 * ((t - bt - 0.20) / 0.13) ** 2)
        for name, g in front.items():
            X[CHANNELS.index(name)] += 145.0 * g * shape

    # interictal epileptiform discharges: sharp spike + slow after-wave
    gains = dict(FOCUS)
    gains.update(SPREAD)
    for it in IED_TIMES:
        spike = -np.exp(-0.5 * ((t - it) / 0.011) ** 2)
        wave = 0.42 * np.exp(-0.5 * ((t - it - 0.085) / 0.048) ** 2)
        for name, g in gains.items():
            X[CHANNELS.index(name)] += 155.0 * g * (spike + wave)
    return t, X


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(17)
    t, X = build(rng)
    nch = len(CHANNELS)

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.105)
    trx = blended_transform_factory(ax.transAxes, ax.transData)

    for i, name in enumerate(CHANNELS):
        y0 = -i * SPACING
        if name in FOCUS:
            col, lw, z = C_IED, 0.72, 4
        elif name in ("O1", "Oz", "O2"):
            col, lw, z = C_ALPHA, 0.65, 3
        else:
            col, lw, z = C_BASE, 0.45, 2
        if name in FOCUS or name in ("O1", "Oz", "O2"):
            ax.add_patch(Rectangle((0, y0 - SPACING * 0.46), DUR,
                                   SPACING * 0.92, facecolor=col, alpha=0.055,
                                   edgecolor="none", zorder=0))
        ax.plot(t, np.clip(X[i], -SPACING * 1.35, SPACING * 1.35) + y0,
                color=col, lw=lw, zorder=z, solid_joinstyle="round")
        ax.text(-0.012, y0, name, transform=trx, ha="right", va="center",
                fontsize=6.4, color=col,
                fontweight="bold" if col != C_BASE else "normal")

    bottom = -(nch - 1) * SPACING
    ax.set_xlim(0, DUR)
    ax.set_ylim(bottom - SPACING * 2.6, SPACING * 1.5)
    ax.axis("off")

    # ---- events ------------------------------------------------------------
    for et, lab in EVENTS:
        ax.plot([et, et], [bottom - SPACING * 0.75, SPACING * 0.9],
                color="#6b6b6b", lw=0.8, ls=(0, (3, 2)), zorder=1, alpha=0.85)
        ax.text(et, SPACING * 1.15, lab, ha="center", va="bottom",
                fontsize=7.2, fontweight="bold", color="#333333")
        ax.plot([et], [SPACING * 0.95], marker="v", ms=3.4, color="#333333",
                zorder=5)
    for it in IED_TIMES:
        ax.text(it, -CHANNELS.index("F7") * SPACING + SPACING * 0.62, "\u2605",
                ha="center", va="bottom", fontsize=6.5, color=C_IED, zorder=6)

    # ---- L-shaped scale bar (replaces the y axis) --------------------------
    x0, y0 = DUR - 0.72, bottom - SPACING * 1.75
    ax.plot([x0, x0, x0 + 0.5], [y0 + 100.0, y0, y0], color="#1a1a1a",
            lw=1.4, solid_joinstyle="miter", solid_capstyle="butt", zorder=6)
    ax.text(x0 - 0.045, y0 + 50.0, "100 \u03bcV", ha="right", va="center",
            fontsize=7.2, fontweight="bold", color="#1a1a1a")
    ax.text(x0 + 0.25, y0 - 34.0, "500 ms", ha="center", va="top",
            fontsize=7.2, fontweight="bold", color="#1a1a1a")

    handles = [
        Line2D([], [], color=C_IED, lw=1.4,
               label="left temporal focus (F7\u2013T7\u2013P7)"),
        Line2D([], [], color=C_ALPHA, lw=1.4,
               label="occipital \u03b1 rhythm (O1\u2013Oz\u2013O2)"),
        Line2D([], [], color=C_BASE, lw=1.0, label="other channels"),
        Line2D([], [], color=C_IED, lw=0, marker="$\u2605$", ms=5,
               label="epileptiform discharge"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False,
              fontsize=7, handlelength=1.5, columnspacing=1.6,
              bbox_to_anchor=(-0.005, -0.060))


    savefig(fig, "139_multichannel_traces.png")


if __name__ == "__main__":
    main()
