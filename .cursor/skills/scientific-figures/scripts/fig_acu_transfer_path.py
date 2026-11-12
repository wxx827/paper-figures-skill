"""Transfer path analysis of interior noise at the driver's ear.

Left: the path model - one source, five transfer paths, one receiver, with the
share of the broadband contribution each path carries.  Top right: the partial
pressure contributions stacked per third-octave band.  Bottom right: the
synthesised sum checked against the measured total, which is the only thing
that makes a path decomposition believable.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import (set_pub_style, savefig, style_axes, CATEGORICAL, COLORS)
from figkit import block, arrow, anchors, panel_label_inches

BANDS = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                  800, 1000, 1250, 1600, 2000], dtype=float)
P_REF = 0.02                                    # 20 uPa expressed in mPa
# path -> (broadband strength mPa, centre frequency Hz, log-width)
PATHS = {
    "Mount, front left": (7.4, 118.0, 0.42),
    "Mount, rear right": (5.6, 205.0, 0.46),
    "Exhaust hanger": (4.3, 78.0, 0.36),
    "Driveshaft bearing": (3.1, 430.0, 0.40),
    "Airborne, dash panel": (4.8, 780.0, 0.62),
}


def contributions(rng):
    """Partial pressure of every path in every band, in mPa."""
    out = {}
    for name, (amp, fc, w) in PATHS.items():
        shape = np.exp(-0.5 * (np.log(BANDS / fc) / w) ** 2)
        jitter = 10.0 ** (0.045 * rng.normal(size=BANDS.size))
        out[name] = amp * shape * jitter + 0.05
    return out


def to_db(p):
    return 20.0 * np.log10(np.asarray(p) / P_REF)


def draw_model(ax, shares):
    ax.set_xlim(0.0, 3.15)
    ax.set_ylim(0.0, 3.70)
    ax.set_aspect("equal")
    ax.axis("off")

    src = block(ax, 0.52, 1.85, 0.80, 2.35, "Engine\nblock", color="#3a3a3a",
                sub="source", fontsize=8.0, sub_color="#d5d5d5", rounding=0.08)
    rcv = anchors(2.80, 1.85, 0.44, 0.66)
    block(ax, rcv["x"], rcv["y"], rcv["w"], rcv["h"], "Ear", color="#555555",
          fontsize=8.0, rounding=0.08)
    ax.text(rcv["x"], rcv["y"] - rcv["h"] / 2 - 0.13, "receiver", ha="center",
            va="top", fontsize=6.8, color=COLORS["sub"])

    ys = np.linspace(3.28, 0.42, len(PATHS))
    for (name, col), y in zip(zip(PATHS, CATEGORICAL), ys):
        chip = block(ax, 1.62, y, 1.02, 0.40, name, color=col, fontsize=6.8,
                     rounding=0.07)
        y_src = float(np.clip(y, 1.85 - 1.00, 1.85 + 1.00))
        arrow(ax, (src["x"] + src["w"] / 2, y_src), chip["l"],
              color=COLORS["rule"], lw=1.0, gap=0.06, ms=8)
        arrow(ax, chip["r"], (rcv["x"] - rcv["w"] / 2, rcv["y"]),
              color=COLORS["rule"], lw=1.0, gap=0.06, ms=8)
        ax.text(chip["x"], y - 0.30, f"{shares[name] * 100:.0f} % of total",
                ha="center", va="top", fontsize=6.4, color=COLORS["sub"])


def main():
    set_pub_style()
    rng = np.random.default_rng(418)
    parts = contributions(rng)
    names = list(PATHS)

    synth = np.sum([parts[n] for n in names], axis=0)
    measured = synth * 10.0 ** (rng.normal(0.0, 0.30, BANDS.size) / 20.0)
    shares = {n: parts[n].sum() / synth.sum() for n in names}
    delta = to_db(measured) - to_db(synth)
    i_bad = int(np.argmax(np.abs(delta)))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.80, 1.0],
                          height_ratios=[1.0, 0.60], wspace=0.16, hspace=0.34,
                          left=0.035, right=0.975, top=0.885, bottom=0.100)
    ax_d = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])

    # ---- (a) path model ---------------------------------------------------
    draw_model(ax_d, shares)
    ax_d.set_title("Source - path - receiver model", fontsize=9.0,
                   fontweight="bold")

    # ---- (b) stacked band contributions -----------------------------------
    x = np.arange(BANDS.size)
    bottom = np.zeros(BANDS.size)
    for name, col in zip(names, CATEGORICAL):
        ax_s.bar(x, parts[name], bottom=bottom, width=0.78, color=col,
                 edgecolor="white", linewidth=0.5, label=name, zorder=3)
        bottom = bottom + parts[name]
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([f"{b:.0f}" if b < 1000 else f"{b / 1000:.2g}k"
                          for b in BANDS], fontsize=6.6)
    ax_s.set_xlim(-0.7, BANDS.size - 0.3)
    ax_s.set_ylabel("Partial contribution (mPa)", fontsize=8.0)
    ax_s.set_xlabel("Third-octave centre frequency (Hz)", fontsize=8.0)
    ax_s.tick_params(labelsize=7.0)
    ax_s.grid(axis="y", lw=0.5, alpha=0.6)
    ax_s.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=2,
                handlelength=1.1, columnspacing=0.9)
    ax_s.set_title("Path contributions per band", fontsize=8.5,
                   fontweight="bold")
    style_axes(ax_s)

    # ---- (c) synthesis check ----------------------------------------------
    ax_v.fill_between(x, to_db(synth), to_db(measured),
                      color=COLORS["accent"], alpha=0.22, lw=0, zorder=2,
                      label="residual")
    ax_v.plot(x, to_db(measured), color=COLORS["ink"], lw=1.5, marker="o",
              ms=3.2, mfc="white", mew=0.8, zorder=4, label="measured total")
    ax_v.plot(x, to_db(synth), color=COLORS["accent"], lw=1.4,
              ls=(0, (4.0, 2.0)), zorder=5, label="synthesised sum")
    ax_v.set_xticks(x)
    ax_v.set_xticklabels([f"{b:.0f}" if b < 1000 else f"{b / 1000:.2g}k"
                          for b in BANDS], fontsize=6.6)
    ax_v.set_xlim(-0.7, BANDS.size - 0.3)
    ax_v.set_ylabel("$L_p$ (dB)", fontsize=8.0)
    ax_v.set_xlabel("Third-octave centre frequency (Hz)", fontsize=8.0)
    ax_v.tick_params(labelsize=7.0)
    ax_v.grid(axis="y", lw=0.5, alpha=0.6)
    ax_v.legend(loc="lower left", fontsize=6.8, frameon=False, ncol=3,
                handlelength=1.4, columnspacing=1.0)
    ax_v.set_title(f"Synthesis check: max residual {np.abs(delta).max():.2f} dB"
                   f" at {BANDS[i_bad]:.0f} Hz", fontsize=8.5,
                   fontweight="bold")
    style_axes(ax_v)

    for ax, lab in zip((ax_d, ax_s, ax_v), ("(a)", "(b)", "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.10, dy=0.30)

    top = max(shares, key=shares.get)
    savefig(fig, "418_acu_transfer_path.png")


if __name__ == "__main__":
    main()
