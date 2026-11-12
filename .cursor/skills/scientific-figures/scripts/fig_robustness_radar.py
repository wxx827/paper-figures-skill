"""Robustness under six corruption families.

The radar on the left is the headline panel.  The severity sweeps on the
right are deliberately drawn as a block of small multiples at roughly half
the linear size and sit on a tinted card, so they read as supporting
evidence rather than as a second main result.  All six small panels share
one y range computed from the full data set.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

OURS = "#3b5b92"
GREY = "#8c8c8c"
GREEN = "#2e7d5b"
ORANGE = "#c9a227"
GRID = "#eeeeee"

METHODS = ["Baseline", "AugMix", "DeepAugment", "Ours"]
MCOL = {"Baseline": GREY, "AugMix": GREEN, "DeepAugment": ORANGE,
        "Ours": OURS}
MLW = {"Baseline": 1.3, "AugMix": 1.3, "DeepAugment": 1.3, "Ours": 2.0}

AXES_LAB = ["Noise", "Blur", "Weather", "Compression", "Adversarial",
            "Occlusion"]
# accuracy (%) at severity 3, one row per method
RADAR = {
    "Baseline":    [42, 38, 45, 52, 18, 47],
    "AugMix":      [58, 55, 60, 63, 26, 55],
    "DeepAugment": [62, 60, 63, 66, 30, 58],
    "Ours":        [69, 67, 71, 73, 41, 65],
}
CLEAN = {"Baseline": 79.2, "AugMix": 79.8, "DeepAugment": 80.1, "Ours": 81.4}
SEV = np.arange(1, 6)


def sweeps():
    """Accuracy as a function of severity, anchored on the radar values."""
    out = {}
    for m in METHODS:
        out[m] = {}
        for k, lab in enumerate(AXES_LAB):
            mid = RADAR[m][k]
            c = CLEAN[m]
            end = max(mid - 0.55 * (c - mid), 3.0)
            # quadratic through (0, clean), (3, mid), (5, end)
            p = np.polyfit([0, 3, 5], [c, mid, end], 2)
            out[m][lab] = np.polyval(p, SEV)
    return out


def main():
    set_pub_style(base_font=8)
    sw = sweeps()

    lo = min(v.min() for m in METHODS for v in sw[m].values())
    hi = max(v.max() for m in METHODS for v in sw[m].values())
    ylo = max(0.0, 5.0 * np.floor((lo - 3) / 5.0))
    yhi = 5.0 * np.ceil((hi + 3) / 5.0)
    print(f"  shared small-panel y range: [{ylo:.0f}, {yhi:.0f}]")

    fig = plt.figure(figsize=(7.5, 4.25))

    # ---- tinted card behind the subordinate block --------------------------
    card = FancyBboxPatch((0.497, 0.105), 0.497, 0.785,
                          boxstyle="round,pad=0,rounding_size=0.012",
                          transform=fig.transFigure, facecolor="#f6f7f9",
                          edgecolor="#e2e5e9", lw=0.8, zorder=-5)
    fig.add_artist(card)

    # ---- (a) radar ---------------------------------------------------------
    ax = fig.add_axes([0.035, 0.125, 0.40, 0.695], polar=True)
    n = len(AXES_LAB)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.r_[ang, ang[:1]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for m in METHODS:
        hot = m == "Ours"
        v = np.array(RADAR[m], dtype=float)
        vc = np.r_[v, v[:1]]
        ax.plot(closed, vc, color=MCOL[m], lw=MLW[m], zorder=6 if hot else 4,
                solid_joinstyle="round")
        ax.fill(closed, vc, color=MCOL[m], alpha=0.16 if hot else 0.08,
                lw=0, zorder=3)
        ax.plot(ang, v, ls="none", marker="o", ms=3.4 if hot else 2.6,
                mfc="white", mec=MCOL[m], mew=1.2 if hot else 0.9,
                zorder=7 if hot else 5)

    ax.set_xticks(ang)
    ax.set_xticklabels(AXES_LAB, fontsize=7.6)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylim(0, 80)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=6.2,
                       color="#8a8a8a")
    for t in ax.get_yticklabels():
        t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75,
                        pad=0.8))
    ax.set_rlabel_position(28.0)
    ax.grid(color="#dcdcdc", lw=0.6)
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)

    mean_gain = np.mean(np.array(RADAR["Ours"]) -
                        np.array(RADAR["DeepAugment"]))
    fig.text(0.030, 0.175, f"Ours gains $+${mean_gain:.1f} points\non average "
             f"over the\nstrongest baseline", ha="left", va="top",
             fontsize=6.9, color=OURS, linespacing=1.45)

    # ---- (b) subordinate small multiples -----------------------------------
    x0, w, gap = 0.573, 0.1225, 0.0285
    rows = (0.500, 0.210)
    small = []
    for k, lab in enumerate(AXES_LAB):
        r, c = divmod(k, 3)
        axs = fig.add_axes([x0 + c * (w + gap), rows[r], w, 0.185])
        small.append(axs)
        axs.set_axisbelow(True)
        axs.grid(color="#e6e8ea", lw=0.5, zorder=0)
        axs.set_facecolor("white")
        for m in METHODS:
            hot = m == "Ours"
            axs.plot(SEV, sw[m][lab], color=MCOL[m],
                     lw=1.5 if hot else 0.9, zorder=6 if hot else 4,
                     marker="o", ms=2.4 if hot else 1.8, mfc="white",
                     mew=0.8 if hot else 0.6, mec=MCOL[m])
        axs.set_xlim(0.7, 5.3)
        axs.set_ylim(ylo, yhi)
        axs.set_xticks(SEV)
        axs.set_yticks(np.arange(ylo, yhi + 1, 20))
        axs.set_title(lab, fontsize=6.8, pad=2.0)
        axs.tick_params(labelsize=6.0, length=2.0)
        if c:
            axs.set_yticklabels([])
        if r == 0:
            axs.set_xticklabels([])
        style_axes(axs)

    fig.text(0.524, (rows[1] + rows[0] + 0.185) / 2, "accuracy (%)",
             rotation=90, ha="center", va="center", fontsize=6.8,
             color="#444444")
    fig.text(0.745, 0.140, "corruption severity (1\u20135)", ha="center",
             va="bottom", fontsize=7.0, color="#444444")
    fig.text(0.512, 0.872, "(b)", fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.762, 0.872, "Degradation with severity (shared y range)",
             ha="center", va="top", fontsize=7.6, color="#333333")

    handles = [Line2D([], [], color=MCOL[m], lw=MLW[m], label=m)
               for m in METHODS]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.008), frameon=False, fontsize=7.4,
                     handlelength=2.0, columnspacing=2.0)
    leg.get_texts()[3].set_fontweight("bold")

    for m in METHODS:
        print(f"  {m:12s} severity-3 mean {np.mean(RADAR[m]):5.1f}%  "
              f"clean {CLEAN[m]:.1f}%")
    savefig(fig, "309_robustness_radar.png")


if __name__ == "__main__":
    main()
