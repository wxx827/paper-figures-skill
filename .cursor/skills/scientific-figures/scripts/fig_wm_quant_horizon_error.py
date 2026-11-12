"""Rollout error against prediction horizon, with the terminal error boxed.

Main panel: open-loop rollout error of four world models as the horizon grows,
markers carrying +/-1 s.d. over 24 held-out episodes.  The shaded band is the
region where our model stays under the usable-error threshold.  The narrow right
panel opens up the distribution behind the very last point of each curve, on the
same y range as the main panel.

Synthetic data lives in ``gen_rollouts`` only -- swap that one function for a
real loader and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import shared_limits

METHODS = ["Genie-style", "IRIS", "DreamerV3", "Ours (WM-8)"]
SHORT = {"Genie-style": "Genie", "IRIS": "IRIS", "DreamerV3": "DreamerV3",
         "Ours (WM-8)": "Ours"}
# cool-blue baselines, one warm accent reserved for Ours
COL = {"Genie-style": "#8e6bbf", "IRIS": "#2a7f7a", "DreamerV3": "#3b5b92",
       "Ours (WM-8)": "#b23a48"}
LW = {m: (2.2 if m == "Ours (WM-8)" else 1.6) for m in METHODS}
MK = {"Genie-style": "^", "IRIS": "s", "DreamerV3": "D", "Ours (WM-8)": "o"}

# method -> (saturating amplitude, time constant, late linear drift)
SPEC = {
    "Genie-style": (0.46, 9.0, 0.0125),
    "IRIS": (0.38, 11.0, 0.0102),
    "DreamerV3": (0.30, 12.5, 0.0081),
    "Ours (WM-8)": (0.17, 15.0, 0.0043),
}

H_MAX = 48
N_EP = 24
THRESH = 0.30          # usable-error threshold
GRID = "#ededed"


def gen_rollouts(seed=1503):
    """{method: (n_episode, H_MAX) rollout error} plus the horizon axis."""
    rng = np.random.default_rng(seed)
    h = np.arange(1, H_MAX + 1, dtype=float)
    out = {}
    for m in METHODS:
        amp, tau, drift = SPEC[m]
        runs = np.empty((N_EP, h.size))
        for e in range(N_EP):
            a = amp * rng.normal(1.0, 0.11)
            d = drift * rng.normal(1.0, 0.16)
            base = a * (1.0 - np.exp(-h / (tau * rng.normal(1.0, 0.10)))) + d * h
            # error compounds, so the episode spread grows with the horizon
            jitter = rng.normal(0.0, 0.010 + 0.0016 * h)
            runs[e] = np.maximum(base + jitter, 0.005)
        out[m] = runs
    return h, out


def reliable_horizon(h, mean_curve, thresh=THRESH):
    """Last horizon at which the mean error is still under the threshold."""
    ok = np.nonzero(mean_curve <= thresh)[0]
    return h[ok[-1]] if ok.size else h[0]


def main():
    set_pub_style(base_font=8)
    h, runs = gen_rollouts()

    mean = {m: runs[m].mean(axis=0) for m in METHODS}
    sd = {m: runs[m].std(axis=0, ddof=1) for m in METHODS}
    tail = {m: runs[m][:, -1] for m in METHODS}

    # one shared y range covering both panels, before anything is drawn
    pool = [mean[m] + sd[m] for m in METHODS] + [mean[m] - sd[m] for m in METHODS]
    pool += [tail[m] for m in METHODS]
    ylo, yhi = shared_limits(pool, pad=0.07)
    ylo = max(0.0, ylo)

    h_star = reliable_horizon(h, mean["Ours (WM-8)"])
    base_star = max(reliable_horizon(h, mean[m]) for m in METHODS[:-1])
    gain = h_star / base_star

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.30], left=0.062,
                          right=0.982, top=0.865, bottom=0.115, wspace=0.045)
    ax = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    for a in (ax, ax_b):
        a.set_axisbelow(True)
        a.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        a.set_ylim(ylo, yhi)
        style_axes(a)

    # ---- reliable region ----------------------------------------------------
    ax.axvspan(0.5, h_star, facecolor="#eaf0f6", lw=0, zorder=1)
    ax.axvline(h_star, color="#3b5b92", lw=1.0, ls=(0, (3.2, 2.0)), zorder=2)
    ax.axvline(base_star, color="#9a9a9a", lw=0.9, ls=(0, (2.4, 2.0)),
               zorder=2)
    ax.axhline(THRESH, color="#8a8a8a", lw=0.9, ls=(0, (2.6, 2.0)), zorder=2)
    mid = (base_star + h_star) / 2.0
    ax.text(mid, yhi * 0.972, "reliable region", ha="center", va="top",
            fontsize=9.0, color="#3b5b92", fontweight="bold")
    ax.text(mid, yhi * 0.915,
            f"Ours usable to $H = {h_star:.0f}$, best baseline to "
            f"$H = {base_star:.0f}$", ha="center", va="top", fontsize=7.4,
            color="#555555")

    y_arrow = ylo + 0.055 * (yhi - ylo)
    ax.annotate("", xy=(h_star, y_arrow), xytext=(base_star, y_arrow),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="#3b5b92",
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text((base_star + h_star) / 2.0, y_arrow + 0.012 * (yhi - ylo),
            f"{gain:.1f}$\\times$ longer usable horizon", ha="center",
            va="bottom", fontsize=8.0, color="#3b5b92", fontweight="bold",
            zorder=8)

    # ---- horizon curves -----------------------------------------------------
    every = 4
    for m in METHODS:
        hot = m == "Ours (WM-8)"
        ax.fill_between(h, mean[m] - sd[m], mean[m] + sd[m], color=COL[m],
                        alpha=0.16 if hot else 0.10, lw=0, zorder=3)
        ax.plot(h, mean[m], color=COL[m], lw=LW[m], zorder=6 if hot else 4,
                solid_capstyle="round")
        idx = np.arange(every - 1, h.size, every)
        ax.errorbar(h[idx], mean[m][idx], yerr=sd[m][idx], fmt=MK[m],
                    ms=4.2 if hot else 3.4, mfc="white", mec=COL[m],
                    mew=1.25 if hot else 0.95, ecolor=COL[m], elinewidth=0.9,
                    capsize=2.2, capthick=0.9, ls="none",
                    zorder=7 if hot else 5)

    ax.set_xlim(0.5, H_MAX + 0.5)
    ax.set_xticks([1, 8, 16, 24, 32, 40, 48])
    ax.set_xlabel("Prediction horizon  $H$ (frames, open loop)", fontsize=9.0)
    ax.set_ylabel("Rollout error  (LPIPS, lower is better)", fontsize=9.0)
    ax.set_title("Error compounds with the horizon", fontsize=10.0,
                 fontweight="bold", pad=5.0)

    handles = [Line2D([], [], color=COL[m], lw=LW[m], marker=MK[m], ms=4.2,
                      mfc="white", mec=COL[m], mew=1.1, label=m)
               for m in METHODS]
    handles += [
        Patch(facecolor="#eaf0f6", lw=0, label="reliable region"),
        Line2D([], [], color="#8a8a8a", lw=0.9, ls=(0, (2.6, 2.0)),
               label=f"threshold ($\\mathrm{{LPIPS}} = {THRESH:.2f}$)"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=8.8, handlelength=1.9, labelspacing=0.5,
                    handletextpad=0.6, borderpad=0.5)
    leg.get_texts()[3].set_fontweight("bold")
    for t in leg.get_texts()[4:]:
        t.set_fontsize(7.6)
        t.set_color("#555555")

    # ---- terminal-error box panel ------------------------------------------
    ax_b.axhline(THRESH, color="#8a8a8a", lw=0.9, ls=(0, (2.6, 2.0)), zorder=2)
    data = [tail[m] for m in METHODS]
    bp = ax_b.boxplot(data, positions=np.arange(len(METHODS)), widths=0.62,
                      patch_artist=True, showfliers=True, zorder=3,
                      medianprops=dict(color="#1a1a1a", lw=1.2),
                      whiskerprops=dict(lw=0.9, color="#555555"),
                      capprops=dict(lw=0.9, color="#555555"),
                      flierprops=dict(marker="o", ms=2.2, mfc="none",
                                      mec="#9a9a9a", mew=0.6))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(lighten(COL[m], 0.62))
        patch.set_edgecolor(COL[m])
        patch.set_linewidth(1.1)
    for j, m in enumerate(METHODS):
        ax_b.plot(j, tail[m].mean(), marker="D", ms=3.2, mfc="white",
                  mec=COL[m], mew=1.1, zorder=5)

    ax_b.set_xlim(-0.62, len(METHODS) - 0.38)
    ax_b.set_xticks(np.arange(len(METHODS)))
    ax_b.set_xticklabels([SHORT[m] for m in METHODS], fontsize=7.6,
                         rotation=38, ha="right")
    for lab, m in zip(ax_b.get_xticklabels(), METHODS):
        lab.set_color(COL[m])
    ax_b.get_xticklabels()[-1].set_fontweight("bold")
    ax_b.tick_params(labelleft=False, length=0)
    ax_b.set_title(f"At $H = {H_MAX}$", fontsize=9.0, fontweight="bold",
                   pad=5.0)
    ax_b.text(0.5, -0.175, f"$n = {N_EP}$ episodes", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=6.8, color="#777777")

    savefig(fig, "1098_wm_quant_horizon_error.png")


if __name__ == "__main__":
    main()
