"""Polar direction-tuning curves with von Mises fits for six neurons."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, add_panel_label

N_DIR = 12
N_TRIALS = 10
WINDOW = 1.0                       # s over which spikes are counted

CLASS_COLOR = {"DS": "#2a5d8f", "OS": "#c9772e", "NS": "#808080"}
CLASS_NAME = {"DS": "direction-selective",
              "OS": "orientation-selective",
              "NS": "untuned"}

NEURONS = [
    dict(tag="Neuron 1", area="V1 L2/3", cls="DS",
         r0=3.0, rp=46.0, rn=3.0, kappa=3.6, pref=45.0),
    dict(tag="Neuron 2", area="V1 L4", cls="OS",
         r0=4.0, rp=38.0, rn=34.0, kappa=4.2, pref=120.0),
    dict(tag="Neuron 3", area="MT", cls="DS",
         r0=6.0, rp=31.0, rn=8.0, kappa=1.2, pref=200.0),
    dict(tag="Neuron 4", area="V1 L2/3", cls="OS",
         r0=2.5, rp=43.0, rn=40.0, kappa=8.5, pref=285.0),
    dict(tag="Neuron 5", area="V1 L5", cls="NS",
         r0=17.0, rp=11.0, rn=9.0, kappa=0.7, pref=15.0),
    dict(tag="Neuron 6", area="MT", cls="DS",
         r0=11.0, rp=36.0, rn=13.0, kappa=2.4, pref=330.0),
]


def dvm(theta, r0, rp, rn, kappa, tp):
    """Sum of two opposed von Mises bumps (direction + null direction)."""
    return (r0
            + rp * np.exp(kappa * (np.cos(theta - tp) - 1.0))
            + rn * np.exp(kappa * (np.cos(theta - tp - np.pi) - 1.0)))


def simulate(rng, cfg):
    theta = np.deg2rad(np.arange(N_DIR) * 360.0 / N_DIR)
    true = dvm(theta, cfg["r0"], cfg["rp"], cfg["rn"], cfg["kappa"],
               np.deg2rad(cfg["pref"]))
    counts = rng.poisson(np.repeat(true[None, :] * WINDOW, N_TRIALS, axis=0))
    rates = counts / WINDOW
    return theta, rates.mean(axis=0), rates.std(axis=0, ddof=1) / np.sqrt(N_TRIALS)


def fit(theta, mean):
    imax = int(np.argmax(mean))
    p0 = [mean.min(), mean.max() - mean.min(),
          (mean.max() - mean.min()) * 0.4, 2.0, theta[imax]]
    bounds = ([0, 0, 0, 0.05, -2 * np.pi], [np.inf, np.inf, np.inf, 40, 4 * np.pi])
    popt, _ = curve_fit(dvm, theta, mean, p0=p0, bounds=bounds, maxfev=40000)
    r0, rp, rn, kappa, tp = popt
    if rn > rp:                      # keep tp at the stronger of the two lobes
        rp, rn = rn, rp
        tp += np.pi
    return r0, rp, rn, kappa, tp % (2 * np.pi)


def metrics(theta, mean, kappa):
    """Vector-strength DSI / OSI (robust for weakly tuned cells) + fit width."""
    r = np.clip(mean, 0.0, None)
    tot = r.sum()
    dsi = np.abs((r * np.exp(1j * theta)).sum()) / tot
    osi = np.abs((r * np.exp(2j * theta)).sum()) / tot
    c = 1.0 + np.log(0.5) / kappa
    hwhm = 180.0 if c < -1 else np.rad2deg(np.arccos(c))
    return dsi, osi, hwhm


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(5)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.3),
                             subplot_kw=dict(projection="polar"))
    fig.subplots_adjust(left=0.055, right=0.965, top=0.935, bottom=0.190,
                        wspace=0.40, hspace=0.76)

    grid = np.linspace(0, 2 * np.pi, 361)

    # Fit every neuron first: all six panels then share one radial scale, so a
    # weakly responsive cell cannot fill the disc like a strongly tuned one.
    fits = []
    for cfg in NEURONS:
        theta, mean, sem = simulate(rng, cfg)
        r0, rp, rn, kappa, tp = fit(theta, mean)
        fits.append((cfg, theta, mean, sem, kappa, tp,
                     dvm(grid, r0, rp, rn, kappa, tp)))
    rmax = max(max(cu.max(), (m + s).max())
               for _, _, m, s, _, _, cu in fits) * 1.20
    step = 20 if rmax > 45 else 10
    ticks = np.arange(step, rmax * 0.90, step)

    for k, (ax, (cfg, theta, mean, sem, kappa, tp, curve)) in enumerate(
            zip(axes.ravel(), fits)):
        col = CLASS_COLOR[cfg["cls"]]
        dsi, osi, hwhm = metrics(theta, mean, kappa)

        ax.fill_between(grid, 0, curve, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(grid, curve, color=col, lw=1.4, zorder=3)
        ax.errorbar(np.append(theta, theta[0]), np.append(mean, mean[0]),
                    yerr=np.append(sem, sem[0]), fmt="o", ms=2.9,
                    mfc="white", mec="#1a1a1a", mew=0.7, ecolor="#5c5c5c",
                    elinewidth=0.7, capsize=1.4, ls="none", zorder=4)

        ax.annotate("", xy=(tp, min(curve.max() * 1.04, rmax * 0.94)),
                    xytext=(tp, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=col,
                                    mutation_scale=7,
                                    shrinkA=0, shrinkB=0), zorder=5)

        ax.set_rlim(0, rmax)
        ax.set_rticks(ticks)
        ax.set_yticklabels([f"{v:.0f}" for v in ticks], fontsize=6.0,
                           color="#6b6b6b")
        ax.set_rlabel_position(148)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"],
                           fontsize=6.4, color="#4d4d4d")
        ax.tick_params(pad=0.4)
        ax.grid(color="#dcdcdc", lw=0.5)
        ax.spines["polar"].set_linewidth(0.8)
        ax.spines["polar"].set_edgecolor("#333333")
        ax.set_title(f"{cfg['tag']}  \u00b7  {cfg['area']}", fontsize=8.2,
                     pad=11, color="#1a1a1a")
        ax.text(0.5, -0.155, CLASS_NAME[cfg["cls"]], transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=col,
                fontweight="bold")
        # a flat tuning curve leaves kappa unidentifiable, so no width is shown
        width = f"HWHM {hwhm:.0f}\u00b0" if max(dsi, osi) >= 0.12 \
            else "HWHM n.d."
        ax.text(0.5, -0.265,
                f"pref {np.rad2deg(tp):.0f}\u00b0   \u00b7   {width}\n"
                f"DSI {dsi:.2f}   \u00b7   OSI {osi:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.6,
                color="#2b2b2b", linespacing=1.5)
        add_panel_label(ax, f"({chr(97 + k)})", x=-0.15, y=1.02, size=10)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.4, mfc="white",
               mec="#1a1a1a", mew=0.7, label="mean \u00b1 SEM (10 trials)"),
        Line2D([], [], color="#555555", lw=1.4, label="von Mises fit"),
        Line2D([], [], color="#555555", lw=1.0, marker=">", ms=3.5,
               label="preferred direction"),
    ]
    handles += [Line2D([], [], color=c, lw=3.2, alpha=0.55,
                       label=CLASS_NAME[k]) for k, c in CLASS_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=7, handlelength=1.5, columnspacing=1.5,
               bbox_to_anchor=(0.51, 0.030))

    savefig(fig, "140_tuning_curves.png")


if __name__ == "__main__":
    main()
