"""Training / validation accuracy curves for four methods on six datasets.

2 x 3 grid.  Hue encodes the method, line style encodes the split (solid =
validation, dashed = train), the shaded ribbon is +-1 sigma over five seeds.
Every panel is drawn on the *same* y range, computed up-front from all
curves, so accuracies stay visually comparable across datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes

OURS = "#3b5b92"
GREY = "#8c8c8c"
GREEN = "#2e7d5b"
ORANGE = "#c9a227"
GRID = "#eeeeee"

METHODS = ["Baseline", "AugMix", "SAM", "Ours"]
MCOL = {"Baseline": GREY, "AugMix": GREEN, "SAM": ORANGE, "Ours": OURS}
MLW = {"Baseline": 1.3, "AugMix": 1.3, "SAM": 1.3, "Ours": 2.0}

# dataset -> (final val accuracy of "Ours", epoch-1 accuracy, time constant)
DATASETS = [
    ("CIFAR-10", 95.1, 63.0, 14.0),
    ("CIFAR-100", 79.6, 47.0, 18.0),
    ("SVHN", 96.8, 76.0, 10.0),
    ("STL-10", 88.2, 53.0, 20.0),
    ("Tiny-ImageNet", 70.4, 41.0, 22.0),
    ("Food-101", 85.3, 49.0, 17.0),
]

# method -> (val offset from "Ours", train-val generalisation gap, late decay)
SPEC = {
    "Baseline": (-5.2, 7.4, 1.60),
    "AugMix": (-3.1, 4.0, 0.95),
    "SAM": (-1.7, 3.0, 0.60),
    "Ours": (0.0, 1.6, 0.22),
}

EPOCHS = np.arange(1, 121)
DECAY_EP = 70.0          # LR decay -> visible step in every curve
NSEED = 5


def curves():
    """Return {dataset: {method: (train_runs, val_runs)}} with 5 seeds each."""
    rng = np.random.default_rng(20260812)
    t = EPOCHS.astype(float)
    step = 1.0 / (1.0 + np.exp(-(t - DECAY_EP) / 2.5))
    out = {}
    for name, top, v0, tau in DATASETS:
        out[name] = {}
        for m in METHODS:
            doff, gap, decay = SPEC[m]
            asym = top + doff - 1.6           # pre-decay plateau
            val_m = asym - (asym - v0) * np.exp(-t / tau) + 1.6 * step
            val_m = val_m - decay * np.clip((t - 85.0) / 35.0, 0, None)
            trn_m = np.minimum(val_m + gap * (1 - np.exp(-t / (0.8 * tau))),
                               98.8)
            tr, vl = [], []
            for _ in range(NSEED):
                off = rng.normal(0, 0.45)
                wob = np.cumsum(rng.normal(0, 0.22, t.size))
                wob = wob - wob * np.linspace(0, 1, t.size)   # anneal away
                jit = rng.normal(0, 0.55, t.size) * np.exp(-t / 60.0)
                vl.append(val_m + off + 0.5 * wob + jit)
                tr.append(trn_m + 0.6 * off + 0.3 * wob
                          + rng.normal(0, 0.3, t.size) * np.exp(-t / 70.0))
            out[name][m] = (np.array(tr), np.array(vl))
    return out


def main():
    set_pub_style(base_font=8)
    data = curves()

    # ---- global y range: sweep *every* band of *every* panel first ---------
    lo, hi = np.inf, -np.inf
    for name, _, _, _ in DATASETS:
        for m in METHODS:
            for runs in data[name][m]:
                mu, sd = runs.mean(0), runs.std(0)
                lo = min(lo, (mu - sd).min())
                hi = max(hi, (mu + sd).max())
    ylo = 10.0 * np.floor((lo - 1.5) / 10.0)
    yhi = min(103.0, np.ceil(hi + 1.5))
    yticks = np.arange(ylo + 10.0, 100.1, 10.0)
    print(f"shared y range: [{ylo:.0f}, {yhi:.0f}]  ticks {yticks}")

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 3, left=0.068, right=0.988, top=0.845,
                          bottom=0.095, hspace=0.30, wspace=0.10)
    axes = []
    for k, (name, _, _, _) in enumerate(DATASETS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        axes.append(ax)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=0.7, zorder=0)

        for m in METHODS:
            tr, vl = data[name][m]
            mu, sd = vl.mean(0), vl.std(0)
            hot = m == "Ours"
            ax.fill_between(EPOCHS, mu - sd, mu + sd, color=MCOL[m],
                            alpha=0.22 if hot else 0.13, lw=0,
                            zorder=3 if hot else 2)
            ax.plot(EPOCHS, tr.mean(0), color=MCOL[m], lw=0.8,
                    ls=(0, (3.2, 1.6)), alpha=0.72, zorder=4)
            ax.plot(EPOCHS, mu, color=MCOL[m], lw=MLW[m],
                    zorder=7 if hot else 5, solid_capstyle="round")

        # best epoch of the proposed method
        mu_ours = data[name]["Ours"][1].mean(0)
        be = int(EPOCHS[np.argmax(mu_ours)])
        ax.axvline(be, color=OURS, lw=0.8, ls=(0, (2.4, 1.8)), alpha=0.65,
                   zorder=6)
        ax.plot([be], [mu_ours.max()], marker="o", ms=3.6, mfc="white",
                mec=OURS, mew=1.2, zorder=8)

        ax.set_xlim(0, EPOCHS[-1])
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(yticks)
        ax.set_xticks([0, 30, 60, 90, 120])
        ax.set_title(name, fontsize=8.4, fontweight="bold", pad=3.5)
        gain = mu_ours.max() - data[name]["Baseline"][1].mean(0).max()
        box = ax.text(0.955, 0.06,
                      f"Ours {mu_ours.max():.1f}%  ($+${gain:.1f})",
                      transform=ax.transAxes, ha="right", va="bottom",
                      fontsize=6.9, color=OURS, fontweight="bold", zorder=9,
                      bbox=dict(boxstyle="round,pad=0.26", fc="white",
                                ec="#d5e0ea", lw=0.6, alpha=0.94))
        ax.text(0.955, 0.055, f"best epoch {be}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2, color="#666666", zorder=9)
        style_axes(ax)
        if k % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Top-1 accuracy (%)", fontsize=8)
        if k // 3:
            ax.set_xlabel("Epoch", fontsize=8)

    handles = [Line2D([], [], color=MCOL[m], lw=MLW[m], label=m)
               for m in METHODS]
    handles += [
        Line2D([], [], color="#4d4d4d", lw=1.3, label="validation"),
        Line2D([], [], color="#4d4d4d", lw=0.9, ls=(0, (3.2, 1.6)),
               label="train"),
        Patch(facecolor="#9fb6cb", alpha=0.5, lw=0,
              label=r"$\pm 1\sigma$ over 5 seeds"),
        Line2D([], [], color=OURS, lw=0.8, ls=(0, (2.4, 1.8)),
               label="best epoch (Ours)"),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_texts()[3].set_fontweight("bold")


    for name, _, _, _ in DATASETS:
        v = {m: data[name][m][1].mean(0).max() for m in METHODS}
        print(f"  {name:14s} " + "  ".join(f"{m} {v[m]:5.2f}" for m in METHODS))
    savefig(fig, "303_training_curves_grid.png")


if __name__ == "__main__":
    main()
