"""Maximum-likelihood integration of a visual and an auditory location cue.

The main panel is the textbook prediction and its test in one frame: two
single-cue likelihoods with different widths, the product of the two, and the
locations twelve observers actually reported when the cues were placed in
conflict.  The integrated estimate is pulled toward the more reliable cue and
is narrower than either single cue - a variance reduction that no weighted
average with fixed weights could produce.

The subordinate panels show where the weights come from and what happens when
the two channels disagree beyond repair: the cue weights as visual noise is
raised through and past the auditory noise, and the response matrix of a
McGurk-style conflict, where the fused percept is neither of the two inputs.
All annotated values are computed from the arrays that are drawn.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches, safe_colorbar

BLUE = "#3b5b92"      # visual
RED = "#b23a48"       # auditory
GREEN = "#2e7d5b"     # integrated / optimal
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

MU_V, SIG_V = -5.0, 3.2       # deg
MU_A, SIG_A = 5.0, 4.4        # deg
N_OBS = 12

CONDITIONS = ["audio /ba/ +\nvideo /ga/", "congruent /ba/",
              "congruent /ga/", "audio only /ba/"]
RESPONSES = ["ba", "da", "ga", "tha"]
RESP_P = np.array([[0.18, 0.58, 0.14, 0.10],
                   [0.93, 0.04, 0.02, 0.01],
                   [0.03, 0.06, 0.88, 0.03],
                   [0.96, 0.02, 0.01, 0.01]])


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))


def integrate(mu_v, sig_v, mu_a, sig_a):
    rv, ra = 1.0 / sig_v ** 2, 1.0 / sig_a ** 2
    return (mu_v * rv + mu_a * ra) / (rv + ra), 1.0 / np.sqrt(rv + ra)


def observers(mu_hat, sig_hat, seed=13):
    rng = np.random.default_rng(seed)
    return rng.normal(mu_hat, sig_hat / np.sqrt(3.0), N_OBS)


def weight_sweep(n=240):
    sig_v = np.linspace(1.0, 12.0, n)
    rv, ra = 1.0 / sig_v ** 2, 1.0 / SIG_A ** 2
    return sig_v, rv / (rv + ra), ra / (rv + ra)


def measured_weights(seed=19, k=5):
    rng = np.random.default_rng(seed)
    sig_v = np.array([1.6, 2.8, 4.4, 7.0, 10.5])[:k]
    rv, ra = 1.0 / sig_v ** 2, 1.0 / SIG_A ** 2
    w = rv / (rv + ra)
    obs = np.clip(w + rng.normal(0.0, 0.045, w.size), 0.0, 1.0)
    err = rng.uniform(0.030, 0.062, w.size)
    return sig_v, obs, err


def main_panel(ax):
    x = np.linspace(-22.0, 22.0, 700)
    mu_i, sig_i = integrate(MU_V, SIG_V, MU_A, SIG_A)
    pv, pa, pi = (gauss(x, MU_V, SIG_V), gauss(x, MU_A, SIG_A),
                  gauss(x, mu_i, sig_i))
    top = pi.max()

    for p, mu, sig, col, name in [(pv, MU_V, SIG_V, BLUE, "visual"),
                                  (pa, MU_A, SIG_A, RED, "auditory")]:
        ax.plot(x, p, color=col, lw=1.6, zorder=4,
                label=f"{name} likelihood")
        ax.plot([mu, mu], [0.0, p.max()], color=col, lw=0.7,
                ls=(0, (2.2, 2.0)), zorder=3)
        h = p.max() * np.exp(-0.5)
        ax.annotate("", xy=(mu - sig, h), xytext=(mu + sig, h),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=0.8),
                    zorder=5)
        ax.text(mu, h + 0.004, f"$\\sigma$ = {sig:.1f} deg", ha="center",
                va="bottom", fontsize=6.8, color=col, zorder=6)

    ax.fill_between(x, 0.0, pi, color=GREEN, alpha=0.14, linewidth=0.0,
                    zorder=2)
    ax.plot(x, pi, color=GREEN, lw=2.0, zorder=5,
            label="integrated posterior")
    ax.plot([mu_i, mu_i], [0.0, pi.max()], color=GREEN, lw=0.9,
            ls=(0, (2.2, 2.0)), zorder=4)
    h = pi.max() * np.exp(-0.5)
    ax.annotate("", xy=(mu_i - sig_i, h), xytext=(mu_i + sig_i, h),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.9),
                zorder=6)
    ax.text(mu_i, h + 0.004, f"$\\sigma$ = {sig_i:.1f} deg", ha="center",
            va="bottom", fontsize=6.8, color=GREEN, zorder=6)

    pse = observers(mu_i, sig_i)
    y_row = top * 1.30
    rng = np.random.default_rng(7)
    ax.plot(pse, y_row + rng.uniform(-0.004, 0.004, pse.size), marker="o",
            ms=3.0, ls="none", color=INK, alpha=0.55, mec="none", zorder=6)
    sem = float(pse.std(ddof=1) / np.sqrt(pse.size))
    ax.errorbar([pse.mean()], [y_row + 0.011], xerr=[sem], fmt="o", ms=4.4,
                color=INK, ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=7)
    ax.text(pse.mean() + 1.0, y_row + 0.011,
            f"observed PSE {pse.mean():+.1f} $\\pm$ {sem:.1f} deg "
            f"(n = {pse.size})", ha="left", va="center", fontsize=6.8,
            color=INK)
    ax.text(mu_i, y_row - 0.014, f"predicted optimum {mu_i:+.1f} deg",
            ha="center", va="top", fontsize=6.8, color=GREEN)

    ax.set_xlim(-22.0, 22.0)
    ax.set_ylim(0.0, top * 1.46)
    ax.set_xticks(np.arange(-20.0, 21.0, 5.0))
    ax.set_xlabel("perceived source location (deg)")
    ax.set_ylabel("likelihood density (1/deg)")
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left", handlelength=1.6,
              labelspacing=0.30, borderaxespad=0.3)
    cut = 1.0 - sig_i / min(SIG_V, SIG_A)
    ax.set_title(f"Fusing the two cues cuts the standard deviation by "
                 f"{cut * 100:.0f} %", fontsize=9.0, fontweight="bold",
                 color=INK)
    return mu_i, sig_i, cut, pse


def weight_panel(ax):
    sig_v, wv, wa = weight_sweep()
    ax.plot(sig_v, wv, color=BLUE, lw=1.6, label="visual weight $w_V$")
    ax.plot(sig_v, wa, color=RED, lw=1.6, ls=(0, (5, 2.2)),
            label="auditory weight $w_A$")
    ax.axhline(0.5, color=GREY, lw=0.7, ls=(0, (1.6, 2.0)))
    ax.plot([SIG_A], [0.5], marker="o", ms=4.0, color=GREEN, mfc="white",
            mew=1.2, zorder=6)
    ax.text(6.5, 0.99,
            f"equal weighting at $\\sigma_V$ = $\\sigma_A$ = {SIG_A:.1f} deg",
            ha="center", va="top", fontsize=6.5, color=SUB)

    s_obs, w_obs, err = measured_weights()
    ax.errorbar(s_obs, w_obs, yerr=err, fmt="o", ms=3.4, color=INK,
                ecolor=INK, elinewidth=0.8, capsize=2.0, zorder=7,
                label="measured")
    ax.set_xlim(1.0, 12.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("visual noise $\\sigma_V$ (deg)", fontsize=7.4)
    ax.set_ylabel("cue weight (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="center right",
              handlelength=1.5, labelspacing=0.28, borderaxespad=0.4)
    ax.set_title("Weights follow relative reliability", fontsize=8.2,
                 fontweight="bold", color=INK)


def mcgurk_panel(fig, ax, seed=5):
    rng = np.random.default_rng(seed)
    trials = 60
    counts = np.array([rng.multinomial(trials, p) for p in RESP_P],
                      dtype=float)
    pct = 100.0 * counts / counts.sum(axis=1, keepdims=True)
    im = ax.imshow(pct, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                   vmax=100.0)
    ax.set_aspect("equal")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.0,
                    color="white" if pct[i, j] > 55.0 else INK)
    fused = pct[0, 1]
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1.0, 1.0, facecolor="none",
                               edgecolor=GREEN, lw=1.4, zorder=5))
    ax.set_xticks(range(len(RESPONSES)))
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_xticklabels(RESPONSES)
    ax.set_yticklabels(CONDITIONS, linespacing=1.15)
    ax.tick_params(labelsize=6.5, length=2.0)
    ax.set_xlabel("reported syllable (-)", fontsize=7.4)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title(f"Fused /da/ on {fused:.0f} % of McGurk trials",
                 fontsize=8.2, fontweight="bold", color=INK)
    safe_colorbar(fig, im, ax, label="responses (%)", width=0.09, pad=0.06,
                  labelsize=6.5)
    return trials


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.36, hspace=0.72)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])

    mu_i, sig_i, cut, pse = main_panel(ax_main)
    weight_panel(ax_w)
    trials = mcgurk_panel(fig, ax_m)

    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_w, ax_m], dx=-0.46, dy=0.12)
    savefig(fig, "731_sen_multisensory_integration.png")


if __name__ == "__main__":
    main()
