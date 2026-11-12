"""Figure 491 - Poisoning, gradient inversion, and what the defences cost.

Top left: a targeted poisoning campaign switched on mid-schedule, swept over
the fraction of malicious clients, so the collapse can be read against the
clean run rather than against an absolute scale.  Top right: what each robust
aggregator recovers under the heaviest attack, with the clean accuracy marked
as a rule on every bar.  Bottom left: the other threat - gradient inversion -
as abstract reconstructions that degrade once noise is added, stamped with
the reconstruction PSNR.  Bottom right: the price of that robustness, since a
defence nobody can afford to run is not a defence.

Panels (b) and (d) plot the same quantity and therefore share one accuracy
range; every quoted number is measured on the plotted arrays.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import shared_limits, shared_norm, image_axes, corner_stamp

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

FIG_W, FIG_H = 10.4, 6.4
CURVE = [0.062, 0.560, 0.500, 0.330]
BARS = [0.680, 0.560, 0.300, 0.330]
TILES = [0.062, 0.090, 0.500, 0.330]
COST = [0.680, 0.090, 0.300, 0.330]

ROUNDS = np.arange(0, 121)
ATTACK_ROUND = 40
CLEAN_TOP = 91.5
# malicious fraction -> accuracy the poisoned model settles at
STRENGTH = [(0.05, 78.0), (0.10, 63.5), (0.20, 41.0), (0.30, 21.5)]

# defence, accuracy under 20% attackers, clean accuracy, overhead %
DEFENCES = [("FedAvg (none)", 41.0, 91.5, 0.0, ACCENT),
            ("norm clipping", 68.2, 90.8, 3.0, BLUE),
            ("coordinate median", 77.5, 89.6, 8.0, BLUE),
            ("trimmed mean", 81.2, 90.1, 11.0, BLUE),
            ("Krum", 84.4, 88.4, 26.0, BLUE),
            ("clipping + DP (ours)", 88.1, 90.4, 14.0, GREEN)]

# column label, blur sigma, noise scale
INVERSION = [("original", None, None),
             ("no defence", 0.8, 0.030),
             (r"DP $\sigma$ = 0.6", 1.9, 0.145),
             (r"DP $\sigma$ = 1.2", 3.3, 0.300)]
TILE = 0.95 / FIG_W
TILE_H = 0.95 / FIG_H
TILE_GAP = 0.12 / FIG_W
TILE_VGAP = 0.09 / FIG_H
IMG_N = 40


def poisoned(floor, tau=9.0):
    """Clean warm-up, then decay towards the poisoned floor."""
    base = CLEAN_TOP * (1.0 - np.exp(-ROUNDS / 14.0))
    out = base.copy()
    late = ROUNDS >= ATTACK_ROUND
    out[late] = floor + (base[late] - floor) * np.exp(
        -(ROUNDS[late] - ATTACK_ROUND) / tau)
    return out


def blob_image(rng, n=IMG_N, k=7):
    """Abstract private sample: a few smooth blobs on a soft background."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    img = np.zeros((n, n))
    for _ in range(k):
        cx, cy = rng.uniform(0.15, 0.85, 2)
        s = rng.uniform(0.07, 0.20)
        img += rng.uniform(0.5, 1.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    img -= img.min()
    return img / max(img.max(), 1e-9)


def psnr(ref, test):
    mse = float(np.mean((ref - test) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main():
    set_pub_style()
    rng = np.random.default_rng(491)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---- poisoning collapse ----------------------------------------------
    ax = fig.add_axes(CURVE)
    clean = poisoned(CLEAN_TOP)
    ax.plot(ROUNDS, clean, color=BLUE, lw=1.7, zorder=6)
    ax.annotate("no attack", xy=(ROUNDS[-1], clean[-1]), xytext=(-3, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.6, color=BLUE, fontweight="bold")
    shades = [lighten(ACCENT, f) for f in (0.55, 0.38, 0.20, 0.0)]
    finals = {}
    for (frac, floor), col in zip(STRENGTH, shades):
        y = poisoned(floor)
        finals[frac] = y[-1]
        ax.plot(ROUNDS, y, color=col, lw=1.5 if frac == 0.30 else 1.2,
                zorder=5)
        ax.annotate(f"{frac:.0%}", xy=(ROUNDS[-1], y[-1]), xytext=(-3, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.4, color=col,
                    fontweight="bold" if frac == 0.30 else "normal")
    ax.axvline(ATTACK_ROUND, color=MUTED, lw=0.8, ls=(0, (3, 2.2)), zorder=2)
    ax.annotate(f"poisoning starts, round {ATTACK_ROUND}",
                xy=(ATTACK_ROUND, 12.0), xytext=(6, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=6.4, color=SUB)
    drop = clean[-1] - finals[0.30]
    ax.text(0.985, 0.06, f"{drop:.0f} pt collapse at 30% malicious clients",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color=ACCENT, fontweight="bold")
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("communication rounds", fontsize=8.0, labelpad=2.2)
    ax.set_ylabel("global accuracy (%)", fontsize=8.0, labelpad=2.5)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(True, axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Model poisoning, swept over the malicious fraction",
                 fontsize=8.5, fontweight="bold", pad=4.0)

    # ---- defence robustness ----------------------------------------------
    names = [d[0] for d in DEFENCES]
    under = np.array([d[1] for d in DEFENCES])
    cleanacc = np.array([d[2] for d in DEFENCES])
    over = np.array([d[3] for d in DEFENCES])
    cols = [d[4] for d in DEFENCES]
    acc_lo, acc_hi = shared_limits([under, cleanacc], pad=0.10)

    axb = fig.add_axes(BARS)
    y = np.arange(len(names))[::-1]
    axb.barh(y, under - acc_lo, left=acc_lo, height=0.62, color=cols,
             edgecolor="white", linewidth=0.6, zorder=3)
    for yy, u, c in zip(y, under, cleanacc):
        axb.plot([c, c], [yy - 0.34, yy + 0.34], lw=1.1, color="#555555",
                 zorder=5)
        axb.text(u - 1.2, yy, f"{u:.1f}", ha="right", va="center",
                 fontsize=6.3, color="white", fontweight="bold", zorder=6)
    axb.set_xlim(acc_lo, acc_hi)
    axb.set_yticks(y)
    axb.set_yticklabels(names, fontsize=6.8)
    axb.tick_params(labelsize=6.8, length=2.2)
    axb.set_xlabel("accuracy under 20% attackers (%)", fontsize=7.5,
                   labelpad=1.8)
    axb.grid(True, axis="x", lw=0.5, color=GRID)
    axb.set_axisbelow(True)
    style_axes(axb)
    axb.set_title("What each aggregator recovers", fontsize=8.5,
                  fontweight="bold", pad=4.0)
    axb.legend(handles=[Line2D([], [], color="#555555", lw=1.1,
                               label="clean accuracy")],
               loc="upper right", frameon=False, fontsize=6.4,
               handlelength=1.1, borderaxespad=0.3)

    # ---- gradient inversion ----------------------------------------------
    samples = [blob_image(rng) for _ in range(2)]
    stack, scores = [], []
    for img in samples:
        row = [img]
        for _, blur, noise in INVERSION[1:]:
            rec = gaussian_filter(img, blur) + rng.normal(0, noise, img.shape)
            row.append(rec)
        scores.append([psnr(img, r) for r in row[1:]])
        stack.append(row)
    norm = shared_norm([r for row in stack for r in row], robust=(1, 99))

    grid_w = len(INVERSION) * TILE + (len(INVERSION) - 1) * TILE_GAP
    x0 = TILES[0] + (TILES[2] - grid_w) / 2.0
    y_top = TILES[1] + 2 * TILE_H + TILE_VGAP
    for r, row in enumerate(stack):
        for c, img in enumerate(row):
            rect = [x0 + c * (TILE + TILE_GAP),
                    y_top - (r + 1) * TILE_H - r * TILE_VGAP, TILE, TILE_H]
            axi = fig.add_axes(rect)
            axi.imshow(img, cmap="gray", norm=norm, interpolation="nearest")
            image_axes(axi, color="#8a8a8a", lw=0.6)
            if c > 0:
                corner_stamp(axi, f"{scores[r][c - 1]:.1f} dB",
                             loc="lower left", fontsize=6.0)
            if r == 0:
                fig.text(rect[0] + TILE / 2.0, y_top + 0.010 / FIG_H,
                         INVERSION[c][0], ha="center", va="bottom",
                         fontsize=6.6, color=INK)
            if c == 0:
                fig.text(rect[0] - 0.07 / FIG_W, rect[1] + TILE_H / 2.0,
                         f"sample {r + 1}", ha="right", va="center",
                         fontsize=6.4, color=SUB, rotation=90)
    fig.text(TILES[0] + TILES[2] / 2.0, y_top + 0.20 / FIG_H,
             "Gradient inversion is blocked by noise, not by aggregation",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold",
             color=INK)
    fig.text(TILES[0] + TILES[2] / 2.0, TILES[1] - 0.10 / FIG_H,
             "stamps are reconstruction PSNR against the private sample",
             ha="center", va="top", fontsize=6.4, color=MUTED)

    # ---- overhead against benefit ----------------------------------------
    axd = fig.add_axes(COST)
    for (name, u, _, ov, col) in DEFENCES:
        lead = col == GREEN
        axd.scatter(ov, u, s=62 if lead else 40, facecolor=col,
                    alpha=0.95 if lead else 0.72,
                    edgecolor="white" if lead else col, linewidth=1.0,
                    zorder=6 if lead else 4)
    offs = {"FedAvg (none)": (6, 4), "norm clipping": (6, -2),
            "coordinate median": (6, -3), "trimmed mean": (6, -3),
            "Krum": (-6, -4), "clipping + DP (ours)": (7, 6)}
    for (name, u, _, ov, col) in DEFENCES:
        dx, dy = offs[name]
        axd.annotate(name, xy=(ov, u), xytext=(dx, dy),
                     textcoords="offset points", fontsize=6.3,
                     ha="left" if dx > 0 else "right", va="center",
                     color=col if col != BLUE else SUB,
                     fontweight="bold" if col == GREEN else "normal")
    axd.annotate("", xy=(0.055, 0.945), xytext=(0.155, 0.865),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    axd.text(0.065, 0.855, "better", transform=axd.transAxes, fontsize=7.0,
             fontweight="bold", color=INK, ha="left", va="top")
    axd.set_xlim(-3.0, 34.0)
    axd.set_ylim(acc_lo, acc_hi)
    axd.set_xlabel("added round time (%)", fontsize=7.5, labelpad=1.8)
    axd.set_ylabel("accuracy under attack (%)", fontsize=7.5, labelpad=2.0)
    axd.tick_params(labelsize=6.8, length=2.2)
    axd.grid(True, lw=0.5, color=GRID)
    axd.set_axisbelow(True)
    style_axes(axd)
    axd.set_title("Robustness is not free", fontsize=8.5, fontweight="bold",
                  pad=4.0)
    best = DEFENCES[-1]
    axd.text(0.985, 0.05,
             f"+{best[1] - DEFENCES[0][1]:.0f} pt for {best[3]:.0f}% "
             "extra round time", transform=axd.transAxes, ha="right",
             va="bottom", fontsize=6.4, color=GREEN, fontweight="bold")

    for tag, x, ytop in (("(a)", CURVE[0], CURVE[1] + CURVE[3]),
                         ("(b)", BARS[0], BARS[1] + BARS[3]),
                         ("(c)", TILES[0], y_top + 0.20 / FIG_H),
                         ("(d)", COST[0], COST[1] + COST[3])):
        fig.text(x - 0.40 / FIG_W, ytop + 0.07 / FIG_H, tag, fontsize=11,
                 fontweight="bold", ha="left", va="bottom")

    savefig(fig, "491_fed_attack_defense.png")


if __name__ == "__main__":
    main()
