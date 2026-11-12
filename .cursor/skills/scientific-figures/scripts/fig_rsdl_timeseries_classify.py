"""Figure 338 - Satellite image time-series classification.

Left: per-class NDVI trajectory families (translucent samples plus the class
mean).  Right top: a subordinate bar panel comparing three sequence models.
Right bottom: the temporal attention the transformer places on each class,
which lines up with the phenological stage that separates it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label, \
    style_axes

CLASSES = ["Maize", "Soybean", "Winter wheat", "Rice", "Grassland",
           "Deciduous forest"]
CLASS_COLORS = ["#c9a227", "#8fbc4a", "#d98c5f", "#3b7ea1", "#c6bf55",
                "#1d6b3f"]
# darkened twins so the pale fills stay legible as text on white
CLASS_TEXT = ["#a5851f", "#5f8a2c", "#b56a3c", "#2f6683", "#96882a",
              "#1d6b3f"]
# peak DOY, peak NDVI, season width, green-up DOY, sample spread
PROFILE = [(212, 0.86, 40, 152, 0.055), (232, 0.80, 36, 174, 0.050),
           (142, 0.77, 32, 80, 0.060), (240, 0.73, 30, 180, 0.048),
           (188, 0.60, 62, 118, 0.070), (200, 0.83, 74, 122, 0.040)]
N_STEPS = 24
DOY = np.linspace(10, 355, N_STEPS)


def trajectories(rng, c, m=26):
    pk, amp, wd, gu, sp = PROFILE[c]
    out = np.zeros((m, N_STEPS))
    for i in range(m):
        dp = pk + rng.normal(0, 9)
        da = amp * (1 + rng.normal(0, 0.075))
        dw = wd * (1 + rng.normal(0, 0.12))
        base = 0.15 + 0.025 * np.sin(DOY / 55.0) + rng.normal(0, 0.012)
        y = base + da * np.exp(-0.5 * ((DOY - dp) / dw) ** 2)
        y += 0.13 * da * np.exp(-0.5 * ((DOY - gu - rng.normal(0, 8))
                                        / 22.0) ** 2)
        y += rng.normal(0, sp * 0.42, N_STEPS)
        out[i] = np.clip(y, 0.04, 0.98)
    return out


def attention_matrix(rng):
    a = np.zeros((6, N_STEPS))
    for c, (pk, _amp, wd, gu, _sp) in enumerate(PROFILE):
        a[c] = (np.exp(-0.5 * ((DOY - pk) / (wd * 0.75)) ** 2)
                + 0.62 * np.exp(-0.5 * ((DOY - gu) / 26.0) ** 2)
                + 0.30 * np.exp(-0.5 * ((DOY - pk - 62) / 34.0) ** 2))
        a[c] *= 1 + 0.10 * rng.standard_normal(N_STEPS)
    return np.clip(a, 0, None) / np.clip(a, 0, None).sum(1, keepdims=True)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(29)

    fams = [trajectories(rng, c) for c in range(6)]
    attn = attention_matrix(np.random.default_rng(4))

    models = ["1D-CNN", "Bi-LSTM", "Transformer"]
    oa = np.array([0.842, 0.871, 0.906])
    f1 = np.array([0.821, 0.858, 0.893])
    bar_c = ["#9aa7b1", "#7d8f9e", "#b23a48"]

    fig = plt.figure(figsize=(10.0, 5.6))
    gs_l = fig.add_gridspec(2, 3, left=0.048, right=0.505, top=0.855,
                            bottom=0.105, wspace=0.10, hspace=0.28)
    gs_r = fig.add_gridspec(2, 1, left=0.590, right=0.960, top=0.855,
                            bottom=0.105, height_ratios=[0.60, 1.00],
                            hspace=0.42)

    # ---- (a) trajectory families -----------------------------------------
    lo, hi = 0.02, 1.02
    for c in range(6):
        ax = fig.add_subplot(gs_l[c // 3, c % 3])
        col = CLASS_COLORS[c]
        for y in fams[c]:
            ax.plot(DOY, y, color=col, lw=0.45, alpha=0.28, zorder=2)
        mu = fams[c].mean(axis=0)
        sd = fams[c].std(axis=0)
        ax.fill_between(DOY, mu - sd, mu + sd, color=col, alpha=0.22,
                        linewidth=0, zorder=3)
        ax.plot(DOY, mu, color=col, lw=1.9, zorder=4,
                path_effects=None)
        ax.plot(DOY, mu, color="white", lw=0.5, zorder=5, alpha=0.55)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, 365)
        ax.set_xticks([60, 180, 300])
        ax.set_xticklabels(["Mar", "Jul", "Nov"], fontsize=6.4)
        ax.set_yticks([0.2, 0.5, 0.8])
        ax.tick_params(labelsize=6.4, length=2, pad=1.5)
        ax.grid(linewidth=0.35, alpha=0.55)
        style_axes(ax)
        if c % 3 != 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("NDVI (-)", fontsize=7.0, labelpad=2)
        if c // 3 == 1:
            ax.set_xlabel("day of year (-)", fontsize=7.0, labelpad=1.5)
        ax.set_title(CLASSES[c], fontsize=7.4, fontweight="bold", pad=2.6,
                     color=CLASS_TEXT[c])
        ax.text(0.965, 0.94, f"n = {fams[c].shape[0]}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
                color="#777777")
        if c == 0:
            add_panel_label(ax, "(a)", x=-0.30, y=1.10, size=11)

    # ---- (b) subordinate model comparison ---------------------------------
    ax_b = fig.add_subplot(gs_r[0, 0])
    x = np.arange(3)
    ax_b.bar(x - 0.185, oa, 0.34, color=bar_c, edgecolor="#333333",
             linewidth=0.5, label="Overall accuracy")
    ax_b.bar(x + 0.185, f1, 0.34, color=bar_c, alpha=0.55, hatch="///",
             edgecolor="#333333", linewidth=0.5, label="Macro F1")
    for xi, a, b in zip(x, oa, f1):
        ax_b.text(xi - 0.185, a + 0.008, f"{a:.3f}", ha="center",
                  va="bottom", fontsize=6.2, color="#333333")
        ax_b.text(xi + 0.185, b + 0.008, f"{b:.3f}", ha="center",
                  va="bottom", fontsize=6.2, color="#666666")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(models, fontsize=7.0)
    ax_b.set_ylim(0.70, 0.985)
    ax_b.set_ylabel("score (-)", fontsize=7.4, labelpad=2)
    ax_b.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_b.grid(axis="y", linewidth=0.4, alpha=0.6)
    ax_b.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left",
                handlelength=1.2, columnspacing=0.9, borderaxespad=0.15)
    style_axes(ax_b)
    ax_b.set_title("Sequence model comparison, 6 classes", fontsize=7.8,
                   fontweight="bold", pad=3.0)
    add_panel_label(ax_b, "(b)", x=-0.115, y=1.06, size=11)

    # ---- (c) temporal attention ------------------------------------------
    ax_c = fig.add_subplot(gs_r[1, 0])
    cmap = LinearSegmentedColormap.from_list(
        "att", PALETTES["cb_sequential_blue"])
    vmax = float(attn.max())
    im = ax_c.imshow(attn, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto",
                     interpolation="nearest")
    for c in range(6):
        t = int(np.argmax(attn[c]))
        ax_c.add_patch(Rectangle((t - 0.5, c - 0.5), 1, 1, fill=False,
                                 edgecolor="#b23a48", linewidth=1.1,
                                 zorder=5))
    ax_c.set_yticks(range(6))
    ax_c.set_yticklabels(CLASSES, fontsize=6.8)
    for t, col in zip(ax_c.get_yticklabels(), CLASS_TEXT):
        t.set_color(col)
    step = 4
    ax_c.set_xticks(range(0, N_STEPS, step))
    ax_c.set_xticklabels([f"{int(d)}" for d in DOY[::step]], fontsize=6.4)
    ax_c.set_xlabel("acquisition day of year (-)", fontsize=7.4, labelpad=1.5)
    ax_c.tick_params(length=2, pad=1.5)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
    ax_c.set_title("Temporal attention weight per class", fontsize=7.8,
                   fontweight="bold", pad=3.0)
    add_panel_label(ax_c, "(c)", x=-0.115, y=1.05, size=11)

    cax = ax_c.inset_axes([1.028, 0.0, 0.026, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.set_yticklabels(["0", f"{vmax / 2:.2f}", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, pad=1.5)
    cb.set_label("attention weight (-)", fontsize=6.6, color="#1a1a1a",
                 labelpad=3)


    savefig(fig, "338_rsdl_timeseries_classify.png")


if __name__ == "__main__":
    main()
