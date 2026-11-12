"""2x3 stacked area proportion charts: composition over time.

Each panel shows how 3-4 shares of a 100% composition evolve from 1990 to
2020. Shares follow smooth logistic transitions between a documented start and
end state with mild correlated noise on top, which is what real composition
series look like; per-panel legends sit under the axes so they never cover the
data, and the net change of the fastest-moving class is called out.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label)

# (title, [(category, colour, share in 1990, share in 2020), ...])
PANELS = [
    ("Land cover", [
        ("Forest", "#2e7d5b", 38, 31), ("Cropland", "#c9a227", 34, 27),
        ("Grassland", "#8fc4a3", 20, 16), ("Built-up", "#8c2f2f", 8, 26)]),
    ("Energy mix", [
        ("Coal", "#6b6b6b", 72, 37), ("Gas", "#c0603f", 18, 25),
        ("Wind", "#2e7d5b", 6, 19), ("Solar", "#e0b13f", 4, 19)]),
    ("Crop types", [
        ("Rice", "#2e7d5b", 40, 30), ("Wheat", "#e0955a", 27, 25),
        ("Maize", "#8c2f2f", 22, 35), ("Other", "#bfbfbf", 11, 10)]),
    ("Water sources", [
        ("Surface", "#3b5b92", 55, 43), ("Groundwater", "#2a8f86", 30, 26),
        ("Rainwater", "#8fc0d9", 10, 17), ("Reclaimed", "#c9a227", 5, 14)]),
    ("Emission sectors", [
        ("Industry", "#8c2f2f", 42, 32), ("Transport", "#e0955a", 18, 28),
        ("Agriculture", "#2e7d5b", 15, 13), ("Power", "#6b6b6b", 25, 27)]),
    ("Age structure", [
        ("0-14 yr", "#8fc4a3", 27, 16), ("15-59 yr", "#3b5b92", 63, 60),
        ("60+ yr", "#c0603f", 10, 24)]),
]


def smooth_noise(rng, n, sd, width=7):
    w = np.exp(-0.5 * (np.arange(-width, width + 1) / (width / 2.0)) ** 2)
    w /= w.sum()
    return np.convolve(rng.normal(0, sd, n + 2 * width), w, "same")[width:width + n]


def panel(ax, rng, idx, spec):
    title, rows = spec
    years = np.arange(1990, 2021)
    n = len(years)
    # logistic transition from the 1990 state to the 2020 state
    s = 1 / (1 + np.exp(-6.0 * (np.linspace(0, 1, n) - 0.45)))
    s = (s - s[0]) / (s[-1] - s[0])
    raw = np.array([a + (b - a) * s + smooth_noise(rng, n, 0.9)
                    for _, _, a, b in rows])
    raw = np.clip(raw, 0.4, None)
    shares = raw / raw.sum(axis=0) * 100

    cum = np.zeros(n)
    for (cat, col, _, _), sh in zip(rows, shares):
        ax.fill_between(years, cum, cum + sh, color=col, alpha=0.9,
                        edgecolor="white", linewidth=0.7, label=cat, zorder=2)
        cum += sh

    delta = shares[:, -3:].mean(1) - shares[:, :3].mean(1)
    j = int(np.argmax(np.abs(delta)))
    ax.text(0.5, 1.055, f"{rows[j][0]}: {delta[j]:+.0f} pp since 1990",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2,
            color="#555555")

    ax.set_ylim(0, 100)
    ax.set_xlim(years[0], years[-1])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Share of total (%)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=16)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=len(rows), frameon=False, handlelength=1.1,
              handletextpad=0.45, columnspacing=1.0)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.17, y=1.06, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(331)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for k, (ax, spec) in enumerate(zip(axes.ravel(), PANELS)):
        panel(ax, rng, k, spec)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955], h_pad=4.2, w_pad=2.6)
    savefig(fig, "37_stacked_area_grid.png")


if __name__ == "__main__":
    main()
