"""Multi-scenario ensemble projection of global mean warming.

Left: the observed record to 2014 followed by three emission scenarios, each
drawn as an ensemble mean with 25-75 % and 5-95 % quantile fans. Warming
thresholds are marked with horizontal guides and the crossing years are
tabulated. Right: violins of the 2081-2100 mean warming in every ensemble
member, sharing the temperature axis with the main panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from pubstyle import set_pub_style, savefig, style_axes

HIST0, SPLIT, END = 1950, 2014, 2100
NMEM = 40
THRESH = [1.5, 2.0, 3.0, 4.0]

SCEN = [
    dict(key="SSP1-2.6", color="#2f6ea8", amp=0.98, shape="saturating"),
    dict(key="SSP2-4.5", color="#d8973c", amp=1.90, shape="linear"),
    dict(key="SSP5-8.5", color="#a8322a", amp=3.60, shape="accelerating"),
]


def hist_forced(t):
    """Observed-like forced warming relative to 1850-1900."""
    return (0.22 + 0.30 * np.tanh((t - 1945) / 16.0)
            + 0.62 * np.tanh((t - 2000) / 22.0))


def ramp(x, shape):
    if shape == "saturating":
        return (1 - np.exp(-3.1 * x)) / (1 - np.exp(-3.1))
    if shape == "linear":
        return x ** 0.88
    return x ** 1.30


def internal(rng, n, nmem, sigma=0.115):
    """Smoothed AR(1)-like unforced variability for each member."""
    v = gaussian_filter1d(rng.normal(size=(nmem, n)), 2.2, axis=1)
    return sigma * v / v.std()


def build(rng):
    years = np.arange(HIST0, END + 1)
    hist = hist_forced(years)
    anchor = hist_forced(SPLIT)
    out = {}
    for sc in SCEN:
        gain = rng.normal(1.0, 0.21, NMEM)[:, None]
        x = np.clip((years - SPLIT) / (END - SPLIT), 0, 1)[None, :]
        forced = anchor + gain * sc["amp"] * ramp(x, sc["shape"])
        forced = np.where(years[None, :] <= SPLIT, hist[None, :], forced)
        out[sc["key"]] = forced + internal(rng, years.size, NMEM)
    return years, hist, out


def crossing(years, mean, level):
    idx = np.where(mean >= level)[0]
    return int(years[idx[0]]) if idx.size else None


def main():
    set_pub_style()
    rng = np.random.default_rng(85)
    years, hist, ens = build(rng)
    obs_mask = years <= SPLIT
    obs = hist[obs_mask] + internal(rng, years.size, 1)[0][obs_mask]

    fig = plt.figure(figsize=(10.6, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.0], left=0.062,
                          right=0.985, top=0.865, bottom=0.185, wspace=0.045)
    ax = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1], sharey=ax)

    ax.axvspan(HIST0, SPLIT, color="#f0f0f0", lw=0, zorder=0)
    for lev in THRESH:
        ax.axhline(lev, color="#9a9a9a", lw=0.7, ls=(0, (1.6, 2)), zorder=1)
        ax.text(1952, lev + 0.06, f"{lev:.1f} $\\degree$C", fontsize=6.8,
                color="#777777", va="bottom", ha="left", zorder=6)
    ax.axhline(0, color="#666666", lw=0.8, zorder=1)

    # historical ensemble spread (identical forcing across scenarios)
    hband = ens["SSP2-4.5"][:, obs_mask]
    ax.fill_between(years[obs_mask], *np.percentile(hband, [5, 95], axis=0),
                    color="#8a8a8a", alpha=0.22, lw=0, zorder=2)
    ax.plot(years[obs_mask], hband.mean(0), color="#4a4a4a", lw=2.1, zorder=5,
            label="Historical ensemble")
    ax.plot(years[obs_mask], obs, color="#111111", lw=1.1, zorder=6,
            label="Observations")

    proj = years >= SPLIT
    rows = []
    for sc in SCEN:
        e = ens[sc["key"]][:, proj]
        yp = years[proj]
        lo5, lo25, med, hi75, hi95 = np.percentile(e, [5, 25, 50, 75, 95],
                                                   axis=0)
        ax.fill_between(yp, lo5, hi95, color=sc["color"], alpha=0.13, lw=0,
                        zorder=3)
        ax.fill_between(yp, lo25, hi75, color=sc["color"], alpha=0.26, lw=0,
                        zorder=3.5)
        mean = e.mean(0)
        ax.plot(yp, mean, color=sc["color"], lw=2.1, zorder=5,
                label=sc["key"], solid_capstyle="round")
        ax.text(END + 0.8, mean[-1], f"{mean[-1]:.1f}", color=sc["color"],
                fontsize=7.6, fontweight="bold", va="center", ha="left")
        rows.append((sc["key"], crossing(yp, mean, 1.5),
                     crossing(yp, mean, 2.0)))

    # crossing-year table in the empty lower-right corner
    lines = ["Year the ensemble mean crosses"]
    for k, y15, y20 in rows:
        y20txt = f"2.0 $\\degree$C in {y20}" if y20 else "2.0 $\\degree$C not reached"
        lines.append(f"{k}:   1.5 $\\degree$C in {y15 or 'n/a'},   {y20txt}")
    ax.text(0.445, 0.035, "\n".join(lines), transform=ax.transAxes,
            fontsize=6.9, color="#333333", va="bottom", ha="left",
            linespacing=1.55, zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dcdcdc",
                      lw=0.6, alpha=0.92))

    ax.axvline(SPLIT, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=4)
    ax.text(SPLIT - 1.5, 6.15, "historical", fontsize=7.4, color="#555555",
            ha="right", va="top", style="italic")
    ax.text(SPLIT + 1.5, 6.15, "projection", fontsize=7.4, color="#555555",
            ha="left", va="top", style="italic")

    ax.set_xlim(HIST0, END + 5)
    ax.set_ylim(-0.6, 6.2)
    ax.set_xticks(np.arange(1950, 2101, 25))
    ax.set_xlabel("Year")
    ax.set_ylabel("Global mean surface temperature anomaly\n"
                  "relative to 1850-1900 ($\\degree$C)", linespacing=1.4)
    style_axes(ax)
    ax.text(-0.055, 1.025, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")

    # ---- 2081-2100 distributions -----------------------------------------
    late = (years >= 2081) & (years <= END)
    dist = [ens[sc["key"]][:, late].mean(1) for sc in SCEN]
    pos = np.arange(1, len(SCEN) + 1)
    parts = ax_v.violinplot(dist, positions=pos, widths=0.78,
                            showextrema=False, showmedians=False)
    for body, sc in zip(parts["bodies"], SCEN):
        body.set_facecolor(sc["color"])
        body.set_alpha(0.30)
        body.set_edgecolor(sc["color"])
        body.set_linewidth(0.9)
    for x, d, sc in zip(pos, dist, SCEN):
        q5, q25, q50, q75, q95 = np.percentile(d, [5, 25, 50, 75, 95])
        ax_v.plot([x, x], [q5, q95], color=sc["color"], lw=1.0, zorder=4)
        ax_v.plot([x, x], [q25, q75], color=sc["color"], lw=5.0,
                  solid_capstyle="butt", zorder=5)
        ax_v.plot([x], [q50], marker="o", ms=4.2, mfc="white",
                  mec=sc["color"], mew=1.2, zorder=6)
        ax_v.text(x, q95 + 0.14, f"{q50:.1f}", ha="center", va="bottom",
                  fontsize=7.2, fontweight="bold", color=sc["color"])
    for lev in THRESH:
        ax_v.axhline(lev, color="#9a9a9a", lw=0.7, ls=(0, (1.6, 2)), zorder=1)
    ax_v.axhline(0, color="#666666", lw=0.8, zorder=1)
    ax_v.set_xlim(0.45, len(SCEN) + 0.55)
    ax_v.set_xticks(pos)
    ax_v.set_xticklabels([sc["key"] for sc in SCEN], fontsize=7.2,
                         rotation=30, ha="right")
    ax_v.tick_params(labelleft=False, length=2.5)
    ax_v.set_xlabel("2081-2100 mean", labelpad=2)
    style_axes(ax_v)
    ax_v.text(0.02, 1.025, "(b)", transform=ax_v.transAxes, fontsize=11,
              fontweight="bold", va="bottom")

    handles = [plt.Line2D([], [], color="#111111", lw=1.1,
                          label="Observations"),
               plt.Line2D([], [], color="#4a4a4a", lw=2.1,
                          label="Historical ensemble")]
    handles += [plt.Line2D([], [], color=sc["color"], lw=2.1, label=sc["key"])
                for sc in SCEN]
    handles += [Patch(facecolor="#666666", alpha=0.26, label="25-75 %"),
                Patch(facecolor="#666666", alpha=0.13, label="5-95 %")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.50, 0.055),
               ncol=7, frameon=False, fontsize=7.8, handlelength=1.7,
               columnspacing=1.4, handletextpad=0.55)

    savefig(fig, "85_ensemble_projection.png")


if __name__ == "__main__":
    main()
