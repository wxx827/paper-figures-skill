"""Day-ahead price surface: hour of day against week of year.

(a) the 24 x 52 mean-price surface, (b) the diurnal price profile sharing the
hour axis, (c) monthly means sharing the week axis, (d) the distribution of
the 1 248 cells.  All four read off one diverging colour scale that is
centred on the annual mean price, so blue is "cheaper than average", orange
is "more expensive than average", and the deep-blue block in spring is the
negative-price window created by midday solar.

Layout note: (b) and (c) are marginals, so they get ~21 % of the width and
~36 % of the height of the main surface; (d) is tertiary and smaller again.
Every colour-mapped element uses the same norm and the same colormap.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, PALETTES, savefig, style_axes

INK = "#333333"
GREY = "#8d959c"
NWEEK = 52


def circ_gauss(t, mu, sigma):
    d = ((t - mu + 12.0) % 24.0) - 12.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def annual_prices(seed=23):
    """8760 hourly day-ahead prices in $/MWh, driven by a net-load stack."""
    n = 365 * 24
    hod = np.arange(n) % 24
    doy = np.arange(n) // 24
    rng = np.random.default_rng(seed)

    shape_w = 0.30 + 0.55 * circ_gauss(hod, 8.5, 2.2) \
        + 0.95 * circ_gauss(hod, 19.0, 2.6)
    shape_s = 0.40 + 0.30 * circ_gauss(hod, 9.0, 2.5) \
        + 1.00 * circ_gauss(hod, 16.5, 3.4)
    mix = 0.5 * (1.0 + np.cos(2 * np.pi * (doy - 15) / 365.0))
    amp = 1.0 - 0.22 * np.sin(2 * np.pi * (doy - 15) / 365.0) ** 2
    wf = np.where(doy % 7 >= 5, 0.89, 1.0)
    load = 3000.0 + 3200.0 * amp * (mix * shape_w + (1 - mix) * shape_s) * wf

    # solar: longer, stronger days in summer; clouds modulate the peak
    daylen = 10.6 + 3.4 * np.cos(2 * np.pi * (doy - 172) / 365.0)
    clear = np.clip(np.sin(np.pi * (hod - (12 - daylen / 2)) / daylen),
                    0.0, None) ** 1.2
    cloud = np.clip(1.0 - 0.30 * np.abs(
        gaussian_filter1d(rng.normal(0, 1, n), 9.0)), 0.35, 1.0)
    sf = clear * cloud
    solar = 2400.0 * sf
    wind = 1700.0 * np.clip(
        0.45 + 0.36 * gaussian_filter1d(rng.normal(0, 1, n), 26.0), 0.03, 1.0)

    net = load - solar - wind
    z = (net - net.min()) / np.ptp(net)
    price = 9.0 + 108.0 * z ** 2.6
    # shoulder seasons with a full solar deck push the market below zero
    shoulder = np.exp(-0.5 * ((doy - 108) / 34.0) ** 2) \
        + 0.85 * np.exp(-0.5 * ((doy - 288) / 30.0) ** 2)
    price -= 34.0 * np.clip(sf - 0.45, 0.0, None) / 0.55 * np.clip(
        shoulder, 0.0, 1.0)
    price += 4.5 * gaussian_filter1d(rng.normal(0, 1, n), 2.0)

    # scarcity events on the tightest hours of the year
    tight = np.argsort(net)[-18:]
    price[tight] *= rng.uniform(1.8, 3.2, tight.size)
    return price


def main():
    set_pub_style(base_font=8)
    price = annual_prices()

    # weekly mean of each hour-of-day -> the 24 x 52 surface
    grid = price[:NWEEK * 7 * 24].reshape(NWEEK, 7, 24).mean(axis=1).T
    # a handful of scarcity weeks would otherwise own the whole colour ramp,
    # so the scale is capped at the 99th percentile and the bar is extended
    vmin = np.floor(grid.min() / 5) * 5
    vmax = np.ceil(np.percentile(grid, 99) / 5) * 5
    vmid = grid.mean()
    cmap = LinearSegmentedColormap.from_list(
        "price", PALETTES["cb_diverging"], N=256)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)

    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(2, 3, left=0.068, right=0.975, top=0.905,
                          bottom=0.105, hspace=0.30, wspace=0.055,
                          width_ratios=[1.0, 0.21, 0.048],
                          height_ratios=[1.0, 0.36])
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1], sharey=ax)
    axc = fig.add_subplot(gs[0, 2])
    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axh = fig.add_subplot(gs[1, 1])

    # ---- (a) price surface -----------------------------------------------
    xe = np.arange(NWEEK + 1) + 0.5
    ye = np.arange(25)
    mesh = ax.pcolormesh(xe, ye, grid, cmap=cmap, norm=norm, shading="flat",
                         rasterized=True)
    ax.set_xlim(0.5, NWEEK + 0.5)
    ax.set_ylim(0, 24)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_yticklabels(["00", "06", "12", "18", "24"])
    ax.set_ylabel("Hour of day")
    ax.tick_params(labelbottom=False)
    ax.set_title("Mean day-ahead price by hour and week", fontsize=8.8,
                 fontweight="bold", loc="left", x=0.0)

    # call out the negative-price window; the label sits directly on the
    # dashed box, so no leader line has to cross the surface
    neg = grid < 0
    if neg.any():
        wk = np.where(neg.any(axis=0))[0]
        hr = np.where(neg.any(axis=1))[0]
        ax.add_patch(plt.Rectangle(
            (wk[0] + 0.5, hr[0]), wk[-1] - wk[0] + 1, hr[-1] - hr[0] + 1,
            fill=False, edgecolor="#0b3d63", lw=0.9, ls=(0, (3, 2)),
            zorder=4))
        ax.text(wk[0] + 0.7, hr[-1] + 1.35,
                f"negative prices: {neg.sum()} of {grid.size} cells, "
                f"min {grid.min():.0f} $/MWh",
                fontsize=6.3, color="#0b3d63", ha="left", va="bottom",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor="#0b3d63", lw=0.5, alpha=0.93))

    # ---- (b) diurnal marginal --------------------------------------------
    dm = grid.mean(axis=1)
    axr.barh(np.arange(24) + 0.5, dm, height=0.86,
             color=[cmap(norm(v)) for v in dm], lw=0, zorder=3)
    axr.axvline(vmid, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=4)
    axr.set_xlim(0, dm.max() * 1.16)
    axr.set_xticks([0, 40, 80])
    axr.tick_params(labelleft=False, labelsize=6.6)
    axr.set_xlabel("$/MWh", fontsize=7)
    axr.set_title("Diurnal mean", fontsize=7.8, fontweight="bold",
                  loc="left", x=0.0)
    axr.grid(axis="x", lw=0.4, alpha=0.5)

    # ---- (c) monthly marginal --------------------------------------------
    mw = (np.arange(12) + 0.5) * NWEEK / 12.0
    edges = np.round(np.arange(13) * NWEEK / 12.0).astype(int)
    mm = np.array([grid[:, edges[i]:edges[i + 1]].mean() for i in range(12)])
    axb.plot(mw, mm, color="#5d6a75", lw=1.2, zorder=3)
    axb.scatter(mw, mm, s=26, c=[cmap(norm(v)) for v in mm],
                edgecolors=INK, linewidths=0.55, zorder=5)
    axb.axhline(vmid, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
    axb.text(0.995, 0.94, f"annual mean {vmid:.0f} $/MWh",
             transform=axb.transAxes, fontsize=6.2, color=GREY, ha="right",
             va="top")
    axb.set_xticks(mw)
    axb.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O",
                         "N", "D"])
    axb.tick_params(axis="x", length=0)
    axb.set_xlabel("Month")
    axb.set_ylabel("$/MWh", fontsize=7.4)
    axb.set_ylim(mm.min() - 9, mm.max() + 15)
    axb.grid(axis="y", lw=0.4, alpha=0.5)
    axb.set_title("Monthly mean", fontsize=7.8, fontweight="bold",
                  loc="left", x=0.0)

    # ---- (d) distribution of the surface ---------------------------------
    cnt, be = np.histogram(np.clip(grid.ravel(), vmin, vmax), bins=24,
                           range=(vmin, vmax))
    ctr = 0.5 * (be[:-1] + be[1:])
    axh.bar(ctr, cnt, width=(be[1] - be[0]) * 0.92,
            color=[cmap(norm(v)) for v in ctr], lw=0, zorder=3)
    axh.axvline(vmid, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=4)
    axh.set_xlim(vmin, vmax)
    axh.set_xticks([0, 40, 80])
    axh.set_yticks([])
    axh.tick_params(labelsize=6.6)
    axh.set_xlabel("$/MWh", fontsize=7)
    axh.set_title("Distribution", fontsize=7.8, fontweight="bold",
                  loc="left", x=0.0)
    axh.spines["left"].set_visible(False)

    # ---- one shared colorbar ---------------------------------------------
    cb = fig.colorbar(mesh, cax=axc, extend="max")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color=INK)
    # ticks must stay strictly inside the norm range: a tick outside a
    # TwoSlopeNorm collapses the colorbar transform and blanks the bar
    ticks = [t for t in (-20, 0, 20, 40, 60, 80) if vmin < t < vmax]
    cb.set_ticks(ticks)
    cb.set_label("Day-ahead price ($/MWh)", fontsize=7)

    for a in (axr, axb, axh):
        style_axes(a)

    fw, fh = fig.get_size_inches()
    for a, lab in ((ax, "(a)"), (axr, "(b)"), (axb, "(c)"), (axh, "(d)")):
        bb = a.get_position()
        a.text(-0.30 / (bb.width * fw), 1.0 + 0.055 / (bb.height * fh), lab,
               transform=a.transAxes, ha="left", va="bottom",
               fontweight="bold", fontsize=9.5)

    print(f"surface min={grid.min():.1f}  mean={vmid:.1f}  "
          f"max={grid.max():.1f} $/MWh  (scale {vmin:.0f}..{vmax:.0f})")
    print(f"negative cells={int((grid < 0).sum())}/{grid.size}  "
          f"hourly max={price.max():.0f} $/MWh")
    savefig(fig, "196_price_heatmap.png")


if __name__ == "__main__":
    main()
