"""Multi-site validation of a satellite surface soil-moisture product.

Panels (a)-(e): one core validation site each, satellite retrieval against the
in-situ network, points coloured by day of year, with 1:1 line, ordinary
least-squares fit, marginal density bars and the full statistics block.
Panel (f) is an intentionally subordinate summary panel (smaller, on a tinted
background) comparing the skill metrics across sites.

Retrieval skill degrades with increasing vegetation optical depth, so the
forested site shows the lowest R2 and the largest unbiased RMSE.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

# name, land cover, VOD, slope, offset, noise sd, n
SITES = [
    ("REMEDHUS", "Cropland", 0.18, 0.94, 0.021, 0.023, 148),
    ("Kenaston", "Grassland", 0.31, 0.88, 0.014, 0.027, 132),
    ("Yanco", "Semi-arid shrub", 0.12, 0.97, 0.010, 0.026, 156),
    ("Little Washita", "Mixed pasture", 0.42, 0.83, 0.048, 0.036, 121),
    ("Fort Cobb", "Forest edge", 0.68, 0.71, 0.038, 0.036, 104),
]
LIM = (0.03, 0.38)
SM_MEAN, SM_SD = 0.205, 0.062


def synth_site(rng, n, slope, offset, sd):
    """In-situ series driven by a seasonal cycle plus rainfall / dry-down
    events; the satellite series adds a multiplicative bias and retrieval
    noise whose magnitude grows with vegetation optical depth."""
    doy = np.sort(rng.integers(5, 361, n)).astype(float)
    seas = 0.9 * np.sin(2 * np.pi * (doy - 55) / 365.0)
    wet = np.zeros(n)
    for _ in range(6):
        t0 = rng.uniform(1, 365)
        wet += rng.uniform(0.6, 1.9) * np.exp(
            -np.clip(doy - t0, 0, None) / rng.uniform(9, 26))
    raw = seas + wet + rng.normal(0, 0.25, n)
    # standardise so every site shares the same in-situ dynamic range and the
    # reported skill is governed only by the retrieval noise
    truth = np.clip(SM_MEAN + SM_SD * (raw - raw.mean()) / raw.std(),
                    0.05, 0.40)
    sat = np.clip(offset + slope * truth + rng.normal(0, sd, n), 0.03, 0.41)
    return doy, truth, sat


def metrics(x, y):
    k, b = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))
    bias = float(np.mean(y - x))
    ub = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0)))
    return dict(slope=k, icept=b, r2=r2, rmse=rmse, mae=mae, bias=bias, ub=ub)


def marginal_bars(ax, x, y, cx, cy):
    """Thin density strips above and to the right of a scatter panel."""
    top = ax.inset_axes([0, 1.015, 1, 0.10])
    right = ax.inset_axes([1.015, 0, 0.10, 1])
    top.set_xlim(*LIM)
    right.set_ylim(*LIM)
    bins = np.linspace(*LIM, 22)
    hx, _ = np.histogram(x, bins=bins)
    hy, _ = np.histogram(y, bins=bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    wdt = bins[1] - bins[0]
    top.bar(ctr, hx, width=wdt * 0.92, color=cx, alpha=0.75, linewidth=0)
    right.barh(ctr, hy, height=wdt * 0.92, color=cy, alpha=0.75, linewidth=0)
    for a, horiz in ((top, True), (right, False)):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.patch.set_alpha(0.0)
    top.set_ylim(0, hx.max() * 1.15)
    right.set_xlim(0, hy.max() * 1.15)
    return top, right


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240189)

    data = {}
    for name, cover, vod, slope, off, sd, n in SITES:
        doy, truth, sat = synth_site(rng, n, slope, off, sd)
        data[name] = dict(doy=doy, truth=truth, sat=sat, cover=cover, vod=vod,
                          m=metrics(truth, sat))

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.50)
    labels = "abcdef"
    cmap = plt.get_cmap("twilight_shifted")

    sc = None
    for i, (name, cover, vod, *_rest) in enumerate(SITES):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        d = data[name]
        m = d["m"]
        ax.plot(LIM, LIM, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=2)
        sc = ax.scatter(d["truth"], d["sat"], c=d["doy"], cmap=cmap, vmin=1,
                        vmax=365, s=11, linewidth=0.25, edgecolor="#3a3a3a",
                        alpha=0.92, zorder=4)
        xg = np.linspace(*LIM, 20)
        ax.plot(xg, m["slope"] * xg + m["icept"], color="#b23a48", lw=1.3,
                zorder=5)
        ax.set_xlim(*LIM)
        ax.set_ylim(*LIM)
        ax.set_aspect("equal")
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.set_yticks([0.1, 0.2, 0.3, 0.4])
        ax.set_xticklabels(["0.1", "0.2", "0.3", "0.4"])
        ax.tick_params(labelsize=6.6)
        style_axes(ax)
        ax.grid(lw=0.45, alpha=0.45)
        if i // 3 == 1 or i == 2:
            ax.set_xlabel("In-situ SM (m$^3$ m$^{-3}$)", labelpad=1.5)
        if i % 3 == 0:
            ax.set_ylabel("Satellite SM (m$^3$ m$^{-3}$)")
        ax.text(0.035, 0.965, name, transform=ax.transAxes, ha="left",
                va="top", fontsize=6.8, fontweight="bold", zorder=6)
        ax.text(0.035, 0.885, cover, transform=ax.transAxes, ha="left",
                va="top", fontsize=6.2, color="#555555", zorder=6)
        ax.text(0.965, 0.035,
                f"$R^2$ {m['r2']:.2f}   $n$ {d['truth'].size}\n"
                f"RMSE {m['rmse']:.3f}\n"
                f"ubRMSE {m['ub']:.3f}\n"
                f"MAE {m['mae']:.3f}\n"
                f"bias {m['bias']:+.3f}".replace("-", "\u2212"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
                linespacing=1.3, zorder=6,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#cccccc",
                          lw=0.5, alpha=0.94))
        marginal_bars(ax, d["truth"], d["sat"], "#9cb8d4", "#d6b28c")
        add_panel_label(ax, f"({labels[i]})", x=-0.24, y=1.16, size=10)

    # ---- (f) subordinate summary panel ------------------------------------
    host = fig.add_subplot(gs[1, 2])
    host.set_xticks([])
    host.set_yticks([])
    for s in host.spines.values():
        s.set_visible(False)
    host.add_patch(FancyBboxPatch(
        (0.02, -0.16), 1.10, 1.30, boxstyle="round,pad=0.02",
        transform=host.transAxes, facecolor="#f1f1ec", edgecolor="#d5d5cc",
        linewidth=0.7, zorder=0, clip_on=False))
    host.text(0.57, 1.055, "Summary across sites", transform=host.transAxes,
              ha="center", va="bottom", fontsize=7.4, fontweight="bold",
              zorder=3)
    add_panel_label(host, "(f)", x=-0.24, y=1.16, size=10)

    sub = host.inset_axes([0.16, 0.06, 0.86, 0.80])
    names = [s[0] for s in SITES]
    r2s = np.array([data[n]["m"]["r2"] for n in names])
    ubs = np.array([data[n]["m"]["ub"] for n in names])
    ypos = np.arange(len(names))[::-1]
    sub.barh(ypos + 0.19, r2s, height=0.34, color="#3b5b92",
             edgecolor="white", linewidth=0.4, label="$R^2$ (\u2013)")
    sub.barh(ypos - 0.19, ubs / 0.08, height=0.34, color="#c0603f",
             edgecolor="white", linewidth=0.4,
             label="ubRMSE / 0.08 m$^3$ m$^{-3}$")
    for yv, v in zip(ypos, r2s):
        sub.text(v + 0.02, yv + 0.19, f"{v:.2f}", va="center", fontsize=6.0,
                 color="#1f3f6b", fontweight="bold")
    for yv, v in zip(ypos, ubs):
        sub.text(v / 0.08 + 0.02, yv - 0.19, f"{v:.3f}", va="center",
                 fontsize=6.0, color="#8c4a12", fontweight="bold")
    sub.axvline(1.0, color="#888888", lw=0.6, ls=(0, (3, 2)))
    sub.set_yticks(ypos)
    sub.set_yticklabels(names, fontsize=6.0)
    sub.set_xlim(0, 1.32)
    sub.set_xticks([0, 0.5, 1.0])
    sub.set_xlabel("normalised skill metric", fontsize=6.2, labelpad=1.0)
    sub.tick_params(labelsize=6.0)
    sub.patch.set_alpha(0.0)
    style_axes(sub)
    sub.grid(axis="x", lw=0.45, alpha=0.5)
    sub.legend(loc="lower center", bbox_to_anchor=(0.42, 1.00), ncol=1,
               fontsize=6.0, frameon=False, handlelength=1.0,
               labelspacing=0.2, borderaxespad=0.0)

    # ---- shared colorbar ---------------------------------------------------
    cax = fig.add_axes([0.30, 0.035, 0.40, 0.014])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal",
                      ticks=[1, 91, 182, 274, 365])
    cb.ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"])
    cb.set_label("Day of year of the matched overpass", fontsize=6.6,
                 labelpad=2.0)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.0, pad=1.5)

    fig.subplots_adjust(top=0.885, bottom=0.145, left=0.075, right=0.955)
    savefig(fig, "189_rs_validation_scatter.png")


if __name__ == "__main__":
    main()
