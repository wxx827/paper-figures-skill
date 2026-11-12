"""Annual NDVI trajectories for four crop and grassland parcels.

Main panel: clear-sky NDVI observations (translucent dots) from a 5-day
revisit, the cloud gaps they leave behind, and the smooth reconstruction
fitted through them. Green-up, peak and senescence dates are marked on the
maize parcel, which is the reference crop of the figure.

Auxiliary panels (58 % of the main panel): the spread of the three phenology
dates across the pixels of each parcel, and the growing-season NDVI integral
that follows from those dates.

Every date and integral printed on the figure is recomputed from the curves
that are drawn, so changing the seed moves the annotations with the data.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches, lowess

# (name, colour, base, amplitude, green-up, rise rate, senescence, fall rate)
PARCELS = [
    ("Winter wheat", "#c9a227", 0.16, 0.72, 62.0, 9.0, 172.0, 8.5),
    ("Maize", "#2e7d5b", 0.18, 0.71, 152.0, 8.0, 256.0, 9.5),
    ("Grassland", "#c0603f", 0.15, 0.42, 104.0, 14.0, 276.0, 16.0),
    ("Paddy rice", "#3b5b92", 0.10, 0.68, 166.0, 7.0, 268.0, 10.5),
]
REF = 1                       # the parcel carrying the phenology callouts
ACCENT = "#b23a48"
REVISIT = 5                   # days between acquisitions
DOY = np.arange(1, 366)


def double_logistic(t, base, amp, sos, k1, eos, k2):
    return base + amp * (1.0 / (1.0 + np.exp(-(t - sos) / k1))
                         - 1.0 / (1.0 + np.exp(-(t - eos) / k2)))


def cloud_mask(rng, dates):
    """Clear/cloudy flag per acquisition, cloudier in the humid mid-season."""
    p_cloud = 0.16 + 0.26 * np.exp(-((dates - 205.0) / 58.0) ** 2)
    clear = rng.random(dates.size) > p_cloud
    # two persistent overcast spells that no per-date draw would produce
    for start, span in ((118, 26), (232, 21)):
        clear[(dates >= start) & (dates < start + span)] = False
    clear[0] = clear[-1] = True
    return clear


def gap_spans(dates, clear, min_len=14):
    """Contiguous cloudy stretches longer than ``min_len`` days."""
    spans, run = [], None
    for d, ok in zip(dates, clear):
        if not ok and run is None:
            run = d
        elif ok and run is not None:
            if d - run >= min_len:
                spans.append((run - REVISIT / 2.0, d - REVISIT / 2.0))
            run = None
    return spans


def phenology(t, y):
    """Green-up, peak and senescence dates from a reconstructed curve."""
    i_pk = int(np.argmax(y))
    base = np.percentile(y, 5)
    half = base + 0.5 * (y[i_pk] - base)
    rise = np.where(y[:i_pk + 1] <= half)[0]
    fall = np.where(y[i_pk:] <= half)[0]
    sos = t[rise[-1]] if rise.size else t[0]
    eos = t[i_pk + fall[0]] if fall.size else t[-1]
    sel = (t >= sos) & (t <= eos)
    integral = float(np.trapezoid(y[sel] - base, t[sel]))
    return float(sos), float(t[i_pk]), float(eos), integral


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(758)

    dates = np.arange(3, 366, REVISIT, dtype=float)
    clear = cloud_mask(rng, dates)
    spans = gap_spans(dates, clear)

    obs, recon, phen = {}, {}, {}
    for name, col, base, amp, sos, k1, eos, k2 in PARCELS:
        truth = double_logistic(dates, base, amp, sos, k1, eos, k2)
        y = truth + rng.normal(0.0, 0.028, dates.size)
        obs[name] = (dates[clear], np.clip(y[clear], -0.05, 1.0))
        xg, yg = lowess(obs[name][0], obs[name][1], frac=0.20, n_out=DOY.size)
        recon[name] = (xg, yg)
        phen[name] = phenology(xg, yg)

    # per-pixel ensembles drive the two auxiliary panels
    n_pix = 36
    ens = {}
    for name, col, base, amp, sos, k1, eos, k2 in PARCELS:
        rows = []
        for _ in range(n_pix):
            y = double_logistic(DOY.astype(float),
                                base + rng.normal(0, 0.012),
                                amp * (1.0 + rng.normal(0, 0.07)),
                                sos + rng.normal(0, 7.0), k1,
                                eos + rng.normal(0, 9.0), k2)
            rows.append(phenology(DOY.astype(float), y))
        ens[name] = np.asarray(rows)

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, (0.82, 1.05, 5.28, 4.45))
    ax_box = axes_inches(fig, (6.94, 3.62, 3.02, 1.88))
    ax_int = axes_inches(fig, (6.94, 1.05, 3.02, 1.66))

    # ---- (a) trajectories -------------------------------------------------
    for k, (x0, x1) in enumerate(spans):
        ax.axvspan(x0, x1, facecolor="#cccccc", alpha=0.55, lw=0, zorder=0)
        ax.text((x0 + x1) / 2.0, 1.045, "cloud\ngap", ha="center", va="top",
                fontsize=6.0, color="#555555", linespacing=1.2, zorder=1)

    for name, col, *_ in PARCELS:
        ox, oy = obs[name]
        ax.plot(ox, oy, ls="none", marker="o", ms=2.8, mfc=col, mec="none",
                alpha=0.38, zorder=2)
        xg, yg = recon[name]
        ax.plot(xg, yg, color=col, lw=1.7, zorder=3, label=name)

    name_ref, col_ref = PARCELS[REF][0], PARCELS[REF][1]
    xg, yg = recon[name_ref]
    sos, pos, eos, integ = phen[name_ref]
    for lab, day, ha, dx in (("green-up", sos, "right", -4.0),
                             ("peak", pos, "center", 0.0),
                             ("senescence", eos, "left", 4.0)):
        yv = float(np.interp(day, xg, yg))
        ax.plot([day, day], [0.0, yv], color=ACCENT, lw=0.9, ls=(0, (4, 2)),
                zorder=4)
        ax.plot([day], [yv], marker="o", ms=4.6, mfc="white", mec=ACCENT,
                mew=1.3, zorder=5)
        ax.text(day + dx, yv + 0.030, "%s\nDOY %.0f" % (lab, day), ha=ha,
                va="bottom", fontsize=6.4, color=ACCENT, linespacing=1.25,
                zorder=6)
    ax.annotate("", xy=(sos, 0.055), xytext=(eos, 0.055),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=ACCENT))
    ax.text((sos + eos) / 2.0, 0.075,
            "%s season length %.0f d" % (name_ref.lower(), eos - sos),
            ha="center", va="bottom", fontsize=6.4, color=ACCENT)

    ax.set_xlim(1, 365)
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks([1, 60, 121, 182, 244, 305, 365])
    ax.set_xlabel("Day of year (d)")
    ax.set_ylabel("NDVI (-)")
    ax.set_title("Clear-sky observations and gap-filled reconstruction",
                 fontsize=9.0, fontweight="bold")
    style_axes(ax)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False,
              labelspacing=0.28, handlelength=1.5,
              bbox_to_anchor=(0.005, 0.995))
    ax.text(0.005, 0.795, "dots: %d of %d acquisitions clear (%.0f %%)"
            % (int(clear.sum()), clear.size, 100.0 * clear.mean()),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color="#777777")

    # ---- (b) phenology-date spread ----------------------------------------
    metrics = ["Green-up", "Peak", "Senescence"]
    w = 0.17
    for i, (name, col, *_) in enumerate(PARCELS):
        off = (i - (len(PARCELS) - 1) / 2.0) * w
        bp = ax_box.boxplot([ens[name][:, m] for m in range(3)],
                            positions=np.arange(3) + off, widths=w * 0.82,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="#1a1a1a", lw=0.8),
                            whiskerprops=dict(color="#555555", lw=0.6),
                            capprops=dict(color="#555555", lw=0.6),
                            boxprops=dict(edgecolor="#555555", lw=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.80)
    ax_box.set_xticks(range(3))
    ax_box.set_xticklabels(metrics, fontsize=6.8)
    ax_box.set_xlim(-0.55, 2.55)
    ax_box.set_ylabel("Day of year (d)", fontsize=7.4)
    ax_box.set_title("Phenology dates across %d pixels per parcel" % n_pix,
                     fontsize=8.5, fontweight="bold")
    ax_box.tick_params(labelsize=6.6)
    style_axes(ax_box)
    ax_box.grid(axis="y", lw=0.5, alpha=0.45)

    # ---- (c) growing-season integral --------------------------------------
    names = [p[0] for p in PARCELS]
    cols = [p[1] for p in PARCELS]
    med = np.array([np.median(ens[n][:, 3]) for n in names])
    err = np.array([[np.median(ens[n][:, 3]) - np.percentile(ens[n][:, 3], 25)
                     for n in names],
                    [np.percentile(ens[n][:, 3], 75) - np.median(ens[n][:, 3])
                     for n in names]])
    ypos = np.arange(len(names))[::-1]
    ax_int.barh(ypos, med, height=0.58, color=cols, edgecolor="white",
                linewidth=0.4, zorder=3)
    ax_int.errorbar(med, ypos, xerr=err, fmt="none", ecolor="#333333",
                    elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
    for i, (y, v) in enumerate(zip(ypos, med)):
        ax_int.text(v + err[1][i] + 1.6, y, "%.0f" % v, ha="left",
                    va="center", fontsize=6.4, color="#333333")
    ax_int.set_yticks(ypos)
    ax_int.set_yticklabels(names, fontsize=6.8)
    ax_int.set_xlim(0, med.max() * 1.22)
    ax_int.set_xlabel("Season NDVI integral (NDVI $\\cdot$ d)", fontsize=7.4,
                      labelpad=1.0)
    ax_int.set_title("Integrated greenness over the season", fontsize=8.5,
                     fontweight="bold")
    ax_int.tick_params(labelsize=6.6)
    style_axes(ax_int)
    ax_int.grid(axis="x", lw=0.5, alpha=0.45)

    panel_labels_inches(fig, [ax, ax_box, ax_int], dx=-0.46, dy=0.16)


    savefig(fig, "758_rs_ndvi_timeseries.png")


if __name__ == "__main__":
    main()
