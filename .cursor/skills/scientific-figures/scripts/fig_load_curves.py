"""System load at four aggregation scales, one seasonal colour set throughout.

(a) mean daily profile, (b) a representative week, (c) daily peak and daily
minimum across a representative month, (d) the seasonal load-duration curves.

All four panels are driven by a single synthetic 8760-hour series, so the
scales agree by construction.  Every panel is a load in MW, therefore the
y-limits and y-ticks are computed once over the union of all plotted data and
reused verbatim: panel-to-panel bar heights and curve heights stay
comparable.  Panel labels are offset by a fixed number of *inches* rather
than a fixed axes fraction, so they line up even though the columns differ
slightly in width.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
SCOL = {"Winter": "#2f6d95", "Spring": "#2a9d8f",
        "Summer": "#c25a3c", "Autumn": "#d9a441"}
INK = "#333333"
GREY = "#8d959c"

# representative Monday-start week and 30-day month for each season
WEEK0 = {"Winter": 14, "Spring": 105, "Summer": 196, "Autumn": 287}
MONTH0 = {"Winter": 0, "Spring": 90, "Summer": 181, "Autumn": 273}
SEASON_DAYS = {
    "Winter": np.r_[np.arange(0, 59), np.arange(334, 365)],
    "Spring": np.arange(59, 151),
    "Summer": np.arange(151, 243),
    "Autumn": np.arange(243, 334),
}


def circ_gauss(h, mu, sigma):
    """Gaussian bump on the 24-hour circle."""
    d = ((h - mu + 12.0) % 24.0) - 12.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def annual_series(seed=11):
    """Synthetic 8760-hour system load in MW (day 0 is a Monday)."""
    n = 365 * 24
    hod = np.arange(n) % 24
    doy = np.arange(n) // 24

    # winter-evening-peak shape blended into a summer-afternoon-peak shape
    shape_w = 0.30 + 0.55 * circ_gauss(hod, 8.5, 2.2) \
        + 0.95 * circ_gauss(hod, 19.0, 2.6)
    shape_s = 0.40 + 0.30 * circ_gauss(hod, 9.0, 2.5) \
        + 1.00 * circ_gauss(hod, 16.5, 3.4)
    mix = 0.5 * (1.0 + np.cos(2 * np.pi * (doy - 15) / 365.0))
    shape = mix * shape_w + (1.0 - mix) * shape_s

    # shoulder seasons sag; weekends shave the controllable part of the load
    amp = 1.0 - 0.22 * np.sin(2 * np.pi * (doy - 15) / 365.0) ** 2
    wf = np.where(doy % 7 == 5, 0.91, np.where(doy % 7 == 6, 0.85, 1.0))

    rng = np.random.default_rng(seed)
    noise = 1.0 + 0.028 * gaussian_filter1d(rng.normal(0, 1, n), 6.0)
    return (3000.0 + 3200.0 * amp * shape * wf) * noise


def main():
    set_pub_style(base_font=8)
    load = annual_series()
    daily = load.reshape(365, 24)

    # ---- assemble every series first, so the shared scale is exact --------
    prof = {s: daily[SEASON_DAYS[s]].mean(axis=0) for s in SEASONS}
    prof = {s: np.r_[v, v[0]] for s, v in prof.items()}          # close 24 h
    week = {s: load[WEEK0[s] * 24:(WEEK0[s] + 7) * 24] for s in SEASONS}
    dpk = {s: daily[MONTH0[s]:MONTH0[s] + 30].max(axis=1) for s in SEASONS}
    dmn = {s: daily[MONTH0[s]:MONTH0[s] + 30].min(axis=1) for s in SEASONS}
    ldc = {s: np.sort(load.reshape(365, 24)[SEASON_DAYS[s]].ravel())[::-1]
           for s in SEASONS}

    allv = np.concatenate([np.concatenate(list(d.values()))
                           for d in (prof, week, dpk, dmn, ldc)])
    lo = np.floor((allv.min() - 250) / 500) * 500
    hi = np.ceil((allv.max() + 250) / 500) * 500
    yticks = np.arange(lo, hi + 1, 1000)

    fig = plt.figure(figsize=(7.3, 5.05))
    gs = fig.add_gridspec(2, 2, left=0.082, right=0.985, top=0.905,
                          bottom=0.135, hspace=0.42, wspace=0.145)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # ---- (a) mean daily profile ------------------------------------------
    hh = np.arange(25)
    for s in SEASONS:
        axa.plot(hh, prof[s], color=SCOL[s], lw=1.5, zorder=4)
    # the two peaks nearly touch around 19:00, so they are marked with dots
    # and read out in the empty top-left corner instead of with leader lines
    for s, dy in (("Summer", 0), ("Winter", 1)):
        k = int(np.argmax(prof[s]))
        axa.plot([k], [prof[s][k]], marker="o", ms=3.4, color=SCOL[s],
                 mec="white", mew=0.7, zorder=6, linestyle="none")
        axa.text(0.35, hi - 130 - dy * 430,
                 f"{s} peak {prof[s][k]:,.0f} MW at {k:02d}:00",
                 fontsize=6.3, color=SCOL[s], ha="left", va="top")
    iv = int(np.argmin(prof["Spring"]))
    axa.annotate(f"overnight valley {prof['Spring'][iv]:,.0f} MW",
                 xy=(iv, prof["Spring"][iv]), xytext=(6.4, lo + 190),
                 fontsize=6.2, color=GREY, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color=GREY))
    axa.set_xlim(0, 24)
    axa.set_xticks(np.arange(0, 25, 4))
    axa.set_xticklabels([f"{h:02d}" for h in range(0, 25, 4)])
    axa.set_xlabel("Hour of day")
    axa.set_title("Mean daily profile", fontsize=8.5, fontweight="bold")

    # ---- (b) representative week -----------------------------------------
    hw = np.arange(168)
    for d in (5, 6):
        axb.axvspan(d * 24, (d + 1) * 24, color="#eceff2", lw=0, zorder=0)
    for s in SEASONS:
        axb.plot(hw, week[s], color=SCOL[s], lw=0.95, zorder=4, alpha=0.95)
    axb.set_xlim(0, 168)
    axb.set_xticks(np.arange(12, 168, 24))
    axb.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axb.tick_params(axis="x", length=0)
    axb.set_xlabel("Representative week")
    axb.set_title("Weekly load shape", fontsize=8.5, fontweight="bold")
    axb.text(0.985, 0.965, "weekends shaded", transform=axb.transAxes,
             ha="right", va="top", fontsize=6.2, color=GREY)

    # ---- (c) daily peak / minimum over one month --------------------------
    dd = np.arange(1, 31)
    for s in SEASONS:
        axc.plot(dd, dpk[s], color=SCOL[s], lw=1.35, zorder=4)
        axc.plot(dd, dmn[s], color=SCOL[s], lw=0.9, ls=(0, (1.6, 1.6)),
                 alpha=0.85, zorder=3)
    ibest = int(np.argmax(dpk["Summer"]))
    axc.annotate(f"monthly maximum {dpk['Summer'][ibest]:,.0f} MW",
                 xy=(dd[ibest], dpk["Summer"][ibest]),
                 xytext=(30.0, hi - 120), fontsize=6.2,
                 color=SCOL["Summer"], ha="right", va="top",
                 arrowprops=dict(arrowstyle="->", lw=0.6,
                                 color=SCOL["Summer"]))
    axc.set_xlim(1, 30)
    axc.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axc.set_xlabel("Day of representative month")
    axc.set_title("Daily peak and daily minimum", fontsize=8.5,
                  fontweight="bold")

    # ---- (d) load-duration curves ----------------------------------------
    for s in SEASONS:
        y = ldc[s]
        axd.plot(np.linspace(0, 100, y.size), y, color=SCOL[s], lw=1.5,
                 zorder=4)
    axd.axvspan(0, 5, color="#f0e2d8", lw=0, zorder=0)
    yr = np.sort(load)[::-1]
    p5 = yr[int(0.05 * yr.size)]
    axd.annotate(f"top 5 % of hours\nneed {yr[0] - p5:,.0f} MW of\n"
                 f"peaking capacity",
                 xy=(5.0, p5), xytext=(17.0, hi - 120), fontsize=6.2,
                 color="#9c5b3c", ha="left", va="top", linespacing=1.35,
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="#9c5b3c"))
    axd.set_xlim(0, 100)
    axd.set_xticks([0, 25, 50, 75, 100])
    axd.set_xlabel("Duration (% of season hours)")
    axd.set_title("Load-duration curves", fontsize=8.5, fontweight="bold")

    # ---- one shared scale for every panel --------------------------------
    for i, ax in enumerate((axa, axb, axc, axd)):
        ax.set_ylim(lo, hi)
        ax.set_yticks(yticks)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
        style_axes(ax)
        if i % 2 == 0:
            ax.set_ylabel("System load (MW)")
        else:
            ax.tick_params(labelleft=False)

    # panel labels offset by a constant physical distance, not axes fraction
    fw, fh = fig.get_size_inches()
    for ax, lab in zip((axa, axb, axc, axd), ["(a)", "(b)", "(c)", "(d)"]):
        bb = ax.get_position()
        ax.text(-0.40 / (bb.width * fw), 1.0 + 0.055 / (bb.height * fh), lab,
                transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", fontsize=10)

    handles = [Line2D([], [], color=SCOL[s], lw=1.6, label=s)
               for s in SEASONS]
    handles += [
        Line2D([], [], color=GREY, lw=1.35, label="Daily peak (c)"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (1.6, 1.6)),
               label="Daily minimum (c)"),
        Patch(facecolor="#eceff2", label="Weekend (b)"),
        Patch(facecolor="#f0e2d8", label="Top 5 % hours (d)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.53, -0.012), handlelength=1.7,
               columnspacing=1.8, labelspacing=0.5)

    print(f"annual peak={load.max():,.0f} MW  min={load.min():,.0f} MW  "
          f"load factor={load.mean() / load.max():.3f}")
    print(f"shared y-range = [{lo:,.0f}, {hi:,.0f}] MW")
    savefig(fig, "194_load_curves.png")


if __name__ == "__main__":
    main()
