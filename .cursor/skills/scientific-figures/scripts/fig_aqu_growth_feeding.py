"""Growth against feeding rate, and the two curves that price it.

Nine feeding rates are grown through one bioenergetic integration,
dW/dt = A g(T) W^0.68 h(r) - M W, in three replicate tanks each.  The
temperature term g(T) is a Gaussian in the seasonal water temperature drawn on
the right-hand axis, so the steepest part of every growth curve necessarily
sits under the summer peak and the curves flatten together through the winter
- the panel cannot show growth the temperature does not support.

Everything quoted is measured off those arrays afterwards.  Specific growth
rate is 100 ln(W2/W1) / days within fixed body-weight stages, which is why it
falls as the fish get bigger even though absolute daily gain is still rising.
The feed conversion ratio is cumulative feed divided by the weight gained in
the same stage, so it climbs for the mirror-image reason: maintenance takes a
larger share of the ration as W grows.  The dose-response panel fits a
quadratic to the nine end weights and reads the optimum off the vertex, rather
than declaring one of the simulated rates to be best.

Illustrative values; this is a layout reference, not a feeding table.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_palette, lighten, darken
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY = "#808080"
TEMP_C, ACCENT = "#c0603f", "#b23a48"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.04, 4.86, 4.16)
AUX1 = (6.62, 3.14, 2.78, 2.06)
AUX2 = (6.62, 1.04, 2.78, 1.44)

DAY0, N_DAY = 60, 330
N_REF = 200                               # isothermal reference trial, days
T_REF = 27.0
RATES = np.linspace(1.2, 5.0, 9)          # % body weight per day
SHOW = [0, 2, 4, 7]
N_TANK = 3
W0 = 12.0
A_GROW, M_MAINT = 0.205, 0.0012
STAGES = np.array([0.0, 40.0, 90.0, 160.0, 260.0, 400.0])
STAGE_LAB = ["12-40", "40-90", "90-160", "160-260", "260-400"]


def water_temp(day):
    return 18.0 + 10.0 * np.sin(2.0 * np.pi * (day - 100.0) / 365.0)


def ration_response(r):
    """Saturating intake with an over-feeding penalty; peaks near 3% BW/day."""
    return r / (r + 1.8) - 0.014 * r ** 2


def grow(rng, n_day=N_DAY, iso=False):
    """Integrate the cohort; ``iso`` holds temperature at the reference value.

    The seasonal run is what the reader sees in panel (a).  The isothermal run
    exists because FCR and SGR per weight stage are otherwise confounded by
    which season the stage happened to fall in - a stage spent in winter shows
    a huge FCR that says nothing about body size.
    """
    day = np.arange(n_day + 1) + (0 if iso else DAY0)
    temp = np.full(n_day + 1, T_REF) if iso else water_temp(day)
    gt = np.exp(-((temp - T_REF) / 9.5) ** 2)

    w = np.zeros((RATES.size, N_TANK, n_day + 1))
    feed = np.zeros_like(w)
    for i, r in enumerate(RATES):
        hr = ration_response(r)
        for k in range(N_TANK):
            tank = float(np.exp(rng.normal(0.0, 0.030)))
            w[i, k, 0] = W0 * float(np.exp(rng.normal(0.0, 0.035)))
            for t in range(n_day):
                gain = (A_GROW * tank * gt[t] * w[i, k, t] ** 0.68 * hr
                        - M_MAINT * w[i, k, t])
                gain = max(gain, 0.0) * float(np.exp(rng.normal(0.0, 0.035)))
                w[i, k, t + 1] = w[i, k, t] + gain
                feed[i, k, t + 1] = (feed[i, k, t]
                                     + r / 100.0 * w[i, k, t])
    return dict(day=day, temp=temp, w=w, feed=feed)


def stage_metrics(d, i):
    """FCR and SGR inside fixed weight stages, averaged over the tanks."""
    fcr = np.full(len(STAGE_LAB), np.nan)
    sgr = np.full(len(STAGE_LAB), np.nan)
    mid = np.full(len(STAGE_LAB), np.nan)
    for s in range(len(STAGE_LAB)):
        f_all, g_all, m_all = [], [], []
        for k in range(N_TANK):
            w = d["w"][i, k]
            inside = np.where((w >= STAGES[s]) & (w < STAGES[s + 1]))[0]
            if inside.size < 4:
                continue
            a, b = inside[0], inside[-1]
            gain = w[b] - w[a]
            if gain <= 1e-6:
                continue
            f_all.append((d["feed"][i, k, b] - d["feed"][i, k, a]) / gain)
            g_all.append(100.0 * np.log(w[b] / w[a]) / (b - a))
            m_all.append(np.sqrt(w[a] * w[b]))
        if f_all:
            fcr[s] = float(np.mean(f_all))
            sgr[s] = float(np.mean(g_all))
            mid[s] = float(np.mean(m_all))
    return fcr, sgr, mid


def draw_growth(ax, d, cols):
    day = d["day"]
    axt = ax.twinx()
    axt.plot(day, d["temp"], color=lighten(TEMP_C, 0.30), lw=1.2,
             ls=(0, (4.0, 2.0)), zorder=2)
    axt.set_ylim(0.0, d["temp"].max() * 2.05)
    axt.set_ylabel("water temperature ($^{\\circ}$C)", fontsize=7.8,
                   labelpad=1.5, color=darken(TEMP_C, 0.05))
    axt.tick_params(labelsize=6.8, pad=1.5, colors=darken(TEMP_C, 0.05))
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_visible(True)
    axt.spines["right"].set_color(darken(TEMP_C, 0.05))
    axt.spines["right"].set_linewidth(0.8)
    k_hot = int(np.argmax(d["temp"]))
    axt.axvline(day[k_hot], color=lighten(TEMP_C, 0.62), lw=0.9,
                ls=(0, (2.4, 2.0)), zorder=1)
    axt.text(day[k_hot] + 3.0, d["temp"].max() * 1.02,
             "temperature peak {:.1f} $^{{\\circ}}$C\nfastest growth here"
             .format(d["temp"].max()), fontsize=6.3,
             color=darken(TEMP_C, 0.12), ha="left", va="bottom",
             linespacing=1.30, zorder=6)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    axt.set_zorder(1)

    sample = np.arange(0, N_DAY + 1, 30)
    out = []
    for j, i in enumerate(SHOW):
        mean = d["w"][i].mean(axis=0)
        sd = d["w"][i].std(axis=0, ddof=1)
        ax.plot(day, mean, color=cols[j], lw=1.55, zorder=6)
        ax.errorbar(day[sample], mean[sample], yerr=sd[sample], fmt="o",
                    ms=3.0, color=cols[j], ecolor=cols[j], elinewidth=0.8,
                    capsize=2.0, markeredgecolor="white",
                    markeredgewidth=0.35, zorder=7)
        sgr_all = 100.0 * np.log(mean[-1] / mean[0]) / N_DAY
        out.append((RATES[i], mean[-1], sd[-1], sgr_all))
        ax.annotate("{:.1f}% BW/d\n{:.0f} g,  SGR {:.2f} %/d"
                    .format(RATES[i], mean[-1], sgr_all),
                    xy=(day[-1], mean[-1]), xytext=(-6, 0),
                    textcoords="offset points", fontsize=6.3,
                    color=darken(cols[j], 0.18), ha="right", va="center",
                    linespacing=1.28, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.82))

    ax.set_xlim(day[0] - 4, day[-1] + 4)
    ax.set_ylim(0.0, d["w"][:, :, -1].max() * 1.30)
    ax.set_xlabel("day of year (day)", fontsize=8.0)
    ax.set_ylabel("mean body weight (g)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Body weight under four feeding rates, three tanks each",
                 fontsize=8.5, fontweight="bold")

    handles = [Line2D([], [], color=cols[j], lw=1.55, marker="o", ms=3.0,
                      label="ration {:.1f}% BW/day".format(RATES[i]))
               for j, i in enumerate(SHOW)]
    handles.append(Line2D([], [], color=lighten(TEMP_C, 0.30), lw=1.2,
                          ls=(0, (4.0, 2.0)),
                          label="water temperature (right axis)"))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.5,
                    frameon=True, framealpha=0.94, borderpad=0.42,
                    handlelength=1.8, labelspacing=0.32)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(12)
    return out


def draw_fcr(ax, d, cols):
    x = np.arange(len(STAGE_LAB))
    best = None
    for j, i in enumerate(SHOW):
        fcr, sgr, mid = stage_metrics(d, i)
        ok = np.isfinite(fcr)
        ax.plot(x[ok], fcr[ok], color=cols[j], lw=1.35, marker="o", ms=3.2,
                markeredgecolor="white", markeredgewidth=0.35, zorder=6)
        ax.annotate("{:.1f}%".format(RATES[i]),
                    xy=(x[ok][-1], fcr[ok][-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=6.1,
                    color=darken(cols[j], 0.18), ha="left", va="center",
                    zorder=8)
        if i == SHOW[2]:
            best = (sgr, ok)

    axs = ax.twinx()
    sgr, ok = best
    axs.plot(x[ok], sgr[ok], color=GREY, lw=1.15, ls=(0, (3.4, 2.0)),
             marker="s", ms=2.8, markeredgecolor="white",
             markeredgewidth=0.3, zorder=5)
    axs.set_ylim(0.0, np.nanmax(sgr) * 1.45)
    axs.set_ylabel("SGR (%/day)", fontsize=7.4, labelpad=1.5, color=SUB)
    axs.tick_params(labelsize=6.4, pad=1.5, colors=SUB)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(True)
    axs.spines["right"].set_color(GREY)
    axs.spines["right"].set_linewidth(0.8)
    axs.annotate("SGR at {:.1f}% BW/day\nfalls as the fish grow"
                 .format(RATES[SHOW[2]]), xy=(x[ok][0], sgr[ok][0]),
                 xytext=(9, 3), textcoords="offset points", fontsize=6.1,
                 color=SUB, ha="left", va="bottom", linespacing=1.28,
                 zorder=8)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LAB, fontsize=6.5)
    ax.set_xlim(-0.45, len(STAGE_LAB) - 0.20)
    ax.set_xlabel("body-weight stage (g)", fontsize=7.6)
    ax.set_ylabel("feed conversion ratio (kg feed / kg gain)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.set_title("FCR rises and SGR falls with size, at {:.0f} $^{{\\circ}}$C"
                 .format(T_REF), fontsize=8.5, fontweight="bold")


def draw_response(ax, d):
    final = d["w"][:, :, -1]
    mean = final.mean(axis=1)
    sd = final.std(axis=1, ddof=1)
    c2, c1, c0 = np.polyfit(RATES, mean, 2)
    grid = np.linspace(RATES[0] - 0.2, RATES[-1] + 0.2, 240)
    curve = c2 * grid ** 2 + c1 * grid + c0
    k = int(np.argmax(curve))
    r_opt, w_opt = float(grid[k]), float(curve[k])
    pred = c2 * RATES ** 2 + c1 * RATES + c0
    r2 = 1.0 - np.sum((mean - pred) ** 2) / np.sum((mean - mean.mean()) ** 2)

    ax.plot(grid, curve, color=darken("#2e7d5b", 0.05), lw=1.5, zorder=5)
    ax.errorbar(RATES, mean, yerr=sd, fmt="o", ms=3.6, color="#2e7d5b",
                ecolor="#2e7d5b", elinewidth=0.8, capsize=2.0,
                markeredgecolor="white", markeredgewidth=0.4, zorder=6)
    ax.vlines(r_opt, 0.0, w_opt, color=ACCENT, lw=0.9, ls=(0, (2.6, 2.0)),
              zorder=4)
    ax.plot([r_opt], [w_opt], marker="o", ms=5.4, color=ACCENT,
            markeredgecolor="white", markeredgewidth=0.7, zorder=8)
    ax.annotate("optimum ration {:.2f}% BW/day\n{:.0f} g at harvest"
                .format(r_opt, w_opt), xy=(r_opt, w_opt), xytext=(0, 12),
                textcoords="offset points", fontsize=6.3, color=ACCENT,
                ha="center", va="bottom", linespacing=1.28, zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=ACCENT,
                                shrinkA=0.0, shrinkB=2.5))
    ax.text(0.975, 0.06, "quadratic fit, R$^2$ = {:.3f}".format(r2),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=SUB, zorder=8)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0.0, max(mean.max(), w_opt) * 1.40)
    ax.set_xlabel("feeding rate (% body weight per day)", fontsize=7.6)
    ax.set_ylabel("harvest weight (g)", fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title("Dose-response: growth peaks at an intermediate ration",
                 fontsize=8.5, fontweight="bold")
    return r_opt, w_opt, r2


def main():
    set_pub_style()
    rng = np.random.default_rng(9050)
    d = grow(rng)
    d_ref = grow(rng, N_REF, iso=True)
    cols = get_palette("cb_sequential_green", 6)[1:5]

    fig = plt.figure(figsize=(FW, FH))
    ax_g = axes_inches(fig, MAIN)
    ax_f = axes_inches(fig, AUX1)
    ax_r = axes_inches(fig, AUX2)

    shown = draw_growth(ax_g, d, cols)
    draw_fcr(ax_f, d_ref, cols)
    r_opt, w_opt, r2 = draw_response(ax_r, d)

    panel_labels_inches(fig, [ax_g, ax_f, ax_r], dx=-0.46, dy=0.20)

    fcr_mid, sgr_mid, _ = stage_metrics(d_ref, SHOW[2])
    ok = np.isfinite(fcr_mid)
    hot = int(np.argmax(d["temp"]))

    print("r_opt={:.2f} w_opt={:.0f} r2={:.3f} finals={:s}"
          .format(r_opt, w_opt, r2,
                  " ".join("{:.0f}".format(v)
                           for v in d["w"][:, :, -1].mean(axis=1))))
    print("fcr={:s}".format(" ".join("{:.2f}".format(v)
                                     for v in fcr_mid[ok])))
    print("sgr={:s}".format(" ".join("{:.2f}".format(v)
                                     for v in sgr_mid[ok])))
    savefig(fig, "905_aqu_growth_feeding.png")


if __name__ == "__main__":
    main()
