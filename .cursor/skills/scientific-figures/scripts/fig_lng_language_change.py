"""Logistic S-curves of grammatical change across two centuries of corpora.

Twelve variables are simulated as binomial attestations per five-year corpus
bin; each one is then fitted with a logistic, and every number the figure
quotes - the fifty-percent year, the maximum rate in percent per decade, the
onset year, the apparent-time profile - is read off those fits rather than
written into the text.  Four variables are highlighted in the main panel over
four labelled historical periods; the supporting panels ask whether later
changes run faster, and whether the youngest speakers alive today already show
the same slope in apparent time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.84, 4.98, 4.56)
AUX_X, AUX_W, AUX_H = 7.12, 2.90, 1.98
AUX_TOP = (AUX_X, 3.42, AUX_W, AUX_H)
AUX_BOT = (AUX_X, 0.84, AUX_W, AUX_H)

YEARS = np.arange(1800, 2006, 5)
N_BIN = 420                     # attestations per corpus bin
NAMED = [("do-support in questions", 1852.0, 15.0, RED),
         ("periphrastic future", 1898.0, 12.0, BLUE),
         ("s-genitive, inanimate head", 1941.0, 10.0, GREEN),
         ("progressive passive", 1976.0, 8.0, GOLD)]
N_EXTRA = 8
ERAS = [(1800, 1870, "industrialisation"), (1870, 1920, "mass literacy"),
        (1920, 1970, "broadcast media"), (1970, 2010, "digital era")]
SURVEY_YEAR = 2000
AGES = np.array([25, 35, 45, 55, 65, 75])


def logistic(t, t50, s):
    return 100.0 / (1.0 + np.exp(-(t - t50) / s))


def simulate(seed=807):
    """Attestations plus logistic fits for every variable."""
    rng = np.random.default_rng(seed)
    truth = [(n, t, s, c) for n, t, s, c in NAMED]
    for i in range(N_EXTRA):
        t50 = rng.uniform(1832.0, 1988.0)
        s = np.clip(20.5 - 0.062 * (t50 - 1830.0) + rng.normal(0.0, 1.9),
                    5.0, 24.0)
        truth.append(("variable %d" % (i + 1), t50, s, GREY))

    out = []
    for name, t50, s, col in truth:
        p = logistic(YEARS, t50, s) / 100.0
        obs = rng.binomial(N_BIN, np.clip(p, 1e-4, 1.0)) / N_BIN * 100.0
        par, _ = curve_fit(logistic, YEARS, obs, p0=[t50, s],
                           bounds=([1780.0, 2.0], [2020.0, 60.0]),
                           maxfev=20000)
        out.append(dict(name=name, color=col, obs=obs, t50=par[0], s=par[1],
                        rate=1000.0 / (4.0 * par[1]),
                        onset=par[0] - np.log(9.0) * par[1]))
    return out


def main_panel(ax, fits):
    for y0, y1, name in ERAS:
        ax.axvspan(y0, y1, color="#f5f5f5" if (y0 // 50) % 2 else "#f0f0f0",
                   lw=0.0, zorder=0)
        ax.text(0.5 * (y0 + y1), -8.2, name, ha="center", va="bottom",
                fontsize=6.6, color=MUTED)
    ax.axhline(50.0, color=GREY, lw=0.8, ls=(0, (2.4, 2.0)), zorder=1)

    tt = np.linspace(YEARS[0], YEARS[-1], 400)
    for f in fits[:len(NAMED)]:
        ax.scatter(YEARS, f["obs"], s=9.0, color=f["color"], alpha=0.45,
                   linewidths=0.0, zorder=3)
        ax.plot(tt, logistic(tt, f["t50"], f["s"]), color=f["color"], lw=1.8,
                zorder=4,
                label="%s\n$t_{50}$ %d,  %.1f %%/decade"
                      % (f["name"], round(f["t50"]), f["rate"]))
        ax.plot([f["t50"]], [50.0], marker="o", ms=5.2, mfc="white",
                mec=f["color"], mew=1.4, zorder=6)
        ax.annotate("%d" % round(f["t50"]), xy=(f["t50"], 50.0),
                    xytext=(0.0, -13.0), textcoords="offset points",
                    ha="center", va="top", fontsize=6.8, color=f["color"])

    ax.set_xlim(YEARS[0] - 4, YEARS[-1] + 4)
    ax.set_ylim(-9.5, 107.0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("year (CE)")
    ax.set_ylabel("share of the innovative variant (%)")
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=6.8,
              handlelength=1.6, labelspacing=0.6, borderaxespad=0.4,
              bbox_to_anchor=(0.005, 0.995))
    ax.set_title("Every variable follows the same S, only the timing differs",
                 fontsize=9.0, fontweight="bold", color=INK)


def rate_panel(ax, fits):
    onset = np.array([f["onset"] for f in fits])
    rate = np.array([f["rate"] for f in fits])
    b, a = np.polyfit(onset, rate, 1)
    xx = np.linspace(onset.min() - 6, onset.max() + 6, 50)
    ax.plot(xx, a + b * xx, color=SUB, lw=1.2, ls=(0, (4.0, 2.4)), zorder=2)
    ax.scatter(onset[len(NAMED):], rate[len(NAMED):], s=20.0, color=GREY,
               alpha=0.7, linewidths=0.0, zorder=3, label="other variables")
    for f in fits[:len(NAMED)]:
        ax.scatter([f["onset"]], [f["rate"]], s=34.0, color=f["color"],
                   edgecolor="white", linewidths=0.8, zorder=4)
    r = np.corrcoef(onset, rate)[0, 1]
    ax.text(0.04, 0.94, "n=%d variables,  r = %.2f\n%+.2f %%/decade per "
            "century" % (onset.size, r, b * 100.0),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB, linespacing=1.5)
    ax.set_xlabel("onset year, 10 % crossing (CE)", fontsize=7.4)
    ax.set_ylabel("max rate (%/decade)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Later changes run faster", fontsize=8.2, fontweight="bold",
                 color=INK)


def apparent_panel(ax, fits, seed=17):
    rng = np.random.default_rng(seed)
    form = SURVEY_YEAR - AGES + 15.0        # formative years of each cohort
    for f in fits[2:4]:
        p = logistic(form, f["t50"], f["s"])
        obs = np.clip(p + rng.normal(0.0, 3.4, p.size), 0.0, 100.0)
        ax.plot(AGES, obs, marker="o", ms=3.8, lw=1.5, color=f["color"],
                mfc="white", mew=1.0, label=f["name"].split(",")[0])
    ax.set_xlabel("speaker age in %d (years)" % SURVEY_YEAR, fontsize=7.4)
    ax.set_ylabel("use of the variant (%)", fontsize=7.4)
    ax.set_xticks(AGES)
    ax.set_ylim(-4.0, 108.0)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5,
              handlelength=1.5, borderaxespad=0.3)
    ax.text(0.97, 0.93, "cohorts acquired the variant\nin %d-%d"
            % (round(form.min()), round(form.max())),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=MUTED, linespacing=1.4)
    ax.set_title("Apparent time across age cohorts", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    fits = simulate()

    fig = plt.figure(figsize=(FW, FH))
    ax_main = fig.add_axes([MAIN[0] / FW, MAIN[1] / FH,
                            MAIN[2] / FW, MAIN[3] / FH])
    ax_rate = fig.add_axes([AUX_TOP[0] / FW, AUX_TOP[1] / FH,
                            AUX_TOP[2] / FW, AUX_TOP[3] / FH])
    ax_app = fig.add_axes([AUX_BOT[0] / FW, AUX_BOT[1] / FH,
                           AUX_BOT[2] / FW, AUX_BOT[3] / FH])

    main_panel(ax_main, fits)
    rate_panel(ax_rate, fits)
    apparent_panel(ax_app, fits, )

    span = fits[3]["t50"] - fits[0]["t50"]
    panel_labels_inches(fig, [ax_main, ax_rate, ax_app], dx=-0.56, dy=0.10)
    savefig(fig, "807_lng_language_change.png")


if __name__ == "__main__":
    main()
