"""Ion-exchange elution profile and the purification table it produces.

The chromatogram is a sum of exponentially modified Gaussians, so every
reported number is measured off the curve rather than asserted: the peak
positions and half-height widths are found on each isolated peak, the
resolution of the critical pair is 1.18 dt / (w1 + w2) computed from those
widths, and the asymmetry factor of the target peak is the ratio of the right
and left half-widths at 10 % of its height.  The salt gradient on the second
axis is the same array that the peaks were placed against.  The two supporting
panels share one mass balance: a target protein and six contaminants, each
with its own per-step recovery, give the purification fold and the step yield
on the right-hand top panel and, as mass fractions of an equal-load lane, the
band intensities of the gel below.  Purity rises while total mass falls,
because those are two readings of the same table.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.special import erfcx

from pubstyle import set_pub_style, savefig, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"         # UV280 trace
OCHRE = "#c0603f"        # target product
GREEN = "#2e7d5b"        # yield / recovery
GOLD = "#c9a227"         # salt gradient
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

CV_END = 20.0            # column volumes
COLUMN_ML = 25.0         # mL per column volume

#            name              mu(CV) sigma  tau   area
PEAKS = [("flow-through", 1.15, 0.17, 0.22, 340.0),
         ("contaminant 1", 6.60, 0.22, 0.15, 130.0),
         ("contaminant 2", 8.55, 0.24, 0.17, 175.0),
         ("target", 10.60, 0.27, 0.21, 430.0),
         ("contaminant 3", 12.70, 0.30, 0.55, 150.0),
         ("strip peak", 16.90, 0.34, 0.28, 95.0)]
TARGET = 3
CRITICAL = 2

SALT_KNOTS = np.array([0.0, 4.0, 15.0, 16.8, 17.0, 19.0, CV_END])
SALT_MM = np.array([50.0, 50.0, 500.0, 500.0, 1000.0, 1000.0, 100.0])

FRAC_START, FRAC_WIDTH, N_FRAC = 5.0, 1.0, 13

STEPS = ["crude\nlysate", "clarified", "IEX\ncapture", "HIC", "SEC\npolish"]
TARGET_REC = [1.0, 0.95, 0.85, 0.90, 0.92]
CONTAM = [
    ("C1", 130.0, 9000.0, [1.0, 0.70, 0.060, 0.15, 0.12]),
    ("C2", 95.0, 7000.0, [1.0, 0.75, 0.080, 0.20, 0.10]),
    ("C3", 72.0, 6500.0, [1.0, 0.85, 0.120, 0.10, 0.08]),
    ("C4", 40.0, 8000.0, [1.0, 0.90, 0.050, 0.18, 0.15]),
    ("C5", 28.0, 5200.0, [1.0, 0.60, 0.035, 0.12, 0.20]),
    ("C6", 18.0, 4300.0, [1.0, 0.55, 0.025, 0.10, 0.10])]
TARGET_MW, TARGET_MG = 55.0, 900.0
LADDER = [250.0, 130.0, 100.0, 70.0, 55.0, 35.0, 25.0, 15.0, 10.0]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 0.95, 4.62, 4.55)
TOP = (7.82, 3.61, 2.68, 1.89)
BOT = (7.82, 0.95, 2.68, 1.89)


def emg(x, mu, sigma, tau, area):
    """Exponentially modified Gaussian, overflow-safe via erfcx.

    The scaled complementary error function keeps the usual
    ``exp(huge) * erfc(huge)`` product finite, but it still overflows once the
    argument is strongly negative, which is exactly the far leading tail of a
    tailed peak.  There the complementary error function has saturated at 2,
    so the closed form collapses to a plain decaying exponential and that
    branch is used instead - otherwise ``0 * inf`` puts NaNs in the tail.
    """
    d = x - mu
    z = (sigma / tau - d / sigma) / np.sqrt(2.0)
    out = np.empty_like(d)
    near = z > -18.0
    out[near] = ((area / (2.0 * tau)) *
                 np.exp(-0.5 * (d[near] / sigma) ** 2) * erfcx(z[near]))
    far = ~near
    out[far] = (area / tau) * np.exp(sigma ** 2 / (2.0 * tau ** 2) -
                                     d[far] / tau)
    return out


def crossings(x, y, level):
    """First and last x where y crosses ``level``, linearly interpolated."""
    idx = np.flatnonzero(y >= level)
    i0, i1 = idx[0], idx[-1]
    lo = np.interp(level, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    hi = np.interp(level, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])
    return float(lo), float(hi)


def peak_metrics(x, comp):
    """Apex, half-height width and 10 %-height asymmetry of one component."""
    i = int(np.argmax(comp))
    top = float(comp[i])
    lo50, hi50 = crossings(x, comp, 0.50 * top)
    lo10, hi10 = crossings(x, comp, 0.10 * top)
    apex = float(x[i])
    return dict(apex=apex, height=top, w50=hi50 - lo50,
                asym=(hi10 - apex) / (apex - lo10))


def chromatogram(rng):
    x = np.linspace(0.0, CV_END, 4000)
    comps = [emg(x, mu, s, t, a) for _n, mu, s, t, a in PEAKS]
    base = 6.0 + 2.4 * x / CV_END + 14.0 * np.exp(-((x - 4.0) / 0.9) ** 2)
    noise = rng.normal(0.0, 1.1, x.size)
    return x, comps, np.sum(comps, axis=0) + base + noise, base


def mass_balance():
    m_t = TARGET_MG * np.cumprod(TARGET_REC)
    m_c = np.array([mg * np.cumprod(rec) for _n, _mw, mg, rec in CONTAM])
    total = m_t + m_c.sum(axis=0)
    purity = 100.0 * m_t / total
    return m_t, m_c, total, purity, purity / purity[0], 100.0 * m_t / m_t[0]


def main_panel(ax, x, comps, uv, rng):
    ax_s = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_s.set_zorder(1)

    salt = np.interp(x, SALT_KNOTS, SALT_MM)
    ax_s.plot(x, salt, color=GOLD, lw=1.4, ls=(0, (5.0, 2.2)), zorder=3)
    ax_s.fill_between(x, 0.0, salt, color=lighten(GOLD, 0.90), lw=0.0,
                      zorder=2)

    tgt = comps[TARGET]
    ax.fill_between(x, 0.0, tgt, where=tgt > 0.02 * tgt.max(),
                    color=lighten(OCHRE, 0.72), lw=0.0, zorder=3)
    ax.plot(x, uv, color=BLUE, lw=1.5, zorder=6)

    met = [peak_metrics(x, c) for c in comps]
    rs = (1.18 * (met[TARGET]["apex"] - met[CRITICAL]["apex"]) /
          (met[TARGET]["w50"] + met[CRITICAL]["w50"]))

    for k, (name, _mu, _s, _t, _a) in enumerate(PEAKS):
        m = met[k]
        col = OCHRE if k == TARGET else SUB
        ax.text(m["apex"], np.interp(m["apex"], x, uv) + 26.0,
                name if k != TARGET else "TARGET",
                ha="center", va="bottom", fontsize=6.6, color=col,
                fontweight="bold" if k == TARGET else "normal")

    y_rs = met[CRITICAL]["height"] * 0.42
    ax.annotate("", xy=(met[CRITICAL]["apex"], y_rs),
                xytext=(met[TARGET]["apex"], y_rs), zorder=8,
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.8,
                                mutation_scale=8))
    ax.text((met[CRITICAL]["apex"] + met[TARGET]["apex"]) / 2.0, y_rs + 16.0,
            r"$R_s$ = " + "{:.2f}".format(rs), ha="center", va="bottom",
            fontsize=7.2, color=INK, zorder=9,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.88))
    ax.annotate(
        "target peak\napex {:.2f} CV = {:.0f} mL\n".format(
            met[TARGET]["apex"], met[TARGET]["apex"] * COLUMN_ML) +
        r"$w_{50}$ = " + "{:.2f} CV,  ".format(met[TARGET]["w50"]) +
        r"$A_s$ = " + "{:.2f}".format(met[TARGET]["asym"]),
        xy=(met[TARGET]["apex"] + 0.30, met[TARGET]["height"] * 0.80),
        xytext=(13.4, met[TARGET]["height"] * 1.02), ha="left", va="bottom",
        fontsize=6.8, color=OCHRE, linespacing=1.35, zorder=9,
        arrowprops=dict(arrowstyle="-", color=OCHRE, lw=0.8))

    pool = []
    trans = ax.get_xaxis_transform()
    for i in range(N_FRAC):
        a = FRAC_START + i * FRAC_WIDTH
        b = a + FRAC_WIDTH
        inside = (met[TARGET]["apex"] - 0.9 < (a + b) / 2.0 <
                  met[TARGET]["apex"] + 0.9)
        if inside:
            pool.append(i + 1)
        ax.add_patch(Rectangle(
            (a, 0.004), FRAC_WIDTH, 0.040, transform=trans,
            facecolor=OCHRE if inside else lighten(BLUE, 0.55 + 0.22 *
                                                   (i % 2)),
            edgecolor="white", linewidth=0.6, zorder=7, clip_on=False))
        if i % 2 == 0 or inside:
            ax.text((a + b) / 2.0, 0.024, "{:d}".format(i + 1),
                    transform=trans, ha="center", va="center", fontsize=6,
                    color="white", zorder=8)
    ax.text(FRAC_START - 0.25, 0.024, "fractions", transform=trans,
            ha="right", va="center", fontsize=6.4, color=SUB)
    ax.text(FRAC_START + N_FRAC * FRAC_WIDTH + 0.25, 0.024,
            "pool {:d}-{:d}".format(pool[0], pool[-1]), transform=trans,
            ha="left", va="center", fontsize=6.4, color=OCHRE)

    ax.set_xlim(0.0, CV_END)
    ax.set_ylim(-90.0, max(uv) * 1.30)
    ax_s.set_ylim(0.0, 1550.0)
    ax.set_xlabel("retention volume (CV,  1 CV = {:.0f} mL)".format(
        COLUMN_ML))
    ax.set_ylabel(r"absorbance UV$_{280}$ (mAU)", color=BLUE)
    ax_s.set_ylabel("NaCl concentration (mM)", color=GOLD)
    ax.tick_params(axis="y", colors=BLUE)
    ax.spines["left"].set_color(BLUE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_s.spines["right"].set_color(GOLD)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["left"].set_visible(False)
    ax_s.tick_params(axis="y", colors=GOLD)
    ax.grid(axis="y", lw=0.5, alpha=0.5)

    handles = [Line2D([], [], color=BLUE, lw=1.5, label=r"UV$_{280}$ (mAU)"),
               Line2D([], [], color=GOLD, lw=1.4, ls=(0, (5.0, 2.2)),
                      label="NaCl gradient (mM)"),
               Line2D([], [], color=lighten(OCHRE, 0.72), lw=5.0,
                      label="target peak, pooled")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False,
              bbox_to_anchor=(0.020, 0.985), handlelength=1.7,
              borderaxespad=0.0, labelspacing=0.32)
    ax.set_title("Elution profile with the salt gradient that produced it",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    return met, rs, pool


def table_panel(ax, total, fold, yld):
    ax_y = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_y.set_zorder(1)
    pos = np.arange(len(STEPS), dtype=float)

    ax.plot(pos, fold, color=OCHRE, lw=1.7, drawstyle="steps-mid",
            marker="o", ms=4.6, mfc="white", mec=OCHRE, mew=1.2, zorder=5)
    ax_y.plot(pos, yld, color=GREEN, lw=1.7, drawstyle="steps-mid",
              marker="s", ms=4.2, mfc="white", mec=GREEN, mew=1.2, zorder=4)
    ax.text(pos[-1], fold[-1] * 1.35,
            r"$\times${:.0f} purification".format(fold[-1]), ha="right",
            va="bottom", fontsize=6.6, color=OCHRE)
    ax_y.text(pos[-1], yld[-1] - 6.0, "{:.0f} % step yield".format(yld[-1]),
              ha="right", va="top", fontsize=6.6, color=GREEN)
    ax.text(0.030, 0.955, "total protein {:.1f} g -> {:.0f} mg".format(
        total[0] / 1000.0, total[-1]), transform=ax.transAxes, ha="left",
        va="top", fontsize=6.5, color=SUB)

    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(STEPS) - 0.45)
    ax.set_ylim(0.55, fold[-1] * 4.2)
    ax_y.set_ylim(0.0, 118.0)
    ax.set_xticks(pos)
    ax.set_xticklabels([s.replace("\n", " ") for s in STEPS], fontsize=6.6,
                       rotation=18.0, ha="right")
    ax.set_ylabel("purification fold (-)", fontsize=7.6, color=OCHRE)
    ax_y.set_ylabel("overall yield (%)", fontsize=7.6, color=GREEN)
    ax.tick_params(axis="y", colors=OCHRE, labelsize=6.8)
    ax_y.tick_params(axis="y", colors=GREEN, labelsize=6.8)
    ax.spines["left"].set_color(OCHRE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_y.spines["right"].set_color(GREEN)
    ax_y.spines["top"].set_visible(False)
    ax_y.spines["left"].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    ax.set_title("Purity climbs, mass does not", fontsize=8.4,
                 fontweight="bold", color=INK)


def gel_panel(ax, m_t, m_c, total):
    rows, lane_px, gap_px = 320, 26, 8
    n_lane = len(STEPS) + 1
    hi, lo = np.log10(290.0), np.log10(8.5)
    grid = np.linspace(hi, lo, rows)
    img = np.zeros((rows, n_lane * (lane_px + gap_px)))

    def stamp(lane, mw, amp, sig=0.017):
        c0 = lane * (lane_px + gap_px) + gap_px // 2
        band = amp * np.exp(-0.5 * ((grid - np.log10(mw)) / sig) ** 2)
        img[:, c0:c0 + lane_px] += band[:, None]

    for mw in LADDER:
        stamp(0, mw, 0.62, sig=0.013)
    for k in range(len(STEPS)):
        stamp(k + 1, TARGET_MW, (m_t[k] / total[k]) ** 0.55)
        for j, (_n, mw, _mg, _r) in enumerate(CONTAM):
            stamp(k + 1, mw, (m_c[j, k] / total[k]) ** 0.55)

    ax.imshow(np.clip(img, 0.0, 1.0), cmap=get_cmap("cb_sequential_blue"),
              vmin=0.0, vmax=1.0, aspect="auto", origin="upper",
              extent=[0.0, float(n_lane), lo, hi], interpolation="bilinear",
              zorder=2)

    for mw in LADDER:
        ax.text(-0.10, np.log10(mw), "{:.0f}".format(mw), ha="right",
                va="center", fontsize=6, color=SUB)
    ax.text(-0.10, hi + 0.035, "kDa", ha="right", va="bottom", fontsize=6.0,
            color=SUB)
    ax.axhline(np.log10(TARGET_MW), color=OCHRE, lw=0.8, ls=(0, (3.0, 2.0)),
               zorder=4)
    ax.text(float(n_lane) + 0.06, np.log10(TARGET_MW),
            "target\n{:.0f} kDa".format(TARGET_MW), ha="left", va="center",
            fontsize=6.2, color=OCHRE, linespacing=1.25)

    ax.set_xticks(np.arange(n_lane) + 0.5)
    ax.set_xticklabels(["ladder"] + [s.replace("\n", " ") for s in STEPS],
                       fontsize=6.4, rotation=18.0, ha="right")
    ax.set_xlim(0.0, float(n_lane))
    ax.set_ylim(lo, hi)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("SDS-PAGE: contaminant bands thin out", fontsize=8.4,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(880)
    x, comps, uv, _base = chromatogram(rng)
    m_t, m_c, total, purity, fold, yld = mass_balance()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_tab = axes_inches(fig, TOP)
    ax_gel = axes_inches(fig, BOT)

    met, rs, pool = main_panel(ax_main, x, comps, uv, rng)
    table_panel(ax_tab, total, fold, yld)
    gel_panel(ax_gel, m_t, m_c, total)

    panel_labels_inches(fig, [ax_main, ax_tab, ax_gel], dx=-0.62, dy=0.14)
    savefig(fig, "880_bpr_chromatography.png")


if __name__ == "__main__":
    main()
