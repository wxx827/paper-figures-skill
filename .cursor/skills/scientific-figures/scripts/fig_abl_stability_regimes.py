"""Three states of the atmospheric boundary layer, told by potential temperature.

The main panel plots idealised potential-temperature soundings for a stable
night, the morning transition and a fully convective afternoon, with height
on the y axis.  The layers each state builds - surface layer, mixed layer,
entrainment zone, residual layer, stable boundary layer and the free
atmosphere the three share aloft - are marked on the curves they belong to.

The inversion heights annotated on the soundings are the same numbers that
drive the diurnal cycle in the first companion panel: one function supplies
both, so the snapshots cannot drift away from the cycle they are taken from.
The second companion panel shows how often each regime occurs, through the
surface-layer stability parameter derived from the Obukhov length.
"""
import textwrap

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
GRID = "#d9d9d9"

COOL = "#3d6b8e"     # stable
MID = "#c9a227"      # neutral / transition
WARM = "#b23a48"     # unstable

Z_TOP = 2000.0
TH_RESID = 292.2     # residual layer potential temperature (K)
TH_NEUTRAL = 291.0
TH_MIXED = 294.5
TH_SFC_NIGHT = 286.0
FA_REF, FA_Z, FA_LAPSE = 296.5, 1600.0, 0.0035    # free atmosphere (K, m, K/m)
Z_REF = 10.0         # reference height for z / L (m)
SNAPS = [(5.0, "Stable", "05:00", COOL),
         (8.5, "Neutral", "08:30", MID),
         (14.0, "Unstable", "14:00", WARM)]


def zi_of_t(t):
    """Boundary-layer depth (m) through the day; the single source of z_i."""
    grow = 1.0 / (1.0 + np.exp(-(t - 9.2) / 1.25))
    fall = 1.0 / (1.0 + np.exp(-(18.3 - t) / 0.55))
    return 190.0 + 1250.0 * grow * fall


def free_atm(z):
    return FA_REF + FA_LAPSE * (z - FA_Z)


def blend(z, z0, w):
    return 0.5 * (1.0 + np.tanh((z - z0) / w))


def soundings(z, zi):
    """Potential temperature (K) for the three regimes at their own z_i."""
    fa = free_atm(z)
    out = {}

    h = zi["Stable"] / 2.6
    th = TH_SFC_NIGHT + (TH_RESID - TH_SFC_NIGHT) * (1.0 - np.exp(-z / h))
    out["Stable"] = th * (1 - blend(z, 1400.0, 70.0)) + fa * blend(z, 1400.0, 70.0)

    th = TH_NEUTRAL + 0.30 * np.exp(-z / 30.0)
    b = blend(z, zi["Neutral"], 45.0)
    th = th * (1 - b) + TH_RESID * b
    out["Neutral"] = th * (1 - blend(z, 1400.0, 70.0)) + fa * blend(z, 1400.0, 70.0)

    th = TH_MIXED + 2.6 * np.exp(-z / 45.0)
    b = blend(z, zi["Unstable"] + 90.0, 62.0)
    out["Unstable"] = th * (1 - b) + fa * b
    return out


def stability_samples(rng, n=4000):
    """Surface-layer z / L, mixing night-time stable and daytime unstable hours."""
    kind = rng.choice(3, size=n, p=[0.40, 0.19, 0.41])
    z_l = np.where(kind == 0, -np.exp(rng.normal(-1.15, 0.95, n)),
                   np.where(kind == 1, rng.normal(0.0, 0.030, n),
                            np.exp(rng.normal(-1.35, 1.05, n))))
    return z_l[np.abs(z_l) <= 2.0]     # drop the tails, never pile them on the edge


def main():
    set_pub_style()
    rng = np.random.default_rng(718)

    z = np.linspace(0.0, Z_TOP, 900)
    zi = {name: float(zi_of_t(t)) for t, name, _, _ in SNAPS}
    prof = soundings(z, zi)
    for k in prof:                       # radiosonde-like jitter
        prof[k] = prof[k] + gaussian_filter1d(rng.normal(0.0, 1.0, z.size), 7) * 0.16

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.88, 0.82, 5.02, 4.68))
    ax1 = axes_inches(fig, (6.72, 3.30, 3.00, 2.20))
    ax2 = axes_inches(fig, (6.72, 0.82, 3.00, 1.95))

    # ---- main panel: three soundings --------------------------------------
    thlo, thhi = 284.6, 299.4
    ax.axhspan(zi["Unstable"], zi["Unstable"] + 190.0, facecolor="#f0f0f0",
               edgecolor="none", zorder=0)
    ax.axhline(1500.0, color=GREY, lw=0.7, ls=(0, (2, 2)), alpha=0.7, zorder=1)
    ax.text(thhi - 0.25, 1760.0, "free atmosphere, "
            f"$\\partial\\theta/\\partial z$ = {FA_LAPSE * 1000:.1f} K/km",
            fontsize=6.8, color=SUB, ha="right", va="bottom")
    ax.axvline(TH_MIXED, color=GREY, lw=0.8, ls=(0, (5, 2)), alpha=0.8,
               zorder=1)

    for t, name, clock, col in SNAPS:
        ax.plot(prof[name], z, color=col, lw=1.8, zorder=4,
                label=f"{name}, {clock}")
        h = zi[name]
        th_h = float(np.interp(h, z, prof[name]))
        ax.plot([th_h - 0.45, th_h + 0.45], [h, h], color=col, lw=1.1,
                zorder=5)
        ax.text(th_h + 0.60, h, f"$z_i$ = {h:.0f} m", fontsize=6.6, color=col,
                ha="left", va="center", zorder=6)

    ax.text(TH_MIXED - 0.35, 780.0, "mixed layer", fontsize=6.8, color=WARM,
            ha="right", va="center", rotation=90)
    ax.text(thhi - 0.25, 120.0, "surface layer,\nsuperadiabatic",
            fontsize=6.8, color=WARM, ha="right", va="center",
            linespacing=1.3)
    ax.text(thlo + 0.25, zi["Unstable"] + 95.0, "entrainment zone",
            fontsize=6.8, color=SUB, ha="left", va="center")
    ax.text(TH_RESID + 0.25, 1030.0, "residual layer", fontsize=6.8,
            color=COOL, ha="left", va="center")
    ax.text(TH_SFC_NIGHT + 0.25, 95.0, "stable boundary layer", fontsize=6.8,
            color=COOL, ha="left", va="center")
    ax.text(TH_NEUTRAL - 0.25, 330.0, "growing\nmixed layer", fontsize=6.8,
            color=MID, ha="right", va="center", linespacing=1.3)

    ax.set_xlim(thlo, thhi)
    ax.set_ylim(0.0, Z_TOP)
    ax.set_xlabel("Potential temperature $\\theta$ (K)", fontsize=8.0)
    ax.set_ylabel("Height above ground $z$ (m)", fontsize=8.0)
    ax.set_title("Idealised soundings for the three stability regimes",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    ax.grid(True, color=GRID, lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)

    # ---- aux 1: diurnal cycle of the boundary-layer depth ------------------
    tg = np.linspace(0.0, 24.0, 600)
    zg = zi_of_t(tg)
    ax1.plot(tg, zg, color=INK, lw=1.5, zorder=4)
    ax1.fill_between(tg, zg, color=MID, alpha=0.14, lw=0.0, zorder=2)
    night = (tg < 7.0) | (tg > 19.0)
    ax1.fill_between(tg, 0.0, Z_TOP, where=night, facecolor="#f0f0f0",
                     edgecolor="none", zorder=0)
    ax1.axhline(1400.0, color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax1.text(0.4, 1450.0, "residual layer top", fontsize=6.5, color=SUB,
             ha="left", va="bottom")
    for t, name, clock, col in SNAPS:
        ax1.plot([t], [zi[name]], marker="o", ms=4.2, mfc=col, mec="white",
                 mew=0.8, zorder=6)
        ax1.annotate(f"{name}\n{zi[name]:.0f} m", xy=(t, zi[name]),
                     xytext=(t + 0.7, zi[name] + 260.0), fontsize=6.4,
                     color=col, ha="left", va="bottom", linespacing=1.25,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
    ax1.set_xlim(0.0, 24.0)
    ax1.set_ylim(0.0, Z_TOP)
    ax1.set_xticks(np.arange(0, 25, 6))
    ax1.set_xlabel("Local time (h)", fontsize=7.2)
    ax1.set_ylabel("Depth $z_i$ (m)", fontsize=7.2)
    ax1.set_title("Diurnal growth and collapse", fontsize=8.5,
                  fontweight="bold")
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)

    # ---- aux 2: distribution of the stability parameter --------------------
    zl = stability_samples(rng)
    thr = 0.10
    edges = np.linspace(-2.0, 2.0, 49)
    cnt, _ = np.histogram(zl, bins=edges)
    share = 100.0 * cnt / cnt.sum()
    mids = 0.5 * (edges[:-1] + edges[1:])
    cols = [WARM if m < -thr else (COOL if m > thr else MID) for m in mids]
    ax2.bar(mids, share, width=(edges[1] - edges[0]) * 0.92, color=cols,
            edgecolor="white", linewidth=0.25, zorder=3)
    for x in (-thr, thr):
        ax2.axvline(x, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=4)

    f_un = 100.0 * np.mean(zl < -thr)
    f_ne = 100.0 * np.mean(np.abs(zl) <= thr)
    f_st = 100.0 * np.mean(zl > thr)
    l_un = Z_REF / np.median(np.abs(zl[zl < -thr]))
    l_st = Z_REF / np.median(zl[zl > thr])
    top = float(share.max())
    ax2.text(-1.95, top * 0.96, f"unstable {f_un:.0f} %\nmedian $L$ = "
             f"-{l_un:.0f} m", fontsize=6.4, color=WARM, ha="left", va="top",
             linespacing=1.3)
    ax2.text(1.95, top * 0.96, f"stable {f_st:.0f} %\nmedian $L$ = "
             f"{l_st:.0f} m", fontsize=6.4, color=COOL, ha="right", va="top",
             linespacing=1.3)
    ax2.text(0.0, top * 1.10, f"near neutral {f_ne:.0f} %", fontsize=6.4,
             color=MID, ha="center", va="bottom")
    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(0.0, top * 1.30)
    ax2.set_xlabel("Stability parameter $z/L$ (-) at $z$ = 10 m",
                   fontsize=7.2)
    ax2.set_ylabel("Share of hours (%)", fontsize=7.2)
    ax2.set_title("How often each regime occurs", fontsize=8.5,
                  fontweight="bold")
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.50, dy=0.10)


    savefig(fig, "718_abl_stability_regimes.png")


if __name__ == "__main__":
    main()
