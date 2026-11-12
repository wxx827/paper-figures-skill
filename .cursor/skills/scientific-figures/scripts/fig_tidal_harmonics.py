"""Harmonic analysis of a 15-day water-level record.

Panel (a) overlays the observed water level on the tide predicted from eight
harmonic constituents and marks the spring and neap phases of the fortnightly
cycle. Panel (b) shares the time axis and shows the non-tidal residual, with a
storm surge highlighted. Panel (c) is a deliberately smaller companion panel
giving the fitted amplitude and Greenwich phase lag of each constituent, next to
a summary card for the fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"
C_WARM = "#d9a441"

MSL = 2.50
# name, amplitude (m), Greenwich phase (deg), speed (deg/h), species
CONST = [
    ("M2", 1.42, 128.0, 28.9841, "semidiurnal"),
    ("S2", 0.48, 213.0, 30.0000, "semidiurnal"),
    ("N2", 0.29, 108.0, 28.4397, "semidiurnal"),
    ("K2", 0.13, 168.0, 30.0821, "semidiurnal"),
    ("K1", 0.36, 315.0, 15.0411, "diurnal"),
    ("O1", 0.25, 288.0, 13.9430, "diurnal"),
    ("P1", 0.11, 310.0, 14.9589, "diurnal"),
    ("M4", 0.08, 40.0, 57.9682, "shallow water"),
]
SPECIES_COLOR = {"semidiurnal": C_DEEP, "diurnal": C_MAIN,
                 "shallow water": C_WARM}


def panel_tag(fig, ax, text, dx=-0.052, dy=0.012, size=10.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def main():
    set_pub_style()
    rng = np.random.default_rng(5)

    t = np.arange(0, 15 * 24 + 0.25, 0.25)          # hours
    pred = np.full_like(t, MSL)
    for _, a, g, w, _s in CONST:
        pred += a * np.cos(np.deg2rad(w * t - g))

    surge = (0.62 * np.exp(-0.5 * ((t - 206.0) / 13.0) ** 2)
             + 0.20 * np.exp(-0.5 * ((t - 62.0) / 20.0) ** 2)
             - 0.16 * np.exp(-0.5 * ((t - 300.0) / 22.0) ** 2))
    red = np.cumsum(rng.standard_normal(t.size)) * 0.010
    red -= np.linspace(red[0], red[-1], t.size)
    obs = pred + surge + red * 0.6 + 0.018 * rng.standard_normal(t.size)
    res = obs - pred
    day = t / 24.0

    fig = plt.figure(figsize=(9.8, 6.6))
    ax_a = fig.add_axes([0.068, 0.550, 0.905, 0.320])
    ax_b = fig.add_axes([0.068, 0.372, 0.905, 0.140], sharex=ax_a)
    ax_c = fig.add_axes([0.068, 0.098, 0.455, 0.170])
    ax_p = ax_c.twinx()

    # ---- (a) observed vs predicted ---------------------------------------
    ax_a.plot(day, obs, color=C_DEEP, lw=1.25, zorder=5,
              label="Observed water level")
    ax_a.plot(day, pred, color=C_ACC, lw=1.0, ls=(0, (4.5, 2.2)), zorder=6,
              label="Harmonic prediction (8 constituents)")
    ax_a.axhline(MSL, color="#9aa5ab", lw=0.7, ls=(0, (1.6, 2.0)), zorder=3)
    ax_a.text(14.92, MSL + 0.06, "mean sea level", ha="right", va="bottom",
              fontsize=7.0, color="#7d878d")

    win = np.arange(0, 15 * 24 - 25, 1.0)
    env = np.array([np.ptp(pred[(t >= h) & (t < h + 25)]) / 2.0 for h in win])
    hs = (win + 12.5) / 24.0
    i_sp, i_np = int(np.argmax(env)), int(np.argmin(env))
    for i, lab, col in ((i_sp, "spring tide", C_DEEP), (i_np, "neap tide",
                                                        C_MAIN)):
        ax_a.axvspan(hs[i] - 0.55, hs[i] + 0.55, color=col, alpha=0.07, lw=0,
                     zorder=1)
        ax_a.text(hs[i], 4.95, lab, ha="center", va="top", fontsize=7.4,
                  color=col, fontweight="bold")
        ax_a.annotate("", xy=(hs[i], MSL + env[i]),
                      xytext=(hs[i], MSL - env[i]),
                      arrowprops=dict(arrowstyle="<|-|>", color=col, lw=0.8,
                                      mutation_scale=6, shrinkA=0, shrinkB=0),
                      zorder=7)
        ax_a.text(hs[i] + 0.16, MSL, f"{2 * env[i]:.2f} m", ha="left",
                  va="center", fontsize=7.0, color=col, zorder=8,
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.85))

    ax_a.set_ylim(0.05, 5.05)
    ax_a.set_yticks([1, 2, 3, 4])
    ax_a.set_ylabel("Water level above chart datum (m)")
    ax_a.grid(axis="y", color="#eeeeee", lw=0.5)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(labelbottom=False)
    style_axes(ax_a)
    leg = ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
                      fontsize=7.8, handlelength=2.0, columnspacing=1.8)
    leg.set_clip_on(False)

    # ---- (b) residual -----------------------------------------------------
    ax_b.axhline(0, color="#9aa5ab", lw=0.7, zorder=3)
    ax_b.fill_between(day, 0, res, color=C_MAIN, alpha=0.28, lw=0, zorder=2)
    ax_b.fill_between(day, 0, res, where=res > 0.30, color=C_ACC, alpha=0.35,
                      lw=0, zorder=3, interpolate=True)
    ax_b.plot(day, res, color=C_DEEP, lw=1.0, zorder=5)

    ipk = int(np.argmax(res))
    ax_b.annotate(f"storm surge  +{res[ipk]:.2f} m",
                  xy=(day[ipk], res[ipk]), xytext=(day[ipk] + 0.85, 0.66),
                  fontsize=7.6, color=C_ACC, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", color=C_ACC, lw=0.8,
                                  connectionstyle="arc3,rad=-0.25"))
    ax_b.set_ylim(-0.48, 0.88)
    ax_b.set_yticks([-0.4, 0, 0.4, 0.8])
    ax_b.set_ylabel("Residual\n(m)", linespacing=1.35)
    ax_b.set_xlim(0, 15)
    ax_b.set_xticks(np.arange(0, 15.1, 1))
    ax_b.set_xlabel("Days from 01 March (hourly record)")
    ax_b.grid(axis="y", color="#eeeeee", lw=0.5)
    ax_b.set_axisbelow(True)
    style_axes(ax_b)

    rms = float(np.sqrt(np.mean(res ** 2)))
    ax_b.text(0.995, 0.93, f"RMS residual {rms:.3f} m",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=7.2,
              color="#555555")

    # ---- (c) constituents (subordinate) -----------------------------------
    names = [c[0] for c in CONST]
    amps = np.array([c[1] for c in CONST])
    phas = np.array([c[2] for c in CONST])
    cols = [SPECIES_COLOR[c[4]] for c in CONST]
    xs = np.arange(len(CONST))

    ax_c.bar(xs, amps, width=0.62, color=cols, edgecolor="white",
             linewidth=0.5, zorder=3)
    for x, a in zip(xs, amps):
        ax_c.text(x, a + 0.045, f"{a:.2f}", ha="center", va="bottom",
                  fontsize=6.6, color="#4a4a4a", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                            edgecolor="none", alpha=0.8))
    ax_c.set_ylim(0, 1.72)
    ax_c.set_yticks([0, 0.5, 1.0, 1.5])
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(names, fontsize=7.6)
    ax_c.set_xlim(-0.7, len(CONST) - 0.3)
    ax_c.set_ylabel("Amplitude (m)", fontsize=7.8)
    ax_c.tick_params(labelsize=7.2, length=2.5)
    ax_c.grid(axis="y", color="#ececec", lw=0.5)
    ax_c.set_axisbelow(True)
    ax_c.spines["top"].set_visible(False)

    ax_p.plot(xs, phas, ls="none", marker="D", ms=4.0, mfc="white",
              mec=C_ACC, mew=1.2, zorder=5)
    ax_p.set_ylim(-25, 415)
    ax_p.set_yticks([0, 90, 180, 270, 360])
    ax_p.set_ylabel("Phase lag (\u00b0)", fontsize=7.8, color=C_ACC)
    ax_p.tick_params(labelsize=7.2, length=2.5, colors=C_ACC)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(True)
    ax_p.spines["right"].set_color(C_ACC)
    ax_p.spines["right"].set_linewidth(0.7)

    handles = [Patch(facecolor=SPECIES_COLOR[k], edgecolor="white",
                     linewidth=0.5) for k in SPECIES_COLOR]
    handles.append(plt.Line2D([], [], ls="none", marker="D", ms=4.0,
                              mfc="white", mec=C_ACC, mew=1.2))
    labels = list(SPECIES_COLOR) + ["phase lag"]
    leg_c = ax_c.legend(handles, labels, loc="upper center",
                        bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=7.0,
                        handlelength=1.1, handleheight=0.85,
                        columnspacing=1.2, handletextpad=0.45)
    leg_c.set_clip_on(False)

    # ---- summary card -----------------------------------------------------
    cx0, cy0, cw, ch = 0.588, 0.062, 0.385, 0.224
    card = FancyBboxPatch((cx0, cy0), cw, ch, transform=fig.transFigure,
                          boxstyle="round,pad=0.004,rounding_size=0.010",
                          facecolor="#f2f5f6", edgecolor="#dde3e6",
                          linewidth=0.7, zorder=-5)
    fig.patches.append(card)

    f_form = (0.36 + 0.25) / (1.42 + 0.48)
    var = 100.0 * (1.0 - np.var(res) / np.var(obs))
    fig.text(cx0 + 0.022, cy0 + ch - 0.030, "Harmonic fit summary",
             fontsize=8.2, fontweight="bold", ha="left", va="top",
             color="#1a1a1a")
    left = ("Record length\nSampling\nConstituents fitted\n"
            "Variance explained\nRMS residual\nForm factor  F")
    right = (f"15 days\nhourly\n{len(CONST)}\n{var:.1f} %\n{rms:.3f} m\n"
             f"{f_form:.2f}  (semidiurnal)")
    fig.text(cx0 + 0.022, cy0 + ch - 0.068, left, fontsize=7.4,
             ha="left", va="top", color="#555555", linespacing=1.72)
    fig.text(cx0 + 0.205, cy0 + ch - 0.068, right, fontsize=7.4,
             ha="left", va="top", color="#1a1a1a", linespacing=1.72)


    for a, lab in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        panel_tag(fig, a, lab)
    savefig(fig, "218_tidal_harmonics.png")


if __name__ == "__main__":
    main()
