"""Diurnal evolution of boundary-layer vertical profiles.

Three panels share an inverted logarithmic pressure axis covering the lowest
three kilometres and show temperature, specific humidity and wind speed at four
local times. Shaded envelopes give the +-1 sigma spread of the sounding
ensemble. The nocturnal surface inversion, the daytime mixed layer and its
capping inversion, the residual layer and the nocturnal low-level jet are
annotated. Geometric height is mirrored on the right-hand axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes

PBOT, PTOP = 1000.0, 690.0
PSFC = 1013.25

TIMES = [
    dict(label="00 LST", color="#26385c", asfc=-7.0, hsfc=0.17, zi=0.26,
         cap=0.0, q0=11.4, llj=8.5, lz=0.32, geo=10.0),
    dict(label="06 LST", color="#3d8b8b", asfc=-4.2, hsfc=0.22, zi=0.40,
         cap=0.0, q0=10.9, llj=5.0, lz=0.44, geo=9.5),
    dict(label="12 LST", color="#dd9038", asfc=7.8, hsfc=0.70, zi=1.75,
         cap=2.3, q0=13.6, llj=0.0, lz=0.6, geo=8.6),
    dict(label="18 LST", color="#a8432f", asfc=2.6, hsfc=0.50, zi=1.15,
         cap=1.6, q0=12.7, llj=1.2, lz=0.5, geo=9.1),
]


def p_to_z(p):
    return 44.331 * (1.0 - (p / PSFC) ** 0.1903)


def z_to_p(z):
    return PSFC * (1.0 - z / 44.331) ** (1.0 / 0.1903)


def profiles(z, tm):
    """Temperature (degC), specific humidity (g/kg) and wind speed (m/s)."""
    t = 288.15 - 6.5 * z + tm["asfc"] * np.exp(-z / tm["hsfc"])
    # capping inversion, relaxing back to the free-tropospheric lapse rate
    t += (tm["cap"] / (1.0 + np.exp(-(z - tm["zi"]) / 0.09))
          * np.exp(-np.maximum(z - tm["zi"], 0.0) / 0.75))
    t -= 273.15

    blend = 1.0 / (1.0 + np.exp((z - tm["zi"]) / 0.16))
    free = tm["q0"] * 0.88 * np.exp(-(z - tm["zi"]) / 1.35)
    q = blend * tm["q0"] + (1 - blend) * free

    ws = (2.6 + tm["geo"] * np.tanh(z / 1.15)
          + tm["llj"] * np.exp(-((z - tm["lz"]) / 0.17) ** 2))
    return t, q, ws


def main():
    set_pub_style()
    p = np.logspace(np.log10(PBOT), np.log10(PTOP), 400)
    z = p_to_z(p)

    fig = plt.figure(figsize=(10.2, 5.9))
    gs = fig.add_gridspec(1, 3, left=0.062, right=0.905, top=0.845,
                          bottom=0.165, wspace=0.10)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1], sharey=ax_t)
    ax_w = fig.add_subplot(gs[0, 2], sharey=ax_t)
    axes = [ax_t, ax_q, ax_w]

    store = {}
    for tm in TIMES:
        t, q, ws = profiles(z, tm)
        store[tm["label"]] = (t, q, ws)
        sig = [0.38 + 0.75 * np.exp(-z / 0.9),
               0.11 * q + 0.06,
               0.42 + 1.05 * np.exp(-z / 0.8)]
        for ax, val, s in zip(axes, (t, q, ws), sig):
            ax.fill_betweenx(p, val - s, val + s, color=tm["color"],
                             alpha=0.14, lw=0, zorder=2)
            ax.plot(val, p, color=tm["color"], lw=1.9, zorder=3,
                    label=tm["label"], solid_capstyle="round")

    yticks = np.arange(1000, 689, -50)
    for ax, lab, tag in zip(axes,
                            ["Temperature ($\\degree$C)",
                             "Specific humidity (g kg$^{-1}$)",
                             "Wind speed (m s$^{-1}$)"],
                            ["(a)", "(b)", "(c)"]):
        ax.set_yscale("log")
        ax.set_ylim(PBOT, PTOP)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.0f}" for v in yticks])
        ax.minorticks_off()
        ax.set_xlabel(lab)
        style_axes(ax)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.025, tag, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
    ax_t.set_ylabel("Pressure (hPa)")
    for ax in (ax_q, ax_w):
        ax.tick_params(labelleft=False)

    ax_t.set_xlim(-9, 30)
    ax_q.set_xlim(1.0, 15.5)
    ax_w.set_xlim(0, 17.5)

    # ---- feature annotations ---------------------------------------------
    p_inv = z_to_p(TIMES[0]["zi"])
    p_cap = z_to_p(TIMES[2]["zi"])
    p_res = z_to_p(TIMES[3]["zi"])

    for ax in axes:
        ax.axhspan(PBOT, p_inv, color="#26385c", alpha=0.07, lw=0, zorder=1)
        ax.axhline(p_cap, color="#dd9038", lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax_t.text(0.025, np.sqrt(PBOT * p_inv), "nocturnal inversion",
              transform=ax_t.get_yaxis_transform(), ha="left", va="center",
              fontsize=7, color="#26385c", zorder=6)
    ax_t.text(0.025, p_cap * 1.004, "capping\ninversion",
              transform=ax_t.get_yaxis_transform(), ha="left", va="top",
              fontsize=7, color="#a86a1e", fontweight="bold",
              linespacing=1.25, zorder=6)

    ax_q.axhspan(p_res, p_cap, color="#8a8a8a", alpha=0.09, lw=0, zorder=1)
    ax_q.text(0.03, np.sqrt(p_res * p_cap), "residual\nlayer",
              transform=ax_q.get_yaxis_transform(), ha="left", va="center",
              fontsize=7, color="#666666", linespacing=1.25, zorder=6)
    ax_q.text(0.03, np.sqrt(p_cap * PBOT), "well-mixed\nlayer",
              transform=ax_q.get_yaxis_transform(), ha="left", va="center",
              fontsize=7, color="#a86a1e", linespacing=1.25, zorder=6)
    ax_q.annotate("entrainment drying", xy=(6.2, p_cap * 0.995),
                  xytext=(9.0, 760), fontsize=7, color="#555555",
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8,
                                  connectionstyle="arc3,rad=0.25"))

    ws0 = store["00 LST"][2]
    i = int(np.argmax(ws0))
    ax_w.plot([ws0[i]], [p[i]], marker="o", ms=4.5, mfc="white",
              mec="#26385c", mew=1.2, zorder=6)
    ax_w.annotate(f"nocturnal low-level jet\n{ws0[i]:.1f} m s$^{{-1}}$ at "
                  f"{p_to_z(p[i]) * 1000:.0f} m",
                  xy=(ws0[i], p[i]), xytext=(0.5, 928),
                  fontsize=7, color="#26385c", ha="left", va="center",
                  linespacing=1.25,
                  arrowprops=dict(arrowstyle="->", color="#26385c", lw=0.8,
                                  connectionstyle="arc3,rad=-0.28"))
    ax_w.text(0.03, 730, "free-atmosphere\ngeostrophic flow",
              transform=ax_w.get_yaxis_transform(), ha="left", va="center",
              fontsize=7, color="#555555", linespacing=1.25)

    # ---- mirrored geometric-height axis ----------------------------------
    ax_h = ax_w.twinx()
    ax_h.set_yscale("log")
    ax_h.set_ylim(PBOT, PTOP)
    zt = np.arange(0.0, 3.01, 0.5)
    zt = zt[(z_to_p(zt) <= PBOT) & (z_to_p(zt) >= PTOP)]
    ax_h.set_yticks(z_to_p(zt))
    ax_h.set_yticklabels([f"{v:g}" for v in zt])
    ax_h.minorticks_off()
    ax_h.set_ylabel("Approximate geometric height (km)", fontsize=8)
    ax_h.spines["right"].set_visible(True)
    ax_h.spines["right"].set_color("#333333")
    ax_h.spines["top"].set_visible(False)
    ax_h.tick_params(length=3)

    handles, labels = ax_t.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="center",
                     bbox_to_anchor=(0.48, 0.055), ncol=4, frameon=False,
                     fontsize=8.2, handlelength=1.9, columnspacing=2.2,
                     title="Local solar time of sounding")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_fontweight("bold")

    savefig(fig, "84_vertical_profile.png")


if __name__ == "__main__":
    main()
