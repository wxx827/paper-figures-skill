"""What actually removes the view: an extinction budget through a haze episode.

The main panel stacks the light-extinction coefficient by chemical component
across five days that start clear and end in dense haze, with the visual range
implied by the very same total drawn over it on a logarithmic axis.  Stacking
is the honest way to draw this because extinction is additive: the components
are summed to give the total, the total is inverted through the Koschmieder
relation to give the visual range, and no number on the panel is quoted from
anywhere else.

The two companion panels are the reasons the budget moves.  The hygroscopic
growth curves show why a humid night can double the extinction without a
single extra microgram of aerosol, and the two stacked bars compare the
composition of the extinction on the clearest and haziest hours of the
episode, each summing to one hundred per cent.

Illustrative synthetic episode built with IMPROVE-style extinction
efficiencies; not a measurement record.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
RED = "#b23a48"
BROWN = "#8c5a3c"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.68, 1.00, 5.45, 4.22)
AUX_T_RECT = (7.52, 3.42, 3.16, 1.80)
AUX_B_RECT = (7.52, 1.00, 3.16, 1.80)

N_HOUR = 120                 # a five-day episode, hourly
RAYLEIGH = 10.0              # Mm^-1, clean-air Rayleigh scattering
KOSCH = 3912.0               # km Mm^-1, Koschmieder constant

# component: (label, dry extinction efficiency m^2/g, hygroscopic, colour)
COMPS = [
    ("ammonium sulphate", 3.0, True, BLUE),
    ("ammonium nitrate", 3.0, True, TEAL),
    ("organic matter", 4.0, False, GOLD),
    ("elemental carbon", 10.0, False, "#3a3a3a"),
    ("fine soil", 1.0, False, BROWN),
    ("coarse mass", 0.6, False, "#d9c4a0"),
    ("Rayleigh air", 0.0, False, "#cfd6dc"),
]


def f_rh(rh, power):
    """Hygroscopic growth factor of the scattering efficiency, dimensionless."""
    return (1.0 - np.clip(rh, 0.0, 95.0) / 100.0) ** (-power)


def synth_episode(rng):
    """Hourly species concentrations and relative humidity, one episode."""
    t = np.arange(N_HOUR, dtype=float)
    build = 1.0 / (1.0 + np.exp(-(t - 62.0) / 13.0))
    diurnal = 0.16 * np.cos(2.0 * np.pi * (t - 6.0) / 24.0)
    noise = rng.normal(0.0, 0.055, N_HOUR)
    scale = np.exp(np.log(1.0 + 8.0 * build) + diurnal + noise)

    conc = {
        "ammonium sulphate": 2.1 * scale,
        "ammonium nitrate": 1.4 * scale ** 1.18,
        "organic matter": 4.2 * scale ** 0.86,
        "elemental carbon": 0.85 * scale ** 0.62,
        "fine soil": 1.4 + 0.8 * build,
        "coarse mass": 7.5 + 9.0 * build,
    }
    rh = np.clip(58.0 + 22.0 * build
                 - 15.0 * np.cos(2.0 * np.pi * (t - 4.0) / 24.0)
                 + rng.normal(0.0, 2.2, N_HOUR), 26.0, 94.0)
    return t, conc, rh


def budget(conc, rh):
    """Per-component extinction, Mm^-1; the stack that sums to the total."""
    out = {}
    for name, eff, hyg, _ in COMPS:
        if name == "Rayleigh air":
            out[name] = np.full_like(rh, RAYLEIGH)
        else:
            g = f_rh(rh, 0.60 if name.endswith("sulphate") else 0.68) \
                if hyg else 1.0
            out[name] = eff * g * conc[name]
    return out


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1011)
    t, conc, rh = synth_episode(rng)
    parts = budget(conc, rh)
    names = [c[0] for c in COMPS]
    stack = np.vstack([parts[n] for n in names])
    b_ext = stack.sum(axis=0)
    vr = KOSCH / b_ext
    assert np.allclose(stack.sum(axis=0), b_ext)

    i_hi = int(np.argmax(b_ext))
    i_lo = int(np.argmin(b_ext))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_c, ax_d):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the stacked budget and the view it leaves --------------------
    ax.stackplot(t, stack, colors=[c[3] for c in COMPS], labels=names,
                 edgecolor="white", linewidth=0.35, zorder=3)
    ax.plot(t, b_ext, color=INK, lw=1.2, zorder=6)
    ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax.set_xlim(0.0, N_HOUR - 1.0)
    ax.set_ylim(0.0, b_ext.max() * 1.30)
    ax.set_xticks(np.arange(0.0, N_HOUR, 24.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("hours from the start of the episode (h)", fontsize=8.4,
                  labelpad=1.5)
    ax.set_ylabel("light-extinction coefficient ($Mm^{-1}$)", fontsize=8.4)
    ax.set_title("Extinction budget through a five-day haze episode",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.legend(loc="upper left", frameon=False, fontsize=6.4, ncol=4,
              handlelength=1.1, handleheight=0.9, columnspacing=0.9,
              labelspacing=0.22, borderaxespad=0.25)

    ax_v = ax.twinx()
    ax_v.plot(t, vr, color=TEAL, lw=1.7, ls=(0, (4.6, 2.0)), zorder=7)
    ax_v.set_yscale("log")
    ax_v.set_ylim(4.0, 400.0)
    ax_v.set_yticks([5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    ax_v.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax_v.minorticks_off()
    ax_v.set_ylabel("visual range (km, log scale)", fontsize=8.4, color=TEAL)
    ax_v.tick_params(labelsize=7.0, labelcolor=TEAL)
    ax_v.spines["top"].set_visible(False)
    ax_v.grid(False)
    ax_v.plot([t[i_hi]], [vr[i_hi]], marker="o", ms=4.8, mfc="white",
              mec=RED, mew=1.4, zorder=9)
    ax_v.annotate(f"worst hour: {b_ext[i_hi]:.0f} $Mm^{{-1}}$, "
                  f"{vr[i_hi]:.1f} km",
                  xy=(t[i_hi], vr[i_hi]), xytext=(t[i_hi] - 4.0, 30.0),
                  ha="right", va="center", fontsize=6.6, color=RED,
                  arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                  shrinkA=1, shrinkB=3), zorder=9,
                  bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                            edgecolor="#e0c8cb", linewidth=0.6, alpha=0.94))
    ax.text(0.012, 0.020,
            "stack sums to the black total by construction;\nthe dashed "
            f"curve is {KOSCH:.0f} / $b_{{ext}}$ in km\nhumidity climbs "
            f"{rh.min():.0f} % to {rh.max():.0f} %, lifting the two\n"
            "hygroscopic components by a factor of "
            f"{f_rh(rh.max(), 0.6) / f_rh(rh.min(), 0.6):.1f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=12,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))

    # ---- aux (b): why humidity matters --------------------------------------
    rhs = np.linspace(20.0, 95.0, 300)
    ax_c.plot(rhs, f_rh(rhs, 0.60), color=BLUE, lw=1.8, zorder=6)
    ax_c.plot(rhs, f_rh(rhs, 0.68), color=TEAL, lw=1.8, ls=(0, (4.4, 2.0)),
              zorder=6)
    ax_c.axvspan(rh.min(), rh.max(), color="#eef2f6", zorder=1)
    ax_c.axhline(1.0, color=FAINT, lw=0.8, ls=(0, (3.0, 2.2)), zorder=2)
    for i, col in ((i_lo, BLUE), (i_hi, RED)):
        ax_c.plot([rh[i]], [f_rh(rh[i], 0.60)], marker="o", ms=4.4,
                  mfc="white", mec=col, mew=1.3, zorder=8)
    ax_c.set_xlim(20.0, 95.0)
    ax_c.set_ylim(0.8, 7.5)
    ax_c.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("relative humidity (%)", fontsize=7.5, labelpad=1.5)
    ax_c.set_ylabel("growth factor f(RH) (dimensionless)", fontsize=7.5)
    ax_c.set_title("Water does most of the work", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.text(0.030, 0.955,
              f"shaded band is the range the episode visits;\ncircles are its "
              f"clearest ({rh[i_lo]:.0f} %) and haziest "
              f"({rh[i_hi]:.0f} %) hours",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45)
    ax_c.legend(handles=[Line2D([], [], color=BLUE, lw=1.8, label="sulphate"),
                         Line2D([], [], color=TEAL, lw=1.8,
                                ls=(0, (4.4, 2.0)), label="nitrate")],
                loc="center left", frameon=False, fontsize=6.3,
                handlelength=1.7, labelspacing=0.26, borderaxespad=0.55)

    # ---- aux (c): composition of the extinction, clear versus hazy ----------
    labels = [f"clearest hour\n{b_ext[i_lo]:.0f} $Mm^{{-1}}$, "
              f"{vr[i_lo]:.0f} km",
              f"haziest hour\n{b_ext[i_hi]:.0f} $Mm^{{-1}}$, "
              f"{vr[i_hi]:.1f} km"]
    shares = np.vstack([100.0 * stack[:, i] / b_ext[i] for i in (i_lo, i_hi)])
    left = np.zeros(2)
    for j, (name, _, _, col) in enumerate(COMPS):
        ax_d.barh([0, 1], shares[:, j], left=left, height=0.52, color=col,
                  edgecolor="white", linewidth=0.6, zorder=4)
        for r in (0, 1):
            if shares[r, j] >= 9.0:
                ax_d.text(left[r] + shares[r, j] / 2.0, r,
                          f"{shares[r, j]:.0f}", ha="center", va="center",
                          fontsize=6.2, color="white"
                          if col not in ("#d9c4a0", "#cfd6dc", GOLD) else INK,
                          zorder=6)
        left += shares[:, j]
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(labels, fontsize=6.6)
    ax_d.set_xlim(0.0, 100.0)
    ax_d.set_ylim(-0.62, 1.75)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("share of the total extinction (%)", fontsize=7.5,
                    labelpad=1.5)
    ax_d.set_title("The same aerosol, a different mixture", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_d.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_d.text(0.5, 0.985,
              f"each bar sums to {left[0]:.0f} %; segments below 9 % are left "
              "unlabelled\ncolours match the stack in panel (a)",
              transform=ax_d.transAxes, ha="center", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_c, ax_d], dx=-0.56, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=1011")
    check_layout(fig, foot, [ax, ax_c, ax_d])
    savefig(fig, "1011_aqm_visibility_extinction.png")


if __name__ == "__main__":
    main()
