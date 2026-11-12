"""Acoustic camera image of an engine bay in one third-octave band.

Left: the beamforming map laid over a schematic of the hardware, drawn with a
12 dB display range so only the sources that matter survive, with the spiral
array geometry inset.  Right: the band the map was computed in, and the
measurement parameters the array diameter and stand-off distance imply.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from pubstyle import set_pub_style, savefig, style_axes, COLORS
from figkit import safe_colorbar, text_card, use_inch_units, halo_text

W, H = 1.60, 1.15                  # field of view (m)
STANDOFF, ARRAY_D, N_MIC = 1.20, 0.95, 36
C0 = 343.0
BANDS = np.array([250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                  3150, 4000, 5000, 6300, 8000], dtype=float)
F_SEL = 4000.0
DYN_RANGE = 12.0
# (x, y, source level dB, label)
SOURCES = [(0.56, 0.63, 92.4, "belt drive"), (1.18, 0.30, 85.1, "manifold"),
           (1.22, 0.82, 82.6, "intake")]


def resolution(f):
    """Rayleigh resolution of the array at the stand-off distance (m)."""
    return 1.22 * (C0 / f) * STANDOFF / ARRAY_D


def beam_map(X, Y, f):
    """Beamforming output level (dB) over the image plane."""
    s = resolution(f) / 2.0
    p = np.full(X.shape, 10.0 ** (6.2))
    for sx, sy, lvl, _ in SOURCES:
        d = np.hypot(X - sx, Y - sy)
        psf = (np.exp(-0.5 * (d / s) ** 2)
               + 0.09 * np.exp(-0.5 * ((d - 2.3 * s) / (0.55 * s)) ** 2))
        p = p + 10.0 ** (lvl / 10.0) * psf
    return 10.0 * np.log10(p)


def spiral_array():
    i = np.arange(N_MIC)
    r = (ARRAY_D / 2.0) * np.sqrt((i + 0.5) / N_MIC)
    th = i * np.pi * (3.0 - np.sqrt(5.0))
    return r * np.cos(th), r * np.sin(th)


def draw_hardware(ax):
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="#f0f0f0",
                           edgecolor="none", zorder=0))
    parts = [((0.24, 0.22), 0.66, 0.60, "#b3b3b3"),
             ((0.24, 0.84), 0.66, 0.16, "#d9d9d9"),
             ((0.98, 0.70), 0.46, 0.22, "#d9d9d9"),
             ((0.96, 0.18), 0.44, 0.16, "#808080")]
    for (x0, y0), w, h, fc in parts:
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=fc, edgecolor="#808080",
                                    lw=0.7, mutation_aspect=1.0, zorder=1))
    for cx, cy, r, fc in [(0.56, 0.63, 0.135, "#808080"),
                          (0.33, 0.72, 0.070, "#808080"),
                          (0.86, 0.32, 0.095, "#4d4d4d")]:
        ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor="#4d4d4d",
                            lw=0.7, zorder=2))
    ax.plot([0.33, 0.56], [0.79, 0.765], color="#4d4d4d", lw=1.0, zorder=2)
    ax.plot([0.33, 0.56], [0.65, 0.495], color="#4d4d4d", lw=1.0, zorder=2)
    for i in range(4):
        ax.add_patch(Rectangle((1.00 + i * 0.10, 0.34), 0.055, 0.13,
                               facecolor="#808080", edgecolor="#4d4d4d",
                               lw=0.6, zorder=2))
    for x, y, s in [(0.57, 0.10, "engine block"), (1.21, 0.94, "intake duct"),
                    (1.18, 0.12, "exhaust manifold")]:
        ax.text(x, y, s, ha="center", va="center", fontsize=6.4,
                color=COLORS["sub"], zorder=3)


def main():
    set_pub_style()
    rng = np.random.default_rng(423)

    gx = np.linspace(0.0, W, 340)
    gy = np.linspace(0.0, H, 245)
    X, Y = np.meshgrid(gx, gy)
    lvl = beam_map(X, Y, F_SEL)
    peak = float(lvl.max())
    iy, ix = np.unravel_index(np.argmax(lvl), lvl.shape)
    px, py = gx[ix], gy[iy]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.46],
                          height_ratios=[1.0, 0.62], wspace=0.16, hspace=0.40,
                          left=0.048, right=0.975, top=0.885, bottom=0.165)
    ax = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[1, 1])

    # ---- (a) acoustic image -----------------------------------------------
    draw_hardware(ax)
    levels = np.linspace(peak - DYN_RANGE, peak, 13)
    masked = np.ma.masked_less(lvl, peak - DYN_RANGE)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        cs = ax.contourf(X, Y, masked, levels=levels, cmap="magma",
                         alpha=0.74, zorder=4)
        cs.set_rasterized(True)
    cs.set_edgecolor("face")
    ax.contour(X, Y, lvl, levels=[peak - 3.0], colors="white",
               linewidths=1.0, zorder=5)

    halo_text(ax, px, py + 0.10, f"dominant source\n{peak:.1f} dB at "
              f"({px:.2f}, {py:.2f}) m", ha="center", va="bottom",
              fontsize=6.8, zorder=8, linespacing=1.35)
    ax.plot([px], [py], marker="+", ms=9, mew=1.4, color="white", zorder=8)

    axi = ax.inset_axes([0.735, 0.035, 0.245, 0.245])
    mx, my = spiral_array()
    axi.scatter(mx, my, s=4.0, color=COLORS["rule"], zorder=3)
    axi.add_patch(Circle((0, 0), ARRAY_D / 2.0, facecolor="none",
                         edgecolor=COLORS["grid"], lw=0.7))
    axi.set_xlim(-0.42, 0.42)
    axi.set_ylim(-0.42, 0.42)
    axi.set_aspect("equal")
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_zorder(7)
    axi.patch.set_visible(True)
    axi.patch.set_alpha(0.92)
    for sp in axi.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color(COLORS["rule"])
    axi.set_title(f"{N_MIC}-mic spiral", fontsize=6.4, color=COLORS["sub"],
                  pad=2.0)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal position (m)")
    ax.set_ylabel("Vertical position (m)")
    ax.set_title(f"Beamforming map, {F_SEL:.0f} Hz third-octave band",
                 fontsize=9.0, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(COLORS["rule"])
    safe_colorbar(fig, cs, ax, label=f"$L_p$ (dB, {DYN_RANGE:.0f} dB display "
                  "range)", side="bottom", width=0.13, pad=0.40, shrink=0.88,
                  ticks=np.arange(np.ceil(peak - DYN_RANGE), peak + 0.1, 3.0))

    # ---- (b) band selection ------------------------------------------------
    spec = 74.0 + 12.0 * np.exp(-0.5 * (np.log(BANDS / 3800.0) / 0.46) ** 2) \
        + 5.0 * np.exp(-0.5 * (np.log(BANDS / 500.0) / 0.30) ** 2) \
        + rng.normal(0.0, 0.6, BANDS.size)
    sel = int(np.argmin(np.abs(BANDS - F_SEL)))
    cols = [COLORS["accent"] if i == sel else "#b3b3b3"
            for i in range(BANDS.size)]
    ax_b.bar(np.arange(BANDS.size), spec, width=0.76, color=cols,
             edgecolor="white", linewidth=0.5, zorder=3)
    ax_b.set_xticks(np.arange(BANDS.size))
    ax_b.set_xticklabels([f"{b:.0f}" if b < 1000 else f"{b / 1000:g}k"
                          for b in BANDS], fontsize=6.2, rotation=90)
    ax_b.set_ylim(60.0, spec.max() * 1.06)
    ax_b.set_ylabel("Band level (dB)", fontsize=8.0)
    ax_b.set_xlabel("Third-octave centre frequency (Hz)", fontsize=8.0)
    ax_b.tick_params(labelsize=7.0)
    ax_b.grid(axis="y", lw=0.5, alpha=0.6)
    ax_b.annotate(f"imaged band\n{spec[sel]:.1f} dB",
                  xy=(sel, spec[sel]), xytext=(-14, 12),
                  textcoords="offset points", ha="right", fontsize=6.8,
                  color=COLORS["accent"], linespacing=1.35,
                  arrowprops=dict(arrowstyle="-", lw=0.8,
                                  color=COLORS["accent"]))
    ax_b.set_title("Band selection", fontsize=8.5, fontweight="bold")
    style_axes(ax_b)

    # ---- (c) measurement parameters ---------------------------------------
    ax_i.set_title("Measurement setup", fontsize=8.5, fontweight="bold")
    cw, ch = use_inch_units(ax_i)
    rank = sorted(SOURCES, key=lambda s: -s[2])
    facts = [("Array", f"{ARRAY_D * 100:.0f} cm / {N_MIC} mics"),
             ("Stand-off", f"{STANDOFF:.2f} m"),
             ("Resolution", f"{resolution(F_SEL) * 100:.0f} cm"),
             ("Peak margin", f"+{rank[0][2] - rank[1][2]:.1f} dB")]
    bw, bh, gap = (cw - 0.10) / 2.0 - 0.06, (ch - 0.24) / 2.0 - 0.06, 0.12
    for i, (name, value) in enumerate(facts):
        x0 = 0.05 + (i % 2) * (bw + gap)
        y0 = 0.05 + (1 - i // 2) * (bh + gap)
        text_card(ax_i, x0, y0, bw, bh, value, title=name, fs=8.6,
                  title_fs=6.8, title_color=COLORS["sub"], ha="left",
                  color=COLORS["ink"], weight="bold", rounding=0.06, pad=0.10)

    for a, lab in zip((ax, ax_b, ax_i), ("(a)", "(b)", "(c)")):
        a.text(-0.09 if a is ax else -0.16, 1.045, lab, transform=a.transAxes,
               fontsize=11, fontweight="bold", va="bottom")

    savefig(fig, "423_acu_beamforming.png")


if __name__ == "__main__":
    main()
