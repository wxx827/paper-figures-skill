"""Steady-state sound field in a rectangular room at one third-octave centre.

Left: sound pressure level over the room plan, summed coherently over the
image sources up to third order (hence the standing-wave lobes) plus a Sabine
diffuse term, with the three first-order reflection paths drawn in.  Right:
the level sampled along the source-to-receiver line against the reverberant
floor, and the room acoustics metrics the same numbers imply.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, get_cmap, style_axes, COLORS
from figkit import safe_colorbar, text_card, use_inch_units, halo_text

LX, LY, LZ = 10.0, 7.0, 3.2          # room dimensions (m)
ALPHA = 0.14                          # mean absorption coefficient (-)
LW = 95.0                             # source sound power level (dB re 1 pW)
F0, C0 = 125.0, 343.0                 # analysis frequency (Hz), sound speed
MAX_ORDER = 3
SRC = (2.1, 1.7)
RCV = (7.9, 5.3)
R_MIN = 0.40                          # near-field clip (m)


def room_constant():
    s = 2.0 * (LX * LY) + 2.0 * LZ * (LX + LY)
    return s * ALPHA / (1.0 - ALPHA), s


def _mirror_axis(pos, length):
    """(coordinate, reflection count) of every image along one axis."""
    out = []
    for m in range(-2, 3):
        out.append((2.0 * m * length + pos, abs(2 * m)))
        out.append((2.0 * m * length - pos, abs(2 * m - 1)))
    return out


def image_sources():
    """All images up to ``MAX_ORDER``: (x, y, pressure reflection factor)."""
    rp = np.sqrt(1.0 - ALPHA)
    out = []
    for xi, ox in _mirror_axis(SRC[0], LX):
        for yi, oy in _mirror_axis(SRC[1], LY):
            if ox + oy <= MAX_ORDER:
                out.append((xi, yi, rp ** (ox + oy)))
    return out


IMAGES = image_sources()


def spl(x, y):
    """Sound pressure level (dB) at plan position (x, y)."""
    rc, _ = room_constant()
    k = 2.0 * np.pi * F0 / C0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.zeros(np.shape(x), dtype=complex)
    for sx, sy, a in IMAGES:
        r = np.maximum(np.hypot(x - sx, y - sy), R_MIN)
        p = p + a * np.exp(-1j * k * r) / r
    e = np.abs(p) ** 2 / (4.0 * np.pi) + 4.0 / rc
    return LW + 10.0 * np.log10(e)


def reflection_points():
    """Wall hit point of each first-order path, with its total path length."""
    first = [(-SRC[0], SRC[1], "left wall"),
             (2.0 * LX - SRC[0], SRC[1], "right wall"),
             (SRC[0], -SRC[1], "front wall")]
    out = []
    for ix, iy, name in first:
        if name == "front wall":
            t = (0.0 - iy) / (RCV[1] - iy)
        else:
            wall_x = 0.0 if ix < 0 else LX
            t = (wall_x - ix) / (RCV[0] - ix)
        px, py = ix + t * (RCV[0] - ix), iy + t * (RCV[1] - iy)
        length = np.hypot(px - SRC[0], py - SRC[1]) + np.hypot(RCV[0] - px,
                                                               RCV[1] - py)
        out.append((px, py, length, name))
    return out


def main():
    set_pub_style()
    rc, surf = room_constant()
    volume = LX * LY * LZ
    rt60 = 0.161 * volume / (surf * ALPHA)
    r_crit = np.sqrt(rc / (16.0 * np.pi))

    gx = np.linspace(0.0, LX, 280)
    gy = np.linspace(0.0, LY, 200)
    X, Y = np.meshgrid(gx, gy)
    field = spl(X, Y)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.56],
                          height_ratios=[1.0, 0.62], wspace=0.20, hspace=0.42,
                          left=0.055, right=0.975, top=0.885, bottom=0.115)
    ax = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) room plan ----------------------------------------------------
    lo, hi = np.percentile(field, [2.0, 99.7])
    levels = np.linspace(lo, hi, 15)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        cs = ax.contourf(X, Y, field, levels=levels, extend="both",
                         cmap=get_cmap("cb_sequential_blue"))
        cs.set_rasterized(True)
    cs.set_edgecolor("face")

    for px, py, length, name in reflection_points():
        ax.plot([SRC[0], px, RCV[0]], [SRC[1], py, RCV[1]],
                color=COLORS["solid"], lw=0.9, ls=(0, (3.2, 2.0)), alpha=0.8,
                zorder=5)
        ax.plot([px], [py], marker="o", ms=3.4, mfc="white",
                mec=COLORS["rule"], mew=0.7, zorder=6)
    ax.plot([SRC[0], RCV[0]], [SRC[1], RCV[1]], color=COLORS["accent"],
            lw=1.4, zorder=6)
    ax.plot([SRC[0]], [SRC[1]], marker="o", ms=8.0, mfc=COLORS["accent"],
            mec="white", mew=1.0, zorder=8)
    ax.plot([RCV[0]], [RCV[1]], marker="^", ms=7.5, mfc="white",
            mec=COLORS["accent"], mew=1.4, zorder=8)
    halo_text(ax, SRC[0], SRC[1] - 0.44, "source", ha="center", va="top",
              fontsize=7.0)
    halo_text(ax, RCV[0], RCV[1] + 0.36, "receiver", ha="center", va="bottom",
              fontsize=7.0)
    ax.add_patch(Rectangle((0, 0), LX, LY, facecolor="none",
                           edgecolor=COLORS["rule"], lw=1.2, zorder=7))

    ax.set_xlim(0, LX)
    ax.set_ylim(0, LY)
    ax.set_aspect("equal")
    ax.set_xlabel("Room x (m)")
    ax.set_ylabel("Room y (m)")
    ax.set_title("Sound pressure level over the room plan", fontsize=9.0,
                 fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(COLORS["rule"])
    safe_colorbar(fig, cs, ax, label="$L_p$ (dB re 20 $\\mu$Pa)",
                  side="bottom", width=0.13, pad=0.42, shrink=0.86,
                  ticks=np.arange(np.ceil(lo), np.floor(hi) + 0.1, 2.0))
    ax.text(-0.075, 1.045, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")

    # ---- (b) level along the direct path ----------------------------------
    t = np.linspace(0.0, 1.0, 300)
    px = SRC[0] + t * (RCV[0] - SRC[0])
    py = SRC[1] + t * (RCV[1] - SRC[1])
    d = np.hypot(px - SRC[0], py - SRC[1])
    lp = spl(px, py)
    l_rev = LW + 10.0 * np.log10(4.0 / rc)

    ax_p.plot(d, lp, color=COLORS["accent"], lw=1.7, zorder=5,
              label="total field")
    ax_p.plot(d[1:], LW - 10.0 * np.log10(4.0 * np.pi * d[1:] ** 2),
              color="#3b5b92", lw=1.1, ls=(0, (4.0, 2.0)), zorder=4,
              label="direct field only")
    ax_p.axhline(l_rev, color=COLORS["baseline"], lw=1.0, ls=(0, (2.0, 2.0)),
                 zorder=3)
    ax_p.text(0.04, l_rev - 0.4, f"reverberant floor {l_rev:.1f} dB",
              transform=ax_p.get_yaxis_transform(), ha="left", va="top",
              fontsize=6.6, color=COLORS["sub"])
    ax_p.axvline(r_crit, color=COLORS["ink"], lw=0.9, ls=(0, (1.6, 1.8)),
                 zorder=3)
    ax_p.text(r_crit + 0.12, 0.94, f"critical distance\n{r_crit:.2f} m",
              transform=ax_p.get_xaxis_transform(), ha="left", va="top",
              fontsize=6.6, color=COLORS["ink"], linespacing=1.3)
    ax_p.set_xlim(0, d[-1])
    ax_p.set_ylim(min(lp.min(), l_rev) - 5.0, lp.max() + 6.0)
    ax_p.set_xlabel("Distance from source (m)", fontsize=8.0)
    ax_p.set_ylabel("$L_p$ (dB)", fontsize=8.0)
    ax_p.tick_params(labelsize=7.0)
    ax_p.legend(loc="upper right", fontsize=7.0, frameon=False)
    ax_p.set_title("Decay along the source-receiver line", fontsize=8.5,
                   fontweight="bold")
    style_axes(ax_p)
    ax_p.text(-0.13, 1.05, "(b)", transform=ax_p.transAxes, fontsize=11,
              fontweight="bold", va="bottom")

    # ---- (c) room acoustics metric cards ----------------------------------
    ax_c.set_title("Room acoustics metrics", fontsize=8.5, fontweight="bold")
    ax_c.text(-0.13, 1.05, "(c)", transform=ax_c.transAxes, fontsize=11,
              fontweight="bold", va="bottom")
    cw, ch = use_inch_units(ax_c)
    cards = [("$T_{60}$ (Sabine)", f"{rt60:.2f} s"),
             ("Room volume", f"{volume:.0f} m$^3$"),
             ("Mean $\\overline{\\alpha}$", f"{ALPHA:.2f} (-)"),
             ("$L_p$ at receiver", f"{spl(*RCV):.1f} dB")]
    bw, bh, gap = (cw - 0.12) / 2.0 - 0.06, (ch - 0.10) / 2.0 - 0.06, 0.12
    for i, (name, value) in enumerate(cards):
        x0 = 0.03 + (i % 2) * (bw + gap)
        y0 = 0.03 + (1 - i // 2) * (bh + gap)
        text_card(ax_c, x0, y0, bw, bh, value, title=name, fs=9.5,
                  title_fs=7.0, title_color=COLORS["sub"], ha="left",
                  color=COLORS["ink"], weight="bold", rounding=0.06,
                  pad=0.10)

    savefig(fig, "417_acu_sound_field.png")


if __name__ == "__main__":
    main()
