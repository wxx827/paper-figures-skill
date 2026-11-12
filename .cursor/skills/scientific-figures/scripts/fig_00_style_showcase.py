"""Design-system showcase for pubstyle.py.

Renders one poster-style sheet covering the whole shared style system:
palettes, journal themes, helper functions, colour-vision safety and the
"what makes a figure look expensive" checklist.

Output: output/00_style_showcase.png
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from pubstyle import (PALETTES, add_colorbar, add_sig_bracket, annotate_box,
                      despine, get_cmap, get_palette, letter_panels, savefig,
                      set_pub_style, sig_stars, smooth_line)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
RULE = "#d0d0d0"

LEFT_PALETTES = ["teal_rust", "blue_red", "greens", "earth",
                 "div_rdbu", "pastel4", "zones", "neutral"]
RIGHT_PALETTES = ["cb_safe", "cb_muted", "cb_diverging", "cb_diverging_pg",
                  "cb_sequential_blue", "cb_sequential_green",
                  "cb_sequential_warm", "cb_sequential_purple"]
CONTINUOUS = {"greens", "div_rdbu", "neutral", "cb_diverging",
              "cb_diverging_pg", "cb_sequential_blue", "cb_sequential_green",
              "cb_sequential_warm", "cb_sequential_purple"}

THEME_NOTES = [
    ("default", "general purpose"),
    ("nature", "7 pt, hairline rules"),
    ("science", "8 pt, more contrast"),
    ("ieee", "narrow 2-column"),
    ("presentation", "projector safe"),
]

CHECKLIST = [
    "Kill the top and right spines; keep the rest hairline (0.8 pt, #333).",
    "Type stays small (7-9 pt); bold only for panel titles and statistics.",
    "Muted, low-chroma hues - never a saturated RGB primary.",
    "One accent colour per figure; grey carries everything secondary.",
    "Statistics live in a white rounded call-out, not floating on data.",
    "Direct-label lines instead of a legend whenever it fits.",
    "Grid, if any, sits behind the data in a whisper of grey.",
    "Consistent panel labels: (a) (b) (c), bold, outside the axes.",
    "Export at 300 dpi with bbox_inches='tight'; PDF/SVG for submission.",
]

# Machado et al. (2009) deuteranopia matrix, severity 1.0, linear RGB.
_DEUTER = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def simulate_deuteranopia(color):
    """Approximate how a colour reads to a deuteranope."""
    rgb = np.array(mpl.colors.to_rgb(color))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    sim = np.clip(_DEUTER @ lin, 0, 1)
    out = np.where(sim <= 0.0031308, sim * 12.92,
                   1.055 * sim ** (1 / 2.4) - 0.055)
    return mpl.colors.to_hex(np.clip(out, 0, 1))


def section_title(fig, y, number, text, note=""):
    """Small caps section header with a hairline rule, at figure height y."""
    fig.text(0.045, y, f"{number}", fontsize=8.5, fontweight="bold",
             color="#b08968", va="bottom", ha="left")
    fig.text(0.063, y, text.upper(), fontsize=9.0, fontweight="bold",
             color=INK, va="bottom", ha="left")
    if note:
        fig.text(0.975, y + 0.0005, note, fontsize=7.5, color=MUTED,
                 va="bottom", ha="right")
    fig.add_artist(Line2D([0.045, 0.975], [y - 0.0055, y - 0.0055],
                          color=RULE, lw=0.7, zorder=0))


def draw_palettes(ax):
    ax.set_xlim(0, 2)
    ax.set_ylim(8, 0)
    ax.axis("off")
    for col, names in enumerate([LEFT_PALETTES, RIGHT_PALETTES]):
        x0 = col * 1.0
        for row, name in enumerate(names):
            cy = row + 0.5
            cols = PALETTES[name]
            ax.text(x0 + 0.305, cy, name, fontsize=8.0, color=INK,
                    ha="right", va="center", fontfamily="monospace")
            ax.text(x0 + 0.322, cy, f"{len(cols)}", fontsize=6.5, color="#a8a8a8",
                    ha="left", va="center")
            bx0, bx1 = x0 + 0.355, x0 + 0.955
            h = 0.40
            if name in CONTINUOUS:
                ramp = get_palette(name, 96)
                w = (bx1 - bx0) / len(ramp)
                for i, c in enumerate(ramp):
                    ax.add_patch(Rectangle((bx0 + i * w, cy - h / 2), w * 1.02, h,
                                           facecolor=c, edgecolor="none"))
                ax.add_patch(Rectangle((bx0, cy - h / 2), bx1 - bx0, h,
                                       facecolor="none", edgecolor="#c9c9c9",
                                       lw=0.5))
            else:
                n = len(cols)
                gap = 0.006
                w = (bx1 - bx0 - gap * (n - 1)) / n
                for i, c in enumerate(cols):
                    ax.add_patch(FancyBboxPatch(
                        (bx0 + i * (w + gap) + w * 0.06, cy - h / 2 + 0.03),
                        w * 0.88, h - 0.06,
                        boxstyle="round,pad=0.012,rounding_size=0.05",
                        facecolor=c, edgecolor="white", lw=0.4,
                        mutation_aspect=0.28))


def theme_demo(fig, cell, theme, note):
    """Same plot rendered under one theme (rcParams scoped to this axes)."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 60)
    with mpl.rc_context():
        set_pub_style(theme=theme)
        ax = fig.add_subplot(cell)
        for i, (amp, ph) in enumerate([(1.0, 0.0), (0.62, 1.4)]):
            c = get_palette("cb_safe")[i]
            y = amp * np.sin(x * 0.7 + ph) + 0.15 * x
            ax.fill_between(x, y - 0.28, y + 0.28, color=c, alpha=0.16, lw=0)
            ax.plot(x, y, color=c)
            idx = np.arange(0, 60, 12)
            ax.plot(x[idx], y[idx] + rng.normal(0, 0.05, idx.size), "o",
                    color=c, mec="white", mew=0.6, ls="none")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Response")
        ax.set_xticks([0, 5, 10])
        ax.set_title(theme, fontweight="bold", color=INK)
        despine(ax)
    bb = ax.get_position()
    fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.038, note, fontsize=7.0,
             color=MUTED, ha="center", va="top")
    return ax


def demo_brackets(ax):
    rng = np.random.default_rng(11)
    labels = ["Ctrl", "Low", "Mid", "High"]
    means = np.array([2.10, 2.55, 3.20, 3.65])
    errs = np.array([0.18, 0.21, 0.19, 0.24])
    cols = get_palette("cb_sequential_blue", 6)[1:5]
    ax.bar(labels, means, yerr=errs, color=cols, edgecolor="#2f2f2f",
           lw=0.6, width=0.62, error_kw=dict(lw=0.8, ecolor="#444444"))
    for i, m in enumerate(means):
        pts = rng.normal(m, 0.10, 8)
        ax.plot(np.full(8, i) + rng.uniform(-0.13, 0.13, 8), pts, "o",
                ms=2.2, color="#33333360", mec="none", zorder=4)
    ax.set_ylim(0, 5.3)
    add_sig_bracket(ax, 0, 2, 4.15, sig_stars(0.0004))
    add_sig_bracket(ax, 1, 3, 4.72, sig_stars(0.012))
    ax.set_ylabel("Yield (t ha$^{-1}$)")
    ax.set_title("add_sig_bracket / sig_stars", fontsize=8.2, color=MUTED)
    despine(ax)


def demo_smooth(ax):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 260)
    y = 1.6 + 0.55 * x - 0.045 * x ** 2 + rng.normal(0, 0.42, x.size)
    ax.scatter(x, y, s=8, color=PALETTES["teal_rust"][0], alpha=0.45,
               edgecolors="none")
    xg, yg = smooth_line(x, y, frac=0.25)
    ax.fill_between(xg, yg - 0.34, yg + 0.34, color="#8a8a8a", alpha=0.18, lw=0)
    ax.plot(xg, yg, color="#222222", lw=1.8)
    annotate_box(ax, f"$R^2$ = 0.71{sig_stars(0.0002)}\nn = 260",
                 loc="lower right", weight="bold")
    ax.set_xlabel("Driver")
    ax.set_ylabel("Response")
    ax.set_title("smooth_line / annotate_box", fontsize=8.2, color=MUTED)
    despine(ax)


def demo_colorbar(fig, ax):
    gx, gy = np.meshgrid(np.linspace(-3, 3, 160), np.linspace(-3, 3, 160))
    z = np.sin(gx) * np.cos(gy * 0.8) * np.exp(-0.08 * (gx ** 2 + gy ** 2))
    im = ax.imshow(z, cmap=get_cmap("cb_diverging", reverse=True),
                   extent=[-3, 3, -3, 3], origin="lower", vmin=-0.9, vmax=0.9,
                   aspect="auto")
    ax.contour(gx, gy, z, levels=[-0.4, 0, 0.4], colors="#33333355",
               linewidths=0.5)
    add_colorbar(fig, im, ax, label="Anomaly", ticks=[-0.8, 0, 0.8],
                 fraction=0.05, pad=0.035)
    ax.set_xlabel("Longitude offset")
    ax.set_ylabel("Latitude offset")
    ax.set_title("add_colorbar / get_cmap", fontsize=8.2, color=MUTED)


def demo_directlabel(ax):
    x = np.arange(2010, 2025)
    names = ["Alpha", "Beta", "Gamma", "Delta"]
    cols = get_palette("cb_safe", 4)
    rng = np.random.default_rng(2)
    for i, (nm, c) in enumerate(zip(names, cols)):
        y = np.cumsum(rng.normal(0.3 + 0.18 * i, 0.45, x.size)) + 4
        ax.plot(x, y, color=c, lw=1.5)
        ax.plot(x[-1], y[-1], "o", color=c, ms=3.2, mec="white", mew=0.6)
        ax.text(x[-1] + 0.35, y[-1], nm, color=c, fontsize=7.5,
                va="center", ha="left", fontweight="bold")
    ax.set_xlim(2010, 2027.5)
    ax.set_xticks([2010, 2015, 2020, 2024])
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.set_title("get_palette / direct labels", fontsize=8.2, color=MUTED)
    despine(ax)


def draw_cvd(ax):
    bad = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
    blocks = [("cb_safe (Okabe-Ito)", get_palette("cb_safe", 6), True),
              ("saturated RGB primaries", bad, False)]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for block, (title, cols, good) in enumerate(blocks):
        y = 0.95 - block * 0.50
        mark = "recommended" if good else "avoid"
        mcol = "#2e7d5b" if good else "#b23a48"
        ax.text(0.0, y, title, fontsize=8.0, fontweight="bold", color=INK,
                va="center", fontfamily="monospace")
        ax.text(1.0, y, mark, fontsize=7.2, color=mcol, va="center", ha="right",
                fontweight="bold")
        for k, (row_label, transform) in enumerate(
                [("normal", lambda c: c), ("deuteranopia", simulate_deuteranopia)]):
            yy = y - 0.16 - k * 0.145
            ax.text(0.0, yy, row_label, fontsize=7.0, color=MUTED,
                    va="center", ha="left")
            bx0, bx1 = 0.21, 1.0
            w = (bx1 - bx0) / len(cols)
            for i, c in enumerate(cols):
                ax.add_patch(Rectangle((bx0 + i * w + 0.004, yy - 0.048),
                                       w - 0.008, 0.096,
                                       facecolor=transform(c),
                                       edgecolor="white", lw=0.6))
    ax.text(0.0, 0.0, "Okabe-Ito stays separable under deuteranopia; the "
                      "primaries collapse into muddy look-alikes.",
            fontsize=7.0, color=MUTED, va="bottom", ha="left", style="italic")


def draw_checklist(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.98,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="#fbfaf8", edgecolor="#e0dcd5",
                                lw=0.8, mutation_aspect=0.45,
                                transform=ax.transAxes, zorder=0))
    ax.text(0.045, 0.935, "WHAT MAKES A FIGURE LOOK EXPENSIVE", fontsize=8.2,
            fontweight="bold", color=INK, va="top")
    for i, line in enumerate(CHECKLIST):
        yy = 0.845 - i * 0.093
        ax.text(0.045, yy, "\u2022", fontsize=8.0, color="#b08968", va="top")
        ax.text(0.078, yy, line, fontsize=7.3, color="#3a3a3a", va="top")


def main():
    set_pub_style()
    fig = plt.figure(figsize=(12.4, 14.2))
    L, R = 0.045, 0.975

    # --- header -----------------------------------------------------------
    fig.add_artist(Line2D([L, R], [0.9495, 0.9495], color="#8a8a8a", lw=1.0))

    # --- 1. palettes ------------------------------------------------------
    section_title(fig, 0.9285, "01", "Palettes",
                  "PALETTES[name] · get_palette(name, n) · get_cmap(name)")
    ax_pal = fig.add_axes([L, 0.720, R - L, 0.192])
    draw_palettes(ax_pal)

    # --- 2. themes --------------------------------------------------------
    section_title(fig, 0.6875, "02", "Journal themes",
                  "set_pub_style(theme=...) · one dataset, five house styles")
    gs_theme = fig.add_gridspec(1, 5, wspace=0.46, left=L, right=R,
                                top=0.645, bottom=0.515,
                                width_ratios=[1, 1, 1, 1, 1.22])
    theme_axes = [theme_demo(fig, gs_theme[i], t, note)
                  for i, (t, note) in enumerate(THEME_NOTES)]

    # --- 3. helpers -------------------------------------------------------
    section_title(fig, 0.4515, "03", "Helper functions",
                  "letter_panels() stamped the (a)-(d) labels below")
    gs_help = fig.add_gridspec(1, 4, wspace=0.56, left=L, right=R,
                               top=0.400, bottom=0.267)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs_help[i]) for i in range(4)]
    demo_brackets(ax_a)
    demo_smooth(ax_b)
    demo_colorbar(fig, ax_c)
    demo_directlabel(ax_d)
    letter_panels([ax_a, ax_b, ax_c, ax_d], x=-0.24, y=1.10, size=10)

    # --- 4. colour vision + checklist -------------------------------------
    section_title(fig, 0.2185, "04", "Colour-vision safety & house rules",
                  "simulate_deuteranopia() · full tables in reference.md")
    gs_bot = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.10,
                              left=L, right=R, top=0.196, bottom=0.030)
    ax_cvd = fig.add_subplot(gs_bot[0])
    ax_chk = fig.add_subplot(gs_bot[1])
    draw_cvd(ax_cvd)
    draw_checklist(ax_chk)

    savefig(fig, "00_style_showcase.png")


if __name__ == "__main__":
    main()
