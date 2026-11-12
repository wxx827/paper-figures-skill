"""Transit photometry of a hot Jupiter: full series, folded profile, geometry.

The transit model integrates a quadratic limb-darkening law over the fraction
of the stellar disc hidden by the planet, evaluated on an equal-area polar
sampling of the planet disc, so the folded profile shows the rounded bottom
and finite ingress/egress that limb darkening produces.  The right-hand column
is deliberately narrow: it carries the geometry cartoon and the parameter
table that support, but do not compete with, the photometry.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

PERIOD = 3.21455        # d
T0 = 1.412              # d
K_RATIO = 0.1103        # Rp / Rs
A_RS = 9.52             # a / Rs
INC = np.radians(87.2)
U1, U2 = 0.42, 0.24
B_IMPACT = A_RS * np.cos(INC)


def limb_intensity(r):
    mu = np.sqrt(np.clip(1.0 - r ** 2, 0.0, 1.0))
    return 1.0 - U1 * (1 - mu) - U2 * (1 - mu) ** 2


def planet_samples(n_r=26, n_a=40):
    """Equal-area sampling of the planet disc, in stellar radii."""
    rr = np.sqrt((np.arange(n_r) + 0.5) / n_r) * K_RATIO
    th = (np.arange(n_a) + 0.5) / n_a * 2 * np.pi
    R, T = np.meshgrid(rr, th, indexing="ij")
    return (R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()


PX, PY = planet_samples()
STAR_FLUX = np.pi * (1.0 - U1 / 3.0 - U2 / 6.0)
CELL = np.pi * K_RATIO ** 2 / PX.size


def transit_flux(t):
    ph = 2 * np.pi * ((t - T0) / PERIOD)
    x = A_RS * np.sin(ph)
    y = A_RS * np.cos(ph) * np.cos(INC)
    behind = np.cos(ph) < 0
    dx = x[:, None] + PX[None, :]
    dy = y[:, None] + PY[None, :]
    r = np.hypot(dx, dy)
    inside = (r < 1.0) & (~behind)[:, None]
    blocked = np.where(inside, limb_intensity(np.clip(r, 0, 1)), 0.0).sum(1)
    return 1.0 - blocked * CELL / STAR_FLUX


def trend(t):
    return (1.0 + 0.00115 * np.sin(2 * np.pi * t / 8.4 + 0.6)
            + 0.00055 * np.sin(2 * np.pi * t / 3.1 + 2.1))


def star_image(n=320):
    g = np.linspace(-1.35, 1.35, n)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    I = np.where(R <= 1.0, limb_intensity(np.clip(R, 0, 1)), np.nan)
    return I, (-1.35, 1.35, -1.35, 1.35)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(89)

    t = np.arange(0.0, 22.0, 10.0 / (60 * 24))
    model = transit_flux(t)
    sigma = 0.00092
    flux = model * trend(t) + rng.normal(0, sigma, t.size)
    detr = flux / trend(t)

    ph_h = (((t - T0 + PERIOD / 2) % PERIOD) - PERIOD / 2) * 24.0
    win = np.abs(ph_h) < 4.6
    fold_t = T0 + ph_h[win] / 24.0
    fold_m = transit_flux(fold_t)
    resid = (detr[win] - fold_m) * 1e6

    edges = np.linspace(-4.6, 4.6, 63)
    idx = np.clip(np.digitize(ph_h[win], edges) - 1, 0, edges.size - 2)
    bx, by, be = [], [], []
    for i in range(edges.size - 1):
        m = idx == i
        if m.sum() < 3:
            continue
        bx.append(0.5 * (edges[i] + edges[i + 1]))
        by.append(detr[win][m].mean())
        be.append(detr[win][m].std(ddof=1) / np.sqrt(m.sum()))
    bx, by, be = np.array(bx), np.array(by), np.array(be)

    depth = 1.0 - transit_flux(np.array([T0]))[0]
    dur_h = PERIOD * 24 / np.pi * np.arcsin(
        np.sqrt((1 + K_RATIO) ** 2 - B_IMPACT ** 2) / A_RS / np.sin(INC))

    fig = plt.figure(figsize=(10.4, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.74, 1.0, 0.30],
                          width_ratios=[1.0, 1.0, 0.80], hspace=0.36,
                          wspace=0.20, left=0.062, right=0.985, top=0.882,
                          bottom=0.070)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[2, :2])
    sub = gs[1:, 2].subgridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.24)
    ax_d = fig.add_subplot(sub[0])
    ax_e = fig.add_subplot(sub[1])

    # ---- (a) full photometric series ---------------------------------------
    ax_a.plot(t, flux, color="#a9bacd", lw=0.0, marker="o", ms=0.7,
              alpha=0.55, zorder=2)
    ax_a.plot(t, trend(t), color=RUST, lw=1.0, ls=(0, (5, 2.6)), zorder=4)
    n_tr = int((t.max() - T0) // PERIOD) + 1
    for k in range(n_tr):
        tc = T0 + k * PERIOD
        if tc > t.max():
            break
        ax_a.plot([tc], [1 - depth - 0.0020], marker="^", ms=4.2, mfc=GOLD,
                  mec="#6d520f", mew=0.5, ls="none", clip_on=False, zorder=6)
    ax_a.set_xlim(0, 22)
    ax_a.set_ylim(1 - depth - 0.0028, 1.0038)
    ax_a.set_xlabel("time  (BJD $-$ 2458000, day)", fontsize=8.2, labelpad=2)
    ax_a.set_ylabel("relative flux", fontsize=8.2, labelpad=3)
    ax_a.tick_params(labelsize=7.0, length=2.6)
    ax_a.grid(True, lw=0.4, color="#eceef1")
    ax_a.set_axisbelow(True)
    ax_a.set_title("Continuous photometry: %d transits at a %.4f d period"
                   % (n_tr, PERIOD), fontsize=8.8, fontweight="bold", pad=3)
    ax_a.text(0.006, 0.055, "10 min cadence,  $\\sigma = %d$ ppm per point"
              % round(sigma * 1e6), transform=ax_a.transAxes, ha="left",
              va="bottom", fontsize=6.8, zorder=8,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                        lw=0.55, alpha=0.94))

    # ---- (b) folded transit -------------------------------------------------
    ax_b.plot(ph_h[win], detr[win], ls="none", marker="o", ms=1.9,
              mfc="#9fb2c6", mec="none", alpha=0.6, zorder=2)
    ax_b.errorbar(bx, by, yerr=be, ls="none", marker="o", ms=4.0, mfc=NAVY,
                  mec="white", mew=0.6, ecolor=NAVY, elinewidth=0.9, zorder=4)
    grid_h = np.linspace(-4.6, 4.6, 900)
    ax_b.plot(grid_h, transit_flux(T0 + grid_h / 24.0), color=GOLD, lw=2.0,
              zorder=5)
    ax_b.axhline(1.0, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax_b.axhline(1.0 - depth, color=CYAN, lw=0.7, ls=(0, (4, 3)), zorder=1)
    for s in (-1, 1):
        ax_b.axvline(s * dur_h / 2, color=INDIGO, lw=0.7, ls=(0, (2, 2.2)),
                     zorder=1)
    y_arrow = 1.0 + 0.085 * depth
    ax_b.annotate("", xy=(-dur_h / 2, y_arrow), xytext=(dur_h / 2, y_arrow),
                  arrowprops=dict(arrowstyle="<->", lw=0.8, color=INDIGO))
    ax_b.text(0, y_arrow + 0.012 * depth, "$T_{14} = %.2f$ h" % dur_h,
              ha="center", va="bottom", fontsize=7.0, color=INDIGO)
    ax_b.text(4.45, 1.0 - depth + 0.025 * depth, "depth $= %d$ ppm"
              % round(depth * 1e6), ha="right", va="bottom", fontsize=7.0,
              color=CYAN)
    ax_b.annotate("limb-darkened,\nrounded floor", xy=(-0.55, 1 - depth),
                  xytext=(-3.05, 1 - 1.06 * depth), fontsize=7.0, color=RUST,
                  ha="center", va="center", linespacing=1.4,
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=RUST,
                                  shrinkB=4,
                                  connectionstyle="arc3,rad=0.18"))
    ax_b.set_xlim(-4.6, 4.6)
    ax_b.set_ylim(1 - 1.16 * depth, 1 + 0.20 * depth)
    ax_b.set_xticklabels([])
    ax_b.set_ylabel("normalised flux", fontsize=8.2, labelpad=3)
    ax_b.tick_params(labelsize=7.0, length=2.6)
    ax_b.grid(True, lw=0.4, color="#eceef1")
    ax_b.set_axisbelow(True)
    ax_b.set_title("Phase-folded and detrended transit with the "
                   "limb-darkened model", fontsize=8.8, fontweight="bold",
                   pad=3)

    # ---- (c) residual strip -------------------------------------------------
    rms = float(np.std(resid))
    ax_c.set_facecolor("#f4f5f7")
    ax_c.axhspan(-rms, rms, color=CYAN, alpha=0.16, lw=0, zorder=0)
    ax_c.axhline(0, color=GREY, lw=0.7, zorder=1)
    ax_c.plot(ph_h[win], resid, ls="none", marker="o", ms=1.6, mfc=INDIGO,
              mec="none", alpha=0.55, zorder=2)
    ax_c.set_xlim(-4.6, 4.6)
    ax_c.set_ylim(-3.4 * rms, 3.4 * rms)
    ax_c.set_xlabel("time from mid-transit (hour)", fontsize=8.2, labelpad=2)
    ax_c.set_ylabel("O $-$ C (ppm)", fontsize=7.2, labelpad=2)
    ax_c.tick_params(labelsize=6.6, length=2.2)
    ax_c.grid(True, axis="y", lw=0.4, color="#e3e5e8")
    ax_c.set_axisbelow(True)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_c.set_title("Supporting statistic  |  residuals,  rms $= %d$ ppm"
                   % round(rms), fontsize=7.2, style="italic",
                   color="#555555", pad=2.5)

    # ---- (d) transit geometry ----------------------------------------------
    img, ext = star_image()
    scmap = mpl.colors.LinearSegmentedColormap.from_list(
        "star", ["#c9962f", "#e8b95f", "#f7dfa5", "#fff8e6"])
    ax_d.imshow(img, extent=ext, origin="lower", cmap=scmap, zorder=2)
    ax_d.add_patch(Circle((0, 0), 1.0, fc="none", ec="#8a6b1f", lw=0.7,
                          zorder=3))
    xs = np.linspace(-1.32, 1.32, 200)
    ax_d.plot(xs, np.full_like(xs, B_IMPACT), color=NAVY, lw=0.8,
              ls=(0, (4, 2.4)), zorder=4)
    for xp, al in [(-0.86, 0.35), (-0.30, 0.60), (0.30, 1.0)]:
        ax_d.add_patch(Circle((xp, B_IMPACT), K_RATIO, fc="#182a47",
                              ec="white", lw=0.4, alpha=al, zorder=5))
    ax_d.annotate("", xy=(0.72, B_IMPACT), xytext=(0.42, B_IMPACT),
                  arrowprops=dict(arrowstyle="->", lw=0.9, color=NAVY))
    ax_d.annotate("", xy=(-1.24, B_IMPACT), xytext=(-1.24, 0.0),
                  arrowprops=dict(arrowstyle="<->", lw=0.7, color=CYAN))
    ax_d.text(-1.18, B_IMPACT / 2, "$b = %.2f$" % B_IMPACT, fontsize=6.8,
              color=CYAN, ha="left", va="center")
    ax_d.plot([-1.32, 1.32], [0, 0], color="#8a6b1f", lw=0.5,
              ls=(0, (1, 2.5)), zorder=4)
    ax_d.set_xlim(-1.38, 1.38)
    ax_d.set_ylim(-1.38, 1.38)
    ax_d.set_aspect("equal")
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_facecolor("#f6f7f9")
    for s in ax_d.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_d.set_title("geometry  |  chord across the disc", fontsize=7.2,
                   style="italic", color="#555555", pad=2.5)
    ax_d.text(0.5, 0.035, "$R_p/R_\\star = %.4f$" % K_RATIO,
              transform=ax_d.transAxes, ha="center", va="bottom",
              fontsize=6.8, color="#33475f", zorder=8,
              bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                        alpha=0.85))

    # ---- (e) parameter table -------------------------------------------------
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    ax_e.set_facecolor("#f6f7f9")
    for s in ax_e.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_e.set_title("fitted system parameters", fontsize=7.2, style="italic",
                   color="#555555", pad=2.5)
    rows = [("$P$", "%.5f d" % PERIOD), ("$T_0$", "%.3f BJD" % T0),
            ("$R_p/R_\\star$", "%.4f" % K_RATIO), ("$a/R_\\star$", "%.2f" % A_RS),
            ("$i$", "%.1f deg" % np.degrees(INC)),
            ("$b$", "%.2f" % B_IMPACT), ("$T_{14}$", "%.2f h" % dur_h),
            ("depth", "%d ppm" % round(depth * 1e6)),
            ("$u_1,\\,u_2$", "%.2f, %.2f" % (U1, U2)),
            ("$R_p$", "1.28 $R_{\\rm Jup}$"),
            ("$T_{\\rm eq}$", "1180 K")]
    y = 0.945
    for k, (nm, val) in enumerate(rows):
        ax_e.text(0.07, y, nm, fontsize=7.0, ha="left", va="center",
                  color="#33475f")
        ax_e.text(0.93, y, val, fontsize=7.0, ha="right", va="center",
                  color="#1a1a1a", family="DejaVu Sans Mono")
        if k < len(rows) - 1:
            ax_e.plot([0.06, 0.94], [y - 0.0425] * 2, color="#e0e3e7",
                      lw=0.5, zorder=1)
        y -= 0.085

    for ax, lab in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"),
                    (ax_d, "(d)"), (ax_e, "(e)")]:
        p = ax.get_position()
        add_panel_label(ax, lab, x=-0.030 / p.width, y=1.0 + 0.026 / p.height,
                        size=10)

    handles = [Line2D([], [], marker="o", ls="none", ms=3, mfc="#9fb2c6",
                      mec="none", label="10 min cadence"),
               Line2D([], [], marker="o", ls="none", ms=4.5, mfc=NAVY,
                      mec="white", mew=0.6, label="binned (9 min)"),
               Line2D([], [], color=GOLD, lw=2.0, label="limb-darkened model"),
               Line2D([], [], color=RUST, lw=1.0, ls=(0, (5, 2.6)),
                      label="stellar variability trend"),
               Line2D([], [], marker="^", ls="none", ms=5, mfc=GOLD,
                      mec="#6d520f", mew=0.5, label="predicted mid-transit")]
    savefig(fig, "238_exoplanet_transit.png")


if __name__ == "__main__":
    main()
