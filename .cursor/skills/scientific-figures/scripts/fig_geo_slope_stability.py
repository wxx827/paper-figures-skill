"""Circular slip-surface analysis of a layered cut slope.

Every factor of safety on this figure comes from the same ordinary method of
slices run over the layered profile: unit weights are integrated through the
soil column, strength is taken from the layer at each slice base, and pore
pressure from the phreatic surface.  The slip circles in the section are
coloured on the same scale as the search chart below them, whose axes are the
circle-centre coordinates - each centre uses the radius that puts its arc
through the toe.  The critical circle is the grid minimum, not a hand pick.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import (set_pub_style, savefig, add_panel_label, get_cmap,
                      lighten, darken)
from figkit import safe_colorbar

# (top, bottom, gamma kN/m3, c' kPa, phi' deg, colour, hatch, name)
LAYERS = [
    (40.0, 22.0, 18.5, 12.0, 30.0, lighten("#c0603f", 0.72), "xx",
     "Fill"),
    (22.0, 12.0, 19.2, 20.0, 24.0, lighten("#2e7d5b", 0.70), "----",
     "Silty clay"),
    (12.0, -8.0, 20.0, 42.0, 20.0, lighten("#8e6bbf", 0.72), "--..",
     "Stiff clay"),
    (-8.0, -26.0, 22.0, 300.0, 38.0, lighten("#3b5b92", 0.62), "\\\\",
     "Mudstone"),
]
CREST, TOE = (44.0, 32.0), (80.0, 8.0)
GW = 9.81
XMIN, XMAX = 0.0, 124.0
NS = 44
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"


def ground(x):
    return np.interp(x, [XMIN, CREST[0], TOE[0], XMAX],
                     [CREST[1], CREST[1], TOE[1], TOE[1]])


def water(x):
    return np.interp(x, [XMIN, 62.0, 96.0, XMAX], [27.5, 14.0, 9.0, 9.0])


def layer_at(y):
    for lay in LAYERS:
        if lay[1] <= y < lay[0]:
            return lay
    return LAYERS[-1]


def fos(xc, yc):
    """Ordinary (Fellenius) method of slices for a toe circle."""
    r = float(np.hypot(TOE[0] - xc, TOE[1] - yc))
    if r < 12.0:
        return np.nan
    xg = np.linspace(max(xc - r + 1e-6, XMIN), min(xc + r - 1e-6, XMAX), 600)
    yb = yc - np.sqrt(np.maximum(r ** 2 - (xg - xc) ** 2, 0.0))
    ins = np.where(yb < ground(xg) - 0.10)[0]
    if ins.size < 20:
        return np.nan
    xe = np.linspace(xg[ins[0]], xg[ins[-1]], NS + 1)
    dx = xe[1] - xe[0]
    xm = 0.5 * (xe[:-1] + xe[1:])
    yb = yc - np.sqrt(np.maximum(r ** 2 - (xm - xc) ** 2, 0.0))
    if yb.min() < LAYERS[-1][0]:
        return np.nan               # circle would cut into the mudstone
    top = ground(xm)
    if np.any(top - yb < -0.01):
        return np.nan
    # sliding is toward +x, so the base angle is positive in the upper
    # (active) slices and negative in the passive slices near the toe
    sina = (xc - xm) / r
    cosa = np.sqrt(np.maximum(1.0 - sina ** 2, 1e-6))
    dl = dx / cosa

    w = np.zeros_like(xm)
    for lt, lb, gam, *_ in LAYERS:
        w += gam * np.maximum(np.minimum(top, lt) - np.maximum(yb, lb), 0.0)
    w *= dx
    c = np.array([layer_at(v)[3] for v in yb])
    tanphi = np.tan(np.deg2rad([layer_at(v)[4] for v in yb]))
    u = GW * np.maximum(water(xm) - yb, 0.0)

    num = np.sum(c * dl + np.maximum(w * cosa - u * dl, 0.0) * tanphi)
    den = np.sum(w * sina)
    return num / den if den > 1e-6 else np.nan


def arc(xc, yc):
    r = float(np.hypot(TOE[0] - xc, TOE[1] - yc))
    xg = np.linspace(max(xc - r, XMIN), min(xc + r, XMAX), 500)
    yb = yc - np.sqrt(np.maximum(r ** 2 - (xg - xc) ** 2, 0.0))
    m = yb < ground(xg) - 0.05
    return np.where(m, xg, np.nan), np.where(m, yb, np.nan)


def main():
    set_pub_style(**{"hatch.linewidth": 0.45})

    cxs = np.linspace(44.0, 92.0, 60)
    cys = np.linspace(24.0, 74.0, 54)
    CX, CY = np.meshgrid(cxs, cys)
    F = np.array([[fos(a, b) for a in cxs] for b in cys])
    j, i = np.unravel_index(np.nanargmin(F), F.shape)
    fmin, cx0, cy0 = F[j, i], CX[j, i], CY[j, i]

    show = [(cx0, cy0)]
    for a in (0.30, 0.62, 0.92):
        for b in (0.24, 0.70):
            xa = cxs[0] + a * (cxs[-1] - cxs[0])
            yb_ = cys[0] + b * (cys[-1] - cys[0])
            if np.isfinite(fos(xa, yb_)):
                show.append((xa, yb_))

    vals = np.array([fos(a, b) for a, b in show])
    cmap = get_cmap("cb_sequential_warm", reverse=True)
    lo = float(np.nanmin(F))
    hi = float(min(np.nanpercentile(F, 88), np.nanmax(F)))
    norm = plt.Normalize(lo, hi)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 0.62],
                          width_ratios=[1.0, 0.56], left=0.055, right=0.955,
                          top=0.870, bottom=0.115, hspace=0.46, wspace=0.20)
    ax = fig.add_subplot(gs[0, :])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])

    # ---- (a) slope section -----------------------------------------------
    xg = np.linspace(XMIN, XMAX, 900)
    gy = ground(xg)
    for lt, lb, gam, c, phi, col, hat, name in LAYERS:
        upper = np.minimum(gy, lt)
        lower = np.minimum(gy, lb)
        ax.fill_between(xg, lower, upper, where=upper > lower, facecolor=col,
                        hatch=hat, edgecolor=darken(col, 0.42),
                        linewidth=0.5, zorder=2)
    ax.plot(xg, gy, color=RULE, lw=1.5, zorder=6)
    ax.plot(xg, water(xg), color="#3d6b8e", lw=1.2, ls=(0, (5, 2.2)),
            zorder=6)
    ax.plot([96.0], [water(np.array(96.0))], marker="v", ms=5.5,
            mfc="#3d6b8e", mec="white", mew=0.7, zorder=7)

    for (a, b), v in zip(show, vals):
        if (a, b) == (cx0, cy0):
            continue
        ax_, ay_ = arc(a, b)
        ax.plot(ax_, ay_, color=cmap(norm(v)), lw=1.2, zorder=4)
    ax_, ay_ = arc(cx0, cy0)
    ax.plot(ax_, ay_, color="#b23a48", lw=2.4, zorder=8)
    ax.annotate(f"critical circle  $F_s$ = {fmin:.2f}\n"
                f"centre ({cx0:.0f}, {cy0:.0f}) m",
                xy=(float(np.nanmean(ax_)), float(np.nanmin(ay_))),
                xytext=(90.0, 26.0), fontsize=7.2, color="#b23a48",
                fontweight="bold", ha="left", va="center", zorder=9,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color="#b23a48", lw=0.9,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_aspect("equal")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-10.0, 40.0)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Elevation (m)")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Cut slope with the searched family of toe circles",
                 fontsize=8.5, fontweight="bold")

    # ---- (b) centre search chart -----------------------------------------
    lv = np.linspace(lo, hi, 13)
    cf = ax_c.contourf(CX, CY, F, levels=lv, cmap=cmap, extend="max")
    cf.set_edgecolor("face")
    cs = ax_c.contour(CX, CY, F, levels=lv[::2], colors="#8c5a2b",
                      linewidths=0.5)
    ax_c.clabel(cs, fmt="%.2f", fontsize=6.0, inline=True)
    if lo < 1.30 < hi:
        ax_c.contour(CX, CY, F, levels=[1.30], colors="#1a1a1a",
                     linewidths=1.2, linestyles=[(0, (5, 2.2))])
    ax_c.plot([cx0], [cy0], marker="+", ms=10, mew=1.8, color="#b23a48",
              zorder=5)
    ax_c.annotate(f"min $F_s$ = {fmin:.2f}", xy=(cx0, cy0),
                  xytext=(7.0, 5.0), textcoords="offset points",
                  fontsize=6.8, color="#b23a48", fontweight="bold")
    for (a, b), v in zip(show[1:], vals[1:]):
        ax_c.plot([a], [b], marker="o", ms=3.4, mfc="white", mec=RULE,
                  mew=0.7, zorder=4)
    ax_c.set_xlabel("Circle centre easting (m)", fontsize=7.4)
    ax_c.set_ylabel("Centre elevation (m)", fontsize=7.4)
    ax_c.tick_params(labelsize=6.6)
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax_c.set_title("Factor of safety over the centre grid", fontsize=8.5,
                   fontweight="bold")
    safe_colorbar(fig, cf, ax_c, label="factor of safety (-)", width=0.11,
                  pad=0.08)

    # ---- (c) soil parameter card -----------------------------------------
    ax_t.axis("off")
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.set_title("Design soil parameters", fontsize=8.5,
                   fontweight="bold", loc="left", x=0.0)
    cols = (0.235, 0.53, 0.72, 0.92)
    ax_t.text(0.02, 0.90, "unit", fontsize=6.8, color=SUB, ha="left")
    for x, hd in zip(cols, ("layer", "$\\gamma$ (kN m$^{-3}$)",
                            "$c'$ (kPa)", "$\\phi'$ ($\\degree$)")):
        ax_t.text(x, 0.90, hd, fontsize=6.8, color=SUB,
                  ha="left" if hd == "layer" else "right")
    ax_t.plot([0.0, 1.0], [0.855, 0.855], color="#cccccc", lw=0.7)
    for k, (lt, lb, gam, c, phi, col, hat, name) in enumerate(LAYERS):
        y = 0.755 - k * 0.165
        ax_t.add_patch(Rectangle((0.02, y - 0.048), 0.175, 0.096,
                                 facecolor=col, edgecolor=darken(col, 0.42),
                                 linewidth=0.6, hatch=hat))
        ax_t.text(cols[0], y, name, fontsize=7.0, color=INK, ha="left",
                  va="center")
        for x, v in zip(cols[1:], (f"{gam:.1f}", f"{c:.0f}", f"{phi:.0f}")):
            ax_t.text(x, y, v, fontsize=7.0, color=INK, ha="right",
                      va="center")
    n_valid = int(np.isfinite(F).sum())
    ax_t.text(0.0, 0.045, f"phreatic surface drawn dashed; {n_valid} valid "
              f"trial circles searched", fontsize=6.4, color=MUTED,
              ha="left", va="bottom")

    handles = [
        Line2D([0], [0], color="#b23a48", lw=2.4, label="critical circle"),
        Line2D([0], [0], color="#3d6b8e", lw=1.2, ls=(0, (5, 2.2)),
               label="phreatic surface"),
        Line2D([0], [0], color=INK, lw=1.2, ls=(0, (5, 2.2)),
               label="$F_s$ = 1.30 design requirement"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.004), ncol=3, frameon=False,
               fontsize=7.0, handlelength=2.0, columnspacing=2.0)

    add_panel_label(ax, "(a)", x=-0.052, y=1.035)
    add_panel_label(ax_c, "(b)", x=-0.100, y=1.045)
    add_panel_label(ax_t, "(c)", x=-0.055, y=1.045)
    verdict = "meets" if fmin >= 1.30 else "falls short of"
    savefig(fig, "411_geo_slope_stability.png")


if __name__ == "__main__":
    main()
