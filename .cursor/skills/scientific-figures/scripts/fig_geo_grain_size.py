"""Grain-size distribution suite for a downstream sample transect.

Six samples are described by log-normal mixtures in phi units; everything else
on the figure is derived from those same distributions.  The main panel is the
cumulative curve set on a logarithmic diameter axis with the Wentworth class
boundaries ruled in.  The ternary panel places each sample in sand-silt-clay
space by integrating its own distribution, and the lower panel plots the Folk
and Ward sorting and skewness coefficients computed from its phi percentiles.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import (set_pub_style, savefig, add_panel_label, get_palette,
                      style_axes)

# (label, [(mean phi, sd phi, weight), ...])
SAMPLES = [
    ("S1 bar head", [(-0.6, 0.45, 0.92), (1.4, 0.80, 0.08)]),
    ("S2 bar tail", [(1.5, 0.55, 0.88), (3.8, 1.00, 0.12)]),
    ("S3 point bar", [(2.8, 0.70, 0.80), (5.2, 1.20, 0.20)]),
    ("S4 levee", [(3.7, 0.90, 0.66), (6.2, 1.40, 0.34)]),
    ("S5 crevasse", [(5.6, 1.15, 0.58), (8.0, 1.30, 0.42)]),
    ("S6 floodbasin", [(7.2, 1.30, 0.55), (9.4, 1.20, 0.45)]),
]
# Wentworth boundaries in mm and the class names between them
BOUNDS = [0.004, 0.063, 2.0]
CLASSES = ["clay", "silt", "sand", "gravel"]
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
S3 = np.sqrt(3.0) / 2.0


def phi_cdf(phi, comps):
    from scipy.special import erf
    out = np.zeros_like(phi)
    for mu, sd, w in comps:
        out += w * 0.5 * (1.0 + erf((phi - mu) / (sd * np.sqrt(2.0))))
    return out


def tern(sand, silt, clay):
    """Sand at lower-left, silt at lower-right, clay at apex."""
    tot = sand + silt + clay
    s, m, c = sand / tot, silt / tot, clay / tot
    return m + 0.5 * c, S3 * c


def main():
    set_pub_style()

    phi = np.linspace(-4.0, 12.0, 3000)
    dia = 2.0 ** (-phi)
    cols = get_palette("cb_sequential_blue", 9)[3:]

    stats = []
    for (name, comps), col in zip(SAMPLES, cols):
        cdf = phi_cdf(phi, comps)
        q = {p: float(np.interp(p / 100.0, cdf, phi))
             for p in (5, 16, 50, 84, 95)}
        sort = ((q[84] - q[16]) / 4.0 + (q[95] - q[5]) / 6.6)
        skew = ((q[16] + q[84] - 2 * q[50]) / (2 * (q[84] - q[16])) +
                (q[5] + q[95] - 2 * q[50]) / (2 * (q[95] - q[5])))
        f = np.interp([-1.0, 4.0, 8.0], phi, cdf)
        gravel = f[0]
        sand, silt = f[1] - f[0], f[2] - f[1]
        clay = 1.0 - f[2]
        stats.append(dict(name=name, color=col, cdf=cdf, q=q, sort=sort,
                          skew=skew, gravel=gravel, sand=sand, silt=silt,
                          clay=clay, d50=2.0 ** (-q[50])))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.16, 0.80], left=0.058,
                          right=0.985, top=0.870, bottom=0.135, hspace=0.40,
                          wspace=0.20)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ---- (a) cumulative curves -------------------------------------------
    for st in stats:
        ax.semilogx(dia, 100.0 * (1.0 - st["cdf"]), color=st["color"],
                    lw=1.8, zorder=4, solid_capstyle="round")
        ax.plot([st["d50"]], [50.0], marker="o", ms=4.2, mfc="white",
                mec=st["color"], mew=1.2, zorder=5)
    for b in BOUNDS:
        ax.axvline(b, color=RULE, lw=0.9, ls=(0, (4, 2.4)), zorder=2)
    edges = [1e-3] + BOUNDS + [3e1]
    for lo, hi, nm in zip(edges[:-1], edges[1:], CLASSES):
        ax.text(np.sqrt(lo * hi), 96.5, nm, ha="center", va="center",
                fontsize=7.2, color=SUB, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="none", alpha=0.88))
    ax.axhline(50.0, color="#cccccc", lw=0.7, zorder=1)
    ax.set_xlim(1e-3, 3e1)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Particle diameter (mm)")
    ax.set_ylabel("Cumulative mass finer (%)")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(color="#ececec", lw=0.5, which="major")
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title("Cumulative grain-size curves", fontsize=8.5,
                 fontweight="bold")

    d50_span = max(s["d50"] for s in stats) / min(s["d50"] for s in stats)
    ax.text(0.985, 0.045, f"$D_{{50}}$ spans {d50_span:.0f}x, sorting "
            f"{min(s['sort'] for s in stats):.2f} to "
            f"{max(s['sort'] for s in stats):.2f} $\\phi$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=SUB,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    # ---- (b) sand-silt-clay ternary --------------------------------------
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3], [0.0, 0.0]])
    ax_t.plot(verts[:, 0], verts[:, 1], color=RULE, lw=1.0, zorder=3)
    for f in (0.25, 0.5, 0.75):
        for a, b in (((f, 0.0), (f / 2 + 0.5, S3 * (1 - f))),
                     ((f / 2, S3 * f), (1 - f / 2, S3 * f)),
                     ((f / 2, S3 * f), (f, 0.0))):
            ax_t.plot([a[0], b[0]], [a[1], b[1]], color="#e6e6e6", lw=0.5,
                      zorder=1)
    for a, b in ((tern(50, 50, 0), tern(50, 0, 50)),
                 ((tern(0, 50, 50)), tern(50, 50, 0)),
                 (tern(50, 0, 50), tern(0, 50, 50))):
        ax_t.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.8,
                  ls=(0, (3, 2)), zorder=2)
    mud = tern(34, 34, 32)
    ax_t.text(mud[0], mud[1], "mud", ha="center", va="center", fontsize=6.4,
              color=MUTED, style="italic", zorder=2)
    for st in stats:
        x, y = tern(st["sand"], st["silt"], st["clay"])
        ax_t.plot([x], [y], marker="o", ms=5.2, mfc=st["color"],
                  mec="white", mew=0.8, zorder=6)
    ax_t.text(-0.03, -0.045, "sand", ha="right", va="top", fontsize=7.0,
              color=INK, fontweight="bold")
    ax_t.text(1.03, -0.045, "silt", ha="left", va="top", fontsize=7.0,
              color=INK, fontweight="bold")
    ax_t.text(0.5, S3 + 0.045, "clay", ha="center", va="bottom",
              fontsize=7.0, color=INK, fontweight="bold")
    ax_t.set_aspect("equal")
    ax_t.set_xlim(-0.14, 1.14)
    ax_t.set_ylim(-0.16, S3 + 0.16)
    ax_t.axis("off")
    ax_t.set_title("Textural classification", fontsize=8.5,
                   fontweight="bold")

    # ---- (c) Folk and Ward sorting vs skewness ---------------------------
    for x in (0.35, 0.71, 1.00):
        ax_s.axvline(x, color="#cccccc", lw=0.7, ls=(0, (3, 2)), zorder=1)
    for x, lab in ((0.53, "well"), (0.855, "moderate"), (1.45, "poor")):
        ax_s.text(x, 0.028, lab, transform=ax_s.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=6.2, color=MUTED,
                  style="italic")
    ax_s.axhline(0.0, color="#808080", lw=0.8, zorder=1)
    for st in stats:
        ax_s.plot([st["sort"]], [st["skew"]], marker="o", ms=5.2,
                  mfc=st["color"], mec="white", mew=0.8, zorder=4)
        ax_s.annotate(st["name"].split()[0], xy=(st["sort"], st["skew"]),
                      xytext=(4.5, 3.0), textcoords="offset points",
                      fontsize=6.2, color=SUB)
    ax_s.set_xlim(0.30, max(s["sort"] for s in stats) + 0.24)
    ax_s.set_xlabel("Sorting $\\sigma_I$ ($\\phi$)", fontsize=7.4)
    ax_s.set_ylabel("Skewness $Sk_I$ (-)", fontsize=7.4)
    ax_s.tick_params(labelsize=6.5)
    ax_s.grid(color="#ececec", lw=0.5)
    ax_s.set_axisbelow(True)
    style_axes(ax_s)
    ax_s.set_title("Sorting against skewness", fontsize=8.5,
                   fontweight="bold", pad=11)

    def d50_text(d):
        return f"{d:.2f} mm" if d >= 1.0 else f"{d * 1000:.0f} $\\mu$m"

    handles = [Line2D([0], [0], color=st["color"], lw=1.8, marker="o", ms=4.4,
                      mfc="white", mec=st["color"],
                      label=f"{st['name']}  $D_{{50}}$ = "
                            f"{d50_text(st['d50'])}")
               for st in stats]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), ncol=6, frameon=False,
               fontsize=6.8, handlelength=1.9, columnspacing=1.3)

    add_panel_label(ax, "(a)", x=-0.085, y=1.045)
    add_panel_label(ax_t, "(b)", x=-0.075, y=1.045)
    add_panel_label(ax_s, "(c)", x=-0.145, y=1.045)
    savefig(fig, "409_geo_grain_size.png")


if __name__ == "__main__":
    main()
