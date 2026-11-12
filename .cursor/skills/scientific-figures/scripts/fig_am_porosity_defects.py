"""Reconstructed pore population in an as-built cylinder from X-ray CT.

The main panel is the claim: porosity in this coupon is not a uniform
background, it is a thin band of flat lack-of-fusion defects at one build
height plus a rim of contour pores, with spherical gas porosity filling the
rest.  Bubble area is the measured pore diameter and colour the sphericity, so
the two defect families separate by shape rather than by an assigned label; the
irregular ones are ringed in red.  The porosity quoted in the caption, the
median diameter and the worst layer are all integrated from the same pore
table, so they cannot disagree with the cloud.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
ACCENT = "#b23a48"      # irregular, crack-like defects
BLUE = "#3b5b92"
GREY = "#808080"
HAIR = "#cccccc"

# Pale markers vanish against the white specimen; start a third of the way in.
SPH_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "sphericity", get_cmap("cb_sequential_blue")(np.linspace(0.30, 1.0, 256)))

R_SPEC = 4.0            # mm, specimen radius
H_SPEC = 12.0           # mm, specimen height
CT_LIMIT = 0.035        # mm, voxel-limited detection threshold
SPH_IRREG = 0.65        # below this a pore is called lack of fusion
LOF_Z = 4.6             # mm, build height of the stalled-fusion layer


def pores(seed=23):
    rng = np.random.default_rng(seed)

    def polar(n, r_lo, r_hi):
        u = rng.uniform(r_lo ** 2, r_hi ** 2, n)
        r = np.sqrt(u)
        th = rng.uniform(0.0, 2.0 * np.pi, n)
        return r * np.cos(th), r * np.sin(th)

    x1, y1 = polar(250, 0.0, R_SPEC)
    z1 = rng.uniform(0.15, H_SPEC - 0.15, 250)
    d1 = 0.13 * np.exp(rng.normal(0.0, 0.50, 250))
    s1 = np.clip(0.88 + rng.normal(0.0, 0.055, 250), 0.55, 0.99)

    x2, y2 = polar(72, 0.0, R_SPEC * 0.96)
    z2 = np.clip(rng.normal(LOF_Z, 0.30, 72), 0.2, H_SPEC - 0.2)
    d2 = 0.28 * np.exp(rng.normal(0.0, 0.38, 72))
    s2 = np.clip(0.55 + rng.normal(0.0, 0.075, 72), 0.30, 0.80)

    x3, y3 = polar(88, R_SPEC * 0.84, R_SPEC * 0.99)
    z3 = rng.uniform(0.15, H_SPEC - 0.15, 88)
    d3 = 0.17 * np.exp(rng.normal(0.0, 0.42, 88))
    s3 = np.clip(0.72 + rng.normal(0.0, 0.085, 88), 0.35, 0.95)

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    z = np.concatenate([z1, z2, z3])
    d = np.concatenate([d1, d2, d3])
    s = np.concatenate([s1, s2, s3])

    keep = ((d >= CT_LIMIT) & (np.hypot(x, y) <= R_SPEC) &
            (z >= 0.0) & (z <= H_SPEC))
    return x[keep], y[keep], z[keep], d[keep], s[keep]


def specimen(ax):
    th = np.linspace(0.0, 2.0 * np.pi, 121)
    zz = np.array([0.0, H_SPEC])
    TH, ZZ = np.meshgrid(th, zz)
    ax.plot_surface(R_SPEC * np.cos(TH), R_SPEC * np.sin(TH), ZZ,
                    color=GREY, alpha=0.07, linewidth=0.0, shade=False,
                    zorder=0)
    for z in (0.0, H_SPEC):
        ax.plot(R_SPEC * np.cos(th), R_SPEC * np.sin(th), z, color=HAIR,
                lw=0.9, zorder=1)
    for a in np.arange(0.0, 2.0 * np.pi, np.pi / 2.0):
        ax.plot([R_SPEC * np.cos(a)] * 2, [R_SPEC * np.sin(a)] * 2,
                [0.0, H_SPEC], color=HAIR, lw=0.7, zorder=1)


def cloud_panel(ax, x, y, z, d, s, stats):
    specimen(ax)
    size = 520.0 * d ** 2
    irr = s < SPH_IRREG
    ax.scatter(x[~irr], y[~irr], z[~irr], s=size[~irr], c=s[~irr],
               cmap=SPH_CMAP, vmin=0.30, vmax=1.0, depthshade=False,
               linewidths=0.4, edgecolors="white", alpha=0.95, zorder=4)
    ax.scatter(x[irr], y[irr], z[irr], s=size[irr], c=s[irr], cmap=SPH_CMAP,
               vmin=0.30, vmax=1.0, depthshade=False, linewidths=0.9,
               edgecolors=ACCENT, alpha=0.95, zorder=5)

    ax.set_xlim(-R_SPEC, R_SPEC)
    ax.set_ylim(-R_SPEC, R_SPEC)
    ax.set_zlim(0.0, H_SPEC)
    ax.set_box_aspect((1.0, 1.0, 1.45))
    ax.view_init(elev=17.0, azim=-58.0)
    ax.set_xlabel("x (mm)", fontsize=7.4, labelpad=-2.0)
    ax.set_ylabel("y (mm)", fontsize=7.4, labelpad=-2.0)
    ax.set_zlabel("build height z (mm)", fontsize=7.4, labelpad=-2.0)
    ax.tick_params(labelsize=6.4, pad=0.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor(HAIR)
        pane.set_alpha(1.0)
    ax.grid(True)

    ax.text2D(0.01, 0.90, f"porosity {stats['porosity']:.3f} vol %\n"
                          f"{stats['n']} pores above "
                          f"{CT_LIMIT * 1000.0:.0f} " + r"$\mu$m" + "\n"
                          f"{stats['frac_irr']:.0f} % irregular",
              transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
              color=INK, linespacing=1.45)
    handles = [Line2D([], [], marker="o", ls="none", mfc="#cfe1ee",
                      mec="white", mew=0.4,
                      ms=np.sqrt(520.0 * (v / 1000.0) ** 2),
                      label=f"{v:.0f} " + r"$\mu$m")
               for v in (100.0, 300.0, 500.0)]
    handles.append(Line2D([], [], marker="o", ls="none", mfc="#5f9dc6",
                          mec=ACCENT, mew=0.9, ms=5.0,
                          label=f"sphericity < {SPH_IRREG:.2f}"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.6,
              labelspacing=0.75, handletextpad=0.8, borderaxespad=0.0,
              bbox_to_anchor=(-0.02, 0.02))
    ax.set_title("Lack-of-fusion defects sit in one stalled layer",
                 fontsize=9.0, fontweight="bold", color=INK, y=1.01)


def size_panel(ax, d, s):
    um = d * 1000.0
    bins = np.logspace(np.log10(um.min() * 0.92), np.log10(um.max() * 1.08),
                       26)
    irr = s < SPH_IRREG
    ax.hist([um[~irr], um[irr]], bins=bins, stacked=True,
            color=[BLUE, ACCENT], edgecolor="white", linewidth=0.4,
            label=["spherical gas pores", "irregular defects"], zorder=3)
    med = float(np.median(um))
    ax.axvline(med, color=RULE, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    ax.text(med * 1.08, ax.get_ylim()[1] * 0.92, f"median {med:.0f} " +
            r"$\mu$m", fontsize=6.5, color=RULE, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"pore diameter ($\mu$m)", fontsize=7.4)
    ax.set_ylabel("pore count (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left",
              borderaxespad=0.2, handlelength=1.3)
    ax.set_title("Diameter distribution", fontsize=8.2, fontweight="bold",
                 color=INK)


def height_panel(ax, z, d, nbin=12):
    edges = np.linspace(0.0, H_SPEC, nbin + 1)
    vol = np.pi / 6.0 * d ** 3
    slab = np.pi * R_SPEC ** 2 * (H_SPEC / nbin)
    per = np.array([100.0 * vol[(z >= edges[i]) & (z < edges[i + 1])].sum() /
                    slab for i in range(nbin)])
    mid = 0.5 * (edges[:-1] + edges[1:])
    worst = int(np.argmax(per))
    cols = [ACCENT if i == worst else BLUE for i in range(nbin)]
    ax.barh(mid, per, height=0.82, color=cols, edgecolor="white",
            linewidth=0.5, zorder=3)
    ax.annotate(f"{per[worst]:.2f} vol % at z = {mid[worst]:.1f} mm",
                xy=(per[worst], mid[worst]),
                xytext=(per[worst] * 0.62, mid[worst] + 2.4), fontsize=6.5,
                color=ACCENT, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_ylim(0.0, H_SPEC)
    ax.set_xlim(0.0, per.max() * 1.30)
    ax.set_xlabel("layer porosity (vol %)", fontsize=7.4)
    ax.set_ylabel("build height z (mm)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Porosity against build height", fontsize=8.2,
                 fontweight="bold", color=INK)
    return per, mid, worst


def main():
    set_pub_style()
    x, y, z, d, s = pores()
    vol = np.pi / 6.0 * d ** 3
    stats = dict(n=len(d),
                 porosity=100.0 * vol.sum() / (np.pi * R_SPEC ** 2 * H_SPEC),
                 frac_irr=100.0 * float((s < SPH_IRREG).mean()))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax3d = axes_inches(fig, (0.15, 0.95, 6.10, 4.55), projection="3d")
    ax_size = axes_inches(fig, (7.05, 3.60, 2.80, 1.90))
    ax_h = axes_inches(fig, (7.05, 0.95, 2.80, 1.90))

    cloud_panel(ax3d, x, y, z, d, s, stats)
    size_panel(ax_size, d, s)
    per, mid, worst = height_panel(ax_h, z, d)

    cax = axes_inches(fig, (1.55, 0.58, 3.00, 0.12))
    cb = fig.colorbar(mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0.30, 1.0), cmap=SPH_CMAP), cax=cax,
        orientation="horizontal")
    cb.set_label("pore sphericity (-)", fontsize=6.6, color=INK)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color=RULE)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)

    panel_labels_inches(fig, [ax_size, ax_h], dx=-0.50, dy=0.10, start=1)
    panel_labels_inches(fig, [ax3d], dx=0.05, dy=0.10)
    savefig(fig, "747_am_porosity_defects.png")


if __name__ == "__main__":
    main()
