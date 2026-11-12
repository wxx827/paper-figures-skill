"""2x3 hyper-parameter search landscape: heatmap + contours + marginal profiles.

Each panel is a two-dimensional slice through an eight-dimensional random
search, with the remaining hyper-parameters held at their optimum. The main
axis shows the interpolated validation AUC (imshow) with iso-AUC contours on
top and a star at the slice optimum; the top and right strips carry the
marginal profiles, i.e. the best AUC reachable at each value of one axis.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter
from pubstyle import set_pub_style, savefig

CMAP = LinearSegmentedColormap.from_list(
    "auc_land",
    ["#31435e", "#436a86", "#6797a2", "#9dc0aa", "#d9d29c", "#e5ad6c",
     "#cf7a4d"])

NX, NY = 52, 44
FLOOR = 0.615                      # AUC of the worst corner of the space
SEED = 1357

SPECS = [
    dict(xn="Learning rate", xr=(0.003, 0.30), xlog=True, xf="{:.3g}",
         yn="Max depth", yr=(2, 12), ylog=False, yf="{:.0f}",
         ux=0.46, vy=0.44, ax=1.9, ay=1.4, axy=0.75, peak=0.9035),
    dict(xn="Subsample", xr=(0.40, 1.00), xlog=False, xf="{:.2f}",
         yn="Colsample bytree", yr=(0.40, 1.00), ylog=False, yf="{:.2f}",
         ux=0.66, vy=0.58, ax=1.1, ay=0.9, axy=-0.30, peak=0.9018),
    dict(xn="L2 penalty $\\lambda$", xr=(1e-3, 1e2), xlog=True, xf="{:.3g}",
         yn="Min child weight", yr=(0.5, 50), ylog=True, yf="{:.3g}",
         ux=0.52, vy=0.38, ax=1.5, ay=1.8, axy=0.45, peak=0.9006),
    dict(xn="Number of trees", xr=(50, 3000), xlog=True, xf="{:.0f}",
         yn="Learning rate", yr=(0.003, 0.30), ylog=True, yf="{:.3g}",
         ux=0.60, vy=0.46, ax=1.3, ay=1.7, axy=1.05, peak=0.9042),
    dict(xn="Colsample bylevel", xr=(0.3, 1.0), xlog=False, xf="{:.2f}",
         yn="$\\gamma$ (split loss)", yr=(1e-3, 5), ylog=True, yf="{:.3g}",
         ux=0.62, vy=0.30, ax=1.0, ay=2.2, axy=-0.25, peak=0.8994),
    dict(xn="Max delta step", xr=(0, 10), xlog=False, xf="{:.0f}",
         yn="Scale pos weight", yr=(0.5, 20), ylog=True, yf="{:.3g}",
         ux=0.34, vy=0.50, ax=1.6, ay=1.2, axy=0.20, peak=0.9010),
]


def landscape(spec):
    """Smooth, bounded AUC surface over the normalised search rectangle."""
    rng = np.random.default_rng(SEED + int(spec["peak"] * 1e4))
    u = np.linspace(0, 1, NX)
    v = np.linspace(0, 1, NY)
    U, V = np.meshgrid(u, v)
    du, dv = U - spec["ux"], V - spec["vy"]
    q = np.clip(spec["ax"] * du ** 2 + spec["ay"] * dv ** 2
                + spec["axy"] * du * dv, 0, None)
    z = spec["peak"] - (spec["peak"] - FLOOR) * (1 - np.exp(-1.35 * q))
    noise = gaussian_filter(rng.normal(0, 1, (NY, NX)), sigma=3.4)
    return z + noise / max(noise.std(), 1e-9) * 0.0024


def to_value(spec, frac, axis):
    lo, hi = spec[f"{axis}r"]
    return lo * (hi / lo) ** frac if spec[f"{axis}log"] else lo + frac * (hi - lo)


def axis_ticks(spec, axis, n_grid, n_ticks=4):
    pos = np.linspace(0, n_grid - 1, n_ticks)
    labels = [spec[f"{axis}f"].format(to_value(spec, p / (n_grid - 1), axis))
              for p in pos]
    return pos, labels


def panel(fig, cell, spec, label, norm):
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=cell,
                                 width_ratios=[5.2, 0.85],
                                 height_ratios=[0.85, 5.2],
                                 wspace=0.05, hspace=0.05)
    axt = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[1, 1])

    z = landscape(spec)
    im = ax.imshow(z, cmap=CMAP, norm=norm, origin="lower", aspect="auto",
                   extent=(-0.5, NX - 0.5, -0.5, NY - 0.5),
                   interpolation="bilinear")
    levels = np.linspace(z.min(), z.max(), 9)[1:-1]
    cs = ax.contour(np.arange(NX), np.arange(NY), z, levels=levels,
                    colors="#ffffff", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, levels[1::3], fmt="%.3f", fontsize=6.0, inline=True,
              colors="#ffffff")

    jy, jx = np.unravel_index(np.argmax(z), z.shape)
    ax.plot(jx, jy, marker="*", ms=11, color="#ffffff",
            markeredgecolor="#1a1a1a", markeredgewidth=0.8, zorder=6)

    xp, xl = axis_ticks(spec, "x", NX)
    yp, yl = axis_ticks(spec, "y", NY)
    ax.set_xticks(xp, xl, fontsize=6.6)
    ax.set_yticks(yp, yl, fontsize=6.6)
    ax.set_xlim(-0.5, NX - 0.5)
    ax.set_ylim(-0.5, NY - 0.5)
    ax.set_xlabel(spec["xn"], fontsize=7.6, labelpad=1.5)
    ax.set_ylabel(spec["yn"], fontsize=7.6, labelpad=1.5)
    ax.tick_params(length=2.4, colors="#333333")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")

    # ---- marginal profiles ----------------------------------------------
    prof_x, prof_y = z.max(axis=0), z.max(axis=1)
    lo_x = prof_x.min() - (prof_x.max() - prof_x.min()) * 0.30
    lo_y = prof_y.min() - (prof_y.max() - prof_y.min()) * 0.30
    axt.fill_between(np.arange(NX), lo_x, prof_x, color="#8fa8bb",
                     alpha=0.35, lw=0)
    axt.plot(np.arange(NX), prof_x, color="#33475c", lw=1.0)
    axt.plot(jx, prof_x[jx], marker="*", ms=6, color="#ffffff",
             markeredgecolor="#1a1a1a", markeredgewidth=0.6, clip_on=False)
    axr.fill_betweenx(np.arange(NY), lo_y, prof_y, color="#8fa8bb",
                      alpha=0.35, lw=0)
    axr.plot(prof_y, np.arange(NY), color="#33475c", lw=1.0)
    axr.plot(prof_y[jy], jy, marker="*", ms=6, color="#ffffff",
             markeredgecolor="#1a1a1a", markeredgewidth=0.6, clip_on=False)
    axt.set_xlim(-0.5, NX - 0.5)
    axt.set_ylim(lo_x, prof_x.max() + (prof_x.max() - lo_x) * 0.12)
    axr.set_ylim(-0.5, NY - 0.5)
    axr.set_xlim(lo_y, prof_y.max() + (prof_y.max() - lo_y) * 0.12)
    for a in (axt, axr):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    bx = to_value(spec, jx / (NX - 1), "x")
    by = to_value(spec, jy / (NY - 1), "y")
    axt.set_title(f"{label}  {spec['xn']}  $\\times$  {spec['yn']}\n"
                  f"best AUC {z.max():.4f}  at  "
                  f"{spec['xf'].format(bx)} / {spec['yf'].format(by)}",
                  fontsize=7.4, fontweight="bold", linespacing=1.5, pad=3,
                  loc="left")
    return im


def main():
    set_pub_style()
    zs = [landscape(s) for s in SPECS]
    norm = Normalize(min(z.min() for z in zs), max(z.max() for z in zs))

    fig = plt.figure(figsize=(11.2, 7.4))
    gs = GridSpec(2, 3, figure=fig, left=0.048, right=0.915, top=0.885,
                  bottom=0.095, hspace=0.46, wspace=0.30)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    im = None
    for k, spec in enumerate(SPECS):
        im = panel(fig, gs[k // 3, k % 3], spec, labels[k], norm)

    cax = fig.add_axes([0.933, 0.16, 0.013, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Validation AUC (5-fold mean)", fontsize=7.6, labelpad=3)
    cb.ax.tick_params(labelsize=6.8, length=2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).
    savefig(fig, "68_hyperparam_heatmap.png")


if __name__ == "__main__":
    main()
