"""Event-related potentials with a row of scalp topographies."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Arc
from scipy.interpolate import RBFInterpolator

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

HEAD_R = 0.55

# 32-channel 10-20 montage, azimuthal-equidistant 2-D projection
MONTAGE = {
    "Fp1": (-0.14, 0.470), "Fp2": (0.14, 0.470),
    "AF3": (-0.20, 0.372), "AF4": (0.20, 0.372),
    "F7": (-0.425, 0.278), "F3": (-0.225, 0.262), "Fz": (0.0, 0.252),
    "F4": (0.225, 0.262), "F8": (0.425, 0.278),
    "FC5": (-0.345, 0.132), "FC1": (-0.125, 0.128), "FC2": (0.125, 0.128),
    "FC6": (0.345, 0.132),
    "T7": (-0.500, 0.0), "C3": (-0.250, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.250, 0.0), "T8": (0.500, 0.0),
    "CP5": (-0.345, -0.132), "CP1": (-0.125, -0.128), "CP2": (0.125, -0.128),
    "CP6": (0.345, -0.132),
    "P7": (-0.425, -0.278), "P3": (-0.225, -0.262), "Pz": (0.0, -0.252),
    "P4": (0.225, -0.262), "P8": (0.425, -0.278),
    "PO3": (-0.20, -0.372), "PO4": (0.20, -0.372),
    "O1": (-0.14, -0.470), "Oz": (0.0, -0.505), "O2": (0.14, -0.470),
}

COND_COLORS = {"Standard": "#7a8a99", "Deviant": "#c4661f", "Target": "#22557f"}

# (latency ms, peak amplitude uV, width ms) per condition, measured at Pz
COMPONENTS = {
    "Standard": [(80, 1.8, 22), (112, -4.0, 26), (196, 2.6, 34),
                 (256, -1.1, 32), (360, 1.3, 70)],
    "Deviant": [(80, 1.9, 22), (115, -4.5, 26), (196, 2.2, 34),
                (251, -3.3, 32), (356, 4.3, 68), (520, 1.1, 110)],
    "Target": [(80, 2.0, 22), (118, -4.8, 26), (198, 2.0, 34),
               (248, -2.5, 32), (352, 8.6, 72), (525, 2.6, 110)],
}

# scalp generators per component: (x, y, amplitude uV, sigma)
TOPO_SNAPSHOTS = [
    ("N1", 118, [(-0.06, -0.30, -5.6, 0.26), (0.09, -0.31, -5.0, 0.26),
                 (0.0, 0.30, 2.0, 0.30)]),
    ("P2", 196, [(0.0, 0.12, 4.4, 0.30), (0.0, -0.34, -1.4, 0.26)]),
    ("N2", 250, [(-0.03, 0.20, -4.2, 0.26), (0.0, -0.30, 1.6, 0.28)]),
    ("P300", 352, [(0.02, -0.16, 9.2, 0.32), (0.0, 0.34, -1.8, 0.26)]),
    ("LPC", 540, [(0.05, -0.30, 3.6, 0.34), (0.0, 0.26, -0.9, 0.28)]),
]

SIG_WINDOWS = [(215, 292, "**"), (305, 452, "***")]


def erp_traces(t, comps, rng, n_sub=20):
    """Grand-average waveform and SEM across simulated subjects (uV)."""
    subs = np.empty((n_sub, t.size))
    smooth = np.hanning(41)
    smooth /= smooth.sum()
    for i in range(n_sub):
        gain = 1.0 + rng.normal(0, 0.20)
        shift = rng.normal(0, 0.055)
        w = np.zeros_like(t)
        for lat, amp, width in comps:
            w += amp * gain * np.exp(-0.5 * ((t - lat * (1 + shift)) / width) ** 2)
        noise = np.convolve(rng.normal(0, 1.0, t.size), smooth, mode="same")
        subs[i] = w + noise * 9.0
    subs -= subs[:, t < 0].mean(axis=1, keepdims=True)
    return subs.mean(axis=0), subs.std(axis=0, ddof=1) / np.sqrt(n_sub)


def scalp_values(generators, names):
    vals = []
    for n in names:
        x, y = MONTAGE[n]
        v = 0.0
        for gx, gy, amp, sig in generators:
            v += amp * np.exp(-0.5 * (((x - gx) ** 2 + (y - gy) ** 2) / sig ** 2))
        vals.append(v)
    return np.asarray(vals)


def draw_head(ax, lw=1.1, color="#333333"):
    ax.add_patch(Circle((0, 0), HEAD_R, fill=False, lw=lw, ec=color, zorder=5))
    nose = np.array([[-0.105, 0.978], [0.0, 1.15], [0.105, 0.978]]) * HEAD_R
    ax.add_patch(Polygon(nose, closed=False, fill=False, lw=lw, ec=color,
                         zorder=5, joinstyle="round"))
    for sgn in (-1, 1):
        ax.add_patch(Arc((sgn * HEAD_R * 0.99, 0.0), HEAD_R * 0.24,
                         HEAD_R * 0.52, angle=0,
                         theta1=-72 if sgn > 0 else 108,
                         theta2=72 if sgn > 0 else 252,
                         lw=lw, ec=color, zorder=4))


def draw_topomap(ax, values, names, vlim, cmap, highlight="Pz"):
    pts = np.array([MONTAGE[n] for n in names])
    rbf = RBFInterpolator(pts, values, kernel="thin_plate_spline", degree=1,
                          smoothing=1e-3)
    g = np.linspace(-HEAD_R, HEAD_R, 220)
    gx, gy = np.meshgrid(g, g)
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    z = rbf(flat).reshape(gx.shape)
    z[np.hypot(gx, gy) > HEAD_R * 0.998] = np.nan

    im = ax.imshow(z, extent=(-HEAD_R, HEAD_R, -HEAD_R, HEAD_R), origin="lower",
                   cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="bilinear",
                   zorder=1)
    levels = np.linspace(-vlim, vlim, 11)
    ax.contour(gx, gy, z, levels=levels, colors="#3a3a3a", linewidths=0.35,
               alpha=0.55, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=2.2, c="#1a1a1a", zorder=6, lw=0)
    hx, hy = MONTAGE[highlight]
    ax.scatter([hx], [hy], s=26, facecolors="none", edgecolors="#111111",
               lw=0.9, zorder=7)
    draw_head(ax)
    ax.set_xlim(-HEAD_R * 1.28, HEAD_R * 1.28)
    ax.set_ylim(-HEAD_R * 1.22, HEAD_R * 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    return im


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7)
    cmap = plt.get_cmap("RdBu_r")

    t = np.arange(-150, 801, 2.0)
    fig = plt.figure(figsize=(7.2, 4.85))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.72, 1.0],
                  width_ratios=[1, 1, 1, 1, 1, 0.10],
                  hspace=0.32, wspace=0.12,
                  left=0.085, right=0.955, top=0.925, bottom=0.045)

    # ---- (a) ERP waveforms -------------------------------------------------
    ax = fig.add_subplot(gs[0, :5])
    for name, comps in COMPONENTS.items():
        m, sem = erp_traces(t, comps, rng)
        c = COND_COLORS[name]
        ax.fill_between(t, m - sem, m + sem, color=c, alpha=0.20, lw=0, zorder=2)
        ax.plot(t, m, color=c, lw=1.5, label=name, zorder=3)

    ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax.axvline(0, color="#333333", lw=0.7, ls="--", dashes=(3, 2), zorder=1)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-7.5, 12.5)

    for lo, hi, stars in SIG_WINDOWS:
        ax.axvspan(lo, hi, color="#c9b458", alpha=0.15, lw=0, zorder=0)
        ax.plot([lo, hi], [-6.4, -6.4], lw=2.4, color="#8a7420",
                solid_capstyle="butt", zorder=4)
        ax.text((lo + hi) / 2, -6.15, stars, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#8a7420")
    ax.text(468, -6.4, "cluster-corrected p < .01", va="center", ha="left",
            fontsize=6.4, color="#8a7420")

    peaks = [("N1", 118, -4.6, (-22, -15)), ("P2", 196, 2.5, (-4, 19)),
             ("N2", 250, -1.8, (20, -19)), ("P300", 352, 10.0, (30, -2)),
             ("LPC", 548, 2.7, (16, 20))]
    for lab, x, y, off in peaks:
        ax.annotate(lab, xy=(x, y), xytext=off, textcoords="offset points",
                    fontsize=7.2, fontweight="bold", color="#2b2b2b",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.55, color="#9a9a9a",
                                    shrinkA=1, shrinkB=3))

    for _, lat, _ in TOPO_SNAPSHOTS:
        ax.plot([lat], [12.15], marker="v", ms=3.4, color="#444444",
                clip_on=False, zorder=5)

    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Amplitude at Pz (\u03bcV)")
    ax.set_xticks([-150, 0, 200, 400, 600, 800])
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=7.2, handlelength=1.4,
              labelspacing=0.3, borderaxespad=0.2,
              title="Oddball condition", title_fontsize=7.2)
    add_panel_label(ax, "(a)", x=-0.075, y=1.02, size=10)
    ax.text(0.995, 1.015, "n = 20 subjects, grand average \u00b1 SEM",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            color="#777777")

    # ---- (b) topographies --------------------------------------------------
    names = list(MONTAGE)
    all_vals = [scalp_values(g, names) for _, _, g in TOPO_SNAPSHOTS]
    vlim = float(np.ceil(max(np.abs(v).max() for v in all_vals)))

    topo_axes = []
    for i, ((lab, lat, _), vals) in enumerate(zip(TOPO_SNAPSHOTS, all_vals)):
        axt = fig.add_subplot(gs[1, i])
        im = draw_topomap(axt, vals, names, vlim, cmap)
        axt.set_title(f"{lab}\n{lat - 25}\u2013{lat + 25} ms", fontsize=7.4,
                      pad=2.5, linespacing=1.3)
        topo_axes.append(axt)

    add_panel_label(topo_axes[0], "(b)", x=-0.16, y=1.06, size=10)

    cax = fig.add_subplot(gs[1, 5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-vlim, -vlim / 2, 0, vlim / 2, vlim])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.2, color="#333333")
    cb.set_label("Amplitude (\u03bcV)", fontsize=7)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + pos.height * 0.16, pos.width * 0.85,
                      pos.height * 0.62])


    savefig(fig, "136_erp_topomap.png")


if __name__ == "__main__":
    main()
