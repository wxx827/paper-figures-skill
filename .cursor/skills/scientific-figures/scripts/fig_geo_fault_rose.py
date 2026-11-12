"""Three complementary views of one structural fabric.

The same 196 measured joint planes drive all three panels: a bidirectional
strike rose, an equal-area lower-hemisphere stereonet carrying the poles, their
density contours and the mean great circle of each set, and a map-view
palaeostress sketch.  Every angle quoted on the figure - set means, the acute
conjugate angle, the inferred maximum horizontal stress azimuth - is a circular
statistic computed from the measurement arrays, not a caption.

Canvas: all three main panels are circular, so the aspect ratio sits below the
0.55-0.80 band under exception E3 of DESIGN_SYSTEM.md section 1.2.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow
from pubstyle import set_pub_style, savefig, add_panel_label, get_cmap

SETS = [
    dict(name="Set J1", n=88, strike=42.0, s_sd=9.0, dip=78.0, d_sd=6.0,
         color="#b23a48"),
    dict(name="Set J2", n=72, strike=118.0, s_sd=11.0, dip=71.0, d_sd=7.0,
         color="#3b5b92"),
    dict(name="Set J3 (bedding)", n=36, strike=6.0, s_sd=14.0, dip=32.0,
         d_sd=8.0, color="#2e7d5b"),
]
SIG1, SIG3 = 42.0, 17.0          # MPa, the modelled principal magnitudes
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"


def circ_mean_axial(deg):
    """Mean direction of an axial (180-periodic) sample, in degrees."""
    a = np.deg2rad(np.asarray(deg) * 2.0)
    return (np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
            / 2.0) % 180.0


def schmidt(trend, plunge):
    """Equal-area lower-hemisphere projection of a line."""
    r = np.sqrt(2.0) * np.sin(np.deg2rad((90.0 - plunge) / 2.0))
    t = np.deg2rad(trend)
    return r * np.sin(t), r * np.cos(t)


def great_circle(strike, dip, n=400):
    """Lower-hemisphere trace of a plane, split where it leaves the disc."""
    t = np.deg2rad(strike - 90.0)
    p = np.deg2rad(90.0 - dip)
    nvec = np.array([np.sin(t) * np.cos(p), np.cos(t) * np.cos(p), -np.sin(p)])
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(nvec[2]) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    u1 = np.cross(nvec, tmp)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(nvec, u1)
    th = np.linspace(0.0, 2 * np.pi, n)
    v = (np.cos(th)[:, None] * u1 + np.sin(th)[:, None] * u2)
    low = v[:, 2] <= 0.0
    pl = np.rad2deg(np.arcsin(np.clip(-v[:, 2], -1.0, 1.0)))
    tr = np.rad2deg(np.arctan2(v[:, 0], v[:, 1])) % 360.0
    x, y = schmidt(tr, pl)
    return np.where(low, x, np.nan), np.where(low, y, np.nan)


def rasterize(artist):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def main():
    set_pub_style()
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(4080813)

    strikes, dips, tags = [], [], []
    for i, s in enumerate(SETS):
        strikes.append(rng.normal(s["strike"], s["s_sd"], s["n"]) % 180.0)
        dips.append(np.clip(rng.normal(s["dip"], s["d_sd"], s["n"]), 5, 89))
        tags.append(np.full(s["n"], i))
    strike = np.concatenate(strikes)
    dip = np.concatenate(dips)
    tag = np.concatenate(tags)
    n_tot = strike.size

    means = [circ_mean_axial(strikes[i]) for i in range(len(SETS))]
    acute = abs(means[1] - means[0])
    acute = min(acute, 180.0 - acute)
    shmax = (means[0] + (means[1] - means[0]) / 2.0) % 180.0

    fig = plt.figure(figsize=(10.4, 5.0))
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.975, top=0.845,
                          bottom=0.150, wspace=0.22)
    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    ax_s = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])

    # ---- (a) bidirectional strike rose ------------------------------------
    step = 10.0
    edges = np.arange(0.0, 360.0 + step, step)
    doubled = np.concatenate([strike, (strike + 180.0) % 360.0])
    counts, _ = np.histogram(doubled, bins=edges)
    cmap = get_cmap("cb_sequential_blue")
    cmax = counts.max()
    ax_r.bar(np.deg2rad(edges[:-1] + step / 2), counts,
             width=np.deg2rad(step) * 0.94, bottom=0.0,
             color=[cmap(0.20 + 0.78 * c / cmax) for c in counts],
             edgecolor="white", linewidth=0.5, zorder=3)
    ax_r.set_theta_zero_location("N")
    ax_r.set_theta_direction(-1)
    ax_r.set_thetagrids(np.arange(0, 360, 45),
                        labels=["N", "045", "E", "135", "S", "225", "W",
                                "315"], fontsize=6.8)
    ax_r.set_rlabel_position(112.5)
    ax_r.tick_params(axis="y", labelsize=6.2, colors=MUTED)
    ax_r.set_ylim(0, cmax * 1.08)
    ax_r.grid(color="#e0e0e0", lw=0.5)
    ax_r.set_axisbelow(True)
    for m, s in zip(means, SETS):
        for a in (m, m + 180.0):
            ax_r.plot([np.deg2rad(a)] * 2, [0, cmax * 1.05], color=s["color"],
                      lw=1.3, ls=(0, (4, 2)), zorder=5)
    ax_r.set_title(f"Strike rose, {step:.0f}$\\degree$ bins "
                   f"(n = {n_tot})", fontsize=8.5, fontweight="bold", pad=9)

    # ---- (b) equal-area stereonet -----------------------------------------
    px, py = schmidt((strike - 90.0) % 360.0, 90.0 - dip)
    gx = np.linspace(-1.0, 1.0, 240)
    GX, GY = np.meshgrid(gx, gx)
    dens = gaussian_kde(np.vstack([px, py]), bw_method=0.22)(
        np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    dens = np.where(GX ** 2 + GY ** 2 <= 1.0, dens, np.nan)
    cs = ax_s.contourf(GX, GY, dens, levels=8, cmap=cmap, alpha=0.85,
                       zorder=1)
    rasterize(cs)
    cs.set_edgecolor("face")
    ax_s.contour(GX, GY, dens, levels=8, colors="#5f9dc6", linewidths=0.4,
                 zorder=2)

    for i, s in enumerate(SETS):
        m = tag == i
        ax_s.plot(px[m], py[m], ls="none", marker="o", ms=2.4,
                  mfc=s["color"], mec="white", mew=0.35, zorder=4)
        mdip = float(np.mean(dips[i]))
        cx, cy = great_circle(means[i], mdip)
        ax_s.plot(cx, cy, color=s["color"], lw=1.3, zorder=5)

    ax_s.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=RULE,
                          lw=1.0, zorder=6))
    for a, lab in zip((0, 90, 180, 270), ("N", "E", "S", "W")):
        xt, yt = np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))
        ax_s.plot([xt, xt * 1.045], [yt, yt * 1.045], color=RULE, lw=0.9,
                  clip_on=False, zorder=6)
        ax_s.text(xt * 1.12, yt * 1.12, lab, ha="center", va="center",
                  fontsize=7.0, color=RULE, fontweight="bold")
    ax_s.plot([0], [0], marker="+", ms=5, color=RULE, mew=0.9, zorder=6)
    ax_s.set_aspect("equal")
    ax_s.set_xlim(-1.22, 1.22)
    ax_s.set_ylim(-1.22, 1.22)
    ax_s.axis("off")
    ax_s.set_title("Poles, equal-area lower hemisphere",
                   fontsize=8.5, fontweight="bold")
    ax_s.text(0.0, -1.24, f"contours: kernel density of {n_tot} poles",
              ha="center", va="top", fontsize=6.4, color=MUTED)

    # ---- (c) inferred palaeostress ----------------------------------------
    ax_p.add_patch(Circle((0, 0), 1.0, facecolor="#f5f5f5", edgecolor=RULE,
                          lw=1.0, zorder=1))
    for i, s in enumerate(SETS[:2]):
        a = np.deg2rad(means[i])
        dvec = np.array([np.sin(a), np.cos(a)])
        nvec = np.array([dvec[1], -dvec[0]])
        for off in np.linspace(-0.78, 0.78, 5):
            h = np.sqrt(max(1.0 - off ** 2, 0.0))
            p0 = nvec * off - dvec * h
            p1 = nvec * off + dvec * h
            ax_p.plot([p0[0], p1[0]], [p0[1], p1[1]], color=s["color"],
                      lw=0.9, alpha=0.60, zorder=2)

    a1 = np.deg2rad(shmax)
    d1 = np.array([np.sin(a1), np.cos(a1)])
    d3 = np.array([d1[1], -d1[0]])
    for sgn in (1, -1):
        p = d1 * sgn * 1.44
        ax_p.add_patch(FancyArrow(p[0], p[1], -d1[0] * sgn * 0.34,
                                  -d1[1] * sgn * 0.34, width=0.045,
                                  head_width=0.15, head_length=0.15,
                                  length_includes_head=True,
                                  facecolor="#b23a48", edgecolor="none",
                                  zorder=5))
        q = d3 * sgn * 1.38
        ax_p.add_patch(FancyArrow(q[0], q[1], -d3[0] * sgn * 0.28,
                                  -d3[1] * sgn * 0.28, width=0.030,
                                  head_width=0.11, head_length=0.12,
                                  length_includes_head=True,
                                  facecolor="#3d6b8e", edgecolor="none",
                                  zorder=5))
    for sgn in (1, -1):
        ax_p.text(d1[0] * sgn * 1.62, d1[1] * sgn * 1.62, "$\\sigma_1$",
                  ha="center", va="center", fontsize=8.0, color="#b23a48",
                  fontweight="bold")
        ax_p.text(d3[0] * sgn * 1.56, d3[1] * sgn * 1.56, "$\\sigma_3$",
                  ha="center", va="center", fontsize=8.0, color="#3d6b8e",
                  fontweight="bold")
    ax_p.annotate("", xy=(-1.42, 1.62), xytext=(-1.42, 1.24),
                  arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0))
    ax_p.text(-1.34, 1.44, "N", ha="left", va="center", fontsize=7.0,
              color=RULE, fontweight="bold")
    ax_p.text(0.0, -1.82, f"$\\sigma_1$ = {SIG1:.0f} MPa at "
              f"{shmax:.0f}$\\degree$,  $\\sigma_3$ = {SIG3:.0f} MPa\n"
              f"conjugate angle 2$\\theta$ = {acute:.0f}$\\degree$",
              ha="center", va="top", fontsize=7.0, color=INK,
              linespacing=1.35)
    ax_p.set_aspect("equal")
    ax_p.set_xlim(-1.88, 1.88)
    ax_p.set_ylim(-1.88, 1.88)
    ax_p.axis("off")
    ax_p.set_title("Map-view palaeostress", fontsize=8.5, fontweight="bold")

    handles = [Line2D([0], [0], color=s["color"], lw=1.4, marker="o", ms=3.6,
                      mfc=s["color"], mec="white",
                      label=f"{s['name']}  {means[i]:.0f}/"
                            f"{np.mean(dips[i]):.0f} (n={s['n']})")
               for i, s in enumerate(SETS)]
    handles += [
        Line2D([0], [0], color="#b23a48", lw=4, label="$\\sigma_1$ direction"),
        Line2D([0], [0], color="#3d6b8e", lw=4, label="$\\sigma_3$ direction"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), ncol=5, frameon=False,
               fontsize=7.0, handlelength=1.8, columnspacing=1.8)

    for ax, lab in zip((ax_r, ax_s, ax_p), ("(a)", "(b)", "(c)")):
        add_panel_label(ax, lab, x=-0.10, y=1.06)
    savefig(fig, "408_geo_fault_rose.png")


if __name__ == "__main__":
    main()
