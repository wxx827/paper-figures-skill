"""Galaxy rotation curve decomposed into baryonic and dark components.

The stellar and gaseous discs use the exact exponential-disc solution
v^2 = 4 pi G Sigma0 Rd y^2 [I0(y)K0(y) - I1(y)K1(y)] with y = R/2Rd, the bulge
is a Plummer sphere and the halo a pseudo-isothermal profile.  The measured
points follow the quadrature sum; the thin dotted curve is the Keplerian
fall-off the visible matter alone would produce, and the gap between the two
is the usual argument for a dark halo.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import i0, i1, k0, k1
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

RMAX = 22.0


def v_disc(r, v_scale, rd):
    y = np.clip(r, 1e-3, None) / (2 * rd)
    f = y ** 2 * (i0(y) * k0(y) - i1(y) * k1(y))
    return v_scale * np.sqrt(np.clip(f, 0, None))


def v_bulge(r, v0, a):
    r = np.clip(r, 1e-3, None)
    return v0 * r / (r ** 2 + a ** 2) ** 0.75


def v_halo(r, vinf, rc):
    r = np.clip(r, 1e-3, None)
    return vinf * np.sqrt(np.clip(1.0 - (rc / r) * np.arctan(r / rc), 0, None))


def components(r):
    d = v_disc(r, 402.0, 2.9)
    g = v_disc(r, 132.0, 7.4)
    b = v_bulge(r, 210.0, 0.75)
    h = v_halo(r, 205.0, 5.2)
    tot = np.sqrt(d ** 2 + g ** 2 + b ** 2 + h ** 2)
    bary = np.sqrt(d ** 2 + g ** 2 + b ** 2)
    return d, g, b, h, tot, bary


def spiral_galaxy(ax, rng, r_rings=(5, 10, 15, 20)):
    """Face-on cartoon: exponential disc, two log-spiral arms, gold bulge."""
    n = 1700
    rr = rng.exponential(6.6, n)
    rr = rr[rr < 21.0]
    th = rng.uniform(0, 2 * np.pi, rr.size)
    ax.scatter(rr * np.cos(th), rr * np.sin(th), s=1.1, c="#cbd8e3",
               linewidths=0, alpha=0.7, zorder=2)
    for off in (0.0, np.pi):
        t = np.linspace(0.2, 3.5 * np.pi, 1400)
        r = 1.4 * np.exp(0.235 * t)
        keep = r < 20.5
        t, r = t[keep], r[keep]
        m = rng.normal(0, 0.055, t.size) * np.maximum(r, 3.0)
        x = (r + m) * np.cos(t + off)
        y = (r + m) * np.sin(t + off)
        ax.scatter(x, y, s=3.0, c="#6f9dbe", linewidths=0, alpha=0.85,
                   zorder=3)
        ax.scatter(x[::11], y[::11], s=9.0, c="#eef4f8",
                   edgecolors="#5d8cb0", linewidths=0.3, alpha=0.95,
                   zorder=4)
    nb = 700
    rb = np.abs(rng.normal(0, 1.5, nb))
    tb = rng.uniform(0, 2 * np.pi, nb)
    ax.scatter(rb * np.cos(tb), rb * np.sin(tb), s=3.0, c="#e8b95f",
               linewidths=0, alpha=0.55, zorder=5)
    for rr_ in r_rings:
        ax.add_artist(plt.Circle((0, 0), rr_, fc="none", ec="#8b9099",
                                 lw=0.5, ls=(0, (3, 3)), alpha=0.8, zorder=6))
        if rr_ % 10 == 0:
            ax.text(rr_, -1.2, "%d" % rr_, fontsize=6.0, color="#5c636b",
                    ha="center", va="top", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))
    ax.set_xlim(-22.5, 22.5)
    ax.set_ylim(-22.5, 22.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8f9fb")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(101)

    r = np.linspace(0.05, RMAX, 900)
    d, g, b, h, tot, bary = components(r)

    r_obs = np.concatenate([np.arange(0.6, 4.0, 0.55),
                            np.arange(4.0, RMAX, 1.15)])
    _, _, _, _, t_obs, _ = components(r_obs)
    err = 5.5 + 7.0 * rng.random(r_obs.size)
    v_obs = t_obs + rng.normal(0, err * 0.75)

    # Keplerian expectation from the enclosed baryonic mass at 6 kpc
    r_k = np.linspace(6.0, RMAX, 300)
    v_kep = np.interp(6.0, r, bary) * np.sqrt(6.0 / r_k)

    cross = float(r[np.argmax(h > bary)])

    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.34],
                          width_ratios=[1.0, 0.42], hspace=0.20,
                          wspace=0.145, left=0.068, right=0.985, top=0.868,
                          bottom=0.078)
    ax = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0])
    sub = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.60], hspace=0.20)
    ax_g = fig.add_subplot(sub[0])
    ax_t = fig.add_subplot(sub[1])

    ax.axvspan(cross, RMAX, color=INDIGO, alpha=0.07, lw=0, zorder=0)
    ax.plot(r, d, color=GOLD, lw=1.5, ls=(0, (5, 2.4)), zorder=3)
    ax.plot(r, g, color=CYAN, lw=1.5, ls=(0, (1.5, 2.0)), zorder=3)
    ax.plot(r, b, color=RUST, lw=1.5, ls=(0, (4, 1.6, 1, 1.6)), zorder=3)
    ax.plot(r, h, color=INDIGO, lw=1.6, ls=(0, (7, 2.6)), zorder=3)
    ax.plot(r, tot, color=NAVY, lw=2.2, zorder=4)
    ax.plot(r_k, v_kep, color=GREY, lw=1.2, ls=(0, (1, 2.2)), zorder=3)
    ax.errorbar(r_obs, v_obs, yerr=err, ls="none", marker="o", ms=4.2,
                mfc="white", mec=NAVY, mew=1.0, ecolor=NAVY, elinewidth=0.9,
                capsize=2.0, zorder=6)

    ax.set_xlim(0, RMAX)
    ax.set_ylim(0, 268)
    ax.set_xticklabels([])
    ax.set_ylabel("circular velocity  $v_c$ (km s$^{-1}$)", fontsize=8.6,
                  labelpad=3)
    ax.tick_params(labelsize=7.2, length=2.8)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)
    ax.set_title("Observed rotation curve against the mass model",
                 fontsize=9.0, fontweight="bold", pad=4)

    ax.axvline(cross, color=INDIGO, lw=0.8, ls=(0, (3, 2.4)), zorder=2)
    ax.text(RMAX - 0.35, 261, "dark-halo dominated", fontsize=7.4,
            color=INDIGO, ha="right", va="top", fontweight="bold")
    ax.annotate("Keplerian fall-off expected\nfrom the visible mass alone",
                xy=(18.4, float(np.interp(18.4, r_k, v_kep))),
                xytext=(13.6, 26), fontsize=7.0, color="#6f767e",
                ha="center", va="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#6f767e",
                                shrinkB=4, connectionstyle="arc3,rad=-0.25"))
    ax.annotate("flat outer curve", xy=(20.6, float(np.interp(20.6, r, tot))),
                xytext=(18.6, 186), fontsize=7.2, color=NAVY, ha="center",
                va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=NAVY,
                                shrinkB=4, connectionstyle="arc3,rad=-0.25"))

    # ---- subordinate column: cartoon and derived quantities ----------------
    spiral_galaxy(ax_g, rng)
    ax_g.set_title("face-on sketch  |  rings every 5 kpc", fontsize=7.2,
                   style="italic", color="#555555", pad=2.5)

    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    ax_t.set_facecolor("#f6f7f9")
    for s in ax_t.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_t.set_title("derived quantities", fontsize=7.2, style="italic",
                   color="#555555", pad=2.5)
    rows = [("$v_{\\rm flat}$", "%.0f km s$^{-1}$" % tot[-1]),
            ("$R_{\\rm d}$ (stars)", "2.9 kpc"),
            ("$R_{\\rm d}$ (H I)", "7.4 kpc"),
            ("halo $r_c$", "5.2 kpc"),
            ("$R$ (halo = baryons)", "%.1f kpc" % cross),
            ("$M_{\\rm bar}(<R_{\\rm max})$", "$6.1\\times10^{10}\\,M_\\odot$"),
            ("$M_{\\rm dyn}(<R_{\\rm max})$", "$2.5\\times10^{11}\\,M_\\odot$"),
            ("$M_{\\rm dyn}/M_{\\rm bar}$", "4.1")]
    y = 0.925
    for k, (nm, val) in enumerate(rows):
        ax_t.text(0.06, y, nm, fontsize=6.9, ha="left", va="center",
                  color="#33475f")
        ax_t.text(0.94, y, val, fontsize=6.9, ha="right", va="center",
                  color="#1a1a1a")
        if k < len(rows) - 1:
            ax_t.plot([0.05, 0.95], [y - 0.058] * 2, color="#e0e3e7", lw=0.5)
        y -= 0.116

    # ---- demoted strip: share of v^2 carried by each mass component --------
    ax_f.set_facecolor("#f4f5f7")
    fb = bary ** 2 / tot ** 2
    ax_f.fill_between(r, 0, fb, color=GOLD, alpha=0.45, lw=0)
    ax_f.fill_between(r, fb, 1.0, color=INDIGO, alpha=0.35, lw=0)
    ax_f.plot(r, fb, color=GOLD, lw=1.2)
    ax_f.axvline(cross, color=INDIGO, lw=0.8, ls=(0, (3, 2.4)))
    ax_f.axhline(0.5, color=GREY, lw=0.6, ls=(0, (2, 2)))
    ax_f.text(1.0, 0.30, "baryons", fontsize=7.0, color="#8a6b1f",
              fontweight="bold", va="center")
    ax_f.text(RMAX - 0.8, 0.62, "dark halo", fontsize=7.0, color=INDIGO,
              fontweight="bold", ha="right", va="center")
    ax_f.set_xlim(0, RMAX)
    ax_f.set_ylim(0, 1)
    ax_f.set_yticks([0, 0.5, 1.0])
    ax_f.set_xlabel("galactocentric radius  $R$ (kpc)", fontsize=8.6,
                    labelpad=2)
    ax_f.set_ylabel("share of $v_c^2$", fontsize=7.4, labelpad=2)
    ax_f.tick_params(labelsize=6.8, length=2.2)
    ax_f.grid(True, axis="x", lw=0.4, color="#e3e5e8")
    ax_f.set_axisbelow(True)
    for s in ax_f.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_f.set_title("Supporting statistic  |  fraction of $v_c^2$ supplied by "
                   "baryons and by the halo", fontsize=7.2, style="italic",
                   color="#555555", pad=2.5)

    for ax_, lab in [(ax, "(a)"), (ax_f, "(b)"), (ax_g, "(c)"),
                     (ax_t, "(d)")]:
        p = ax_.get_position()
        add_panel_label(ax_, lab, x=-0.030 / p.width,
                        y=1.0 + 0.026 / p.height, size=10)

    handles = [
        Line2D([], [], color=NAVY, lw=2.2, label="total model"),
        Line2D([], [], marker="o", ls="none", ms=4.6, mfc="white", mec=NAVY,
               mew=1.0, label="H I observations"),
        Line2D([], [], color=GOLD, lw=1.5, ls=(0, (5, 2.4)),
               label="stellar disc"),
        Line2D([], [], color=CYAN, lw=1.5, ls=(0, (1.5, 2.0)),
               label="H I gas disc"),
        Line2D([], [], color=RUST, lw=1.5, ls=(0, (4, 1.6, 1, 1.6)),
               label="bulge"),
        Line2D([], [], color=INDIGO, lw=1.6, ls=(0, (7, 2.6)),
               label="dark halo"),
        Line2D([], [], color=GREY, lw=1.1, ls=(0, (1, 2.2)),
               label="Keplerian fall-off")]

    savefig(fig, "239_rotation_curve.png")


if __name__ == "__main__":
    main()
