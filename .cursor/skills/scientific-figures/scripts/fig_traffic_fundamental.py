"""Traffic-flow fundamental diagram: the three canonical projections.

A synthetic freeway loop-detector day (5-min aggregates) is plotted as
flow-density, speed-density and flow-speed clouds.  Observations are
coloured by hour of day through a cyclic diurnal ramp; the calibrated
theoretical curve is drawn in two hues so the uncongested and congested
branches read apart at a glance, with the critical density and the
capacity plateau called out.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig

# ---- shared transport palette (kept identical across the 203-212 series)
FREE = "#2f7d78"      # uncongested branch
CONG = "#b5563c"      # congested branch
INK = "#2b2b2b"
GREY = "#8c8c8c"
DIURNAL = ["#46566b", "#3f8f88", "#d9b96a", "#c0603f", "#46566b"]

VF, KC, KJ = 98.0, 24.0, 145.0          # km/h, veh/km, veh/km
VC = VF * 0.88                           # speed at capacity
QMAX = KC * VC                           # veh/h
W = QMAX / (KJ - KC)                     # backward wave speed, km/h


def fd_speed(k):
    k = np.asarray(k, dtype=float)
    v_free = VF * (1.0 - 0.12 * (k / KC) ** 2)
    v_cong = W * (KJ - k) / np.maximum(k, 1e-9)
    return np.where(k <= KC, v_free, v_cong)


def panel_tag(fig, ax, text, dx=-0.40, dy=0.13, size=11):
    """(a)/(b)/... at a constant *inch* offset from the axes' top-left."""
    pos = ax.get_position()
    fig.text(pos.x0 + dx / fig.get_figwidth(),
             pos.y1 + dy / fig.get_figheight(),
             text, fontsize=size, fontweight="bold", ha="left", va="baseline",
             color=INK)


def panel_title(fig, ax, text, dy=0.13, size=8.8):
    pos = ax.get_position()
    fig.text(pos.x0, pos.y1 + dy / fig.get_figheight(), text, fontsize=size,
             fontweight="bold", ha="left", va="baseline", color="#33474f")


def make_day(rng, n=300):
    t = np.linspace(0.0, 24.0, n, endpoint=False)
    base = (4.5
            + 30.0 * np.exp(-0.5 * ((t - 8.0) / 1.05) ** 2)
            + 38.0 * np.exp(-0.5 * ((t - 17.7) / 1.35) ** 2)
            + 11.0 * np.exp(-0.5 * ((t - 13.0) / 2.60) ** 2)
            + 3.0 * np.exp(-0.5 * ((t - 21.5) / 1.80) ** 2))
    k = base * rng.lognormal(0.0, 0.17, n) + rng.normal(0, 0.6, n)
    k = np.clip(k, 1.2, KJ - 6.0)
    v = fd_speed(k)
    drop = np.where(k > KC, 0.97, 1.0)            # capacity-drop hysteresis
    spread = np.where(k > KC, 0.075, 0.042)
    v = v * drop * np.exp(rng.normal(0.0, spread, n))
    v = np.clip(v, 6.0, VF * 1.04)
    return t, k, v


def main():
    set_pub_style()
    rng = np.random.default_rng(7)
    t, k, v = make_day(rng)
    q = k * v

    cmap = mpl.colors.LinearSegmentedColormap.from_list("diurnal", DIURNAL)
    norm = mpl.colors.Normalize(0, 24)
    cols = cmap(norm(t))

    kf = np.linspace(0.0, KC, 160)
    kc_ = np.linspace(KC, KJ, 200)
    qf, qc = kf * fd_speed(kf), kc_ * fd_speed(kc_)
    vf_, vc_ = fd_speed(kf), fd_speed(kc_)

    fig = plt.figure(figsize=(10.2, 3.95))
    gs = GridSpec(1, 3, left=0.055, right=0.988, top=0.855, bottom=0.255,
                  wspace=0.255)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    sc_kw = dict(s=11, linewidths=0.0, alpha=0.80, zorder=3)
    ln_kw = dict(lw=2.0, zorder=4, solid_capstyle="round")

    # ---------------------------------------------------------------- (a)
    ax = axes[0]
    ax.scatter(k, q, c=cols, **sc_kw)
    ax.plot(kf, qf, color=FREE, **ln_kw)
    ax.plot(kc_, qc, color=CONG, **ln_kw)
    ax.axvline(KC, color=GREY, lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    ax.axhline(QMAX, color=GREY, lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    ax.plot([KC], [QMAX], marker="o", ms=5.2, mfc="white", mec=INK, mew=1.1,
            zorder=6)
    ax.set_xlim(0, KJ)
    ax.set_ylim(0, QMAX * 1.30)
    ax.set_xlabel("density $k$  (veh/km)", labelpad=2)
    ax.set_ylabel("flow $q$  (veh/h)", labelpad=2)
    ax.annotate(f"capacity\n{QMAX:.0f} veh/h", xy=(KC, QMAX),
                xytext=(KC + 30, QMAX * 1.16), fontsize=7.2, color=INK,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY,
                                shrinkA=1, shrinkB=4))
    ax.annotate(f"$k_c$ = {KC:.0f} veh/km", xy=(KC, QMAX * 0.16),
                xytext=(KC + 8, QMAX * 0.10), fontsize=7.2, color="#55606b",
                ha="left", va="center")
    ax.text(KC * 0.32, QMAX * 0.84, "free\nflow", color=FREE, fontsize=7.4,
            fontweight="bold", ha="center", va="center", linespacing=1.15)
    ax.text(KJ * 0.74, QMAX * 0.60, "congested", color=CONG, fontsize=7.4,
            fontweight="bold", ha="center", va="center")

    # ---------------------------------------------------------------- (b)
    ax = axes[1]
    ax.scatter(k, v, c=cols, **sc_kw)
    ax.plot(kf, vf_, color=FREE, **ln_kw)
    ax.plot(kc_, vc_, color=CONG, **ln_kw)
    ax.axvline(KC, color=GREY, lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    ax.plot([KC], [VC], marker="o", ms=5.2, mfc="white", mec=INK, mew=1.1,
            zorder=6)
    ax.set_xlim(0, KJ)
    ax.set_ylim(0, VF * 1.22)
    ax.set_xlabel("density $k$  (veh/km)", labelpad=2)
    ax.set_ylabel("space-mean speed $v$  (km/h)", labelpad=2)
    ax.annotate(f"$v_f$ = {VF:.0f} km/h", xy=(2.0, VF), fontsize=7.2,
                xytext=(3.0, VF * 1.11), color=FREE, fontweight="bold",
                ha="left", va="center")
    ax.annotate(f"$v_c$ = {VC:.0f} km/h", xy=(KC, VC),
                xytext=(KC + 12, VC * 1.14), fontsize=7.2, color=INK,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY,
                                shrinkA=1, shrinkB=4))
    ax.annotate("jam density\n$k_j$ = %.0f veh/km" % KJ, xy=(KJ, 0.0),
                xytext=(KJ - 4, VF * 0.30), fontsize=7.2, color=CONG,
                fontweight="bold", ha="right", va="bottom", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY,
                                shrinkA=3, shrinkB=1))

    # ---------------------------------------------------------------- (c)
    ax = axes[2]
    ax.scatter(v, q, c=cols, **sc_kw)
    ax.plot(vf_, qf, color=FREE, **ln_kw)
    ax.plot(vc_, qc, color=CONG, **ln_kw)
    ax.axhline(QMAX, color=GREY, lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    ax.plot([VC], [QMAX], marker="o", ms=5.2, mfc="white", mec=INK, mew=1.1,
            zorder=6)
    ax.set_xlim(0, VF * 1.14)
    ax.set_ylim(0, QMAX * 1.30)
    ax.set_xlabel("space-mean speed $v$  (km/h)", labelpad=2)
    ax.set_ylabel("flow $q$  (veh/h)", labelpad=2)
    # one flow rate is served by two distinct speeds -> tie-line between the
    # congested and the uncongested branch at a common q
    q_ref = QMAX * 0.72
    v_lo = float(np.interp(q_ref, qc[::-1], vc_[::-1]))
    v_hi = float(np.interp(q_ref, qf, vf_))
    ax.plot([v_lo, v_hi], [q_ref, q_ref], color=INK, lw=0.75,
            ls=(0, (2.2, 1.8)), zorder=5)
    ax.plot([v_lo, v_hi], [q_ref, q_ref], ls="none", marker="o", ms=3.6,
            mfc="white", mec=INK, mew=0.9, zorder=6)
    ax.text((v_lo + v_hi) / 2.0, q_ref - QMAX * 0.075,
            "one flow rate,\ntwo speed regimes", fontsize=7.0,
            color="#55606b", ha="center", va="top", linespacing=1.2)
    ax.text(v_lo - 2.5, q_ref, f"{v_lo:.0f}", fontsize=6.8,
            color=CONG, fontweight="bold", ha="right", va="center")
    ax.text(v_hi + 2.5, q_ref, f"{v_hi:.0f} km/h", fontsize=6.8,
            color=FREE, fontweight="bold", ha="left", va="center")

    for ax in axes:
        ax.grid(axis="both", color="#e8e8e8", lw=0.55, zorder=0)
        ax.tick_params(length=2.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax, tag, ttl in zip(axes, "(a) (b) (c)".split(),
                            ["Flow $-$ density", "Speed $-$ density",
                             "Flow $-$ speed"]):
        panel_tag(fig, ax, tag)
        panel_title(fig, ax, ttl, dy=0.135)

    # ---- shared diurnal colorbar + branch legend ------------------------
    cax = fig.add_axes([0.415, 0.105, 0.265, 0.030])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[0, 6, 12, 18, 24])
    cb.set_label("hour of day  (h)", fontsize=7.2, labelpad=2.5)
    cb.ax.set_xticklabels(["00", "06", "12", "18", "24"])
    cb.ax.tick_params(labelsize=6.8, length=2.2, width=0.7)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#555555")

    handles = [Line2D([], [], color=FREE, lw=2.2, label="uncongested branch"),
               Line2D([], [], color=CONG, lw=2.2, label="congested branch"),
               Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white",
                      mec=INK, mew=1.0, label="capacity point")]
    leg = fig.legend(handles=handles, loc="lower left",
                     bbox_to_anchor=(0.055, 0.045), frameon=False,
                     fontsize=7.2, handlelength=1.9, labelspacing=0.42,
                     borderpad=0.2)
    leg.set_clip_on(False)

    savefig(fig, "203_traffic_fundamental.png")


if __name__ == "__main__":
    main()
