"""Interpreted 2D seismic section: three horizons cut by one normal fault.

The section is built forward.  A reflectivity series is laid out in a flat
reference frame, then a structural warp - regional dip, an anticline that
relaxes with depth, and a listric normal fault - maps every reflector onto its
own two-way time at each trace.  Convolution with a Ricker wavelet turns that
into the variable-density display, so the picked horizons are guaranteed to
follow real events rather than being drawn on top of them.

Two-way time increases downward on every panel.  The quoted fault throw is
measured from the picked horizon arrays, and the well-tie correlation is
computed between the synthetic and the extracted trace that are drawn.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, shared_norm, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"
FAULT = "#1a1a1a"

HZN = [("Top Chalk", 1010.0, 0.16, "#b23a48"),
       ("Base Shale", 1420.0, -0.14, "#c9a227"),
       ("Top Reservoir", 1760.0, 0.19, "#2e7d5b")]

NTR, DT = 240, 2.0
T0, T1 = 800.0, 2200.0
FREQ = 27.0
THROW = 60.0
WELL_CDP = 96


def structure(x, t):
    """Regional dip plus a depth-relaxing anticline, in milliseconds."""
    relax = 1.0 - 0.32 * (t - T0) / (T1 - T0)
    anticline = -78.0 * np.exp(-((x - 95.0) / 46.0) ** 2) * relax
    syncline = 26.0 * np.exp(-((x - 200.0) / 40.0) ** 2) * relax
    return 0.052 * x + anticline + syncline


def fault_x(t):
    """Trace index of the fault plane at time ``t`` (listric, dipping right)."""
    return 150.0 + 0.030 * (t - T0) + 4.0e-6 * (t - T0) ** 2


def warp(x, tref):
    t = tref + structure(x, tref)
    return np.where(x > fault_x(t), t + THROW, t)


def ricker(freq, dt, n=121):
    tt = (np.arange(n) - n // 2) * dt * 1e-3
    a = (np.pi * freq * tt) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def main():
    set_pub_style()
    rng = np.random.default_rng(2711)

    t_ax = np.arange(T0, T1 + DT, DT)
    nt = t_ax.size
    x_ax = np.arange(NTR, dtype=float)

    # reference reflectivity series: background layering plus the three markers
    t_bg = np.sort(rng.uniform(T0 + 30.0, T1 - 30.0, 58))
    r_bg = rng.normal(0.0, 0.055, t_bg.size)
    t_ref = np.concatenate([t_bg, [h[1] for h in HZN]])
    r_ref = np.concatenate([r_bg, [h[2] for h in HZN]])
    order = np.argsort(t_ref)
    t_ref, r_ref = t_ref[order], r_ref[order]

    refl = np.zeros((nt, NTR))
    for tr, rc in zip(t_ref, r_ref):
        tt = warp(x_ax, tr)
        idx = np.round((tt - T0) / DT).astype(int)
        ok = (idx >= 0) & (idx < nt)
        refl[idx[ok], x_ax[ok].astype(int)] += rc

    w = ricker(FREQ, DT)
    sec = np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0,
                              refl)
    sec = sec + gaussian_filter(rng.normal(0.0, 1.0, sec.shape), 1.1) * 0.012

    hz_t = np.array([warp(x_ax, h[1]) for h in HZN])
    d = np.abs(np.diff(hz_t[2]))
    throw_meas = float(d.max())

    # ---- well tie ----------------------------------------------------------
    xw = np.full(t_ref.size, float(WELL_CDP))
    t_w = np.array([float(warp(np.array([xw[i]]), t_ref[i])[0])
                    for i in range(t_ref.size)])
    r_w = np.zeros(nt)
    for tt, rc in zip(t_w, r_ref):
        i = int(round((tt - T0) / DT))
        if 0 <= i < nt:
            r_w[i] += rc
    synth = np.convolve(r_w, w, mode="same")
    trace = sec[:, WELL_CDP]
    cc = float(np.corrcoef(synth, trace)[0, 1])
    ai = 7.2e3 * np.cumprod((1.0 + np.clip(r_w, -0.4, 0.4)) /
                            (1.0 - np.clip(r_w, -0.4, 0.4)))

    # ---- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.80, 1.06, 4.80, 4.22))
    ax_tie = axes_inches(fig, (7.05, 3.58, 2.85, 1.70))
    ax_str = axes_inches(fig, (7.05, 1.06, 2.85, 1.70))

    norm = shared_norm([sec], robust=(1.0, 99.0), symmetric=True)
    im = ax.imshow(sec, cmap=get_cmap("cb_diverging"), norm=norm,
                   extent=[0.0, float(NTR), T1, T0], aspect="auto",
                   interpolation="bilinear")
    ax.set_xlabel("CDP number (-)", fontsize=8.0)
    ax.set_ylabel("Two-way time (ms)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    safe_colorbar(fig, im, ax, label="amplitude (-)", width=0.11, pad=0.07)

    for (name, tr, _, col), z in zip(HZN, hz_t):
        cut = np.r_[np.abs(np.diff(z)) > 0.6 * THROW, False]
        zz = np.where(cut, np.nan, z)
        ax.plot(x_ax + 0.5, zz, color=col, lw=1.6, zorder=5)
        ax.text(3.0, z[3] - 14.0, name, color=col, fontsize=7.0,
                fontweight="bold", va="bottom", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.75))

    tf = np.linspace(T0 + 20.0, T1 - 20.0, 60)
    ax.plot(fault_x(tf), tf, color=FAULT, lw=1.5, ls=(0, (5, 2.2)), zorder=6)
    ax.annotate("normal fault\nthrow %.0f ms" % throw_meas,
                xy=(fault_x(1760.0), 1760.0), xytext=(fault_x(1760.0) + 26.0,
                                                      1980.0),
                fontsize=6.8, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=FAULT, lw=0.9,
                                shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
                zorder=7)
    ax.axvline(WELL_CDP, color=INK, lw=1.0)
    ax.text(WELL_CDP + 2.0, T0 + 18.0, "well A", fontsize=6.8, color=INK,
            ha="left", va="top", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.8))
    ax.set_title("Anticlinal crest at CDP 95, downthrown block to the right",
                 fontsize=8.8, fontweight="bold")

    # ---- well tie: wiggle with variable area --------------------------------
    ax_tie.set_ylim(T1, T0)
    ax_tie.set_xlim(-0.9, 3.15)
    ax_tie.set_xticks([0.0, 1.6, 2.6])
    ax_tie.set_xticklabels(["AI", "Synthetic", "Seismic"], fontsize=7.0)
    ax_tie.set_ylabel("Two-way time (ms)", fontsize=7.4)
    ax_tie.set_xlabel("impedance and amplitude, scaled (-)", fontsize=7.0)
    ax_tie.tick_params(labelsize=6.6)
    ax_tie.grid(axis="y", color=GRID, lw=0.4)
    ax_tie.set_axisbelow(True)

    ai_x = -0.85 + 1.15 * (ai - ai.min()) / (ai.max() - ai.min())
    ax_tie.plot(ai_x, t_ax, color="#3b5b92", lw=0.7)
    sc = 0.9 / max(np.abs(trace).max(), np.abs(synth).max())
    for x_c, sig, col in ((1.6, synth, "#2e7d5b"), (2.6, trace, "#333333")):
        xx = x_c + sig * sc
        ax_tie.plot(xx, t_ax, color=col, lw=0.6)
        ax_tie.fill_betweenx(t_ax, x_c, xx, where=xx > x_c, facecolor=col,
                             alpha=0.75, lw=0)
    for (name, tr, _, col), z in zip(HZN, hz_t):
        zt = z[WELL_CDP]
        if T0 < zt < T1:
            ax_tie.axhline(zt, color=col, lw=1.0, ls=(0, (4, 2)))
    ax_tie.text(0.03, 0.035, "cross-correlation %.3f" % cc,
                transform=ax_tie.transAxes, ha="left", va="bottom",
                fontsize=6.6, color=INK,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.92))
    ax_tie.set_title("Synthetic ties the section at well A", fontsize=8.5,
                     fontweight="bold")

    # ---- horizon time-structure ----------------------------------------------
    for (name, tr, _, col), z in zip(HZN, hz_t):
        cut = np.r_[np.abs(np.diff(z)) > 0.6 * THROW, False]
        ax_str.plot(x_ax, np.where(cut, np.nan, z), color=col, lw=1.2,
                    label=name)
    ax_str.set_ylim(T1, T0)
    ax_str.set_xlim(0.0, float(NTR))
    ax_str.axvspan(fault_x(T0), fault_x(T1), facecolor=MUTED, alpha=0.16,
                   lw=0)
    ax_str.text(0.5 * (fault_x(T0) + fault_x(T1)), T0 + 26.0, "fault zone",
                ha="center", va="top", fontsize=6.2, color=SUB)
    ax_str.set_xlabel("CDP number (-)", fontsize=7.4)
    ax_str.set_ylabel("Two-way time (ms)", fontsize=7.4)
    ax_str.tick_params(labelsize=6.6)
    ax_str.grid(color=GRID, lw=0.4)
    ax_str.set_axisbelow(True)
    ax_str.legend(frameon=False, fontsize=6.2, loc="lower left",
                  labelspacing=0.3, handlelength=1.3, handletextpad=0.5)
    ax_str.set_title("Picked horizon times share one structural shape",
                     fontsize=8.5, fontweight="bold")

    relief = float(np.nanmax(hz_t[0]) - np.nanmin(hz_t[0]))
    panel_labels_inches(fig, [ax, ax_tie, ax_str], dx=-0.58, dy=0.34)
    savefig(fig, "813_seis_section_horizons.png")


if __name__ == "__main__":
    main()
