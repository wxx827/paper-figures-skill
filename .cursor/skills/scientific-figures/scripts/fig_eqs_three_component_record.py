"""Regional record section and the three-component record of one station.

Every trace is synthesised from the same crustal model: a 32 km layer over a
mantle half-space with the source at 12 km depth.  Direct Pg and Sg arrive on
the straight-line distance, Pn and Sn are head waves that only exist beyond
the critical distance, and Lg follows at its group velocity, so the move-out
curves drawn over the gather are the same arrival times that positioned the
wavelets.  The three-component panel splits each arrival into radial and
transverse motion about the true source-station azimuth, and the hodogram
panel recovers that azimuth back from the data with a principal-component fit
to the horizontal particle motion inside the P window; the sign ambiguity is
resolved with the vertical component.  Amplitudes are trace-normalised, as in
any regional gather.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#22406e"
BLUE = "#3b5b92"
PALE = "#a8bcd8"
RED = "#b23a48"
GOLD = "#c9a227"
TEAL = "#2a7f7a"
GREY = "#8a8a8a"
TRACE = "#33383d"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

V1P, V2P = 6.05, 8.05        # km/s, crust and mantle P velocity
VPVS = 1.73
H_CRUST, Z_SRC = 32.0, 12.0  # km
V_LG = 3.10                  # km/s, Lg group velocity
DT, T_MAX = 0.02, 200.0      # s
DIST = np.linspace(26.0, 520.0, 16)      # km
AZ_TRUE = 118.0              # deg, source to station azimuth
PICK_STATION = 8             # index used by the two side panels
P_WIN = 2.4                  # s, particle-motion window


def v1s():
    return V1P / VPVS


def v2s():
    return V2P / VPVS


def t_direct(x, v):
    return np.sqrt(np.asarray(x, dtype=float) ** 2 + Z_SRC ** 2) / v


def t_head(x, v1, v2):
    ic = np.arcsin(v1 / v2)
    return np.asarray(x, dtype=float) / v2 + (2.0 * H_CRUST - Z_SRC) \
        * np.cos(ic) / v1


def x_critical(v1, v2):
    ic = np.arcsin(v1 / v2)
    return (H_CRUST + (H_CRUST - Z_SRC)) * np.tan(ic)


def x_crossover(v1, v2):
    return brentq(lambda x: t_direct(x, v1) - t_head(x, v1, v2), 20.0, 900.0)


def gabor(t, f, cycles=1.5):
    s = cycles / f
    return np.exp(-(t / s) ** 2) * np.sin(2.0 * np.pi * f * t)


def band_noise(rng, n, f_lo, f_hi):
    w = rng.normal(size=n)
    freq = np.fft.rfftfreq(n, DT)
    fc, bw = 0.5 * (f_lo + f_hi), 0.5 * (f_hi - f_lo)
    resp = np.exp(-((freq - fc) / bw) ** 4)
    out = np.fft.irfft(np.fft.rfft(w) * resp, n)
    return out / (np.abs(out).max() + 1e-12)


def phase_train(t, arrivals):
    """Sum of Gabor wavelets, one per (time, amplitude, frequency) arrival."""
    out = np.zeros_like(t)
    for t0, amp, f in arrivals:
        if amp <= 0.0:
            continue
        m = np.abs(t - t0) < 6.0 / f
        out[m] += amp * gabor(t[m] - t0, f)
    return out


def taper(x, x0, width=25.0):
    """Zero below ``x0``, one well beyond it, smooth in between."""
    return 0.5 * (1.0 + np.tanh((np.asarray(x, dtype=float) - x0) / width))


def arrivals_at(x):
    """Times, amplitudes and centre frequencies of the five regional phases."""
    r = np.sqrt(x ** 2 + Z_SRC ** 2)
    g = (100.0 / r) ** 1.15
    gx = (100.0 / x) ** 1.15
    xc_p = x_critical(V1P, V2P)
    xc_s = x_critical(v1s(), v2s())
    return dict(
        Pg=(t_direct(x, V1P), 1.00 * g, 3.0),
        Pn=(t_head(x, V1P, V2P), 0.62 * gx * taper(x, xc_p), 2.4),
        Sg=(t_direct(x, v1s()), 1.75 * g, 1.8),
        Sn=(t_head(x, v1s(), v2s()), 0.90 * gx * taper(x, xc_s), 1.5),
        Lg=(x / V_LG, 1.95 * gx, 0.55))


def synth_vertical(rng, t, x):
    a = arrivals_at(x)
    p = phase_train(t, [a["Pg"], a["Pn"]])
    s = phase_train(t, [a["Sg"], a["Sn"]])
    lg = phase_train(t, [a["Lg"]])
    t_s = min(a["Sg"][0], a["Sn"][0])
    coda = (np.exp(-np.clip(t - t_s, 0.0, None) / 42.0) * (t > t_s)
            * band_noise(rng, t.size, 0.4, 4.0) * 0.34 * a["Sg"][1])
    noise = band_noise(rng, t.size, 0.2, 8.0) * 0.05 * a["Pg"][1]
    return 0.85 * p + 0.55 * s + 1.00 * lg + coda + noise


def synth_three(rng, t, x, az_deg):
    """Vertical, north and east traces built from radial / transverse motion."""
    a = arrivals_at(x)
    p = phase_train(t, [a["Pg"], a["Pn"]])
    s = phase_train(t, [a["Sg"], a["Sn"]])
    lg = phase_train(t, [a["Lg"]])
    t_s = min(a["Sg"][0], a["Sn"][0])
    env = np.exp(-np.clip(t - t_s, 0.0, None) / 42.0) * (t > t_s)

    z = (0.85 * p + 0.50 * s + 1.00 * lg
         + 0.30 * a["Sg"][1] * env * band_noise(rng, t.size, 0.4, 4.0)
         + 0.075 * a["Pg"][1] * band_noise(rng, t.size, 0.2, 8.0))
    rad = (0.62 * p + 0.38 * s + 0.80 * lg
           + 0.28 * a["Sg"][1] * env * band_noise(rng, t.size, 0.4, 4.0)
           + 0.075 * a["Pg"][1] * band_noise(rng, t.size, 0.2, 8.0))
    tra = (0.05 * p + 1.05 * s + 0.30 * lg
           + 0.28 * a["Sg"][1] * env * band_noise(rng, t.size, 0.4, 4.0)
           + 0.075 * a["Pg"][1] * band_noise(rng, t.size, 0.2, 8.0))
    c, sn = np.cos(np.radians(az_deg)), np.sin(np.radians(az_deg))
    return z, rad * c - tra * sn, rad * sn + tra * c


def polarization(n, e, z):
    """Back-azimuth in degrees from a PCA of the horizontal P-wave motion."""
    m = np.vstack([n, e])
    m = m - m.mean(axis=1, keepdims=True)
    w, v = np.linalg.eigh(m @ m.T)
    vec = v[:, -1]
    rad = vec[0] * n + vec[1] * e
    if float(np.dot(rad, z)) < 0.0:
        vec = -vec
    az = np.degrees(np.arctan2(vec[1], vec[0])) % 360.0
    lin = 1.0 - w[0] / w[-1]
    return az, (az + 180.0) % 360.0, lin


def section_panel(ax, t, traces, xc, x_over):
    gain = 30.0
    for x, tr in zip(DIST, traces):
        y = x + gain * tr / np.abs(tr).max()
        ax.plot(t, y, color=TRACE, lw=0.45, zorder=5)
        ax.fill_between(t, x, y, where=y > x, facecolor=TRACE, alpha=0.55,
                        linewidth=0.0, zorder=4)

    xx = np.linspace(DIST[0], DIST[-1], 220)
    curves = [(t_direct(xx, V1P), "Pg", DEEP, "-"),
              (t_head(xx, V1P, V2P), "Pn", BLUE, (0, (4.0, 2.0))),
              (t_direct(xx, v1s()), "Sg", RED, "-"),
              (t_head(xx, v1s(), v2s()), "Sn", "#c0603f", (0, (4.0, 2.0))),
              (xx / V_LG, "Lg", TEAL, (0, (1.6, 1.8)))]
    for tt, lab, col, ls in curves:
        ax.plot(tt, xx, color=col, lw=1.2, ls=ls, zorder=8)
        ax.text(tt[-1] + 1.5, xx[-1] + 4.0, lab, fontsize=7.2,
                fontweight="bold", color=col, ha="left", va="bottom",
                zorder=9)

    ax.axhline(x_over, color=GOLD, lw=1.0, ls=(0, (3.4, 2.2)), zorder=7)
    ax.text(178.0, x_over + 5.0,
            "Pn overtakes Pg at {:.0f} km".format(x_over), fontsize=6.4,
            color="#8a6f12", ha="right", va="bottom", fontweight="bold",
            zorder=9)
    ax.axhline(xc, color=GREY, lw=0.9, ls=(0, (2.2, 2.0)), zorder=7)
    ax.text(178.0, xc + 5.0,
            "head waves start at the critical distance, {:.0f} km".format(xc),
            fontsize=6.4, color=SUB, ha="right", va="bottom", zorder=9)
    ax.axhline(DIST[PICK_STATION], color=RED, lw=0.9, zorder=7)
    ax.text(178.0, DIST[PICK_STATION] + 5.0,
            "station in panels (b) and (c)", fontsize=6.4, color=RED,
            ha="right", va="bottom", fontweight="bold", zorder=9)

    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 570.0)
    ax.set_xlabel("time after origin (s)")
    ax.set_ylabel("epicentral distance (km)")
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.45, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Vertical-component record section, 16 stations",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)


def three_panel(ax, t, comps, tp, ts, x_true, x_est):
    names = ["Z (vertical)", "N (north)", "E (east)"]
    for k, (tr, lab) in enumerate(zip(comps, names)):
        off = 2 - k
        ax.plot(t, off + 0.42 * tr / np.abs(tr).max(), color=TRACE, lw=0.5,
                zorder=5)
        ax.text(2.0, off + 0.40, lab, fontsize=6.2, color=SUB, ha="left",
                va="bottom", zorder=7)
    for tpick, lab, col in [(tp, "P", DEEP), (ts, "S", RED)]:
        ax.axvline(tpick, color=col, lw=1.0, ls=(0, (3.2, 2.0)), zorder=6)
        ax.text(tpick + 1.2, 2.62, lab, fontsize=7.0, fontweight="bold",
                color=col, ha="left", va="center", zorder=8)
    ax.annotate("", xy=(tp, -0.70), xytext=(ts, -0.70),
                arrowprops=dict(arrowstyle="<|-|>", color=GOLD, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(0.5 * (tp + ts), -0.80,
            "S - P = {:.1f} s gives {:.0f} km at direct-wave speed\n"
            "true distance {:.0f} km: Pn and Sn arrive first here"
            .format(ts - tp, x_est, x_true), fontsize=6.2, color="#8a6f12",
            ha="center", va="top", fontweight="bold", linespacing=1.4,
            zorder=8)
    ax.set_xlim(0.0, 120.0)
    ax.set_ylim(-1.42, 2.85)
    ax.set_yticks([])
    ax.set_xlabel("time after origin (s)", fontsize=6.9)
    ax.set_ylabel("normalised ground velocity", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Three components at {:.0f} km".format(x_true),
                 fontsize=8.0, fontweight="bold", color=INK, pad=3.0)


def hodogram_panel(ax, n_win, e_win, az_est, baz_est, lin):
    lim = 1.18 * max(np.abs(n_win).max(), np.abs(e_win).max())
    ax.plot(e_win, n_win, color=TRACE, lw=0.7, zorder=5)
    ax.plot([e_win[0]], [n_win[0]], marker="o", ms=3.2, mfc=GOLD,
            mec="white", mew=0.6, zorder=7)
    a = np.radians(az_est)
    ax.plot([-lim * np.sin(a), lim * np.sin(a)],
            [-lim * np.cos(a), lim * np.cos(a)], color=RED, lw=1.2,
            ls=(0, (4.0, 2.2)), zorder=6)
    at = np.radians(AZ_TRUE)
    ax.plot([-lim * np.sin(at), lim * np.sin(at)],
            [-lim * np.cos(at), lim * np.cos(at)], color=DEEP, lw=1.0,
            zorder=6)
    ax.axhline(0.0, color=GREY, lw=0.5, zorder=3)
    ax.axvline(0.0, color=GREY, lw=0.5, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("east motion (normalised)", fontsize=6.9)
    ax.set_ylabel("north motion (normalised)", fontsize=6.9)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.45, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=DEEP, lw=1.0,
                      label="true azimuth {:.0f} deg".format(AZ_TRUE)),
               Line2D([], [], color=RED, lw=1.2, ls=(0, (4.0, 2.2)),
                      label="fitted {:.0f} deg, baz {:.0f} deg"
                      .format(az_est, baz_est))]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.5, labelspacing=0.22, borderaxespad=0.28)
    ax.text(0.97, 0.04, "rectilinearity {:.2f}".format(lin),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.1,
            color=MUTED, zorder=8)
    ax.set_title("P-window particle motion, {:.1f} s".format(P_WIN),
                 fontsize=8.0, fontweight="bold", color=INK, pad=3.0)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(19640328)

    t = np.arange(0.0, T_MAX, DT)
    traces = [synth_vertical(rng, t, x) for x in DIST]

    x_pick = float(DIST[PICK_STATION])
    z, n, e = synth_three(rng, t, x_pick, AZ_TRUE)
    a = arrivals_at(x_pick)
    tp = float(min(a["Pg"][0], a["Pn"][0]))
    ts = float(min(a["Sg"][0], a["Sn"][0]))
    x_est = (ts - tp) / (1.0 / v1s() - 1.0 / V1P)

    w = (t >= tp - 0.25) & (t <= tp - 0.25 + P_WIN)
    az_est, baz_est, lin = polarization(n[w], e[w], z[w])

    xc = x_critical(V1P, V2P)
    x_over = x_crossover(V1P, V2P)

    assert 40.0 < xc < 160.0, "critical distance off the plotted range"
    assert x_over > xc, "crossover must lie beyond the critical distance"
    assert abs(az_est - AZ_TRUE) < 8.0, "polarization fit lost the azimuth"
    assert lin > 0.75, "P window is not rectilinear enough to fit"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_sec = axes_inches(fig, MAIN)
    ax_3c = axes_inches(fig, AUX_T)
    ax_hod = axes_inches(fig, AUX_B)

    section_panel(ax_sec, t, traces, xc, x_over)
    three_panel(ax_3c, t, (z, n, e), tp, ts, x_pick, x_est)
    hodogram_panel(ax_hod, n[w], e[w], az_est, baz_est, lin)



    panel_labels_inches(fig, [ax_sec, ax_3c, ax_hod], dx=-0.60, dy=0.12)
    savefig(fig, "1053_eqs_three_component_record.png")


if __name__ == "__main__":
    main()
