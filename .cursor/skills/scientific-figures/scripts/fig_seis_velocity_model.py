"""Interval velocity model with traced rays, and what migration does to it.

The model is a six-layer section with dipping, folded interfaces and a
downward gradient inside each layer.  Rays are traced through it properly:
each ray marches in straight segments, and at every interface crossing the
direction is updated by the vector form of Snell's law using the local
interface normal, so the ray fan bends where the contrast is and flattens
where it is not.

The 1D companion is the velocity column read out of the same grid at 5 km,
with the RMS and average curves derived from it by vertical integration.  The
migration pair is a diffraction experiment: the unmigrated panel is the sum of
the diffraction hyperbolas radiated by every scatter point on a faulted
reflector, the migrated panel is the same scatter points imaged.  The focusing
widths quoted on the two panels are measured from those arrays.

The velocity panel is true scale (no vertical exaggeration), so its colour bar
is pinned in inches by ``figkit.safe_colorbar`` rather than carved out of the
nominal axes box.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    shared_norm)

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"
RAY = "#b23a48"

XMAX, ZMAX = 10.0, 4.0
V_LAYER = np.array([1700.0, 2150.0, 2650.0, 3250.0, 3850.0, 4500.0])
V_GRAD = 320.0                      # m/s per km below each layer top
SRC_X, REFLECT_K = 0.8, 3
X_PROFILE = 5.0
X_FAULT, T_SLICE = 3.0, 0.715
ANG_LO, ANG_HI = 5.0, 30.0


def interfaces(x):
    """Depths (km) of the five interfaces at horizontal position ``x``."""
    x = np.asarray(x, dtype=float)
    return np.array([
        0.45 + 0.020 * x + 0.09 * np.cos(x / 2.2),
        1.05 + 0.035 * x - 0.24 * np.exp(-((x - 4.2) / 2.0) ** 2),
        1.85 + 0.050 * x - 0.32 * np.exp(-((x - 4.6) / 2.4) ** 2),
        2.65 + 0.055 * x - 0.24 * np.exp(-((x - 5.0) / 2.8) ** 2),
        3.35 + 0.040 * x + 0.05 * np.cos(x / 3.0)])


def iface(k, x):
    return float(interfaces(np.array([x]))[k, 0])


def normal(k, x, h=1.0e-4):
    """Unit normal of interface ``k`` at ``x``, pointing downward."""
    slope = (iface(k, x + h) - iface(k, x - h)) / (2.0 * h)
    n = np.array([-slope, 1.0])
    return n / np.linalg.norm(n)


def velocity_grid(xg, zg):
    zi = interfaces(xg)
    tops = np.vstack([np.zeros_like(xg), zi])
    v = np.empty((zg.size, xg.size))
    for j in range(xg.size):
        col = np.searchsorted(zi[:, j], zg)
        v[:, j] = V_LAYER[np.clip(col, 0, 5)] + V_GRAD * (
            zg - tops[np.clip(col, 0, 5), j])
    return v


def snell(d, n, v1, v2):
    """Vector Snell refraction; falls back to reflection past the critical angle."""
    n = n if np.dot(d, n) > 0.0 else -n
    ci = float(np.dot(d, n))
    eta = v2 / v1
    st = eta * np.sqrt(max(1.0 - ci * ci, 0.0))
    if st >= 1.0:
        t = d - 2.0 * ci * n
    else:
        t = eta * d + (np.sqrt(1.0 - st * st) - eta * ci) * n
    return t / np.linalg.norm(t)


def trace_ray(angle_deg, step=0.008, n_max=3000):
    d = np.array([np.sin(np.deg2rad(angle_deg)),
                  np.cos(np.deg2rad(angle_deg))])
    p = np.array([SRC_X, 0.0])
    pts, k, down = [p.copy()], 0, True
    for _ in range(n_max):
        q = p + d * step
        if not (0.0 <= q[0] <= XMAX) or q[1] > ZMAX:
            break
        if down and k <= 4 and q[1] > iface(k, q[0]):
            n = normal(k, q[0])
            if k == REFLECT_K:
                d = d - 2.0 * float(np.dot(d, n)) * n
                down = False
            else:
                d = snell(d, n, V_LAYER[k], V_LAYER[k + 1])
                k += 1
        elif (not down) and k >= 1 and q[1] < iface(k - 1, q[0]):
            n = normal(k - 1, q[0])
            d = snell(d, n, V_LAYER[k], V_LAYER[k - 1])
            k -= 1
        p = p + d * step
        pts.append(p.copy())
        if (not down) and p[1] <= 0.0:
            break
    return np.array(pts)


def ricker(freq, dt, n=81):
    tt = (np.arange(n) - n // 2) * dt
    a = (np.pi * freq * tt) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def reflector_t(x):
    """Two-way time of the faulted reflector, in seconds."""
    return np.where(np.asarray(x) < X_FAULT, 0.600, 0.820) + 0.012 * x


def migration_pair(rng):
    """Diffraction modelling of a faulted reflector, before and after imaging."""
    nx, nt, dt = 170, 320, 0.004
    x = np.linspace(0.0, 6.0, nx)
    t = np.arange(nt) * dt
    v_mig = 2600.0
    xs = np.linspace(0.02, 5.98, 420)
    ts = reflector_t(xs)
    amp = np.where(xs < X_FAULT, 1.0, 0.92)

    unm = np.zeros((nt, nx))
    for x0, t0, a in zip(xs, ts, amp):
        hyp = np.sqrt(t0 ** 2 + (2.0 * (x - x0) * 1000.0 / v_mig) ** 2)
        w = a / (1.0 + 1.15 * np.abs(x - x0)) ** 1.15
        i = np.round(hyp / dt).astype(int)
        ok = i < nt
        unm[i[ok], np.arange(nx)[ok]] += w[ok]

    mig = np.zeros((nt, nx))
    tt, aa = reflector_t(x), np.where(x < X_FAULT, 1.0, 0.92)
    for j in range(nx):
        i = int(round(tt[j] / dt))
        if i < nt:
            mig[i, j] += aa[j]

    wav = ricker(28.0, dt)
    unm = np.apply_along_axis(lambda c: np.convolve(c, wav, "same"), 0, unm)
    mig = np.apply_along_axis(lambda c: np.convolve(c, wav, "same"), 0, mig)
    unm = unm / np.abs(unm).max() + rng.normal(0.0, 0.004, unm.shape)
    mig = mig / np.abs(mig).max() + rng.normal(0.0, 0.004, mig.shape)
    return x, t, unm, mig


def slice_energy(t, sec, t0):
    """RMS amplitude on a time slice, as a percentage of the section peak."""
    i = int(np.argmin(np.abs(t - t0)))
    row = sec[max(i - 2, 0):i + 3]
    return 100.0 * float(np.sqrt((row ** 2).mean()) / np.abs(sec).max())


def main():
    set_pub_style()
    rng = np.random.default_rng(9042)

    xg = np.linspace(0.0, XMAX, 340)
    zg = np.linspace(0.0, ZMAX, 220)
    vel = velocity_grid(xg, zg)
    zi = interfaces(xg)

    rays = [trace_ray(a) for a in np.linspace(ANG_LO, ANG_HI, 9)]

    # ---- 1D column at X_PROFILE ---------------------------------------------
    jc = int(np.argmin(np.abs(xg - X_PROFILE)))
    v1d = vel[:, jc]
    dz = np.gradient(zg)
    dt_ow = dz / (v1d / 1000.0)
    t_ow = np.cumsum(dt_ow)
    v_rms = np.sqrt(np.cumsum(v1d ** 2 * dt_ow) / np.maximum(t_ow, 1e-9))
    v_avg = np.cumsum(dz) * 1000.0 / np.maximum(t_ow, 1e-9)

    xm, tm, unm, mig = migration_pair(rng)
    t_tip = float(reflector_t(np.array([X_FAULT + 0.01]))[0])
    e_unm = slice_energy(tm, unm, T_SLICE)
    e_mig = slice_energy(tm, mig, T_SLICE)

    # ---- canvas ---------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.82, 3.25, 5.20, 2.08))
    ax_v = axes_inches(fig, (7.30, 3.25, 2.55, 2.08))
    ax_u = axes_inches(fig, (0.82, 0.98, 4.25, 1.34))
    ax_m = axes_inches(fig, (5.55, 0.98, 4.25, 1.34))

    # ---- velocity model --------------------------------------------------------
    pm = ax.pcolormesh(xg, zg, vel, cmap=get_cmap("cb_sequential_blue"),
                       shading="gouraud", rasterized=True)
    for k in range(zi.shape[0]):
        ax.plot(xg, zi[k], color="#f0f0f0", lw=1.1, zorder=3)
        ax.plot(xg, zi[k], color=RULE, lw=0.55, zorder=4)
    for r in rays:
        ax.plot(r[:, 0], r[:, 1], color=RAY, lw=0.85, alpha=0.9, zorder=5)
    ax.plot([SRC_X], [0.0], marker="v", ms=6.0, mfc=RAY, mec="white",
            mew=0.7, clip_on=False, zorder=6)
    hs = ax.text(SRC_X + 0.14, 0.16, "shot", fontsize=6.6, color=RAY,
                 fontweight="bold", ha="left", va="top", zorder=6)
    hs.set_path_effects([pe.withStroke(linewidth=1.9, foreground="white")])
    hp = ax.text(X_PROFILE + 0.10, ZMAX - 0.10, "profile", fontsize=6.4,
                 color=INK, ha="left", va="bottom", zorder=6)
    hp.set_path_effects([pe.withStroke(linewidth=1.9, foreground="white")])
    ax.axvline(X_PROFILE, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=5)
    ax.set_xlim(0.0, XMAX)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_aspect(1.0)
    ax.set_xlabel("Distance (km)", fontsize=8.0)
    ax.set_ylabel("Depth (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    safe_colorbar(fig, pm, ax, label="interval velocity (m/s)", width=0.11,
                  pad=0.07)
    ax.set_title("Rays bend at every contrast and reflect off interface 4",
                 fontsize=8.8, fontweight="bold")

    # ---- 1D velocity -----------------------------------------------------------
    ax_v.plot(v1d, zg, color=RULE, lw=1.2, label="Interval")
    ax_v.plot(v_rms, zg, color="#b23a48", lw=1.3, label="RMS")
    ax_v.plot(v_avg, zg, color="#2e7d5b", lw=1.1, ls=(0, (4, 2)),
              label="Average")
    for k in range(zi.shape[0]):
        ax_v.axhline(zi[k, jc], color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax_v.set_ylim(ZMAX, 0.0)
    ax_v.set_xlim(1400.0, 5200.0)
    ax_v.set_xlabel("Velocity (m/s)", fontsize=7.4)
    ax_v.set_ylabel("Depth (km)", fontsize=7.4)
    ax_v.tick_params(labelsize=6.6)
    ax_v.grid(color=GRID, lw=0.4)
    ax_v.set_axisbelow(True)
    ax_v.legend(frameon=False, fontsize=6.4, loc="lower left",
                labelspacing=0.3, handlelength=1.4, handletextpad=0.5)
    ax_v.set_title("Column at %.0f km, RMS from the same grid" % X_PROFILE,
                   fontsize=8.5, fontweight="bold")

    # ---- migration pair ---------------------------------------------------------
    norm = shared_norm([unm, mig], robust=(1.0, 99.0), symmetric=True)
    for axx, sec, name, en in ((ax_u, unm, "Before migration", e_unm),
                               (ax_m, mig, "After migration", e_mig)):
        axx.imshow(sec, cmap=get_cmap("cb_diverging"), norm=norm,
                   extent=[xm[0], xm[-1], tm[-1], tm[0]], aspect="auto",
                   interpolation="bilinear")
        axx.set_ylim(1.12, 0.34)
        axx.set_xlabel("Distance (km)", fontsize=7.4)
        axx.tick_params(labelsize=6.6)
        axx.set_title(name, fontsize=8.5, fontweight="bold")
        axx.axhline(T_SLICE, color=INK, lw=0.7, ls=(0, (4, 2.4)))
        lab = axx.text(0.978, 0.06,
                       "energy on the %.2f s slice: %.1f %%" % (T_SLICE, en),
                       transform=axx.transAxes, ha="right", va="bottom",
                       fontsize=6.4, color=INK)
        lab.set_path_effects([pe.withStroke(linewidth=2.2,
                                            foreground="white")])
        axx.annotate("fault tip", xy=(X_FAULT, t_tip),
                     xytext=(X_FAULT - 0.45, t_tip + 0.20), fontsize=6.4,
                     color=INK, ha="right", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                     shrinkA=0, shrinkB=2))
    ax_u.set_ylabel("Two-way time (s)", fontsize=7.4)
    ax_m.tick_params(labelleft=False)

    panel_labels_inches(fig, [ax, ax_v, ax_u, ax_m], dx=-0.52, dy=0.30)
    savefig(fig, "815_seis_velocity_model.png")


if __name__ == "__main__":
    main()
