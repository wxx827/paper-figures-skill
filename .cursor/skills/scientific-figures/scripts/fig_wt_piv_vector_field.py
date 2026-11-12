"""PIV of the separated wake behind an aerofoil: field, deficit, shear stress.

Main panel: a phase-locked particle-image-velocimetry plane behind a NACA 0012
section held near stall in a low-speed tunnel, drawn on an equal-aspect axes so
the vortex street keeps its true spacing.  Spanwise vorticity is the background,
a decimated quiver carries the direction and speed of the same field, the
section itself is a solid black silhouette (no data is measured through the
model), and the vortex cores are ringed where the vorticity field actually has
its local extrema rather than where the eye guesses they are.

The two companion panels are what a PIV plane is normally reduced to.  The
ensemble-mean streamwise profiles on three vertical traverses show the wake
deficit filling in with downstream distance, and the Reynolds shear stress on
the same three traverses shows the two-lobed signature that does the filling.
Both are computed from an ensemble of shedding phases of the same velocity
model that produced the snapshot, so the profiles and the picture agree.

Illustrative synthetic data: the wake is a Gaussian deficit plus a staggered row
of Lamb-Oseen vortices convecting at a fraction of the free-stream speed.
"""
import textwrap

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import maximum_filter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"

FIG_W, FIG_H = 10.6, 6.4
MAIN_RECT = (0.50, 1.20, 5.36, 3.50)
AUX_T_RECT = (7.28, 3.52, 3.02, 1.86)
AUX_B_RECT = (7.28, 0.82, 3.02, 1.86)

U_INF = 15.0            # m/s, free-stream speed in the working section
CHORD = 100.0           # mm, model chord
ALPHA = 12.0            # deg, geometric incidence
RE_C = 2.1e5            # chord Reynolds number (dimensionless)

NX, NY = 380, 250
X0, X1 = -45.0, 265.0   # mm, field of view
Y0, Y1 = -126.0, 76.0   # mm

N_VORT = 11             # shed vortices held in the field of view
SPACING = 36.0          # mm, streamwise spacing of the street
STAGGER = 7.0           # mm, lateral offset of the two rows
STATIONS = (140.0, 195.0, 250.0)   # mm, traverse positions (x from the LE)


def section_polygon(n=141):
    """NACA 0012 outline in mm, leading edge at the origin, pitched to ALPHA."""
    beta = np.linspace(0.0, np.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * 0.12 * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                       + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)
    xs = np.concatenate([xc[::-1], xc[1:]])
    ys = np.concatenate([yt[::-1], -yt[1:]])
    a = np.deg2rad(ALPHA)
    return (CHORD * (xs * np.cos(a) + ys * np.sin(a)),
            CHORD * (-xs * np.sin(a) + ys * np.cos(a)))


def trailing_edge():
    a = np.deg2rad(ALPHA)
    return CHORD * np.cos(a), -CHORD * np.sin(a)


def wake_axis(x):
    """Centreline of the wake, drifting down under the induced downwash (mm)."""
    xte, yte = trailing_edge()
    s = np.clip(x - xte, 0.0, None)
    return yte - 11.0 * (1.0 - np.exp(-s / 70.0))


def lamb_oseen(X, Y, xc, yc, gamma, rc):
    """Velocity of one Lamb-Oseen vortex; gamma in mm^2/s-equivalent units."""
    dx, dy = X - xc, Y - yc
    r2 = dx * dx + dy * dy + 1e-9
    f = gamma / (2.0 * np.pi * r2) * (1.0 - np.exp(-r2 / (rc * rc)))
    return -f * dy, f * dx


def velocity(X, Y, phase):
    """Streamwise / transverse velocity in m/s at one shedding phase."""
    xte, _ = trailing_edge()
    u = np.full(np.shape(X), U_INF, dtype=float)
    v = np.zeros(np.shape(X), dtype=float)

    # Bound circulation: a wide, weak core so it bends the arrows without
    # printing a false blob of vorticity outside the model.
    du, dv = lamb_oseen(X, Y, 0.30 * CHORD, -0.05 * CHORD,
                        -0.26 * U_INF * CHORD, 1.60 * CHORD)
    u, v = u + du, v + dv

    # Mean wake deficit, widening and filling in downstream.
    s = np.clip(X - xte, 0.0, None)
    ud = 0.72 * U_INF / (1.0 + s / 42.0) ** 0.62
    b = 9.5 * np.sqrt(1.0 + s / 32.0)
    eta = (Y - wake_axis(X)) / b
    u = u - np.where(X > xte, ud * np.exp(-eta ** 2), 0.0)

    # Staggered Lamb-Oseen street convecting at 0.86 U_inf.
    for k in range(N_VORT):
        xv = xte + 14.0 + (k + phase) * SPACING
        sgn = 1.0 if k % 2 else -1.0
        yv = wake_axis(xv) + sgn * STAGGER
        age = np.clip((xv - xte) / SPACING, 0.0, None)
        rc = 4.6 * np.sqrt(1.0 + age / 3.4)
        g = sgn * 3.4e2 * np.exp(-age / 16.0)
        du, dv = lamb_oseen(X, Y, xv, yv, g, rc)
        u, v = u + du, v + dv
    return u, v


def masked_grid():
    """Snapshot field on the PIV grid with the model interior blanked."""
    x = np.linspace(X0, X1, NX)
    y = np.linspace(Y0, Y1, NY)
    X, Y = np.meshgrid(x, y)
    u, v = velocity(X, Y, 0.24)
    px, py = section_polygon()
    inside = Path(np.column_stack([px, py])).contains_points(
        np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    u[inside] = np.nan
    v[inside] = np.nan
    dx = (x[1] - x[0]) * 1e-3
    dy = (y[1] - y[0]) * 1e-3
    vort = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    return X, Y, u, v, vort, inside


def vortex_cores(X, Y, vort, inside, n_keep=5):
    """Local vorticity extrema in the wake: the cores the eye should trust."""
    xte, _ = trailing_edge()
    w = np.where(np.isfinite(vort) & (X > xte + 8.0), vort, 0.0)
    peak = np.nanmax(np.abs(w))
    found = []
    for sgn in (+1.0, -1.0):
        f = sgn * w
        loc = (f == maximum_filter(f, size=27)) & (f > 0.42 * peak)
        for j, i in zip(*np.nonzero(loc)):
            found.append((abs(f[j, i]), float(X[j, i]), float(Y[j, i]), sgn))
    found.sort(reverse=True)
    return [(x, y, s, m) for m, x, y, s in found[:n_keep]], peak


def ensemble_profiles(n_phase=72, seed=4207):
    """Ensemble mean and Reynolds stresses on the three vertical traverses."""
    rng = np.random.default_rng(seed)
    yq = np.linspace(-95.0, 55.0, 241)
    out = {}
    for xs in STATIONS:
        us, vs = [], []
        for ph in rng.permutation(n_phase) / n_phase:
            u, v = velocity(np.full_like(yq, xs), yq, float(ph))
            us.append(u)
            vs.append(v)
        us = np.array(us)
        vs = np.array(vs)
        um, vm = us.mean(axis=0), vs.mean(axis=0)
        uv = ((us - um) * (vs - vm)).mean(axis=0)
        out[xs] = (yq, um, -uv)
    return out


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    X, Y, u, v, vort, inside = masked_grid()
    cores, peak = vortex_cores(X, Y, vort, inside)
    prof = ensemble_profiles()
    px, py = section_polygon()
    xte, yte = trailing_edge()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_u = axes_inches(fig, AUX_T_RECT)
    ax_r = axes_inches(fig, AUX_B_RECT)
    for a in (ax_u, ax_r):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the PIV plane -----------------------------------------------
    ax.set_aspect("equal")
    lim = float(np.nanpercentile(np.abs(vort), 99.2))
    im = ax.pcolormesh(X, Y, vort, cmap=get_cmap("cb_diverging"),
                       vmin=-lim, vmax=lim, shading="gouraud", zorder=2,
                       rasterized=True)
    sk = 13
    ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], u[::sk, ::sk], v[::sk, ::sk],
              color="#2b2b2b", scale=430.0, width=0.0022, headwidth=3.6,
              headlength=4.2, headaxislength=3.6, alpha=0.85, zorder=5)
    ax.add_patch(Polygon(np.column_stack([px, py]), closed=True,
                         facecolor="#111111", edgecolor="#111111", lw=0.8,
                         zorder=7))

    for k, (cx, cy, sgn, mag) in enumerate(cores):
        ax.add_patch(Circle((cx, cy), 13.0, facecolor="none",
                            edgecolor=INK if sgn > 0 else INK, lw=1.0,
                            ls=(0, (3.2, 1.8)) if sgn < 0 else "-", zorder=8))
        halo_text(ax, cx, cy + 15.5, f"{sgn * mag:+.0f}", ha="center",
                  va="bottom", fontsize=6.2, color="white", halo="#1a1a1a",
                  halo_lw=1.9, zorder=9)
    for xs in STATIONS:
        ax.plot([xs, xs], [-95.0, 55.0], color=GOLD, lw=1.0,
                ls=(0, (3.4, 2.0)), zorder=6)
        halo_text(ax, xs, 57.0, f"{xs / CHORD:.1f}c", ha="center", va="bottom",
                  fontsize=6.4, color=GOLD, halo="white", halo_lw=2.0,
                  zorder=9)
    ax.plot([xte], [yte], marker="o", ms=3.4, mfc="white", mec=INK, mew=1.0,
            zorder=9)

    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_xticks(np.arange(0.0, 261.0, 50.0))
    ax.set_yticks(np.arange(-100.0, 76.0, 50.0))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("streamwise position x from the leading edge (mm)",
                  fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("transverse position y (mm)", fontsize=8.4)
    ax.set_title(f"Phase-locked PIV plane, NACA 0012 at {ALPHA:.0f} deg, "
                 f"$U_\\infty$ = {U_INF:.0f} m/s, $Re_c$ = "
                 f"{RE_C / 1e5:.1f} x 10$^5$",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.017, 0.035,
            f"rings mark the {len(cores)} strongest local vorticity extrema\n"
            f"peak |$\\omega_z$| {peak:.0f} 1/s; the bar is clipped at the "
            f"99.2nd percentile\nvectors decimated {sk}:1 each way; no data "
            "inside the model silhouette",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=11,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    safe_colorbar(fig, im, ax, label="spanwise vorticity $\\omega_z$ (1/s)",
                  width=0.13, pad=0.10, shrink=0.88, labelsize=7.0,
                  extend="both")

    # ---- aux (b): wake deficit ---------------------------------------------
    cols = (BLUE, GREEN, RED)
    ax_u.axvline(1.0, color=FAINT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    deficits, span = [], []
    for (xs, col) in zip(STATIONS, cols):
        yq, um, _ = prof[xs]
        ax_u.plot(um / U_INF, yq, color=col, lw=1.7, zorder=5,
                  label=f"x = {xs / CHORD:.1f}c")
        edge = float(np.mean([um[0], um[-1]]))
        deficits.append(100.0 * (edge - um.min()) / U_INF)
        span.append(um / U_INF)
    lo = min(float(s.min()) for s in span)
    hi = max(float(s.max()) for s in span)
    ax_u.set_xlim(lo - 0.05 * (hi - lo), hi + 0.06 * (hi - lo))
    ax_u.set_ylim(-95.0, 55.0)
    ax_u.tick_params(labelsize=6.8)
    ax_u.set_xlabel("mean streamwise velocity $\\bar{u}/U_\\infty$ "
                    "(dimensionless)", fontsize=7.5, labelpad=1.5)
    ax_u.set_ylabel("y (mm)", fontsize=7.5)
    ax_u.set_title("The wake fills in with distance", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_u.text(0.030, 0.045,
              f"peak deficit falls from {deficits[0]:.0f} % to "
              f"{deficits[-1]:.0f} % of $U_\\infty$\nbetween the first and "
              "last traverse",
              transform=ax_u.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_u.legend(loc="upper left", frameon=False, fontsize=6.6,
                handlelength=1.7, labelspacing=0.28, borderaxespad=0.25)

    # ---- aux (c): Reynolds shear stress ------------------------------------
    ax_r.axvline(0.0, color=FAINT, lw=0.8, zorder=1)
    peaks = []
    for (xs, col) in zip(STATIONS, cols):
        yq, _, ruv = prof[xs]
        ax_r.plot(ruv, yq, color=col, lw=1.7, zorder=5)
        peaks.append(float(np.abs(ruv).max()))
    ax_r.set_ylim(-95.0, 55.0)
    ax_r.tick_params(labelsize=6.8)
    ax_r.set_xlabel("Reynolds shear stress $-\\overline{u'v'}$ "
                    "(m$^2$/s$^2$)", fontsize=7.5, labelpad=1.5)
    ax_r.set_ylabel("y (mm)", fontsize=7.5)
    ax_r.set_title("Two lobes of opposite sign do the mixing", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_r.text(0.030, 0.045,
              f"peak magnitude {peaks[0]:.1f} m$^2$/s$^2$ at "
              f"{STATIONS[0] / CHORD:.1f}c, {peaks[-1]:.1f} m$^2$/s$^2$ at "
              f"{STATIONS[-1] / CHORD:.1f}c;\nthe sign change straddles the "
              "wake centreline",
              transform=ax_r.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_r.legend(handles=[Line2D([], [], color=c, lw=1.7,
                                label=f"{s / CHORD:.1f}c")
                         for s, c in zip(STATIONS, cols)],
                loc="upper left", frameon=False, fontsize=6.6, ncol=3,
                handlelength=1.4, columnspacing=1.0, borderaxespad=0.25)

    panel_labels_inches(fig, [ax, ax_u, ax_r], dx=-0.52, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 930")
    check_layout(fig, foot, [ax, ax_u, ax_r])
    savefig(fig, "930_wt_piv_vector_field.png")


if __name__ == "__main__":
    main()
