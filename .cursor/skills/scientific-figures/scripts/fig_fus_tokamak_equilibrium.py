"""Poloidal flux map of a single-null tokamak, and the q profile it implies.

The equilibrium is not drawn, it is solved.  A toroidal current distribution
J ~ (1 - rho^2)^1.7 is laid down as ~350 circular filaments on nested Miller
surfaces (elongation and triangularity ramping with rho, plus a Shafranov
shift), and the poloidal flux of every filament is the exact circular-loop
Green function written with complete elliptic integrals.  Eight poloidal-field
coils and a four-segment central solenoid then take the currents that a damped
least-squares fit picks so that psi is constant on the requested D-shaped
boundary *and* grad psi vanishes at the requested X-point, which is what makes
the separatrix close with a corner and send two open legs into the divertor.

Everything quoted afterwards is measured on that field.  The magnetic axis is
the maximum of psi, the last closed surface is the contour through the
X-point, elongation and triangularity come from its bounding box, and the
safety factor of each surface is the contour integral q = (B0 R0 / 2 pi) *
integral dl / (R |grad psi|) evaluated on the extracted polyline, so q95 and
the q = 1 and q = 2 radii are read off the same geometry the left panel draws.
The plasma current is normalised to put q95 at 3.3; the pressure profile is
the assumed p(psi_n), and its flux-surface volume average gives beta.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from scipy.special import ellipk, ellipe
from scipy.interpolate import RegularGridInterpolator
from contourpy import contour_generator, LineType

from pubstyle import set_pub_style, savefig, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches

MU0 = 4.0e-7 * np.pi

RED = "#b23a48"          # plasma / hot core
ORANGE = "#c0603f"
BLUE = "#3b5b92"         # magnets, coils
PURPLE = "#8e6bbf"       # scrape-off layer
GREY = "#808080"         # structure
STEEL = "#9a9a9a"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

R0, AMIN = 3.05, 1.05            # m
KAP_U, DEL_U = 1.75, 0.33        # upper elongation / triangularity
KAP_L, DEL_L = 1.72, 0.52        # lower (X-point) shaping
SHIFT = 0.13                     # Shafranov shift of the magnetic axis, m
B0 = 5.3                         # T on axis
Q95_TARGET = 3.3
P_AXIS = 0.62e6                  # Pa, assumed on-axis pressure

N_RHO, N_THETA = 12, 40          # plasma current filaments
NR, NZ = 240, 300                # flux grid

# (R, Z) of the shaping coils; all sit outside the flux grid window.
COILS = [(0.30, -1.60), (0.30, -0.55), (0.30, 0.55), (0.30, 1.60),
         (1.05, 2.45), (2.35, 2.55), (5.05, 1.85), (5.15, -1.55),
         (4.05, -2.50), (2.55, -2.55), (1.10, -2.30)]
DIV_COIL = 9                     # index of the divertor coil

R_LO, R_HI = 1.30, 4.90
Z_LO, Z_HI = -2.35, 2.20
FLOOR_R = np.array([R_LO, 1.95, 2.42, 2.72, 3.45, R_HI])
FLOOR_Z = np.array([-2.05, -2.05, -2.26, -2.26, -2.00, -2.00])

VIEW_R = (-0.05, 6.26)
VIEW_Z = (-2.95, 2.45)

FW, FH = 10.4, 6.4
MAIN = (0.60, 1.04, 4.86, 4.16)
AUX1 = (6.58, 3.28, 2.80, 1.92)
AUX2 = (6.58, 1.04, 2.80, 1.52)


# ---------------------------------------------------------------- geometry
def miller(rho, theta):
    """Nested D-shaped surfaces; rho = 1 is the requested plasma boundary."""
    kap = 1.0 + (np.where(np.sin(theta) >= 0.0, KAP_U, KAP_L) - 1.0) * rho ** 1.4
    dlt = np.where(np.sin(theta) >= 0.0, DEL_U, DEL_L) * rho ** 1.8
    r = (R0 + SHIFT * (1.0 - rho ** 2) +
         AMIN * rho * np.cos(theta + np.arcsin(dlt) * np.sin(theta)))
    z = kap * AMIN * rho * np.sin(theta)
    return r, z


def green(r, z, rc, zc):
    """Poloidal flux per radian at (r, z) from a unit circular loop."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    k2 = 4.0 * rc * r / ((r + rc) ** 2 + (z - zc) ** 2)
    k2 = np.clip(k2, 1e-14, 1.0 - 1e-10)
    k = np.sqrt(k2)
    return (MU0 / (np.pi * k)) * np.sqrt(rc * r) * (
        (1.0 - 0.5 * k2) * ellipk(k2) - ellipe(k2))


def filaments():
    """Current filaments and their weights, summing to unit plasma current."""
    rho = (np.arange(N_RHO) + 0.5) / N_RHO
    th = (np.arange(N_THETA) + 0.5) * 2.0 * np.pi / N_THETA
    rr, tt = np.meshgrid(rho, th, indexing="ij")
    fr, fz = miller(rr, tt)
    w = (1.0 - rr ** 2) ** 1.7 * rr
    w = w / w.sum()
    return fr.ravel(), fz.ravel(), w.ravel()


def psi_from(fr, fz, w, r, z, chunk=48):
    """Sum the Green function of every filament onto (r, z)."""
    out = np.zeros_like(np.asarray(r, dtype=float))
    for i0 in range(0, fr.size, chunk):
        sl = slice(i0, i0 + chunk)
        for rc, zc, wc in zip(fr[sl], fz[sl], w[sl]):
            out += wc * green(r, z, rc, zc)
    return out


def dgreen(r, z, rc, zc, h=1.5e-3):
    dr = (green(r + h, z, rc, zc) - green(r - h, z, rc, zc)) / (2.0 * h)
    dz = (green(r, z + h, rc, zc) - green(r, z - h, rc, zc)) / (2.0 * h)
    return dr, dz


def solve_coils(fr, fz, w, x_pt):
    """Damped least squares for coil currents: psi = psi_b on the boundary,
    grad psi = 0 at the X-point."""
    th = np.linspace(0.0, 2.0 * np.pi, 181)[:-1]
    keep = np.abs(np.angle(np.exp(1j * (th + np.pi / 2.0)))) > 0.36
    th = th[keep]
    br, bz = miller(np.ones_like(th), th)

    nc = len(COILS)
    a_rows, b_rows = [], []
    psi_pl = psi_from(fr, fz, w, br, bz)
    for k in range(br.size):
        row = np.zeros(nc + 1)
        for j, (rc, zc) in enumerate(COILS):
            row[j] = green(br[k], bz[k], rc, zc) * 1e6
        row[nc] = -1.0
        a_rows.append(row)
        b_rows.append(-psi_pl[k])

    hx = 1.5e-3
    gr_pl = (psi_from(fr, fz, w, x_pt[0] + hx, x_pt[1]) -
             psi_from(fr, fz, w, x_pt[0] - hx, x_pt[1])) / (2.0 * hx)
    gz_pl = (psi_from(fr, fz, w, x_pt[0], x_pt[1] + hx) -
             psi_from(fr, fz, w, x_pt[0], x_pt[1] - hx)) / (2.0 * hx)
    for comp, target in ((0, gr_pl), (1, gz_pl)):
        row = np.zeros(nc + 1)
        for j, (rc, zc) in enumerate(COILS):
            row[j] = dgreen(x_pt[0], x_pt[1], rc, zc)[comp] * 1e6
        a_rows.append(row * 40.0)
        b_rows.append(-float(target) * 40.0)

    for j in range(nc):                      # Tikhonov damping
        row = np.zeros(nc + 1)
        row[j] = 2.5e-2
        a_rows.append(row)
        b_rows.append(0.0)

    sol = np.linalg.lstsq(np.array(a_rows), np.array(b_rows), rcond=None)[0]
    return sol[:nc] * 1e6, float(sol[nc])


# ---------------------------------------------------------------- analysis
def lines_at(cg, level):
    out = cg.lines(level)
    return out[0] if isinstance(out, tuple) else out


def closed_around(cg, level, pt):
    """The closed contour of ``level`` that encircles ``pt``, or None."""
    for ln in lines_at(cg, level):
        if len(ln) < 24:
            continue
        if np.hypot(*(ln[0] - ln[-1])) > 1e-6:
            continue
        if Path(ln).contains_point(pt):
            return ln
    return None


def surface_metrics(ln, grad_r, grad_z):
    r, z = ln[:, 0], ln[:, 1]
    dl = np.hypot(np.diff(r), np.diff(z))
    rm = 0.5 * (r[:-1] + r[1:])
    zm = 0.5 * (z[:-1] + z[1:])
    pts = np.column_stack([zm, rm])
    gmag = np.hypot(grad_r(pts), grad_z(pts))
    gmag = np.maximum(gmag, 1e-9)
    q = (B0 * R0 / (2.0 * np.pi)) * np.sum(dl / (rm * gmag))
    area = 0.5 * np.abs(np.sum(r[:-1] * z[1:] - r[1:] * z[:-1]))
    rbar = np.mean(rm)
    return dict(q=q, area=area, vol=2.0 * np.pi * rbar * area,
                r_out=float(r.max()), r_in=float(r.min()))


def shape_metrics(ln):
    r, z = ln[:, 0], ln[:, 1]
    r_hi, r_lo = float(r.max()), float(r.min())
    z_hi, z_lo = float(z.max()), float(z.min())
    a = 0.5 * (r_hi - r_lo)
    rgeo = 0.5 * (r_hi + r_lo)
    du = (rgeo - float(r[np.argmax(z)])) / a
    dl_ = (rgeo - float(r[np.argmin(z)])) / a
    return dict(a=a, rgeo=rgeo, kappa=(z_hi - z_lo) / (2.0 * a),
                delta=0.5 * (du + dl_), du=du, dl=dl_)


def build():
    fr, fz, w = filaments()
    x_pt = miller(np.array([1.0]), np.array([-np.pi / 2.0]))
    x_pt = (float(x_pt[0][0]), float(x_pt[1][0]))
    ic, _psi_b = solve_coils(fr, fz, w, x_pt)

    rv = np.linspace(R_LO, R_HI, NR)
    zv = np.linspace(Z_LO, Z_HI, NZ)
    rg, zg = np.meshgrid(rv, zv)
    psi = psi_from(fr, fz, w, rg, zg)
    for (rc, zc), cur in zip(COILS, ic):
        psi += cur * green(rg, zg, rc, zc)

    k_ax = np.unravel_index(np.argmax(psi), psi.shape)
    r_ax, z_ax = float(rv[k_ax[1]]), float(zv[k_ax[0]])
    psi_ax = float(psi[k_ax])

    dpz, dpr = np.gradient(psi, zv, rv)
    grad_r = RegularGridInterpolator((zv, rv), dpr, bounds_error=False,
                                     fill_value=None)
    grad_z = RegularGridInterpolator((zv, rv), dpz, bounds_error=False,
                                     fill_value=None)
    psi_i = RegularGridInterpolator((zv, rv), psi, bounds_error=False,
                                    fill_value=None)

    # refine the X-point: Newton steps on grad psi = 0
    def at(fn, r, z):
        return float(np.asarray(fn([[z, r]])).ravel()[0])

    xr, xz = x_pt
    for _ in range(40):
        g = np.array([at(grad_r, xr, xz), at(grad_z, xr, xz)])
        h = 4e-3
        jrr = (at(grad_r, xr + h, xz) - at(grad_r, xr - h, xz)) / (2 * h)
        jrz = (at(grad_r, xr, xz + h) - at(grad_r, xr, xz - h)) / (2 * h)
        jzz = (at(grad_z, xr, xz + h) - at(grad_z, xr, xz - h)) / (2 * h)
        jac = np.array([[jrr, jrz], [jrz, jzz]])
        try:
            step = np.linalg.solve(jac, g)
        except np.linalg.LinAlgError:
            break
        xr -= float(np.clip(step[0], -0.04, 0.04))
        xz -= float(np.clip(step[1], -0.04, 0.04))
    psi_x = at(psi_i, xr, xz)

    cg = contour_generator(x=rv, y=zv, z=psi, line_type=LineType.Separate)
    psi_n = np.linspace(0.03, 0.995, 46)
    rows = []
    for pn in psi_n:
        ln = closed_around(cg, psi_ax - pn * (psi_ax - psi_x), (r_ax, z_ax))
        if ln is None:
            continue
        m = surface_metrics(ln, grad_r, grad_z)
        m["psi_n"] = float(pn)
        rows.append(m)
    lcfs = closed_around(cg, psi_ax - 0.9985 * (psi_ax - psi_x),
                         (r_ax, z_ax))
    shp = shape_metrics(lcfs)

    pn = np.array([m["psi_n"] for m in rows])
    q = np.array([m["q"] for m in rows])
    for _ in range(2):        # damp the ripple left by the filament grid
        q[1:-1] = 0.25 * q[:-2] + 0.5 * q[1:-1] + 0.25 * q[2:]
    vol = np.array([m["vol"] for m in rows])
    rho = ((np.array([m["r_out"] for m in rows]) - r_ax) /
           (shp["rgeo"] + shp["a"] - r_ax))

    # The field was built with unit plasma current, so psi and therefore
    # 1/q scale linearly with Ip; fix Ip by demanding q95 = Q95_TARGET.
    scale = float(np.interp(0.95, pn, q)) / Q95_TARGET
    q = q / scale
    ip_ma = scale / 1e6

    press = P_AXIS * (1.0 - pn ** 1.15) ** 1.6
    dvol = np.diff(np.concatenate([[0.0], vol]))
    p_avg = float(np.sum(press * dvol) / np.sum(dvol))
    beta_t = 2.0 * MU0 * p_avg / B0 ** 2
    beta_n = beta_t * 100.0 * shp["a"] * B0 / ip_ma

    return dict(rv=rv, zv=zv, psi=psi, psi_ax=psi_ax, psi_x=psi_x,
                r_ax=r_ax, z_ax=z_ax, x_pt=(xr, xz), lcfs=lcfs, shp=shp,
                rho=rho, q=q, pn=pn, press=press, ip_ma=ip_ma,
                beta_t=beta_t, beta_n=beta_n, coils=ic, p_avg=p_avg)


# ---------------------------------------------------------------- drawing
def superellipse(rc, zc, a, b, n=3.1, npts=400):
    t = np.linspace(0.0, 2.0 * np.pi, npts)
    ct, st = np.cos(t), np.sin(t)
    r = rc + a * np.sign(ct) * np.abs(ct) ** (2.0 / n)
    z = zc + b * np.sign(st) * np.abs(st) ** (2.0 / n)
    return r, z


def vessel_ring(scale):
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    kap = np.where(np.sin(th) >= 0.0, 1.74, 1.94)
    dlt = np.where(np.sin(th) >= 0.0, 0.36, 0.56)
    a = AMIN * scale
    r = R0 + a * np.cos(th + np.arcsin(dlt) * np.sin(th))
    z = kap * a * np.sin(th)
    return r, z


def main_panel(ax, d):
    rv, zv, psi = d["rv"], d["zv"], d["psi"]
    rg, zg = np.meshgrid(rv, zv)
    pn = (d["psi_ax"] - psi) / (d["psi_ax"] - d["psi_x"])
    floor = np.interp(rg, FLOOR_R, FLOOR_Z)
    wr, wz = vessel_ring(1.18)
    inside = Path(np.column_stack([wr, wz])).contains_points(
        np.column_stack([rg.ravel(), zg.ravel()])).reshape(rg.shape)
    pnm = np.ma.masked_where((zg < floor) | ~inside, pn)
    warm = get_cmap("cb_sequential_warm")

    # machine hardware, drawn first
    for zc in np.linspace(-2.05, 2.05, 8):
        ax.add_patch(Rectangle((0.14, zc - 0.24), 0.32, 0.48,
                               facecolor=lighten(BLUE, 0.30),
                               edgecolor="white", lw=0.6, zorder=2))
    ax.text(0.30, -2.42, "central\nsolenoid", ha="center", va="top",
            fontsize=6.1, color=BLUE, linespacing=1.2)
    tr, tz = superellipse(3.05, 0.0, 2.85, 2.70)
    ax.plot(tr, tz, color=BLUE, lw=2.6, alpha=0.85, zorder=2,
            solid_joinstyle="round")
    tr2, tz2 = superellipse(3.05, 0.0, 2.62, 2.48)
    ax.plot(tr2, tz2, color=lighten(BLUE, 0.45), lw=1.4, zorder=2)
    ax.text(5.95, 0.05, "toroidal-field coil\n(B$_0$ = {:.1f} T)".format(B0),
            ha="right", va="bottom", fontsize=6.2, color=BLUE,
            linespacing=1.25)

    ro, zo = vessel_ring(1.27)
    ri, zi = vessel_ring(1.18)
    ax.fill(ro, zo, facecolor=lighten(GREY, 0.42), edgecolor=GREY, lw=0.8,
            zorder=3)
    ax.fill(ri, zi, facecolor="white", edgecolor=GREY, lw=0.7, zorder=3)
    ax.annotate("vacuum vessel\nand first wall", xy=(4.33, 0.95),
                xytext=(4.62, 1.72), ha="left", va="center", fontsize=6.1,
                color=SUB, linespacing=1.25, zorder=10,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))

    cur = d["coils"]
    cmax = np.max(np.abs(cur))
    for j, (rc, zc) in enumerate(COILS):
        if rc < 0.5:
            continue
        s = 0.055 + 0.085 * abs(cur[j]) / cmax
        ax.add_patch(Rectangle((rc - s, zc - s * 0.82), 2 * s, 1.64 * s,
                               facecolor=BLUE if cur[j] > 0 else "white",
                               edgecolor=BLUE, lw=0.9, zorder=6))
    ax.annotate("divertor coil", xy=(COILS[DIV_COIL][0] + 0.12,
                                     COILS[DIV_COIL][1]),
                xytext=(3.05, -2.80), ha="left", va="bottom", fontsize=6.1,
                color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE,
                                            lw=0.7))

    # flux surfaces
    lev = np.linspace(0.0, 1.0, 13)
    ax.contourf(rg, zg, np.ma.masked_where(pnm > 1.0, pnm), levels=lev,
                cmap=warm.reversed(), alpha=0.32, zorder=4)
    closed = np.arange(0.10, 0.96, 0.10)
    ax.contour(rg, zg, pnm, levels=closed,
               colors=[warm(0.34 + 0.52 * (1.0 - v)) for v in closed],
               linewidths=0.85, zorder=5)
    sol = np.array([1.006, 1.020, 1.045, 1.080, 1.130])
    ax.contour(rg, zg, pnm, levels=sol, colors=[PURPLE] * sol.size,
               linewidths=0.65, linestyles=[(0, (3.4, 1.9))] * sol.size,
               zorder=5)
    ax.contour(rg, zg, pnm, levels=[1.0], colors=[RED], linewidths=2.0,
               zorder=7)

    # divertor targets
    ax.plot([1.98, 2.44], [-1.99, -2.25], color="#4d4d4d", lw=3.4,
            solid_capstyle="butt", zorder=8)
    ax.plot([2.70, 3.22], [-2.25, -2.02], color="#4d4d4d", lw=3.4,
            solid_capstyle="butt", zorder=8)
    ax.plot([2.44, 2.70], [-2.25, -2.25], color=STEEL, lw=2.6,
            solid_capstyle="butt", zorder=8)
    ax.annotate("inner target", xy=(2.10, -2.07), xytext=(1.16, -2.62),
                ha="center", va="top", fontsize=6.0, color="#4d4d4d",
                zorder=10, arrowprops=dict(arrowstyle="-", color="#4d4d4d",
                                           lw=0.7))
    ax.annotate("outer target", xy=(3.10, -2.11), xytext=(4.72, -2.16),
                ha="right", va="top", fontsize=6.0, color="#4d4d4d",
                zorder=10, arrowprops=dict(arrowstyle="-", color="#4d4d4d",
                                           lw=0.7))

    # annotations
    r_ax, z_ax = d["r_ax"], d["z_ax"]
    ax.plot([r_ax], [z_ax], marker="+", ms=8.0, mew=1.6, color=INK, zorder=9)
    ax.annotate("magnetic axis\nR = {:.2f} m,  Z = {:+.2f} m".format(
        r_ax, z_ax), xy=(r_ax, z_ax), xytext=(4.30, 0.72), ha="left",
        va="bottom", fontsize=6.3, color=INK, linespacing=1.25, zorder=10,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    xr, xz = d["x_pt"]
    ax.plot([xr], [xz], marker="X", ms=6.2, mec="white", mew=0.8, color=RED,
            zorder=9)
    ax.annotate("X-point\n({:.2f}, {:+.2f}) m".format(xr, xz), xy=(xr, xz),
                xytext=(0.62, -1.62), ha="left", va="center", fontsize=6.3,
                color=RED, linespacing=1.25, zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.annotate("LCFS (separatrix)", xy=(4.09, -0.15), xytext=(4.56, -0.72),
                ha="left", va="center", fontsize=6.3, color=RED, zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.annotate("scrape-off layer,\nopen lines to the targets",
                xy=(4.20, -1.05), xytext=(4.42, -1.72), ha="left",
                va="top", fontsize=6.1, color=PURPLE, linespacing=1.25,
                zorder=10, arrowprops=dict(arrowstyle="-", color=PURPLE,
                                           lw=0.7))

    s = d["shp"]
    ax.text(0.10, 2.36, ("R$_0$ = {:.2f} m,  a = {:.2f} m\n"
                         "$\\kappa$ = {:.2f},  $\\delta$ = {:.2f}\n"
                         "I$_p$ = {:.1f} MA,  q$_{{95}}$ = {:.2f}").format(
        s["rgeo"], s["a"], s["kappa"], s["delta"], d["ip_ma"], Q95_TARGET),
        ha="left", va="top", fontsize=6.4, color=INK, linespacing=1.42,
        zorder=11, bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                             edgecolor=HAIR, lw=0.6))

    ax.axvline(0.0, color=HAIR, lw=0.8, ls=(0, (5.0, 3.0)), zorder=1)
    ax.text(-0.02, -2.88, "machine axis", ha="left", va="bottom",
            fontsize=6, color=MUTED, rotation=90.0)

    ax.set_aspect("equal")
    ax.set_xlim(*VIEW_R)
    ax.set_ylim(*VIEW_Z)
    ax.set_xlabel("major radius R (m)")
    ax.set_ylabel("height Z (m)")
    ax.tick_params(labelsize=6.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Poloidal flux surfaces of a lower-single-null equilibrium",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)
    handles = [Line2D([], [], color=warm(0.72), lw=0.9,
                      label="closed flux surface"),
               Line2D([], [], color=RED, lw=2.0, label="LCFS / separatrix"),
               Line2D([], [], color=PURPLE, lw=0.7, ls=(0, (3.4, 1.9)),
                      label="SOL, open field line"),
               Line2D([], [], color=BLUE, lw=2.2, label="magnet system")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=False,
              bbox_to_anchor=(0.004, 0.545), handlelength=1.7,
              labelspacing=0.30, borderaxespad=0.0)


def q_panel(ax, d):
    rho, q = d["rho"], d["q"]
    ax.plot(rho, q, color=BLUE, lw=1.7, zorder=5)
    ax.fill_between(rho, 0.0, q, color=lighten(BLUE, 0.86), lw=0.0, zorder=2)

    r95 = float(np.interp(0.95, d["pn"], rho))
    q95 = float(np.interp(0.95, d["pn"], q))
    ax.plot([r95], [q95], marker="o", ms=4.6, mfc="white", mec=BLUE, mew=1.3,
            zorder=7)
    ax.annotate("q$_{{95}}$ = {:.2f}".format(q95), xy=(r95, q95),
                xytext=(r95 - 0.06, q95 + 0.9), ha="right", va="bottom",
                fontsize=6.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

    for qr, col in ((1.0, RED), (2.0, ORANGE)):
        ax.axhline(qr, color=col, lw=0.8, ls=(0, (3.2, 2.0)), zorder=3)
        if q.min() < qr < q.max():
            rr = float(np.interp(qr, q, rho))
            ax.plot([rr], [qr], marker="s", ms=3.6, color=col, zorder=7)
            ax.text(rr, qr - 0.30, "q = {:.0f}\n$\\rho$ = {:.2f}".format(
                qr, rr), ha="center", va="top", fontsize=6.0, color=col,
                linespacing=1.2)
        else:
            ax.text(0.02, qr + 0.12, "q = {:.0f} not present".format(qr),
                    ha="left", va="bottom", fontsize=6.0, color=col)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, min(9.5, float(q.max()) * 1.12))
    ax.set_xlabel("normalised minor radius $\\rho$ (-)", fontsize=7.4)
    ax.set_ylabel("safety factor q (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    ax.set_title("q from the contour integral, not a formula", fontsize=8.2,
                 fontweight="bold", color=INK)


def p_panel(ax, d):
    rho, p = d["rho"], d["press"] / 1e3
    ax.plot(rho, p, color=RED, lw=1.7, zorder=5)
    ax.fill_between(rho, 0.0, p, color=lighten(RED, 0.86), lw=0.0, zorder=2)
    ax.axhline(d["p_avg"] / 1e3, color=SUB, lw=0.8, ls=(0, (4.0, 2.2)),
               zorder=4)
    ax.text(0.03, d["p_avg"] / 1e3 + 10.0,
            "volume average $\\langle p \\rangle$ = {:.0f} kPa".format(
                d["p_avg"] / 1e3), ha="left", va="bottom", fontsize=6.1,
            color=SUB)
    ax.text(0.975, p.max() * 1.06,
            ("$\\beta_t$ = 2$\\mu_0\\langle p\\rangle$/B$_0^2$ = "
             "{:.2f} %\n$\\beta_N$ = {:.2f} % m T MA$^{{-1}}$").format(
                 d["beta_t"] * 100.0, d["beta_n"]), ha="right", va="top",
            fontsize=6.2, color=INK, linespacing=1.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, p.max() * 1.16)
    ax.set_xlabel("normalised minor radius $\\rho$ (-)", fontsize=7.4)
    ax.set_ylabel("plasma pressure p (kPa)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    ax.set_title("Pressure profile and the beta it carries", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    np.random.default_rng(918)
    d = build()

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN)
    ax_q = axes_inches(fig, AUX1)
    ax_p = axes_inches(fig, AUX2)
    main_panel(ax_m, d)
    q_panel(ax_q, d)
    p_panel(ax_p, d)

    s = d["shp"]
    panel_labels_inches(fig, [ax_m, ax_q, ax_p], dx=-0.56, dy=0.13)
    savefig(fig, "918_fus_tokamak_equilibrium.png")


if __name__ == "__main__":
    main()
