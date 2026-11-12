"""A finned wall solved in two dimensions, and the efficiency that follows.

The main panel is a finite-volume solution of steady conduction through a
finned steel wall: the bottom face is held at the wall temperature, every
exposed face loses heat by convection to the same air, and nothing else is
prescribed.  Fin efficiency is then measured three independent ways from that
one solution - by summing the convective faces of the fins, by integrating the
temperature excess along a fin centreline, and by the textbook closed form for a
straight rectangular fin - and the script refuses to draw unless the three
agree.  The upper side panel puts the measured point on the classical efficiency
curves for three fin profiles; the lower one is the centreline itself, with the
shaded area that the efficiency literally is: the temperature excess the fin
really carries, divided by the excess an infinitely conductive fin would carry.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import iv
from matplotlib.colors import LinearSegmentedColormap

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
DEEP = "#134a76"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
EMBER = "#c0603f"
GREY = "#8a8a8a"

DEG = r"$^\circ$C"

SEED = 1041
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

DX = 2.0e-4                # grid pitch, m
W_DOM, H_DOM = 0.0288, 0.0300      # domain width and height, m
H_BASE = 0.0040            # thickness of the plain wall, m
T_FIN = 0.0016             # fin thickness, m
H_FIN = 0.0240             # fin height above the wall, m
N_FIN = 4                  # fins across the domain
K_SOLID = 45.0             # steel conductivity, W/(m K)
H_CONV = 90.0              # convective coefficient, W/(m2 K)
T_BASE = 85.0              # wall temperature, degC
T_INF = 25.0               # air temperature, degC


def geometry():
    """Solid mask on the finite-volume grid, plus the fin column indices."""
    nx = int(round(W_DOM / DX)) + 1
    ny = int(round(H_DOM / DX)) + 1
    x = np.arange(nx) * DX
    y = np.arange(ny) * DX
    j_base = int(round(H_BASE / DX))
    j_tip = j_base + int(round(H_FIN / DX))
    half = int(round(0.5 * T_FIN / DX))
    pitch = W_DOM / N_FIN
    centres = [int(round(((k + 0.5) * pitch) / DX)) for k in range(N_FIN)]
    solid = np.zeros((ny, nx), dtype=bool)
    solid[:j_base + 1, :] = True
    for c in centres:
        solid[j_base:j_tip + 1, c - half:c + half + 1] = True
    return x, y, solid, j_base, j_tip, centres, half


def solve_field(solid, j_base):
    """Steady conduction with Robin faces, assembled as a finite-volume system."""
    ny, nx = solid.shape
    n = int(solid.sum())
    ids = -np.ones((ny, nx), dtype=int)
    ids[solid] = np.arange(n)

    kindp = np.zeros((ny + 2, nx + 2), dtype=int)      # 0 adiabatic, 1 solid, 2 air
    kindp[1:-1, 1:-1] = np.where(solid, 1, 2)
    kindp[-1, 1:-1] = 2                                # air above the tips
    idp = -np.ones((ny + 2, nx + 2), dtype=int)
    idp[1:-1, 1:-1] = ids

    free = solid.copy()
    free[0, :] = False                                 # the held wall face
    idx = ids[free]
    diag = np.zeros(n)
    b = np.zeros(n)
    rows, cols, vals = [], [], []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_id = idp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx][free]
        nb_kind = kindp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx][free]
        m_s = nb_kind == 1
        rows.append(idx[m_s])
        cols.append(nb_id[m_s])
        vals.append(np.full(int(m_s.sum()), K_SOLID))
        diag[idx[m_s]] -= K_SOLID
        m_f = nb_kind == 2
        diag[idx[m_f]] -= H_CONV * DX
        b[idx[m_f]] -= H_CONV * DX * T_INF
    held = ids[0, :]
    diag[held] = 1.0
    b[held] = T_BASE

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    a = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    t = spsolve(a, b)
    field = np.full((ny, nx), np.nan)
    field[solid] = t
    return field


def convective_load(field, solid, row_lo=0):
    """Heat leaving every exposed face at or above ``row_lo``, W per m of depth."""
    ny, nx = solid.shape
    kindp = np.zeros((ny + 2, nx + 2), dtype=int)
    kindp[1:-1, 1:-1] = np.where(solid, 1, 2)
    kindp[-1, 1:-1] = 2
    take = solid.copy()
    take[:row_lo, :] = False
    total = 0.0
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = kindp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
        m = take & (nb == 2)
        total += H_CONV * DX * float(np.sum(field[m] - T_INF))
    return total


def eta_rect(mlc):
    mlc = np.asarray(mlc, dtype=float)
    return np.tanh(mlc) / np.where(mlc > 1e-9, mlc, 1e-9)


def eta_triangular(ml):
    ml = np.asarray(ml, dtype=float)
    z = np.where(ml > 1e-9, ml, 1e-9)
    return iv(1, 2.0 * z) / (z * iv(0, 2.0 * z))


def eta_parabolic(ml):
    ml = np.asarray(ml, dtype=float)
    return 2.0 / (1.0 + np.sqrt((2.0 * ml) ** 2 + 1.0))


def field_panel(fig, ax, x, y, field, centres, half, j_base, j_tip):
    img = np.ma.masked_invalid(field)
    # The dark end of magma turns a cool fin into a black bar; keep the ramp
    # inside the readable middle so the whole 30 K span stays legible.
    cmap = LinearSegmentedColormap.from_list(
        "magma_mid", plt.get_cmap("magma")(np.linspace(0.22, 0.97, 256)))
    cmap.set_bad("#eceff2")
    im = ax.imshow(img, origin="lower", cmap=cmap, interpolation="nearest",
                   extent=[0.0, 1e3 * W_DOM, 0.0, 1e3 * H_DOM],
                   vmin=float(np.nanmin(field)), vmax=T_BASE)
    ax.set_aspect("equal")
    levels = np.arange(60.0, T_BASE, 4.0)
    ax.contour(1e3 * x, 1e3 * y, field, levels=levels, colors="white",
               linewidths=0.45, alpha=0.62, zorder=4)

    c = centres[1]
    ax.plot([1e3 * x[c]] * 2, [1e3 * y[j_base], 1e3 * y[j_tip]], color="white",
            lw=0.9, ls=(0, (3.0, 2.0)), zorder=6)
    halo_text(ax, 1e3 * x[c] + 0.55, 1e3 * y[j_tip] - 3.0,
              "centreline\nread out in (c)", fontsize=6.2, ha="left",
              va="top", linespacing=1.28, zorder=8)

    tip = float(field[j_tip, centres[1]])
    halo_text(ax, 1e3 * W_DOM * 0.5, 0.9, "wall face held at %.0f " % T_BASE
              + DEG, fontsize=6.6, ha="center", va="bottom",
              fontweight="bold", zorder=8)
    ax.text(1e3 * W_DOM * 0.5, 1e3 * H_DOM - 0.55,
            "air at %.0f " % T_INF + DEG + ", h = %.0f W/(m$^2\\cdot$K) on "
            "every exposed face" % H_CONV, fontsize=6.5, color=INK,
            ha="center", va="top", zorder=8)

    ax.set_xlabel("width across the finned wall (mm)")
    ax.set_ylabel("height above the wall face (mm)")
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
        s.set_visible(True)
    ax.set_title("%d steel fins, %.1f mm thick and %.0f mm tall, on a %.0f mm "
                 "wall" % (N_FIN, 1e3 * T_FIN, 1e3 * H_FIN, 1e3 * H_BASE),
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    safe_colorbar(fig, im, ax, label="temperature (" + DEG + ")", width=0.11,
                  pad=0.09, labelsize=6.6)
    return tip


def efficiency_panel(ax, mlc_case, eta_flux):
    ml = np.linspace(0.0, 3.0, 320)
    ax.plot(ml, eta_rect(ml), color=DEEP, lw=1.7, zorder=6)
    ax.plot(ml, eta_triangular(ml), color=GREEN, lw=1.5, zorder=6)
    ax.plot(ml, eta_parabolic(ml), color=GOLD, lw=1.5, zorder=6)
    ax.plot([mlc_case], [eta_flux], marker="o", ms=5.2, mfc="white", mec=RED,
            mew=1.5, zorder=9)
    ax.plot([mlc_case, mlc_case], [0.0, eta_flux], color=RED, lw=0.8,
            ls=(0, (2.2, 1.8)), zorder=5)
    ax.annotate("this wall: %.2f from the\ntwo-dimensional solve"
                % eta_flux, xy=(mlc_case, eta_flux),
                xytext=(1.42, 0.075), fontsize=6.2, color=RED,
                ha="left", va="bottom", linespacing=1.30, zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=3,
                                shrinkB=4))
    for name, col, y in (("rectangular", DEEP, 0.335),
                         ("triangular", GREEN, 0.245),
                         ("parabolic", GOLD, 0.155)):
        ax.text(0.10, y, name, fontsize=6.2, color=col, ha="left",
                va="center", fontweight="bold", zorder=8)
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("fin parameter " + r"$L_c\sqrt{2h/(kt)}$" +
                  " (dimensionless)", fontsize=6.8)
    ax.set_ylabel("fin efficiency (dimensionless)", fontsize=6.5)
    ax.grid(lw=0.5, alpha=0.40)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Where this fin sits on the classical curves", fontsize=8.0,
                 fontweight="bold", color=INK)


def profile_panel(ax, z_mm, t_2d, t_1d, eta_area, tip):
    ax.fill_between(z_mm, T_INF, t_2d, color="#f6e0d6", edgecolor="none",
                    zorder=2)
    ax.plot(z_mm, t_2d, color=RED, lw=1.8, zorder=7)
    ax.plot(z_mm, t_1d, color=DEEP, lw=1.2, ls=(0, (4.0, 2.0)), zorder=6)
    ax.axhline(T_BASE, color=GREY, lw=0.9, ls=(0, (2.4, 1.8)), zorder=5)
    ax.axhline(T_INF, color=BLUE, lw=0.9, zorder=5)
    ax.text(1e3 * H_FIN - 0.30, T_BASE - 1.4,
            "an infinitely conductive fin would sit here", fontsize=6.0,
            color=SUB, ha="right", va="top", zorder=8)
    ax.text(1e3 * H_FIN - 0.30, 78.4, "one-dimensional fin equation",
            fontsize=6.0, color=DEEP, ha="right", va="top", zorder=8)
    ax.text(1e3 * H_FIN - 0.30, 73.0, "two-dimensional solve", fontsize=6.0,
            color=RED, ha="right", va="top", zorder=8)
    ax.text(1e3 * H_FIN - 0.25, T_INF + 1.2, "air, %.0f " % T_INF + DEG,
            fontsize=6.0, color=BLUE, ha="right", va="bottom", zorder=8)
    ax.text(0.35, 0.5 * (T_INF + T_BASE), "shaded area / full rectangle\n"
            "= %.3f, the fin efficiency" % eta_area, fontsize=6.2, color=RED,
            ha="left", va="center", linespacing=1.30, fontweight="bold",
            zorder=8)
    ax.plot([1e3 * H_FIN], [tip], marker="o", ms=4.0, mfc="white", mec=RED,
            mew=1.2, zorder=9)
    ax.text(1e3 * H_FIN - 0.30, tip - 2.4, "tip %.1f " % tip + DEG,
            fontsize=6.1, color=RED, ha="right", va="top", zorder=9)
    ax.set_xlim(0.0, 1e3 * H_FIN)
    ax.set_ylim(T_INF - 2.0, T_BASE + 4.0)
    ax.set_xlabel("distance from the fin root (mm)", fontsize=6.8)
    ax.set_ylabel("temperature (" + DEG + ")", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.36)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Efficiency is an area ratio on this panel", fontsize=8.0,
                 fontweight="bold", color=INK)


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
    np.random.default_rng(SEED)

    x, y, solid, j_base, j_tip, centres, half = geometry()
    field = solve_field(solid, j_base)

    t_fin = (2 * half + 1) * DX
    l_fin = (j_tip - j_base) * DX
    l_c = l_fin + 0.5 * t_fin
    m_fin = np.sqrt(2.0 * H_CONV / (K_SOLID * t_fin))
    theta_b = T_BASE - T_INF

    q_total = convective_load(field, solid, row_lo=0)
    q_fins = convective_load(field, solid, row_lo=j_base + 1)
    q_in = K_SOLID * float(np.sum(T_BASE - field[1, :]))
    q_ideal_fins = H_CONV * N_FIN * (2.0 * l_fin + t_fin) * theta_b
    q_bare = H_CONV * W_DOM * theta_b

    eta_flux = q_fins / q_ideal_fins
    eta_1d = float(eta_rect(m_fin * l_c))

    z = y[j_base:j_tip + 1] - y[j_base]
    t_2d = field[j_base:j_tip + 1, centres[1]]
    t_1d = T_INF + theta_b * np.cosh(m_fin * (l_c - z)) / np.cosh(m_fin * l_c)
    eta_area = float(np.trapezoid(t_2d - T_INF, z) / (l_fin * theta_b))

    assert abs(q_in - q_total) < 1e-6 * abs(q_in), \
        "what enters through the held face has to leave through the air faces"
    assert abs(eta_flux - eta_1d) < 0.06, \
        "the solved efficiency must track the closed form"
    assert abs(eta_area - eta_flux) < 0.06, \
        "integrating the centreline must reproduce the flux ratio"
    assert q_total > 3.0 * q_bare, "these fins should be worth having"
    assert np.all(np.diff(t_2d) < 0.0), "a fin has to cool toward its tip"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_eta = axes_inches(fig, AUX_T)
    ax_pro = axes_inches(fig, AUX_B)

    tip = field_panel(fig, ax_main, x, y, field, centres, half, j_base, j_tip)
    efficiency_panel(ax_eta, m_fin * l_c, eta_flux)
    profile_panel(ax_pro, 1e3 * z, t_2d, t_1d, eta_area, tip)



    panel_labels_inches(fig, [ax_main, ax_eta, ax_pro], dx=-0.62, dy=0.14)
    savefig(fig, "1041_ht_fin_efficiency.png")


if __name__ == "__main__":
    main()
