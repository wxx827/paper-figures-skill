"""Figure 1157 - View-dependent radiance and what degree-2 SH can hold.

One BRDF drives the whole figure.  At the marked surface point P (normal N0,
light L) the outgoing radiance is

    L(v) = ambient*a + kd*a*(N0.L)_+ + ks*(N0.H)_+^shine * spec,  H = |L+v|

(a) renders the object from six view directions with that same shading, so the
    highlight really does travel across the six tiles.
(b) projects the luminance of L(v) onto the nine real spherical harmonics of
    degree <= 2 by numerical quadrature over the sphere.
(c) is L(v) itself over the whole hemisphere of view directions at P, with the
    six view directions of (a) marked where they actually sit.
(d) renders the object twice with the specular term off and on, plus their
    difference.

Every printed number - the degrees off normal, the specular percentages, the
SH coefficients, the reconstruction error - is measured from these arrays.
This is the one warm figure of the batch: a specular highlight has to be warm.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
HAIR, CARD = "#d5d5d5", "#f5f5f5"
BLUE, DEEP = "#5f9dc6", "#134a76"
SEED = 1562  # original id; figure number is 1157

ALBEDO = np.array([0.255, 0.415, 0.545])       # cool steel-blue body
SPEC = np.array([1.000, 0.905, 0.735])         # warm highlight
AMBIENT, KD, KS, SHINE = 0.320, 0.80, 0.92, 44.0
BG = np.array([0.930, 0.940, 0.950])


def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


N0 = unit([0.34, 0.44, 0.83])                  # normal at the probed point P
LDIR = unit([-0.42, 0.70, 0.58])               # direction to the light
E1 = unit(np.cross(N0, [0.0, 0.0, 1.0]))
E2 = np.cross(N0, E1)
MIRROR = unit(2.0 * np.dot(N0, LDIR) * N0 - LDIR)


def frame_dir(alpha, phi):
    return unit(np.sin(alpha) * (np.cos(phi) * E1 + np.sin(phi) * E2)
                + np.cos(alpha) * N0)


A_MIR = float(np.arccos(np.clip(np.dot(MIRROR, N0), -1, 1)))
P_MIR = float(np.arctan2(np.dot(MIRROR, E2), np.dot(MIRROR, E1)))
VIEWS = [frame_dir(0.16, P_MIR),
         frame_dir(A_MIR, P_MIR),
         frame_dir(A_MIR + 0.42, P_MIR),
         frame_dir(0.78, P_MIR + 2.30),
         frame_dir(1.02, P_MIR + 3.70),
         frame_dir(1.24, P_MIR + 5.10)]


def shade(normals, view, spec_on=True):
    """The one BRDF; ``normals`` is (..., 3), ``view`` a single direction."""
    h = unit(LDIR + view)
    ndl = np.clip(normals @ LDIR, 0.0, None)
    out = AMBIENT * ALBEDO + KD * ALBEDO * ndl[..., None]
    if spec_on:
        ndh = np.clip(normals @ h, 0.0, None)
        out = out + KS * (ndh ** SHINE)[..., None] * SPEC
    return np.clip(out, 0.0, 1.0)


def radiance(view, spec_on=True):
    return shade(N0[None, :], view, spec_on)[0]


def spec_fraction(view):
    full = float(radiance(view).sum())
    diff = float(radiance(view, spec_on=False).sum())
    return 100.0 * (full - diff) / max(full, 1e-9)


def basis(view):
    t1 = unit(np.cross([0.0, 0.0, 1.0], view)) if abs(view[2]) < 0.995 \
        else unit(np.cross([0.0, 1.0, 0.0], view))
    return t1, np.cross(view, t1)


def render(view, n=170, spec_on=True, mark=True):
    """Orthographic render of the unit sphere seen along ``view``."""
    t1, t2 = basis(view)
    a, b = np.meshgrid(np.linspace(-1.34, 1.34, n),
                       np.linspace(-1.34, 1.34, n))
    rr = a ** 2 + b ** 2
    inside = rr <= 1.0
    zz = np.sqrt(np.clip(1.0 - rr, 0.0, None))
    nrm = (a[..., None] * t1 + b[..., None] * t2 + zz[..., None] * view)

    img = np.ones((n, n, 3)) * BG
    img *= (1.0 - 0.10 * np.clip((b + 1.34) / 2.68, 0, 1))[..., None]
    shadow = np.exp(-(((a - 0.10) / 1.05) ** 2 + ((b + 1.18) / 0.16) ** 2))
    img *= (1.0 - 0.26 * shadow)[..., None]
    img[inside] = shade(nrm[inside], view, spec_on)

    px = py = None
    if mark and float(np.dot(N0, view)) > 0.06:
        px, py = float(N0 @ t1), float(N0 @ t2)
    return np.clip(img, 0, 1), (px, py)


def lobe(n=190):
    """L(v) over the hemisphere of view directions at P (orthographic)."""
    a, b = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n))
    rr = a ** 2 + b ** 2
    inside = rr <= 1.0
    zz = np.sqrt(np.clip(1.0 - rr, 0.0, None))
    v = (a[..., None] * E1 + b[..., None] * E2 + zz[..., None] * N0)
    img = np.ones((n, n, 4))
    img[..., :3] = 1.0
    img[..., 3] = 0.0
    flat = v[inside]
    flat = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
    h = flat + LDIR[None, :]
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    ndl = max(float(np.dot(N0, LDIR)), 0.0)
    col = AMBIENT * ALBEDO + KD * ALBEDO * ndl
    col = np.tile(col, (flat.shape[0], 1))
    col = col + KS * (np.clip(h @ N0, 0, None) ** SHINE)[..., None] * SPEC
    img[inside, :3] = np.clip(col, 0, 1)
    img[inside, 3] = 1.0
    return img


SH_NAMES = [r"$Y_0^0$", r"$Y_1^{-1}$", r"$Y_1^0$", r"$Y_1^1$",
            r"$Y_2^{-2}$", r"$Y_2^{-1}$", r"$Y_2^0$", r"$Y_2^1$", r"$Y_2^2$"]
SH_DEG = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])


def sh_basis(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([
        np.full_like(x, 0.282095),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z, 0.546274 * (x * x - y * y)], axis=-1)


def sh_fit(nt=180, npz=360):
    th = (np.arange(nt) + 0.5) * np.pi / nt
    ph = (np.arange(npz) + 0.5) * 2.0 * np.pi / npz
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)],
                 axis=-1)
    vis = (v @ N0) > 0.0
    h = v + LDIR[None, None, :]
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    ndl = max(float(np.dot(N0, LDIR)), 0.0)
    rgb = (AMBIENT * ALBEDO + KD * ALBEDO * ndl)[None, None, :] * \
        np.ones(v.shape[:2] + (3,))
    rgb = rgb + KS * (np.clip(h @ N0, 0, None) ** SHINE)[..., None] * SPEC
    lum = np.clip(rgb, 0, 1) @ np.array([0.2126, 0.7152, 0.0722])
    lum = np.where(vis, lum, 0.0)

    Y = sh_basis(v)
    dw = (np.sin(T) * (np.pi / nt) * (2.0 * np.pi / npz))
    coef = np.einsum("ij,ijk->k", lum * dw, Y)
    fit = Y @ coef
    err = np.abs(fit - lum)[vis]
    return coef, float(np.sqrt(np.mean(err ** 2))), float(err.max()), \
        float(lum[vis].max())


def main():
    set_pub_style(base_font=8.5)
    coef, rms, emax, lmax = sh_fit()
    frac = [spec_fraction(v) for v in VIEWS]
    ang = [np.degrees(np.arccos(np.clip(np.dot(v, N0), -1, 1))) for v in VIEWS]
    rad = [radiance(v) for v in VIEWS]

    full, _ = render(VIEWS[1], n=210, spec_on=True, mark=False)
    diff_only, _ = render(VIEWS[1], n=210, spec_on=False, mark=False)
    delta = np.abs(full - diff_only) @ np.array([0.2126, 0.7152, 0.0722])

    fig = plt.figure(figsize=(10.4, 6.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.34, 1.0], left=0.040,
                             right=0.982, top=0.836, bottom=0.072,
                             hspace=0.40)
    top = outer[0].subgridspec(2, 5,
                               width_ratios=[0.80, 0.80, 1.52, 0.80, 1.30],
                               wspace=0.17, hspace=0.44)
    bot = outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.22],
                               wspace=0.13)
    slots = [top[0, 0], top[1, 0], top[0, 1], top[1, 1], top[0, 3], top[1, 3]]

    # ---- (a) six views -----------------------------------------------------
    for i, (sl, v) in enumerate(zip(slots, VIEWS)):
        ax = fig.add_subplot(sl)
        img, (px, py) = render(v, n=150)
        ax.imshow(img, extent=(-1.34, 1.34, -1.34, 1.34), origin="lower",
                  interpolation="bilinear", zorder=1)
        if px is not None:
            ax.add_patch(Circle((px, py), 0.145, facecolor="none",
                                edgecolor="white", lw=1.6, zorder=4))
            ax.add_patch(Circle((px, py), 0.145, facecolor="none",
                                edgecolor="#0b3d63", lw=0.8, zorder=5))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#9aa7b2")
            s.set_linewidth(0.7)
        ax.set_xlabel(f"$\\mathbf{{d}}_{i + 1}$   {ang[i]:.0f}$^\\circ$ off "
                      f"$\\mathbf{{n}}$\nspecular at P {frac[i]:.0f} %",
                      fontsize=6.8, color=SUB, labelpad=2.5, linespacing=1.5)
        ax.add_patch(mpl.patches.Rectangle((-1.30, 1.06), 0.34, 0.20,
                                           facecolor=rad[i],
                                           edgecolor="#3a3a3a", lw=0.6,
                                           zorder=6))
        ax.text(-0.90, 1.16, "$L(\\mathbf{d})$", ha="left", va="center",
                fontsize=6.4, color=INK, zorder=6)
        if i == 0:
            add_panel_label(ax, "(a)", x=-0.115, y=1.03)

    # ---- (b) spherical harmonic coefficients -------------------------------
    axs = fig.add_subplot(top[:, 2])
    shades = ["#9cc3dd", "#5f9dc6", "#134a76"]
    xs = np.arange(9)
    axs.bar(xs, coef, width=0.68, color=[shades[d] for d in SH_DEG],
            edgecolor="white", linewidth=0.5, zorder=3)
    axs.axhline(0.0, color="#3a3a3a", lw=0.9, zorder=4)
    lo, hi = float(coef.min()), float(coef.max())
    pad = 0.68 * (hi - lo)
    for g, (i0, i1) in enumerate([(0, 0), (1, 3), (4, 8)]):
        yb = hi + 0.13 * (hi - lo)
        axs.plot([i0 - 0.34, i0 - 0.34, i1 + 0.34, i1 + 0.34],
                 [yb - 0.035 * (hi - lo), yb, yb, yb - 0.035 * (hi - lo)],
                 color="#8a8a8a", lw=0.8, zorder=5)
        axs.text((i0 + i1) / 2.0, yb + 0.02 * (hi - lo),
                 f"$\\ell$ = {g}", ha="center", va="bottom", fontsize=7.0,
                 color=SUB, zorder=5)
    axs.set_xticks(xs)
    axs.set_xticklabels(SH_NAMES, fontsize=6.8)
    axs.set_xlim(-0.75, 8.75)
    axs.set_ylim(lo - 0.16 * (hi - lo), hi + pad)
    axs.set_ylabel("coefficient of the luminance of $L(\\mathbf{v})$",
                   fontsize=7.6)
    axs.tick_params(labelsize=7.0, length=2.4)
    axs.grid(True, axis="y", lw=0.45, color="#ececec")
    axs.set_axisbelow(True)
    axs.set_title("Nine coefficients, degree $\\leq$ 2", fontsize=8.6,
                  fontweight="bold", pad=4.0)
    axs.text(0.975, 0.630, f"RMS error {100 * rms / lmax:.1f} % of peak\n"
             f"worst error {100 * emax / lmax:.0f} % at the highlight",
             transform=axs.transAxes, ha="right", va="top", fontsize=6.8,
             color=SUB, linespacing=1.55, zorder=6,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.94))
    add_panel_label(axs, "(b)", x=-0.185, y=1.015)

    # ---- (c) the radiance lobe over the view hemisphere --------------------
    axl = fig.add_subplot(top[:, 4])
    axl.imshow(lobe(), extent=(-1, 1, -1, 1), origin="lower",
               interpolation="bilinear", zorder=2)
    axl.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor="#9aa7b2",
                         lw=0.8, zorder=4))
    for r_ in (1 / 3, 2 / 3):
        axl.add_patch(Circle((0, 0), r_, facecolor="none", edgecolor="white",
                             lw=0.5, alpha=0.55, zorder=3))
    spots = np.array([[float(v @ E1), float(v @ E2)] for v in VIEWS])
    for i, (px, py) in enumerate(spots):
        axl.plot(px, py, marker="o", ms=4.2, mfc="white", mec="#0b3d63",
                 mew=1.0, zorder=6)
        rr = float(np.hypot(px, py))
        if rr < 0.06:
            ox, oy = px, py + 0.14
        else:
            ux, uy = px / rr, py / rr
            # two views that share an azimuth project onto the same ray, so a
            # purely radial badge lands on the outer one's marker and hides it
            ox, oy = px + 0.20 * ux, py + 0.20 * uy
            others = np.delete(spots, i, axis=0)
            if np.hypot(*(others - [ox, oy]).T).min() < 0.15:
                ox, oy = px - 0.19 * uy, py + 0.19 * ux
        axl.text(ox, oy, f"{i + 1}", ha="center", va="center", fontsize=6.8,
                 fontweight="bold", color="#0b3d63", zorder=7,
                 bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                           ec="#b9c9d8", lw=0.5, alpha=0.95))
    axl.text(0.0, -1.28, "centre = looking straight down $\\mathbf{n}$,\n"
             "rim = grazing", ha="center", va="top", fontsize=6.8, color=SUB,
             linespacing=1.5)
    axl.set_xlim(-1.30, 1.30)
    axl.set_ylim(-1.30, 1.30)
    axl.set_aspect("equal")
    axl.axis("off")
    axl.set_title("$L(\\mathbf{v})$ at P over every view", fontsize=8.6,
                  fontweight="bold", pad=4.0)
    add_panel_label(axl, "(c)", x=-0.045, y=1.005)

    # ---- (d) diffuse vs diffuse + specular ---------------------------------
    tiles = [(diff_only, "diffuse only", None),
             (full, "diffuse + specular", None),
             (delta, "difference (luminance)", get_cmap("cb_sequential_blue"))]
    for j, (img, lab, cm) in enumerate(tiles):
        ax = fig.add_subplot(bot[0, j])
        if cm is None:
            ax.imshow(img, extent=(-1.34, 1.34, -1.34, 1.34), origin="lower",
                      interpolation="bilinear")
        else:
            im = ax.imshow(img, extent=(-1.34, 1.34, -1.34, 1.34),
                           origin="lower", cmap=cm, vmin=0.0,
                           vmax=float(delta.max()), interpolation="bilinear")
            cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.025)
            cb.outline.set_linewidth(0.6)
            cb.outline.set_edgecolor("#333333")
            cb.ax.tick_params(labelsize=6.4, width=0.6, length=2.0)
            ax.add_patch(Circle((0, 0), 1.0, facecolor="none",
                                edgecolor="#9aa7b2", lw=0.7, zorder=4))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#9aa7b2")
            s.set_linewidth(0.7)
        ax.set_xlabel(lab, fontsize=7.4, color=SUB, labelpad=2.5)
        if j == 0:
            ax.set_title(f"The specular term touches "
                         f"{100 * (delta > 0.01 * delta.max()).mean():.0f} % "
                         f"of the frame and changes it by up to "
                         f"{delta.max():.2f}", fontsize=8.6,
                         fontweight="bold", pad=4.0, loc="left", x=0.052)
            add_panel_label(ax, "(d)", x=-0.108, y=1.015)

    axn = fig.add_subplot(bot[0, 3])
    axn.set_facecolor(CARD)
    axn.set_xticks([])
    axn.set_yticks([])
    for s in axn.spines.values():
        s.set_color(HAIR)
        s.set_linewidth(0.7)
    axn.text(0.055, 0.94, "Reading the three panels together", fontsize=7.6,
             fontweight="bold", color=INK, ha="left", va="top",
             transform=axn.transAxes)
    order = np.argsort(frac)[::-1]
    bullets = [
        f"The mirror direction is d{order[0] + 1} at {ang[order[0]]:.0f} deg "
        f"off n, where the specular term is {frac[order[0]]:.0f} % of the "
        f"radiance; the next brightest view, d{order[1] + 1}, is already down "
        f"to {frac[order[1]]:.0f} %.",
        "The lobe in (c) is one narrow spot on an otherwise flat "
        "hemisphere - exactly the shape nine coefficients cannot hold.",
        f"Degree-2 SH is off by {100 * emax / lmax:.0f} % there while holding "
        f"{100 * rms / lmax:.1f} % RMS overall: the fit looks fine and the "
        f"highlight still vanishes.",
    ]
    body = "\n\n".join(textwrap.fill(f"- {b}", 44, subsequent_indent="  ")
                       for b in bullets)
    axn.text(0.055, 0.855, body, fontsize=6.4, color=SUB, ha="left",
             va="top", linespacing=1.50, transform=axn.transAxes)

    savefig(fig, "1157_wm_nr_view_dependence.png")


if __name__ == "__main__":
    main()
