"""BRDF reflection lobes for four surface models, sliced in the incidence plane.

Every lobe comes from one vector-valued BRDF evaluated on the same set of
outgoing directions, so the four shapes are directly comparable: only the model
changes.  The main polar panel is peak-normalised because a mirror-like GGX
lobe is three orders of magnitude taller than a Lambertian one and a shared
linear radius would flatten the diffuse material into the axis - the true peak
of each lobe is printed in the legend instead, so no magnitude is hidden.

The two auxiliary panels carry the evidence.  The roughness strip re-evaluates
the same GGX model at four widths on one shared radial scale, and the
reflectance panel integrates each BRDF over the full hemisphere (not just the
plotted slice) against incidence angle, so the energy bound at unity is a real
test rather than a decoration.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GUIDE = "#808080"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

THETA_I = np.deg2rad(45.0)          # incidence angle used for the lobe slice
RHO_D = 0.62                        # Lambertian albedo
PHONG_KS, PHONG_N = 0.55, 48.0
WARD_RS, WARD_AX, WARD_AY = 0.42, 0.11, 0.34
GGX_ALPHA, GGX_F0 = 0.18, 0.35
ROUGHNESS = [0.05, 0.15, 0.35, 0.70]


def light_dir(theta_i):
    """Unit vector toward the light, in the x-z incidence plane."""
    return np.array([-np.sin(theta_i), 0.0, np.cos(theta_i)])


def mirror_dir(theta_i):
    return np.array([np.sin(theta_i), 0.0, np.cos(theta_i)])


def _unit(v, axis=0):
    return v / np.clip(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12, None)


def brdf_lambert(wo, theta_i):
    return np.full(wo.shape[1], RHO_D / np.pi)


def brdf_phong(wo, theta_i):
    r = mirror_dir(theta_i)
    c = np.clip(r @ wo, 0.0, 1.0)
    return PHONG_KS * (PHONG_N + 2.0) / (2.0 * np.pi) * c ** PHONG_N


def brdf_ward(wo, theta_i):
    """Ward anisotropic lobe; the tangent axis is x, so it stretches sideways."""
    wi = light_dir(theta_i)
    h = _unit(wi[:, None] + wo)
    ch = np.clip(h[2], 1e-4, 1.0)
    tan2 = (1.0 - ch ** 2) / ch ** 2
    hx, hy = h[0], h[1]
    rad2 = np.clip(hx ** 2 + hy ** 2, 1e-12, None)
    cp2, sp2 = hx ** 2 / rad2, hy ** 2 / rad2
    ci = np.cos(theta_i)
    co = np.clip(wo[2], 1e-3, 1.0)
    pref = WARD_RS / (4.0 * np.pi * WARD_AX * WARD_AY * np.sqrt(ci * co))
    return pref * np.exp(-tan2 * (cp2 / WARD_AX ** 2 + sp2 / WARD_AY ** 2))


def brdf_ggx(wo, theta_i, alpha=GGX_ALPHA, f0=GGX_F0):
    """Cook-Torrance microfacet lobe: GGX distribution, Smith G, Schlick F."""
    wi = light_dir(theta_i)
    h = _unit(wi[:, None] + wo)
    ch = np.clip(h[2], 1e-4, 1.0)
    a2 = alpha ** 2
    d = a2 / (np.pi * (ch ** 2 * (a2 - 1.0) + 1.0) ** 2)
    ci = np.cos(theta_i)
    co = np.clip(wo[2], 2e-2, 1.0)
    g1i = 2.0 * ci / (ci + np.sqrt(a2 + (1.0 - a2) * ci ** 2))
    g1o = 2.0 * co / (co + np.sqrt(a2 + (1.0 - a2) * co ** 2))
    coh = np.clip(np.sum(wo * h, axis=0), 0.0, 1.0)
    f = f0 + (1.0 - f0) * (1.0 - coh) ** 5
    return d * g1i * g1o * f / (4.0 * ci * co)


MATERIALS = [
    ("Lambertian diffuse", brdf_lambert, BLUE, "-"),
    ("Blinn-Phong specular", brdf_phong, GREEN, "-"),
    ("Ward anisotropic", brdf_ward, GOLD, "-"),
    ("Cook-Torrance GGX", brdf_ggx, PURPLE, "-"),
]


def slice_dirs(n=721):
    """Outgoing directions in the incidence plane, signed about the normal."""
    t = np.deg2rad(np.linspace(-88.0, 88.0, n))
    return t, np.stack([np.sin(t), np.zeros_like(t), np.cos(t)])


def hemisphere_grid(n_t=120, n_p=181):
    t = np.linspace(1e-3, np.pi / 2 - 1e-3, n_t)
    p = np.linspace(0.0, 2.0 * np.pi, n_p)
    T, P = np.meshgrid(t, p, indexing="ij")
    wo = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)])
    return T, wo.reshape(3, -1), t, p


def directional_reflectance(fn, theta_i):
    """Hemispherical-directional reflectance: integral of f_r cos over the dome."""
    T, wo, t, p = hemisphere_grid()
    f = fn(wo, theta_i).reshape(T.shape)
    integ = f * np.cos(T) * np.sin(T)
    return np.trapezoid(np.trapezoid(integ, p, axis=1), t)


def polar_angle(t_signed):
    """Signed angle from the normal -> polar plot angle (surface along 0-180)."""
    return np.pi / 2.0 - t_signed


def dress_polar(ax, rmax, tick_fs=6.6):
    ax.set_thetamin(0.0)
    ax.set_thetamax(180.0)
    ax.set_ylim(0.0, rmax)
    ax.set_thetagrids(np.arange(0.0, 181.0, 30.0),
                      labels=["90", "60", "30", "0", "30", "60", "90"],
                      fontsize=tick_fs)
    ax.tick_params(axis="x", pad=1.0)
    ax.grid(True, color=GRID, lw=0.5)
    ax.spines["polar"].set_color(RULE)
    ax.spines["polar"].set_linewidth(0.7)


def draw_geometry(ax, rmax, arrow_len=0.94, label_fs=6.8):
    """Incident ray, surface normal and mirror direction on a lobe panel."""
    a_in = polar_angle(-THETA_I)
    a_mir = polar_angle(THETA_I)
    ax.annotate("", xy=(a_in, 0.0), xytext=(a_in, arrow_len * rmax),
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=10))
    ax.plot([np.pi / 2, np.pi / 2], [0.0, 0.86 * rmax], color=GUIDE, lw=0.8,
            ls=(0, (3, 2.2)), zorder=2)
    ax.plot([a_mir, a_mir], [0.0, 0.86 * rmax], color=GUIDE, lw=0.8,
            ls=(0, (1.6, 2.0)), zorder=2)
    ax.text(a_in, 1.02 * rmax, r"$\omega_i$", ha="center", va="bottom",
            fontsize=label_fs, color=RULE)
    ax.text(np.pi / 2, 0.90 * rmax, "n", ha="center", va="bottom",
            fontsize=label_fs, color=MUTED)
    ax.text(a_mir, 0.90 * rmax, "mirror", ha="center", va="bottom",
            fontsize=label_fs, color=MUTED)


def main():
    set_pub_style()
    t, wo = slice_dirs()
    lobes, peaks = {}, {}
    for name, fn, _c, _ls in MATERIALS:
        f = fn(wo, THETA_I)
        lobes[name] = f
        peaks[name] = float(np.max(f))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.62],
                          height_ratios=[1.0, 0.74],
                          left=0.045, right=0.975, top=0.855, bottom=0.140,
                          wspace=0.26, hspace=0.52)

    # ---- main: four peak-normalised lobes on one radial scale --------------
    ax = fig.add_subplot(gs[:, 0], projection="polar")
    dress_polar(ax, 1.10, tick_fs=7.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    draw_geometry(ax, 1.10, label_fs=7.2)

    for name, fn, col, ls in MATERIALS:
        r = lobes[name] / peaks[name]
        ax.plot(polar_angle(t), r, color=col, lw=1.5, ls=ls, zorder=5,
                label=rf"{name}  (peak $f_r$ = {peaks[name]:.2f} sr$^{{-1}}$)")
        ax.fill(polar_angle(t), r, color=col, alpha=0.10, lw=0, zorder=3)

    ax.set_title(r"Reflected lobes at $\theta_i$ = "
                 f"{np.rad2deg(THETA_I):.0f} deg", fontsize=9.0,
                 fontweight="bold", color=INK, pad=6)
    ax.text(0.5, 0.145, "outgoing angle from normal (deg)",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.2,
            color=SUB)
    ax.text(0.5, 0.100, r"radius: peak-normalised $f_r$ (-), outer ring = 1.00",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.8,
            color=MUTED)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.095), ncol=2,
                    frameon=False, fontsize=7.0, handlelength=1.5,
                    columnspacing=1.0, labelspacing=0.30)
    for txt in leg.get_texts():
        txt.set_color(SUB)

    # ---- aux 1: roughness sweep, one shared radial scale -------------------
    gsr = gs[0, 1].subgridspec(2, 2, wspace=0.34, hspace=0.60)
    ramp = get_palette("cb_sequential_blue", 6)[2:5]
    rough_cols = [ramp[0], ramp[1], ramp[2], ACCENT]
    mini_axes = []
    for k, (a_r, col) in enumerate(zip(ROUGHNESS, rough_cols)):
        axr = fig.add_subplot(gsr[k // 2, k % 2], projection="polar")
        dress_polar(axr, 1.12, tick_fs=6.0)
        axr.set_thetagrids(np.arange(0.0, 181.0, 45.0),
                           labels=["90", "45", "0", "45", "90"], fontsize=6.0)
        axr.set_yticklabels([])
        axr.set_yticks([0.5, 1.0])
        f = brdf_ggx(wo, THETA_I, alpha=a_r)
        pk = float(np.max(f))
        axr.plot(polar_angle(t), f / pk, color=col, lw=1.3, zorder=5)
        axr.fill(polar_angle(t), f / pk, color=col, alpha=0.14, lw=0, zorder=3)
        axr.plot([polar_angle(THETA_I)] * 2, [0.0, 1.0], color=GUIDE, lw=0.7,
                 ls=(0, (1.6, 2.0)), zorder=2)
        axr.set_title(rf"$\alpha$ = {a_r:.2f}", fontsize=7.4, color=INK,
                      pad=1.5)
        axr.text(0.5, 0.055, rf"peak {pk:.3g} sr$^{{-1}}$",
                 transform=axr.transAxes, ha="center", va="center",
                 fontsize=6.2, color=MUTED)
        mini_axes.append(axr)

    mini_axes[0].text(0.44, 1.42, "GGX roughness sweep",
                      transform=mini_axes[0].transAxes, ha="left", va="bottom",
                      fontsize=8.0, fontweight="bold", color=INK)

    # ---- aux 2: hemispherical reflectance vs incidence ---------------------
    axe = fig.add_subplot(gs[1, 1])
    ti = np.deg2rad(np.linspace(0.0, 85.0, 60))
    worst, worst_at, worst_col = 0.0, 0.0, RULE
    for name, fn, col, _ls in MATERIALS:
        rho = np.array([directional_reflectance(fn, x) for x in ti])
        axe.plot(np.rad2deg(ti), rho, color=col, lw=1.4)
        if rho.max() > worst:
            worst = float(rho.max())
            worst_at = float(np.rad2deg(ti[int(np.argmax(rho))]))
            worst_col = col
    axe.annotate(f"highest {worst:.2f} at {worst_at:.0f} deg",
                 xy=(worst_at, worst), xytext=(6.0, 8.0),
                 textcoords="offset points", ha="left", va="bottom",
                 fontsize=6.4, color=worst_col)
    axe.axhline(1.0, color=GUIDE, lw=0.9, ls=(0, (3.2, 2.4)))
    axe.text(84.0, 1.02, "energy bound", ha="right", va="bottom",
             fontsize=6.4, color=GUIDE)
    axe.set_xlim(0.0, 85.0)
    axe.set_ylim(0.0, max(1.18, worst * 1.10))
    axe.set_xlabel(r"incidence angle $\theta_i$ (deg)", fontsize=8.0)
    axe.set_ylabel("reflectance (-)", fontsize=8.0)
    axe.set_title("Hemispherical reflectance stays below unity",
                  fontsize=8.5, fontweight="bold", color=INK, pad=4)
    axe.tick_params(labelsize=6.6)
    axe.grid(True, axis="y", color=GRID, lw=0.5)


    panel_labels_inches(fig, [ax, mini_axes[0], axe], dx=-0.02, dy=0.36)
    savefig(fig, "648_cg_brdf_lobes.png")


if __name__ == "__main__":
    main()
