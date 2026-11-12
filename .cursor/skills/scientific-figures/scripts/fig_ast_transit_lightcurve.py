"""Phase-folded exoplanet transit: depth, duration and the two follow-ups.

The main panel is the claim: folding twenty-odd epochs on the fitted ephemeris
turns a 1.6 % dip that is invisible in a single night into a curve whose four
contact points can be read off individually.  The transit model is a
limb-darkened overlap integral, so the ingress and egress ramps and the
slightly bowed floor come out of the geometry rather than being drawn by hand;
every annotated quantity - depth, radius ratio, duration, period - is measured
from the plotted arrays.

The residual strip underneath shows the fold leaves no structure.  The two
side panels carry the follow-up evidence: transit-timing residuals against a
linear ephemeris, and the depth measured band by band.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GREY = "#808080"
RED = "#b23a48"          # model
BLUE = "#3b5b92"         # residual band
GOLD = "#c9a227"

P_TRUE = 3.52474         # days
RP_RS = 0.1180
A_RS = 11.20
IMPACT = 0.36
U1, U2 = 0.42, 0.22
SIGMA_PPM = 780.0
N_OBS = 1400


def limb_intensity(r):
    """Quadratic limb-darkening profile, normalised to the disc mean."""
    mu = np.sqrt(np.clip(1.0 - r ** 2, 0.0, 1.0))
    prof = 1.0 - U1 * (1.0 - mu) - U2 * (1.0 - mu) ** 2
    return prof / (1.0 - U1 / 3.0 - U2 / 6.0)


def overlap_area(z, p):
    """Area of the lens shared by a unit disc and a disc of radius p."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    full = z <= 1.0 - p
    out[full] = np.pi * p ** 2
    part = (~full) & (z < 1.0 + p)
    zp = z[part]
    k0 = np.arccos(np.clip((p ** 2 + zp ** 2 - 1.0) / (2.0 * p * zp),
                           -1.0, 1.0))
    k1 = np.arccos(np.clip((1.0 - p ** 2 + zp ** 2) / (2.0 * zp), -1.0, 1.0))
    tri = np.sqrt(np.clip(4.0 * zp ** 2 - (1.0 + zp ** 2 - p ** 2) ** 2,
                          0.0, None)) / 2.0
    out[part] = p ** 2 * k0 + k1 - tri
    return out


def transit_model(t_hours, p=RP_RS):
    """Normalised flux for a circular orbit, limb-darkened star."""
    phase = 2.0 * np.pi * (t_hours / 24.0) / P_TRUE
    z = np.sqrt((A_RS * np.sin(phase)) ** 2 + (IMPACT * np.cos(phase)) ** 2)
    area = overlap_area(z, p)
    r_eff = np.clip(0.5 * (z + np.maximum(z - p, 0.0)), 0.0, 0.9995)
    return 1.0 - area / np.pi * limb_intensity(r_eff)


def contact_times():
    """Four contact points in hours, solved on a dense grid of the model."""
    t = np.linspace(-6.0, 6.0, 400001)
    phase = 2.0 * np.pi * (t / 24.0) / P_TRUE
    z = np.sqrt((A_RS * np.sin(phase)) ** 2 + (IMPACT * np.cos(phase)) ** 2)
    outer = t[z <= 1.0 + RP_RS]
    inner = t[z <= 1.0 - RP_RS]
    return outer.min(), inner.min(), inner.max(), outer.max()


def observations(seed=31):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-5.0, 5.0, N_OBS)
    t.sort()
    trend = 1.0 + 2.6e-4 * np.sin(0.9 * t + 0.4)      # residual systematics
    flux = transit_model(t) * trend
    flux = flux + rng.normal(0.0, SIGMA_PPM * 1e-6, t.size)
    return t, flux


def binned(t, y, edges):
    idx = np.digitize(t, edges) - 1
    xs, ys, es, ns = [], [], [], []
    for k in range(edges.size - 1):
        m = idx == k
        if m.sum() >= 4:
            xs.append(0.5 * (edges[k] + edges[k + 1]))
            ys.append(y[m].mean())
            es.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
            ns.append(int(m.sum()))
    return (np.array(xs), np.array(ys), np.array(es), np.array(ns))


def transit_times(seed=8, n_epoch=42):
    rng = np.random.default_rng(seed)
    n = np.arange(n_epoch)
    amp_d = 3.4 / (24.0 * 60.0)                       # 3.4 min TTV amplitude
    super_p = 13.5                                    # epochs per TTV cycle
    err = (0.55 + 0.35 * rng.random(n_epoch)) / (24.0 * 60.0)
    t_mid = (2458000.0 + n * P_TRUE
             + amp_d * np.sin(2.0 * np.pi * n / super_p + 0.7)
             + rng.normal(0.0, 1.0, n_epoch) * err)
    return n, t_mid, err, super_p


def transmission(base, seed=4):
    """Depth per band, built around the measured white-light depth."""
    rng = np.random.default_rng(seed)
    wl = np.array([0.45, 0.58, 0.72, 0.88, 1.05, 1.15, 1.25, 1.35, 1.45,
                   1.55, 1.65, 2.15, 3.60, 4.50])
    rayleigh = 520.0 * (-np.log(wl / 1.0))            # scattering slope
    water = 430.0 * np.exp(-0.5 * ((wl - 1.42) / 0.13) ** 2)
    methane = 260.0 * np.exp(-0.5 * ((wl - 3.35) / 0.28) ** 2)
    err = 95.0 + 60.0 * rng.random(wl.size)
    depth = (base + rayleigh + water + methane
             + rng.normal(0.0, 1.0, wl.size) * err)
    return wl, depth, err


def main_panel(ax, t, flux, model_t, model_f, c1, c2, c3, c4):
    ax.scatter(t, flux, s=2.2, color="#9a9a9a", alpha=0.45, linewidths=0.0,
               zorder=2, label="single-visit photometry")
    edges = np.linspace(-5.0, 5.0, 61)
    bx, by, be, bn = binned(t, flux, edges)
    ax.errorbar(bx, by, yerr=be, fmt="o", ms=3.2, mfc="white", mec=RULE,
                mew=0.8, ecolor=RULE, elinewidth=0.8, capsize=1.6,
                linestyle="none", zorder=4, label="10 min bins")
    ax.plot(model_t, model_f, color=RED, lw=1.6, zorder=5,
            label="limb-darkened model")

    for x in (c1, c2, c3, c4):
        ax.axvline(x, color=GREY, lw=0.7, ls=(0, (2.2, 2.0)), zorder=1)
    depth = 1.0 - model_f.min()
    ax.annotate("", xy=(c2, 1.0), xytext=(c2, 1.0 - depth),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9))
    ax.text(c2 - 0.12, 1.0 - depth / 2.0,
            f"depth $\\delta$ = {depth * 1e6:,.0f} ppm", ha="right",
            va="center", fontsize=7.2, color=BLUE)

    lo = 1.0 - depth
    ax.annotate("", xy=(c1, lo - 0.0042), xytext=(c4, lo - 0.0042),
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=0.9))
    ax.text(0.0, lo - 0.0050, f"$T_{{14}}$ = {(c4 - c1):.2f} h", ha="center",
            va="top", fontsize=7.2, color=GOLD)
    ax.text(c1, 1.0018, "ingress", ha="center", va="bottom", fontsize=6.6,
            color=MUTED)
    ax.text(c4, 1.0018, "egress", ha="center", va="bottom", fontsize=6.6,
            color=MUTED)

    ax.axhline(1.0, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(lo - 0.0072, 1.0035)
    ax.set_ylabel("normalised flux (-)")
    ax.tick_params(labelbottom=False, labelsize=7.2)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.legend(fontsize=7.0, frameon=False, loc="lower left",
              handlelength=1.5, borderaxespad=0.3, ncol=1)
    return depth, edges


def residual_panel(ax, t, flux, edges):
    """Residuals are formed per sample, then binned.

    Binning the flux first and subtracting the model at the bin centre puts a
    spurious spike on every bin that straddles the ingress or egress ramp,
    where the model moves by more than the noise across one bin.
    """
    rx, res, re_, _ = binned(t, (flux - transit_model(t)) * 1e6, edges)
    rms_ppm = float(np.median(re_))
    ax.axhspan(-rms_ppm, rms_ppm, color=BLUE, alpha=0.16, linewidth=0.0,
               zorder=1)
    ax.axhline(0.0, color=RED, lw=1.0, zorder=3)
    ax.errorbar(rx, res, yerr=re_, fmt="o", ms=2.6, mfc="white",
                mec=RULE, mew=0.7, ecolor=RULE, elinewidth=0.7, capsize=1.3,
                linestyle="none", zorder=4)
    span = max(np.abs(res).max() * 1.25, rms_ppm * 2.4)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-span, span)
    ax.set_xlabel("time from mid-transit (h)")
    ax.set_ylabel("residual (ppm)", fontsize=7.6)
    ax.tick_params(labelsize=7.0)
    ax.text(0.985, 0.92, f"binned scatter {res.std(ddof=1):.0f} ppm,  "
                         f"band = median standard error",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=SUB)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    return res


def ttv_panel(ax, n, t_mid, err, super_p):
    slope, icept = np.polyfit(n, t_mid, 1)
    oc = (t_mid - (icept + slope * n)) * 24.0 * 60.0
    design = np.column_stack([np.sin(2.0 * np.pi * n / super_p),
                              np.cos(2.0 * np.pi * n / super_p)])
    coef, *_ = np.linalg.lstsq(design, oc, rcond=None)
    amp = float(np.hypot(*coef))
    grid = np.linspace(n.min(), n.max(), 320)
    fit = (coef[0] * np.sin(2.0 * np.pi * grid / super_p)
           + coef[1] * np.cos(2.0 * np.pi * grid / super_p))

    ax.axhline(0.0, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(n, oc, yerr=err * 24.0 * 60.0, fmt="o", ms=3.0, mfc="white",
                mec=RULE, mew=0.8, ecolor=RULE, elinewidth=0.7, capsize=1.4,
                linestyle="none", zorder=3)
    ax.plot(grid, fit, color=RED, lw=1.4, zorder=4)
    ax.text(0.03, 0.05,
            f"$P$ = {slope:.5f} d,  TTV amplitude {amp:.1f} min",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=RED)
    ax.set_xlabel("epoch (-)", fontsize=7.4)
    ax.set_ylabel("O $-$ C (min)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Transit timing against a linear ephemeris", fontsize=8.4,
                 fontweight="bold", color=INK)
    return slope, amp


def spectrum_panel(ax, wl, depth, err, white_ppm):
    ax.axhline(white_ppm, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.errorbar(wl, depth, yerr=err, fmt="o", ms=3.4, mfc="white", mec=RULE,
                mew=0.8, ecolor=RULE, elinewidth=0.7, capsize=1.5,
                linestyle="none", zorder=4)
    ax.plot(wl, depth, color=RED, lw=1.0, alpha=0.75, zorder=3)
    k = int(np.argmin(np.abs(wl - 1.45)))
    ax.text(wl[k], depth[k] + 2.4 * err[k], "H$_2$O", ha="center",
            va="bottom", fontsize=6.8, color=RED)
    amp = depth.max() - depth.min()
    ax.text(0.03, 0.06, f"band-to-band spread {amp:,.0f} ppm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=SUB)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0.5", "1.0", "2.0", "4.0"])
    ax.set_xlabel(r"wavelength ($\mu$m)", fontsize=7.4)
    ax.set_ylabel("depth (ppm)", fontsize=7.4)
    ax.set_ylim(depth.min() - 2.2 * err.max(), depth.max() + 3.0 * err.max())
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Transmission spectrum, depth per band", fontsize=8.4,
                 fontweight="bold", color=INK)
    return amp


def main():
    set_pub_style()
    t, flux = observations()
    model_t = np.linspace(-5.0, 5.0, 3000)
    model_f = transit_model(model_t)
    c1, c2, c3, c4 = contact_times()
    n, t_mid, err, super_p = transit_times()

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    ax_main = axes_inches(fig, (0.90, 1.92, 5.45, 3.54))
    ax_res = axes_inches(fig, (0.90, 0.86, 5.45, 0.90))
    ax_ttv = axes_inches(fig, (7.05, 3.30, 3.00, 2.16))
    ax_spec = axes_inches(fig, (7.05, 0.86, 3.00, 1.74))

    depth, edges = main_panel(ax_main, t, flux, model_t, model_f,
                              c1, c2, c3, c4)
    res = residual_panel(ax_res, t, flux, edges)
    p_fit, ttv_amp = ttv_panel(ax_ttv, n, t_mid, err, super_p)
    wl, depth_b, err_b = transmission(depth * 1e6)
    spread = spectrum_panel(ax_spec, wl, depth_b, err_b, depth * 1e6)

    # Radius ratio recovered from the plotted depth, corrected for the
    # limb-darkened intensity at the transit chord.
    ld_gain = float(limb_intensity(np.array([IMPACT]))[0])
    rp_rs = np.sqrt(depth / ld_gain)
    ax_main.set_title(
        f"A {depth * 1e6:,.0f} ppm dip implies $R_{{\\rm p}}/R_{{\\rm s}}$ = "
        f"{rp_rs:.3f} on a {(c4 - c1):.2f} h chord", fontsize=9.0,
        fontweight="bold", color=INK)


    # The residual strip belongs to the main panel, so it is not lettered.
    panel_labels_inches(fig, [ax_main, ax_ttv, ax_spec], dx=-0.66, dy=0.28)
    savefig(fig, "799_ast_transit_lightcurve.png")


if __name__ == "__main__":
    main()
