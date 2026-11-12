"""413 - Composite wireline log of a clastic reservoir interval.

Six tracks share one depth axis that increases downward, the hard convention of
the domain.  Every curve is generated from a bed model and then interpreted
back out of the curves: the lithology and pay flags in the right-hand track are
classified from the plotted GR / RT / RHOB / NPHI samples, not from the bed
model, so the interpretation track cannot disagree with the curves beside it.
The neutron-density track carries the message - where the density curve crosses
to the left of the neutron curve the separation is filled in warm colour, which
is the classic gas effect.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, lighten, darken

INK, RULE, SUB = "#1a1a1a", "#333333", "#555555"
MUTED, HAIR, GRID = "#777777", "#cccccc", "#e2e2e2"

GAS_FILL = "#c0603f"        # density-left-of-neutron separation
SHL_FILL = "#9fb08c"        # neutron-left-of-density separation
C_GR, C_SP, C_RT = "#2e7d5b", "#8e6bbf", "#b23a48"
C_DEN, C_NEU, C_PE = "#3b5b92", "#c0603f", "#555555"

Z0, Z1, DZ = 2000.0, 2300.0, 0.15
GR_CLEAN, GR_SHALE = 15.0, 115.0
RHO_MA, RHO_FL = 2.65, 1.00

# facies -> log response (mean, sd); RT is lognormal, sd in log10 units
FACIES = {
    "shale":    dict(name="Shale", color=lighten("#2e7d5b", 0.58),
                     gr=(104, 7.0), sp=(3, 3.0), rt=(3.0, 0.10),
                     rhob=(2.535, 0.020), nphi=(0.335, 0.016), pe=(3.30, 0.14)),
    "silt":     dict(name="Silty sand", color=lighten("#c0603f", 0.62),
                     gr=(66, 7.0), sp=(-26, 4.0), rt=(7.0, 0.12),
                     rhob=(2.445, 0.020), nphi=(0.215, 0.016), pe=(2.60, 0.12)),
    "sand_wet": dict(name="Water sand", color=lighten("#3b5b92", 0.62),
                     gr=(34, 5.0), sp=(-58, 4.0), rt=(2.1, 0.09),
                     rhob=(2.290, 0.018), nphi=(0.275, 0.014), pe=(1.90, 0.10)),
    "sand_gas": dict(name="Gas sand", color=lighten("#c9a227", 0.42),
                     gr=(28, 5.0), sp=(-64, 4.0), rt=(58.0, 0.12),
                     rhob=(2.170, 0.026), nphi=(0.105, 0.014), pe=(1.85, 0.10)),
    "lime":     dict(name="Limestone", color=lighten("#8e6bbf", 0.24),
                     gr=(16, 4.0), sp=(-8, 3.0), rt=(140.0, 0.14),
                     rhob=(2.660, 0.016), nphi=(0.045, 0.010), pe=(5.00, 0.15)),
}

# (thickness in m, facies) from the top of the logged interval downward
BEDS = [
    (26, "shale"), (9, "sand_wet"), (14, "shale"), (11, "silt"),
    (18, "shale"), (16, "sand_gas"), (6, "shale"), (13, "sand_gas"),
    (5, "silt"), (10, "sand_wet"), (21, "shale"), (12, "lime"),
    (17, "shale"), (9, "silt"), (14, "sand_gas"), (7, "shale"),
    (12, "sand_wet"), (24, "shale"), (10, "lime"), (16, "shale"),
    (8, "silt"), (12, "sand_wet"), (10, "shale"),
]

# track key -> (width ratio, header rows)  header row = (label, left, right)
TRACKS = [
    ("gr", 1.00), ("sp", 0.60), ("rt", 0.92), ("dn", 1.34),
    ("pe", 0.56), ("int", 0.66),
]


def step_profile(z, key):
    """Piecewise-constant facies property along the depth grid."""
    out = np.empty(z.size)
    top = Z0
    for thick, fac in BEDS:
        m = (z >= top) & (z < top + thick)
        out[m] = FACIES[fac][key][0]
        top += thick
    out[z >= top] = FACIES[BEDS[-1][1]][key][0]
    return out


def step_sd(z, key):
    out = np.empty(z.size)
    top = Z0
    for thick, fac in BEDS:
        m = (z >= top) & (z < top + thick)
        out[m] = FACIES[fac][key][1]
        top += thick
    out[z >= top] = FACIES[BEDS[-1][1]][key][1]
    return out


def synth_log(rng, z, key, bed_sigma=0.55, noise_sigma=0.30, log10=False):
    """Bed response blurred by tool resolution plus correlated tool noise."""
    base = step_profile(z, key)
    sd = step_sd(z, key)
    if log10:
        base = np.log10(base)
    base = gaussian_filter1d(base, bed_sigma / DZ, mode="nearest")
    n = gaussian_filter1d(rng.standard_normal(z.size), noise_sigma / DZ,
                          mode="nearest")
    n /= n.std() + 1e-12
    val = base + gaussian_filter1d(sd, bed_sigma / DZ, mode="nearest") * n
    return 10.0 ** val if log10 else val


def runs_of(mask, min_len=1):
    """[(i0, i1), ...] index bounds of True runs at least ``min_len`` long."""
    out, i, n = [], 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def track_header(ax, rows):
    """Log-header rows above a track: ``left  NAME (unit)  right``."""
    for i, (label, lo, hi, color) in enumerate(rows[::-1]):
        y = 1.016 + 0.058 * i
        ax.text(0.5, y, label, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.5, color=color, fontweight="bold",
                clip_on=False)
        ax.text(0.0, y, lo, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.0, color=MUTED, clip_on=False)
        ax.text(1.0, y, hi, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.0, color=MUTED, clip_on=False)


def frame(ax, z_ticks, labelleft=False):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(Z1, Z0)
    ax.set_xticks([])
    ax.set_yticks(z_ticks)
    for x in (0.2, 0.4, 0.6, 0.8):
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)
    for zz in np.arange(Z0, Z1 + 1, 10.0):
        ax.axhline(zz, color=GRID, lw=0.4, zorder=0)
    ax.tick_params(labelleft=labelleft, labelsize=7.0, length=2.4,
                   color=RULE, pad=2.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)


def main():
    set_pub_style(base_font=8, **{"hatch.linewidth": 0.45})
    rng = np.random.default_rng(413)

    z = np.arange(Z0, Z1 + DZ, DZ)
    gr = synth_log(rng, z, "gr")
    sp = synth_log(rng, z, "sp", bed_sigma=1.20)
    rt = synth_log(rng, z, "rt", log10=True)
    rhob = synth_log(rng, z, "rhob", bed_sigma=0.40)
    nphi = synth_log(rng, z, "nphi", bed_sigma=0.40)
    pe = synth_log(rng, z, "pe", bed_sigma=0.40)

    # track coordinates: standard petrophysical scales, both porosity curves
    # on the compatible sandstone pair 1.95-2.95 g/cm3 vs 0.45 to -0.15 v/v
    x_gr = np.clip(gr / 150.0, 0, 1)
    x_sp = np.clip((sp + 80.0) / 100.0, 0, 1)
    x_rt = np.clip((np.log10(rt) - np.log10(0.2)) / 4.0, 0, 1)
    x_den = np.clip((rhob - 1.95) / 1.00, 0, 1)
    x_neu = np.clip((0.45 - nphi) / 0.60, 0, 1)
    x_pe = np.clip(pe / 10.0, 0, 1)

    sep = x_neu - x_den                 # > 0 : density left of neutron = gas
    vsh = np.clip((gr - GR_CLEAN) / (GR_SHALE - GR_CLEAN), 0.0, 1.0)
    phi_d = (RHO_MA - rhob) / (RHO_MA - RHO_FL)
    phi_e = np.clip(0.5 * (phi_d + nphi) * (1.0 - vsh), 0.0, 0.40)

    # ---- interpretation classified from the curves themselves --------------
    is_gas = (sep > 0.055) & (gr < 60.0) & (rt > 15.0)
    for i0, i1 in runs_of(~is_gas, 1):        # bridge sub-metre gaps
        if (i1 - i0) * DZ < 0.9 and i0 > 0 and i1 < z.size:
            is_gas[i0:i1] = True
    keep = np.zeros_like(is_gas)
    for i0, i1 in runs_of(is_gas, int(round(1.5 / DZ))):
        keep[i0:i1] = True
    is_gas = keep

    lith = np.empty(z.size, dtype=object)
    for i in range(z.size):
        if pe[i] > 4.2 and gr[i] < 34.0:
            lith[i] = "lime"
        elif is_gas[i]:
            lith[i] = "sand_gas"
        elif gr[i] < 50.0:
            lith[i] = "sand_wet"
        elif gr[i] < 82.0:
            lith[i] = "silt"
        else:
            lith[i] = "shale"

    net_gas = float(is_gas.sum()) * DZ
    gas_runs = runs_of(is_gas, 1)
    phi_gas = float(phi_e[is_gas].mean()) if is_gas.any() else 0.0

    # ---- canvas ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, len(TRACKS),
                          width_ratios=[w for _, w in TRACKS],
                          left=0.055, right=0.845, top=0.775, bottom=0.105,
                          wspace=0.055)
    axs = {k: fig.add_subplot(gs[0, i]) for i, (k, _) in enumerate(TRACKS)}
    z_ticks = np.arange(Z0, Z1 + 1, 25.0)
    for i, (k, _) in enumerate(TRACKS):
        frame(axs[k], z_ticks, labelleft=(i == 0))
    axs["gr"].set_ylabel("Measured depth (m)", fontsize=8.0)

    # ---- track 1: gamma ray with sand / shale flags ------------------------
    ax = axs["gr"]
    cut = 60.0 / 150.0
    ax.axvline(cut, color=MUTED, lw=0.7, ls=(0, (3.0, 2.0)), zorder=2)
    ax.fill_betweenx(z, x_gr, cut, where=x_gr < cut, color=lighten("#c9a227", 0.55),
                     lw=0, zorder=1)
    ax.fill_betweenx(z, cut, x_gr, where=x_gr > cut, color=lighten("#2e7d5b", 0.62),
                     lw=0, zorder=1)
    ax.plot(x_gr, z, color=C_GR, lw=0.75, zorder=4)
    track_header(ax, [("GR (API)", "0", "150", C_GR)])
    ax.text(0.03, 0.996, "sand | shale split at 60 API", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.0, color=SUB, zorder=6,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none",
                      alpha=0.85))

    # ---- track 2: spontaneous potential ------------------------------------
    ax = axs["sp"]
    base_sp = np.percentile(sp, 88)
    xb = (base_sp + 80.0) / 100.0
    ax.axvline(xb, color=MUTED, lw=0.7, ls=(0, (3.0, 2.0)), zorder=2)
    ax.fill_betweenx(z, x_sp, xb, where=x_sp < xb, color=lighten("#8e6bbf", 0.72),
                     lw=0, zorder=1)
    ax.plot(x_sp, z, color=C_SP, lw=0.75, zorder=4)
    track_header(ax, [("SP (mV)", "-80", "+20", C_SP)])
    ax.text(0.96, 0.996, "shale\nbase line", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.0, color=SUB, linespacing=1.2, zorder=6,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none",
                      alpha=0.85))

    # ---- track 3: deep resistivity (log scale) -----------------------------
    ax = axs["rt"]
    for dec in (1.0, 10.0, 100.0):
        ax.axvline((np.log10(dec) - np.log10(0.2)) / 4.0, color=GRID, lw=0.5,
                   zorder=0)
    x_hc = (np.log10(15.0) - np.log10(0.2)) / 4.0
    ax.fill_betweenx(z, x_hc, x_rt, where=x_rt > x_hc, color=lighten("#b23a48", 0.74),
                     lw=0, zorder=1)
    ax.plot(x_rt, z, color=C_RT, lw=0.75, zorder=4)
    track_header(ax, [("RT (\u03a9\u00b7m)", "0.2", "2000", C_RT)])

    # ---- track 4: neutron-density crossover (the message) ------------------
    ax = axs["dn"]
    ax.fill_betweenx(z, x_den, x_neu, where=sep > 0, color=GAS_FILL, alpha=0.72,
                     lw=0, zorder=2, interpolate=True)
    ax.fill_betweenx(z, x_den, x_neu, where=sep <= 0, color=SHL_FILL, alpha=0.48,
                     lw=0, zorder=1, interpolate=True)
    ax.plot(x_den, z, color=C_DEN, lw=0.85, zorder=5)
    ax.plot(x_neu, z, color=C_NEU, lw=0.85, ls=(0, (3.2, 1.6)), zorder=5)
    track_header(ax, [("RHOB (g/cm\u00b3)", "1.95", "2.95", C_DEN),
                      ("NPHI (v/v)", "0.45", "-0.15", C_NEU)])

    # ---- track 5: photoelectric factor -------------------------------------
    ax = axs["pe"]
    ax.plot(x_pe, z, color=C_PE, lw=0.75, zorder=4)
    ax.fill_betweenx(z, 0.0, x_pe, color="#e8e8e8", lw=0, zorder=1)
    track_header(ax, [("PE (b/e)", "0", "10", C_PE)])

    # ---- track 6: interpretation -------------------------------------------
    ax = axs["int"]
    edges = np.r_[z - DZ / 2, z[-1] + DZ / 2]
    start = 0
    for i in range(1, z.size + 1):
        if i == z.size or lith[i] != lith[start]:
            spec = FACIES[lith[start]]
            ax.add_patch(Rectangle((0.0, edges[start]), 0.52,
                                   edges[i] - edges[start],
                                   facecolor=spec["color"], edgecolor="none",
                                   zorder=2))
            start = i
    for i0, i1 in gas_runs:
        ax.add_patch(Rectangle((0.56, edges[i0]), 0.44, edges[i1] - edges[i0],
                               facecolor=lighten(GAS_FILL, 0.30),
                               edgecolor=darken(GAS_FILL, 0.35), lw=0.5,
                               hatch="////", zorder=3))
    ax.axvline(0.54, color=RULE, lw=0.6, zorder=4)
    track_header(ax, [("Interpretation", "", "", INK), ("", "", "", INK)])
    for xc, lab in ((0.26, "lithology"), (0.78, "pay")):
        ax.text(xc, Z0 - 1.0, lab, ha="center", va="bottom", fontsize=6.0,
                color=MUTED, clip_on=False)

    # zone call-outs for the three thickest gas intervals
    tr = ax.get_yaxis_transform()
    thick = sorted(gas_runs, key=lambda r: r[1] - r[0], reverse=True)[:3]
    for k, (i0, i1) in enumerate(sorted(thick)):
        zm = 0.5 * (z[i0] + z[min(i1, z.size - 1)])
        ax.annotate(f"Zone {'ABC'[k]}   {z[i0]:.0f}\u2013{z[min(i1, z.size - 1)]:.0f} m\n"
                    f"{(i1 - i0) * DZ:.1f} m net gas, "
                    f"$\\phi_e$ = {phi_e[i0:i1].mean():.2f}",
                    xy=(1.0, zm), xytext=(1.30, zm), xycoords=tr, textcoords=tr,
                    fontsize=6.2, color=INK, va="center", ha="left",
                    linespacing=1.3, annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.28", fc="#fbf3ef",
                              ec=lighten(GAS_FILL, 0.45), lw=0.6),
                    arrowprops=dict(arrowstyle="-", color=darken(GAS_FILL, 0.2),
                                    lw=0.7, shrinkA=1.5, shrinkB=1.5))

    # ---- keys ---------------------------------------------------------------
    handles = [Patch(facecolor=FACIES[k]["color"], edgecolor=HAIR, lw=0.5,
                     label=FACIES[k]["name"])
               for k in ("sand_gas", "sand_wet", "silt", "shale", "lime")]
    handles += [
        Patch(facecolor=GAS_FILL, alpha=0.72, lw=0,
              label="density-left-of-neutron (gas effect)"),
        Patch(facecolor=SHL_FILL, alpha=0.48, lw=0,
              label="neutron-left-of-density (shale)"),
        Patch(facecolor=lighten(GAS_FILL, 0.30), hatch="////",
              edgecolor=darken(GAS_FILL, 0.35), lw=0.5, label="flagged net pay"),
        Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3.0, 2.0)),
               label="cut-off / base line"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
               ncol=5, frameon=False, fontsize=6.8, handlelength=1.5,
               handleheight=1.1, columnspacing=1.3)

    savefig(fig, "413_geo_well_log.png")


if __name__ == "__main__":
    main()
