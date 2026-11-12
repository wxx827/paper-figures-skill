"""Composite well log: six side-by-side tracks over one shared depth axis.

The signature plot of formation evaluation.  A bed model is drawn first
(thicknesses, lithology, fluid), every curve is then generated *from* that
model - blocky response convolved with a tool resolution kernel plus
correlated high-frequency noise - so the gamma ray, the resistivity pair, the
density-neutron overlay and the lithology column all agree bed by bed.

Everything quoted in the footnote and in the pay-flag column is derived from
the plotted arrays: the shale and clean baselines are percentiles of the GR
curve, volume of shale comes from those baselines, porosity from the bulk
density, water saturation from Archie on the deep resistivity, and net-to-gross
from the resulting flag.  Change the seed and every number moves.

Layout is absolute (``figkit.axes_inches``) because track gutters must be
physically equal; only the leftmost track carries depth tick labels.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

GR_C = "#2e7d5b"          # gamma ray curve
RES_D = "#b23a48"         # deep resistivity
RES_S = "#3b5b92"         # shallow resistivity
RHOB_C = "#b23a48"        # bulk density
NPHI_C = "#3b5b92"        # neutron porosity
HC = "#b23a48"            # hydrocarbon
WATER = "#3b5b92"
MATRIX = "#d9d9d9"

LITH_FC = {"sand": lighten("#c9a227", 0.55),
           "shale": lighten("#2e7d5b", 0.58),
           "lime": lighten("#3b5b92", 0.58)}
LITH_HATCH = {"sand": "...", "shale": "---", "lime": "///"}
LITH_NAME = {"sand": "Sandstone", "shale": "Shale", "lime": "Limestone"}

# Bed property ranges: (GR API, RHOB g/cm3, NPHI v/v, deep Rt ohm-m)
PROPS = {
    "shale": ((94.0, 122.0), (2.49, 2.62), (0.27, 0.38), (1.6, 4.0)),
    "sand": ((22.0, 44.0), (2.20, 2.38), (0.17, 0.28), (0.9, 2.6)),
    "lime": ((11.0, 26.0), (2.57, 2.70), (0.03, 0.11), (28.0, 120.0)),
}

TOP, BASE, STEP = 2000.0, 2400.0, 0.1524      # 0.5 ft sampling
GR_LIM = (0.0, 150.0)
RES_LIM = (0.2, 2000.0)
RHOB_LIM = (1.95, 2.95)
NPHI_LIM = (0.45, -0.15)                       # reversed, drawn on RHOB axis
RW, ARCHIE_M, ARCHIE_N = 0.045, 2.0, 2.0


def build_beds(rng):
    """Random bed stack; the three thickest sands are made hydrocarbon bearing."""
    beds, d, prev = [], TOP, "shale"
    while d < BASE:
        if prev == "shale":
            lith = str(rng.choice(["sand", "lime", "sand"]))
        else:
            lith = "shale" if rng.random() < 0.72 else str(
                rng.choice(["sand", "lime"]))
        th = float(rng.uniform(5.0, 24.0))
        top, bot = d, min(d + th, BASE)
        beds.append({"top": top, "bot": bot, "lith": lith, "hc": False})
        d, prev = bot, lith
    sands = [b for b in beds if b["lith"] == "sand"]
    sands.sort(key=lambda b: b["bot"] - b["top"], reverse=True)
    for b in sands[:3]:
        b["hc"] = True
    return beds


def blocky(md, beds, key, rng):
    """Piecewise-constant bed response for one property index."""
    out = np.empty_like(md)
    for b in beds:
        lo, hi = PROPS[b["lith"]][key]
        v = float(rng.uniform(lo, hi))
        if b["hc"]:
            if key == 1:
                v -= 0.11                      # light hydrocarbon lowers RHOB
            elif key == 2:
                v -= 0.11                      # and suppresses neutron
            elif key == 3:
                v *= float(rng.uniform(18.0, 34.0))
        m = (md >= b["top"]) & (md < b["bot"])
        out[m] = v
    out[md >= beds[-1]["bot"]] = out[md < beds[-1]["bot"]][-1]
    return out


def logify(base, sd, rng, res_m=0.55, log_space=False):
    """Tool-resolution smoothing plus correlated high-frequency noise."""
    a = np.log10(base) if log_space else base
    a = gaussian_filter1d(a, max(res_m / STEP / 2.355, 0.6))
    n = gaussian_filter1d(rng.normal(size=a.size), 1.7)
    n *= sd / (n.std() + 1e-12)
    a = a + n
    return 10.0 ** a if log_space else a


def nphi_on_rhob(nphi):
    """Map neutron porosity onto the shared bulk-density axis."""
    f = (NPHI_LIM[0] - nphi) / (NPHI_LIM[0] - NPHI_LIM[1])
    return RHOB_LIM[0] + f * (RHOB_LIM[1] - RHOB_LIM[0])


def runs(flag, md):
    """Contiguous True intervals of ``flag`` as (top, bottom) depth pairs."""
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = md[i]
        elif not f and start is not None:
            out.append((start, md[i]))
            start = None
    if start is not None:
        out.append((start, md[-1]))
    return out


def header_axes(fig, rect, title):
    ax = axes_inches(fig, rect)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(RULE)
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=7.0,
            fontweight="bold", color=INK)
    return ax


def header_row(ax, y, name, lo, hi, color, ls="-"):
    ax.plot([0.32, 0.68], [y, y], color=color, ls=ls, lw=1.2, clip_on=False)
    ax.text(0.5, y + 0.055, name, ha="center", va="bottom", fontsize=6.1,
            color=INK)
    ax.text(0.02, y, lo, ha="left", va="center", fontsize=6.0, color=SUB)
    ax.text(0.98, y, hi, ha="right", va="center", fontsize=6.0, color=SUB)


def header_swatch(ax, x, y, label, fc, hatch=None, w=0.17, h=0.14, fs=6.0):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=fc, hatch=hatch,
                           edgecolor=RULE, linewidth=0.5))
    ax.text(x + w + 0.05, y, label, ha="left", va="center", fontsize=fs,
            color=SUB)


def track_axes(fig, rect, first=False):
    ax = axes_inches(fig, rect)
    ax.set_ylim(BASE, TOP)
    ax.tick_params(labelsize=6.6, length=2.4, width=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color(RULE)
    ax.yaxis.set_major_locator(plt.MultipleLocator(50.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10.0))
    if first:
        ax.set_ylabel("Depth (m MD)", fontsize=8.0)
    else:
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", color=GRID, lw=0.4, alpha=0.8)
    return ax


def main():
    set_pub_style()
    rng = np.random.default_rng(4021)

    md = np.arange(TOP, BASE + STEP, STEP)
    beds = build_beds(rng)

    gr = np.clip(logify(blocky(md, beds, 0, rng), 5.2, rng), *GR_LIM)
    rhob = np.clip(logify(blocky(md, beds, 1, rng), 0.018, rng, res_m=0.35),
                   RHOB_LIM[0] + 0.01, RHOB_LIM[1] - 0.01)
    nphi = np.clip(logify(blocky(md, beds, 2, rng), 0.014, rng, res_m=0.40),
                   NPHI_LIM[1] + 0.01, NPHI_LIM[0] - 0.01)
    rt = np.clip(logify(blocky(md, beds, 3, rng), 0.055, rng, res_m=0.75,
                        log_space=True), *RES_LIM)
    inv = np.where(np.array([any(b["hc"] and b["top"] <= d < b["bot"]
                                 for b in beds) for d in md]), 0.11, 1.35)
    rxo = np.clip(logify(rt * inv, 0.05, rng, res_m=0.45, log_space=True),
                  *RES_LIM)

    # ---- petrophysics, all from the curves that are actually drawn ---------
    gr_clean, gr_shale = np.percentile(gr, [8.0, 92.0])
    vsh = np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)
    phid = (2.65 - rhob) / (2.65 - 1.0)
    phit = np.clip(0.5 * (phid + nphi), 0.0, 0.40)
    phie = np.clip(phit - 0.26 * vsh, 0.005, 0.40)
    sw = np.clip((RW / (phie ** ARCHIE_M * rt)) ** (1.0 / ARCHIE_N), 0.04, 1.0)
    res_flag = (vsh < 0.40) & (phie > 0.085)
    pay = res_flag & (sw < 0.55)
    wet = res_flag & ~pay

    thick = np.full_like(md, STEP)
    ntg = float(thick[pay].sum() / (md[-1] - md[0]))
    phi_pay = float(np.average(phie[pay], weights=thick[pay])) if pay.any() else 0.0
    sw_pay = float(np.average(sw[pay], weights=thick[pay])) if pay.any() else 0.0
    zones = [z for z in runs(pay, md) if z[1] - z[0] > 2.0]

    # ---- canvas ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    widths = [1.52, 1.52, 1.78, 0.76, 1.46, 0.70]
    titles = ["Gamma ray", "Resistivity", "Density - neutron", "Lithology",
              "Volumetric analysis", "Pay"]
    gap, x0, y_b, y_t = 0.24, 0.94, 0.80, 4.86
    xs, x = [], x0
    for w in widths:
        xs.append(x)
        x += w + gap

    heads = [header_axes(fig, (xx, 4.94, w, 0.80), t)
             for xx, w, t in zip(xs, widths, titles)]
    axes = [track_axes(fig, (xx, y_b, w, y_t - y_b), first=(i == 0))
            for i, (xx, w) in enumerate(zip(xs, widths))]
    ax_gr, ax_res, ax_dn, ax_lith, ax_vol, ax_pay = axes

    # pay bands echoed across the curve tracks
    for ax in (ax_gr, ax_res, ax_dn, ax_vol):
        for z0, z1 in zones:
            ax.axhspan(z0, z1, facecolor=HC, alpha=0.09, zorder=0)

    # ---- track 1: gamma ray with clean / shale baselines -------------------
    ax_gr.set_xlim(*GR_LIM)
    ax_gr.set_xticks([0, 50, 100, 150])
    ax_gr.set_xlabel("GR (API)", fontsize=7.4)
    gr_cut = gr_clean + 0.40 * (gr_shale - gr_clean)     # the Vsh = 0.4 cutoff
    ax_gr.fill_betweenx(md, gr_cut, gr, where=gr < gr_cut,
                        facecolor=LITH_FC["sand"], alpha=0.95, lw=0)
    ax_gr.fill_betweenx(md, gr_cut, gr, where=gr >= gr_cut,
                        facecolor=LITH_FC["shale"], alpha=0.95, lw=0)
    for v, ls in ((gr_clean, (0, (3, 2))), (gr_shale, (0, (3, 2)))):
        ax_gr.axvline(v, color=MUTED, lw=0.7, ls=ls)
    ax_gr.axvline(gr_cut, color=RULE, lw=0.8)
    ax_gr.plot(gr, md, color=GR_C, lw=0.55)

    # ---- track 2: dual resistivity, crossover fill -------------------------
    ax_res.set_xscale("log")
    ax_res.set_xlim(*RES_LIM)
    ax_res.set_xticks([0.2, 2.0, 20.0, 200.0, 2000.0])
    ax_res.set_xticklabels(["0.2", "2", "20", "200", "2k"])
    ax_res.xaxis.set_minor_locator(plt.NullLocator())
    ax_res.set_xlabel("Resistivity (ohm-m)", fontsize=7.4)
    ax_res.fill_betweenx(md, rxo, rt, where=rt > rxo * 1.5, facecolor=HC,
                         alpha=0.30, lw=0)
    ax_res.fill_betweenx(md, rt, rxo, where=rxo > rt * 1.5, facecolor=WATER,
                         alpha=0.18, lw=0)
    ax_res.plot(rxo, md, color=RES_S, lw=0.5, ls=(0, (3, 1.6)))
    ax_res.plot(rt, md, color=RES_D, lw=0.6)

    # ---- track 3: density-neutron overlay with crossover -------------------
    ax_dn.set_xlim(*RHOB_LIM)
    ax_dn.set_xticks([2.0, 2.4, 2.8])
    ax_dn.set_xlabel(r"RHOB ($g/cm^3$)  /  NPHI (v/v)", fontsize=7.4)
    nx = nphi_on_rhob(nphi)
    ax_dn.fill_betweenx(md, rhob, nx, where=nx > rhob,
                        facecolor=LITH_FC["sand"], alpha=0.95, lw=0)
    ax_dn.fill_betweenx(md, nx, rhob, where=nx <= rhob,
                        facecolor=LITH_FC["shale"], alpha=0.75, lw=0)
    ax_dn.plot(rhob, md, color=RHOB_C, lw=0.55)
    ax_dn.plot(nx, md, color=NPHI_C, lw=0.5, ls=(0, (3, 1.6)))

    # ---- track 4: lithology column -----------------------------------------
    ax_lith.set_xlim(0.0, 1.0)
    ax_lith.set_xticks([])
    ax_lith.set_xlabel("Interpreted", fontsize=7.4)
    for b in beds:
        ax_lith.add_patch(Rectangle(
            (0.0, b["top"]), 1.0, b["bot"] - b["top"],
            facecolor=LITH_FC[b["lith"]], hatch=LITH_HATCH[b["lith"]],
            edgecolor="#8a8a8a", linewidth=0.35))

    # ---- track 5: volumetric analysis --------------------------------------
    ax_vol.set_xlim(0.0, 1.0)
    ax_vol.set_xticks([0.0, 0.5, 1.0])
    ax_vol.set_xlabel("Volume (v/v)", fontsize=7.4)
    c1 = vsh
    c2 = np.clip(1.0 - phie, c1, 1.0)
    c3 = np.clip(c2 + phie * sw, c2, 1.0)
    ax_vol.fill_betweenx(md, 0.0, c1, facecolor=LITH_FC["shale"], lw=0)
    ax_vol.fill_betweenx(md, c1, c2, facecolor=MATRIX, lw=0)
    ax_vol.fill_betweenx(md, c2, c3, facecolor=WATER, alpha=0.85, lw=0)
    ax_vol.fill_betweenx(md, c3, 1.0, facecolor=HC, lw=0)

    # ---- track 6: pay flag --------------------------------------------------
    ax_pay.set_xlim(0.0, 1.0)
    ax_pay.set_xticks([])
    ax_pay.set_xlabel("Flag", fontsize=7.4)
    ax_pay.add_patch(Rectangle((0.0, TOP), 1.0, BASE - TOP,
                               facecolor="#f0f0f0", lw=0))
    for z0, z1 in runs(wet, md):
        ax_pay.add_patch(Rectangle((0.0, z0), 1.0, z1 - z0, facecolor=WATER,
                                   alpha=0.85, lw=0))
    for z0, z1 in runs(pay, md):
        ax_pay.add_patch(Rectangle((0.0, z0), 1.0, z1 - z0, facecolor=HC,
                                   lw=0))
    for k, (z0, z1) in enumerate(zones):
        ax_pay.text(0.5, 0.5 * (z0 + z1), "ABCDEF"[k], ha="center",
                    va="center", fontsize=6.4, fontweight="bold",
                    color="white")

    # ---- headers ------------------------------------------------------------
    header_row(heads[0], 0.36, "GR", "0", "150", GR_C)
    heads[0].text(0.5, 0.10, "baselines dashed, cutoff solid", ha="center",
                  va="center", fontsize=6.0, color=MUTED)
    header_row(heads[1], 0.46, "RT deep", "0.2", "2000", RES_D)
    header_row(heads[1], 0.14, "RXO shallow", "0.2", "2000", RES_S,
               ls=(0, (3, 1.6)))
    header_row(heads[2], 0.46, r"RHOB ($g/cm^3$)", "1.95", "2.95", RHOB_C)
    header_row(heads[2], 0.14, "NPHI (v/v)", "0.45", "-0.15", NPHI_C,
               ls=(0, (3, 1.6)))
    for i, k in enumerate(("sand", "shale", "lime")):
        header_swatch(heads[3], 0.06, 0.58 - 0.22 * i, LITH_NAME[k],
                      LITH_FC[k], LITH_HATCH[k])
    for i, (lab, fc) in enumerate([("Vsh", LITH_FC["shale"]),
                                   ("Matrix", MATRIX),
                                   ("Water", WATER), ("HC", HC)]):
        header_swatch(heads[4], 0.06 + 0.48 * (i % 2), 0.52 - 0.28 * (i // 2),
                      lab, fc, w=0.12)
    for i, (lab, fc) in enumerate([("Pay", HC), ("Wet", WATER),
                                   ("None", "#f0f0f0")]):
        header_swatch(heads[5], 0.08, 0.58 - 0.22 * i, lab, fc, w=0.26,
                      h=0.13)

    savefig(fig, "810_wlg_composite_log.png")


if __name__ == "__main__":
    main()
