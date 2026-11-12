"""Electrochemical characterisation panel: CV, Nyquist and Bode plots.

(a) Cyclic voltammograms at five scan rates, generated as a closed loop from a
    triangular potential sweep (faradaic peak + capacitive plateau, smoothed at
    the vertices), with a power-law inset that extracts the b-value.
(b) Nyquist plot of a Randles circuit (Rs + Rct||CPE + Warburg) for two
    electrodes, with the equivalent circuit drawn from patches.
(c) Bode magnitude and phase for the same two electrodes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

RATES = np.array([0.010, 0.020, 0.050, 0.100, 0.200])   # V s-1
E1, E2 = -0.20, 0.60                                    # V vs Ag/AgCl
EPA, EPC = 0.322, 0.242
CDL = 3.5e-3                                            # F cm-2

CELLS = [dict(name="pristine NiCo$_2$O$_4$", rs=3.1, rct=42.0, cdl=1.8e-4,
              sigma=26.0, col="#4c78a8"),
         dict(name="N-doped NiCo$_2$O$_4$", rs=2.6, rct=11.5, cdl=3.6e-4,
              sigma=14.0, col="#c0603f")]


def circ_smooth(y, frac=0.013):
    n = y.size
    k = max(3, int(frac * n))
    kern = np.exp(-0.5 * np.linspace(-3, 3, 2 * k + 1) ** 2)
    kern /= kern.sum()
    pad = np.concatenate([y[-k:], y, y[:k]])
    return np.convolve(pad, kern, mode="same")[k:-k]


def peak(u, w):
    """Asymmetric voltammetric peak: gaussian rise, diffusional tail."""
    out = np.where(u <= 0.0, np.exp(-0.5 * (u / w) ** 2),
                   1.0 / (1.0 + (np.abs(u) / (1.25 * w)) ** 2) ** 0.85)
    return out


def voltammogram(v, n=2400):
    """Closed CV loop for scan rate ``v`` (V s-1); returns E, j (mA cm-2)."""
    half = n // 2
    e_up = np.linspace(E1, E2, half, endpoint=False)
    e_dn = np.linspace(E2, E1, n - half, endpoint=False)
    e = np.concatenate([e_up, e_dn])
    d = np.concatenate([np.ones(half), -np.ones(n - half)])

    ip = 5.0 * v ** 0.78                       # mixed-control peak current
    jc = CDL * v * 1e3                         # capacitive plateau, mA cm-2
    j = np.where(d > 0, ip * peak(e - EPA, 0.055),
                 -0.92 * ip * peak(EPC - e, 0.058))
    j = j + d * jc
    return e, circ_smooth(j)


def randles(cell, w):
    zw = cell["sigma"] * (1.0 - 1.0j) / np.sqrt(w)
    y = 1.0 / (cell["rct"] + zw) + 1.0j * w * cell["cdl"]
    return cell["rs"] + 1.0 / y


def draw_circuit(ax):
    """Randles equivalent circuit sketched with plain patches."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.15, 4.35)     # padding keeps the sketch near-isotropic
    ax.axis("off")
    ax.patch.set_alpha(0.0)
    lk = dict(color="#333333", lw=0.8, solid_capstyle="butt")

    ax.plot([0.2, 1.4], [1.6, 1.6], **lk)
    ax.add_patch(Rectangle((1.4, 1.28), 1.15, 0.64, facecolor="white",
                           edgecolor="#333333", lw=0.8))
    ax.text(1.975, 1.6, "R$_s$", ha="center", va="center", fontsize=6)
    ax.plot([2.55, 3.15], [1.6, 1.6], **lk)
    ax.plot([3.15, 3.15], [0.62, 2.58], **lk)

    # upper branch: constant phase element
    ax.plot([3.15, 5.05], [2.58, 2.58], **lk)
    ax.plot([5.05, 5.05], [2.16, 3.00], **lk)
    ax.plot([5.55, 5.55], [2.16, 3.00], **lk)
    ax.plot([5.55, 7.45], [2.58, 2.58], **lk)
    ax.text(5.30, 3.12, "CPE", ha="center", va="bottom", fontsize=6)

    # lower branch: charge transfer resistance in series with Warburg
    ax.plot([3.15, 3.75], [0.62, 0.62], **lk)
    ax.add_patch(Rectangle((3.75, 0.30), 1.15, 0.64, facecolor="white",
                           edgecolor="#333333", lw=0.8))
    ax.text(4.325, 0.62, "R$_{ct}$", ha="center", va="center", fontsize=6)
    ax.plot([4.90, 5.55], [0.62, 0.62], **lk)
    ax.add_patch(Rectangle((5.55, 0.30), 0.80, 0.64, facecolor="white",
                           edgecolor="#333333", lw=0.8))
    ax.text(5.95, 0.62, "W", ha="center", va="center", fontsize=6)
    ax.plot([6.35, 7.45], [0.62, 0.62], **lk)

    ax.plot([7.45, 7.45], [0.62, 2.58], **lk)
    ax.plot([7.45, 8.6], [1.6, 1.6], **lk)


def main():
    set_pub_style(base_font=9)

    fig = plt.figure(figsize=(7.6, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0],
                          height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.28,
                          left=0.075, right=0.975, top=0.925, bottom=0.105)
    axc = fig.add_subplot(gs[:, 0])
    axn = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1])

    # ---------------- (a) cyclic voltammetry ------------------------------
    cmap = LinearSegmentedColormap.from_list(
        "cv", ["#9dc3d9", "#5b93b8", "#3f6f8f", "#8a5a3c", "#a8382f"])
    peaks = []
    for i, v in enumerate(RATES):
        e, j = voltammogram(v)
        col = cmap(i / (len(RATES) - 1))
        axc.plot(e, j, color=col, lw=1.05, zorder=4 + i,
                 label=f"{int(v * 1000)} mV s$^{{-1}}$")
        peaks.append(j.max())
        if i == len(RATES) - 1:
            top = j.max()
    peaks = np.array(peaks)

    axc.axhline(0.0, color="#bfbfbf", lw=0.6, zorder=1)
    for ep, col in ((EPA, "#a8382f"), (EPC, "#3f6f8f")):
        axc.axvline(ep, color=col, lw=0.6, ls=(0, (3.5, 2.5)), zorder=2,
                    alpha=0.7)
    axc.text(EPA + 0.015, 2.62, "$E_{pa}$ = 0.32 V", fontsize=6.8,
             color="#a8382f", ha="left", va="center", fontweight="bold")
    axc.text(EPC - 0.015, -2.28, "$E_{pc}$ = 0.24 V", fontsize=6.8,
             color="#3f6f8f", ha="right", va="center", fontweight="bold")
    axc.text(E2, 2.85, "$\\Delta E_p$ = 80 mV", fontsize=6.8, color="#4d4d4d",
             ha="right", va="top")

    e_hi, j_hi = voltammogram(RATES[-1])
    for idx, sgn in ((int(0.30 * e_hi.size), 1), (int(0.80 * e_hi.size), -1)):
        axc.annotate("", xy=(e_hi[idx + sgn * 22], j_hi[idx + sgn * 22]),
                     xytext=(e_hi[idx], j_hi[idx]),
                     arrowprops=dict(arrowstyle="-|>", color="#a8382f",
                                     lw=1.0, mutation_scale=8))
    axc.text(0.46, 1.45, "anodic", fontsize=6.6, color="#8a5a3c",
             ha="center", va="bottom", style="italic")
    axc.text(0.05, -1.55, "cathodic", fontsize=6.6, color="#3f6f8f",
             ha="center", va="top", style="italic")

    axc.set_xlim(E1 - 0.02, E2 + 0.02)
    axc.set_ylim(-2.55, 3.05)
    axc.set_xlabel("Potential (V vs. Ag/AgCl)")
    axc.set_ylabel("Current density (mA cm$^{-2}$)")
    axc.legend(loc="lower right", fontsize=6.5, ncol=1, handlelength=1.3,
               labelspacing=0.28, borderaxespad=0.4)
    add_panel_label(axc, "(a)", x=-0.115, y=1.01, size=10)

    ins = axc.inset_axes([0.085, 0.635, 0.30, 0.29])
    slope, icept = np.polyfit(np.log10(RATES), np.log10(peaks), 1)
    vg = np.logspace(-2.15, -0.6, 50)
    ins.plot(vg, 10 ** (icept + slope * np.log10(vg)), color="#4d4d4d",
             lw=0.8, ls=(0, (3, 2)))
    ins.plot(RATES, peaks, ls="none", marker="o", ms=3.0, mfc="white",
             mec="#a8382f", mew=0.9)
    ins.set_xscale("log")
    ins.set_yscale("log")
    ins.set_xlabel("$v$ (V s$^{-1}$)", fontsize=6, labelpad=1)
    ins.set_ylabel("$i_p$", fontsize=6, labelpad=1)
    ins.tick_params(labelsize=6.0, length=1.8, pad=1.2, which="both")
    ins.text(0.95, 0.10, f"$b$ = {slope:.2f}", transform=ins.transAxes,
             ha="right", va="bottom", fontsize=6, color="#a8382f",
             fontweight="bold")
    for s in ins.spines.values():
        s.set_linewidth(0.6)
    ins.spines["top"].set_visible(False)
    ins.spines["right"].set_visible(False)

    # ---------------- (b) Nyquist ----------------------------------------
    w = 2.0 * np.pi * np.logspace(-2, 5, 900)
    for cell in CELLS:
        z = randles(cell, w)
        axn.plot(z.real, -z.imag, color=cell["col"], lw=1.15,
                 label=cell["name"], zorder=4)
        axn.plot(z.real[::110], -z.imag[::110], ls="none", marker="o", ms=2.6,
                 mfc="white", mec=cell["col"], mew=0.7, zorder=5)
    axn.set_xlim(0, 92)
    axn.set_ylim(0, 92)
    axn.set_aspect("equal")
    axn.set_xlabel("Z' ($\\Omega$)", labelpad=1.5)
    axn.set_ylabel("–Z'' ($\\Omega$)", labelpad=1.5)
    axn.tick_params(labelsize=7)
    axn.text(58, 27, "45° Warburg diffusion", fontsize=6.4, color="#5c5c5c",
             ha="left", va="bottom", rotation=45, rotation_mode="anchor")
    axn.annotate("$R_s$", xy=(3.1, 0.0), xytext=(15.0, 15.0), fontsize=6.6,
                 color="#4d4d4d", ha="center",
                 arrowprops=dict(arrowstyle="-|>", color="#4d4d4d", lw=0.6,
                                 shrinkB=1.0, mutation_scale=6))
    axn.annotate("$R_{ct}$ = 11.5 $\\Omega$", xy=(9.0, 5.4), xytext=(18, 50),
                 fontsize=6.6, color="#c0603f", ha="left",
                 arrowprops=dict(arrowstyle="-|>", color="#c0603f", lw=0.6,
                                 shrinkB=1.0, mutation_scale=6))
    axn.legend(loc="upper left", fontsize=6.4, handlelength=1.2,
               borderaxespad=0.3, labelspacing=0.3)
    add_panel_label(axn, "(b)", x=-0.155, y=1.02, size=10)

    # ---------------- (c) Bode -------------------------------------------
    f = np.logspace(-2, 5, 900)
    axp = axb.twinx()
    for cell in CELLS:
        z = randles(cell, 2.0 * np.pi * f)
        axb.plot(f, np.abs(z), color=cell["col"], lw=1.15, zorder=4)
        axp.plot(f, -np.angle(z, deg=True), color=cell["col"], lw=0.9,
                 ls=(0, (3.5, 2.0)), zorder=3)
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(1e-2, 1e5)
    axb.set_ylim(1, 4e3)
    axb.set_xlabel("Frequency (Hz)", labelpad=1.5)
    axb.set_ylabel("|Z| ($\\Omega$)", labelpad=1.5)
    axb.tick_params(labelsize=7)
    axp.set_ylim(0, 90)
    axp.set_yticks([0, 30, 60, 90])
    axp.set_ylabel("Phase angle (°)", labelpad=2)
    axp.tick_params(labelsize=7, colors="#4d4d4d")
    axp.spines["right"].set_visible(True)
    axp.spines["right"].set_color("#4d4d4d")
    axp.spines["top"].set_visible(False)
    axb.legend(handles=[Line2D([], [], color="#5c5c5c", lw=1.1, label="|Z|"),
                        Line2D([], [], color="#5c5c5c", lw=0.9,
                               ls=(0, (3.5, 2.0)), label="phase")],
               loc="lower left", fontsize=6.4, ncol=2, handlelength=1.4,
               borderaxespad=0.35, columnspacing=1.0)
    add_panel_label(axb, "(c)", x=-0.155, y=1.02, size=10)
    circ = axb.inset_axes([0.55, 0.625, 0.435, 0.34])
    draw_circuit(circ)
    circ.text(4.4, 4.05, "Randles equivalent circuit", fontsize=6.2,
              color="#4d4d4d", ha="center", va="bottom")

    savefig(fig, "122_cv_eis_panel.png")


if __name__ == "__main__":
    main()
