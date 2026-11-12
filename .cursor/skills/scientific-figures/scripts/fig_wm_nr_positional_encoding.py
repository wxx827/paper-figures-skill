"""Figure 1154 - Positional encoding: what each frequency band buys you.

One frequency ladder drives all three panels.  Band ``k`` of the encoding is

    gamma_k(p) = [sin(2^k pi p), cos(2^k pi p)],   f_k = F0 * 2^k   (cycles
                                                                    per view)

(a) draws the first eight bands of that ladder.
(b) passes ONE scene through a Butterworth low-pass whose cutoff is the top
    band admitted at that L, f_L = F0 * 2^(L-1), so the four tiles are four
    views of the same array and every PSNR is measured against the same
    ground truth.
(c) turns the same f_L into the finest recoverable half-wavelength in pixels,
    and marks the four tiles on it.

Change F0, L_SHOWN or the scene and all three panels plus every label follow:
each number printed is an f-string over these arrays.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_palette, add_panel_label
from _wm_scenes import build

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
HAIR, CARD = "#d5d5d5", "#f5f5f5"
ACCENT = "#134a76"
SEED = 1559  # original id; figure number is 1154

F0 = 0.5                      # cycles per view width of band k = 0
N_BAND = 8                    # bands drawn in panel (a)
L_SHOWN = (2, 4, 8, 12)       # encodings rendered in panel (b)
TILE = (180, 240)             # rows, cols
ROMAN = ("i", "ii", "iii", "iv")


def f_band(k):
    return F0 * 2.0 ** np.asarray(k, dtype=float)


def f_cut(L):
    """Top band admitted by an encoding with L bands (k = 0 .. L-1)."""
    return f_band(np.asarray(L) - 1)


def band_limit(img, cut, order=3):
    """Isotropic Butterworth low-pass, cutoff in cycles per view width."""
    h, w = img.shape[:2]
    fx = np.fft.fftfreq(w) * w
    fy = np.fft.fftfreq(h) * h * (w / h)
    fr = np.hypot(*np.meshgrid(fx, fy))
    gain = 1.0 / (1.0 + (fr / cut) ** (2 * order))
    out = np.dstack([np.real(np.fft.ifft2(np.fft.fft2(img[..., c]) * gain))
                     for c in range(3)])
    return np.clip(out, 0.0, 1.0)


def psnr(a, b):
    return 10.0 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))


def main():
    set_pub_style(base_font=8.5)
    gt = build("road", seed=SEED, shape=TILE, t=0.22,
               cam=dict(dof=1.0, sharpen=1.15))["rgb"]
    recon = [band_limit(gt, f_cut(L)) for L in L_SHOWN]
    q = [psnr(r, gt) for r in recon]
    nyq = TILE[1] / 2.0                       # cycles per view width
    exact = [f_cut(L) >= nyq for L in L_SHOWN]
    tag = ["= ground truth" if e else f"PSNR {v:.1f} dB"
           for e, v in zip(exact, q)]
    # derive the gap from the ROUNDED numbers that are actually printed, so a
    # reader subtracting the two tile captions lands on the same value
    shown = [round(v, 1) for v in q]
    last_gain = shown[2] - shown[1]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.98, 1.0],
                          height_ratios=[1.28, 1.0], left=0.058, right=0.978,
                          top=0.845, bottom=0.088, wspace=0.20, hspace=0.42)
    axw = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[:, 1])
    inner = gs[1, 0].subgridspec(1, 4, wspace=0.055)
    axt = [fig.add_subplot(inner[0, i]) for i in range(4)]

    # ---- (a) the frequency ladder -----------------------------------------
    cols = get_palette("cb_sequential_blue", N_BAND + 3)[3:]
    x = np.linspace(0.0, 1.0, 6000)
    step = 1.0
    for k in range(N_BAND):
        y0 = (N_BAND - 1 - k) * step
        fk = float(f_band(k))
        # a full-width axhline would run straight through the row labels
        # sitting in the left margin, so keep the baseline inside the data
        axw.plot([0.0, 1.0], [y0, y0], color="#ededed", lw=0.5, zorder=1)
        axw.plot(x, y0 + 0.40 * np.cos(2 * np.pi * fk * x), color=cols[k],
                 lw=0.7, alpha=0.50, zorder=2)
        axw.plot(x, y0 + 0.40 * np.sin(2 * np.pi * fk * x), color=cols[k],
                 lw=1.05, zorder=3)
        # ".0f" would print 240/64 as "4 px" and break the reader's division
        axw.text(-0.020, y0, f"$k$ = {k}   {fk:g} cyc   $\\lambda$ = "
                 f"{TILE[1] / fk:.4g} px", ha="right", va="center",
                 fontsize=6.9, color=INK if k in (0, N_BAND - 1) else SUB,
                 zorder=5)
    axw.text(-0.372, -0.58, "solid  $\\sin(2^k\\pi p)$\n"
             "faint   $\\cos(2^k\\pi p)$", ha="left", va="top", fontsize=6.9,
             color=SUB, linespacing=1.6, zorder=9,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=HAIR,
                       lw=0.5, alpha=0.92))
    axw.set_xlim(-0.385, 1.004)
    axw.set_ylim(-1.12, (N_BAND - 1) * step + 0.62)
    axw.set_yticks([])
    axw.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axw.set_xticklabels(["0", "", "0.5", "", "1"])
    axw.set_xlabel("normalised position $p$ across the view", fontsize=8.0,
                   labelpad=1.5)
    axw.xaxis.set_label_coords(0.62, -0.075)
    axw.spines["left"].set_visible(False)
    axw.spines["bottom"].set_bounds(0.0, 1.0)
    axw.tick_params(labelsize=7.0, length=2.4)
    axw.set_title(f"{N_BAND} bands span {f_band(0):g} to "
                  f"{f_band(N_BAND - 1):g} cycles per view", fontsize=8.8,
                  fontweight="bold", pad=4.0)
    add_panel_label(axw, "(a)", x=-0.078, y=1.008)

    # ---- (b) the same scene at four encoding lengths ----------------------
    for i, (ax, L, img) in enumerate(zip(axt, L_SHOWN, recon)):
        ax.imshow(img, interpolation="bilinear", zorder=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#9aa7b2")
            s.set_linewidth(0.7)
        ax.text(0.035, 0.955, f"({ROMAN[i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, fontweight="bold",
                color="white", zorder=5)
        ax.set_xlabel(f"$L$ = {L}   $f_L$ = {f_cut(L):g} cyc\n{tag[i]}",
                      fontsize=7.2, color=SUB, labelpad=2.5, linespacing=1.5)
    axt[0].set_title(f"Detail stops arriving once $f_L$ reaches Nyquist "
                     f"({nyq:g} cyc)", fontsize=8.8, fontweight="bold",
                     pad=4.0, loc="left")
    add_panel_label(axt[0], "(b)", x=-0.235, y=1.015)

    # ---- (c) band count vs recoverable detail -----------------------------
    Ls = np.arange(1, 15)
    detail = TILE[1] / (2.0 * f_cut(Ls))          # half-wavelength in pixels
    sub_pixel = Ls[detail < 1.0][0]
    axb.fill_between(Ls, 1e-3, np.minimum(detail, 1.0), color="#eef3f7", lw=0,
                     zorder=1)
    axb.plot(Ls, detail, color=ACCENT, lw=1.6, marker="o", ms=3.2,
             mfc="white", mec=ACCENT, mew=0.9, zorder=4)
    axb.axhline(1.0, color="#8a8a8a", lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    axb.text(14.2, 1.16, "one pixel", ha="right", va="bottom", fontsize=6.9,
             color=SUB, zorder=6)
    axb.axvspan(sub_pixel - 0.5, 14.5, color="#f0f0f0", lw=0, zorder=0)
    axb.text(sub_pixel + 0.25, 0.0032, f"bands past $L$ = {sub_pixel}\n"
             "are finer than a pixel", ha="left", va="bottom", fontsize=6.9,
             color=SUB, linespacing=1.5, zorder=6)

    dsh = TILE[1] / (2.0 * f_cut(np.array(L_SHOWN)))
    axb.plot(L_SHOWN, dsh, ls="none", marker="D", ms=5.4, mfc=ACCENT,
             mec="white", mew=0.8, zorder=7)
    for i, (L, d) in enumerate(zip(L_SHOWN, dsh)):
        dy, dx, ha = ((1.85, 0.45, "left") if i < 2 else
                      (0.36, 0.45, "left") if i == 2 else
                      (2.90, -0.45, "right"))
        axb.annotate(f"({ROMAN[i]}) {tag[i]}", xy=(L, d),
                     xytext=(L + dx, d * dy), fontsize=6.9, color=ACCENT,
                     ha=ha, va="center", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.22", fc="white",
                               ec="#b9c9d8", lw=0.5, alpha=0.94),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                     shrinkA=1, shrinkB=3))
    axb.set_yscale("log")
    axb.set_xlim(0.4, 14.5)
    axb.set_ylim(2.2e-3, 4.0e2)
    axb.set_xticks([1, 4, 8, 12])
    axb.set_xlabel("number of bands $L$", fontsize=8.0)
    axb.set_ylabel("finest recoverable detail (pixels)", fontsize=8.0)
    axb.set_title("Detail scale halves with every band added", fontsize=8.8,
                  fontweight="bold", pad=4.0)
    axb.tick_params(labelsize=7.0, length=2.4)
    axb.grid(True, which="major", lw=0.45, color="#ececec")
    axb.set_axisbelow(True)
    add_panel_label(axb, "(c)", x=-0.160, y=1.008)

    # ---- headline ---------------------------------------------------------
    savefig(fig, "1154_wm_nr_positional_encoding.png")


if __name__ == "__main__":
    main()
