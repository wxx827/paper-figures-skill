"""Campbell diagram of a rotor: excitation orders against structural modes.

The map is the measured vibration level over the (speed, frequency) plane.
Straight lines through the origin are the engine orders, horizontal lines the
measured natural frequencies, and every intersection inside the operating
range is a resonance - circled here and listed, sorted by the level the map
actually shows at that point, on the right.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator
from pubstyle import set_pub_style, savefig, add_colorbar, COLORS
from figkit import panel_label_inches

RPM = np.linspace(500.0, 6200.0, 260)
FREQ = np.linspace(5.0, 800.0, 420)
IDLE, REDLINE = 800.0, 6000.0
# natural frequency (Hz), half-width (Hz), amplification, label
MODES = [(118.0, 9.0, 8.0, "Bending 1"), (262.0, 13.0, 6.0, "Bending 2"),
         (455.0, 18.0, 4.5, "Torsion 1"), (640.0, 24.0, 3.2, "Casing")]
ORDERS = {1: 0.55, 2: 1.00, 3: 0.42, 4: 0.30}
LINE_W = 6.0                                   # order-line width in Hz


def amplification(f):
    h = np.ones_like(f)
    for fc, q, g, _ in MODES:
        h = h + g / (1.0 + ((f - fc) / q) ** 2)
    return h


def response(rng):
    """Vibration velocity (mm/s rms) on the (speed, frequency) grid."""
    R, F = np.meshgrid(RPM, FREQ)
    amp = np.full(R.shape, 0.006)
    for k, a in ORDERS.items():
        amp = amp + 0.10 * a * (R / 1000.0) ** 1.2 * amplification(F) * \
            np.exp(-0.5 * ((F - k * R / 60.0) / LINE_W) ** 2)
    ker = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2)
    ker /= ker.sum()
    noise = rng.lognormal(0.0, 0.12, size=amp.shape)
    for axis in (0, 1):
        noise = np.apply_along_axis(
            lambda v: np.convolve(v, ker, mode="same"), axis, noise)
    return amp * noise / noise.mean()


def crossings(amp_db):
    """Every order / mode intersection inside the operating range."""
    out = []
    for k in ORDERS:
        for fc, _, _, name in MODES:
            r = 60.0 * fc / k
            if not IDLE <= r <= REDLINE:
                continue
            i = int(np.argmin(np.abs(FREQ - fc)))
            j = int(np.argmin(np.abs(RPM - r)))
            out.append((k, name, fc, r, float(amp_db[i, j])))
    return sorted(out, key=lambda c: -c[4])


def main():
    set_pub_style()
    rng = np.random.default_rng(422)
    amp = response(rng)
    amp_db = 20.0 * np.log10(amp / 1e-3)        # dB re 1 um/s
    hits = crossings(amp_db)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.46], wspace=0.30,
                          left=0.062, right=0.955, top=0.885, bottom=0.105)
    ax = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[0, 1])

    # ---- (a) Campbell map -------------------------------------------------
    pm = ax.pcolormesh(RPM, FREQ, amp_db, cmap="magma", shading="gouraud",
                       vmin=np.percentile(amp_db, 2.0),
                       vmax=np.percentile(amp_db, 99.8), rasterized=True)
    for fc, _, _, name in MODES:
        ax.axhline(fc, color="#cccccc", lw=0.9, ls=(0, (5.0, 2.5)), zorder=4)
        ax.text(RPM[0] + 80.0, fc + 9.0, f"{name}   $f_n$ = {fc:.0f} Hz",
                fontsize=6.6, color="white", va="bottom", zorder=6)
    for k in ORDERS:
        ax.plot(RPM, k * RPM / 60.0, color="white", lw=0.9, alpha=0.9,
                zorder=5)
        r_end = min(RPM[-1], 60.0 * FREQ[-1] / k)
        ax.text(r_end - 60.0, k * r_end / 60.0 - 12.0, f"{k}X", ha="right",
                va="top", fontsize=7.2, fontweight="bold", color="white",
                zorder=6)
    for k, name, fc, r, lvl in hits:
        ax.plot([r], [fc], marker="o", ms=9.0, mfc="none",
                mec=COLORS["accent"], mew=1.5, zorder=8)

    ax.axvspan(RPM[0], IDLE, color="#1a1a1a", alpha=0.35, lw=0, zorder=3)
    ax.axvspan(REDLINE, RPM[-1], color="#1a1a1a", alpha=0.35, lw=0, zorder=3)
    ax.text(IDLE - 40.0, FREQ[-1] - 18.0, "below idle", rotation=90,
            ha="right", va="top", fontsize=6.4, color="white", zorder=6)
    ax.text(REDLINE + 40.0, FREQ[-1] - 18.0, "above redline", rotation=90,
            ha="left", va="top", fontsize=6.4, color="white", zorder=6)

    ax.set_xlim(RPM[0], RPM[-1])
    ax.set_ylim(FREQ[0], FREQ[-1])
    ax.set_xlabel("Rotor speed (rpm)")
    ax.set_ylabel("Frequency (Hz)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(COLORS["rule"])
    ax.tick_params(which="both", color=COLORS["rule"])
    ax.set_title("Order excitation against structural modes", fontsize=9.0,
                 fontweight="bold")
    add_colorbar(fig, pm, ax,
                 label="vibration velocity (dB re 1 $\\mu$m/s)")

    # ---- (b) critical speed list ------------------------------------------
    ax_t.set_xlim(0.0, 1.0)
    ax_t.set_ylim(0.0, 1.0)
    ax_t.axis("off")
    ax_t.set_title("Critical speeds in the operating range", fontsize=8.5,
                   fontweight="bold")
    cols = (0.045, 0.185, 0.640, 0.820, 0.975)
    heads = ("Order", "Mode", "Speed", "$f_n$", "Level")
    aligns = ("left", "left", "right", "right", "right")
    row_h = min(0.90 / (len(hits) + 1), 0.105)
    y = 0.95
    for x, h, a in zip(cols, heads, aligns):
        ax_t.text(x, y, h, ha=a, va="center", fontsize=6.8,
                  color=COLORS["sub"], fontweight="bold")
    ax_t.plot([0.02, 0.98], [y - row_h * 0.42] * 2, color=COLORS["grid"],
              lw=0.8)
    for i, (k, name, fc, r, lvl) in enumerate(hits):
        y = 0.95 - (i + 1) * row_h
        top = i == 0
        if i % 2 == 0:
            ax_t.add_patch(Rectangle((0.02, y - row_h * 0.42), 0.96,
                                     row_h * 0.84, facecolor="#f5f5f5",
                                     edgecolor="none", zorder=0))
        if top:
            ax_t.add_patch(Rectangle((0.02, y - row_h * 0.42), 0.96,
                                     row_h * 0.84, facecolor="#fbeff0",
                                     edgecolor=COLORS["accent"], lw=1.0,
                                     zorder=1))
        col = COLORS["accent"] if top else COLORS["ink"]
        vals = (f"{k}X", name, f"{r:.0f} rpm", f"{fc:.0f} Hz",
                f"{lvl:.0f} dB")
        for x, v, a in zip(cols, vals, aligns):
            ax_t.text(x, y, v, ha=a, va="center", fontsize=7.0, color=col,
                      fontweight="bold" if top else "normal", zorder=3)
    ax_t.text(0.02, 0.95 - (len(hits) + 1) * row_h - 0.03,
              f"Worst case: {hits[0][0]}X on the {hits[0][1]} mode at "
              f"{hits[0][3]:.0f} rpm.\nLevels are read from the map at the "
              "crossing point.", ha="left", va="top", fontsize=6.6,
              color=COLORS["muted"], linespacing=1.5)

    for a, lab in zip((ax, ax_t), ("(a)", "(b)")):
        panel_label_inches(fig, a, lab, dx=-0.40, dy=0.30)

    savefig(fig, "422_acu_campbell.png")


if __name__ == "__main__":
    main()
