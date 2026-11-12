"""Quiet-standing balance: plantar pressure, centre-of-pressure path, sway
ellipse and the frequency content of the sway velocity.

The main panel is a 5 mm sensor grid, so the pressure map keeps the blocky
resolution a real mat delivers, with the analytic foot outline drawn over it.
The COP path is the same signal in all three panels - the ellipse panel is a
zoom of the squiggle between the feet, and the spectrum is the Welch estimate
of its velocity - so the summary statistics can only agree with the picture.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from scipy.signal import butter, filtfilt, lfilter, welch
from scipy.stats import chi2
from pubstyle import (set_pub_style, savefig, style_axes, get_cmap,
                      get_palette, COLORS)
from figkit import safe_colorbar, panel_labels_inches, shared_limits

AP_C = "#b23a48"         # anterior-posterior (house red, the larger sway)
ML_C = "#3b5b92"         # medio-lateral (house blue)

FS = 100.0               # COP sampling rate (Hz)
DURATION = 30.0          # trial length (s)
CELL_MM = 5.0            # pressure mat cell size
FOOT_X = 54.0            # half distance between foot axes (mm)
TOE_OUT_DEG = 8.0
LOAD = {"left": 0.45, "right": 0.55}     # share of body weight per foot

# (u, v, sigma_u, sigma_v, kPa) in foot-local millimetres
REGIONS = [(0.0, -80.0, 20.0, 24.0, 195.0),      # heel
           (-16.0, 58.0, 13.0, 17.0, 165.0),     # 1st metatarsal head
           (3.0, 63.0, 15.0, 15.0, 140.0),       # 2nd-3rd metatarsal heads
           (23.0, 52.0, 13.0, 15.0, 100.0),      # 5th metatarsal head
           (-20.0, 96.0, 11.0, 12.0, 115.0),     # hallux
           (9.0, 92.0, 15.0, 9.0, 48.0),         # lesser toes
           (22.0, -5.0, 10.0, 32.0, 42.0)]       # lateral midfoot band

# Foot silhouette as a smooth union of ellipses: (u, v, ru, rv)
LOBES = [(0.0, -80.0, 29.0, 31.0),               # heel
         (5.0, -25.0, 26.0, 46.0),               # midfoot, arched medially
         (0.0, 45.0, 41.0, 44.0),                # forefoot
         (-5.0, 88.0, 35.0, 28.0),               # toe box
         (-21.0, 103.0, 15.0, 14.0)]             # hallux tip
BLEND_MM = 8.0                                   # fillet radius of the union


def foot_local(x, y, side):
    """Rotate global mm coordinates into the foot's own frame."""
    sgn = -1.0 if side == "left" else 1.0
    a = np.deg2rad(sgn * TOE_OUT_DEG)
    u, v = x - sgn * FOOT_X, y
    return u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)


def foot_shape(u, v):
    """Implicit foot outline: positive inside, in millimetres.

    A hard ``max`` over the lobes leaves visible creases where they meet, so
    the union is taken with a log-sum-exp smooth maximum, which fillets the
    joints by roughly ``BLEND_MM``.
    """
    f = np.stack([min(ru, rv) * (1.0 - np.hypot((u - cu) / ru, (v - cv) / rv))
                  for cu, cv, ru, rv in LOBES])
    top = f.max(axis=0)
    return top + BLEND_MM * np.log(
        np.exp((f - top) / BLEND_MM).sum(axis=0))


def pressure_field(x, y):
    p = np.zeros_like(x)
    inside = np.zeros_like(x, dtype=bool)
    for side, share in LOAD.items():
        u, v = foot_local(x, y, side)
        f = foot_shape(u, v)
        soft = np.clip(f / 6.0, 0.0, 1.0)
        q = sum(a * np.exp(-0.5 * (((u - cu) / su) ** 2 + ((v - cv) / sv) ** 2))
                for cu, cv, su, sv, a in REGIONS)
        p = p + 2.0 * share * q * soft
        inside |= f > 0.0
    return np.where(inside, p, np.nan), inside


def cop_signal(rng):
    """Leaky-integrated noise, low-passed the way real postural sway is."""
    n = int(DURATION * FS)
    t = np.arange(n) / FS
    b, a = butter(4, 2.5 / (FS / 2), btype="low")
    ml = filtfilt(b, a, lfilter([1.0], [1.0, -0.988],
                                rng.normal(0.0, 1.0, n)))
    ap = filtfilt(b, a, lfilter([1.0], [1.0, -0.992],
                                rng.normal(0.0, 1.0, n)))
    ml = (ml - ml.mean()) / ml.std() * 6.5
    ap = (ap - ap.mean()) / ap.std() * 9.5
    ml = ml + 3.0 * np.sin(2 * np.pi * 0.06 * t)
    ap = ap + 4.0 * np.sin(2 * np.pi * 0.045 * t + 1.1)
    return t, ml + 4.0, ap - 6.0


def confidence_ellipse(ml, ap, conf=0.95):
    cov = np.cov(np.vstack([ml, ap]))
    val, vec = np.linalg.eigh(cov)
    k = chi2.ppf(conf, 2)
    a, b = np.sqrt(k * val[::-1])
    ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
    return a, b, ang, np.pi * a * b


def coloured_path(ax, x, y, t, cmap, lw=0.9, zorder=5):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap,
                        norm=mpl.colors.Normalize(t[0], t[-1]), lw=lw,
                        zorder=zorder)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


def main():
    set_pub_style()
    rng = np.random.default_rng(514)
    t, ml, ap = cop_signal(rng)
    time_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cop_time", get_palette("cb_sequential_warm")[2:])

    xe = np.arange(-115.0, 115.1, CELL_MM)
    ye = np.arange(-130.0, 140.1, CELL_MM)
    xc, yc = np.meshgrid(xe + CELL_MM / 2, ye + CELL_MM / 2)
    press, inside = pressure_field(xc, yc)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.26, hspace=0.42,
                          left=0.055, right=0.945, top=0.880, bottom=0.100)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # ---- (a) pressure map + COP path ---------------------------------------
    im = ax_m.imshow(press, origin="lower", interpolation="nearest",
                     extent=(xe[0], xe[-1] + CELL_MM, ye[0],
                             ye[-1] + CELL_MM),
                     cmap=get_cmap("cb_sequential_blue"), zorder=2)
    gx, gy = np.meshgrid(np.linspace(-118.0, 118.0, 460),
                         np.linspace(-132.0, 142.0, 520))
    for side in LOAD:
        u, v = foot_local(gx, gy, side)
        ax_m.contour(gx, gy, foot_shape(u, v), levels=[0.0],
                     colors=[COLORS["rule"]], linewidths=0.8, zorder=4)
    coloured_path(ax_m, ml, ap, t, time_cmap, lw=0.9, zorder=6)
    ax_m.plot(ml[0], ap[0], marker="o", ms=4.0, mfc="white",
              mec=COLORS["ink"], mew=0.9, zorder=7)
    ax_m.plot(ml[-1], ap[-1], marker="s", ms=4.0, mfc=COLORS["ink"],
              mec=COLORS["ink"], mew=0.9, zorder=7)
    ax_m.annotate("start", xy=(ml[0], ap[0]), xytext=(ml[0] + 22.0,
                                                      ap[0] + 26.0),
                  fontsize=6.4, color=COLORS["ink"], zorder=7,
                  arrowprops=dict(arrowstyle="-", lw=0.6,
                                  color=COLORS["muted"]))
    ax_m.annotate("end", xy=(ml[-1], ap[-1]), xytext=(ml[-1] + 22.0,
                                                      ap[-1] - 30.0),
                  fontsize=6.4, color=COLORS["ink"], zorder=7,
                  arrowprops=dict(arrowstyle="-", lw=0.6,
                                  color=COLORS["muted"]))

    share = {s: np.nansum(np.where(np.isfinite(press), press, 0.0)
                          * (foot_shape(*foot_local(xc, yc, s)) > 0.0))
             for s in LOAD}
    total = sum(share.values())
    ax_m.text(0.985, 0.015, f"left {100 * share['left'] / total:.0f} %  |  "
              f"right {100 * share['right'] / total:.0f} % of total load",
              transform=ax_m.transAxes, ha="right", va="bottom", fontsize=6.6,
              color=COLORS["sub"])
    ax_m.set_xlim(xe[0], xe[-1] + CELL_MM)
    ax_m.set_ylim(ye[0], ye[-1] + CELL_MM)
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("Medio-lateral position (mm)", fontsize=8.0)
    ax_m.set_ylabel("Anterior-posterior position (mm)", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0)
    ax_m.set_title("Plantar pressure and the COP path it hides",
                   fontsize=8.5, fontweight="bold")
    style_axes(ax_m)
    safe_colorbar(fig, im, ax_m, label="plantar pressure (kPa)", width=0.11,
                  pad=0.08, shrink=0.85)

    # ---- (b) sway ellipse ---------------------------------------------------
    a, b, ang, area = confidence_ellipse(ml, ap)
    lc = coloured_path(ax_e, ml, ap, t, time_cmap, lw=0.7, zorder=4)
    ax_e.add_patch(Ellipse((ml.mean(), ap.mean()), 2 * a, 2 * b, angle=ang,
                           facecolor=AP_C, alpha=0.10, edgecolor=AP_C,
                           lw=1.2, zorder=5))
    ax_e.plot(ml.mean(), ap.mean(), marker="+", ms=6, color=AP_C, mew=1.2,
              zorder=6)
    lim = shared_limits([ml - ml.mean(), ap - ap.mean()], pad=0.22,
                        symmetric=True)
    ax_e.set_xlim(ml.mean() + lim[0], ml.mean() + lim[1])
    ax_e.set_ylim(ap.mean() + lim[0], ap.mean() + lim[1])
    ax_e.set_aspect("equal")
    path_mm = float(np.sum(np.hypot(np.diff(ml), np.diff(ap))))
    ax_e.text(0.03, 0.97, f"95 % ellipse {area:.0f} mm$^2$\n"
              f"path {path_mm:.0f} mm\nmean speed "
              f"{path_mm / DURATION:.1f} mm/s", transform=ax_e.transAxes,
              ha="left", va="top", fontsize=6.6, color=COLORS["ink"],
              linespacing=1.45)
    ax_e.set_xlabel("Medio-lateral (mm)", fontsize=7.5)
    ax_e.set_ylabel("Anterior-posterior (mm)", fontsize=7.5)
    ax_e.tick_params(labelsize=6.5)
    ax_e.set_title("COP sway ellipse", fontsize=8.5, fontweight="bold")
    style_axes(ax_e)
    safe_colorbar(fig, lc, ax_e, label="time (s)", width=0.10, pad=0.07,
                  labelsize=6.2)

    # ---- (c) sway velocity spectrum ----------------------------------------
    nper = int(8 * FS)
    curves = {}
    for name, sig, colour in (("AP", ap, AP_C), ("ML", ml, ML_C)):
        v = np.gradient(sig, 1.0 / FS)
        f, pxx = welch(v, fs=FS, nperseg=nper, noverlap=nper // 2)
        band = (f >= 0.05) & (f <= 5.0)
        c = np.cumsum(pxx[band]) / np.sum(pxx[band])
        show = (f > 0.0) & (f <= 3.0)
        curves[name] = (f[show], pxx[show], colour,
                        float(np.interp(0.5, c, f[band])))
    for name, (f, pxx, colour, f50) in curves.items():
        ax_p.semilogy(f, pxx, color=colour, lw=1.3,
                      label=f"{name}, $f_{{50}}$ = {f50:.2f} Hz")
        ax_p.axvline(f50, color=colour, lw=0.8, ls=(0, (3.0, 2.2)), alpha=0.8)
    ax_p.set_xlim(0.0, 3.0)
    lo, hi = shared_limits([np.log10(c[1]) for c in curves.values()],
                           pad=0.10)
    ax_p.set_ylim(10.0 ** lo, 10.0 ** hi)
    ax_p.set_xlabel("Frequency (Hz)", fontsize=7.5)
    ax_p.set_ylabel("PSD ((mm/s)$^2$/Hz)", fontsize=7.5)
    ax_p.tick_params(labelsize=6.5)
    ax_p.legend(loc="lower left", frameon=False, fontsize=6.6)
    ax_p.set_title("Sway velocity spectrum", fontsize=8.5, fontweight="bold")
    style_axes(ax_p)

    panel_labels_inches(fig, [ax_m, ax_e, ax_p], dx=-0.42, dy=0.10)
    savefig(fig, "514_bio_balance_cop.png")


if __name__ == "__main__":
    main()
