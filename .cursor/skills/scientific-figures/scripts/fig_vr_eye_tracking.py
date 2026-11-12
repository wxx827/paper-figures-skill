"""One gaze recording, read three ways: heat map, scan path and dwell budget.

Everything on the page comes from a single stream of 120 Hz gaze samples.  The
heat map is that stream binned and smoothed; the numbered circles are the
fixations the samples cluster into, sized by their own dwell time; and the
area-of-interest shares are those same dwell times sorted by which rectangle
each fixation actually landed in - tested point by point, not assigned by the
script that generated the walk.

Because the assignment is measured rather than declared, a fixation that
drifts off a panel counts as background, which is why the dwell shares do not
add up to a tidy hundred percent.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import safe_colorbar, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GUIDE = "#808080"
GRID = "#d9d9d9"
HAIR = "#cccccc"
ACCENT = "#b23a48"
BLUE = "#3b5b92"

X_DEG, Y_DEG = 40.0, 30.0          # scene half-extents, degrees of arc
SAMPLE_HZ = 120.0
SACCADE_MS = 40.0
BLUR_DEG = 1.9
TARGET_AOI = "Menu panel"

AOIS = [
    ("Horizon band", -40.0, 14.0, 80.0, 12.0),
    ("Menu panel", -36.0, -8.0, 24.0, 20.0),
    ("Avatar", -4.0, -14.0, 16.0, 26.0),
    ("HUD readout", 16.0, 0.0, 20.0, 10.0),
    ("Hand controller", 18.0, -26.0, 18.0, 14.0),
]
VISIT_ORDER = ["Horizon band", "Menu panel", "Menu panel", "Avatar",
               "Menu panel", "HUD readout", "Avatar", "Hand controller",
               "Avatar", "Menu panel", "HUD readout", "Horizon band",
               "Avatar", "Menu panel", "Hand controller", "Avatar",
               "Menu panel", "Avatar", "HUD readout", "Menu panel",
               "Avatar", "Horizon band", "Hand controller", "Menu panel"]


def aoi_box(name):
    for rec in AOIS:
        if rec[0] == name:
            return rec[1:]
    raise KeyError(name)


def which_aoi(x, y):
    for name, x0, y0, w, h in AOIS:
        if (x0 <= x <= x0 + w) and (y0 <= y <= y0 + h):
            return name
    return None


def make_fixations(rng):
    fx, fy, dur = [], [], []
    for name in VISIT_ORDER:
        x0, y0, w, h = aoi_box(name)
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        fx.append(cx + rng.normal(0.0, 0.24 * w))
        fy.append(cy + rng.normal(0.0, 0.24 * h))
        dur.append(float(np.clip(rng.lognormal(5.42, 0.42), 110.0, 950.0)))
    fx = np.clip(np.array(fx), -X_DEG + 1.5, X_DEG - 1.5)
    fy = np.clip(np.array(fy), -Y_DEG + 1.5, Y_DEG - 1.5)
    return fx, fy, np.array(dur)


def gaze_stream(fx, fy, dur, rng):
    xs, ys = [], []
    for k in range(fx.size):
        n = max(int(round(dur[k] * SAMPLE_HZ / 1000.0)), 3)
        xs.append(fx[k] + rng.normal(0.0, 0.72, n))
        ys.append(fy[k] + rng.normal(0.0, 0.72, n))
        if k + 1 < fx.size:
            m = max(int(round(SACCADE_MS * SAMPLE_HZ / 1000.0)), 2)
            s = np.linspace(0.0, 1.0, m)
            xs.append(fx[k] + s * (fx[k + 1] - fx[k]) + rng.normal(0, 0.4, m))
            ys.append(fy[k] + s * (fy[k + 1] - fy[k]) + rng.normal(0, 0.4, m))
    return np.concatenate(xs), np.concatenate(ys)


def density(gx, gy, nx=320, ny=240):
    """Samples per square degree, expressed as a percentage of the recording."""
    h, xe, ye = np.histogram2d(gx, gy, bins=(nx, ny),
                               range=[[-X_DEG, X_DEG], [-Y_DEG, Y_DEG]])
    cell = (2 * X_DEG / nx) * (2 * Y_DEG / ny)
    sm = gaussian_filter(h.T, sigma=BLUR_DEG / (2 * Y_DEG / ny))
    total = sm.sum() * cell
    return 100.0 * sm / max(total, 1e-9)


def main():
    set_pub_style()
    rng = np.random.default_rng(6530)
    fx, fy, dur = make_fixations(rng)
    gx, gy = gaze_stream(fx, fy, dur, rng)
    dens = density(gx, gy)

    onset = np.cumsum(np.concatenate([[0.0], dur[:-1] + SACCADE_MS]))
    labels = [which_aoi(x, y) for x, y in zip(fx, fy)]
    amps = np.hypot(np.diff(fx), np.diff(fy))
    hit = [k for k, lab in enumerate(labels) if lab == TARGET_AOI]
    ttff = onset[hit[0]] if hit else np.nan

    shares = {}
    for name, *_r in AOIS:
        shares[name] = float(np.sum(dur[[k for k, lab in enumerate(labels)
                                         if lab == name]]))
    total_dwell = float(dur.sum())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.90],
                          left=0.050, right=0.965, top=0.880, bottom=0.150,
                          wspace=0.52, hspace=0.62)

    # ---- (a) heat map plus scan path --------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#fbfbfb")
    for name, x0, y0, w, h in AOIS:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#f0f0f0",
                               edgecolor=HAIR, linewidth=0.8,
                               linestyle=(0, (3.0, 2.0)), zorder=1))
        ax.text(x0 + 0.6, y0 + h - 0.6, name, ha="left", va="top",
                fontsize=6.4, color=MUTED, zorder=2)

    vmax = float(dens.max())
    alpha = np.clip(dens / vmax, 0.0, 1.0) ** 0.55
    im = ax.imshow(dens, origin="lower", extent=[-X_DEG, X_DEG, -Y_DEG, Y_DEG],
                   cmap=get_cmap("cb_sequential_warm"), vmin=0.0, vmax=vmax,
                   alpha=alpha, interpolation="bilinear", zorder=3)

    ax.plot(fx, fy, color=BLUE, lw=0.9, alpha=0.85, zorder=5)
    sizes = 26.0 + 150.0 * (dur / dur.max())
    longest = int(np.argmax(dur))
    for k in range(fx.size):
        edge = ACCENT if k in (0, longest) else BLUE
        ax.scatter([fx[k]], [fy[k]], s=sizes[k], facecolor="white",
                   edgecolor=edge, linewidths=1.0, zorder=6)
        ax.text(fx[k], fy[k], f"{k + 1}", ha="center", va="center",
                fontsize=6.0, color=edge, zorder=7)

    ax.set_xlim(-X_DEG, X_DEG)
    ax.set_ylim(-Y_DEG, Y_DEG)
    ax.set_aspect("equal")
    ax.set_xlabel("horizontal gaze angle (deg)", fontsize=8.0)
    ax.set_ylabel("vertical gaze angle (deg)", fontsize=8.0)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Gaze density with the scan path that produced it",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6)
    safe_colorbar(fig, im, ax, label=r"gaze density (% per deg$^2$)",
                  width=0.11, pad=0.09)

    ax.text(0.012, 0.985,
            f"{fx.size} fixations, {total_dwell / 1000.0:.1f} s dwell; "
            f"first fixation on {TARGET_AOI.lower()} at {ttff:.0f} ms",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=SUB, zorder=8)
    ax.text(0.012, 0.945,
            f"scan path {amps.sum():.0f} deg over {amps.size} saccades, "
            f"mean amplitude {amps.mean():.1f} deg",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=SUB, zorder=8)
    ax.text(0.988, 0.018,
            f"red rings: first fixation and the longest ({dur.max():.0f} ms)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            color=ACCENT, zorder=8)

    # ---- (b) fixation duration distribution --------------------------------
    axh = fig.add_subplot(gs[0, 1])
    lo = np.floor(dur.min() / 50.0) * 50.0
    hi = np.ceil(dur.max() / 50.0) * 50.0
    bins = np.linspace(lo, hi, 13)
    counts, _b, _p = axh.hist(dur, bins=bins, color=BLUE, alpha=0.72,
                              edgecolor="white", linewidth=0.6, zorder=3)
    med = float(np.median(dur))
    axh.axvline(med, color=ACCENT, lw=1.2, ls=(0, (3.2, 2.2)), zorder=5)
    axh.set_ylim(0.0, counts.max() * 1.30)
    axh.text(med + (hi - lo) * 0.02, counts.max() * 1.24,
             f"median {med:.0f} ms", ha="left", va="top", fontsize=6.5,
             color=ACCENT)
    axh.set_xlim(lo, hi)
    axh.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axh.set_xlabel("fixation duration (ms)", fontsize=8.0)
    axh.set_ylabel("fixations (count)", fontsize=8.0)
    axh.set_title("Short fixations dominate the count",
                  fontsize=8.5, fontweight="bold", color=INK, pad=4)
    axh.tick_params(labelsize=6.6)
    axh.grid(True, axis="y", color=GRID, lw=0.5)

    # ---- (c) dwell share by area of interest -------------------------------
    axs = fig.add_subplot(gs[1, 1])
    shares["Background"] = total_dwell - sum(shares.values())
    names = [n for n, *_ in AOIS] + ["Background"]
    secs = np.array([shares[n] / 1000.0 for n in names])
    vals = np.array([100.0 * shares[n] / total_dwell for n in names])
    order = np.argsort(vals)
    y = np.arange(len(names))
    cols = [ACCENT if names[i] == TARGET_AOI else
            (GUIDE if names[i] == "Background" else BLUE) for i in order]
    axs.barh(y, vals[order], height=0.60, color=cols, edgecolor="white",
             linewidth=0.5, zorder=3)
    for k, i in enumerate(order):
        axs.text(vals[i] + 0.8, k, f"{vals[i]:.1f} %  ({secs[i]:.1f} s)",
                 ha="left", va="center", fontsize=6.4, color=SUB)
    axs.set_yticks(y)
    axs.set_yticklabels([names[i] for i in order], fontsize=6.6)
    axs.set_xlim(0.0, vals.max() * 1.55)
    axs.set_ylim(-0.62, len(names) - 0.38)
    axs.set_xlabel("share of total dwell time (%)", fontsize=8.0)
    axs.set_title("Dwell concentrates on a few areas",
                  fontsize=8.5, fontweight="bold", color=INK, pad=4)
    axs.tick_params(labelsize=6.6)
    axs.grid(True, axis="x", color=GRID, lw=0.5)


    panel_labels_inches(fig, [ax, axh, axs], dx=-0.36, dy=0.28)
    savefig(fig, "653_vr_eye_tracking.png")


if __name__ == "__main__":
    main()
