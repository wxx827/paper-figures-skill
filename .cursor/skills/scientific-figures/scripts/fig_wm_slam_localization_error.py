"""Localisation error analysis: time series, spatial map, distribution, ablation.

One reference sensor stack (wheel + IMU + lidar) is repeated over many runs on
the same route.  Error is driven by where the robot is - the open hall in the
middle of the building starves the scan matcher, the cluttered corridors feed
it - so the time series, the spatial scatter and the histogram are three views
of the same array.  Panel (d) re-runs the whole survey for four sensor stacks.

All four panels share one blue ladder, and panels (b) and (c) share the one
colour range computed over the reference errors that are actually drawn.
Every printed number is an f-string over those arrays.

Single-hue blue batch (BRIEF_WM_BLUE.md section 2).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

ERR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "wm_blue_err", [BLUE[6], BLUE[5], BLUE[4], BLUE[3], BLUE[2], BLUE[1],
                    BLUE[0]], N=256)

W, H = 20.0, 12.0
T = 0.30
DT = 0.5
N_RUNS = 40
CONFIGS = ["Wheel only", "Wheel + IMU", "Wheel + IMU\n+ lidar",
           "Full stack\n(+ camera)"]
GAINS = [3.20, 1.90, 1.00, 0.68]
REF = 2                                   # the stack used in (a), (b), (c)

WAY = [(2.4, 2.2), (17.6, 2.2), (17.8, 6.0), (17.6, 9.8), (10.0, 9.8),
       (10.0, 6.0), (2.6, 6.0), (2.4, 9.8), (6.4, 9.8), (6.4, 2.6)]
BLOCKS = [(4.6, 3.4, 2.2, 1.6), (12.2, 3.2, 3.0, 1.8), (12.4, 7.0, 1.6, 1.9),
          (2.0, 7.4, 1.4, 1.6), (7.6, 7.2, 1.4, 1.4)]


def route(n=260):
    pts = np.asarray(WAY, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])]), s


def smooth1(a, k=15):
    ker = np.ones(k) / k
    pad = np.concatenate([np.repeat(a[:1], k), a, np.repeat(a[-1:], k)])
    return np.convolve(pad, ker, "same")[k:-k]


def base_profile(P):
    """Geometry-driven error floor: worst in the open middle of the hall."""
    d = np.full(len(P), 6.0)
    for x0, y0, w, h in BLOCKS:
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        d = np.minimum(d, np.hypot(P[:, 0] - cx, P[:, 1] - cy))
    edge = np.minimum.reduce([P[:, 0], W - P[:, 0], P[:, 1], H - P[:, 1]])
    return 0.055 + 0.052 * np.clip(d, 0, 6.0) + 0.030 * np.clip(edge, 0, 5.0)


def runs_for(gain, base, rng, n_runs=N_RUNS):
    n = len(base)
    out = np.empty((n_runs, n))
    for r in range(n_runs):
        w = smooth1(rng.normal(0, 1.0, n), 21)
        w /= np.abs(w).max() or 1.0
        out[r] = base * gain * np.exp(0.42 * w + rng.normal(0, 0.09, n))
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(44)

    P, s = route()
    t = np.arange(len(P)) * DT
    base = base_profile(P)
    all_runs = [runs_for(g, base, rng) for g in GAINS]
    ref = all_runs[REF]
    mean, std = ref.mean(axis=0), ref.std(axis=0)
    spatial = ref.mean(axis=0)
    flat = ref.ravel()

    # one colour range shared by (b) and (c), taken from what is drawn
    vmin, vmax = float(spatial.min()), float(spatial.max())
    norm = mpl.colors.Normalize(vmin, vmax)

    med = float(np.median(flat))
    p95 = float(np.percentile(flat, 95))
    rmse_by_cfg = [np.sqrt((r ** 2).mean(axis=1)) for r in all_runs]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, left=0.062, right=0.972, top=0.895,
                          bottom=0.075, wspace=0.215, hspace=0.315,
                          width_ratios=[1.0, 1.0])

    # ---- (a) error against time with a confidence band -------------------
    ax = ax_a = fig.add_subplot(gs[0, 0])
    ax.fill_between(t, mean - std, mean + std, color=BLUE[5], alpha=0.85,
                    lw=0, zorder=2, label="$\\pm$1 s.d. over %d runs"
                    % ref.shape[0])
    ax.plot(t, mean, color=BLUE[1], lw=1.5, zorder=4, label="mean error")
    ax.plot(t, ref[0], color=BLUE[3], lw=0.9, ls=(0, (3.0, 1.6)), zorder=3,
            label="single run")
    worst = int(np.argmax(mean))
    ax.axvline(t[worst], color=ACCENT, lw=0.8, ls=(0, (2.4, 1.6)), zorder=1)
    ax.annotate("worst stretch of the route\npeak mean %.2f m" % mean[worst],
                xy=(t[worst], mean[worst]), xytext=(0.60, 0.84),
                textcoords="axes fraction", fontsize=6.6, color=ACCENT,
                fontweight="bold", ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4",
                          lw=0.6, alpha=0.96),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                shrinkA=2.0, shrinkB=3.0))
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)", fontsize=8.0, labelpad=1.8)
    ax.set_ylabel("absolute position error (m)", fontsize=8.0, labelpad=2.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, lw=0.4, color="#e6ebf0")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=1.7,
              labelspacing=0.28)
    ax.set_title("Error over time, %s" % CONFIGS[REF].replace("\n", " "),
                 fontsize=8.8, fontweight="bold", color=INK, pad=3.5)

    # ---- (b) spatial distribution ----------------------------------------
    ax = ax_b = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(BLUE[7])
    for x0, y0, w, h in [(0, 0, W, T), (0, H - T, W, T), (0, 0, T, H),
                         (W - T, 0, T, H)]:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=BLUE[1], ec="none",
                               zorder=2))
    for x0, y0, w, h in BLOCKS:
        ax.add_patch(Rectangle((x0, y0), w, h, fc=BLUE[5], ec=BLUE[1],
                               lw=0.7, zorder=2))
    ax.plot(P[:, 0], P[:, 1], color=BLUE[6], lw=2.6, zorder=3)
    sc = ax.scatter(P[:, 0], P[:, 1], c=spatial, cmap=ERR_CMAP, norm=norm,
                    s=13, linewidths=0.25, edgecolors="white", zorder=5)
    ax.plot(*P[worst], marker="o", ms=10.0, mfc="none", mec=ACCENT, mew=1.2,
            ls="none", zorder=7)
    ax.annotate("same peak as (a)", xy=tuple(P[worst]), xytext=(0.62, 0.30),
                textcoords="axes fraction", fontsize=6.6, color=ACCENT,
                fontweight="bold", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c3ccd4",
                          lw=0.6, alpha=0.96),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                shrinkA=2.0, shrinkB=5.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color(INK)
    cax = ax.inset_axes([0.045, 0.075, 0.30, 0.036], zorder=9)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=ERR_CMAP, norm=norm,
                                   orientation="horizontal",
                                   ticks=[vmin, (vmin + vmax) / 2.0, vmax])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_xticklabels(["%.2f" % vmin, "%.2f" % ((vmin + vmax) / 2.0),
                           "%.2f" % vmax])
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK,
                      pad=1.2)
    cb.set_label("mean error (m)", fontsize=6.5, color=INK, labelpad=1.8)
    ax.set_title("Where the error lives along the route", fontsize=8.8,
                 fontweight="bold", color=INK, pad=3.5)

    # ---- (c) histogram plus cumulative distribution ----------------------
    ax = ax_c = fig.add_subplot(gs[1, 0])
    bins = np.linspace(flat.min(), np.percentile(flat, 99.5), 34)
    cnt, edges = np.histogram(flat, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, cnt / cnt.sum() * 100.0, width=np.diff(edges) * 0.92,
           color=[ERR_CMAP(norm(c)) for c in centres], ec=BLUE[1], lw=0.4,
           zorder=3)
    ax.set_xlabel("absolute position error (m)", fontsize=8.0, labelpad=1.8)
    ax.set_ylabel("share of samples (%)", fontsize=8.0, labelpad=2.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, axis="y", lw=0.4, color="#e6ebf0")
    ax.set_axisbelow(True)

    axc = ax.twinx()
    xs = np.sort(flat)
    cdf = np.arange(1, len(xs) + 1) / len(xs) * 100.0
    axc.plot(xs, cdf, color=BLUE[0], lw=1.6, zorder=5)
    axc.set_ylim(0, 103)
    axc.set_ylabel("cumulative share (%)", fontsize=8.0, labelpad=2.5,
                   color=BLUE[0])
    axc.tick_params(labelsize=7.0, colors=BLUE[0])
    axc.spines["right"].set_visible(True)
    axc.spines["right"].set_color(BLUE[0])
    axc.spines["right"].set_linewidth(0.8)
    for xv, lab, yy in ((med, "median %.2f m" % med, 0.34),
                        (p95, "p95 %.2f m" % p95, 0.68)):
        axc.axvline(xv, color=ACCENT, lw=0.9, ls=(0, (2.4, 1.6)), zorder=4)
        axc.text(xv, yy * 103, "  " + lab, fontsize=6.6, color=ACCENT,
                 fontweight="bold", va="center", ha="left", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white",
                           ec="#c3ccd4", lw=0.5, alpha=0.95))
    ax.set_xlim(bins[0], bins[-1])
    ax.set_title("Error distribution over %s samples"
                 % format(flat.size, ","), fontsize=8.8, fontweight="bold",
                 color=INK, pad=3.5)

    # ---- (d) sensor-stack ablation ---------------------------------------
    ax = ax_d = fig.add_subplot(gs[1, 1])
    bp = ax.boxplot(rmse_by_cfg, widths=0.56, patch_artist=True,
                    medianprops=dict(color="white", lw=1.3),
                    whiskerprops=dict(color=BLUE[1], lw=0.9),
                    capprops=dict(color=BLUE[1], lw=0.9),
                    flierprops=dict(marker="o", ms=2.4, mfc=BLUE[4],
                                    mec="none", alpha=0.8))
    shades = [BLUE[5], BLUE[4], BLUE[2], BLUE[0]]
    for patch, col in zip(bp["boxes"], shades):
        patch.set_facecolor(col)
        patch.set_edgecolor(BLUE[1])
        patch.set_linewidth(0.7)
    for i, r in enumerate(rmse_by_cfg, start=1):
        ax.text(i, np.max(r) + 0.045, "%.2f" % np.median(r), ha="center",
                va="bottom", fontsize=6.6, color=INK, fontweight="bold")
    ax.set_xticks(range(1, len(CONFIGS) + 1))
    ax.set_xticklabels(CONFIGS, fontsize=7.0, linespacing=1.25)
    ax.set_ylabel("per-run RMSE (m)", fontsize=8.0, labelpad=2.0)
    ax.set_ylim(0, max(np.max(r) for r in rmse_by_cfg) * 1.20)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, axis="y", lw=0.4, color="#e6ebf0")
    ax.set_axisbelow(True)
    gain = np.median(rmse_by_cfg[0]) / np.median(rmse_by_cfg[-1])
    ax.text(0.975, 0.955, "full stack is %.1f$\\times$ tighter\nthan wheel "
            "odometry alone" % gain, transform=ax.transAxes, ha="right",
            va="top", fontsize=6.8, color=ACCENT, fontweight="bold",
            linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4",
                      lw=0.6, alpha=0.96))
    ax.set_title("Sensor stack ablation, %d runs each" % N_RUNS,
                 fontsize=8.8, fontweight="bold", color=INK, pad=3.5)

    for k, axx in enumerate((ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(axx, "(%s)" % "abcd"[k], x=-0.075, y=1.025, size=10)

    savefig(fig, "1139_wm_slam_localization_error.png")


if __name__ == "__main__":
    main()
