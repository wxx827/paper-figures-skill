"""Spike raster + peri-stimulus time histogram for two single units."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

DT = 0.001
T0, T1 = -0.30, 1.00
STIM_ON, STIM_OFF = 0.0, 0.50

COND_COLORS = ["#2a5d8f", "#c9772e", "#1f7a63"]
COND_NAMES = ["100% contrast", "35% contrast", "10% contrast"]


def rate_profile(t, base, transient, sustained, peak_lat, decay, off_dip=0.55):
    """Inhomogeneous Poisson intensity (spikes/s) for a visual unit."""
    r = np.full_like(t, base)
    r += transient * np.exp(-0.5 * ((t - peak_lat) / 0.022) ** 2)
    stim = (t >= STIM_ON) & (t <= STIM_OFF)
    ts = t[stim] - STIM_ON
    r[stim] += sustained * (1.0 - np.exp(-ts / 0.025)) * np.exp(-ts / decay)
    post = (t > STIM_OFF) & (t < STIM_OFF + 0.18)
    r[post] *= off_dip
    return np.clip(r, 0.3, None)


def gauss_kernel(sigma_s, dt=DT):
    half = int(np.ceil(4 * sigma_s / dt))
    x = np.arange(-half, half + 1) * dt
    k = np.exp(-0.5 * (x / sigma_s) ** 2)
    return k / (k.sum() * dt)


def simulate(rng, t, params, n_trials):
    """Return (spike_matrix, per-trial smoothed rate) for one condition."""
    r = rate_profile(t, **params)
    counts = (rng.random((n_trials, t.size)) < r * DT).astype(float)
    k = gauss_kernel(0.018) * DT      # unit-area kernel -> convolution gives Hz
    rates = np.array([np.convolve(row, k, mode="same") / DT for row in counts])
    return counts, rates


def draw_unit(fig, gs_col, t, unit, rng, panel_labels):
    ax_r = fig.add_subplot(gs_col[0])
    ax_p = fig.add_subplot(gs_col[1], sharex=ax_r)

    n_trials = 28
    row = 0
    trial_edges = []
    for ci, params in enumerate(unit["conds"]):
        counts, rates = simulate(rng, t, params, n_trials)
        offsets = []
        for k in range(n_trials):
            idx = np.flatnonzero(counts[k])
            offsets.append(t[idx] * 1e3)
        ax_r.eventplot(offsets, lineoffsets=np.arange(row, row + n_trials) + 0.5,
                       linelengths=0.86, linewidths=0.55,
                       colors=COND_COLORS[ci])
        trial_edges.append((row, row + n_trials))
        row += n_trials

        m = rates.mean(axis=0)
        sem = rates.std(axis=0, ddof=1) / np.sqrt(n_trials)
        ax_p.fill_between(t * 1e3, m - sem, m + sem, color=COND_COLORS[ci],
                          alpha=0.22, lw=0)
        ax_p.plot(t * 1e3, m, color=COND_COLORS[ci], lw=1.4,
                  label=COND_NAMES[ci], zorder=3)

    # stimulus epoch
    for ax in (ax_r, ax_p):
        ax.axvspan(STIM_ON * 1e3, STIM_OFF * 1e3, color="#f0c05a", alpha=0.16,
                   lw=0, zorder=0)
        ax.axvline(0, color="#333333", lw=0.7, ls="--", dashes=(3, 2),
                   zorder=1)

    # condition colour strip just outside the raster
    tr = blended_transform_factory(ax_r.transAxes, ax_r.transData)
    for ci, (lo, hi) in enumerate(trial_edges):
        ax_r.add_patch(Rectangle((-0.105, lo + 0.5), 0.030, hi - lo - 1.0,
                                 transform=tr, facecolor=COND_COLORS[ci],
                                 edgecolor="none", clip_on=False, zorder=5))

    ax_r.set_ylim(row, 0)                 # trial 1 at the top
    ax_r.set_xlim(T0 * 1e3, T1 * 1e3)
    ax_r.set_ylabel("Trial", labelpad=13)
    ax_r.set_yticks([0, 28, 56, 84])
    ax_r.tick_params(labelbottom=False)
    ax_r.set_title(unit["title"], fontsize=8.5, pad=10)
    style_axes(ax_r)

    ax_p.set_xlabel("Time from stimulus onset (ms)")
    ax_p.set_ylabel("Firing rate\n(spikes s$^{-1}$)")
    ax_p.set_xticks([-300, 0, 300, 600, 900])
    style_axes(ax_p)
    ax_p.margins(x=0)

    ax_r.text(0.25 * 1e3, -0.8, "stimulus  (0\u2013500 ms)", ha="center",
              va="bottom", fontsize=6.8, color="#8a6a12", fontweight="bold",
              zorder=6, clip_on=False)

    add_panel_label(ax_r, panel_labels[0], x=-0.20, y=1.05, size=10)
    add_panel_label(ax_p, panel_labels[1], x=-0.20, y=1.02, size=10)
    return ax_r, ax_p


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(11)
    t = np.arange(T0, T1, DT)

    units = [
        dict(title="Unit 1  \u2014  V1 simple cell (transient)",
             conds=[
                 dict(base=4.0, transient=62, sustained=26, peak_lat=0.052,
                      decay=0.16),
                 dict(base=4.0, transient=38, sustained=15, peak_lat=0.062,
                      decay=0.17),
                 dict(base=4.0, transient=15, sustained=6.0, peak_lat=0.078,
                      decay=0.19),
             ]),
        dict(title="Unit 2  \u2014  MT cell (sustained)",
             conds=[
                 dict(base=7.5, transient=26, sustained=42, peak_lat=0.068,
                      decay=0.75, off_dip=0.8),
                 dict(base=7.5, transient=16, sustained=27, peak_lat=0.075,
                      decay=0.70, off_dip=0.8),
                 dict(base=7.5, transient=7, sustained=11, peak_lat=0.090,
                      decay=0.65, off_dip=0.85),
             ]),
    ]

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.35, 1.0],
                  hspace=0.12, wspace=0.34,
                  left=0.115, right=0.985, top=0.915, bottom=0.185)

    axes = []
    for i, unit in enumerate(units):
        labels = [("(a)", "(c)"), ("(b)", "(d)")][i]
        axes.append(draw_unit(fig, [gs[0, i], gs[1, i]], t, unit, rng, labels))

    ymax = max(ax[1].get_ylim()[1] for ax in axes)
    for _, ax_p in axes:
        ax_p.set_ylim(0, ymax)

    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.55, 0.055),
               handlelength=1.4, columnspacing=1.8)

    savefig(fig, "135_raster_psth.png")


if __name__ == "__main__":
    main()
