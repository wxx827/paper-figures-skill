"""Six-channel surface EMG through two gait cycles: raw traces, linear
envelopes, activation timing and the two muscle synergies behind them.

The raw waterfall follows the electrophysiology convention and carries an
L-shaped scale bar instead of a y axis - stacked traces have no meaningful
common ordinate, and a scale bar states amplitude and time without pretending
otherwise.  Envelopes are computed from those same traces (full-wave
rectification, 6 Hz low pass), the timing bars are the threshold crossings of
the envelopes, and the synergies come from an NMF factorisation of the
cycle-averaged envelope matrix, so every panel is downstream of panel (a).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
from sklearn.decomposition import NMF
from pubstyle import set_pub_style, savefig, style_axes, COLORS
from figkit import panel_labels_inches

GREEN = "#2e7d5b"        # plantarflexors
GOLD = "#c9a227"         # dorsiflexor
PURPLE = "#8e6bbf"       # knee extensors
BLUE = "#3b5b92"         # hamstring
SYN = ["#b23a48", "#3b5b92"]

FS = 1000.0              # sampling rate (Hz)
CYCLE_S = 1.10           # stride time (s)
N_CYCLE = 2
OFFSET_MV = 2.3          # vertical spacing of the raw waterfall (mV)
THRESH_MVC = 12.0        # activation threshold (%MVC)
MIN_BURST_MS = 40.0      # bursts shorter than this are debounced away
MVC_MV = 1.45            # amplitude that counts as maximal contraction

# (tag, full name, colour, linestyle, peak mV, [(centre %, width %, weight)])
MUSCLES = [
    ("TA", "Tibialis anterior", GOLD, "-", 0.72,
     [(3.0, 7.0, 1.00), (88.0, 12.0, 0.85)]),
    ("GM", "Gastrocnemius med.", GREEN, "-", 0.95, [(43.0, 11.0, 1.00)]),
    ("SOL", "Soleus", GREEN, (0, (3.4, 2.0)), 0.80, [(38.0, 15.0, 0.95)]),
    ("VL", "Vastus lateralis", PURPLE, "-", 0.88, [(8.0, 9.0, 1.00)]),
    ("RF", "Rectus femoris", PURPLE, (0, (3.4, 2.0)), 0.60,
     [(5.0, 6.0, 1.00), (62.0, 7.0, 0.75)]),
    ("BF", "Biceps femoris", BLUE, "-", 0.70,
     [(95.0, 10.0, 1.00), (8.0, 7.0, 0.55)]),
]


def _bump(p, centre, width, amp):
    d = (p - centre + 50.0) % 100.0 - 50.0
    return amp * np.exp(-0.5 * (d / width) ** 2)


def simulate(rng):
    t = np.arange(0.0, N_CYCLE * CYCLE_S, 1.0 / FS)
    phase = (t % CYCLE_S) / CYCLE_S * 100.0
    cyc = (t // CYCLE_S).astype(int)
    b_hp, a_hp = butter(4, [20.0 / (FS / 2), 450.0 / (FS / 2)], btype="band")
    b_lp, a_lp = butter(4, 6.0 / (FS / 2), btype="low")

    raw, env = {}, {}
    for tag, _, _, _, peak, bursts in MUSCLES:
        drive = 0.035 + sum(_bump(phase, c, w, a) for c, w, a in bursts)
        drive = drive * (0.92 + 0.16 * rng.random(N_CYCLE))[cyc] * peak
        x = filtfilt(b_hp, a_hp, rng.normal(0.0, 1.0, t.size)) * drive
        raw[tag] = x
        env[tag] = filtfilt(b_lp, a_lp, np.abs(x)) / MVC_MV * 100.0
    return t, raw, env


def bursts_of(mask, t):
    d = np.diff(mask.astype(int))
    starts = list(t[1:][d == 1])
    ends = list(t[1:][d == -1])
    if mask[0]:
        starts.insert(0, t[0])
    if mask[-1]:
        ends.append(t[-1])
    return [(s, e - s) for s, e in zip(starts, ends)
            if (e - s) * 1000.0 >= MIN_BURST_MS]


def synergies(env, n_pts=101):
    """Cycle-average the envelopes, factorise into two synergies, return VAF."""
    n = int(round(CYCLE_S * FS))
    cycle = np.stack([np.mean([env[tag][k * n:(k + 1) * n]
                               for k in range(N_CYCLE)], axis=0)
                      for tag, *_ in MUSCLES], axis=1)
    idx = np.linspace(0, n - 1, n_pts).astype(int)
    x = np.clip(cycle[idx], 0.0, None)
    model = NMF(n_components=2, init="nndsvda", random_state=0, max_iter=900)
    w = model.fit_transform(x)
    h = model.components_
    vaf = 1.0 - np.sum((x - w @ h) ** 2) / np.sum(x ** 2)
    order = np.argsort(-h.sum(axis=1))
    h = h[order] / h[order].max(axis=1, keepdims=True)
    return h, vaf


def main():
    set_pub_style()
    rng = np.random.default_rng(513)
    t, raw, env = simulate(rng)
    h, vaf = synergies(env)
    tags = [m[0] for m in MUSCLES]
    colours = {m[0]: m[2] for m in MUSCLES}
    t0, t1 = -0.22, N_CYCLE * CYCLE_S

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.66, 0.42], wspace=0.30,
                          hspace=0.34, left=0.055, right=0.985, top=0.880,
                          bottom=0.095)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_env = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_act = fig.add_subplot(gs[2, 0], sharex=ax_raw)
    ax_sh = fig.add_subplot(gs[0, 1])
    ax_sy = fig.add_subplot(gs[1:, 1])

    # ---- (a) raw waterfall with an L-shaped scale bar ----------------------
    for i, (tag, name, colour, ls, *_) in enumerate(MUSCLES):
        base = (len(MUSCLES) - 1 - i) * OFFSET_MV
        ax_raw.plot(t, raw[tag] + base, color=colour, lw=0.28, alpha=0.95)
        ax_raw.text(-0.03, base, f"{tag}\n{name}", ha="right", va="center",
                    fontsize=6.4, color=COLORS["ink"], linespacing=1.30)
    ax_raw.set_xlim(t0, t1)
    ax_raw.set_ylim(-1.9 * OFFSET_MV * 0.62, len(MUSCLES) * OFFSET_MV * 0.92)
    ax_raw.axis("off")

    sx, sy = 1.86, -1.05 * OFFSET_MV * 0.62
    ax_raw.plot([sx, sx], [sy, sy + 1.0], color=COLORS["rule"], lw=1.5,
                solid_capstyle="butt")
    ax_raw.plot([sx, sx + 0.20], [sy, sy], color=COLORS["rule"], lw=1.5,
                solid_capstyle="butt")
    ax_raw.text(sx - 0.015, sy + 0.5, "1 mV", ha="right", va="center",
                fontsize=6.4, color=COLORS["ink"])
    ax_raw.text(sx + 0.10, sy - 0.16, "200 ms", ha="center", va="top",
                fontsize=6.4, color=COLORS["ink"])
    ax_raw.set_title("Raw surface EMG, two consecutive strides",
                     fontsize=8.5, fontweight="bold")

    # ---- (b) linear envelopes ----------------------------------------------
    for tag, _, colour, ls, *_ in MUSCLES:
        ax_env.plot(t, env[tag], color=colour, lw=1.1, ls=ls)
    ax_env.axhline(THRESH_MVC, color=COLORS["rule"], lw=0.8,
                   ls=(0, (1.6, 1.8)))
    ax_env.text(t1, THRESH_MVC + 1.5, f"activation threshold "
                f"{THRESH_MVC:.0f} %MVC", ha="right", va="bottom",
                fontsize=6.4, color=COLORS["sub"])
    ax_env.set_xlim(t0, t1)
    ax_env.set_ylim(0.0, max(env[k].max() for k in env) * 1.22)
    ax_env.set_ylabel("Envelope (%MVC)", fontsize=8.0)
    ax_env.tick_params(labelsize=7.0, labelbottom=False)
    ax_env.set_title("Linear envelope: rectified, 6 Hz low pass",
                     fontsize=8.5, fontweight="bold")
    style_axes(ax_env)

    # ---- (c) activation timing --------------------------------------------
    duty = {}
    for i, (tag, _, colour, *_) in enumerate(MUSCLES):
        y = len(MUSCLES) - 1 - i
        segs = bursts_of(env[tag] > THRESH_MVC, t)
        ax_act.broken_barh(segs, (y - 0.32, 0.64), facecolor=colour,
                           edgecolor="none", alpha=0.85)
        duty[tag] = 100.0 * sum(d for _, d in segs) / (N_CYCLE * CYCLE_S)
    ax_act.set_yticks(range(len(MUSCLES)))
    ax_act.set_yticklabels(tags[::-1], fontsize=6.6)
    ax_act.set_ylim(-0.7, len(MUSCLES) - 0.3)
    ax_act.set_xlim(t0, t1)
    ax_act.set_xticks(np.arange(0.0, 2.21, 0.5))
    ax_act.set_xlabel("Time (s)", fontsize=8.0)
    ax_act.tick_params(axis="x", labelsize=7.0)
    ax_act.set_title("Activation timing above threshold", fontsize=8.5,
                     fontweight="bold")
    style_axes(ax_act)
    for ax in (ax_env, ax_act):
        for k in range(N_CYCLE + 1):
            ax.axvline(k * CYCLE_S, color=COLORS["baseline"], lw=0.7,
                       ls=(0, (3.0, 2.4)), zorder=1)

    # ---- (d) share of the cycle each muscle is active ----------------------
    order = np.argsort([duty[k] for k in tags])
    y = np.arange(len(tags))
    ax_sh.barh(y, [duty[tags[i]] for i in order], height=0.62,
               color=[colours[tags[i]] for i in order], edgecolor="white",
               linewidth=0.6)
    for yi, i in zip(y, order):
        ax_sh.text(duty[tags[i]] + 1.2, yi, f"{duty[tags[i]]:.0f} %",
                   ha="left", va="center", fontsize=6.4, color=COLORS["ink"])
    ax_sh.set_yticks(y)
    ax_sh.set_yticklabels([tags[i] for i in order], fontsize=6.6)
    ax_sh.set_xlim(0.0, max(duty.values()) * 1.28)
    ax_sh.set_xlabel("Time above threshold (% of gait cycle)", fontsize=7.5)
    ax_sh.tick_params(axis="x", labelsize=6.5)
    ax_sh.set_title("Activation share", fontsize=8.5, fontweight="bold")
    style_axes(ax_sh)

    # ---- (e) two NMF synergies ---------------------------------------------
    yy = np.arange(len(tags))[::-1]
    bh = 0.36
    for k in range(2):
        ax_sy.barh(yy + (bh / 2 if k == 0 else -bh / 2), h[k], height=bh,
                   color=SYN[k], alpha=0.9, edgecolor="white", linewidth=0.6,
                   label=f"synergy {k + 1}")
    ax_sy.set_yticks(yy)
    ax_sy.set_yticklabels(tags, fontsize=6.6)
    ax_sy.set_xlim(0.0, 1.18)
    ax_sy.set_ylim(-0.7, len(tags) - 0.3)
    ax_sy.set_xlabel("Synergy weight, normalised (-)", fontsize=7.5)
    ax_sy.tick_params(axis="x", labelsize=6.5)
    ax_sy.legend(loc="lower right", frameon=False, fontsize=6.8,
                 handlelength=1.2)
    ax_sy.set_title(f"Muscle synergies, VAF = {100.0 * vaf:.1f} %",
                    fontsize=8.5, fontweight="bold")
    style_axes(ax_sy)

    handles = [Line2D([], [], color=c, lw=1.4, ls=ls, label=f"{tag} - {name}")
               for tag, name, c, ls, *_ in MUSCLES]

    panel_labels_inches(fig, [ax_raw, ax_env, ax_act, ax_sh, ax_sy],
                        dx=-0.50, dy=0.08)
    savefig(fig, "513_bio_emg_activation.png")


if __name__ == "__main__":
    main()
