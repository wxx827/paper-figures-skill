"""Serial position curves in free recall, and what a filled delay does to them.

Main panel: probability of recall against serial position for a 16-item list
under immediate recall, a 15 s filled delay and a continuous-distractor
schedule.  The primacy and recency components of the fitted two-exponential
curve are separated underneath the data, and the recency span quoted in items
is read off the point where the fitted recency term drops below one tenth of
its terminal value.

Auxiliary (both at 58 % of the main panel width): recency strength against
the length of the filled delay, which is the classic dissociation; and the
probability of first recall, which shows that immediate recall starts at the
end of the list while delayed recall starts at the beginning.

Fictional 40-participant experiment, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
TEAL = "#2e7d5b"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

LIST_LEN = 16
N_SUBJ = 40
N_LISTS = 12           # lists per participant per condition
POS = np.arange(1, LIST_LEN + 1)

# (label, asymptote, primacy weight, primacy decay, recency weight, colour)
CONDITIONS = [
    ("immediate recall", 0.30, 0.34, 2.6, 0.52, BLUE),
    ("15 s filled delay", 0.24, 0.30, 2.8, 0.10, ORANGE),
    ("continuous distractor", 0.22, 0.28, 2.7, 0.34, TEAL),
]
DELAYS = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 30.0])


def curve(p, base, w_pri, tau_pri, w_rec, tau_rec=2.2, n=LIST_LEN):
    """Two-exponential serial position model, clipped to a probability."""
    p = np.asarray(p, dtype=float)
    pri = w_pri * np.exp(-(p - 1.0) / tau_pri)
    rec = w_rec * np.exp(-(n - p) / tau_rec)
    return np.clip(base + pri + rec, 0.01, 0.99)


def fit_curve(p, y):
    def f(pp, base, w_pri, tau_pri, w_rec, tau_rec):
        return curve(pp, base, w_pri, tau_pri, w_rec, tau_rec)
    popt, _ = curve_fit(f, p, y, p0=[0.28, 0.32, 2.6, 0.45, 2.2],
                        bounds=([0.02, 0.0, 0.6, 0.0, 0.6],
                                [0.70, 0.90, 8.0, 0.90, 8.0]),
                        maxfev=20000)
    return popt


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(959)

    obs, fits = {}, {}
    for label, base, w_p, t_p, w_r, _ in CONDITIONS:
        p_true = curve(POS, base, w_p, t_p, w_r)
        counts = rng.binomial(N_SUBJ * N_LISTS, p_true) / (N_SUBJ * N_LISTS)
        obs[label] = counts
        fits[label] = fit_curve(POS, counts)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the three serial position curves -----------------------------
    ax.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    fine = np.linspace(1.0, float(LIST_LEN), 300)
    spans = {}
    for label, _, _, _, _, col in CONDITIONS:
        y = obs[label]
        base, w_p, t_p, w_r, t_r = fits[label]
        se = np.sqrt(y * (1 - y) / (N_SUBJ * N_LISTS))
        ax.plot(fine, 100 * curve(fine, *fits[label]), color=col, lw=1.7,
                zorder=4)
        ax.errorbar(POS, 100 * y, yerr=196 * se, fmt="o", ms=4.4, mfc=col,
                    mec="white", mew=0.8, ecolor=col, elinewidth=0.8,
                    capsize=1.6, ls="none", zorder=6)
        # recency span: where the fitted recency term falls below 10 % of peak
        span = float(t_r * np.log(10.0))
        spans[label] = span
        if label != "15 s filled delay":
            ax.plot(fine, 100 * (base + w_r * np.exp(-(LIST_LEN - fine) / t_r)),
                    color=col, lw=0.9, ls=(0, (1.6, 1.8)), alpha=0.85,
                    zorder=3)
        ax.plot(fine, 100 * (base + w_p * np.exp(-(fine - 1.0) / t_p)),
                color=col, lw=0.9, ls=(0, (4.0, 2.0)), alpha=0.55, zorder=3)

    ax.axvspan(0.4, 3.5, facecolor="#f1f4f7", edgecolor="none", zorder=0)
    ax.axvspan(13.5, 16.6, facecolor="#f1f4f7", edgecolor="none", zorder=0)
    ax.text(1.95, 92.0, "primacy", ha="center", va="bottom", fontsize=7.4,
            color=SUB)
    ax.text(15.05, 92.0, "recency", ha="center", va="bottom", fontsize=7.4,
            color=SUB)

    for label, _, _, _, _, col in CONDITIONS:
        base, w_p, t_p, w_r, t_r = fits[label]
        ax.plot([], [], color=col, lw=1.5, label=label)
    ax.legend(loc="upper center", frameon=False, fontsize=7.0,
              handlelength=1.7, labelspacing=0.30, borderpad=0.12, ncol=3,
              bbox_to_anchor=(0.52, 1.012))

    lines = []
    for label, _, _, _, _, col in CONDITIONS:
        base, w_p, t_p, w_r, t_r = fits[label]
        lines.append(f"{label}: primacy gain {100 * w_p:.0f} pp over "
                     f"{t_p:.1f} items, recency gain {100 * w_r:.0f} pp, "
                     f"recency span {spans[label]:.1f} items")
    ax.text(3.30, 88.0, "\n".join(lines) +
            "\ndashed limb = fitted primacy term, dotted limb = fitted "
            "recency term\n"
            f"fitted from {N_SUBJ} participants x {N_LISTS} lists = "
            f"{N_SUBJ * N_LISTS} recall opportunities per position",
            ha="left", va="top", fontsize=6.6, color=SUB, linespacing=1.55)

    ax.set_xlim(0.4, 16.6)
    ax.set_ylim(8.0, 100.0)
    ax.set_xticks(POS)
    ax.set_yticks(np.arange(10, 101, 15))
    ax.set_xlabel(f"serial position in the {LIST_LEN}-item list "
                  "(1 = first presented)", fontsize=8.5)
    ax.set_ylabel("probability of recall (%)", fontsize=8.5)
    ax.set_title("A filled delay removes the recency limb and leaves primacy "
                 "almost intact", fontsize=9.0, fontweight="bold", pad=5.0)

    # ---- aux (b): recency strength against delay length ---------------------
    n_delay = N_SUBJ * N_LISTS * 3
    rec_w, rec_lo, rec_hi = [], [], []
    for d in DELAYS:
        w_r = 0.52 * np.exp(-d / 6.5) + 0.055
        y = curve(POS, 0.28, 0.31, 2.7, w_r)
        counts = rng.binomial(n_delay, y) / n_delay
        rec_w.append(100 * fit_curve(POS, counts)[3])
        boot = [100 * fit_curve(POS, rng.binomial(n_delay, counts) / n_delay)[3]
                for _ in range(120)]
        rec_lo.append(np.percentile(boot, 2.5))
        rec_hi.append(np.percentile(boot, 97.5))
    rec_w = np.array(rec_w)

    def decay(d, amp, tau, floor):
        return amp * np.exp(-d / tau) + floor

    (amp, tau, floor), _ = curve_fit(decay, DELAYS, rec_w,
                                     p0=[50.0, 7.0, 5.0],
                                     bounds=([1.0, 0.5, 0.0],
                                             [200.0, 60.0, 40.0]))
    half = float(tau * np.log(2.0))

    ax_t.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    ax_t.fill_between(DELAYS, rec_lo, rec_hi, facecolor=lighten(ACCENT, 0.78),
                      edgecolor="none", zorder=2)
    dd = np.linspace(0.0, 30.0, 200)
    ax_t.plot(dd, decay(dd, amp, tau, floor), color=FAINT, lw=1.0,
              ls=(0, (3.4, 2.0)), zorder=4)
    ax_t.plot(DELAYS, rec_w, marker="o", ms=4.2, lw=1.5, color=ACCENT,
              mfc="white", mec=ACCENT, mew=1.1, zorder=5)
    ax_t.axvline(half, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    ax_t.text(half + 0.8, 62.0,
              f"fitted time constant {tau:.1f} s\nhalf-life {half:.1f} s",
              ha="left", va="top", fontsize=6.3, color=SUB, linespacing=1.4)
    ax_t.set_xlim(-1.5, 31.5)
    ax_t.set_ylim(0.0, 66.0)
    ax_t.set_xticks(DELAYS)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("filled delay before recall (s)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("fitted recency gain (pp)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Recency decays with the length of the delay",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_t.text(30.8, 40.0, f"band = 120 bootstrap re-fits,\n"
                          f"n={n_delay} recalls per position", ha="right",
              va="top", fontsize=6.1, color=MUTED, linespacing=1.4)

    # ---- aux (c): probability of first recall -------------------------------
    ax_b.grid(color=GRID, lw=0.5, alpha=0.6, zorder=0)
    for label, _, _, _, _, col in CONDITIONS:
        if label == "continuous distractor":
            continue
        base, w_p, t_p, w_r, t_r = fits[label]
        strength = (w_p * np.exp(-(POS - 1.0) / t_p) +
                    w_r * np.exp(-(LIST_LEN - POS) / t_r) + 0.04) ** 2.6
        pfr = strength / strength.sum()
        draws = rng.multinomial(N_SUBJ * N_LISTS, pfr) / (N_SUBJ * N_LISTS)
        ax_b.plot(POS, 100 * draws, marker="o", ms=3.4, lw=1.3, color=col,
                  mfc="white", mec=col, mew=0.9, zorder=4, label=label)
        start = float(np.sum(POS * draws) / draws.sum())
        ax_b.axvline(start, color=col, lw=0.9, ls=(0, (3.0, 2.2)), zorder=2)
        y_lab = 33.0 if label == "immediate recall" else 25.0
        ax_b.text(start - 0.4, y_lab, f"mean start {start:.1f}", ha="right",
                  va="top", fontsize=6.1, color=col)
    ax_b.set_xlim(0.4, 16.6)
    ax_b.set_ylim(0.0, 50.0)
    ax_b.set_xticks([1, 4, 8, 12, 16])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("serial position of the first item recalled",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("probability (%)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Immediate recall starts at the end of the list",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.2,
                handlelength=1.3, labelspacing=0.24, borderpad=0.10,
                bbox_to_anchor=(-0.012, 1.03))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "959_cog_serial_position.png")


if __name__ == "__main__":
    main()
