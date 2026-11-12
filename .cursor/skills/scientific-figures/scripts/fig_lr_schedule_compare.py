"""Learning-rate schedules and the validation accuracy they produce.

Two stacked panels on a shared epoch axis: the schedule itself on top, the
resulting validation accuracy below.  Three key transition points (end of
warm-up, one-cycle peak, first step decay) are carried through both panels
by full-height dashed rules so the cause and its effect line up vertically.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

OURS = "#3b5b92"
GREY = "#8c8c8c"
LGREY = "#b8b8b8"
GREEN = "#2e7d5b"
ORANGE = "#c9a227"
PURPLE = "#8e6bbf"
GRID = "#eeeeee"

T = 120
EP = np.arange(0, T + 1, dtype=float)
LR0 = 0.10
WARM = 8.0
PEAK1C = 36.0

SCHEDULES = ["Step decay", "Linear", "Cosine", "One-cycle",
             "Warm-up + cosine (Ours)"]
SCOL = {"Step decay": GREY, "Linear": PURPLE, "Cosine": GREEN,
        "One-cycle": ORANGE, "Warm-up + cosine (Ours)": OURS}
SLW = {s: (2.0 if "Ours" in s else 1.3) for s in SCHEDULES}
# final validation accuracy each schedule converges to
AFIN = {"Step decay": 92.9, "Linear": 93.2, "Cosine": 93.7,
        "One-cycle": 94.0, "Warm-up + cosine (Ours)": 94.6}
TAU = {"Step decay": 27.0, "Linear": 24.0, "Cosine": 22.0,
       "One-cycle": 26.0, "Warm-up + cosine (Ours)": 15.0}

MARKS = [(WARM, "warm-up ends"), (PEAK1C, "one-cycle peak"),
         (60.0, "1st step decay")]


def lr_curves():
    t = EP
    out = {}
    out["Step decay"] = LR0 * np.where(t < 60, 1.0, np.where(t < 90, 0.1, 0.01))
    out["Linear"] = LR0 * np.clip(1.0 - t / T, 0.004, None)
    out["Cosine"] = LR0 * 0.5 * (1 + np.cos(np.pi * t / T)) + 1e-4
    warm = np.clip(t / WARM, 0, 1)
    cos = 0.5 * (1 + np.cos(np.pi * np.clip((t - WARM) / (T - WARM), 0, 1)))
    out["Warm-up + cosine (Ours)"] = LR0 * warm * np.where(t < WARM, 1.0, cos) \
        + 1e-4
    up = np.clip(t / PEAK1C, 0, 1)
    dn = 0.5 * (1 + np.cos(np.pi * np.clip((t - PEAK1C) / (T - PEAK1C), 0, 1)))
    out["One-cycle"] = (0.02 * LR0 + 1.95 * LR0 * up) * np.where(t < PEAK1C,
                                                                 1.0, dn) + 1e-4
    return out


def acc_curves(lrs):
    rng = np.random.default_rng(3306)
    t = EP
    out = {}
    for s in SCHEDULES:
        lr = lrs[s]
        d = 1.0 - lr / np.maximum.accumulate(np.maximum(lr, 1e-9))
        tau = TAU[s]
        w = 0.62 if "Ours" in s else 0.52
        a0, af = 46.0, AFIN[s]
        base = a0 + (af - a0) * (w * (1 - np.exp(-t / tau)) + (1 - w) * d)
        jit = rng.normal(0, 0.42, t.size) * np.exp(-t / 70.0)
        wob = np.cumsum(rng.normal(0, 0.10, t.size))
        wob -= wob * np.linspace(0, 1, t.size)
        out[s] = base + jit + wob
    return out


def main():
    set_pub_style(base_font=8)
    lrs = lr_curves()
    accs = acc_curves(lrs)

    fig = plt.figure(figsize=(7.0, 4.75))
    gs = fig.add_gridspec(2, 1, left=0.088, right=0.985, top=0.845,
                          bottom=0.095, hspace=0.20, height_ratios=[1.0, 1.35])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.axvspan(0, WARM, color="#f4f7fa", lw=0, zorder=0)

    # ---- (a) schedules -----------------------------------------------------
    for s in SCHEDULES:
        hot = "Ours" in s
        ax1.plot(EP, lrs[s], color=SCOL[s], lw=SLW[s],
                 zorder=7 if hot else 4, label=s)
    ax1.set_ylabel("Learning rate  $\\eta$", fontsize=8)
    ax1.set_ylim(-0.004, 0.215)
    ax1.set_yticks([0.05, 0.10, 0.15, 0.20])
    ax1.tick_params(labelbottom=False)
    ax1.set_title("Schedule shape", fontsize=8.5, loc="left", pad=3)
    add_panel_label(ax1, "(a)", x=-0.075, y=1.01, size=9.5)
    style_axes(ax1)

    # ---- (b) resulting accuracy -------------------------------------------
    for s in SCHEDULES:
        hot = "Ours" in s
        ax2.plot(EP, accs[s], color=SCOL[s], lw=SLW[s],
                 zorder=7 if hot else 4)
    ax2.set_xlabel("Epoch", fontsize=8)
    ax2.set_ylabel("Validation accuracy (%)", fontsize=8)
    ax2.set_xlim(0, T)
    ax2.set_ylim(44, 97)
    ax2.set_yticks([50, 60, 70, 80, 90])
    ax2.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax2.set_title("Resulting validation accuracy", fontsize=8.5, loc="left",
                  pad=3)
    add_panel_label(ax2, "(b)", x=-0.075, y=1.01, size=9.5)
    style_axes(ax2)

    # final-accuracy read-out, sorted best first
    order = sorted(SCHEDULES, key=lambda s: -AFIN[s])
    for j, s in enumerate(order):
        hot = "Ours" in s
        ax2.text(0.985, 0.40 - 0.083 * j,
                 f"{accs[s][-1]:.1f}%   {s.replace(' (Ours)', '')}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=6.8 if not hot else 7.3, color=SCOL[s],
                 fontweight="bold" if hot else "normal", zorder=9)
    ax2.text(0.985, 0.475, "final accuracy", transform=ax2.transAxes,
             ha="right", va="top", fontsize=6.6, color="#888888",
             style="italic", zorder=9)

    # ---- transition rules through both panels ------------------------------
    for j, (xm, lab) in enumerate(MARKS):
        for ax in (ax1, ax2):
            ax.axvline(xm, color="#7a7a7a", lw=0.8, ls=(0, (2.6, 2.0)),
                       zorder=2, alpha=0.85)
        ax1.annotate(lab, xy=(xm, 0.215), xytext=(xm, 0.238),
                     textcoords="data", ha="center", va="bottom",
                     fontsize=6.5, color="#5a5a5a", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="#9a9a9a",
                                     shrinkA=0, shrinkB=1))
        ax2.plot([xm], [accs["Warm-up + cosine (Ours)"][int(xm)]], marker="o",
                 ms=3.4, mfc="white", mec=OURS, mew=1.1, zorder=8)

    handles = [Line2D([], [], color=SCOL[s], lw=SLW[s], label=s)
               for s in SCHEDULES]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_texts()[4].set_fontweight("bold")

    for s in order:
        print(f"  {s:26s} peak lr {lrs[s].max():.3f}  final acc "
              f"{accs[s][-1]:.2f}%")
    savefig(fig, "306_lr_schedule_compare.png")


if __name__ == "__main__":
    main()
