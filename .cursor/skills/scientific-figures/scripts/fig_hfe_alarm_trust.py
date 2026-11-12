"""Alarm reliability and operator trust, read through signal detection theory.

Main panel: the evidence distribution a monitoring alarm sees when nothing is
wrong and when something is, with the alert threshold drawn through both.  The
four signal-detection outcomes are shaded under the curves, an inset ROC sweeps
the same threshold from liberal to conservative, and the quoted hit rate, false
alarm rate, d-prime and positive predictive value are all obtained by
integrating the two density arrays that are plotted.

Auxiliary (both at 58 % of the main panel width): positive predictive value
against event prevalence for three threshold settings, which is where the
"cry wolf" problem lives; and the alarm budget of a 12-hour shift split into
true and false alerts, with the operator response probability that a simple
trust model returns for that budget.

Fictional monitoring system, illustrative values only; nothing here is
calibrated against a real clinical or flight-deck alarm.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
TEAL = "#2e7d5b"
RUST = "#c0603f"
GOLD = "#c9a227"
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

MU_N, SD_N = 4.00, 1.00          # sensor score with no event present
MU_S, SD_S = 6.70, 1.15          # sensor score with the event present
PREVALENCE = 0.030               # events per monitored epoch
EPOCHS_PER_SHIFT = 620           # monitored epochs in a 12-hour shift

SETTINGS = [("liberal", 5.05), ("as fielded", 5.75), ("conservative", 6.55)]


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def rates(x, pdf_n, pdf_s, crit):
    """Hit and false-alarm rate by integrating the plotted density arrays."""
    keep = x >= crit
    hit = float(np.trapezoid(pdf_s[keep], x[keep]))
    fa = float(np.trapezoid(pdf_n[keep], x[keep]))
    return float(np.clip(hit, 0.0, 1.0)), float(np.clip(fa, 0.0, 1.0))


def ppv(hit, fa, prev):
    num = hit * prev
    den = num + fa * (1.0 - prev)
    return num / den if den > 0 else 0.0


def compliance(p):
    """Probability the operator acts on an alert, given its track record."""
    return 1.0 / (1.0 + np.exp(-7.4 * (p - 0.36)))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(948)

    x = np.linspace(0.0, 11.0, 2201)
    pdf_n = normal_pdf(x, MU_N, SD_N)
    pdf_s = normal_pdf(x, MU_S, SD_S)
    d_prime = (MU_S - MU_N) / np.sqrt(0.5 * (SD_N ** 2 + SD_S ** 2))

    crit_now = SETTINGS[1][1]
    hit_now, fa_now = rates(x, pdf_n, pdf_s, crit_now)
    ppv_now = ppv(hit_now, fa_now, PREVALENCE)

    # ROC by sweeping the same threshold across the same arrays
    grid = np.linspace(0.6, 10.4, 260)
    roc = np.array([rates(x, pdf_n, pdf_s, c) for c in grid])
    auc = float(np.trapezoid(roc[::-1, 0], roc[::-1, 1]))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the two evidence distributions and the threshold ------------
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    right = x >= crit_now
    left = x <= crit_now
    ax.fill_between(x[left], 0.0, pdf_s[left], facecolor=lighten(RUST, 0.60),
                    edgecolor="none", zorder=2)
    ax.fill_between(x[right], 0.0, pdf_n[right], facecolor=lighten(GOLD, 0.42),
                    edgecolor="none", zorder=3)
    ax.fill_between(x[right], 0.0, pdf_s[right], facecolor=lighten(TEAL, 0.55),
                    edgecolor="none", alpha=0.92, zorder=4)
    ax.fill_between(x[left], 0.0, pdf_n[left], facecolor="#eeeeee",
                    edgecolor="none", zorder=1)
    ax.plot(x, pdf_n, color=BLUE, lw=1.6, zorder=6)
    ax.plot(x, pdf_s, color=ACCENT, lw=1.6, zorder=6)
    ax.axvline(crit_now, color=INK, lw=1.5, zorder=7)

    for name, c in SETTINGS:
        if name == "as fielded":
            continue
        ax.axvline(c, color=FAINT, lw=0.9, ls=(0, (3.2, 2.2)), zorder=5)

    ax.text(crit_now - 0.10, 0.422, f"alert threshold = {crit_now:.2f}",
            rotation=90, ha="right", va="top", fontsize=6.5, color=INK)
    ax.text(1.92, 0.222, "no event\n(noise)", ha="center", va="bottom",
            fontsize=7.2, color=BLUE, linespacing=1.3)
    ax.text(8.85, 0.145, "event present\n(signal + noise)", ha="center",
            va="bottom", fontsize=7.2, color=ACCENT, linespacing=1.3)

    ax.annotate("", xy=(MU_S, 0.458), xytext=(MU_N, 0.458),
                arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(0.5 * (MU_N + MU_S), 0.453,
            f"d' = {d_prime:.2f} (pooled SD units)",
            ha="center", va="top", fontsize=6.8, color=SUB)

    ax.text(3.05, 0.055, "correct\nrejection", ha="center", va="bottom",
            fontsize=6.4, color=MUTED, linespacing=1.3)
    ax.text(4.95, 0.020, "miss", ha="center", va="bottom", fontsize=6.4,
            color="#a2543c")
    ax.text(6.28, 0.033, "false\nalarm", ha="center", va="bottom",
            fontsize=6.4, color="#8a6f14", linespacing=1.3)
    ax.text(7.55, 0.115, "hit", ha="center", va="bottom", fontsize=6.9,
            color="#1f5b41")

    ax.set_xlim(0.6, 10.6)
    ax.set_ylim(0.0, 0.56)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xlabel("alarm evidence score (arbitrary units)", fontsize=8.5)
    ax.set_ylabel("probability density (1 / score unit)", fontsize=8.5)
    ax.set_title("A high-sensitivity threshold on a rare event: "
                 f"prevalence = {100 * PREVALENCE:.1f} % of epochs",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # inset ROC on the same swept arrays
    axi = ax.inset_axes([0.700, 0.520, 0.290, 0.415])
    style_axes(axi)
    axi.plot([0, 1], [0, 1], color="#c4c4c4", lw=0.8, ls=(0, (3, 2.4)))
    axi.plot(roc[:, 1], roc[:, 0], color=BLUE, lw=1.4)
    for name, c in SETTINGS:
        h, f = rates(x, pdf_n, pdf_s, c)
        mk = "o" if name == "as fielded" else "s"
        col = ACCENT if name == "as fielded" else FAINT
        axi.plot([f], [h], marker=mk, ms=4.0, mfc=col, mec="white", mew=0.7,
                 zorder=5)
    axi.set_xlim(-0.03, 1.03)
    axi.set_ylim(-0.03, 1.03)
    axi.set_xticks([0, 0.5, 1.0])
    axi.set_yticks([0, 0.5, 1.0])
    axi.tick_params(labelsize=6.0, length=2.2)
    axi.set_xlabel("false alarm rate", fontsize=6.4, labelpad=1.2)
    axi.set_ylabel("hit rate", fontsize=6.4, labelpad=1.2)
    axi.text(0.96, 0.06, f"AUC = {auc:.3f}", ha="right", va="bottom",
             fontsize=6.2, color=SUB)

    ax.text(0.70, 0.548,
            f"solid rule = fielded threshold, dashed = liberal and\n"
            f"conservative settings (panels b, c)\n"
            f"hit rate {100 * hit_now:.1f} %, false alarms "
            f"{100 * fa_now:.1f} %, predictive\nvalue "
            f"{100 * ppv_now:.1f} % -- {1.0 / ppv_now:.0f} alerts "
            f"per true event",
            ha="left", va="top", fontsize=6.7, color=SUB, linespacing=1.55)

    # ---- aux (b): predictive value against prevalence ----------------------
    prev = np.logspace(np.log10(0.002), np.log10(0.50), 240)
    cols = {"liberal": GOLD, "as fielded": BLUE, "conservative": TEAL}
    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    for name, c in SETTINGS:
        h, f = rates(x, pdf_n, pdf_s, c)
        y = 100.0 * np.array([ppv(h, f, p) for p in prev])
        ax_t.plot(prev * 100.0, y, color=cols[name], lw=1.5, zorder=4,
                  label=f"{name} (thr {c:.2f})")
    ax_t.axvline(PREVALENCE * 100.0, color=FAINT, lw=0.9,
                 ls=(0, (3.0, 2.2)), zorder=3)
    ax_t.plot([PREVALENCE * 100.0], [100.0 * ppv_now], marker="o", ms=4.2,
              mfc=ACCENT, mec="white", mew=0.7, zorder=6)
    ax_t.text(PREVALENCE * 100.0 * 1.16, 100.0 * ppv_now + 4.0,
              f"{100 * ppv_now:.0f} %", fontsize=6.4, color=ACCENT,
              ha="left", va="bottom")
    ax_t.set_xscale("log")
    ax_t.set_xlim(0.2, 50.0)
    ax_t.set_ylim(0.0, 108.0)
    ax_t.set_yticks([0, 25, 50, 75, 100])
    ax_t.set_xticks([0.2, 1.0, 5.0, 20.0, 50.0])
    ax_t.set_xticklabels(["0.2", "1", "5", "20", "50"])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("event prevalence (% of epochs)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("predictive value (%)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Rare events poison the alert stream", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.legend(loc="upper left", frameon=False, fontsize=6.1,
                handlelength=1.3, labelspacing=0.28, borderpad=0.12,
                bbox_to_anchor=(-0.012, 1.03))

    # ---- aux (c): the alarm budget of one 12-hour shift --------------------
    names = [s[0] for s in SETTINGS]
    true_alerts, false_alerts, resp = [], [], []
    for name, c in SETTINGS:
        h, f = rates(x, pdf_n, pdf_s, c)
        t = EPOCHS_PER_SHIFT * PREVALENCE * h
        fl = EPOCHS_PER_SHIFT * (1.0 - PREVALENCE) * f
        true_alerts.append(t)
        false_alerts.append(fl)
        resp.append(compliance(ppv(h, f, PREVALENCE)))
    true_alerts = np.array(true_alerts)
    false_alerts = np.array(false_alerts)
    resp = np.array(resp)
    # observed operator responses, one shift of Bernoulli draws per setting
    n_alerts = np.maximum(np.round(true_alerts + false_alerts), 1).astype(int)
    obs = rng.binomial(n_alerts, resp) / n_alerts

    pos = np.arange(len(names))
    ax_b.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.bar(pos, false_alerts, width=0.56, color=lighten(GOLD, 0.30),
             edgecolor="white", linewidth=0.7, zorder=3, label="false alerts")
    ax_b.bar(pos, true_alerts, width=0.56, bottom=false_alerts, color=TEAL,
             edgecolor="white", linewidth=0.7, zorder=3, label="true alerts")
    for k in pos:
        tot = false_alerts[k] + true_alerts[k]
        ax_b.text(k, tot + 1.6, f"{tot:.0f} alerts", ha="center", va="bottom",
                  fontsize=6.4, color=INK)
        ax_b.text(k, tot + 7.0, f"response {100 * obs[k]:.0f} %",
                  ha="center", va="bottom", fontsize=6.3, color=ACCENT)
    ax_b.set_xticks(pos)
    ax_b.set_xticklabels([f"{n}\nthr {c:.2f}" for n, c in SETTINGS],
                         fontsize=6.4, linespacing=1.35)
    ax_b.set_xlim(-0.62, len(names) - 0.38)
    ax_b.set_ylim(0.0, max(false_alerts + true_alerts) * 1.42)
    ax_b.tick_params(axis="y", labelsize=6.6)
    ax_b.set_ylabel("alerts per 12-h shift (n)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title(f"Alarm budget, n={EPOCHS_PER_SHIFT} monitored epochs",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_b.legend(loc="upper right", frameon=False, fontsize=6.2,
                handlelength=1.2, labelspacing=0.28, borderpad=0.12,
                bbox_to_anchor=(1.01, 1.03))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "948_hfe_alarm_trust.png")


if __name__ == "__main__":
    main()
