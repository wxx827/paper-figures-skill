"""One day of monitored traffic seen by a network intrusion detection system.

The three stacked panels share a single 24 h axis so that a spike in the link
rate, the anomaly score it produces and the alerts the analyst actually
receives can be read off the same instant.  Red bands mark the ground-truth
attack windows; everything outside them is benign, so the two backup jobs that
push the score over the threshold are genuine false alarms rather than missed
detections.  The side panels are the evidence: the ROC curve is swept over the
same score array that is plotted above, and the confusion matrix counts the
same minutes at the operating threshold drawn as a dashed line.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches

RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"      # house red: attacks and missed / breached quantities
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

THR = 0.50                                  # operating threshold on the score
# (start hour, end hour, label, score plateau, extra Mbps)
ATTACKS = [(6.60, 7.45, "DDoS volumetric", 0.88, 940.0),
           (13.20, 13.95, "SSH brute force", 0.72, 55.0),
           (19.60, 21.10, "Slow exfiltration", 0.58, 38.0)]
# benign events that also lift the score - the source of the false alarms
BENIGN = [(3.05, 3.55, 0.61, 320.0), (16.30, 16.70, 0.57, 260.0)]

ALERT_TYPES = [("Port scan", BLUE), ("Brute force", GOLD),
               ("Exfiltration", PURPLE), ("DDoS", ACCENT)]


def bump(t, t0, t1, height, rise=0.10):
    """Smooth plateau between t0 and t1, used for both traffic and score."""
    return height * (0.5 * (1.0 + np.tanh((t - t0) / rise)) *
                     0.5 * (1.0 + np.tanh((t1 - t) / rise)))


def build(rng):
    t = np.arange(0.0, 24.0, 1.0 / 60.0)                 # one sample a minute
    diurnal = 210.0 + 150.0 * np.sin((t - 7.5) / 24.0 * 2 * np.pi) ** 2
    traffic = diurnal + rng.normal(0.0, 14.0, t.size)
    score = 0.11 + np.abs(rng.normal(0.0, 0.055, t.size))
    label = np.zeros(t.size, dtype=bool)

    for t0, t1, _name, s, mbps in ATTACKS:
        traffic += bump(t, t0, t1, mbps, rise=0.05)
        score += bump(t, t0, t1, s, rise=0.05) * rng.uniform(0.86, 1.0, t.size)
        label |= (t >= t0) & (t <= t1)
    for t0, t1, s, mbps in BENIGN:
        traffic += bump(t, t0, t1, mbps, rise=0.05)
        score += bump(t, t0, t1, s, rise=0.05) * rng.uniform(0.84, 1.0, t.size)

    return t, np.clip(traffic, 0.0, None), np.clip(score, 0.0, 1.0), label


def roc(score, label):
    thr = np.unique(np.round(np.linspace(0.0, 1.0, 401), 4))[::-1]
    tpr = np.array([(score >= c)[label].mean() for c in thr])
    fpr = np.array([(score >= c)[~label].mean() for c in thr])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, thr, auc


def main():
    set_pub_style()
    rng = np.random.default_rng(577)
    t, traffic, score, label = build(rng)
    fpr, tpr, thr, auc = roc(score, label)

    pred = score >= THR
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    tn = int(np.sum(~pred & ~label))
    det = tp / (tp + fn)
    far = fp / (fp + tn)

    fig = plt.figure(figsize=(10.4, 6.4))
    # Nested grids: the time stack wants to be tight, the evidence column
    # wants room for its axis labels, so the two get independent spacing.
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.55], wspace=0.22,
                             left=0.065, right=0.985, top=0.895, bottom=0.115)
    left = outer[0].subgridspec(3, 1, height_ratios=[1.0, 0.85, 0.75],
                                hspace=0.20)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.0, 0.58], hspace=0.46)
    ax_t = fig.add_subplot(left[0])
    ax_s = fig.add_subplot(left[1])
    ax_a = fig.add_subplot(left[2])
    ax_r = fig.add_subplot(right[0])
    ax_c = fig.add_subplot(right[1])

    # ---- (a) link rate ----------------------------------------------------
    for ax in (ax_t, ax_s, ax_a):
        for t0, t1, _n, _s, _m in ATTACKS:
            ax.axvspan(t0, t1, color=ACCENT, alpha=0.13, lw=0, zorder=0)
    ax_t.plot(t, traffic, color=BLUE, lw=0.8)
    ax_t.set_ylabel("link rate (Mbps)", fontsize=8.0)
    ax_t.set_ylim(0.0, traffic.max() * 1.14)
    ax_t.set_title("Monitored uplink, anomaly score and analyst alerts on one "
                   "24 h axis", fontsize=9.0, fontweight="bold")
    for t0, t1, name, _s, _m in ATTACKS:
        ax_t.text(0.5 * (t0 + t1), traffic.max() * 1.07, name, ha="center",
                  va="top", fontsize=6.4, color=ACCENT)

    # ---- (b) anomaly score ------------------------------------------------
    ax_s.plot(t, score, color=RULE, lw=0.75, alpha=0.9)
    ax_s.fill_between(t, THR, score, where=score >= THR, color=ACCENT,
                      alpha=0.55, lw=0.0)
    ax_s.axhline(THR, color=GOLD, lw=1.1, ls="--")
    ax_s.text(9.6, THR + 0.035, f"decision threshold {THR:.2f}",
              fontsize=6.6, color=GOLD, va="bottom", ha="center")
    ax_s.set_ylabel("anomaly score (-)", fontsize=8.0)
    ax_s.set_ylim(0.0, 1.05)
    for t0, t1, _s, _m in BENIGN:
        ax_s.annotate("benign backup job", xy=(0.5 * (t0 + t1), 0.63),
                      xytext=(0.5 * (t0 + t1), 0.97), ha="center",
                      va="top", fontsize=6.2, color=SUB,
                      arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED))

    # ---- (c) alert stream -------------------------------------------------
    alerts = []                       # one raised alert every 7 flagged bins
    for tt in t[pred][::7]:
        if 6.4 < tt < 7.6:
            alerts.append((tt, 3, True))
        elif 13.0 < tt < 14.1:
            alerts.append((tt, 1, True))
        elif tt > 19.4:
            alerts.append((tt, 2, True))
        else:
            alerts.append((tt, 0, False))
    for k, (name, col) in enumerate(ALERT_TYPES):
        xs = [a[0] for a in alerts if a[1] == k and a[2]]
        ax_a.scatter(xs, np.full(len(xs), k), s=16, color=col, marker="o",
                     edgecolors="white", linewidths=0.45, zorder=4)
        xs = [a[0] for a in alerts if a[1] == k and not a[2]]
        ax_a.scatter(xs, np.full(len(xs), k), s=16, facecolors="none",
                     edgecolors=MUTED, linewidths=0.7, marker="o", zorder=4)
    ax_a.set_yticks(range(len(ALERT_TYPES)))
    ax_a.set_yticklabels([n for n, _c in ALERT_TYPES], fontsize=6.8)
    ax_a.set_ylim(-0.7, len(ALERT_TYPES) - 0.3)
    ax_a.set_xlabel("time of day (hours)", fontsize=8.0)
    ax_a.set_ylabel("alert class", fontsize=8.0)

    for ax in (ax_t, ax_s, ax_a):
        ax.set_xlim(0.0, 24.0)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.tick_params(labelsize=7.0, length=2.6)
        ax.grid(axis="x", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    for ax in (ax_t, ax_s):
        ax.tick_params(labelbottom=False)

    # The lane labels already name the classes, so the legend only has to
    # explain the one thing colour cannot say: which alerts were spurious.
    handles = [Line2D([], [], lw=0, marker="o", ms=3.4, mfc=ACCENT,
                      mec="white", label="Alert on a real attack"),
               Line2D([], [], lw=0, marker="o", ms=3.4, mfc="none",
                      mec=MUTED, label="False alarm")]
    ax_a.legend(handles=handles, loc="upper right",
                bbox_to_anchor=(1.0, 1.06), ncol=1, fontsize=6.6,
                frameon=False, handletextpad=0.35, labelspacing=0.35)

    # ---- (d) ROC ----------------------------------------------------------
    ax_r.plot(fpr, tpr, color=BLUE, lw=1.5)
    ax_r.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
    ax_r.plot([far], [det], marker="o", ms=5.0, mfc=ACCENT, mec="white",
              mew=0.9, zorder=5)
    ax_r.annotate(f"operating point\nTPR {det:.1%}, FPR {far:.1%}",
                  xy=(far, det), xytext=(far + 0.16, det - 0.26),
                  fontsize=6.6, color=ACCENT, linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
    ax_r.text(0.96, 0.06, f"AUC = {auc:.3f}", ha="right", va="bottom",
              fontsize=7.2, color=RULE)
    ax_r.set_xlim(0.0, 1.0)
    ax_r.set_ylim(0.0, 1.02)
    ax_r.set_xlabel("false positive rate (-)", fontsize=7.6)
    ax_r.set_ylabel("true positive rate (-)", fontsize=7.6)
    ax_r.tick_params(labelsize=6.8, length=2.4)
    ax_r.set_title("Detector ROC over the same day", fontsize=8.5,
                   fontweight="bold")
    ax_r.grid(lw=0.4, alpha=0.35)
    ax_r.set_axisbelow(True)

    # ---- (e) confusion matrix --------------------------------------------
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    frac = cm / cm.sum(axis=1, keepdims=True)
    ax_c.imshow(frac, cmap=get_cmap("cb_sequential_blue"), vmin=0.0, vmax=1.0,
                aspect="auto")
    for i in range(2):
        for j in range(2):
            ax_c.text(j, i, f"{cm[i, j]:.0f} min\n{frac[i, j]:.1%}",
                      ha="center", va="center", fontsize=6.8,
                      linespacing=1.25,
                      color="white" if frac[i, j] > 0.55 else "#1a1a1a")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["no alert", "alert"], fontsize=6.8)
    ax_c.set_yticks([0, 1])
    ax_c.set_yticklabels(["benign", "attack"], fontsize=6.8)
    ax_c.set_xlabel("detector decision", fontsize=7.6)
    ax_c.set_ylabel("ground truth", fontsize=7.6)
    ax_c.set_title("Minutes classified at the threshold", fontsize=8.5,
                   fontweight="bold")
    ax_c.tick_params(length=0.0)
    for s in ax_c.spines.values():
        s.set_visible(False)


    panel_labels_inches(fig, [ax_t, ax_s, ax_a, ax_r, ax_c],
                        dx=-0.44, dy=0.05)
    savefig(fig, "577_sec_intrusion_detection.png")


if __name__ == "__main__":
    main()
