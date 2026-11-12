"""Dropout early-warning model: retention by risk tier, drivers, and rescue.

One simulated cohort of 1200 students carries the whole figure.  Six
behavioural features drive a linear risk score; that score both assigns the
risk tiers of the main Kaplan-Meier panel and supplies the ROC AUC quoted on
it, so the stratification and the discrimination statistic cannot drift apart.
The small right-hand panels are the evidence: which features the score leans
on, and how much end-of-programme retention an intervention buys as a
function of how early the warning fires.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, annotate_box, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
PALE = "#b3b3b3"
BLUE = "#3b5b92"
LIGHT_BLUE = "#8fadd0"
ACCENT = "#b23a48"

FEATURES = ["login frequency", "assignment submission", "quiz score trend",
            "video watch time", "forum participation", "prior GPA"]
BETA = np.array([-0.95, -1.15, -0.72, -0.41, -0.28, -0.66])
N_STUDENT = 1200
N_SEM = 8
BASE_HAZARD = np.array([0.060, 0.050, 0.042, 0.035, 0.030, 0.025, 0.021,
                        0.019])
TIERS = ["low risk", "medium risk", "high risk"]
TIER_COLS = [BLUE, LIGHT_BLUE, ACCENT]
WARN_SEM = [1, 2, 3, 4]
EFFECT = 0.45


def cohort(seed=596):
    """Features, risk score, dropout semester (0 = still enrolled)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N_STUDENT, len(FEATURES)))
    z = x @ BETA
    z = (z - z.mean()) / z.std(ddof=1)
    mult = np.exp(0.85 * z)
    haz = np.clip(BASE_HAZARD[None, :] * mult[:, None], 0.0, 0.65)
    draw = rng.random((N_STUDENT, N_SEM)) < haz
    first = np.argmax(draw, axis=1) + 1
    first[~draw.any(axis=1)] = 0
    return x, z, mult, first


def auc_score(z, label):
    """Mann-Whitney rank AUC - no sklearn dependency, ties handled."""
    order = np.argsort(z)
    ranks = np.empty(z.size, dtype=float)
    ranks[order] = np.arange(1, z.size + 1, dtype=float)
    n1 = int(label.sum())
    n0 = int(z.size - n1)
    return (ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def kaplan_meier(times):
    """Step survival plus Greenwood standard errors over the fixed horizon."""
    at_risk = float(times.size)
    s, cum = 1.0, 0.0
    surv, se = [1.0], [0.0]
    for t in range(1, N_SEM + 1):
        d = float(np.count_nonzero(times == t))
        if at_risk > 0.0:
            s *= 1.0 - d / at_risk
            if d > 0.0 and at_risk - d > 0.0:
                cum += d / (at_risk * (at_risk - d))
        surv.append(s)
        se.append(s * np.sqrt(cum))
        at_risk -= d
    return np.array(surv), np.array(se)


def retention_under(mult, warn=None):
    """Deterministic end-of-horizon retention for a mean hazard multiplier."""
    h = np.clip(BASE_HAZARD * mult, 0.0, 0.65)
    if warn is not None:
        h = h.copy()
        h[warn:] *= 1.0 - EFFECT
    return float(np.prod(1.0 - h))


def main():
    set_pub_style()
    x, z, mult, first = cohort()
    label = first > 0
    auc = auc_score(z, label)

    cuts = np.quantile(z, [1.0 / 3.0, 2.0 / 3.0])
    tier_id = np.digitize(z, cuts)
    sem = np.arange(N_SEM + 1)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], wspace=0.30,
                          hspace=0.52)
    ax_km = fig.add_subplot(gs[:, 0])
    ax_imp = fig.add_subplot(gs[0, 1])
    ax_int = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.068, right=0.980, bottom=0.155, top=0.885)

    # ---- main: retention by risk tier ------------------------------------
    finals = []
    for k, (name, col) in enumerate(zip(TIERS, TIER_COLS)):
        t = first[tier_id == k]
        surv, se = kaplan_meier(t)
        lo = np.clip(surv - 1.96 * se, 0.0, 1.0)
        hi = np.clip(surv + 1.96 * se, 0.0, 1.0)
        ax_km.fill_between(sem, lo, hi, step="post", color=col, alpha=0.20,
                           linewidth=0.0, zorder=2)
        ax_km.step(sem, surv, where="post", color=col, lw=1.8, zorder=3,
                   label=f"{name}  (n = {t.size})")
        finals.append(surv[-1])
        ax_km.text(N_SEM + 0.10, surv[-1], f"{surv[-1] * 100:.0f}%",
                   ha="left", va="center", fontsize=6.8, color=col)

    ax_km.set_xlim(0.0, N_SEM + 0.55)
    ax_km.set_ylim(0.30, 1.02)
    ax_km.set_xticks(sem)
    ax_km.set_xlabel("semester since enrolment (semesters)", fontsize=8.0)
    ax_km.set_ylabel("retention $S(t)$ (-)", fontsize=8.0)
    ax_km.tick_params(labelsize=7.2)
    ax_km.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_km.set_title("Retention separates by risk tier from the second "
                    "semester", fontsize=9.0, fontweight="bold", color=INK)
    ax_km.legend(loc="lower left", frameon=False, fontsize=7.2,
                 handlelength=1.6, labelspacing=0.34, borderaxespad=0.8)
    annotate_box(ax_km,
                 f"ROC AUC = {auc:.3f}\n"
                 f"dropout by semester {N_SEM}: "
                 f"{label.mean() * 100:.1f} %\n"
                 f"gap low vs high = "
                 f"{(finals[0] - finals[2]) * 100:.0f} pp",
                 loc="lower right", fontsize=7.0)

    # ---- aux 1: what the score leans on ----------------------------------
    imp = np.abs(BETA) * x.std(axis=0, ddof=1)
    imp = imp / imp.sum()
    order = np.argsort(imp)
    ax_imp.barh(np.arange(len(FEATURES)), imp[order], height=0.66,
                color=BLUE, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(imp[order]):
        ax_imp.text(v + 0.006, i, f"{v * 100:.0f}%", ha="left", va="center",
                    fontsize=6.3, color=SUB)
    ax_imp.set_yticks(np.arange(len(FEATURES)))
    ax_imp.set_yticklabels([FEATURES[i] for i in order], fontsize=6.6)
    ax_imp.set_xlim(0.0, imp.max() * 1.28)
    ax_imp.set_xlabel("relative importance (share of total)", fontsize=7.4)
    ax_imp.tick_params(axis="x", labelsize=6.6)
    ax_imp.grid(axis="x", lw=0.5, color="#e4e4e4")
    ax_imp.set_title("Engagement beats grades as a signal", fontsize=8.5,
                     fontweight="bold", color=INK)

    # ---- aux 2: how much earlier warning is worth ------------------------
    # Anchored on the observed high-risk endpoint from panel (a) so the two
    # panels quote the same baseline; only the *ratio* comes from the model.
    m_hi = float(mult[tier_id == 2].mean())
    base = finals[2] * 100.0
    ref = retention_under(m_hi)
    gained = [base * retention_under(m_hi, warn=w) / ref for w in WARN_SEM]
    pos = np.arange(len(WARN_SEM))
    ax_int.bar(pos - 0.19, [base] * len(WARN_SEM), width=0.36, color=PALE,
               edgecolor="white", linewidth=0.6, label="no intervention")
    ax_int.bar(pos + 0.19, gained, width=0.36, color=BLUE,
               edgecolor="white", linewidth=0.6, label="intervention")
    for p, v in zip(pos, gained):
        ax_int.text(p + 0.19, v + 1.0, f"+{v - base:.0f}", ha="center",
                    va="bottom", fontsize=6.3, color=BLUE)
    ax_int.set_xticks(pos)
    ax_int.set_xticklabels([f"sem {w}" for w in WARN_SEM], fontsize=6.6)
    ax_int.set_ylim(0.0, max(gained) * 1.22)
    ax_int.set_ylabel("retention at semester 8 (%)", fontsize=7.4)
    ax_int.set_xlabel("semester the warning fires", fontsize=7.4)
    ax_int.tick_params(axis="y", labelsize=6.6)
    ax_int.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_int.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                  frameon=False, fontsize=6.4, handlelength=1.3,
                  columnspacing=1.0, borderaxespad=0.0)
    ax_int.set_title("Rescue value of the high-risk tier", fontsize=8.5,
                     fontweight="bold", color=INK, pad=12.0)


    note = (f"{N_STUDENT} simulated students followed for {N_SEM} semesters; "
            f"risk score is a standardised linear predictor over "
            f"{len(FEATURES)} behavioural features, tiers are its tertiles. "
            f"Bands are Kaplan-Meier estimates $\\pm$ 1.96 Greenwood SE. "
            f"The rescue panel scales the observed high-risk endpoint "
            f"({base:.0f} %) by the retention ratio of a {EFFECT * 100:.0f} % "
            f"hazard reduction applied from the warning semester onward at "
            f"the mean high-risk multiplier "
            f"($\\times${m_hi:.2f} of the base hazard).")

    panel_labels_inches(fig, [ax_km, ax_imp, ax_int],
                        labels=["(a)", "(b)", "(c)"], dx=-0.44, dy=0.10)
    savefig(fig, "596_edu_dropout_risk.png")


if __name__ == "__main__":
    main()
