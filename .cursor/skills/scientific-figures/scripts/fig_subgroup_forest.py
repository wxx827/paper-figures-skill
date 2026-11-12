"""Subgroup forest plot with event tables, interaction tests and overall diamond."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import stats

from pubstyle import set_pub_style, savefig

INK = "#1a1a1a"
SQ = "#2f5d8a"
DIA = "#16324f"
BAND = "#f3f6f9"

N_TOTAL = 912          # matches the CONSORT population
HR_LOW_UACR = 0.92     # little benefit without albuminuria
HR_HIGH_UACR = 0.55    # strong benefit in the albuminuric stratum


def build(rng):
    """Patient-level simulation, then subgroup risk ratios from the counts."""
    n = N_TOTAL
    arm = np.zeros(n, dtype=int)
    arm[rng.permutation(n)[: n // 2]] = 1          # 1 = intervention

    age65 = rng.random(n) < 0.55
    male = rng.random(n) < 0.62
    egfr = rng.choice([0, 1, 2], size=n, p=[0.30, 0.38, 0.32])   # <45,45-59,>=60
    dm = rng.random(n) < 0.66
    uacr_hi = rng.random(n) < 0.58
    region = rng.choice([0, 1, 2], size=n, p=[0.30, 0.42, 0.28])

    p_ctrl = np.clip(0.15 + 0.10 * (egfr == 0) + 0.04 * (egfr == 1)
                     + 0.05 * dm + 0.09 * uacr_hi + 0.03 * age65, 0.05, 0.60)
    hr_i = np.where(uacr_hi, HR_HIGH_UACR, HR_LOW_UACR)
    p = np.where(arm == 1, p_ctrl * hr_i, p_ctrl)
    event = rng.random(n) < p

    subgroups = [
        ("Age", [("< 65 years", ~age65), ("\u2265 65 years", age65)]),
        ("Sex", [("Male", male), ("Female", ~male)]),
        ("Baseline eGFR (mL/min/1.73 m$^2$)",
         [("< 45", egfr == 0), ("45 to 59", egfr == 1), ("\u2265 60", egfr == 2)]),
        ("Type 2 diabetes", [("Yes", dm), ("No", ~dm)]),
        ("Baseline UACR (mg/g)",
         [("< 300", ~uacr_hi), ("\u2265 300", uacr_hi)]),
        ("Region", [("North America", region == 0), ("Europe", region == 1),
                    ("Asia-Pacific", region == 2)]),
    ]

    def rr(mask):
        it, ic = mask & (arm == 1), mask & (arm == 0)
        a, na = int(event[it].sum()), int(it.sum())
        b, nb = int(event[ic].sum()), int(ic.sum())
        logrr = np.log((a / na) / (b / nb))
        se = np.sqrt(1 / a - 1 / na + 1 / b - 1 / nb)
        return dict(a=a, na=na, b=b, nb=nb, hr=np.exp(logrr),
                    lo=np.exp(logrr - 1.96 * se), hi=np.exp(logrr + 1.96 * se),
                    log=logrr, se=se, n=na + nb)

    out = []
    for name, levels in subgroups:
        stats_ = [(lab, rr(m)) for lab, m in levels]
        logs = np.array([s["log"] for _, s in stats_])
        ses = np.array([s["se"] for _, s in stats_])
        w = 1.0 / ses ** 2
        q = float(np.sum(w * (logs - np.sum(w * logs) / np.sum(w)) ** 2))
        p_int = float(stats.chi2.sf(q, len(stats_) - 1))
        out.append((name, stats_, p_int))
    return out, rr(np.ones(n, dtype=bool))


def main():
    set_pub_style()
    rng = np.random.default_rng(2024)
    groups, overall = build(rng)

    rows = []                                   # (kind, payload)
    for name, levels, p_int in groups:
        rows.append(("header", (name, p_int)))
        for lab, s in levels:
            rows.append(("level", (lab, s)))
        rows.append(("gap", None))
    rows.pop()
    rows.append(("rule", None))
    rows.append(("overall", overall))
    nrow = len(rows)

    fig, (axl, axm, axr) = plt.subplots(
        1, 3, figsize=(7.4, 6.9), sharey=True,
        gridspec_kw={"width_ratios": [2.90, 1.70, 1.25], "wspace": 0.0})
    fig.subplots_adjust(left=0.012, right=0.988, top=0.885, bottom=0.075)

    for ax in (axl, axm, axr):
        ax.set_ylim(-nrow - 3.4, 1.9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_yticks([])
    axl.set_xlim(0, 1)
    axr.set_xlim(0, 1)
    axl.set_xticks([])
    axr.set_xticks([])

    # ---- alternating bands over the whole width ---------------------------
    blocks, y0 = [], None
    for i, (kind, _) in enumerate(rows):
        if kind == "header":
            y0 = i
        elif kind in ("gap", "rule") and y0 is not None:
            blocks.append((y0, i))
            y0 = None
    if y0 is not None:
        blocks.append((y0, nrow))
    for k, (a, b) in enumerate(blocks):
        if k % 2 == 0:
            for ax in (axl, axm, axr):
                ax.axhspan(-b + 0.5, -a + 0.5, color=BAND, zorder=0)

    # ---- forest axis -------------------------------------------------------
    axm.set_xscale("log")
    axm.set_xlim(0.30, 3.3)
    axm.xaxis.set_major_locator(plt.NullLocator())
    axm.xaxis.set_minor_locator(plt.NullLocator())
    axm.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ylo, yhi = axm.get_ylim()
    ybase = -nrow + 0.05

    def yf(v):
        return (v - ylo) / (yhi - ylo)

    axm.axvline(1.0, ymin=yf(ybase), ymax=yf(0.90), color="#8a8a8a", lw=0.9,
                ls=(0, (4, 3)), zorder=2)
    wmax = max(1.0 / payload[1]["se"] ** 2
               for kind, payload in rows if kind == "level")

    for i, (kind, payload) in enumerate(rows):
        y = -i
        if kind == "header":
            name, p_int = payload
            axl.text(0.015, y, name, ha="left", va="center", fontsize=7.0,
                     fontweight="bold", color=INK)
            axr.text(0.995, y, f"{p_int:.2f}" if p_int >= 0.01 else "< 0.01",
                     ha="right", va="center", fontsize=6.8, color=INK,
                     fontweight="bold" if p_int < 0.05 else "normal")
        elif kind == "level":
            lab, s = payload
            axl.text(0.055, y, lab, ha="left", va="center", fontsize=6.8,
                     color=INK)
            axl.text(0.620, y, f"{s['n']}", ha="right", va="center",
                     fontsize=6.8, color=INK)
            axl.text(0.810, y, f"{s['a']}/{s['na']}", ha="right", va="center",
                     fontsize=6.8, color=INK)
            axl.text(0.998, y, f"{s['b']}/{s['nb']}", ha="right", va="center",
                     fontsize=6.8, color=INK)
            axm.plot([s["lo"], s["hi"]], [y, y], lw=1.0, color="#3f3f3f",
                     zorder=3, solid_capstyle="butt")
            for b in (s["lo"], s["hi"]):
                axm.plot([b, b], [y - 0.22, y + 0.22], lw=1.0, color="#3f3f3f",
                         zorder=3)
            size = 16 + 150 * ((1.0 / s["se"] ** 2) / wmax)
            axm.scatter(s["hr"], y, s=size, marker="s", color=SQ,
                        edgecolor="white", linewidth=0.5, zorder=4)
            axr.text(0.02, y,
                     f"{s['hr']:.2f} ({s['lo']:.2f}\u2013{s['hi']:.2f})",
                     ha="left", va="center", fontsize=6.8, color=INK)
        elif kind == "rule":
            for ax in (axl, axm, axr):
                ax.axhline(y + 0.15, color="#b8b8b8", lw=0.7, zorder=3,
                           clip_on=False)
        elif kind == "overall":
            s = payload
            axl.text(0.015, y, "Overall", ha="left", va="center", fontsize=7.2,
                     fontweight="bold", color=INK)
            axl.text(0.620, y, f"{s['n']}", ha="right", va="center",
                     fontsize=6.9, fontweight="bold", color=INK)
            axl.text(0.810, y, f"{s['a']}/{s['na']}", ha="right", va="center",
                     fontsize=6.9, fontweight="bold", color=INK)
            axl.text(0.998, y, f"{s['b']}/{s['nb']}", ha="right", va="center",
                     fontsize=6.9, fontweight="bold", color=INK)
            axm.add_patch(Polygon(
                [(s["lo"], y), (s["hr"], y + 0.34), (s["hi"], y),
                 (s["hr"], y - 0.34)], closed=True, facecolor=DIA,
                edgecolor="white", linewidth=0.6, zorder=5))
            axr.text(0.02, y,
                     f"{s['hr']:.2f} ({s['lo']:.2f}\u2013{s['hi']:.2f})",
                     ha="left", va="center", fontsize=6.9, fontweight="bold",
                     color=INK)
            axm.axvline(s["hr"], color=DIA, lw=0.8, ls=(0, (2, 2)), alpha=0.6,
                        zorder=1, ymin=yf(ybase), ymax=yf(0.90))

    # ---- column headers ----------------------------------------------------
    hy = 1.35
    axl.text(0.015, hy, "Subgroup", ha="left", va="center", fontsize=7.0,
             fontweight="bold")
    for x, lab in ((0.620, "No. of\npatients"),
                   (0.810, "Events\nintervention"),
                   (0.998, "Events\nplacebo")):
        axl.text(x, hy, lab, ha="right", va="center", fontsize=6.6,
                 fontweight="bold", linespacing=1.4)
    axr.text(0.02, hy, "Hazard ratio\n(95% CI)", ha="left", va="center",
             fontsize=6.6, fontweight="bold", linespacing=1.4)
    axr.text(0.995, hy, "P for\ninteraction", ha="right", va="center",
             fontsize=6.6, fontweight="bold", linespacing=1.4)
    for ax in (axl, axm, axr):
        ax.axhline(0.72, color="#333333", lw=0.8, zorder=3, clip_on=False)

    # ---- forest scale ------------------------------------------------------
    ticks = [0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    axm.plot([0.30, 3.3], [ybase, ybase], lw=0.8, color="#333333",
             clip_on=False)
    for t in ticks:
        axm.plot([t, t], [ybase, ybase - 0.16], lw=0.8, color="#333333",
                 clip_on=False)
        axm.text(t, ybase - 0.34, f"{t:g}", ha="center", va="top",
                 fontsize=6.6, color=INK, clip_on=False)
    axm.text(1.0, ybase - 1.15, "Hazard ratio (95% CI)", ha="center",
             va="top", fontsize=7.0, fontweight="bold", clip_on=False)
    axm.annotate("", xy=(0.34, ybase - 1.95), xytext=(0.95, ybase - 1.95),
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#2f5d8a"),
                 annotation_clip=False)
    axm.annotate("", xy=(2.9, ybase - 1.95), xytext=(1.05, ybase - 1.95),
                 arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#b23a48"),
                 annotation_clip=False)
    axm.text(0.55, ybase - 2.35, "Favours\nintervention", ha="center",
             va="top", fontsize=6.4, color="#2f5d8a", fontweight="bold",
             linespacing=1.4, clip_on=False)
    axm.text(1.80, ybase - 2.35, "Favours\nplacebo", ha="center", va="top",
             fontsize=6.4, color="#b23a48", fontweight="bold",
             linespacing=1.4, clip_on=False)


    savefig(fig, "117_subgroup_forest.png")


if __name__ == "__main__":
    main()
