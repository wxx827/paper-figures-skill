"""ADMET profile of three clinical candidates against a marketed reference:
a six-axis desirability radar with the target band shaded, the per-property
scores that build it, and the Lipinski rule-of-five verdict for each series."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GREY = "#808080"

TARGET_LO, TARGET_HI = 60.0, 100.0     # desirable band on every axis

# compound, colour, raw ADMET readouts
#   Papp (1e-6 cm/s), Vd (L/kg), microsomal t1/2 (min), CL (mL/min/kg),
#   hERG IC50 (uM), kinetic solubility (uM)
COMPOUNDS = [
    ("AZ-114, lead", BLUE, dict(papp=22.0, vd=1.4, thalf=48.0, cl=12.0,
                                herg=28.0, sol=120.0,
                                mw=421.0, logp=3.4, hbd=1, hba=6, tpsa=78.0)),
    ("AZ-207, backup", GREEN, dict(papp=14.0, vd=0.9, thalf=33.0, cl=18.0,
                                   herg=15.0, sol=85.0,
                                   mw=388.0, logp=2.6, hbd=2, hba=5,
                                   tpsa=92.0)),
    ("AZ-233, tox flag", RED, dict(papp=31.0, vd=4.8, thalf=9.0, cl=41.0,
                                   herg=1.6, sol=22.0,
                                   mw=552.0, logp=5.6, hbd=3, hba=9,
                                   tpsa=118.0)),
    ("Reference drug", GREY, dict(papp=8.0, vd=2.1, thalf=55.0, cl=9.0,
                                  herg=40.0, sol=210.0,
                                  mw=464.0, logp=4.1, hbd=2, hba=7,
                                  tpsa=86.0)),
]

AXES = [("Absorption", "Caco-2 $P_{app}$ (10$^{-6}$ cm/s)"),
        ("Distribution", "$V_d$ (L/kg)"),
        ("Metabolism", "microsomal $T_{1/2}$ (min)"),
        ("Excretion", "CL (mL/min/kg)"),
        ("Safety", "hERG IC$_{50}$ ($\\mu$M)"),
        ("Solubility", "kinetic solubility ($\\mu$M)")]

RO5 = [("MW (Da)", "mw", 500.0, "{:.0f}"),
       ("clogP (-)", "logp", 5.0, "{:.1f}"),
       ("HBD (n)", "hbd", 5.0, "{:.0f}"),
       ("HBA (n)", "hba", 10.0, "{:.0f}"),
       ("TPSA (\u00c5$^2$)", "tpsa", 140.0, "{:.0f}")]


def desirability(p):
    """Map raw ADMET readouts onto a common 0-100 desirability scale."""
    d_abs = 100.0 * min(p["papp"] / 25.0, 1.0)
    d_dis = 100.0 * float(np.exp(-((np.log10(p["vd"]) - np.log10(1.8))
                                   / 0.42) ** 2))
    d_met = 100.0 * min(p["thalf"] / 60.0, 1.0)
    d_exc = 100.0 * float(np.clip((45.0 - p["cl"]) / 40.0, 0.0, 1.0))
    d_saf = 100.0 * float(np.clip(np.log10(p["herg"]) / np.log10(40.0),
                                  0.0, 1.0))
    d_sol = 100.0 * float(np.clip(np.log10(p["sol"] / 5.0)
                                  / np.log10(200.0 / 5.0), 0.0, 1.0))
    return np.array([d_abs, d_dis, d_met, d_exc, d_saf, d_sol])


def ro5_violations(p):
    return sum(1 for _lab, key, lim, _f in RO5[:4] if p[key] > lim)


def radar_panel(ax, scores):
    n = len(AXES)
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    loop = np.concatenate([th, th[:1]])
    dense = np.linspace(0.0, 2 * np.pi, 361)

    ax.fill_between(dense, TARGET_LO, TARGET_HI, color=lighten(GREEN, 0.90),
                    edgecolor="none", zorder=1)
    ax.plot(dense, np.full_like(dense, TARGET_LO), color=lighten(GREEN, 0.45),
            lw=0.8, ls=(0, (4, 2.5)), zorder=2)

    for name, col, _p in COMPOUNDS:
        s = scores[name]
        v = np.concatenate([s, s[:1]])
        ax.plot(loop, v, color=col, lw=1.6, zorder=5)
        ax.fill(loop, v, color=col, alpha=0.13, zorder=3)
        ax.plot(th, s, ls="none", marker="o", ms=3.2, mfc="white", mec=col,
                mew=1.1, zorder=6, label=f"{name}   mean {s.mean():.0f}")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Polar tick labels inherit the axes clip box, so the right-hand ones lose
    # their first letter; place them by hand outside the rim instead.
    ax.set_thetagrids(np.degrees(th), [""] * n)
    for a, (name, _unit) in zip(th, AXES):
        sx, cy = np.sin(a), np.cos(a)
        ha = "center" if abs(sx) < 0.15 else ("left" if sx > 0 else "right")
        va = "center" if abs(cy) < 0.15 else ("bottom" if cy > 0 else "top")
        ax.text(a, 109.0, name, ha=ha, va=va, fontsize=7.6, color=INK,
                clip_on=False, zorder=8)
    ax.set_rlim(0, 100)
    ax.set_rgrids([20, 40, 60, 80, 100],
                  labels=["20", "40", "60", "80", "100"], angle=90.0,
                  fontsize=6.2, color=MUTED)
    ax.set_rlabel_position(96.0)
    ax.grid(True, color="#dfe3e7", lw=0.6)
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title("ADMET desirability, 0-100 scale per axis", fontsize=9.0,
                 fontweight="bold", pad=26)
    ax.text(np.deg2rad(90), 78, "target band", fontsize=6.2,
            color=lighten(GREEN, 0.20), ha="center", va="bottom", rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2,
              fontsize=6.8, frameon=False, handlelength=1.4,
              columnspacing=1.4, labelspacing=0.35)


def score_panel(ax, scores):
    n = len(AXES)
    y = np.arange(n)[::-1]
    h = 0.19
    offs = np.linspace(1.5, -1.5, len(COMPOUNDS)) * h
    for (name, col, _p), off in zip(COMPOUNDS, offs):
        ax.barh(y + off, scores[name], height=h, color=col,
                edgecolor="white", linewidth=0.4, zorder=3, label=name)
    ax.axvspan(TARGET_LO, TARGET_HI, color=lighten(GREEN, 0.88), zorder=1)
    ax.axvline(TARGET_LO, color=lighten(GREEN, 0.45), lw=0.8,
               ls=(0, (4, 2.5)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([a[0] for a in AXES], fontsize=6.6)
    ax.set_xlim(0, 104)
    ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
    ax.set_xlabel("Desirability (%), target $\\geq$ 60", fontsize=7.5)
    ax.set_title("Per-property score", fontsize=8.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def ro5_panel(ax, scores):
    ncol = len(RO5) + 1
    nrow = len(COMPOUNDS)
    ax.set_xlim(0, ncol)
    ax.set_ylim(-0.95, nrow)
    ax.axis("off")

    for j, (lab, _k, lim, _f) in enumerate(RO5):
        ax.text(j + 0.5, nrow - 0.06, lab, ha="center", va="bottom",
                fontsize=6.3, color=INK)
        ax.text(j + 0.5, nrow - 0.32, f"$\\leq$ {lim:g}", ha="center",
                va="bottom", fontsize=6.1, color=MUTED)
    ax.text(ncol - 0.5, nrow - 0.06, "Verdict", ha="center", va="bottom",
            fontsize=6.3, color=INK)

    for i, (name, col, p) in enumerate(COMPOUNDS):
        y = nrow - 1 - i
        ax.text(-0.14, y + 0.5, name, ha="right", va="center", fontsize=6.5,
                color=INK)
        ax.add_patch(Rectangle((-0.10, y + 0.10), 0.06, 0.80, facecolor=col,
                               edgecolor="none", zorder=3))
        for j, (_lab, key, lim, fmt) in enumerate(RO5):
            bad = p[key] > lim
            ax.add_patch(Rectangle(
                (j + 0.06, y + 0.10), 0.88, 0.80,
                facecolor=lighten(RED, 0.80) if bad else lighten(BLUE, 0.88),
                edgecolor="white", linewidth=0.8, zorder=2))
            ax.text(j + 0.5, y + 0.50, fmt.format(p[key]), ha="center",
                    va="center", fontsize=6.4, color=RED if bad else INK,
                    fontweight="bold" if bad else "normal", zorder=4)
        nv = ro5_violations(p)
        ax.add_patch(Rectangle(
            (ncol - 0.94, y + 0.10), 0.88, 0.80,
            facecolor=lighten(RED, 0.80) if nv else lighten(GREEN, 0.86),
            edgecolor="white", linewidth=0.8, zorder=2))
        ax.text(ncol - 0.5, y + 0.50,
                "pass" if nv == 0 else f"{nv} fail",
                ha="center", va="center", fontsize=6.2,
                color=RED if nv else GREEN, zorder=4)

    ax.text(0.0, -0.30, "Lipinski rule of five uses the first four columns; "
                        "TPSA is the Veber criterion (Å$^2$)",
            fontsize=6.0, color=MUTED, ha="left", va="top")
    ax.set_title("Drug-likeness verdict", fontsize=8.5, fontweight="bold",
                 pad=12)


def main():
    set_pub_style()
    rng = np.random.default_rng(9518)
    scores = {}
    for name, col, p in COMPOUNDS:
        s = desirability(p)
        # analytical scores carry a small assay-to-assay spread
        s = np.clip(s + rng.normal(0.0, 1.6, s.size), 0.0, 100.0)
        scores[name] = s

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86], wspace=0.44, hspace=0.72,
                          left=0.075, right=0.965, top=0.870, bottom=0.135)
    ax_r = fig.add_subplot(gs[:, 0], projection="polar")
    # leave a rim of white around the circle for the hand-placed axis labels
    pr = ax_r.get_position()
    ax_r.set_position([pr.x0 + 0.012, pr.y0 + 0.020, pr.width * 0.90,
                       pr.height * 0.90])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])

    radar_panel(ax_r, scores)
    score_panel(ax_b, scores)
    ro5_panel(ax_t, scores)

    lead = COMPOUNDS[0]
    tox = COMPOUNDS[2]
    s_lead = scores[lead[0]]
    s_tox = scores[tox[0]]
    worst_axis = AXES[int(np.argmin(s_tox))][0].lower()

    panel_labels_inches(fig, [ax_r, ax_b, ax_t], dx=-0.42, dy=0.12)
    savefig(fig, "518_drug_admet_radar.png")


if __name__ == "__main__":
    main()
