"""Counterfactual intervention scenarios against hospital surge capacity.

Main panel: hospital occupancy under four SEIHR scenarios, with the surge
capacity drawn as a level line and the load above it filled in red, so the
overshoot rather than the peak height is what the eye picks up. Auxiliary
panels: the same four scenarios summarised as cumulative infections, deaths
and peak load, each on its own axis because the three quantities have
different units.
"""
import numpy as np
import matplotlib.pyplot as plt
from figkit import panel_label_inches
from pubstyle import set_pub_style, savefig, style_axes

SCEN = ["No intervention", "Case isolation", "Vaccination", "Combined"]
SCEN_C = ["#b23a48", "#c9a227", "#3b5b92", "#2e7d5b"]
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
ALERT = "#b23a48"
DASH = (0, (4.0, 2.4))

R0 = 2.8
SIGMA = 1 / 3.5              # 1 / latent period (days)
GAMMA = 1 / 5.5              # 1 / infectious period (days)
P_HOSP = 0.012               # infections needing a hospital bed
DELTA = 1 / 8.0              # 1 / length of stay (days)
HFR = 0.15                   # deaths among the hospitalised
CAPACITY = 90.0              # surge beds per 100k
DAYS, DT = 300.0, 0.05
I0 = 2e-4
ISO_DAY, ISO_CUT = 35.0, 0.45        # isolation removes 45% of transmission
VAX_DAY, VAX_RATE, VE = 15.0, 0.0115, 0.90


def run(isolation=False, vaccination=False):
    n = int(DAYS / DT) + 1
    t = np.arange(n) * DT
    S = np.empty(n); E = np.empty(n); I = np.empty(n)
    H = np.empty(n); R = np.empty(n); D = np.empty(n); C = np.empty(n)
    S[0], E[0], I[0] = 1.0 - I0, 0.0, I0
    H[0] = R[0] = D[0] = C[0] = 0.0
    beta0 = R0 * GAMMA
    for k in range(n - 1):
        beta = beta0
        if isolation:
            beta *= 1.0 - ISO_CUT / (1.0 + np.exp(-(t[k] - ISO_DAY) / 2.5))
        vax = (VAX_RATE * VE * S[k] if (vaccination and t[k] > VAX_DAY)
               else 0.0)
        inf = beta * S[k] * I[k]
        lat, rec = SIGMA * E[k], GAMMA * I[k]
        out = DELTA * H[k]
        S[k + 1] = S[k] - DT * (inf + vax)
        E[k + 1] = E[k] + DT * (inf - lat)
        I[k + 1] = I[k] + DT * (lat - rec)
        H[k + 1] = H[k] + DT * (P_HOSP * rec - out)
        R[k + 1] = R[k] + DT * ((1 - P_HOSP) * rec + (1 - HFR) * out + vax)
        D[k + 1] = D[k] + DT * HFR * out
        C[k + 1] = C[k] + DT * lat
    return dict(t=t, beds=H * 1e5, attack=100 * C, deaths=D * 1e5)


def main():
    set_pub_style(base_font=9)
    runs = [run(), run(isolation=True), run(vaccination=True),
            run(isolation=True, vaccination=True)]

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.55], hspace=0.72,
                          wspace=0.24, left=0.082, right=0.955, top=0.845,
                          bottom=0.115)
    ax = fig.add_subplot(gs[:, 0])
    aux = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # ---- (a) hospital occupancy --------------------------------------------
    for r, name, col in zip(runs, SCEN, SCEN_C):
        ax.plot(r["t"], r["beds"], color=col, lw=1.8, zorder=5, label=name)
        ax.fill_between(r["t"], CAPACITY, r["beds"],
                        where=r["beds"] > CAPACITY, interpolate=True,
                        color=ALERT, alpha=0.16, lw=0, zorder=3)
    ax.axhline(CAPACITY, color=RULE, ls=DASH, lw=1.0, zorder=6)
    ax.text(DAYS * 0.985, CAPACITY + 6, f"surge capacity {CAPACITY:.0f} beds "
            "per 100k", fontsize=6.8, color=RULE, ha="right", va="bottom")

    worst = runs[0]
    k = int(np.argmax(worst["beds"]))
    over = float((worst["beds"] > CAPACITY).sum() * DT)
    ax.annotate(f"peak {worst['beds'][k]:.0f} beds per 100k\n"
                f"= {worst['beds'][k] / CAPACITY:.1f}x capacity, "
                f"{over:.0f} days above",
                xy=(worst["t"][k], worst["beds"][k]),
                xytext=(worst["t"][k] + 40, worst["beds"][k] * 0.98),
                fontsize=6.8, color=ALERT, linespacing=1.35, va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ALERT))
    best = runs[3]
    b_pk = float(best["beds"].max())
    ax.annotate("combined package peaks at\n"
                f"{b_pk:.0f} beds = {100 * b_pk / CAPACITY:.0f}% of capacity",
                xy=(best["t"][int(np.argmax(best["beds"]))], b_pk),
                xytext=(140, CAPACITY * 1.75), fontsize=6.8, color=SCEN_C[3],
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=SCEN_C[3]))

    ax.set_xlim(0, DAYS)
    ax.set_ylim(0, max(r["beds"].max() for r in runs) * 1.12)
    ax.set_xlabel("Days since first case (days)")
    ax.set_ylabel("Hospital beds occupied (per 100k)")
    ax.tick_params(labelsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=4,
              fontsize=7.0, frameon=False, handlelength=1.5,
              columnspacing=1.4, borderaxespad=0.0)
    style_axes(ax)

    # ---- (b-d) scenario summaries ------------------------------------------
    metrics = [("Cumulative infections (%)",
                [r["attack"][-1] for r in runs], "{:.0f}%"),
               ("Cumulative deaths (per 100k)",
                [r["deaths"][-1] for r in runs], "{:.0f}"),
               ("Peak hospital load (per 100k)",
                [r["beds"].max() for r in runs], "{:.0f}")]
    for axx, (title, vals, fmt) in zip(aux, metrics):
        ypos = np.arange(len(SCEN))[::-1]
        axx.barh(ypos, vals, height=0.66, color=SCEN_C, zorder=3)
        for y, v in zip(ypos, vals):
            axx.text(v + max(vals) * 0.03, y, fmt.format(v), va="center",
                     ha="left", fontsize=6.4, color=INK)
        axx.set_xlim(0, max(vals) * 1.26)
        axx.set_ylim(-0.65, len(SCEN) - 0.05)
        axx.set_yticks([])
        axx.set_title(title, fontsize=7.6, fontweight="bold", loc="left",
                      pad=3)
        axx.tick_params(axis="x", labelsize=6.4)
        axx.tick_params(axis="y", length=0)
        style_axes(axx)
    aux[2].axvline(CAPACITY, color=RULE, ls=DASH, lw=0.9, zorder=5)
    aux[2].text(CAPACITY + 5, len(SCEN) - 0.10, "capacity", fontsize=6.2,
                color=RULE, ha="left", va="top")

    panel_label_inches(fig, ax, "(a)", dx=-0.44, dy=0.20)
    for axx, lab in zip(aux, ["(b)", "(c)", "(d)"]):
        panel_label_inches(fig, axx, lab, dx=-0.30, dy=0.05)

    savefig(fig, "433_epi_intervention_scenarios.png")


if __name__ == "__main__":
    main()
