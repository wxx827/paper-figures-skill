"""SEIR compartment curves with a basic-reproduction-number sweep.

Main panel: deterministic SEIR trajectories for one baseline R0, with the
infectious peak called out and the herd-immunity threshold 1 - 1/R0 drawn as a
level line so the final-size overshoot is visible. Auxiliary panel: peak
prevalence and time-to-peak across a range of R0 on twin axes.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

C_S = "#3b5b92"     # susceptible  - house blue
C_E = "#c9a227"     # exposed      - house gold
C_I = "#b23a48"     # infectious   - house red, the single emphasis
C_R = "#2e7d5b"     # recovered    - house green
RULE = "#333333"
MUTED = "#777777"
DASH = (0, (4.0, 2.4))          # one dash style for every threshold line

R0_BASE = 2.6
SIGMA = 1.0 / 3.4               # 1 / latent period (days)
GAMMA = 1.0 / 6.5               # 1 / infectious period (days)
DAYS = 240.0
DT = 0.05
I0 = 3e-4


def seir(r0, days=DAYS, dt=DT):
    """Forward-Euler SEIR on fractions of a closed population."""
    beta = r0 * GAMMA
    n = int(round(days / dt)) + 1
    t = np.arange(n) * dt
    S = np.empty(n); E = np.empty(n); I = np.empty(n); R = np.empty(n)
    S[0], E[0], I[0], R[0] = 1.0 - I0, 0.0, I0, 0.0
    for k in range(n - 1):
        inf = beta * S[k] * I[k]
        lat = SIGMA * E[k]
        rec = GAMMA * I[k]
        S[k + 1] = S[k] - dt * inf
        E[k + 1] = E[k] + dt * (inf - lat)
        I[k + 1] = I[k] + dt * (lat - rec)
        R[k + 1] = R[k] + dt * rec
    return t, S, E, I, R


def main():
    set_pub_style(base_font=9)
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58], wspace=0.34,
                          left=0.075, right=0.915, top=0.845, bottom=0.135)

    # ---- (a) baseline trajectory ------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    t, S, E, I, R = seir(R0_BASE)
    for y, c, lab in ((S, C_S, "Susceptible"), (E, C_E, "Exposed"),
                      (I, C_I, "Infectious"), (R, C_R, "Recovered")):
        ax.plot(t, 100 * y, color=c, lw=1.9 if c == C_I else 1.5, label=lab,
                zorder=4 if c == C_I else 3)

    k = int(np.argmax(I))
    t_pk, i_pk = t[k], 100 * I[k]
    ax.plot([t_pk], [i_pk], marker="o", ms=4.4, mfc="white", mec=C_I, mew=1.2,
            zorder=6)
    ax.annotate(f"peak {i_pk:.1f}% infectious\non day {t_pk:.0f}",
                xy=(t_pk, i_pk), xytext=(t_pk + 26, i_pk + 13),
                fontsize=7.0, color=C_I, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_I))

    hit = 100 * (1.0 - 1.0 / R0_BASE)
    ax.axhline(hit, color=RULE, ls=DASH, lw=0.9, zorder=2)
    ax.text(DAYS * 0.985, hit + 1.8,
            f"herd-immunity threshold $1-1/R_0$ = {hit:.0f}%",
            ha="right", va="bottom", fontsize=6.8, color=RULE)
    final = 100 * R[-1]
    t_hit = t[np.argmax(R >= hit / 100.0)]
    ax.annotate(f"final size {final:.0f}%\novershoot +{final - hit:.0f} pp",
                xy=(DAYS * 0.985, final), xytext=(DAYS * 0.985, final + 9),
                ha="right", fontsize=6.8, color=C_R, linespacing=1.35)
    ax.plot([t_hit, t_hit], [0, hit], color=RULE, ls=DASH, lw=0.7, zorder=2)
    ax.text(t_hit, 1.5, f"day {t_hit:.0f}", ha="center", va="bottom",
            fontsize=6.4, color=MUTED)

    ax.set_xlim(0, DAYS)
    ax.set_ylim(0, 104)
    ax.set_xlabel("Time since introduction (days)")
    ax.set_ylabel("Population in compartment (%)")
    ax.set_title(f"SEIR trajectory at $R_0$ = {R0_BASE:.1f}", fontsize=9.0,
                 fontweight="bold", loc="left", pad=4)
    ax.legend(loc="center right", fontsize=7.2, frameon=False,
              handlelength=1.5, borderaxespad=0.2,
              bbox_to_anchor=(1.0, 0.44))
    style_axes(ax)
    add_panel_label(ax, "(a)", x=-0.115, y=1.04)

    # ---- (b) R0 sweep ------------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    r0s = np.linspace(1.25, 4.0, 24)
    peaks, tpeaks = [], []
    for r0 in r0s:
        tt, _, _, ii, _ = seir(r0)
        j = int(np.argmax(ii))
        peaks.append(100 * ii[j])
        tpeaks.append(tt[j])
    peaks = np.asarray(peaks); tpeaks = np.asarray(tpeaks)

    axb.plot(r0s, peaks, color=C_I, lw=1.7, zorder=4)
    axb.set_xlabel("Basic reproduction number $R_0$ (-)")
    axb.set_ylabel("Peak prevalence (% infectious)", color=C_I)
    axb.tick_params(axis="y", colors=C_I, labelsize=6.8)
    axb.tick_params(axis="x", labelsize=6.8)
    axb.set_xlim(r0s[0], r0s[-1])
    axb.set_ylim(0, peaks.max() * 1.14)

    axt = axb.twinx()
    axt.plot(r0s, tpeaks, color=C_S, lw=1.7, ls=(0, (5.0, 1.8)), zorder=3)
    axt.set_ylabel("Time to peak (days)", color=C_S)
    axt.tick_params(axis="y", colors=C_S, labelsize=6.8)
    axt.set_ylim(0, tpeaks.max() * 1.14)
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_color(C_S)
    axt.spines["left"].set_visible(False)

    axb.axvline(R0_BASE, color=MUTED, ls=DASH, lw=0.8, zorder=1)
    axb.text(R0_BASE + 0.06, peaks.max() * 1.07, "baseline", fontsize=6.6,
             color=MUTED, ha="left", va="top")
    axb.set_title("Peak height and timing vs $R_0$", fontsize=8.5,
                  fontweight="bold", loc="left", pad=4)
    axb.spines["top"].set_visible(False)
    axb.tick_params(length=3)
    add_panel_label(axb, "(b)", x=-0.20, y=1.04)

    savefig(fig, "424_epi_sir_family.png")


if __name__ == "__main__":
    main()
