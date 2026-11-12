"""Single- and multiple-dose pharmacokinetics of an oral candidate: plasma
concentration on a semi-log axis with a two-compartment fit and the derived
exposure metrics, the q12h accumulation to steady state below it, and the
dose-proportionality check beside them."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, style_axes, lighten, darken
from figkit import panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
BASE = "#808080"

TIMES = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72],
                 dtype=float)
N_SUBJ = 6
TAU = 12.0                     # dosing interval for the multiple-dose panel
MEC = 45.0                     # minimum efficacious concentration, ng/mL

# dose (mg/kg), bioavailable scale (ng/mL per mg/kg), colour
DOSES = [(5.0, 11.4, lighten(BLUE, 0.52)),
         (15.0, 11.0, BLUE),
         (50.0, 9.4, darken(BLUE, 0.34))]

KA, K1, K2, W1 = 1.35, 0.28, 0.041, 0.62   # true absorption / disposition


def pk_model(t, scale, ka, k1, k2, w1):
    """Oral two-compartment profile: fast distribution, slow terminal phase."""
    t = np.asarray(t, dtype=float)
    return scale * (w1 * np.exp(-k1 * t) + (1 - w1) * np.exp(-k2 * t)
                    - np.exp(-ka * t))


def observed(rng, dose, scale):
    """Mean +- SD across N_SUBJ animals, log-normal residual error."""
    truth = pk_model(TIMES, dose * scale, KA, K1, K2, W1)
    reps = truth[None, :] * np.exp(rng.normal(0.0, 0.20, (N_SUBJ, TIMES.size)))
    return reps.mean(axis=0), reps.std(axis=0, ddof=1), reps


def fit_profile(mean):
    p0 = [mean.max() * 2.2, KA, K1, K2, W1]
    bounds = ([mean.max() * 0.3, 0.2, 0.05, 0.005, 0.2],
              [mean.max() * 20.0, 6.0, 1.2, 0.30, 0.95])
    popt, _ = curve_fit(pk_model, TIMES, mean, p0=p0, bounds=bounds,
                        maxfev=40000)
    return popt


def metrics(mean, popt):
    """Cmax / Tmax from the observations, terminal slope from the fit."""
    i = int(np.argmax(mean))
    lam = popt[3]
    thalf = np.log(2.0) / lam
    auc_t = float(np.trapezoid(mean, TIMES))
    auc_inf = auc_t + mean[-1] / lam
    return dict(cmax=mean[i], tmax=TIMES[i], thalf=thalf, auc=auc_inf,
                lam=lam)


def single_dose_panel(ax, data):
    for (dose, _s, col), d in data.items():
        ax.errorbar(TIMES, d["mean"], yerr=d["sd"], ls="none", marker="o",
                    ms=3.4, mfc="white", mec=col, mew=1.0, ecolor=col,
                    elinewidth=0.8, capsize=1.8, zorder=5)
        tg = np.linspace(0.05, TIMES[-1], 400)
        ax.plot(tg, pk_model(tg, *d["popt"]), color=col, lw=1.5, zorder=4,
                label=f"{dose:g} mg/kg   $C_{{max}}$ {d['m']['cmax']:.0f} "
                      f"ng/mL   $AUC_{{0-\\infty}}$ {d['m']['auc']:.0f} "
                      f"ng$\\cdot$h/mL")

    ax.axhline(MEC, color=RED, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.text(TIMES[-1], MEC * 1.12, f"MEC {MEC:g} ng/mL", ha="right",
            va="bottom", fontsize=6.3, color=RED)

    mid = DOSES[1]
    dm = data[mid]["m"]
    ax.annotate(f"$T_{{max}}$ = {dm['tmax']:g} h", xy=(dm["tmax"], dm["cmax"]),
                xytext=(dm["tmax"] + 4.5, dm["cmax"] * 2.3), fontsize=6.4,
                color=SUB, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                shrinkA=1, shrinkB=2))
    t_h = 52.0
    ax.annotate(f"terminal $T_{{1/2}}$ = {dm['thalf']:.1f} h",
                xy=(t_h, pk_model(t_h, *data[mid]["popt"])),
                xytext=(37.0, 170.0), fontsize=6.4, color=SUB, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                shrinkA=1, shrinkB=2))

    ax.set_yscale("log")
    ax.set_xlim(0, 74)
    ax.set_ylim(2.0, 1300)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_ylabel("Plasma concentration (ng/mL, log scale)", fontsize=8.0)
    ax.set_title("Single oral dose, mean $\\pm$ SD of "
                 f"{N_SUBJ} animals per group", fontsize=9.0,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.0, labelbottom=False)
    ax.grid(True, which="major", color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.6, frameon=False,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.32)
    style_axes(ax)


def multi_dose_panel(ax, data):
    dose, _s, col = DOSES[1]
    popt = data[DOSES[1]]["popt"]
    tg = np.linspace(0.0, 72.0, 1200)
    conc = np.zeros_like(tg)
    for n in range(int(72.0 / TAU) + 1):
        m = tg >= n * TAU
        conc[m] += pk_model(tg[m] - n * TAU, *popt)

    ax.plot(tg, conc, color=col, lw=1.5, zorder=5)
    ax.fill_between(tg, 0.0, conc, color=lighten(col, 0.80), zorder=2)
    for n in range(int(72.0 / TAU) + 1):
        ax.plot([n * TAU], [0.0], marker="^", ms=3.4, color=col,
                clip_on=False, zorder=6)

    first = conc[(tg >= 0) & (tg < TAU)]
    last = conc[(tg >= 72.0 - TAU) & (tg <= 72.0)]
    t_last = tg[(tg >= 72.0 - TAU) & (tg <= 72.0)]
    css = float(np.trapezoid(last, t_last) / TAU)
    rac = float(np.trapezoid(last, t_last)
                / np.trapezoid(first, tg[(tg >= 0) & (tg < TAU)]))
    ax.axhline(css, color=BASE, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
    ax.text(1.0, css * 1.06, f"$C_{{ss,avg}}$ = {css:.0f} ng/mL", fontsize=6.3,
            color=SUB, ha="left", va="bottom")
    ax.annotate(f"accumulation ratio $R_{{ac}}$ = {rac:.2f}",
                xy=(66.0, last.max()), xytext=(44.0, conc.max() * 1.06),
                fontsize=6.4, color=INK, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                shrinkA=1, shrinkB=2))
    ax.set_xlim(0, 74)
    ax.set_ylim(0, conc.max() * 1.28)
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel("Time after first dose (h)", fontsize=8.0)
    ax.set_ylabel("Concentration\n(ng/mL, linear)", fontsize=7.5)
    ax.set_title(f"Multiple dosing, {dose:g} mg/kg every {TAU:g} h "
                 "(arrows mark doses)", fontsize=8.5, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.8)
    ax.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    return dict(css=css, rac=rac, trough=float(conc[-1]))


def linearity_panel(ax, data):
    xs, ys = [], []
    for (dose, _s, col), d in data.items():
        auc_i = np.array([np.trapezoid(r, TIMES) for r in d["reps"]])
        ax.plot(np.full(auc_i.size, dose), auc_i, ls="none", marker="o",
                ms=3.0, mfc="white", mec=col, mew=0.9, zorder=5)
        ax.plot([dose], [auc_i.mean()], marker="_", ms=9, mew=1.4, color=col,
                zorder=6)
        xs.append(np.full(auc_i.size, dose))
        ys.append(auc_i)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    slope = np.polyfit(np.log10(xs), np.log10(ys), 1)
    xg = np.logspace(np.log10(4.0), np.log10(62.0), 60)
    ax.plot(xg, 10.0 ** np.polyval(slope, np.log10(xg)), color=INK, lw=1.1,
            zorder=4)
    ax.plot(xg, ys[xs == 5.0].mean() * (xg / 5.0), color=BASE, lw=0.9,
            ls=(0, (4, 2.5)), zorder=3)
    ax.text(0.045, 0.94, f"power-law slope {slope[0]:.2f}\n"
                         "(1.00 = dose proportional)",
            transform=ax.transAxes, fontsize=6.3, color=INK, ha="left",
            va="top", linespacing=1.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4.0, 62.0)
    ax.set_xticks([5, 15, 50])
    ax.set_xticklabels(["5", "15", "50"])
    ax.set_xlabel("Dose (mg/kg, log scale)", fontsize=7.5)
    ax.set_ylabel("$AUC_{0-72h}$ (ng$\\cdot$h/mL)", fontsize=7.5)
    ax.set_title("Dose proportionality", fontsize=8.5, fontweight="bold",
                 pad=4)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, which="major", color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def table_panel(ax, data):
    cols = [("$C_{max}$\n(ng/mL)", lambda m, d: f"{m['cmax']:.0f}"),
            ("$T_{max}$\n(h)", lambda m, d: f"{m['tmax']:g}"),
            ("$T_{1/2}$\n(h)", lambda m, d: f"{m['thalf']:.1f}"),
            ("$AUC_{0-\\infty}$\n(ng$\\cdot$h/mL)",
             lambda m, d: f"{m['auc']:.0f}"),
            ("CL/F\n(L/h/kg)",
             lambda m, d: f"{1000.0 * d / m['auc']:.2f}")]
    nrow, ncol = len(DOSES), len(cols)
    ax.set_xlim(-0.05, ncol)
    ax.set_ylim(-0.85, nrow + 0.12)
    ax.axis("off")
    for j, (head, _f) in enumerate(cols):
        ax.text(j + 0.5, nrow + 0.06, head, ha="center", va="bottom",
                fontsize=6.1, color=INK, linespacing=1.30)
    for i, (dose, s, col) in enumerate(DOSES):
        y = nrow - 1 - i
        d = data[(dose, s, col)]
        ax.text(-0.12, y + 0.5, f"{dose:g} mg/kg", ha="right", va="center",
                fontsize=6.5, color=INK)
        for j, (_h, fn) in enumerate(cols):
            ax.add_patch(plt.Rectangle((j + 0.05, y + 0.12), 0.90, 0.76,
                                       facecolor=lighten(col, 0.86),
                                       edgecolor="white", linewidth=0.8,
                                       zorder=2))
            ax.text(j + 0.5, y + 0.50, fn(d["m"], dose), ha="center",
                    va="center", fontsize=6.4, color=INK, zorder=3)
    ax.text(0.0, -0.30, "CL/F from $AUC_{0-\\infty}$; $T_{1/2}$ from the "
                        "terminal slope of the fitted model",
            fontsize=6.0, color=MUTED, ha="left", va="top")
    ax.set_title("Derived exposure metrics", fontsize=8.5, fontweight="bold",
                 pad=16)


def main():
    set_pub_style()
    rng = np.random.default_rng(7519)
    data = {}
    for dose, s, col in DOSES:
        mean, sd, reps = observed(rng, dose, s)
        popt = fit_profile(mean)
        data[(dose, s, col)] = dict(mean=mean, sd=sd, reps=reps, popt=popt,
                                    m=metrics(mean, popt))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.60], wspace=0.34, hspace=0.42,
                          left=0.070, right=0.965, top=0.885, bottom=0.115)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])

    single_dose_panel(ax_s, data)
    ss = multi_dose_panel(ax_m, data)
    linearity_panel(ax_l, data)
    table_panel(ax_t, data)

    mid = data[DOSES[1]]["m"]

    panel_labels_inches(fig, [ax_s, ax_m, ax_l, ax_t], dx=-0.42, dy=0.08)
    savefig(fig, "519_drug_pk_profile.png")


if __name__ == "__main__":
    main()
