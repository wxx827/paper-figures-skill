"""Figure 667 - Calibrating a digital twin against plant measurements.

Main panel: 48 h of bearing-outlet temperature.  The as-designed model
(nameplate parameters) is drawn against the measurements, the calibration
instant splits the record into a fitting window and a held-out validation
window, and the shaded band between the two model traces is exactly what the
calibration moved.

Right column: what changed and whether it was enough.  Panel (b) contrasts the
prior on the load gain with the posterior implied by the fit; panel (c) plots
the residual against operating load, which is where an uncalibrated gain shows
up as a slope rather than as scatter.

Every printed number - both RMSEs, the reduction, the gain and its interval -
is computed from the arrays that are plotted (DESIGN_SYSTEM section 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, annotate_box, lighten
from figkit import axes_inches, panel_label_inches, shared_limits

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"

PRE = "#808080"        # as-designed model - deliberately neutral
POST = "#3b5b92"       # house blue - the calibrated twin
MEAS = "#1a1a1a"       # measurements
ACCENT = "#b23a48"     # house red - the single emphasis: calibration instant

T_END = 48.0           # hours of record
T_CAL = 18.0           # calibration instant, hours
THETA_PRE = (56.0, 0.492, 1.70)     # nameplate offset, load gain, swirl term
DEG = "$^{\\circ}$C"


def design(t, load):
    """Model basis: offset, load gain, 8 h swirl harmonic."""
    return np.column_stack([np.ones_like(t), load,
                            np.sin(2.0 * np.pi * t / 8.0)])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(667)

    t = np.linspace(0.0, T_END, 577)
    load = np.clip(58.0 + 22.0 * np.sin(2.0 * np.pi * t / 24.0 - 0.6)
                   + 6.0 * np.sin(2.0 * np.pi * t / 7.3), 22.0, 96.0)
    theta_true = np.array([61.5, 0.415, 2.60])
    wander = np.cumsum(rng.normal(0.0, 0.055, t.size))
    wander -= wander.mean()
    truth = design(t, load) @ theta_true + wander

    # Measurements every 45 min.
    idx = np.arange(0, t.size, 9)
    tm, lm = t[idx], load[idx]
    meas = truth[idx] + rng.normal(0.0, 0.55, idx.size)

    fit = tm <= T_CAL
    Xf = design(tm[fit], lm[fit])
    theta_post, *_ = np.linalg.lstsq(Xf, meas[fit], rcond=None)

    sim_pre = design(t, load) @ np.array(THETA_PRE)
    sim_post = design(t, load) @ theta_post
    mpre = design(tm, lm) @ np.array(THETA_PRE)
    mpost = design(tm, lm) @ theta_post

    val = ~fit
    rmse_pre = float(np.sqrt(np.mean((meas[val] - mpre[val]) ** 2)))
    rmse_post = float(np.sqrt(np.mean((meas[val] - mpost[val]) ** 2)))
    drop = 100.0 * (1.0 - rmse_post / rmse_pre)

    # Posterior spread of the load gain from the ordinary-least-squares fit.
    resid_f = meas[fit] - Xf @ theta_post
    dof = max(Xf.shape[0] - Xf.shape[1], 1)
    cov = (resid_f @ resid_f / dof) * np.linalg.inv(Xf.T @ Xf)
    b_hat, b_sd = float(theta_post[1]), float(np.sqrt(cov[1, 1]))
    b_prior, b_prior_sd = THETA_PRE[1], 0.075

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) measured vs modelled temperature ------------------------------
    axm = axes_inches(fig, (0.86, 0.76, 5.22, 4.66))
    axm.axvspan(0.0, T_CAL, facecolor="#f5f5f5", edgecolor="none", zorder=0)
    axm.fill_between(t, sim_pre, sim_post, facecolor=lighten(ACCENT, 0.78),
                     edgecolor="none", zorder=1,
                     label="correction applied by calibration")
    axm.plot(t, sim_pre, lw=1.3, color=PRE, ls=(0, (4.5, 2.2)), zorder=3,
             label="model, as designed")
    axm.plot(t, sim_post, lw=1.6, color=POST, zorder=4,
             label="model, after calibration")
    axm.plot(tm, meas, ls="none", marker="o", ms=3.0, mfc="white",
             mec=MEAS, mew=0.8, zorder=5, label="plant measurement")
    axm.axvline(T_CAL, color=ACCENT, lw=1.2, zorder=6)
    axm.annotate("calibration", xy=(T_CAL, 0.965), xycoords=("data", "axes fraction"),
                 xytext=(4.0, 0.0), textcoords="offset points", ha="left",
                 va="top", fontsize=7.0, color=ACCENT, fontweight="bold")

    lo, hi = shared_limits([truth, sim_pre, sim_post, meas], pad=0.10)
    axm.set_ylim(lo, hi)
    axm.set_xlim(0.0, T_END)
    axm.set_xticks(np.arange(0, 49, 6))
    axm.set_xlabel("time since start of trial (hours)", fontsize=8.0)
    axm.set_ylabel(f"bearing outlet temperature ({DEG})", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.6)
    axm.grid(axis="y", color=GRID, lw=0.6)
    axm.text(T_CAL / 2.0, hi - 0.025 * (hi - lo), "fitting window",
             ha="center", va="top", fontsize=6.8, color=MUTED)
    axm.text((T_CAL + T_END) / 2.0, hi - 0.025 * (hi - lo),
             "held-out validation window", ha="center", va="top",
             fontsize=6.8, color=MUTED)
    annotate_box(axm, f"validation RMSE\n{rmse_pre:.2f} {DEG}  "
                      f"$\\rightarrow$  {rmse_post:.2f} {DEG}\n"
                      f"{drop:.0f} % lower", loc="lower left", fontsize=7.2)
    # Only the top ~15 % of the panel is free of data, which is one legend row
    # short; parking the four entries above the frame keeps the traces clear.
    axm.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), fontsize=7.0,
               frameon=False, ncol=2, handlelength=1.9, labelspacing=0.28,
               columnspacing=1.4)
    axm.set_title("One record, two versions of the same model",
                  fontsize=9.0, fontweight="bold", pad=26.0)
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    # ---- (b) load gain: prior against posterior ----------------------------
    axp = axes_inches(fig, (7.02, 3.36, 3.02, 2.06))
    gb = np.linspace(b_prior - 4.2 * b_prior_sd, b_prior + 3.0 * b_prior_sd, 500)

    def gauss(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    for mu, sd, col, name in (
            (b_prior, b_prior_sd, PRE,
             f"prior {b_prior:.3f} $\\pm$ {b_prior_sd:.3f}"),
            (b_hat, b_sd, POST,
             f"posterior {b_hat:.3f} $\\pm$ {b_sd:.3f}")):
        d = gauss(gb, mu, sd)
        axp.fill_between(gb, 0.0, d, facecolor=lighten(col, 0.74),
                         edgecolor="none", zorder=2)
        axp.plot(gb, d, lw=1.4, color=col, zorder=3, label=name)
        axp.axvline(mu, color=col, lw=0.9, ls=(0, (2.4, 1.8)), zorder=4)
    axp.set_xlim(gb[0], gb[-1])
    axp.set_ylim(0.0, None)
    axp.set_xlabel(f"load gain ({DEG} per % load)", fontsize=7.4)
    axp.set_ylabel("density (-)", fontsize=7.4)
    axp.tick_params(labelsize=6.5, length=2.2)
    axp.grid(axis="y", color=GRID, lw=0.6)
    axp.legend(loc="upper right", fontsize=6.5, frameon=False,
               handlelength=1.4, labelspacing=0.25)
    axp.set_title("Load gain tightens by "
                  f"{b_prior_sd / b_sd:.0f}$\\times$", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)

    # ---- (c) residual against operating load -------------------------------
    axr = axes_inches(fig, (7.02, 0.76, 3.02, 1.94))
    axr.axhline(0.0, color=BASE, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    for res, col, name in ((meas - mpre, PRE, "as designed"),
                           (meas - mpost, POST, "calibrated")):
        axr.plot(lm, res, ls="none", marker="o", ms=2.6, mfc=col,
                 mec="white", mew=0.35, alpha=0.85, zorder=3, label=name)
        k = np.polyfit(lm, res, 1)
        xs = np.linspace(lm.min(), lm.max(), 60)
        axr.plot(xs, np.polyval(k, xs), lw=1.3, color=col, zorder=4)
    rlo, rhi = shared_limits([meas - mpre, meas - mpost], pad=0.12)
    axr.set_ylim(rlo, rhi)
    axr.set_xlim(lm.min() - 2.0, lm.max() + 2.0)
    axr.set_xlabel("operating load (%)", fontsize=7.4)
    axr.set_ylabel(f"residual ({DEG})", fontsize=7.4)
    axr.tick_params(labelsize=6.5, length=2.2)
    axr.grid(axis="y", color=GRID, lw=0.6)
    axr.legend(loc="upper right", fontsize=6.5, frameon=False,
               handlelength=1.2, labelspacing=0.25, ncol=2, columnspacing=0.8)
    axr.set_title("Residual stops tracking load", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)

    for ax, lab in ((axm, "(a)"), (axp, "(b)"), (axr, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.12)
    savefig(fig, "667_dt_model_calibration.png")


if __name__ == "__main__":
    main()
