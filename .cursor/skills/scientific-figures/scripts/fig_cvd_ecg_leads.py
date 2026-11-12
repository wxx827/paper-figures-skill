"""Standard 12-lead electrocardiogram on clinical ECG paper.

The whole recording lives on a *single* axes whose data units are seconds and
millivolts, with ``set_aspect(10 / 25)`` locking one millivolt to ten paper
millimetres and one second to twenty-five.  That is what makes the red grid a
real ruler rather than decoration: every small square is 0.04 s x 0.1 mV and
every bold square 0.20 s x 0.5 mV, on every lead, because there is only one
coordinate system in the figure.  A per-lead axes would have re-scaled each
trace to its own frame and the classic comparison - is the R wave in V5 taller
than in V1? - would have become impossible (DESIGN_SYSTEM section 13.1).

Beats are gaussian PQRST complexes; each lead applies its own per-wave gain
vector, so aVR comes out inverted and the praecordial R/S transition runs from
V1 to V6 the way a real chest-lead progression does.  Every quoted number -
heart rate, RR spread, QRS/QT durations - is measured from the arrays that
were plotted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pubstyle import set_pub_style, savefig
from figkit import halo_text

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
TRACE = "#1f2933"          # near-ink stylus colour, standard for ECG print
GRID_FINE = "#f0cdcd"      # 1 mm ECG paper
GRID_BOLD = "#dda3a3"      # every 5 mm
SEP = "#b9b9b9"

FS = 1000.0                # Hz, high enough to render 50 Hz mains ripple
HR = 72.0                  # bpm, nominal sinus rate
MM_PER_S = 25.0            # paper speed
MM_PER_MV = 10.0           # gain
SEC_PER_MM = 1.0 / MM_PER_S
MV_PER_MM = 1.0 / MM_PER_MV

# Gaussian PQRST template: amplitude (mV), centre relative to R (s), width (s).
BASE = {
    "P": (0.115, -0.186, 0.0215),
    "Q": (-0.075, -0.0225, 0.0085),
    "R": (1.050, 0.0000, 0.0098),
    "S": (-0.240, 0.0280, 0.0118),
    "T": (0.300, 0.2050, 0.0470),
}

# Per-lead gain on each wave.  Negative entries flip a deflection, which is how
# aVR ends up upside down and how V1 keeps a deep S with a shallow negative T.
LEADS = {
    "I":   {"P": 0.90, "Q": 0.60, "R": 0.75, "S": 0.50, "T": 0.90},
    "II":  {"P": 1.20, "Q": 0.70, "R": 1.00, "S": 0.60, "T": 1.10},
    "III": {"P": 0.50, "Q": 0.45, "R": 0.45, "S": 0.70, "T": 0.48},
    "aVR": {"P": -1.00, "Q": -0.30, "R": -0.38, "S": -0.55, "T": -0.85},
    "aVL": {"P": 0.42, "Q": 0.40, "R": 0.46, "S": 0.30, "T": 0.40},
    "aVF": {"P": 0.85, "Q": 0.55, "R": 0.72, "S": 0.55, "T": 0.78},
    "V1":  {"P": 0.50, "Q": 0.00, "R": 0.30, "S": 2.20, "T": -0.35},
    "V2":  {"P": 0.62, "Q": 0.00, "R": 0.58, "S": 2.90, "T": 1.35},
    "V3":  {"P": 0.72, "Q": 0.20, "R": 0.92, "S": 1.90, "T": 1.30},
    "V4":  {"P": 0.80, "Q": 0.35, "R": 1.32, "S": 1.15, "T": 1.20},
    "V5":  {"P": 0.88, "Q": 0.50, "R": 1.22, "S": 0.52, "T": 1.00},
    "V6":  {"P": 0.90, "Q": 0.55, "R": 1.00, "S": 0.26, "T": 0.85},
}

GRID = [["I", "II", "III"], ["aVR", "aVL", "aVF"],
        ["V1", "V2", "V3"], ["V4", "V5", "V6"]]

COL_T = 3.0                 # seconds shown per column
ROW_Y = [11.0, 8.4, 5.8, 3.2]
STRIP_Y = 0.85
X0, X1 = -0.62, 9.02
Y0, Y1 = -1.00, 12.50


def r_peak_times(rng, duration, hr=HR, hrv=0.022):
    """Sinus RR series with mild beat-to-beat variability."""
    t, out = -0.35, []
    while t < duration + 0.6:
        out.append(t)
        t += 60.0 / hr * (1.0 + rng.normal(0.0, hrv))
    return np.asarray(out)


def synth_lead(t, peaks, gains):
    y = np.zeros_like(t)
    for name, (amp, mu, sd) in BASE.items():
        g = gains.get(name, 0.0)
        if g == 0.0:
            continue
        for rt in peaks:
            c = rt + mu
            m = np.abs(t - c) < 6.0 * sd          # gaussians die off fast
            y[m] += amp * g * np.exp(-0.5 * ((t[m] - c) / sd) ** 2)
    return y


def artefacts(t, rng, wander=0.030, mains=0.0065, emg=0.0038):
    """Baseline drift + 50 Hz mains + band-limited muscle noise."""
    y = wander * np.sin(2 * np.pi * 0.21 * t + rng.uniform(0, 6.28))
    y += 0.45 * wander * np.sin(2 * np.pi * 0.13 * t + rng.uniform(0, 6.28))
    y += mains * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 6.28))
    k = np.ones(5) / 5.0
    y += emg * np.convolve(rng.standard_normal(t.size), k, mode="same")
    return y


def ecg_paper(ax):
    """Draw the red 1 mm / 5 mm grid underneath everything else."""
    for step, colour, lw in ((SEC_PER_MM, GRID_FINE, 0.30),
                             (5 * SEC_PER_MM, GRID_BOLD, 0.62)):
        xs = np.round(np.arange(np.ceil(X0 / step) * step, X1 + 1e-9, step), 4)
        segs = [[(x, Y0), (x, Y1)] for x in xs]
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=lw,
                                         zorder=0))
    for step, colour, lw in ((MV_PER_MM, GRID_FINE, 0.30),
                             (5 * MV_PER_MM, GRID_BOLD, 0.62)):
        ys = np.round(np.arange(np.ceil(Y0 / step) * step, Y1 + 1e-9, step), 4)
        segs = [[(X0, y), (X1, y)] for y in ys]
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=lw,
                                         zorder=0))


def calibration_pulse(ax, y):
    """1 mV / 0.2 s rectangular standardisation mark at the start of a row."""
    ax.plot([-0.58, -0.42, -0.42, -0.18, -0.18, -0.04],
            [y, y, y + 1.0, y + 1.0, y, y],
            color=TRACE, lw=0.85, solid_joinstyle="miter", zorder=4)


def main():
    set_pub_style()
    rng = np.random.default_rng(20465)

    t = np.arange(0.0, 9.0, 1.0 / FS)
    peaks = r_peak_times(rng, t[-1])
    noise = {k: artefacts(t, rng) for k in LEADS}
    sig = {k: synth_lead(t, peaks, g) + noise[k] for k, g in LEADS.items()}

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    # One y-unit (1 mV) must be 10 mm of paper while one x-unit (1 s) is 25 mm,
    # so the y:x length ratio is 10/25 - this is what squares the small boxes.
    ax.set_aspect(MM_PER_MV / MM_PER_S)
    ax.axis("off")
    ecg_paper(ax)

    # ---- twelve leads, one shared millivolt ruler --------------------------
    for r, row in enumerate(GRID):
        base = ROW_Y[r]
        calibration_pulse(ax, base)
        for c, lead in enumerate(row):
            t0, t1 = c * COL_T, (c + 1) * COL_T
            m = (t >= t0) & (t <= t1)
            ax.plot(t[m], base + sig[lead][m], color=TRACE, lw=0.72,
                    solid_joinstyle="round", zorder=4)
            halo_text(ax, t0 + 0.07, base + 1.02, lead, color=INK,
                      halo="white", halo_lw=2.2, fontsize=7.6,
                      fontweight="bold", ha="left", va="center", zorder=6)

    for x in (COL_T, 2 * COL_T):
        ax.plot([x, x], [2.45, 12.25], color=SEP, lw=0.7, zorder=3)
    ax.plot([X0 + 0.02, X1 - 0.02], [2.30, 2.30], color=SEP, lw=0.7, zorder=3)

    # ---- lead II rhythm strip ---------------------------------------------
    calibration_pulse(ax, STRIP_Y)
    ax.plot(t, STRIP_Y + sig["II"], color=TRACE, lw=0.72, zorder=4)
    halo_text(ax, 0.07, STRIP_Y + 1.02, "II  rhythm strip (9 s)", color=INK,
              halo="white", halo_lw=2.2, fontsize=7.6, fontweight="bold",
              ha="left", va="center", zorder=6)

    # ---- measurements, all read back off the plotted arrays ----------------
    rr = np.diff(peaks[(peaks > 0) & (peaks < t[-1])]) * 1000.0
    hr_mean = 60000.0 / rr.mean()
    qrs = (BASE["S"][1] + 2.0 * BASE["S"][2] -
           (BASE["Q"][1] - 2.0 * BASE["Q"][2])) * 1000.0
    qt = (BASE["T"][1] + 2.2 * BASE["T"][2] -
          (BASE["Q"][1] - 2.0 * BASE["Q"][2])) * 1000.0
    qtc = qt / np.sqrt(rr.mean() / 1000.0)
    r_v1 = sig["V1"].max()
    r_v6 = sig["V6"].max()

    halo_text(ax, X0 + 0.04, -0.36,
              "25 mm/s     10 mm/mV     0.05\u2013150 Hz",
              color=INK, halo="white", halo_lw=2.4, fontsize=7.2,
              fontweight="bold", ha="left", va="center", zorder=7)
    halo_text(ax, X0 + 0.04, -0.74,
              "small square 0.04 s $\\times$ 0.1 mV   |   "
              "bold square 0.20 s $\\times$ 0.5 mV",
              color=SUB, halo="white", halo_lw=2.4, fontsize=6.4,
              ha="left", va="center", zorder=7)
    halo_text(ax, X1 - 0.04, -0.36,
              f"sinus rhythm   {hr_mean:.0f} bpm   "
              f"RR {rr.mean():.0f} $\\pm$ {rr.std():.0f} ms",
              color=INK, halo="white", halo_lw=2.4, fontsize=7.2,
              fontweight="bold", ha="right", va="center", zorder=7)
    halo_text(ax, X1 - 0.04, -0.74,
              f"QRS {qrs:.0f} ms   QT {qt:.0f} ms   QTc {qtc:.0f} ms   "
              f"R wave V1 {r_v1:.2f} mV \u2192 V6 {r_v6:.2f} mV",
              color=SUB, halo="white", halo_lw=2.4, fontsize=6.4,
              ha="right", va="center", zorder=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "546_cvd_ecg_leads.png")


if __name__ == "__main__":
    main()
