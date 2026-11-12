"""Figure 375 - Multi-sequence MRI of one lesion, eight contrasts.

Every tile renders the same synthetic anatomy and the same lesion; only the
tissue-to-signal lookup changes, which is exactly what a pulse sequence is.
The red zoom box sits on identical pixel coordinates in all eight tiles, so the
signal behaviour of the lesion can be read off column by column.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, COLORS
from figkit import image_axes, corner_stamp, add_zoom_inset, ZOOM

N = 256


# ---------------------------------------------------------------------------
# Synthetic brain phantom (layered ellipses + gyral folding, as in figure 253)
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _sym(a):
    return 0.5 * (a + a[:, ::-1])


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def _rad(x, y, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.hypot(xr / rx, yr / ry)


# tissue codes
AIR, SCALP, SKULL, CSF, GM, WM, VENT, VESSEL, EDEMA, RIM, CORE = range(11)

LESION = (0.30, -0.26)          # centre in normalised grid coordinates


def brain_labels(n=N, seed=4):
    rng = np.random.default_rng(seed)
    x, y = _grid(n)
    wob = 1.0 + 0.015 * _sym(_sm(rng, n, 9))

    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob
    lab = np.full((n, n), AIR, dtype=int)
    lab[rr <= 1.000] = SCALP
    lab[rr <= 0.962] = SKULL
    lab[rr <= 0.900] = CSF
    brain = rr <= 0.875
    lab[brain] = GM

    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    lab[brain & (rw <= 1.0)] = WM

    g1, g2 = _sym(_sm(rng, n, 7.0)), _sym(_sm(rng, n, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.62) & brain & (rr > 0.58)
    lab[sulci] = CSF

    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13),
                                 (0.0, 0.10, 0.030, 0.10, 0)]:
        lab[_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0] = VENT
    lab[(np.abs(x) < 0.010) & brain] = CSF

    for vx, vy, vrx, vry, va in [(0.0, 0.62, 0.030, 0.020, 0),
                                 (0.40, -0.06, 0.012, 0.070, 24),
                                 (-0.40, -0.06, 0.012, 0.070, -24),
                                 (-0.16, 0.42, 0.010, 0.060, -34),
                                 (0.16, 0.42, 0.010, 0.060, 34)]:
        lab[(_rad(x, y, vx, vy, vrx, vry, va) <= 1.0) & (lab != AIR)] = VESSEL

    cx, cy = LESION
    dl = _rad(x, y, cx, cy, 0.160, 0.140, -18) * (1.0 + 0.11 * _sm(rng, n, 9))
    lab[(dl <= 1.00) & brain] = EDEMA
    lab[(dl <= 0.62) & brain] = RIM
    lab[(dl <= 0.40) & brain] = CORE
    return lab, rng


# tissue -> relative signal in each sequence (display units, 0..1)
SEQUENCES = [
    ("T1w", "TR 550 / TE 12 ms"),
    ("T2w", "TR 4500 / TE 96 ms"),
    ("FLAIR", "TI 2200 / TE 120 ms"),
    ("DWI  b = 1000", "TE 82 ms"),
    ("ADC map", "b = 0, 1000 s/mm2"),
    ("T1w + Gd", "TR 550 / TE 12 ms"),
    ("SWI", "TE 20 ms, minIP"),
    ("PD-w", "TR 3000 / TE 15 ms"),
]

SIGNAL = {
    #          T1    T2   FLAIR  DWI   ADC   T1C   SWI    PD
    AIR:    [0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02],
    SCALP:  [0.88, 0.52, 0.30, 0.22, 0.32, 0.88, 0.60, 0.62],
    SKULL:  [0.10, 0.08, 0.07, 0.08, 0.10, 0.10, 0.14, 0.12],
    CSF:    [0.12, 0.95, 0.07, 0.14, 0.95, 0.12, 0.70, 0.86],
    GM:     [0.46, 0.56, 0.55, 0.52, 0.46, 0.46, 0.38, 0.74],
    WM:     [0.72, 0.34, 0.36, 0.44, 0.40, 0.72, 0.54, 0.58],
    VENT:   [0.10, 0.97, 0.06, 0.12, 0.97, 0.10, 0.72, 0.88],
    VESSEL: [0.30, 0.24, 0.30, 0.30, 0.44, 0.92, 0.05, 0.42],
    EDEMA:  [0.54, 0.78, 0.74, 0.46, 0.74, 0.54, 0.50, 0.70],
    RIM:    [0.34, 0.62, 0.66, 0.84, 0.26, 0.97, 0.34, 0.72],
    CORE:   [0.22, 0.88, 0.82, 0.30, 0.90, 0.24, 0.30, 0.80],
}

# per-sequence acquisition character: (blur px, noise sd, texture gain)
CHARACTER = [(0.8, 0.012, 1.00), (0.9, 0.014, 1.00), (1.0, 0.016, 0.85),
             (2.1, 0.030, 0.70), (2.1, 0.026, 0.70), (0.8, 0.012, 1.00),
             (0.7, 0.018, 1.15), (0.9, 0.013, 1.00)]

# what the lesion does in each sequence, for the per-tile caption
LESION_NOTE = ["core hypointense", "core hyperintense", "edema bright, CSF null",
               "rim restricted", "core high ADC", "rim enhances",
               "veins hypointense", "mild hyperintense"]


def render(lab, k, rng):
    """Tissue map -> one displayed sequence image in [0, 1]."""
    n = lab.shape[0]
    img = np.zeros((n, n), dtype=float)
    for code, vals in SIGNAL.items():
        img[lab == code] = vals[k]

    blur, noise, tex = CHARACTER[k]
    inside = lab != AIR
    img = img + tex * 0.035 * _sm(rng, n, 2.5) * inside
    img = img * (1.0 + 0.07 * _sm(rng, n, 26) * inside)     # B1 bias field
    img = gaussian_filter(img, blur)
    img = img + noise * rng.standard_normal((n, n))
    return np.clip(img, 0.0, 1.0)


def main():
    set_pub_style(base_font=8)
    lab, rng = brain_labels()
    imgs = [render(lab, k, np.random.default_rng(31 + k))
            for k in range(len(SEQUENCES))]

    # one crop window, expressed once, reused by every tile
    cx = int(round(LESION[0] * N / 2 + (N - 1) / 2))
    cy = int(round(LESION[1] * N / 2 + (N - 1) / 2))
    half = 30
    box = (cx - half, cy - half, 2 * half, 2 * half)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 4, wspace=0.030, hspace=0.155,
                          left=0.012, right=0.988, top=0.905, bottom=0.055)

    for k, (name, param) in enumerate(SEQUENCES):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        image_axes(ax, color="#d5d5d5", lw=0.6)
        ax.imshow(imgs[k], cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="bilinear")
        ax.set_title(name, fontsize=8.8, fontweight="bold", pad=3.5,
                     color=COLORS["ink"])
        corner_stamp(ax, param, loc="upper left", fontsize=6.4,
                     color="#e0e0e0")
        corner_stamp(ax, LESION_NOTE[k], loc="lower left", fontsize=6.8)
        add_zoom_inset(ax, imgs[k], box, (0.615, 0.015, 0.37, 0.37),
                       color=ZOOM, connect=True, cmap="gray", vmin=0.0,
                       vmax=1.0, interpolation="bilinear")


    savefig(fig, "375_med_mri_sequences.png")


if __name__ == "__main__":
    main()
