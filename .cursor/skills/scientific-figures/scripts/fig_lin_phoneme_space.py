"""Acoustic vowel space with the consonant inventory and diphthong glides.

Both axes of the main panel run backwards.  That is not a slip: plotting F1
downward and F2 leftward is the hard convention of phonetics, because it makes
the scatter fall into the same arrangement as the articulatory vowel
quadrilateral - close vowels at the top, back vowels at the right.  The dashed
polygon through the peripheral centroids is that quadrilateral, drawn from the
same tokens as the scatter.

Colour encodes vowel height rather than identity, so ten categories are read
with three colours instead of ten, and the IPA glyph does the identifying.
Ellipses are two standard deviations of the measured token cloud, the grid
gives the consonant inventory with its token frequency, and the glide panel
tracks both formants across the normalised duration of three diphthongs.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from pubstyle import set_pub_style, savefig, get_cmap, darken
from figkit import (axes_inches, halo_text, panel_labels_inches,
                    safe_colorbar, shared_norm)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

HEIGHT_COLOR = {"close": "#3b5b92", "mid": "#2e7d5b", "open": "#b23a48"}

# IPA glyph, F1 (Hz), F2 (Hz), height class, peripheral order (-1 = interior)
VOWELS = [
    ("i", 285.0, 2280.0, "close", 0),
    ("\u026a", 405.0, 1935.0, "close", -1),
    ("e", 415.0, 2115.0, "mid", 1),
    ("\u025b", 585.0, 1790.0, "mid", 2),
    ("\u00e6", 705.0, 1655.0, "open", 3),
    ("\u0251", 760.0, 1105.0, "open", 4),
    ("\u0254", 615.0, 915.0, "mid", 5),
    ("o", 480.0, 845.0, "mid", 6),
    ("\u028a", 465.0, 1145.0, "close", -1),
    ("u", 320.0, 810.0, "close", 7),
    ("\u0259", 505.0, 1500.0, "mid", -1),
]
N_TOKEN = 42

PLACES = ["Bilab.", "Labiod.", "Dental", "Alveol.", "Postalv.", "Palatal",
          "Velar", "Glottal"]
MANNERS = ["Plosive", "Nasal", "Trill", "Fricative", "Affricate",
           "Approx.", "Lateral"]
# (manner, place): IPA glyphs, voiceless then voiced
INVENTORY = {
    ("Plosive", "Bilab."): "p b",
    ("Plosive", "Alveol."): "t d",
    ("Plosive", "Palatal"): "c",
    ("Plosive", "Velar"): "k g",
    ("Plosive", "Glottal"): "\u0294",
    ("Nasal", "Bilab."): "m",
    ("Nasal", "Alveol."): "n",
    ("Nasal", "Palatal"): "\u0272",
    ("Nasal", "Velar"): "\u014b",
    ("Trill", "Alveol."): "r",
    ("Fricative", "Labiod."): "f v",
    ("Fricative", "Dental"): "\u03b8 \u00f0",
    ("Fricative", "Alveol."): "s z",
    ("Fricative", "Postalv."): "\u0283 \u0292",
    ("Fricative", "Velar"): "x",
    ("Fricative", "Glottal"): "h",
    ("Affricate", "Postalv."): "t\u0283 d\u0292",
    ("Approx.", "Palatal"): "j",
    ("Approx.", "Velar"): "w",
    ("Lateral", "Alveol."): "l",
}

# IPA glyph, start (F1, F2), end (F1, F2)
DIPHTHONGS = [
    ("a\u026a", (735.0, 1290.0), (395.0, 2010.0), "#b23a48"),
    ("a\u028a", (740.0, 1310.0), (410.0, 960.0), "#3b5b92"),
    ("\u0254\u026a", (600.0, 890.0), (400.0, 1960.0), "#c9a227"),
]

FW, FH = 10.4, 6.4


def sample_vowels(rng):
    """Token clouds whose spread grows with F1, as real vowel data do."""
    out = {}
    for glyph, f1, f2, klass, per in VOWELS:
        s1 = 0.052 * f1 + 9.0
        s2 = 0.036 * f2 + 32.0
        rho = -0.28
        cov = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
        out[glyph] = rng.multivariate_normal([f1, f2], cov, N_TOKEN)
    return out


def ellipse_from(points, n_sd=2.0, **kw):
    """Two-sigma ellipse, computed in plot coordinates (x = F2, y = F1)."""
    p = points[:, ::-1]
    c = p.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(p.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * n_sd * np.sqrt(vals)
    return Ellipse(tuple(c), width=w, height=h, angle=ang, **kw)


def nearest_centroid(tokens):
    """Honest separability number, computed from the plotted tokens."""
    glyphs = list(tokens)
    cent = np.array([tokens[g].mean(0) for g in glyphs])
    scale = np.array([120.0, 320.0])
    hit, conf = 0, {}
    total = 0
    for i, g in enumerate(glyphs):
        d = np.linalg.norm((tokens[g][:, None, :] - cent[None]) / scale,
                           axis=2)
        pred = np.argmin(d, axis=1)
        hit += int(np.sum(pred == i))
        total += pred.size
        for p in pred[pred != i]:
            key = tuple(sorted((g, glyphs[p])))
            conf[key] = conf.get(key, 0) + 1
    worst = max(conf.items(), key=lambda kv: kv[1]) if conf else (("", ""), 0)
    return hit / total, worst


def draw_space(ax, tokens):
    for glyph, f1, f2, klass, per in VOWELS:
        col = HEIGHT_COLOR[klass]
        p = tokens[glyph]
        ax.scatter(p[:, 1], p[:, 0], s=7, color=col, alpha=0.42,
                   edgecolors="none", zorder=3)
        ax.add_patch(ellipse_from(p, facecolor=col, alpha=0.11,
                                  edgecolor=col, linewidth=0.9, zorder=4))
    per = [v for v in VOWELS if v[4] >= 0]
    per.sort(key=lambda v: v[4])
    poly = np.array([[tokens[v[0]][:, 1].mean(), tokens[v[0]][:, 0].mean()]
                     for v in per])
    poly = np.vstack([poly, poly[:1]])
    ax.plot(poly[:, 0], poly[:, 1], color="#8a8a8a", lw=0.8,
            ls=(0, (4, 2.4)), zorder=2)
    for glyph, f1, f2, klass, per_i in VOWELS:
        c = tokens[glyph].mean(0)
        halo_text(ax, c[1], c[0], glyph, ha="center", va="center",
                  fontsize=10.0, fontweight="bold",
                  color=darken(HEIGHT_COLOR[klass], 0.25), halo="white",
                  halo_lw=2.6, zorder=8)

    ax.set_xlim(2520.0, 620.0)                 # F2 increases to the left
    ax.set_ylim(880.0, 230.0)                  # F1 increases downward
    ax.set_xlabel("F2 (Hz), axis reversed", fontsize=8.0)
    ax.set_ylabel("F1 (Hz), axis reversed", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Monophthong space, 11 categories", fontsize=8.5,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], marker="o", ls="none", ms=4.4, mfc=c,
                      mec="none", label=f"{k} vowels")
               for k, c in HEIGHT_COLOR.items()]
    handles.append(Line2D([0], [0], color="#8a8a8a", lw=0.9, ls=(0, (4, 2.4)),
                          label="peripheral quadrilateral"))
    ax.legend(handles=handles, frameon=False, fontsize=7.0, loc="lower left",
              handletextpad=0.4, labelspacing=0.3, borderpad=0.15)


def draw_grid(fig, ax, rng):
    freq = np.full((len(MANNERS), len(PLACES)), np.nan)
    for (m, p), glyphs in INVENTORY.items():
        freq[MANNERS.index(m), PLACES.index(p)] = rng.uniform(3.0, 96.0)
    norm = shared_norm([freq[np.isfinite(freq)]])
    cmap = get_cmap("cb_sequential_blue")
    cmap = cmap.copy()
    cmap.set_bad("#f7f7f7")
    im = ax.imshow(np.ma.masked_invalid(freq), cmap=cmap, norm=norm,
                   aspect="auto", zorder=2)
    for (m, p), glyphs in INVENTORY.items():
        i, j = MANNERS.index(m), PLACES.index(p)
        dark = norm(freq[i, j]) > 0.55
        ax.text(j, i, glyphs, ha="center", va="center", fontsize=7.4,
                color="white" if dark else INK, zorder=4)
    ax.set_xticks(np.arange(len(PLACES)))
    ax.set_xticklabels(PLACES, fontsize=6.2, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(MANNERS)))
    ax.set_yticklabels(MANNERS, fontsize=6.2)
    ax.set_xticks(np.arange(-0.5, len(PLACES), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MANNERS), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Consonant inventory by place and manner", fontsize=8.0,
                 fontweight="bold")
    safe_colorbar(fig, im, ax, label="occurrence (per 1000 phonemes)",
                  width=0.10, pad=0.06, labelsize=6.4)
    return freq


def draw_glides(ax, rng):
    t = np.linspace(0.0, 100.0, 120)
    s = 1.0 / (1.0 + np.exp(-(t - 52.0) / 11.0))
    for glyph, (f1a, f2a), (f1b, f2b), col in DIPHTHONGS:
        jit = 0.012 * np.cumsum(rng.normal(0.0, 1.0, t.size))
        f1 = f1a + (f1b - f1a) * s * (1.0 + jit)
        f2 = f2a + (f2b - f2a) * s * (1.0 + jit)
        ax.plot(t, f2, color=col, lw=1.3, zorder=4, label=f"/{glyph}/")
        ax.plot(t, f1, color=col, lw=1.3, ls=(0, (3.2, 1.8)), zorder=4)
        ax.annotate("", xy=(t[-1], f2[-1]), xytext=(t[-14], f2[-14]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9))
    ax.text(2.0, 320.0, "solid F2   dashed F1", fontsize=6.2, color=MUTED,
            ha="left", va="bottom")
    ax.set_xlim(0.0, 104.0)
    ax.set_ylim(280.0, 2250.0)
    ax.set_xlabel("normalised duration (%)", fontsize=7.6)
    ax.set_ylabel("frequency (Hz)", fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title("Diphthong formant glides", fontsize=8.0, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper left",
              handlelength=1.4, columnspacing=1.0, borderpad=0.15)


def main():
    set_pub_style()
    rng = np.random.default_rng(633)
    tokens = sample_vowels(rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_v = axes_inches(fig, (0.70, 0.98, 5.05, 4.20))
    ax_c = axes_inches(fig, (6.60, 2.86, 2.90, 2.32))
    ax_g = axes_inches(fig, (6.60, 0.98, 2.90, 1.28))

    draw_space(ax_v, tokens)
    draw_grid(fig, ax_c, rng)
    draw_glides(ax_g, rng)

    panel_labels_inches(fig, [ax_v, ax_c, ax_g], dx=-0.34, dy=0.20)

    acc, (pair, n_conf) = nearest_centroid(tokens)
    n_tok = N_TOKEN * len(VOWELS)


    # console codepages choke on IPA, so escape it
    print(f"n={n_tok}  nearest-centroid acc={acc:.3f}  "
          f"worst pair={pair[0]!a}-{pair[1]!a} ({n_conf})")
    savefig(fig, "633_lin_phoneme_space.png")


if __name__ == "__main__":
    main()
