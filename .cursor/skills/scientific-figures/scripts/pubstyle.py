"""Shared publication-quality matplotlib style for scientific figures.

Import this from any generator script:

    from pubstyle import set_pub_style, PALETTES, savefig

Design goals: publication-quality journal look. Thin spines, a Times New
Roman serif type stack, muted-but-saturated palettes, tight layout, 300 dpi
export.

Typography contract (see ``DESIGN_SYSTEM.md`` section 3): the house face is
Times New Roman.  Because that font is proprietary and cannot be shipped,
``fonts/`` carries Tinos (SIL OFL), which is *metrically* identical to Times
New Roman -- same advance widths, so a figure lays out byte-for-byte the same
on a machine that has the real thing and on one that does not.  Never set
``font.family`` in a figure script; call :func:`set_pub_style` once.

The module is strictly additive: every symbol that existed before keeps its
name, its defaults and its behaviour, so the 50+ figure scripts that already
import it are unaffected.  New capabilities live behind keyword arguments
(``set_pub_style(theme=...)``, ``savefig(..., formats=...)``) or brand-new
helpers (``get_palette``, ``sig_stars``, ``add_sig_bracket``, ...).

See ``reference.md`` for the full design-system documentation.
"""
from __future__ import annotations

import os
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---- Typography: the Times New Roman stack --------------------------------
# Resolution order.  The first entry is the real (proprietary) face; entries
# 2-6 are metric-compatible clones -- identical advance widths, so switching
# between them never reflows a label.  ``Tinos`` ships in ``fonts/`` so the
# stack always resolves, even in a bare Docker image.
#
# ``STIXGeneral`` is the one exception: it is bundled with matplotlib and it
# *looks* like Times, but ``check_typography.py --metrics`` measures it 121
# units/1000em away, so text set in it will reflow.  It sits last because it
# still reads as a serif journal face, which DejaVu does not.  DejaVu itself is
# deliberately absent: falling back to it silently is the bug this stack exists
# to prevent, and ``assert_font_ok`` raises rather than let it through.
SERIF_STACK = [
    "Times New Roman",   # Windows / machines with the MS core fonts
    "Tinos",             # bundled, SIL OFL, metric-compatible (verified)
    "Nimbus Roman",      # URW clone, default on most Linux distros
    "Liberation Serif",  # default on most Linux distros
    "TeX Gyre Termes",   # ships with TeX Live
    "Times",             # macOS
    "STIXGeneral",       # matplotlib's own; visual match, NOT metric-compatible
]

# Serif CJK faces, so a Chinese label keeps the Times texture instead of
# flipping the whole figure to sans.  Matplotlib >= 3.6 walks the family list
# glyph by glyph, so these sit *after* the Latin stack rather than replacing it.
_CN_SERIF_FONTS = ["SimSun", "Songti SC", "STSong", "Noto Serif CJK SC",
                   "Source Han Serif SC", "AR PL UMing CN"]

# Sans CJK faces, kept as a second choice: a sans Chinese glyph is far better
# than a missing-glyph box.
_CN_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
             "Source Han Sans SC", "PingFang SC", "WenQuanYi Micro Hei"]

FONTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "fonts")

# Anything in this set means the stack failed to resolve and matplotlib fell
# through to its own default, which does not have Times metrics.
_FORBIDDEN_FALLBACKS = {"DejaVu Sans", "DejaVu Serif", "DejaVu Sans Display",
                        "DejaVu Serif Display", "Bitstream Vera Sans"}

_FONTS_REGISTERED = False


def _register_bundled_fonts():
    """Add every font in ``fonts/`` to matplotlib's registry.

    Idempotent and never fatal: a missing directory just means the caller
    relies on system fonts, which :func:`assert_font_ok` will verify anyway.
    Returns the list of family names that were registered.
    """
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return []
    _FONTS_REGISTERED = True
    d = os.path.abspath(FONTS_DIR)
    if not os.path.isdir(d):
        return []
    from matplotlib import font_manager
    added = []
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith((".ttf", ".otf", ".ttc")):
            continue
        path = os.path.join(d, fn)
        try:
            font_manager.fontManager.addfont(path)
            added.append(font_manager.FontProperties(fname=path).get_name())
        except Exception as exc:            # a corrupt file must not kill a run
            warnings.warn(f"could not register bundled font {fn}: {exc}")
    return sorted(set(added))


def available_fonts():
    """Family names matplotlib can currently resolve."""
    from matplotlib import font_manager
    return {f.name for f in font_manager.fontManager.ttflist}


def resolved_font():
    """The family matplotlib will actually use for body text right now.

    Resolves the configured family list the same way the renderer does, so
    this is the ground truth -- not what ``rcParams`` merely *asked* for.
    """
    from matplotlib import font_manager
    prop = font_manager.FontProperties(
        family=mpl.rcParams["font.family"],
        weight=mpl.rcParams.get("font.weight", "normal"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = font_manager.findfont(prop, fallback_to_default=True)
    return font_manager.FontProperties(fname=path).get_name()


def assert_font_ok(strict=None):
    """Fail loudly when the serif stack did not resolve.

    A silent fallback to DejaVu is the single defect that makes a figure look
    wrong on someone else's machine while every other check still passes, so
    it is treated as an error rather than a warning.  Set the environment
    variable ``PUBSTYLE_FONT_STRICT=0`` to downgrade it to a warning (useful
    when deliberately rendering a CJK figure on a bare system).
    """
    if strict is None:
        strict = os.environ.get("PUBSTYLE_FONT_STRICT", "1") != "0"
    name = resolved_font()
    if name not in _FORBIDDEN_FALLBACKS:
        return name
    msg = (
        f"font stack did not resolve: matplotlib fell back to {name!r}, which "
        f"does not have Times metrics, so this figure will not match the "
        f"gallery.\nInstall one of {SERIF_STACK[:5]} or make sure "
        f"{os.path.abspath(FONTS_DIR)} contains the bundled Tinos TTFs "
        f"(see fonts/README.md)."
    )
    if strict:
        raise RuntimeError(msg)
    warnings.warn(msg)
    return name


# ---- Curated palettes pulled from the reference figures -------------------
PALETTES = {
    # teal / rust diverging pair (threshold + marginal figures)
    "teal_rust": ["#2a7f7a", "#c0603f"],
    # blue / red used in ML prediction plots
    "blue_red": ["#3b5b92", "#b23a48"],
    # green sequential (ecosystem, circular bars)
    "greens": ["#0b3d2e", "#2e7d5b", "#6bbf8a", "#bfe3cd"],
    # warm earth categorical (line charts with counties)
    "earth": ["#1a1a1a", "#3d6b8e", "#c9a227", "#8c2f2f", "#d98c5f"],
    # corrplot diverging (RdBu-like but softer)
    "div_rdbu": ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7",
                 "#d1e5f0", "#67a9cf", "#2166ac"],
    # pastel pair-plot backgrounds
    "pastel4": ["#f2b134", "#4a90d9", "#7cb342", "#8e6bbf"],
    # coupling-zone bands
    "zones": ["#f6c85f", "#c9d6a3", "#b8c9e8", "#c9b8e8", "#f0a58a"],

    # ---- colour-blind-safe additions --------------------------------------
    # Okabe-Ito qualitative set: distinguishable under all common CVD types
    "cb_safe": ["#0072b2", "#e69f00", "#009e73", "#cc79a7",
                "#56b4e9", "#d55e00", "#f0e442", "#111111"],
    # muted variant of Okabe-Ito for dense scatter / fills
    "cb_muted": ["#4c78a8", "#dda15e", "#5aa17f", "#c48ba9",
                 "#84b6d6", "#c4714b", "#cfc178", "#5b5b5b"],
    # blue <-> orange diverging (safe replacement for red/green)
    "cb_diverging": ["#0b3d63", "#2e6f9e", "#8fc0d9", "#f2f2f2",
                     "#f4c48a", "#d98032", "#8c4a12"],
    # purple <-> green diverging, for a second diverging channel
    "cb_diverging_pg": ["#40004b", "#7b3294", "#c2a5cf", "#f7f7f7",
                        "#a6dba0", "#3f8a4f", "#0d3d1f"],
    "cb_sequential_blue": ["#f2f7fb", "#cfe1ee", "#9cc3dd", "#5f9dc6",
                           "#2e74a8", "#134a76", "#08263f"],
    "cb_sequential_green": ["#f2f9f4", "#d3ecdb", "#a4d7b8", "#6bba8e",
                            "#3d9469", "#1f6a47", "#0b3d2a"],
    "cb_sequential_warm": ["#fdf5ec", "#fbe0bf", "#f4c088", "#e69a54",
                           "#cc722f", "#a04c18", "#66300c"],
    "cb_sequential_purple": ["#f7f4fa", "#e0d7ec", "#c1acd8", "#9c7dbe",
                             "#77539f", "#54327a", "#331a4d"],
    # neutral greys for reference lines, baselines and "other" categories
    "neutral": ["#f5f5f5", "#d9d9d9", "#b3b3b3", "#808080",
                "#4d4d4d", "#262626"],
}

# Palettes that describe a continuous ramp; ``get_palette`` interpolates
# these instead of cycling through them.
_CONTINUOUS = {
    "greens", "div_rdbu", "cb_diverging", "cb_diverging_pg",
    "cb_sequential_blue", "cb_sequential_green", "cb_sequential_warm",
    "cb_sequential_purple", "neutral",
}

# Recommended canvas sizes in inches (width, height).
FIGSIZES = {
    "single": (3.5, 2.6),     # one journal column
    "single_tall": (3.5, 4.2),
    "double": (7.2, 4.5),     # full page width
    "double_tall": (7.2, 9.0),
    "square": (4.6, 4.6),
    "slide": (10.0, 5.6),     # 16:9 presentation
    # DESIGN_SYSTEM.md section 1.1 names two further tiers for dense panel
    # grids; they were documented before they existed here.  Added, never
    # renamed: every key above keeps its original value.
    "wide": (10.4, 6.4),      # 2x3 / 3x3 grids with per-panel axis labels
    "full": (12.6, 7.6),      # >= 3x4, or grids containing 3D / map panels
}

# ---- Semantic house colours (DESIGN_SYSTEM.md section 2.1) -----------------
# The neutrals every figure needs, by role rather than by hex, so a script can
# say what it means.  Purely additive: PALETTES is untouched.
COLORS = {
    "ink": "#1a1a1a",          # panel titles, emphasis annotation
    "rule": "#333333",         # axis lines, ticks, primary annotation
    "sub": "#555555",          # subtitles, footnote body
    "muted": "#777777",        # in-figure side notes, units
    "faint": "#8a8a8a",        # weakest legible text
    "baseline": "#808080",     # reference / median / zero lines
    "grid": "#cccccc",         # separators
    "hairline": "#d5d5d5",
    "card": "#f5f5f5",         # background grouping blocks
    "card_alt": "#f0f0f0",
    "solid": "#3a3a3a",        # large dark fills (never pure black)
    "accent": "#b23a48",       # the single emphasis colour per figure
}

# ---- Type scale (DESIGN_SYSTEM.md section 3) -------------------------------
# The single source of truth for point sizes.  DESIGN_SYSTEM.md used to state
# this ladder in prose only, so scripts drifted to 8.5 / 8.6 / 9.0 for the same
# role; check_typography.py now lints against these numbers.
TYPE_SCALE = {
    "header_title": 11.0,     # figure header, bold
    "header_subtitle": 8.0,   # grey one-liner under the title
    "panel_label": 11.0,      # (a) (b) (c), bold
    "panel_title": 9.0,       # ax.set_title, bold, short sentence
    "axis_label": 8.0,
    "tick": 7.5,
    "legend": 7.5,
    "annotation": 7.0,        # call-out boxes
    "dense": 6.5,             # matrix cells, contour labels
    "footer": 6.5,            # grey provenance line
}
# Hard bounds.  6.0 pt is the smallest that survives 300 dpi print; 12.5 pt is
# the largest that still reads as a figure rather than a poster.
TYPE_MIN = 6.0
TYPE_MAX = 12.5

# House qualitative order, positions 1-5 of DESIGN_SYSTEM.md section 2.1.
CATEGORICAL = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]
# Same family, for the rare 6th/7th category.
CATEGORICAL_EXTRA = ["#8c2f2f", "#c0603f", "#3d6b8e", "#2a7f7a", "#7b3294"]

# ---- Journal / medium themes ----------------------------------------------
# Each theme only *overrides* the shared base defined in set_pub_style().
THEMES = {
    "default": {
        "base_font": 9,
        "rc": {},
    },
    "nature": {
        # Nature: 7pt minimum type, very fine rules, compact ticks
        "base_font": 7,
        "rc": {
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
            "xtick.major.size": 2.4, "ytick.major.size": 2.4,
            "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.0,
            "patch.linewidth": 0.5,
            "grid.linewidth": 0.45,
            "axes.titlepad": 3.5,
            "figure.figsize": (3.5, 2.6),
        },
    },
    "science": {
        # Science: slightly larger type than Nature, a touch more contrast
        "base_font": 8,
        "rc": {
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#2b2b2b",
            "xtick.major.width": 0.7, "ytick.major.width": 0.7,
            "xtick.major.size": 2.8, "ytick.major.size": 2.8,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.4,
            "axes.titlepad": 4.0,
            "figure.figsize": (3.6, 2.7),
        },
    },
    "ieee": {
        # IEEE two-column: narrow canvas, dense grid, Times-like metrics
        "base_font": 7,
        "rc": {
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7, "ytick.major.width": 0.7,
            "xtick.major.size": 2.4, "ytick.major.size": 2.4,
            "xtick.direction": "in", "ytick.direction": "in",
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "legend.fontsize": 6,
            "grid.linewidth": 0.4,
            "axes.titlepad": 3.0,
            "figure.figsize": (3.3, 2.4),
        },
    },
    "presentation": {
        # Projector / slide deck: big type, thick strokes, high contrast
        "base_font": 13,
        "rc": {
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#2b2b2b",
            "xtick.major.width": 1.2, "ytick.major.width": 1.2,
            "xtick.major.size": 4.5, "ytick.major.size": 4.5,
            "lines.linewidth": 2.6,
            "lines.markersize": 7.0,
            "patch.linewidth": 1.0,
            "grid.linewidth": 0.9,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "axes.titlepad": 8.0,
            "figure.figsize": (10.0, 5.6),
        },
    },
}

def set_pub_style(base_font=9, theme="default", **overrides):
    """Apply the global publication style. Call once at script start.

    Parameters
    ----------
    base_font : int
        Base type size in points.  When a non-default ``theme`` is requested
        and ``base_font`` is left at its default, the theme's own size wins.
    theme : str
        One of ``default``, ``nature``, ``science``, ``ieee``,
        ``presentation``.
    **overrides
        Extra rcParams applied last, e.g. ``set_pub_style(**{"axes.grid": True})``.
    """
    if theme not in THEMES:
        raise ValueError(
            f"unknown theme {theme!r}; choose from {sorted(THEMES)}")
    spec = THEMES[theme]
    # Only fall back to the theme size if the caller kept the legacy default.
    if base_font == 9 and theme != "default":
        base_font = spec["base_font"]

    _register_bundled_fonts()

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": list(SERIF_STACK),
        # STIX is Times-metric, so maths and body text share a colour and a
        # cap height.  Pairing a serif body with the default sans maths is the
        # most common way a figure reads as "made in matplotlib".
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        # Type 42 = embed the real outlines, so a journal's prepress does not
        # substitute the font.  fonttype 3 would be smaller but is unreliable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Keep SVG text as text (editable, and no font substitution) rather
        # than converting glyphs to paths.
        "svg.fonttype": "none",
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "axes.titlepad": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.formatter.use_mathtext": True,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": base_font - 1,
        "legend.frameon": False,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.35,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "patch.linewidth": 0.6,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 0.9,
        "errorbar.capsize": 2.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # constrained_layout stays OFF so existing tight_layout() calls behave
        # exactly as before; the pads only matter if a script opts in.
        "figure.constrained_layout.use": False,
        "figure.constrained_layout.h_pad": 0.05,
        "figure.constrained_layout.w_pad": 0.05,
        "figure.constrained_layout.hspace": 0.04,
        "figure.constrained_layout.wspace": 0.04,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "mathtext.default": "regular",
    })
    mpl.rcParams.update(spec["rc"])
    if theme != "default":
        # Keep derived sizes consistent with the theme's base font.
        mpl.rcParams.update({
            "font.size": base_font,
            "axes.titlesize": base_font + 1,
            "axes.labelsize": base_font,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
        })
        if "legend.fontsize" not in spec["rc"]:
            mpl.rcParams["legend.fontsize"] = base_font - 1
    if overrides:
        mpl.rcParams.update(overrides)

    # Verify the stack actually resolved.  Doing this here means a broken font
    # environment fails at style time with a readable message, instead of
    # producing 1200 subtly-wrong PNGs that pass every other check.
    assert_font_ok()


def set_cn_font(prefer_serif=True):
    """Enable Chinese glyphs without giving up the Times look.

    Call *after* :func:`set_pub_style`.

    The CJK face is appended to a concrete ``font.family`` list, not to
    ``font.serif``.  That distinction is the whole function: matplotlib only
    walks a family list glyph by glyph when the list names real families, so
    leaving ``font.family="serif"`` and extending ``font.serif`` looks correct,
    resolves without error, and still renders every Chinese character as an
    empty box.  The list is also filtered to fonts that exist, which keeps
    ``findfont`` from warning about the five stack members a given machine
    happens not to have.

    ``prefer_serif`` looks for a Song/Ming face first, the CJK counterpart of a
    Times body; set it False to prefer a Hei/Gothic face.

    Returns True when a CJK font was found, False (silently) otherwise, so
    callers can fall back to English labels.
    """
    try:
        available = available_fonts()
    except Exception:
        return False
    order = (_CN_SERIF_FONTS + _CN_FONTS) if prefer_serif else \
            (_CN_FONTS + _CN_SERIF_FONTS)
    cjk = next((n for n in order if n in available), None)
    if cjk is None:
        return False
    latin = [n for n in SERIF_STACK if n in available] or ["serif"]
    mpl.rcParams["font.family"] = latin + [cjk]
    # A CJK face usually carries U+2212; keep the real minus sign.
    mpl.rcParams["axes.unicode_minus"] = False
    return True


# ---- Colour helpers --------------------------------------------------------
def get_palette(name, n=None, reverse=False):
    """Return ``n`` colours from a named palette.

    Continuous palettes (sequential / diverging) are interpolated in RGB;
    categorical palettes are cycled.  ``n=None`` returns the palette as-is.
    """
    if name not in PALETTES:
        raise KeyError(f"unknown palette {name!r}; choose from {sorted(PALETTES)}")
    cols = list(PALETTES[name])
    if reverse:
        cols = cols[::-1]
    if n is None:
        return cols
    if n <= 0:
        return []
    if name in _CONTINUOUS:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name, cols)
        stops = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        return [mpl.colors.to_hex(cmap(s)) for s in stops]
    return [cols[i % len(cols)] for i in range(n)]


def get_cmap(name, reverse=False, n=256):
    """Build a matplotlib colormap from a named palette."""
    cols = get_palette(name, reverse=reverse)
    return mpl.colors.LinearSegmentedColormap.from_list(name, cols, N=n)


def set_color_cycle(name, n=None):
    """Make a named palette the default property cycle for new axes."""
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=get_palette(name, n))


def lighten(color, amount=0.5):
    """Blend a colour toward white (``amount`` in 0..1)."""
    r, g, b = mpl.colors.to_rgb(color)
    f = float(np.clip(amount, 0.0, 1.0))
    return mpl.colors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def darken(color, amount=0.3):
    """Blend a colour toward black (``amount`` in 0..1)."""
    r, g, b = mpl.colors.to_rgb(color)
    f = 1.0 - float(np.clip(amount, 0.0, 1.0))
    return mpl.colors.to_hex((r * f, g * f, b * f))


# ---- IO --------------------------------------------------------------------
def out_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def _extra_formats():
    """Formats requested library-wide through ``PUBSTYLE_FORMATS``.

    Journals want vector art, but adding ``formats=("png", "pdf")`` to 1200
    call sites is not a refactor anyone should do by hand.  Setting
    ``PUBSTYLE_FORMATS=png,pdf`` makes every :func:`savefig` in the library
    emit both, which is how ``run_all.py --vector`` works.
    """
    raw = os.environ.get("PUBSTYLE_FORMATS", "").strip()
    if not raw:
        return []
    return [f.strip().lstrip(".").lower() for f in raw.split(",") if f.strip()]


def savefig(fig, name, formats=("png",), dpi=None, close=True,
            transparent=False, outdir=None):
    """Save into the skill's output/ folder and print the path.

    ``formats`` may list several targets, e.g. ``("png", "pdf", "svg")``;
    the file stem is taken from ``name``.  Returns the first path written.
    Pass ``outdir`` to write elsewhere (used by the GUI exporter).

    The environment variable ``PUBSTYLE_FORMATS`` adds formats on top of
    whatever the caller asked for, so a batch run can produce vector copies
    without touching the figure scripts.
    """
    stem, ext = os.path.splitext(name)
    ext = ext.lstrip(".").lower()
    fmts = [ext] if (ext and tuple(formats) == ("png",)) else list(formats)
    if not fmts:
        fmts = ["png"]
    for extra in _extra_formats():
        if extra not in fmts:
            fmts.append(extra)
    dest = os.path.abspath(outdir) if outdir else out_dir()
    os.makedirs(dest, exist_ok=True)
    kw = {"transparent": transparent}
    if dpi is not None:
        kw["dpi"] = dpi
    paths = []
    for fmt in fmts:
        path = os.path.join(dest, f"{stem}.{fmt}")
        fig.savefig(path, **kw)
        paths.append(path)
        print("saved:", path)
    if close:
        plt.close(fig)
    return paths[0]


# ---- Figure header ---------------------------------------------------------
# House chrome: a left-aligned bold title, a grey subtitle that carries the
# detail, and a small grey footer for provenance.  `_strip_fig_chrome.py` once
# removed every `fig.suptitle` in the library, which left DESIGN_SYSTEM.md
# section 5 documenting a title layer that no code produced; four consecutive
# batch audits then asked for this helper.  This is it.
def header_height(fig, title=True, subtitle=False, panel_labels=True,
                  title_size=None, subtitle_size=None, label_size=None,
                  top=0.12, gap=0.045, label_gap=0.10):
    """Inches of headroom :func:`add_figure_header` will want.

    Use it to size the ``rect`` you pass to ``tight_layout`` *before* drawing
    the header::

        h = header_height(fig, subtitle=True) / fig.get_size_inches()[1]
        fig.tight_layout(rect=[0, 0.03, 1, 1 - h])
        add_figure_header(fig, "...", subtitle="...")
        panel_labels_inches(fig, axes)

    ``panel_labels`` reserves the row that ``(a) (b) (c)`` occupies above the
    axes.  It defaults to True because the house style letters almost every
    multi-panel figure, and a letter colliding with the subtitle is a defect
    that four batch audits kept reporting.  Set it False for a single-panel
    figure that carries no letter.
    """
    ts = TYPE_SCALE["header_title"] if title_size is None else title_size
    ss = TYPE_SCALE["header_subtitle"] if subtitle_size is None else subtitle_size
    ls = TYPE_SCALE["panel_label"] if label_size is None else label_size
    h = top
    if title:
        h += ts * 1.25 / 72.0
    if subtitle:
        h += gap + ss * 1.25 / 72.0
    if panel_labels:
        h += label_gap + ls * 1.25 / 72.0
    return h


def add_figure_header(fig, title, subtitle=None, footer=None, corner=None,
                      left=0.11, top=0.12, bottom=0.07, gap=0.045,
                      title_size=None, subtitle_size=None, footer_size=None,
                      align="left", reserve=False):
    """Stamp the house title / subtitle / footer block on a figure.

    Positions are in **inches from the figure edge**, not axes fractions, so
    the block sits in the same physical place on a 3.5 in single-column figure
    and on a 12.6 in dashboard.  That is the whole point: axes-fraction
    offsets are why panel labels in this library drifted between x=-0.045 and
    x=-0.155.

    Parameters
    ----------
    title : str
        Sentence case, no trailing period, and it should state the
        *conclusion* rather than name the chart type.  ``Loop closure removes
        unbounded drift``, not ``SLAM trajectory plot``.
    subtitle : str, optional
        Grey supporting line: cohort size, units, model version.
    footer : str, optional
        Small grey provenance line at the bottom left.
    corner : str, optional
        Small grey stamp at the top right, e.g. a dataset tag.
    align : {"left", "center"}
        ``left`` is the house default; ``center`` exists for the rare
        symmetric dashboard.
    reserve : bool
        Shrink the axes area to make room. Off by default because most
        scripts already call ``tight_layout(rect=...)``; see
        :func:`header_height`.

    Call this *after* ``tight_layout()``.  Returns the created artists.
    """
    if align not in ("left", "center"):
        raise ValueError(f"align must be 'left' or 'center', got {align!r}")
    fw, fh = fig.get_size_inches()
    ts = TYPE_SCALE["header_title"] if title_size is None else title_size
    ss = TYPE_SCALE["header_subtitle"] if subtitle_size is None else subtitle_size
    fs = TYPE_SCALE["footer"] if footer_size is None else footer_size

    if reserve:
        need = header_height(fig, subtitle=subtitle is not None,
                             title_size=ts, subtitle_size=ss, top=top, gap=gap)
        fig.subplots_adjust(top=max(0.05, 1.0 - need / fh))

    x = (left / fw) if align == "left" else 0.5
    ha = "left" if align == "left" else "center"
    artists = {}

    y = 1.0 - top / fh
    artists["title"] = fig.text(x, y, title, ha=ha, va="top", fontsize=ts,
                                fontweight="bold", color=COLORS["ink"])
    y -= (ts * 1.25 / 72.0) / fh

    if subtitle:
        y -= gap / fh
        artists["subtitle"] = fig.text(x, y, subtitle, ha=ha, va="top",
                                       fontsize=ss, color=COLORS["sub"])

    if corner:
        artists["corner"] = fig.text(
            1.0 - left / fw, 1.0 - top / fh, corner, ha="right", va="top",
            fontsize=ss, color=COLORS["muted"])

    if footer:
        artists["footer"] = fig.text(
            left / fw, bottom / fh, footer, ha="left", va="bottom",
            fontsize=fs, color=COLORS["muted"])

    return artists


def add_footer(fig, text, left=0.11, bottom=0.07, size=None, color=None,
               align="left", **kwargs):
    """Small grey provenance line at the bottom of the figure.

    The footer half of :func:`add_figure_header`, separated out because a
    figure often wants provenance without a title block. Returns the artist, so
    callers can measure it for overlap checks.
    """
    fw, fh = fig.get_size_inches()
    x = (left / fw) if align == "left" else 0.5
    return fig.text(x, bottom / fh, text,
                    ha="left" if align == "left" else "center", va="bottom",
                    fontsize=TYPE_SCALE["footer"] if size is None else size,
                    color=COLORS["muted"] if color is None else color,
                    **kwargs)


# ---- Axes decoration -------------------------------------------------------
def panel_title(ax, text, size=None, weight="bold", pad=5, color=None,
                **kwargs):
    """Set a panel title at the house size instead of a hand-picked one.

    Exists because the library had 8.5 / 8.6 / 9.0 in circulation for the same
    role.  ``size`` still accepts an override for the rare dense panel.
    """
    return ax.set_title(
        text,
        fontsize=TYPE_SCALE["panel_title"] if size is None else size,
        fontweight=weight, pad=pad,
        color=COLORS["ink"] if color is None else color, **kwargs)


def _stamp_panel_label(ax, label, x, y, weight, size, units, color=None,
                       **kwargs):
    """The one implementation behind every panel-letter helper.

    ``units="axes"`` treats ``x``/``y`` as axes fractions -- the historical
    behaviour, kept so the existing library renders identically.
    ``units="inches"`` offsets from the axes' top-left corner in inches, which
    is what new figures should use: an axes-fraction offset means a different
    physical gap in every panel size, and that is exactly how this library
    ended up with x running from -0.045 to -0.155 for the same label.
    """
    kw = dict(ha="left", va="bottom", fontweight=weight, fontsize=size,
              **kwargs)
    if color is not None:
        kw["color"] = color
    if units == "axes":
        return ax.text(x, y, label, transform=ax.transAxes, **kw)
    if units == "inches":
        fig = ax.get_figure()
        fw, fh = fig.get_size_inches()
        bb = ax.get_position()
        return fig.text(bb.x0 + x / fw, bb.y1 + y / fh, label, **kw)
    raise ValueError(f"units must be 'axes' or 'inches', got {units!r}")


def add_panel_label(ax, label, x=-0.12, y=1.05, weight="bold", size=11,
                    units="axes", color=None, **kwargs):
    """Stamp one panel letter. Signature and defaults are unchanged.

    Pass ``units="inches"`` (with e.g. ``x=-0.34, y=0.10``) for the offset
    convention new figures should prefer.
    """
    return _stamp_panel_label(ax, label, x, y, weight, size, units,
                              color=color, **kwargs)


def panel_labels_inches(fig, axes, labels=None, fmt="({})", start=0,
                        dx=-0.34, dy=0.10, size=None, weight="bold",
                        uppercase=False, color=None, **kwargs):
    """Stamp ``(a) (b) (c) ...`` at equal *inch* offsets. House default.

    Mirrors ``figkit.panel_labels_inches`` so a script can reach it without
    importing figkit; both route through :func:`_stamp_panel_label`.
    """
    axes = _as_axes_list(axes)
    if size is None:
        size = TYPE_SCALE["panel_label"]
    for ax, lab in zip(axes, _panel_labels(len(axes), labels, fmt, start,
                                           uppercase)):
        _stamp_panel_label(ax, lab, dx, dy, weight, size, "inches",
                           color=color, **kwargs)
    return axes


def style_axes(ax):
    """Trim spines to the data-facing L-shape look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def despine(ax, top=True, right=True, left=False, bottom=False, offset=None):
    """Hide the requested spines; optionally push the rest outward."""
    flags = {"top": top, "right": right, "left": left, "bottom": bottom}
    for side, hide in flags.items():
        spine = ax.spines[side]
        if hide:
            spine.set_visible(False)
        elif offset:
            spine.set_position(("outward", offset))
    if left and right:
        ax.yaxis.set_ticks_position("none")
    elif left:
        ax.yaxis.set_ticks_position("right")
    if top and bottom:
        ax.xaxis.set_ticks_position("none")
    elif bottom:
        ax.xaxis.set_ticks_position("top")
    return ax


def _as_axes_list(axes):
    """Normalise a list / ndarray / dict-values of axes, dropping None."""
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = list(axes)
    return [a for a in axes if a is not None]


def _panel_labels(n, labels=None, fmt="({})", start=0, uppercase=False):
    """``(a) (b) (c) ...`` -- lowercase and parenthesised is the house form."""
    if labels is not None:
        return list(labels)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if uppercase else \
               "abcdefghijklmnopqrstuvwxyz"
    return [fmt.format(alphabet[(start + i) % 26]) for i in range(n)]


def letter_panels(axes, labels=None, fmt="({})", start=0, x=-0.12, y=1.05,
                  size=11, weight="bold", uppercase=False, units="axes",
                  color=None, **kwargs):
    """Stamp (a) (b) (c) ... on a list / array / dict-values of axes.

    Defaults are unchanged, so existing figures render identically. Pass
    ``units="inches"`` (with ``x=-0.34, y=0.10``) for the offset convention
    new figures should use.
    """
    axes = _as_axes_list(axes)
    for ax, lab in zip(axes, _panel_labels(len(axes), labels, fmt, start,
                                           uppercase)):
        _stamp_panel_label(ax, lab, x, y, weight, size, units, color=color,
                           **kwargs)
    return axes


_LOC_MAP = {
    "upper left": (0.0, 1.0, "left", "top"),
    "upper right": (1.0, 1.0, "right", "top"),
    "upper center": (0.5, 1.0, "center", "top"),
    "lower left": (0.0, 0.0, "left", "bottom"),
    "lower right": (1.0, 0.0, "right", "bottom"),
    "lower center": (0.5, 0.0, "center", "bottom"),
    "center left": (0.0, 0.5, "left", "center"),
    "center right": (1.0, 0.5, "right", "center"),
    "center": (0.5, 0.5, "center", "center"),
}


def annotate_box(ax, text, loc="upper left", fontsize=None, color="#1a1a1a",
                 facecolor="white", edgecolor="#cccccc", alpha=0.92,
                 weight="normal", pad=0.35, margin=0.03, linewidth=0.6,
                 **kwargs):
    """Statistics call-out in a white rounded box (the house annotation)."""
    if loc not in _LOC_MAP:
        raise ValueError(f"unknown loc {loc!r}; choose from {sorted(_LOC_MAP)}")
    fx, fy, ha, va = _LOC_MAP[loc]
    x = fx + margin * (1 if fx == 0.0 else -1 if fx == 1.0 else 0)
    y = fy + margin * (1 if fy == 0.0 else -1 if fy == 1.0 else 0)
    if fontsize is None:
        fontsize = mpl.rcParams["font.size"] - 1
    return ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color,
        fontsize=fontsize, fontweight=weight, zorder=6,
        bbox=dict(boxstyle=f"round,pad={pad}", facecolor=facecolor,
                  edgecolor=edgecolor, linewidth=linewidth, alpha=alpha),
        **kwargs)


def sig_stars(p, ns="n.s."):
    """p-value -> ``***`` / ``**`` / ``*`` / ``n.s.``."""
    if p is None or not np.isfinite(p):
        return ns
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns


def add_sig_bracket(ax, x1, x2, y, text="*", height=None, color="#333333",
                    lw=0.9, fontsize=None, weight="bold", gap=None):
    """Draw a significance bracket between ``x1`` and ``x2`` at height ``y``."""
    span = float(np.ptp(ax.get_ylim())) or 1.0
    h = span * 0.022 if height is None else height
    g = span * 0.008 if gap is None else gap
    if fontsize is None:
        fontsize = mpl.rcParams["font.size"] - 1
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], lw=lw, color=color,
            solid_joinstyle="miter", clip_on=False, zorder=5)
    return ax.text((x1 + x2) / 2.0, y + g, text, ha="center", va="bottom",
                   color=color, fontsize=fontsize, fontweight=weight,
                   zorder=5, clip_on=False)


def add_colorbar(fig, mappable, ax, label="", orientation="vertical",
                 fraction=0.046, pad=0.03, shrink=0.92, aspect=22,
                 ticks=None, labelsize=None, extend="neither"):
    """House-style colorbar: hairline outline, short ticks, small label."""
    cb = fig.colorbar(mappable, ax=ax, orientation=orientation,
                      fraction=fraction, pad=pad, shrink=shrink,
                      aspect=aspect, ticks=ticks, extend=extend)
    if labelsize is None:
        labelsize = mpl.rcParams["font.size"] - 1
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=labelsize, width=0.7, length=2.5,
                      color="#333333")
    if label:
        cb.set_label(label, fontsize=labelsize, color="#1a1a1a")
    return cb


# ---- Data helpers ----------------------------------------------------------
def smooth_line(x, y, frac=0.3, n=200, xmin=None, xmax=None):
    """Gaussian-kernel LOWESS-style smoother.

    Returns ``(xg, yg)`` on a dense grid; ``frac`` is the kernel bandwidth as
    a fraction of the x-range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    lo = xs.min() if xmin is None else xmin
    hi = xs.max() if xmax is None else xmax
    xg = np.linspace(lo, hi, n)
    span = xs.max() - xs.min()
    h = frac * span if span > 0 else 1.0
    w = np.exp(-0.5 * ((xs[None, :] - xg[:, None]) / h) ** 2)
    denom = w.sum(axis=1)
    denom[denom == 0] = 1.0
    return xg, (w @ ys) / denom
