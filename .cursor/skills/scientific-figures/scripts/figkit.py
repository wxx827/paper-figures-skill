"""Shared figure-building helpers for the scientific-figures library.

Import alongside pubstyle:

    from pubstyle import set_pub_style, PALETTES, savefig
    from figkit import add_zoom_inset, text_card, inch_axes

``pubstyle`` owns *style* (rcParams, palettes, save paths).  ``figkit`` owns
*construction*: the geometry, layering and anti-overlap tricks that thirty-odd
agents each rediscovered while building the 352-figure gallery.  Every helper
here replaces between 2 and 39 near-identical private copies that were living
inside individual ``fig_*.py`` scripts.

Sections
--------
Canvas                 inch-based axes placement, box anchors
Zoom & inset           image tiles, colour-boxed magnifier insets
Labels & annotation    leader-line labels, inch-accurate panel letters, halos
Cards & primitives     rounded cards and architecture-diagram elements
Scales                 shared limits / shared norms ("don't let the figure lie")
Layout safety          clipping, background cards, aspect-safe colorbars
Data                   local-linear smoother

The module is standalone: it imports from ``pubstyle`` when available but
falls back to its own constants so it can be imported from anywhere.
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import (Circle, ConnectionPatch, FancyArrowPatch,
                                FancyBboxPatch, Rectangle)

# House neutrals, straight out of DESIGN_SYSTEM.md section 2.1, so that a
# diagram drawn with figkit primitives matches a chart drawn with pubstyle.
INK = "#1a1a1a"        # titles, block labels
RULE = "#333333"       # axis-weight strokes, arrows
SUB = "#555555"        # secondary text
MUTED = "#777777"      # tensor shapes, units, side notes
HAIR = "#cccccc"       # hairline separators, card edges
SOFT = "#f5f5f5"       # background cards
BLOCK = "#3b5b92"      # default diagram block fill (house blue)
ACCENT = "#b23a48"     # single emphasis colour (house red)
# ROI / zoom-box colours.  DESIGN_SYSTEM section 2.3 bans the saturated reds
# (#e8232a and friends) that ad-hoc zoom code kept reaching for; these two are
# house colours that still separate cleanly from photographic content, and two
# is the limit - a third boxed crop per tile is unreadable.
ZOOM = "#b23a48"
ZOOM_ALT = "#c9a227"

__all__ = [
    # canvas
    "inch_canvas", "inch_axes", "axes_inches", "use_inch_units", "anchors",
    # zoom & inset
    "image_axes", "roi_rect", "add_zoom_inset", "ZOOM", "ZOOM_ALT",
    # labels & annotation
    "place_outside_labels", "panel_label_inches", "panel_labels_inches",
    "halo_text", "add_scale_bar", "corner_stamp",
    # cards & diagram primitives
    "text_card", "panel_card", "block", "arrow", "connect", "route",
    "feature_map", "op_circle", "tensor_label",
    # scales
    "shared_limits", "apply_shared_limits", "shared_norm", "norm01",
    # layout safety
    "unclip_artist", "unclip_legend", "background_card", "safe_colorbar",
    # data
    "lowess",
]


# ===========================================================================
# Canvas
# ===========================================================================
def inch_axes(fig, rect_inches=None, aspect="equal", absolute=False,
              zorder=None):
    """Axes whose *data units are inches*; the single best layout habit here.

    Diagram code normally mixes three coordinate systems (data, axes fraction,
    figure fraction), so "make this gap the same as that gap" turns into
    arithmetic that silently breaks whenever the figure is resized.  On an
    inch canvas a gap of 0.12 is 0.12 in on paper everywhere, arrow gaps and
    corner radii are directly comparable, and ``set_aspect("equal")`` cannot
    shrink the axes box because the box already has the requested aspect.

    ``rect_inches`` is ``(x0, y0, w, h)`` measured from the figure's
    lower-left corner; ``None`` means the whole figure.  With
    ``absolute=True`` the data limits are the figure-wide inch coordinates,
    so several panels share one ruler.
    """
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = (0.0, 0.0, fw, fh) if rect_inches is None else rect_inches
    ax = fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh])
    if absolute:
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0, y0 + h)
    else:
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
    if aspect:
        ax.set_aspect(aspect)
    ax.axis("off")
    if zorder is not None:
        ax.set_zorder(zorder)
    return ax


def inch_canvas(w, h, aspect="equal", **fig_kw):
    """``(fig, ax)`` for a whole-figure inch canvas - the diagram entry point.

    Ten architecture scripts had opened with a byte-identical private copy of
    this; it is the standard opening for any figure that is drawn rather than
    plotted.
    """
    fig = plt.figure(figsize=(w, h), **fig_kw)
    return fig, inch_axes(fig, (0.0, 0.0, w, h), aspect=aspect)


def axes_inches(fig, rect_inches, **kwargs):
    """A *normal* (ticked) axes positioned by inches instead of fractions.

    Lets a mixed figure keep one mental model: panels are laid out on an inch
    grid, so column gutters stay physically equal even when the panels have
    different widths.
    """
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = rect_inches
    return fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh], **kwargs)


def use_inch_units(ax, origin=(0.0, 0.0), off=True):
    """Re-map an existing axes so its data units are inches; returns ``(w, h)``.

    For the common case of one hand-drawn panel inside an otherwise normal
    ``gridspec`` figure.  Call it *after* the layout is final (post
    ``tight_layout``), otherwise the position it reads is stale.
    """
    fig = ax.figure
    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    w, h = bb.width * fw, bb.height * fh
    ax.set_xlim(origin[0], origin[0] + w)
    ax.set_ylim(origin[1], origin[1] + h)
    ax.set_aspect("equal")
    if off:
        ax.axis("off")
    return w, h


def anchors(x, y, w, h):
    """Edge/centre anchor dict for a box centred on ``(x, y)``.

    Arrows should attach to *named* points, never to hand-typed coordinates;
    that is what makes a diagram survive a later "move this block 0.2 left".
    """
    return dict(x=x, y=y, w=w, h=h, c=(x, y),
                l=(x - w / 2, y), r=(x + w / 2, y),
                t=(x, y + h / 2), b=(x, y - h / 2),
                tl=(x - w / 2, y + h / 2), tr=(x + w / 2, y + h / 2),
                bl=(x - w / 2, y - h / 2), br=(x + w / 2, y - h / 2))


# ===========================================================================
# Zoom & inset
# ===========================================================================
def image_axes(ax, color="#8a8a8a", lw=0.6, frame=True):
    """Turn an axes into a bare image tile: square, tickless, hairline frame.

    Thirty-nine scripts carried a copy of this.  The point is the *frame*:
    ``pubstyle`` hides the top/right spines, which looks wrong on an image
    (the picture appears to bleed), so image tiles have to re-enable all four
    and thin them down to a hairline.
    """
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_linewidth(lw)
        s.set_color(color)
    return ax


def roi_rect(ax, box_xywh, color=ZOOM, lw=1.15, zorder=6, ls="-"):
    """Draw the region-of-interest rectangle on an image, in pixel units.

    The ``-0.5`` offset matters: ``imshow`` centres pixel *(0, 0)* on data
    coordinate *(0, 0)*, so a rectangle placed at the integer corner is half a
    pixel off and the box visibly clips the feature it is meant to frame.
    """
    x0, y0, bw, bh = (int(v) for v in box_xywh)
    r = Rectangle((x0 - 0.5, y0 - 0.5), bw, bh, facecolor="none",
                  edgecolor=color, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(r)
    return r


def add_zoom_inset(ax, img, box_xywh, inset_loc, color=ZOOM, *,
                   match_aspect=True, box_lw=1.15, border_lw=1.6,
                   connect=False, interpolation="nearest", zorder=6,
                   **imshow_kw):
    """Colour-boxed magnifier: mark a crop on the image, paste it enlarged.

    Two traps are handled here.

    1. ``inset_axes`` bounds are fractions of the axes box *after*
       ``set_aspect("equal")`` has shrunk it, not of the nominal box.  So a
       square tile really does give a square inset for ``w == h`` - but the
       crop pasted into it is only undistorted if the crop window has the same
       aspect.  With ``match_aspect=True`` the requested ``box_xywh`` is grown
       about its centre (and clamped to the image) until its pixel aspect
       matches the inset's on-paper aspect, then drawn with ``aspect="auto"``
       so it fills the frame exactly with no letterboxing and no stretching.
       ``apply_aspect()`` is called first so the measurement is valid even
       before the figure has ever been drawn.
    2. The inset is a child axes and would be painted *under* the parent's
       image, so its zorder is lifted explicitly.

    ``box_xywh`` is ``(x, y, w, h)`` in image pixels, ``inset_loc`` is
    ``(x, y, w, h)`` in axes fractions.  Box and frame share ``color`` so the
    reader can pair them at a glance; keep the same crop across a row of
    methods or the comparison is meaningless.
    """
    x0, y0, bw, bh = (int(v) for v in box_xywh)
    ih, iw = img.shape[0], img.shape[1]

    if match_aspect:
        # On-paper aspect of the inset: needs the *applied* axes box, which
        # only exists after apply_aspect() has run (see doc note above).
        try:
            ax.apply_aspect()
            bb = ax.get_position()
            fw, fh = ax.figure.get_size_inches()
            want = ((inset_loc[2] * bb.width * fw) /
                    max(inset_loc[3] * bb.height * fh, 1e-9))
        except Exception:
            want = inset_loc[2] / max(inset_loc[3], 1e-9)
        have = bw / max(bh, 1e-9)
        if want > have:
            bw = min(iw, int(round(bh * want)))
        elif want < have:
            bh = min(ih, int(round(bw / want)))
        x0 = int(np.clip(x0 - (bw - int(box_xywh[2])) // 2, 0, max(iw - bw, 0)))
        y0 = int(np.clip(y0 - (bh - int(box_xywh[3])) // 2, 0, max(ih - bh, 0)))

    roi_rect(ax, (x0, y0, bw, bh), color=color, lw=box_lw, zorder=zorder)

    axi = ax.inset_axes(inset_loc)
    axi.imshow(img[y0:y0 + bh, x0:x0 + bw], interpolation=interpolation,
               aspect="auto", **imshow_kw)
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(border_lw)
        s.set_color(color)
    axi.set_zorder(zorder + 1)
    axi.patch.set_visible(True)

    if connect:
        right = inset_loc[0] > 0.4
        ax.add_artist(ConnectionPatch(
            xyA=(x0 + bw if right else x0, y0 + bh), coordsA=ax.transData,
            xyB=(1.0 if right else 0.0, 1.0), coordsB=axi.transAxes,
            color=color, lw=0.6, alpha=0.85, zorder=zorder))
    return axi


# ===========================================================================
# Labels & annotation
# ===========================================================================
def place_outside_labels(ax, items, r_edge=1.0, r_stub=1.09, x_knee=1.16,
                         x_text=1.22, min_gap=0.235, y_span=1.26,
                         name_size=7.2, stat_size=6.6, leader_color="#9a9a9a",
                         leader_lw=0.7, name_color=INK, stat_color=SUB,
                         dot_size=2.0):
    """Elbow leader labels around a pie/donut that can neither overlap nor cross.

    Naive placement puts each label at its wedge angle and thin wedges collide;
    nudging the collisions apart one at a time re-orders them and the leader
    lines cross, which reads as an error.  The fix is two sweeps per side: a
    top-down pass pushes every anchor at least ``min_gap`` below its
    predecessor, then a bottom-up pass lifts anything that overflowed the
    column.  Because both passes only ever preserve the original y ordering,
    crossing leaders are geometrically impossible.

    ``items`` is a sequence of ``(angle_deg, name, stat, colour)``.  Returns
    ``{name: y}`` for the final label positions.
    """
    placed = {}
    for side in (+1, -1):
        grp = [it for it in items
               if (np.cos(np.deg2rad(it[0])) >= 0) == (side > 0)]
        if not grp:
            continue
        grp.sort(key=lambda it: -np.sin(np.deg2rad(it[0])))
        ys = [r_edge * np.sin(np.deg2rad(it[0])) for it in grp]
        for i in range(1, len(ys)):                     # pass 1: top -> bottom
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        ys[-1] = max(ys[-1], -y_span)
        for i in range(len(ys) - 2, -1, -1):            # pass 2: bottom -> top
            ys[i] = max(ys[i], ys[i + 1] + min_gap)
        ys[0] = min(ys[0], y_span)
        for i in range(1, len(ys)):                     # re-settle the overflow
            ys[i] = min(ys[i], ys[i - 1] - min_gap)

        for (ang, name, stat, col), y in zip(grp, ys):
            a = np.deg2rad(ang)
            ax.plot([r_edge * np.cos(a), r_stub * np.cos(a),
                     side * x_knee, side * (x_text - 0.02)],
                    [r_edge * np.sin(a), r_stub * np.sin(a), y, y],
                    lw=leader_lw, color=leader_color, clip_on=False, zorder=1)
            ax.plot([side * (x_text - 0.02)], [y], marker="o", ms=dot_size,
                    color=col, clip_on=False, zorder=2)
            ha = "left" if side > 0 else "right"
            ax.text(side * x_text, y + 0.065, name, ha=ha, va="center",
                    fontsize=name_size, color=name_color, clip_on=False)
            if stat:
                ax.text(side * x_text, y - 0.075, stat, ha=ha, va="center",
                        fontsize=stat_size, color=stat_color, clip_on=False)
            placed[name] = y
    return placed


def panel_label_inches(fig, ax, text, dx=-0.30, dy=0.10, size=11,
                       weight="bold", color=INK, **kwargs):
    """Panel letter at a fixed *inch* offset from the axes' top-left corner.

    ``ax.text(-0.12, 1.05, transform=ax.transAxes)`` measures the offset in
    units of the panel's own width, so a panel spanning two columns pushes its
    letter roughly twice as far from the frame as its neighbours and the
    column of letters visibly zig-zags.  Reading ``ax.get_position()`` and
    dividing by the figure size converts the offset to paper inches, which is
    identical for every panel regardless of span.

    Delegates to ``pubstyle._stamp_panel_label`` so the library has exactly one
    panel-letter implementation; the defaults here are unchanged, so existing
    figures render identically.  ``fig`` is kept in the signature for
    backward compatibility but the axes' own figure is authoritative.
    """
    try:
        from pubstyle import _stamp_panel_label
    except Exception:
        bb = ax.get_position()
        fw, fh = fig.get_size_inches()
        return fig.text(bb.x0 + dx / fw, bb.y1 + dy / fh, text, fontsize=size,
                        fontweight=weight, color=color, ha="left",
                        va="bottom", **kwargs)
    return _stamp_panel_label(ax, text, dx, dy, weight, size, "inches",
                              color=color, **kwargs)


def panel_labels_inches(fig, axes, labels=None, fmt="({})", start=0,
                        uppercase=False, **kwargs):
    """Stamp ``(a) (b) (c) ...`` on a sequence of axes at equal inch offsets."""
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    axes = [a for a in axes if a is not None]
    if labels is None:
        alpha = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ" if uppercase
                 else "abcdefghijklmnopqrstuvwxyz")
        labels = [fmt.format(alpha[(start + i) % 26]) for i in range(len(axes))]
    return [panel_label_inches(fig, ax, lab, **kwargs)
            for ax, lab in zip(axes, labels)]


def halo_text(ax, x, y, s, color="white", halo="#1a1a1a", halo_lw=1.7,
              fontsize=6.4, zorder=9, **kwargs):
    """Text with a contrasting outline, for labels that sit on imagery.

    Any label placed on a photo, map or heat map will eventually cross a patch
    of its own colour and vanish.  A stroke path-effect keeps it legible over
    arbitrary content without the opaque bbox that would hide the data.
    """
    t = ax.text(x, y, s, color=color, fontsize=fontsize, zorder=zorder,
                **kwargs)
    t.set_path_effects([pe.withStroke(linewidth=halo_lw, foreground=halo)])
    return t


def add_scale_bar(ax, length, label=None, loc=(0.035, 0.055), height=0.018,
                  n_seg=2, color="#111111", alt="white", fontsize=6.0,
                  text_color="white", halo="#1a1a1a", zorder=6):
    """Checkerboard scale bar on an image axes; ``length`` is in data units.

    Eleven scripts had their own copy, each hard-wiring the image size into
    the placement arithmetic.  Here only the *length* is data-driven (so the
    bar stays metrically correct) while the position is in axes fractions, so
    the same call works on an inverted ``imshow`` axes, a map, or a plain plot.
    """
    span = abs(np.diff(ax.get_xlim())[0]) or 1.0
    frac = abs(length) / span
    fx, fy = loc
    seg = frac / max(n_seg, 1)
    for i in range(max(n_seg, 1)):
        ax.add_patch(Rectangle(
            (fx + i * seg, fy), seg, height, transform=ax.transAxes,
            facecolor=alt if i % 2 else color, edgecolor=color,
            linewidth=0.5, zorder=zorder, clip_on=False))
    if label:
        halo_text(ax, fx + frac / 2, fy + height + 0.020, label,
                  transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=fontsize, color=text_color, halo=halo,
                  zorder=zorder + 1)
    return frac


def corner_stamp(ax, text, loc="lower left", pad=0.028, fontsize=6.4,
                 weight="normal", color="white", halo="#1a1a1a", zorder=9):
    """Method / metric caption stamped inside an image tile.

    Captions written *outside* the tile steal vertical space from every row of
    a comparison grid; stamping them inside keeps the images as large as
    possible, which is the whole point of a qualitative figure.
    """
    v, h = loc.split()
    x = pad if h == "left" else 1.0 - pad
    y = pad if v == "lower" else 1.0 - pad
    return halo_text(ax, x, y, text, transform=ax.transAxes,
                     ha=h, va="bottom" if v == "lower" else "top",
                     fontsize=fontsize, fontweight=weight, color=color,
                     halo=halo, zorder=zorder)


# ===========================================================================
# Cards & diagram primitives
# ===========================================================================
def text_card(ax, x, y, w, h, text, fc=SOFT, ec=HAIR, fs=7.0, color=INK,
              weight="normal", ha="left", va="center", pad=0.07, lw=0.6,
              zorder=3, rounding=0.05, alpha=1.0, linespacing=1.25,
              title=None, title_fs=None, title_color=None, **text_kw):
    """Rounded card holding a block of text; ``(x, y)`` is the lower-left corner.

    Uses ``FancyBboxPatch`` with ``pad=0`` and an explicit ``rounding_size``
    so the drawn card is exactly ``w x h``.  The default ``boxstyle`` pad would
    inflate the card beyond the geometry you laid out, which is how a row of
    "aligned" cards ends up with unequal gutters.  ``mutation_aspect=1``
    keeps the corner radius circular on an inch canvas.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, alpha=alpha,
        mutation_aspect=1.0))
    tx = {"left": x + pad, "center": x + w / 2.0, "right": x + w - pad}[ha]
    ty = y + h / 2.0
    if title:
        ax.text(tx, y + h - pad, title, ha=ha, va="top",
                fontsize=title_fs or (fs + 0.6), color=title_color or color,
                fontweight="bold", zorder=zorder + 2)
        ty = y + (h - pad * 2.4) / 2.0
    t = ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, color=color,
                fontweight=weight, zorder=zorder + 2,
                linespacing=linespacing, **text_kw)
    return t


def panel_card(ax, x, y, w, h, fc=SOFT, ec=HAIR, lw=0.8, ls="-", zorder=1,
               rounding=0.08, label=None, label_fs=7.0, label_color=SUB):
    """Grouping card *behind* a cluster of diagram elements, centred on (x, y).

    Thirty-six scripts drew one.  Group membership is otherwise invisible: a
    reader cannot tell that four blocks form "the encoder" unless something
    encloses them.  Keep ``zorder`` below every element it contains.
    """
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
        mutation_aspect=1.0))
    if label:
        ax.text(x - w / 2 + 0.06, y + h / 2 - 0.06, label, ha="left",
                va="top", fontsize=label_fs, color=label_color,
                zorder=zorder + 1)
    return anchors(x, y, w, h)


def block(ax, x, y, w, h, label="", color=BLOCK, sub=None, ec="white",
          text_color="white", fontsize=8, sub_color=None, lw=0.9, ls="-",
          alpha=1.0, zorder=3, rounding=0.055, hatch=None, weight="normal",
          n_stack=1, stack_off=0.07):
    """Rounded module box centred on ``(x, y)``; returns its anchor dict.

    Centre-based geometry (rather than ``Rectangle``'s corner-based) is what
    makes a column of blocks trivially alignable: they share an ``x``.  The
    white edge is not decoration - it separates touching blocks of similar
    hue, which otherwise merge into one shape at print size.  ``n_stack > 1``
    draws offset ghost copies, the house idiom for "this module repeats".
    """
    for i in range(n_stack - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + i * stack_off, y - h / 2 + i * stack_off), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
            edgecolor=ec, linewidth=0.7, mutation_aspect=1.0,
            alpha=0.40 + 0.16 * (n_stack - 1 - i), zorder=zorder - i))
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, hatch=hatch,
        zorder=zorder, mutation_aspect=1.0))
    if label:
        ax.text(x, y + (0.09 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                fontweight=weight, zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.10, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.5, alpha=0.92,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=RULE, rad=0.0, lw=1.1, ls="-", ms=11, zorder=6,
          alpha=1.0, gap=0.0, gap_a=None, gap_b=None, style="-|>"):
    """Connection arrow with exact endpoints and an explicit, uniform gap.

    ``FancyArrowPatch``'s ``shrinkA/shrinkB`` default to 2 points, so every
    arrow silently starts and ends 2 pt away from where you asked - fine in
    isolation, visibly ragged once arrows of different lengths and angles meet
    the same block edge.  Here shrink is pinned to 0 and any breathing room is
    taken off the *geometry* instead: ``gap`` is trimmed along the segment in
    data units, i.e. in inches on an inch canvas, so every arrow leaves the
    same physical clearance from the box edge it points at.
    """
    (x0, y0), (x1, y1) = tuple(p0), tuple(p1)
    ga = gap if gap_a is None else gap_a
    gb = gap if gap_b is None else gap_b
    if ga or gb:
        dx, dy = x1 - x0, y1 - y0
        d = float(np.hypot(dx, dy))
        if d > 1e-9:
            ux, uy = dx / d, dy / d
            x0, y0 = x0 + ux * ga, y0 + uy * ga
            x1, y1 = x1 - ux * gb, y1 - uy * gb
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=ms, linewidth=lw, linestyle=ls,
                        color=color, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def connect(ax, a, b, side=None, gap=0.06, **kwargs):
    """Arrow between two anchor dicts, picking sensible edges automatically.

    Keeps diagram code declarative (``connect(ax, enc, dec)``) so that moving a
    block does not require re-deriving its neighbours' attachment points.
    """
    if side is None:
        side = "h" if abs(b["x"] - a["x"]) >= abs(b["y"] - a["y"]) else "v"
    if side == "h":
        p0 = a["r"] if b["x"] >= a["x"] else a["l"]
        p1 = b["l"] if b["x"] >= a["x"] else b["r"]
    else:
        p0 = a["t"] if b["y"] >= a["y"] else a["b"]
        p1 = b["b"] if b["y"] >= a["y"] else b["t"]
    return arrow(ax, p0, p1, gap=gap, **kwargs)


def route(ax, pts, color=RULE, lw=1.1, ls="-", ms=11, zorder=6, gap=0.0):
    """Orthogonal / poly-line routing with the arrow head on the last segment.

    Drawing the whole path as one ``FancyArrowPatch`` bends the head; drawing
    the elbow with ``plot`` and reserving the arrow for the final leg keeps the
    head aligned with the direction the reader actually follows.
    """
    pts = [tuple(p) for p in pts]
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder)
    return arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms,
                 zorder=zorder, gap_b=gap)


def feature_map(ax, x, y, w, h, n_layers=3, color=BLOCK, depth=0.055,
                alpha0=0.42, ec="white", lw=0.7, zorder=3, label=None,
                label_color=MUTED, label_fs=6.5):
    """Stacked feature-map cube (C x H x W) as offset slabs, front slab solid.

    A single rectangle cannot say "this tensor has channels"; the offset stack
    does it without the perspective drawing that a true 3D box would need and
    that never matches the flat blocks around it.  The returned anchors carry
    an extra ``top`` key for the *stack* top, so a label placed above the cube
    does not collide with the rear slabs.
    """
    for i in range(n_layers):
        k = n_layers - 1 - i
        a = alpha0 + (1.0 - alpha0) * (i / max(n_layers - 1, 1))
        ax.add_patch(Rectangle((x - w / 2 + k * depth, y - h / 2 + k * depth),
                               w, h, facecolor=color, alpha=a, edgecolor=ec,
                               linewidth=lw, zorder=zorder + i))
    a = anchors(x, y, w, h)
    a["top"] = y + h / 2 + (n_layers - 1) * depth
    if label:
        ax.text(x, a["top"] + 0.06, label, ha="center", va="bottom",
                fontsize=label_fs, color=label_color, zorder=zorder + n_layers)
    return a


def op_circle(ax, x, y, sym="+", r=0.135, color=RULE, fc="white", fontsize=9,
              zorder=7, lw=1.0):
    """Small circled operator (+, x, c) for residual adds, gates and concats.

    Must be a ``Circle`` on an *equal-aspect* axes - on a normal plot axes the
    same patch renders as an ellipse (see DESIGN_SYSTEM section 12.8), which is
    why architecture diagrams belong on an inch canvas.
    """
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=color, lw=lw,
                        zorder=zorder))
    ax.text(x, y + 0.004, sym, ha="center", va="center", fontsize=fontsize,
            color=color, zorder=zorder + 1)
    return dict(x=x, y=y, w=2 * r, h=2 * r, c=(x, y), l=(x - r, y),
                r=(x + r, y), t=(x, y + r), b=(x, y - r))


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kwargs):
    """Shape annotation such as ``256x64x64`` next to a block.

    Deliberately one weight lighter and one size smaller than the block label:
    shapes are reference material, not the message, and matching their weight
    to the block name makes the diagram read as noise.
    """
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kwargs)


# ===========================================================================
# Scales  -  the anti-lying tools
# ===========================================================================
def shared_limits(datasets, pad=0.05, symmetric=False, lo=None, hi=None,
                  log=False):
    """One ``(lo, hi)`` covering every dataset - use it on all sibling panels.

    Per-panel autoscaling is the most common way a correct figure tells a false
    story: matplotlib expands each panel to fill its own frame, so a weak
    response and a strong one look identical, and a control condition can end
    up appearing to be the largest effect on the page.  Compute the range once
    over all the data, then set it on every panel.

    ``pad`` is a fraction of the span (a fraction of the log span when
    ``log=True``).  ``symmetric=True`` mirrors the range about zero, which is
    what a diverging colour map or a signed-difference axis needs.
    """
    vals = []
    for d in datasets:
        a = np.asarray(d, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if log:
            a = a[a > 0]
        if a.size:
            vals.append(a)
    if not vals:
        return (0.0, 1.0)
    allv = np.concatenate(vals)
    a_lo = float(allv.min()) if lo is None else float(lo)
    a_hi = float(allv.max()) if hi is None else float(hi)
    if symmetric:
        m = max(abs(a_lo), abs(a_hi))
        a_lo, a_hi = -m, m
    if log:
        l0, l1 = np.log10(max(a_lo, 1e-300)), np.log10(max(a_hi, 1e-300))
        m = pad * (l1 - l0 or 1.0)
        return (10.0 ** (l0 - m), 10.0 ** (l1 + m))
    span = a_hi - a_lo
    if span <= 0:
        span = abs(a_hi) or 1.0
    m = pad * span
    return (a_lo - m, a_hi + m)


def apply_shared_limits(axes, datasets, axis="y", **kwargs):
    """Compute ``shared_limits`` and set them on every axes in one call."""
    lim = shared_limits(datasets, **kwargs)
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    for ax in axes:
        if ax is None:
            continue
        (ax.set_ylim if axis == "y" else ax.set_xlim)(*lim)
    return lim


def shared_norm(arrays, robust=None, symmetric=False, vmin=None, vmax=None):
    """One ``Normalize`` for a whole row of heat maps / images.

    Same failure as ``shared_limits`` but harsher, because a colour map has no
    tick labels to warn the reader: three panels autoscaled independently all
    saturate to the same colours and the difference the figure exists to show
    disappears.  Pass the returned object as ``norm=`` to every ``imshow`` and
    to the single shared colorbar.

    ``robust=(2, 98)`` clips to percentiles so one hot pixel cannot flatten the
    rest.  ``symmetric=True`` returns a plain ``Normalize`` mirrored about zero
    rather than ``TwoSlopeNorm`` - it is centred just the same but has no
    out-of-range tick failure mode (DESIGN_SYSTEM section 12.5).
    """
    vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return mpl.colors.Normalize(0.0, 1.0)
    allv = np.concatenate(vals)
    if robust:
        p_lo, p_hi = robust
        lo, hi = np.percentile(allv, [p_lo, p_hi])
    else:
        lo, hi = allv.min(), allv.max()
    lo = float(lo) if vmin is None else float(vmin)
    hi = float(hi) if vmax is None else float(vmax)
    if symmetric:
        m = max(abs(lo), abs(hi)) or 1.0
        lo, hi = -m, m
    if hi <= lo:
        hi = lo + 1e-9
    return mpl.colors.Normalize(vmin=lo, vmax=hi)


def norm01(a, lo=None, hi=None):
    """Rescale to 0..1 for display, NaN-safe.

    Only ever for *rendering* a synthetic field.  Never normalise per panel
    before a comparison - that is exactly the lie ``shared_norm`` prevents.
    """
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    return (a - lo) / (hi - lo + 1e-12)


# ===========================================================================
# Layout safety
# ===========================================================================
def unclip_artist(artist):
    """Detach an artist from its axes clip box and keep it in the tight bbox."""
    try:
        artist.set_clip_on(False)
        artist.set_clip_path(None)
    except Exception:
        pass
    try:
        artist.set_in_layout(True)
    except Exception:
        pass
    return artist


def unclip_legend(leg):
    """Make an ``add_artist``-ed legend immune to clipping and tight-bbox loss.

    A legend created by ``ax.legend()`` is handled specially by matplotlib, but
    a *second* legend attached with ``ax.add_artist(leg)`` is just another
    child artist: it inherits the axes clip path, and ``bbox_inches="tight"``
    computes the crop without it.  The result is a legend whose text is sliced
    down the middle - and it only ever happens on figures with two or more
    legends, which is why it survives review.  Clearing the clip path on the
    legend *and* its texts and handles fixes both halves of the problem.
    """
    unclip_artist(leg)
    for t in leg.get_texts():
        unclip_artist(t)
    for h in (getattr(leg, "legend_handles", None) or
              getattr(leg, "legendHandles", None) or []):
        unclip_artist(h)
    frame = getattr(leg, "get_frame", None)
    if frame is not None:
        unclip_artist(frame())
    return leg


def background_card(fig, rect, facecolor=SOFT, edgecolor="none", zorder=-5,
                    radius=0.0, alpha=1.0, linewidth=0.0, transform=None):
    """Figure-level grouping block that stays *behind* the axes it groups.

    Axes default to ``zorder=0``, so a card at ``zorder=0`` is a tie, and
    matplotlib breaks ties by position in ``fig.get_children()`` - which lists
    ``fig.patches`` *after* the axes.  The popular
    ``fig.patches.append(Rectangle(..., zorder=0))`` recipe therefore paints
    the card over the very panels it was meant to sit behind.  ``zorder=0`` is
    not "behind", it is "on top".  This helper uses ``fig.add_artist`` (which
    sorts before the axes) *and* a strictly negative zorder, so it is safe
    either way, and marks the card out of the layout so
    ``bbox_inches="tight"`` is not enlarged by it.

    ``rect`` is ``(x0, y0, w, h)`` in figure fractions.
    """
    x0, y0, w, h = rect
    kw = dict(facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
              alpha=alpha, zorder=zorder,
              transform=transform or fig.transFigure)
    if radius:
        patch = FancyBboxPatch((x0, y0), w, h,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               mutation_aspect=1.0, **kw)
    else:
        patch = Rectangle((x0, y0), w, h, **kw)
    fig.add_artist(patch)
    patch.set_in_layout(False)
    return patch


def safe_colorbar(fig, mappable, ax, label="", side="right", width=0.11,
                  pad=0.07, shrink=1.0, labelsize=None, ticks=None,
                  extend="neither", outline=RULE):
    """Colorbar that cannot detach from an ``set_aspect("equal")`` panel.

    ``fig.colorbar(mappable, ax=ax)`` carves the bar out of the axes' *nominal*
    rectangle - the one you passed to ``add_axes`` / ``gridspec`` - not the
    smaller box the equal-aspect image is actually drawn in.  Two things then
    go wrong at once: the bar is parked at the right edge of the nominal box,
    which on a wide slot can be an inch of whitespace away from the image, and
    it is given the nominal height, so it is taller than the image it labels.
    Shrinking the nominal box also re-centres the image, pushing the two
    further apart.  Reading the applied box and pinning the bar to *that*
    keeps them together at any figure size.

    ``width`` and ``pad`` are in inches, so bars are identical across panels of
    different sizes.  Returns the ``Colorbar``.
    """
    try:
        ax.apply_aspect()
    except Exception:
        pass
    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    w_f, p_f = width / fw, pad / fw
    hp_f, vp_f = pad / fh, width / fh
    if side in ("right", "left"):
        h = bb.height * shrink
        y0 = bb.y0 + (bb.height - h) / 2.0
        x0 = bb.x1 + p_f if side == "right" else bb.x0 - p_f - w_f
        rect, orientation = [x0, y0, w_f, h], "vertical"
    else:
        w = bb.width * shrink
        x0 = bb.x0 + (bb.width - w) / 2.0
        y0 = bb.y0 - hp_f - vp_f if side == "bottom" else bb.y1 + hp_f
        rect, orientation = [x0, y0, w, vp_f], "horizontal"

    try:
        cax = ax.inset_axes(rect, transform=fig.transFigure)
    except Exception:
        cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation,
                      ticks=ticks, extend=extend)
    if labelsize is None:
        labelsize = mpl.rcParams["font.size"] - 1
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(outline)
    cb.ax.tick_params(labelsize=labelsize, width=0.7, length=2.5,
                      color=outline)
    if side == "bottom":
        cb.ax.xaxis.set_ticks_position("bottom")
    if label:
        cb.set_label(label, fontsize=labelsize, color=INK)
    return cb


# ===========================================================================
# Data
# ===========================================================================
def lowess(x, y, frac=0.4, n_out=140, return_scatter=False):
    """Tricube-weighted local *linear* smoother; no statsmodels dependency.

    The Gaussian kernel average in ``pubstyle.smooth_line`` is a local
    *constant* fit, so it flattens the two ends of every trend it draws - and
    the ends are usually where the claim lives.  A local linear fit follows the
    slope right up to the boundary.  ``return_scatter=True`` also returns the
    local residual spread, for a shaded band.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        empty = np.array([])
        return (empty, empty, empty) if return_scatter else (empty, empty)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xout = np.linspace(xs[0], xs[-1], n_out)
    r = int(np.clip(np.ceil(frac * xs.size), 3, xs.size))
    yout = np.empty(n_out)
    sout = np.empty(n_out)
    for i, x0 in enumerate(xout):
        d = np.abs(xs - x0)
        h = max(np.partition(d, r - 1)[r - 1], 1e-9)
        w = np.clip(d / h, 0.0, 1.0)
        w = (1.0 - w ** 3) ** 3
        sw = w.sum()
        mx = (w * xs).sum() / sw
        my = (w * ys).sum() / sw
        vx = (w * (xs - mx) ** 2).sum()
        b = (w * (xs - mx) * (ys - my)).sum() / vx if vx > 1e-12 else 0.0
        yout[i] = my + b * (x0 - mx)
        res = ys - (my + b * (xs - mx))
        sout[i] = np.sqrt((w * res ** 2).sum() / sw)
    return (xout, yout, sout) if return_scatter else (xout, yout)
