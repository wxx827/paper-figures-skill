"""Shared measurement tool for the WM batches.  One ruler, one implementation.

    cd scripts && python _wm_metrics.py 1165 1176

Prints, per figure: warm area %, the WM3 "budget" %, deep-blue %, ink ratio at
both thresholds, the *declared* aspect ratio read out of the script, the
*delivered* aspect ratio measured on the PNG, and the drift between them.

Why this file exists
--------------------
`BRIEF_WM.md` section 1.8 defines two quantities.  The warm predicate has
always been three lines of code in the brief, and four groups across two
batches have reproduced it to within 0.01 pp.  The ink ratio was prose until
batch 4, then became code - and batch 5 showed the remaining failure mode:
one group pasted the code and matched exactly, the other retyped it, dropped
the `thumbnail` line, and came out 1.3-2.6 pp low, which was enough to put a
figure on the wrong side of the 50 % cap.

So: do not re-implement either predicate.  Import from here, or run this.

`closed_ratio` is the batch-4 helper promoted to shared code: it hands back
the decimal places you must print for a quotient to be reproducible from the
two numbers you printed above it.

`aspect` / `declared_aspect` are the batch-7 addition.  `savefig.bbox =
"tight"` *expands* the canvas to fit every artist, so the `figsize=` in the
source is a request, not the delivered shape (DESIGN_SYSTEM section 12.11).
Section 1.2 of DESIGN_SYSTEM constrains the delivered shape, so an audit that
reads `figsize=` is auditing the wrong number.  `declared_aspect` parses the
script with `ast`, not a regex, because a third of the WM scripts write
`W, H = 11.9, 3.5` at module level and then `figsize=(w, h)` inside a helper
that takes them as parameters - a regex sees only the two bare names.
"""
import ast
import glob
import operator
import os
import sys

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "output"))

WARM_HUE = (330.0, 90.0)          # wraps through 0; -30 deg == 330 deg
INK_LOOSE, INK_STRICT = 0.94, 0.75
SAMPLE = (900, 900)
DPI = 300.0                       # pubstyle's savefig.dpi
ASPECT_BAND = (0.55, 0.80)        # DESIGN_SYSTEM section 1.2


def _load(path):
    """Every predicate is measured at the same sampling scale."""
    im = Image.open(path).convert("RGB")
    im.thumbnail(SAMPLE)
    return np.asarray(im).astype(float) / 255.0


def _luminance(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def ink(path, thr=INK_LOOSE):
    """Share of non-white pixels.  thr=0.94 loose, thr=0.75 strict."""
    return float((_luminance(_load(path)) < thr).mean())


def warm(path):
    """Warm area share: hue in [-30, 90] deg and S > 0.25 and V > 0.35."""
    hsv = rgb_to_hsv(_load(path))
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    hot = (h >= WARM_HUE[0]) | (h <= WARM_HUE[1])
    return float((hot & (s > 0.25) & (v > 0.35)).mean())


def budget(path):
    """Warm area plus any other non-blue high-chroma pixels (WM3 ruler)."""
    hsv = rgb_to_hsv(_load(path))
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    hot = ((h >= WARM_HUE[0]) | (h <= WARM_HUE[1])) & (s > 0.25) & (v > 0.35)
    other = (s >= 0.50) & (v >= 0.55) & ~((h >= 185) & (h <= 265))
    return float((hot | other).mean())


def deep_blue(path):
    hsv = rgb_to_hsv(_load(path))
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    return float(((h >= 195) & (h <= 265) & (s > 0.30) & (v < 0.75)).mean())


# ---- canvas shape ----------------------------------------------------------
def size_inches(path):
    """Delivered canvas size in inches, from the PNG's own pixel count."""
    with Image.open(path) as im:
        w, h = im.size
    return w / DPI, h / DPI


def aspect(path):
    """Delivered h/w, measured on the PNG.

    This is the number DESIGN_SYSTEM section 1.2 constrains to 0.55-0.80.
    Pixels or inches give the same quotient; inches is what you compare
    against a `figsize=`.
    """
    w, h = size_inches(path)
    return h / w


def in_band(a, band=ASPECT_BAND):
    return band[0] <= a <= band[1]


_BINOP = {ast.Add: operator.add, ast.Sub: operator.sub,
          ast.Mult: operator.mul, ast.Div: operator.truediv}


def _number(node, env):
    """Fold a numeric expression against ``env``; None when a name is unknown."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        v = env.get(node.id)
        return v if isinstance(v, float) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _number(node.operand, env)
        return None if v is None else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOP:
        a, b = _number(node.left, env), _number(node.right, env)
        if a is None or b is None:
            return None
        return _BINOP[type(node.op)](a, b)
    return None


def _pair(node, env):
    """A two-element size, either written out or hiding behind one name."""
    if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == 2:
        w, h = _number(node.elts[0], env), _number(node.elts[1], env)
        return (w, h) if w and h else None
    if isinstance(node, ast.Name):
        v = env.get(node.id)
        return v if isinstance(v, tuple) else None
    return None


def _constants(tree):
    """Every name bound to a number or a pair of numbers, in source order.

    Flat, not scoped: these scripts bind each of `W` / `H` / `FW` / `FH` once,
    and a scope-accurate resolver would buy nothing but a false sense of rigour.
    """
    env = {}
    assigns = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign)),
                     key=lambda n: n.lineno)
    for node in assigns:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                val = _number(node.value, env)
                if val is None:
                    val = _pair(node.value, env)
                if val is not None:
                    env[tgt.id] = val
            elif isinstance(tgt, (ast.Tuple, ast.List)) and \
                    isinstance(node.value, (ast.Tuple, ast.List)) and \
                    len(tgt.elts) == len(node.value.elts):
                # W, H = 11.9, 3.5
                for name, expr in zip(tgt.elts, node.value.elts):
                    v = _number(expr, env)
                    if isinstance(name, ast.Name) and v is not None:
                        env[name.id] = v
    return env


def _enclosing(tree):
    """node -> the FunctionDef it sits in, for the parameter-binding step."""
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for kid in ast.walk(fn):
                owner.setdefault(kid, fn)
    return owner


def _bind_params(tree, fn, env):
    """`def canvas(w, h)` called once as `canvas(W, H)` - bind w, h.

    This is the pattern a regex cannot see: the size lives in module-level
    `W, H`, and the `figsize=` two hundred lines away names only `w, h`.
    The parameters are dropped from the inherited env first, because several
    scripts also use `w` / `h` as throwaway box dimensions elsewhere, and
    inheriting one of those is how you silently report the wrong figure.
    """
    names = [a.arg for a in fn.args.args]
    env = {k: v for k, v in env.items() if k not in names}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                and call.func.id == fn.name:
            bound = dict(env)
            for name, expr in list(zip(names, call.args)) + \
                    [(kw.arg, kw.value) for kw in call.keywords
                     if kw.arg in names]:
                v = _number(expr, env)
                if v is None:
                    v = _pair(expr, env)
                if v is not None:
                    bound[name] = v
            return bound
    return env


def _parse(script_path):
    """`utf-8-sig` because a handful of older scripts carry a BOM."""
    with open(script_path, encoding="utf-8-sig") as fh:
        return ast.parse(fh.read())


def _calls(tree):
    return sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)),
                  key=lambda n: n.lineno)


def _figsize_in(tree, env, owner=None, seed=None, within=None):
    """First `figsize=` that resolves to two numbers, in source order."""
    owner = _enclosing(tree) if owner is None else owner
    body = set(ast.walk(within)) if within is not None else None
    for call in _calls(tree):
        if body is not None and call not in body:
            continue
        for kw in call.keywords:
            if kw.arg != "figsize":
                continue
            scope = _bind_params(tree, owner[call], env) if call in owner \
                else dict(env)
            scope.update(seed or {})
            size = _pair(kw.value, scope)
            if size is not None:
                return size
    return None


def _sibling_imports(tree):
    """local name -> (module, attribute) for `from figkit import inch_canvas`."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                out[alias.asname or alias.name] = (node.module, alias.name)
    return out


def _figsize_via_helper(script_path, tree, env, owner):
    """`inch_canvas(W, H)` - the twelve diagram scripts build the canvas one
    module away, so the script itself contains no `figsize=` at all."""
    here = os.path.dirname(os.path.abspath(script_path))
    origin = _sibling_imports(tree)
    for call in _calls(tree):
        if not isinstance(call.func, ast.Name) or call.func.id not in origin:
            continue
        mod, attr = origin[call.func.id]
        path = os.path.join(here, mod + ".py")
        if not os.path.exists(path):
            continue
        try:
            mtree = _parse(path)
        except (OSError, SyntaxError):
            continue
        fn = next((n for n in ast.walk(mtree)
                   if isinstance(n, ast.FunctionDef) and n.name == attr), None)
        if fn is None:
            continue
        caller = _bind_params(tree, owner[call], env) if call in owner \
            else dict(env)
        seed = {}
        for name, expr in zip([a.arg for a in fn.args.args], call.args):
            v = _number(expr, caller)
            if v is not None:
                seed[name] = v
        size = _figsize_in(mtree, _constants(mtree), seed=seed, within=fn)
        if size is not None:
            return size
    return None


def declared_figsize(script_path):
    """The `figsize=` a script asks for, as (w, h) inches, or None.

    AST, never a regex: `BRIEF_WM.md` and DESIGN_SYSTEM section 1.3 both note
    that indirect spellings are the common case here, and a regex silently
    reports nothing for them.  Three spellings appear in the WM batches:

        figsize=(11.0, 4.2)          written out
        FW, FH = 10.4, 6.4  ...  figsize=(FW, FH)
        W, H = 12.0, 3.9    ...  inch_canvas(W, H)   -> figkit, another module
    """
    tree = _parse(script_path)
    env, owner = _constants(tree), _enclosing(tree)
    size = _figsize_in(tree, env, owner)
    if size is None:
        size = _figsize_via_helper(script_path, tree, env, owner)
    return size


def declared_aspect(script_path):
    """h/w as declared in the source.  Compare against `aspect()`, not to it.

    A figure whose two numbers disagree is not necessarily broken - but it is
    un-auditable from source, which is how a whole batch drifted out of band
    while every script looked compliant.
    """
    size = declared_figsize(script_path)
    return None if size is None else size[1] / size[0]


_SCRIPT_INDEX = {}


def script_for(num, scripts_dir=HERE):
    """The script that writes `<num>_*.png`, found by its own savefig call."""
    if scripts_dir not in _SCRIPT_INDEX:
        index = {}
        for path in sorted(glob.glob(os.path.join(scripts_dir, "fig_*.py"))):
            try:
                tree = _parse(path)
            except (OSError, SyntaxError):
                continue          # another window may be mid-write
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and \
                        isinstance(node.value, str) and \
                        node.value[:4].isdigit() and "_" in node.value:
                    index.setdefault(int(node.value[:4]), path)
        _SCRIPT_INDEX[scripts_dir] = index
    return _SCRIPT_INDEX[scripts_dir].get(int(num))


def closed_ratio(num, den, lo=2, hi=4):
    """Fewest decimals at which the printed quotient survives the round trip.

    Returns ``(num_str, den_str, quotient_str, dp)``.  Print all three and a
    reader who divides the first two gets the third, which is the only way a
    derived number on a figure can be trusted.
    """
    for dp in range(lo, hi + 1):
        ns, ds = f"{num:.{dp}f}", f"{den:.{dp}f}"
        if abs(float(ns) / float(ds) - num / den) < 0.5 * 10 ** -(dp + 1):
            return ns, ds, f"{float(ns) / float(ds):.{dp + 1}f}", dp
    ns, ds = f"{num:.{hi}f}", f"{den:.{hi}f}"
    return ns, ds, f"{float(ns) / float(ds):.{hi + 1}f}", hi


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    print(f"{'num':>5} {'warm%':>7} {'budget%':>8} {'dblue%':>7} "
          f"{'ink.94':>7} {'ink.75':>7} {'decl':>6} {'h/w':>6} {'drift':>7}  "
          f"name")
    for n in range(lo, hi + 1):
        hits = sorted(glob.glob(os.path.join(OUT, f"{n}_*.png")))
        if not hits:
            continue
        p = hits[0]
        try:
            got = aspect(p)
        except OSError:
            # Another agent may be mid-write; skip rather than report a lie.
            print(f"{n:>5} {'(being written)':>47}  {os.path.basename(p)}")
            continue
        src = script_for(n)
        want = declared_aspect(src) if src else None
        # drift is a subtraction of the two columns to its left, so a reader
        # can check it without trusting this script (see closed_ratio).
        decl = f"{want:>6.3f}" if want else "     ?"
        drift = f"{got - want:>+7.3f}" if want else "      ?"
        flag = "" if in_band(got) else "  <- out of band"
        print(f"{n:>5} {100 * warm(p):>7.2f} {100 * budget(p):>8.2f} "
              f"{100 * deep_blue(p):>7.2f} {100 * ink(p):>7.1f} "
              f"{100 * ink(p, INK_STRICT):>7.1f} {decl} {got:>6.3f} {drift}  "
              f"{os.path.basename(p)}{flag}")


if __name__ == "__main__":
    main()
