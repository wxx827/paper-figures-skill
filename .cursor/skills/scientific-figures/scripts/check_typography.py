"""Audit the typography that ``check_figures.py`` cannot see.

``check_figures.py`` verifies that a PNG exists, is not blank and carries
300 dpi metadata.  All of that passes even when matplotlib silently fell back
to DejaVu because the Times stack did not resolve -- which is the one defect
that makes a figure look wrong on someone else's machine.  A PNG has no font
metadata, so the only way to know which face was actually used is to render a
PDF and read the fonts the renderer embedded.

    cd scripts
    python check_typography.py                 # env + source lint + render probe
    python check_typography.py --metrics       # prove the clone is Times-metric
    python check_typography.py --source-only   # lint only, no rendering
    python check_typography.py --render 40     # probe 40 scripts instead of 12
    python check_typography.py --pdf ../output/01_threshold_regression.pdf
    python check_typography.py --strict        # warnings also fail the run

Exit code is 1 when there is at least one error (0 with warnings only, unless
``--strict``), so this is safe to wire straight into CI.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTPUT_DIR = os.path.join(ROOT, "output")

sys.path.insert(0, HERE)

# ---- What counts as an acceptable face -------------------------------------
# PDF BaseFont names are PostScript names, not family names: Times New Roman
# embeds as "TimesNewRomanPSMT" and carries a random six-letter subset prefix.
# Match on a normalised, lowercased, punctuation-free form.
ALLOWED_FACES = [
    "timesnewroman", "times", "tinos", "nimbusroman", "liberationserif",
    "texgyretermes", "termes", "stix", "cmr", "cmmi", "cmsy", "cmex",
    # Song/Ming CJK faces reached through set_cn_font().
    "simsun", "songti", "stsong", "notoserifcjk", "sourcehanserif",
]
# DejaVu Sans Mono is a deliberate exception: the house stats call-out uses it
# for tabular figures, where a monospaced advance is the point. Any other
# DejaVu means the serif stack failed to resolve.
ALLOWED_EXCEPTIONS = ["dejavusansmono"]
FORBIDDEN_FACES = ["dejavu", "bitstreamvera"]

# Faces that are legitimately in SERIF_STACK without matching Times metrics.
# STIXGeneral is the last resort: it ships with matplotlib, so it guarantees a
# serif face exists at all, but it is roughly 12% narrower on some glyphs. It
# only ever renders when the six faces ahead of it are all missing, so the
# reflow it causes is the price of never falling through to DejaVu.
METRIC_EXCEPTIONS = {"STIXGeneral"}

FONTSIZE_RE = re.compile(
    r"\b(?:fontsize|labelsize|title_fontsize|titlesize)\s*=\s*"
    r"(\d+(?:\.\d+)?)\b")
RCPARAMS_RE = re.compile(r"rcParams\s*(?:\[|\.update)")
SET_TITLE_RE = re.compile(r"""set_title\(\s*(['"])(.*?)\1""", re.S)
HEADER_TITLE_RE = re.compile(
    r"""add_figure_header\(\s*\w+\s*,\s*(['"])(.*?)\1""", re.S)
PANEL_LABEL_RE = re.compile(
    r"""add_panel_label\(\s*[^,]+,\s*(?:f?)(['"])(.*?)\1""", re.S)
BASEFONT_RE = re.compile(rb"/BaseFont\s*/([#A-Za-z0-9+\-_,\.]+)")
FONTFILE_RE = re.compile(rb"/FontFile\d?")

NOISE = ("RequestsDependencyWarning", "urllib3 (", "warnings.warn(")


class Report:
    """Collects errors and warnings and prints them grouped by check."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.notes = []

    def error(self, check, msg):
        self.errors.append((check, msg))

    def warn(self, check, msg):
        self.warnings.append((check, msg))

    def note(self, msg):
        self.notes.append(msg)

    def dump(self, strict=False):
        for line in self.notes:
            print(line)
        for label, items in (("ERROR", self.errors), ("WARN", self.warnings)):
            if not items:
                continue
            print(f"\n{'=' * 62}\n{len(items)} {label.lower()}(s)")
            by_check = {}
            for check, msg in items:
                by_check.setdefault(check, []).append(msg)
            for check, msgs in by_check.items():
                print(f"\n--- {check} ({len(msgs)}) ---")
                for m in msgs[:40]:
                    print(f"  {label}: {m}")
                if len(msgs) > 40:
                    print(f"  ... and {len(msgs) - 40} more")
        print(f"\n{'=' * 62}")
        print(f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return 1 if (self.errors or (strict and self.warnings)) else 0


def normalise_face(name):
    """``FHXFSG+TimesNewRomanPS-BoldMT`` -> ``timesnewromanpsboldmt``."""
    if isinstance(name, bytes):
        name = name.decode("latin-1", "replace")
    if "+" in name:
        name = name.split("+", 1)[1]
    return re.sub(r"[^a-z0-9]", "", name.lower())


def classify_face(name):
    """Return ``"ok"``, ``"forbidden"`` or ``"unknown"`` for a PDF font name."""
    flat = normalise_face(name)
    if any(flat.startswith(x) or x in flat for x in ALLOWED_EXCEPTIONS):
        return "ok"
    if any(bad in flat for bad in FORBIDDEN_FACES):
        return "forbidden"
    if any(flat.startswith(good) or good in flat for good in ALLOWED_FACES):
        return "ok"
    return "unknown"


# ---- Check 1: does the stack resolve in this environment? ------------------
def check_environment(rep):
    import pubstyle as ps

    added = ps._register_bundled_fonts()
    rep.note(f"bundled fonts registered : {added or '(none found)'}")

    available = ps.available_fonts()
    present = [n for n in ps.SERIF_STACK if n in available]
    rep.note(f"serif stack resolvable   : {present or '(NONE)'}")

    if not present:
        rep.error("environment",
                  "no font from SERIF_STACK is installed and fonts/ did not "
                  "register anything; every figure will render in a fallback "
                  "face. See fonts/README.md.")
        return

    ps.set_pub_style()
    resolved = ps.resolved_font()
    rep.note(f"body text resolves to    : {resolved}")
    if resolved in ps._FORBIDDEN_FALLBACKS:
        rep.error("environment",
                  f"body text resolved to {resolved!r}, not a Times-metric face")


# ---- Check 2: which fonts did the renderer actually embed? -----------------
def inspect_pdf(path):
    """``(faces, embedded_count)`` read straight from the PDF bytes.

    Avoids a pypdf dependency: matplotlib writes font descriptors as plain
    dictionary objects, so ``/BaseFont`` is greppable even though the content
    streams are compressed.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    faces = sorted({m.decode("latin-1", "replace")
                    for m in BASEFONT_RE.findall(raw)})
    return faces, len(FONTFILE_RE.findall(raw))


def check_pdfs(rep, paths):
    for path in paths:
        rel = os.path.relpath(path, ROOT)
        try:
            faces, embedded = inspect_pdf(path)
        except OSError as exc:
            rep.error("pdf-fonts", f"{rel}: cannot read ({exc})")
            continue
        if not faces:
            rep.warn("pdf-fonts", f"{rel}: no /BaseFont found (no text?)")
            continue
        for face in faces:
            verdict = classify_face(face)
            if verdict == "forbidden":
                rep.error(
                    "pdf-fonts",
                    f"{rel}: embedded {face!r} -- the serif stack did not "
                    f"resolve, so this figure does not match the gallery")
            elif verdict == "unknown":
                rep.warn("pdf-fonts",
                         f"{rel}: embedded unrecognised face {face!r}")
        if embedded == 0:
            rep.error("pdf-fonts",
                      f"{rel}: fonts are referenced but not embedded; set "
                      f"pdf.fonttype=42 (a journal's prepress will substitute "
                      f"them otherwise)")


# ---- Check 3: render a sample and watch for findfont warnings --------------
def probe_scripts(rep, scripts):
    """Run scripts with PDF output on, then inspect what they embedded."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8",
               PUBSTYLE_FORMATS="pdf")
    produced = []
    for script in scripts:
        base = os.path.basename(script)
        try:
            p = subprocess.run([sys.executable, base], cwd=HERE,
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=600, env=env)
        except subprocess.TimeoutExpired:
            rep.warn("render", f"{base}: timed out, skipped")
            continue
        if p.returncode != 0:
            rep.error("render", f"{base}: exited {p.returncode}")
            continue
        stderr = p.stderr or ""
        for line in stderr.splitlines():
            if any(n in line for n in NOISE):
                continue
            if "findfont" in line:
                rep.error("findfont",
                          f"{base}: {line.strip()[:160]}")
        for line in (p.stdout or "").splitlines():
            if line.startswith("saved:") and line.strip().endswith(".pdf"):
                produced.append(line.split("saved:", 1)[1].strip())
    check_pdfs(rep, produced)
    rep.note(f"render probe             : {len(scripts)} script(s), "
             f"{len(produced)} PDF(s) inspected")


# ---- Check 4: source lint --------------------------------------------------
def looks_title_case(text):
    """True when a title capitalises words it should not.

    Sentence case is the house rule, so only the first word and proper nouns
    are capitalised. Words in ALL CAPS (acronyms), words with internal capitals
    (``ResNet``) and words with digits are ignored, because those are legitimate
    regardless of position.
    """
    words = [w.strip("(),:;-") for w in text.split()]
    words = [w for w in words[1:] if len(w) > 2]
    suspicious = 0
    for w in words:
        if not w[:1].isupper():
            continue
        if w.isupper() or any(c.isdigit() for c in w):
            continue
        if any(c.isupper() for c in w[1:]):
            continue
        suspicious += 1
    return suspicious >= 2


def lint_source(rep, scripts):
    import pubstyle as ps

    lo, hi = ps.TYPE_MIN, ps.TYPE_MAX
    for script in scripts:
        base = os.path.basename(script)
        try:
            with open(script, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue

        for m in FONTSIZE_RE.finditer(src):
            val = float(m.group(1))
            line = src.count("\n", 0, m.start()) + 1
            if val < lo:
                rep.error("type-scale",
                          f"{base}:{line}: fontsize {val} is below the {lo} pt "
                          f"floor (illegible at 300 dpi)")
            elif val > hi:
                rep.warn("type-scale",
                         f"{base}:{line}: fontsize {val} exceeds the {hi} pt "
                         f"ceiling")

        for m in RCPARAMS_RE.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            rep.error("rcparams",
                      f"{base}:{line}: writes rcParams directly; style must "
                      f"come from set_pub_style()")

        for rx, kind in ((SET_TITLE_RE, "panel title"),
                         (HEADER_TITLE_RE, "figure title")):
            for m in rx.finditer(src):
                text = m.group(2).strip()
                if not text or "{" in text or "\\n" in text:
                    continue
                line = src.count("\n", 0, m.start()) + 1
                if text.endswith("."):
                    rep.warn("title-case",
                             f"{base}:{line}: {kind} ends with a period: "
                             f"{text!r}")
                if looks_title_case(text):
                    rep.warn("title-case",
                             f"{base}:{line}: {kind} looks Title Case, house "
                             f"rule is Sentence case: {text!r}")

        for m in PANEL_LABEL_RE.finditer(src):
            text = m.group(2).strip()
            line = src.count("\n", 0, m.start()) + 1
            # Templated labels -- f-string indexing ``({'abc'[i]})`` or
            # percent formatting ``(%s)`` -- resolve to the right thing at
            # runtime, so only the surrounding parentheses can be checked.
            if "{" in text or "%" in text:
                if not (text.startswith("(") and text.endswith(")")):
                    rep.warn("panel-label",
                             f"{base}:{line}: panel label {text!r} is not "
                             f"parenthesised")
                continue
            if not re.fullmatch(r"\([a-z]\)", text):
                rep.warn("panel-label",
                         f"{base}:{line}: panel label {text!r}; house form is "
                         f"lowercase '(a)'")


# ---- Repair: raise sub-floor type sizes ------------------------------------
def fix_type_floor(scripts, floor, dry_run=False):
    """Raise every ``fontsize=`` literal below ``floor`` up to it.

    This is the one lint failure that is safe to repair mechanically: the
    value is a numeric keyword literal, and the design system has a single
    correct answer for anything under the floor. Everything else this module
    reports (Title Case, a title that names the chart type) needs a human to
    write new prose, so it stays a warning.

    Returns ``[(path, line, old, new), ...]``.
    """
    changes = []
    for script in scripts:
        try:
            with open(script, encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            continue

        hits = []

        def repl(m):
            val = float(m.group(1))
            if val >= floor:
                return m.group(0)
            hits.append((src.count("\n", 0, m.start()) + 1, val))
            return m.group(0).replace(m.group(1), f"{floor:g}")

        new_src = FONTSIZE_RE.sub(repl, src)
        if not hits:
            continue
        for line, old in hits:
            changes.append((os.path.basename(script), line, old, floor))
        if not dry_run:
            with open(script, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(new_src)
    return changes


# ---- Check 5: is the bundled clone really metric-compatible? ---------------
SAMPLE = ("Path-length optimality across all successful runs 0123456789 "
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz .,;:()[]%-+")


def advance_widths(family, text=SAMPLE):
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font
    path = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    face = FT2Font(path)
    face.set_size(1000, 72)
    widths = []
    for ch in text:
        glyph = face.load_char(ord(ch))
        widths.append(round(glyph.linearHoriAdvance / 65536.0, 3))
    return path, widths


def check_metrics(rep):
    import pubstyle as ps

    ps._register_bundled_fonts()
    available = ps.available_fonts()
    present = [n for n in ps.SERIF_STACK if n in available]
    if len(present) < 2:
        rep.warn("metrics",
                 f"only {present} available; need two faces to compare")
        return

    reference = present[0]
    try:
        ref_path, ref_w = advance_widths(reference)
    except Exception as exc:
        rep.warn("metrics", f"cannot measure {reference}: {exc}")
        return
    rep.note(f"\nmetric reference         : {reference} "
             f"({os.path.basename(ref_path)})")

    for name in present[1:]:
        try:
            path, widths = advance_widths(name)
        except Exception as exc:
            rep.warn("metrics", f"cannot measure {name}: {exc}")
            continue
        delta = max(abs(a - b) for a, b in zip(ref_w, widths))
        verdict = "identical" if delta == 0 else f"max delta {delta:g}/1000em"
        if name in METRIC_EXCEPTIONS:
            verdict += "  (known exception)"
        rep.note(f"  vs {name:<18} {verdict:<40} "
                 f"({os.path.basename(path)})")
        if delta > 5 and name not in METRIC_EXCEPTIONS:
            rep.warn("metrics",
                     f"{name} differs from {reference} by {delta:g}/1000em; "
                     f"labels will reflow when it is used as a fallback")


# ---- Driver ----------------------------------------------------------------
def discover(only=None):
    names = sorted(f for f in os.listdir(HERE)
                   if f.startswith("fig_") and f.endswith(".py"))
    if only:
        names = [f for f in names if only.lower() in f.lower()]
    return [os.path.join(HERE, f) for f in names]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Audit fonts and title typography across the library.")
    p.add_argument("--only", metavar="SUBSTR",
                   help="substring filter on the script filename")
    p.add_argument("--render", type=int, default=12, metavar="N",
                   help="render N sampled scripts to PDF and inspect the "
                        "embedded fonts (default 12, 0 to skip)")
    p.add_argument("--source-only", action="store_true",
                   help="skip rendering; lint sources only")
    p.add_argument("--metrics", action="store_true",
                   help="only prove the fallback faces are metric-compatible")
    p.add_argument("--pdf", nargs="+", metavar="FILE",
                   help="inspect these existing PDFs and exit")
    p.add_argument("--seed", type=int, default=0,
                   help="sampling seed for --render")
    p.add_argument("--strict", action="store_true",
                   help="treat warnings as failures")
    p.add_argument("--fix", action="store_true",
                   help="raise fontsize literals below the floor up to it, "
                        "then re-lint")
    p.add_argument("--dry-run", action="store_true",
                   help="with --fix, report the edits without writing")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    rep = Report()

    if a.pdf:
        check_pdfs(rep, [os.path.abspath(p) for p in a.pdf])
        return rep.dump(a.strict)

    if a.metrics:
        check_metrics(rep)
        return rep.dump(a.strict)

    check_environment(rep)
    check_metrics(rep)

    scripts = discover(a.only)
    if not scripts:
        rep.error("discover", "no fig_*.py matched the selection")
        return rep.dump(a.strict)

    if a.fix:
        import pubstyle as ps
        changes = fix_type_floor(scripts, ps.TYPE_MIN, dry_run=a.dry_run)
        verb = "would raise" if a.dry_run else "raised"
        rep.note(f"\n--fix                    : {verb} {len(changes)} "
                 f"sub-floor fontsize literal(s) to {ps.TYPE_MIN:g} pt")
        for base, line, old, new in changes[:20]:
            rep.note(f"    {base}:{line}  {old:g} -> {new:g}")
        if len(changes) > 20:
            rep.note(f"    ... and {len(changes) - 20} more")
        if changes and not a.dry_run:
            rep.note("    re-run run_all.py so the PNGs match the sources")

    rep.note(f"\nlinting                  : {len(scripts)} script(s)")
    lint_source(rep, scripts)

    if not a.source_only and a.render > 0:
        sample = scripts if len(scripts) <= a.render else \
            random.Random(a.seed).sample(scripts, a.render)
        probe_scripts(rep, sorted(sample))

    return rep.dump(a.strict)


if __name__ == "__main__":
    sys.exit(main())
