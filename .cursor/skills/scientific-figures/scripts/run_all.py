"""Run every figure generator in this folder.

Scripts are discovered from disk, so figures added later are picked up
automatically -- there is no hard-coded list.

    cd scripts
    python run_all.py                     # all of them, one at a time
    python run_all.py --parallel 8        # 8 subprocesses at a time
    python run_all.py --range 100-200     # only figures numbered 100-200
    python run_all.py --range 173-        # 173 to the end
    python run_all.py --only rs_          # substring filter on the filename
    python run_all.py --list              # show what would run, run nothing

Each script runs in its own subprocess with ``MPLBACKEND=Agg``. That keeps
matplotlib's global state from being shared across workers (it is not
thread-safe) and stops one crashing figure from taking the whole batch down.

For a full audit of the PNGs afterwards -- byte size, pixel size, embedded DPI,
warnings -- use ``check_figures.py`` instead; this tool only regenerates.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTPUT_DIR = os.path.join(ROOT, "output")

TIMEOUT = 600

# ``savefig(fig, "40_lollipop.png")`` -> 40
SAVEFIG_RE = re.compile(
    r"""savefig\(\s*[^,]+,\s*f?["'](\d+)_[^"'{}]*\.(?:png|pdf|svg)["']""")
# ``fig_00_style_showcase.py`` -> 0
STEM_NUM_RE = re.compile(r"^fig_(\d+)_")
# Noise from the Anaconda shell profile, not from the figure code.
NOISE = ("RequestsDependencyWarning", "urllib3 (", "warnings.warn(")


def discover(scripts_dir=HERE):
    names = sorted(f for f in os.listdir(scripts_dir)
                   if f.startswith("fig_") and f.endswith(".py"))
    return [os.path.join(scripts_dir, f) for f in names]


def _png_numbers_by_stem(output_dir=OUTPUT_DIR):
    """``{"lollipop": 40, ...}`` from the existing output/ filenames."""
    table = {}
    if not os.path.isdir(output_dir):
        return table
    for f in os.listdir(output_dir):
        m = re.match(r"^(\d+)_(.+)\.png$", f)
        if m:
            table.setdefault(m.group(2), int(m.group(1)))
    return table


def script_number(script, png_table):
    """The figure number a script owns, or None when it cannot be determined.

    Preferred source is the script's own ``savefig()`` call, so the number is
    right even before the PNG exists.
    """
    base = os.path.basename(script)
    m = STEM_NUM_RE.match(base)
    if m:
        return int(m.group(1))
    try:
        with open(script, encoding="utf-8") as fh:
            m = SAVEFIG_RE.search(fh.read())
        if m:
            return int(m.group(1))
    except OSError:
        pass
    return png_table.get(base[4:-3])


def parse_range(spec):
    """``"100-200"``, ``"173-"``, ``"-54"`` or ``"42"`` -> (lo, hi)."""
    spec = spec.strip()
    m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", spec)
    if m and (m.group(1) or m.group(2)):
        lo = int(m.group(1)) if m.group(1) else 0
        hi = int(m.group(2)) if m.group(2) else 10 ** 9
    elif spec.isdigit():
        lo = hi = int(spec)
    else:
        raise argparse.ArgumentTypeError(
            f"bad --range {spec!r}; use 100-200, 173-, -54 or 42")
    if lo > hi:
        raise argparse.ArgumentTypeError(f"bad --range {spec!r}; lo > hi")
    return lo, hi


def select(scripts, only=None, rng=None):
    """Filter the discovered scripts, and report anything --range skipped."""
    if only:
        scripts = [s for s in scripts if only.lower() in
                   os.path.basename(s).lower()]
    unnumbered = []
    if rng:
        lo, hi = rng
        png_table = _png_numbers_by_stem()
        keep = []
        for s in scripts:
            n = script_number(s, png_table)
            if n is None:
                unnumbered.append(os.path.basename(s))
            elif lo <= n <= hi:
                keep.append((n, s))
        scripts = [s for _, s in sorted(keep)]
    return scripts, unnumbered


def run_one(script, formats=None):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    if formats:
        # pubstyle.savefig reads this, so a batch can emit vector copies
        # without editing 1200 call sites.
        env["PUBSTYLE_FORMATS"] = ",".join(formats)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, os.path.basename(script)],
                           cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=TIMEOUT, env=env)
        code, err = p.returncode, p.stderr or ""
    except subprocess.TimeoutExpired:
        code, err = -9, f"TIMEOUT after {TIMEOUT}s"
    tail = [l for l in err.strip().splitlines()
            if l.strip() and not any(n in l for n in NOISE)]
    return {"script": os.path.basename(script), "code": code,
            "secs": time.perf_counter() - t0, "tail": "\n".join(tail[-6:])}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Regenerate every fig_*.py in this folder.")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="run N scripts concurrently (default 1 = sequential)")
    p.add_argument("--range", type=parse_range, metavar="LO-HI",
                   help="only figures whose number falls in this range, "
                        "e.g. 100-200, 173-, -54")
    p.add_argument("--only", metavar="SUBSTR",
                   help="substring filter on the script filename")
    p.add_argument("--list", action="store_true",
                   help="print the selection and exit without running")
    p.add_argument("--vector", action="store_true",
                   help="also emit PDF next to every PNG (journal submission)")
    p.add_argument("--formats", metavar="LIST",
                   help="explicit extra formats, e.g. pdf,svg "
                        "(overrides --vector)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.parallel < 1:
        sys.exit("--parallel must be >= 1")

    scripts, unnumbered = select(discover(), only=a.only, rng=a.range)
    if unnumbered:
        print(f"note: {len(unnumbered)} script(s) have no discoverable figure "
              f"number and were skipped by --range: "
              f"{', '.join(unnumbered[:5])}"
              f"{' ...' if len(unnumbered) > 5 else ''}\n")
    if not scripts:
        sys.exit("no fig_*.py matched the selection")

    if a.list:
        for s in scripts:
            print(os.path.basename(s))
        print(f"\n{len(scripts)} script(s) selected")
        return 0

    if a.formats:
        formats = [f.strip().lstrip(".").lower()
                   for f in a.formats.split(",") if f.strip()]
    elif a.vector:
        formats = ["pdf"]
    else:
        formats = []

    mode = "sequential" if a.parallel == 1 else f"parallel={a.parallel}"
    extra = f" | extra formats: {', '.join(formats)}" if formats else ""
    print(f"running {len(scripts)} script(s) ({mode}){extra}\n")

    t0 = time.perf_counter()
    results = []
    if a.parallel == 1:
        for i, s in enumerate(scripts, 1):
            print(f"[{i}/{len(scripts)}] {os.path.basename(s)}", flush=True)
            results.append(run_one(s, formats))
    else:
        with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
            futures = {ex.submit(run_one, s, formats): s for s in scripts}
            for i, fut in enumerate(cf.as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                flag = "ok  " if r["code"] == 0 else f"FAIL({r['code']})"
                print(f"[{i}/{len(scripts)}] {flag} {r['script']} "
                      f"({r['secs']:.1f}s)", flush=True)
    elapsed = time.perf_counter() - t0

    results.sort(key=lambda r: r["script"])
    failed = [r for r in results if r["code"] != 0]
    if failed:
        print("\n" + "=" * 60)
        print(f"{len(failed)} failure(s):")
        for r in failed:
            print(f"\n--- {r['script']} (exit {r['code']}) ---")
            print(r["tail"] or "(no stderr captured)")

    print(f"\n{len(results)} script(s) | {len(failed)} failed | "
          f"{elapsed:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
