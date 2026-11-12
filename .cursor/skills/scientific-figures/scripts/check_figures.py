"""Batch health check for every ``fig_*.py`` generator.

Runs each script in a subprocess, records exit code / runtime / warnings, then
inspects the PNG it was supposed to produce (existence, byte size, pixel size,
embedded DPI).  Prints a summary table and writes ``figure_health.md``.

    cd scripts
    python check_figures.py                 # full run (slow, a few minutes)
    python check_figures.py --quick         # inspect existing PNGs only
    python check_figures.py --only heatmap  # substring filter
    python check_figures.py --jobs 8        # more parallelism

Scripts are discovered from disk, so figures added later are covered
automatically.  While other agents are editing the library a few scripts may
fail transiently -- the report records what happened, it never edits them.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTPUT_DIR = os.path.join(ROOT, "output")
REPORT = os.path.join(ROOT, "figure_health.md")

MIN_KB = 20.0
TIMEOUT = 300

# ``savefig(fig, "40_lollipop.png")`` and friends.
SAVEFIG_RE = re.compile(
    r"""savefig\(\s*[^,]+,\s*f?["']([^"'{}]+\.(?:png|pdf|svg))["']""")
WARN_RE = re.compile(r"(?i)\b(\w*Warning)\b\s*:?")
# Noise from the Anaconda shell profile, not from the figure code.
NOISE = ("RequestsDependencyWarning", "urllib3 (", "warnings.warn(")


def discover(scripts_dir=HERE, only=None):
    names = sorted(f for f in os.listdir(scripts_dir)
                   if f.startswith("fig_") and f.endswith(".py"))
    if only:
        names = [f for f in names if only.lower() in f.lower()]
    return [os.path.join(scripts_dir, f) for f in names]


def expected_pngs(script):
    """PNG filenames the script claims to write, from its savefig() calls."""
    try:
        with open(script, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return []
    names = [n for n in SAVEFIG_RE.findall(src) if n.lower().endswith(".png")]
    if names:
        return sorted(dict.fromkeys(names))
    # Fallback: match output/<num>_<stem>.png by the script stem.
    stem = os.path.basename(script)[4:-3]
    if os.path.isdir(OUTPUT_DIR):
        return sorted(f for f in os.listdir(OUTPUT_DIR)
                      if f.lower().endswith(".png")
                      and re.sub(r"^\d+_", "", f[:-4]) == stem)
    return []


def collect_warnings(stderr):
    out = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line or any(n in line for n in NOISE):
            continue
        if WARN_RE.search(line):
            out.append(line if len(line) <= 240 else line[:237] + "...")
    return sorted(dict.fromkeys(out))


def inspect_png(name):
    """Return a dict of on-disk facts about output/<name>."""
    path = os.path.join(OUTPUT_DIR, name)
    info = {"name": name, "exists": os.path.exists(path), "kb": 0.0,
            "px": None, "dpi": None, "notes": []}
    if not info["exists"]:
        info["notes"].append("missing")
        return info
    info["kb"] = os.path.getsize(path) / 1024.0
    if info["kb"] < MIN_KB:
        info["notes"].append(f"only {info['kb']:.1f} KB (<{MIN_KB:.0f})")
    try:
        from PIL import Image
        with Image.open(path) as im:
            info["px"] = im.size
            dpi = im.info.get("dpi")
        if dpi:
            info["dpi"] = round(float(dpi[0]))
            if info["dpi"] < 290:
                info["notes"].append(f"{info['dpi']} dpi (<300)")
        else:
            info["notes"].append("no DPI metadata")
    except ImportError:
        info["notes"].append("Pillow missing, pixel check skipped")
    except Exception as exc:
        info["notes"].append(f"unreadable: {exc}")
    return info


def run_script(script):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, os.path.basename(script)],
                           cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=TIMEOUT, env=env)
        code, err = p.returncode, p.stderr or ""
    except subprocess.TimeoutExpired:
        code, err = -9, f"TIMEOUT after {TIMEOUT}s"
    return code, time.perf_counter() - t0, err


def check(script, quick=False):
    name = os.path.basename(script)
    row = {"script": name, "code": None, "secs": 0.0, "warnings": [],
           "error_tail": "", "pngs": []}
    if not quick:
        row["code"], row["secs"], err = run_script(script)
        row["warnings"] = collect_warnings(err)
        if row["code"] != 0:
            tail = [l for l in err.strip().splitlines()
                    if l.strip() and not any(n in l for n in NOISE)]
            row["error_tail"] = "\n".join(tail[-6:])
    row["pngs"] = [inspect_png(n) for n in expected_pngs(script)]
    if not row["pngs"]:
        row["pngs"] = [{"name": "(no savefig found)", "exists": False,
                        "kb": 0.0, "px": None, "dpi": None,
                        "notes": ["could not determine output filename"]}]
    row["ok"] = (row["code"] in (0, None)
                 and all(p["exists"] and not p["notes"] for p in row["pngs"]))
    return row


# ---- Reporting -------------------------------------------------------------
def fmt_px(px):
    return f"{px[0]}x{px[1]}" if px else "-"


def print_table(rows, quick):
    w = max((len(r["script"]) for r in rows), default=10)
    head = (f"{'script':<{w}}  {'exit':>4}  {'secs':>6}  {'png':<34}"
            f"  {'KB':>7}  {'pixels':>11}  {'dpi':>4}  notes")
    print(head)
    print("-" * len(head))
    for r in rows:
        code = "-" if r["code"] is None else str(r["code"])
        secs = "-" if quick else f"{r['secs']:.1f}"
        for i, p in enumerate(r["pngs"]):
            notes = "; ".join(p["notes"])
            if i == 0 and r["warnings"]:
                notes = (notes + "; " if notes else "") + \
                        f"{len(r['warnings'])} warning(s)"
            print(f"{r['script'] if i == 0 else '':<{w}}  "
                  f"{code if i == 0 else '':>4}  {secs if i == 0 else '':>6}  "
                  f"{p['name']:<34}  {p['kb']:>7.1f}  {fmt_px(p['px']):>11}  "
                  f"{str(p['dpi'] or '-'):>4}  {notes}")


def write_report(rows, quick, elapsed, path=REPORT):
    failed = [r for r in rows if r["code"] not in (0, None)]
    bad_png = [(r, p) for r in rows for p in r["pngs"]
               if p["notes"] or not p["exists"]]
    warned = [r for r in rows if r["warnings"]]
    total_png = sum(len(r["pngs"]) for r in rows)

    L = []
    a = L.append
    a("# Figure health report")
    a("")
    a(f"- Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    a(f"- Mode: {'quick (no scripts executed)' if quick else 'full run'}")
    a(f"- Scripts checked: **{len(rows)}**")
    if not quick:
        a(f"- Failed: **{len(failed)}**  ·  with warnings: **{len(warned)}**")
        a(f"- Wall time: {elapsed:.1f} s")
    a(f"- Expected PNGs: **{total_png}**  ·  with problems: **{len(bad_png)}**")
    a("")
    a("Other agents may be editing scripts while this runs; a transient "
      "failure here is not necessarily a real defect. This tool never "
      "modifies the scripts it checks.")
    a("")

    a("## Failures")
    a("")
    if quick:
        a("_Skipped in `--quick` mode._")
    elif not failed:
        a("None - every script exited 0.")
    else:
        for r in failed:
            a(f"### `{r['script']}` (exit {r['code']}, {r['secs']:.1f}s)")
            a("")
            a("```")
            a(r["error_tail"] or "(no stderr captured)")
            a("```")
            a("")
    a("")

    a("## PNG problems")
    a("")
    if not bad_png:
        a("None - every expected PNG exists, is >20 KB and carries 300 dpi.")
    else:
        a("| script | png | issue | KB | pixels | dpi |")
        a("|---|---|---|---:|---|---:|")
        for r, p in bad_png:
            a(f"| `{r['script']}` | `{p['name']}` | {'; '.join(p['notes'])} "
              f"| {p['kb']:.1f} | {fmt_px(p['px'])} | {p['dpi'] or '-'} |")
    a("")

    a("## Warnings")
    a("")
    if quick:
        a("_Skipped in `--quick` mode._")
    elif not warned:
        a("None.")
    else:
        for r in warned:
            a(f"### `{r['script']}`")
            a("")
            for line in r["warnings"]:
                a(f"- `{line}`")
            a("")
    a("")

    a("## Runtime ranking")
    a("")
    if quick:
        a("_Skipped in `--quick` mode._")
    else:
        a("| # | script | seconds | exit |")
        a("|---:|---|---:|---:|")
        for i, r in enumerate(sorted(rows, key=lambda x: -x["secs"]), 1):
            a(f"| {i} | `{r['script']}` | {r['secs']:.1f} | {r['code']} |")
    a("")

    a("## All figures")
    a("")
    a("| script | exit | secs | png | KB | pixels | dpi | notes |")
    a("|---|---:|---:|---|---:|---|---:|---|")
    for r in rows:
        code = "-" if r["code"] is None else str(r["code"])
        secs = "-" if quick else f"{r['secs']:.1f}"
        for i, p in enumerate(r["pngs"]):
            first = i == 0
            notes = "; ".join(p["notes"]) or "ok"
            a(f"| {'`' + r['script'] + '`' if first else ''} "
              f"| {code if first else ''} | {secs if first else ''} "
              f"| `{p['name']}` | {p['kb']:.1f} | {fmt_px(p['px'])} "
              f"| {p['dpi'] or '-'} | {notes} |")
    a("")

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print(f"\nreport written: {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run every fig_*.py and audit the PNGs it produces.")
    p.add_argument("--quick", action="store_true",
                   help="skip execution; only inspect existing PNGs")
    p.add_argument("--only", help="substring filter on the script filename")
    p.add_argument("--jobs", type=int, default=4,
                   help="parallel subprocesses (default 4)")
    p.add_argument("--typography", action="store_true",
                   help="also audit fonts and titles (check_typography.py); "
                        "a PNG carries no font metadata, so the checks here "
                        "cannot see a wrong typeface")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    scripts = discover(only=a.only)
    if not scripts:
        sys.exit("no fig_*.py found")
    print(f"checking {len(scripts)} scripts "
          f"({'quick' if a.quick else f'full, jobs={a.jobs}'})\n")

    t0 = time.perf_counter()
    if a.quick or a.jobs <= 1:
        rows = [check(s, a.quick) for s in scripts]
    else:
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            rows = list(ex.map(lambda s: check(s, a.quick), scripts))
    elapsed = time.perf_counter() - t0

    rows.sort(key=lambda r: r["script"])
    print_table(rows, a.quick)

    failed = [r for r in rows if r["code"] not in (0, None)]
    bad = [p for r in rows for p in r["pngs"] if p["notes"]]
    print(f"\n{len(rows)} scripts | {len(failed)} failed | "
          f"{len(bad)} PNG problem(s) | {elapsed:.1f}s")
    write_report(rows, a.quick, elapsed)

    if a.typography:
        print(f"\n{'=' * 62}\ntypography audit\n{'=' * 62}")
        import check_typography
        argv2 = ["--source-only"] if a.quick else []
        if a.only:
            argv2 += ["--only", a.only]
        return check_typography.main(argv2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
