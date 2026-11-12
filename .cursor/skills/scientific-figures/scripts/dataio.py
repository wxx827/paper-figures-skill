"""Load a client's data file without making them clean it first.

The figure library is worth nothing to someone whose CSV will not even open.
Real files arrive with a UTF-8 BOM that turns the first column name into
``\ufeffdate``, or in GBK because they were exported from Excel on a Chinese
Windows box, or semicolon-separated because the exporting locale uses a comma
as the decimal mark. This module absorbs that so the plotting code can assume
a clean DataFrame.

    from dataio import load_table, profile

    df = load_table("sales.csv")       # csv, tsv, xlsx, xls, parquet, json
    print(profile(df))                  # column roles, missingness, cardinality

Only pandas is required. Excel needs ``openpyxl``, Parquet needs ``pyarrow``;
both fail with a message naming the package to install rather than a traceback.
"""
from __future__ import annotations

import csv
import io
import os
import unicodedata

# Order matters. utf-8-sig first so a BOM is consumed rather than becoming part
# of the first column name; gb18030 before gbk because it is a strict superset;
# latin-1 last because it never fails and would mask everything after it.
ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "big5", "shift_jis",
             "cp1252", "latin-1"]

DELIMITERS = [",", ";", "\t", "|"]

MISSING_TOKENS = ["", "NA", "N/A", "n/a", "na", "null", "NULL", "None",
                  "-", "--", "?", "#N/A", "#DIV/0!", "nan", "NaN",
                  "无", "缺失", "未知"]

EXCEL_EXT = {".xlsx", ".xlsm", ".xls", ".ods"}
PARQUET_EXT = {".parquet", ".pq"}
JSON_EXT = {".json", ".jsonl", ".ndjson"}


class DataError(Exception):
    """A problem the user can act on, reported without a traceback."""


def _require(package, why):
    try:
        __import__(package)
    except ImportError:
        raise DataError(f"{why} needs the {package!r} package: "
                        f"pip install {package}")


def sniff_encoding(path, sample_bytes=262144):
    """First encoding in :data:`ENCODINGS` that decodes the head of the file.

    Deliberately not chardet: a statistical guess that is *probably* right is
    worse here than a deterministic ladder, because a wrong guess corrupts
    column names silently instead of failing.
    """
    with open(path, "rb") as fh:
        raw = fh.read(sample_bytes)
    for enc in ENCODINGS:
        try:
            raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        return enc
    return "latin-1"


def sniff_delimiter(path, encoding, sample_bytes=65536):
    """Delimiter of a text table, by csv.Sniffer with a column-count fallback."""
    with open(path, "r", encoding=encoding, errors="replace") as fh:
        sample = fh.read(sample_bytes)
    if not sample.strip():
        raise DataError(f"{os.path.basename(path)} is empty")
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        pass
    # Fallback: the delimiter that splits the first few lines most consistently
    # into more than one field.
    lines = [ln for ln in sample.splitlines()[:20] if ln.strip()]
    best, best_score = ",", 0
    for d in DELIMITERS:
        counts = [ln.count(d) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        # Consistent count across rows is the signal; ties go to more columns.
        score = max(counts) if len(set(counts)) == 1 else 1
        if score > best_score:
            best, best_score = d, score
    return best


def load_table(path, sheet=None, encoding=None, delimiter=None, nrows=None):
    """Read any of the common table formats into a DataFrame.

    Parameters
    ----------
    path : str
    sheet : str or int, optional
        Excel worksheet; defaults to the first.
    encoding, delimiter : str, optional
        Override the sniffers.
    nrows : int, optional
        Read only the first N rows, for a quick look at a huge file.
    """
    _require("pandas", "reading a data file")
    import pandas as pd

    if not os.path.exists(path):
        raise DataError(f"no such file: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext in EXCEL_EXT:
        _require("openpyxl", "reading Excel")
        df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet,
                           nrows=nrows, na_values=MISSING_TOKENS)
    elif ext in PARQUET_EXT:
        _require("pyarrow", "reading Parquet")
        df = pd.read_parquet(path)
        if nrows:
            df = df.head(nrows)
    elif ext in JSON_EXT:
        enc = encoding or sniff_encoding(path)
        df = pd.read_json(path, encoding=enc,
                          lines=ext in (".jsonl", ".ndjson"))
        if nrows:
            df = df.head(nrows)
    else:
        enc = encoding or sniff_encoding(path)
        sep = delimiter or sniff_delimiter(path, enc)
        df = pd.read_csv(path, encoding=enc, sep=sep, nrows=nrows,
                         na_values=MISSING_TOKENS, skip_blank_lines=True,
                         engine="python")

    if df.empty:
        raise DataError(f"{os.path.basename(path)} parsed to zero rows")

    df.columns = [str(c).strip() for c in df.columns]
    df = _drop_unnamed(df)
    df = _coerce(df)
    return df


def _drop_unnamed(df):
    """Drop the index column pandas invents when Excel exported one."""
    drop = [c for c in df.columns
            if c.lower().startswith("unnamed:") and df[c].isna().all()]
    return df.drop(columns=drop) if drop else df


# Characters Excel sprinkles through numeric columns. Stripped one at a time
# with regex=False: pandas may be backed by Arrow, whose regex engine rejects
# the \uXXXX escapes a character class would otherwise need.
NUMERIC_NOISE = ",，$¥€£%＄％ \u00a0\u3000"


def is_text_dtype(s):
    """True for a text column under any pandas string backend.

    ``s.dtype == object`` used to be enough. Modern pandas may hand back an
    Arrow-backed ``str`` dtype instead, and testing for ``object`` silently
    classifies every text column as "other" -- which is how a table full of
    perfectly good categories ends up supporting no categorical chart at all.
    """
    import pandas as pd
    return (pd.api.types.is_string_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


def _coerce(df):
    """Turn numeric-looking and date-looking text columns into real dtypes.

    Client exports routinely carry numbers as text with thousands separators,
    percent signs or currency marks. Leaving them as strings makes every
    numeric chart silently unavailable.
    """
    import pandas as pd

    for col in df.columns:
        s = df[col]
        if not is_text_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        present = int(s.notna().sum())
        if present == 0:
            continue
        need = max(1, int(0.8 * present))

        cleaned = s.astype("string").str.strip()
        for ch in NUMERIC_NOISE:
            cleaned = cleaned.str.replace(ch, "", regex=False)
        num = pd.to_numeric(cleaned, errors="coerce")
        if int(num.notna().sum()) >= need:
            df[col] = num
            continue

        dt = _to_datetime(s)
        if dt is not None and int(dt.notna().sum()) >= need:
            df[col] = dt
    return df


def _to_datetime(s):
    """Parse to datetime, tolerating pandas versions and unparseable columns."""
    import pandas as pd
    for kwargs in ({"format": "mixed"}, {}):
        try:
            return pd.to_datetime(s, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
    return None


# ---- Column roles ----------------------------------------------------------
def column_roles(df, max_categories=30):
    """``{"numeric": [...], "categorical": [...], "datetime": [...]}``

    A numeric column with very few distinct values (a 1/0 flag, a Likert score)
    is reported in both ``numeric`` and ``categorical``, because either reading
    is legitimate and the caller knows which chart it is drawing.
    """
    import pandas as pd

    roles = {"numeric": [], "categorical": [], "datetime": [], "other": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            roles["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            roles["numeric"].append(col)
            if s.nunique(dropna=True) <= min(12, max_categories):
                roles["categorical"].append(col)
        elif pd.api.types.is_bool_dtype(s) or is_text_dtype(s):
            if s.nunique(dropna=True) <= max_categories:
                roles["categorical"].append(col)
            else:
                roles["other"].append(col)
        else:
            roles["other"].append(col)
    return roles


def display_width(text):
    """Terminal columns a string occupies, counting CJK glyphs as two.

    Chinese column headers are the normal case for this library's users, and
    padding them with len() leaves the table visibly ragged.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1
               for ch in str(text))


def _pad(text, width, align="<"):
    text = str(text)
    while display_width(text) > width:
        text = text[:-1]
    fill = " " * max(0, width - display_width(text))
    return text + fill if align == "<" else fill + text


def profile(df, max_categories=30):
    """Human-readable summary: shape, dtypes, missingness, cardinality."""
    roles = column_roles(df, max_categories)
    out = io.StringIO()
    n = len(df)
    out.write(f"{n} row(s) x {len(df.columns)} column(s)\n\n")
    widths = (max(6, min(26, max(display_width(c) for c in df.columns))),
              12, 9, 9)
    head = (f"{_pad('column', widths[0])} {_pad('role', widths[1])} "
            f"{_pad('missing', widths[2], '>')} "
            f"{_pad('distinct', widths[3], '>')}  example")
    out.write(head + "\n" + "-" * display_width(head) + "\n")
    for col in df.columns:
        s = df[col]
        if col in roles["datetime"]:
            role = "datetime"
        elif col in roles["numeric"]:
            role = "numeric"
        elif col in roles["categorical"]:
            role = "categorical"
        else:
            role = "text"
        miss = int(s.isna().sum())
        pct = f"{miss}({100 * miss / n:.0f}%)" if miss else "-"
        sample = s.dropna()
        ex = "" if sample.empty else str(sample.iloc[0])[:28]
        out.write(f"{_pad(col, widths[0])} {_pad(role, widths[1])} "
                  f"{_pad(pct, widths[2], '>')} "
                  f"{_pad(s.nunique(dropna=True), widths[3], '>')}  {ex}\n")
    return out.getvalue()


def pick(df, wanted, role, roles=None, exclude=()):
    """Resolve a user-supplied column name, or choose a sensible default.

    Raises :class:`DataError` naming the available columns, because "KeyError:
    'revenue'" is not an error message a client can act on.
    """
    roles = roles or column_roles(df)
    if wanted:
        if wanted in df.columns:
            return wanted
        lowered = {str(c).lower(): c for c in df.columns}
        if wanted.lower() in lowered:
            return lowered[wanted.lower()]
        raise DataError(
            f"no column named {wanted!r}. Available: "
            + ", ".join(str(c) for c in df.columns[:25])
            + (" ..." if len(df.columns) > 25 else ""))
    for col in roles.get(role, []):
        if col not in exclude:
            return col
    return None
