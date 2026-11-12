"""Generate the sample data files used to demonstrate ``plot_data.py``.

Three files, chosen to exercise the parts of the loader that break on real
client data rather than on tidy textbook data:

  sales.csv          UTF-8, mixed types, a date column, missing values, and
                     numbers formatted the way Excel exports them ("1,234",
                     "12.5%") so the coercion path is tested.
  clinical_gbk.csv   GBK-encoded with Chinese column names, which is what an
                     export from a Chinese Windows machine actually looks like.
  trial_effects.csv  estimate / interval / label columns, for the forest plot.

    cd examples
    python make_examples.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)


def sales():
    n = 480
    regions = ["North", "South", "East", "West"]
    channels = ["Direct", "Partner", "Online"]
    region = RNG.choice(regions, n, p=[0.3, 0.25, 0.25, 0.2])
    channel = RNG.choice(channels, n, p=[0.35, 0.25, 0.4])
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        RNG.integers(0, 540, n), unit="D")

    base = {"North": 1.35, "South": 0.95, "East": 1.1, "West": 0.8}
    spend = np.abs(RNG.gamma(3.2, 900, n))
    revenue = np.array([s * base[r] * RNG.normal(2.4, 0.35)
                        for s, r in zip(spend, region)])
    revenue += RNG.normal(0, 600, n)
    units = np.maximum(1, (revenue / RNG.normal(85, 12, n)).round())
    satisfaction = np.clip(RNG.normal(4.1, 0.6, n), 1, 5).round(1)

    df = pd.DataFrame({
        "date": dates,
        "region": region,
        "channel": channel,
        # Thousands separators and a percent sign: the loader must coerce these
        # back to numbers or every numeric chart silently becomes unavailable.
        "spend": [f"{v:,.0f}" for v in spend],
        "revenue": revenue.round(2),
        "units": units.astype(int),
        "satisfaction": satisfaction,
        "margin": [f"{v:.1f}%" for v in np.clip(RNG.normal(18, 5, n), 1, 60)],
    })
    # Real files have holes.
    for col, frac in (("satisfaction", 0.06), ("revenue", 0.02)):
        df.loc[RNG.choice(n, int(n * frac), replace=False), col] = np.nan
    df.to_csv(os.path.join(HERE, "sales.csv"), index=False, encoding="utf-8")
    return "sales.csv", len(df)


def clinical_gbk():
    n = 260
    arms = ["对照组", "低剂量组", "高剂量组"]
    arm = RNG.choice(arms, n, p=[0.34, 0.33, 0.33])
    hazard = {"对照组": 1.0, "低剂量组": 0.72, "高剂量组": 0.55}
    days = np.array([RNG.exponential(320 / hazard[a]) for a in arm])
    days = np.clip(days, 5, 1080).round()
    event = (days < 900).astype(int)
    baseline = np.clip(RNG.normal(62, 11, n), 20, 95).round(1)
    response = np.clip(baseline * 0.4
                       + np.array([12 * (1 / hazard[a]) for a in arm])
                       + RNG.normal(0, 7, n), 0, 100).round(1)

    df = pd.DataFrame({
        "患者编号": [f"P{i:04d}" for i in range(1, n + 1)],
        "组别": arm,
        "生存天数": days.astype(int),
        "是否发生事件": event,
        "基线评分": baseline,
        "疗效评分": response,
        "年龄": RNG.integers(28, 82, n),
    })
    path = os.path.join(HERE, "clinical_gbk.csv")
    df.to_csv(path, index=False, encoding="gbk")
    return "clinical_gbk.csv", len(df)


def trial_effects():
    arms = ["Dose 10 mg", "Dose 20 mg", "Dose 40 mg", "Combination",
            "Active comparator", "Extended release", "Once daily",
            "Twice daily"]
    est = RNG.normal(0.78, 0.16, len(arms)).round(3)
    half = np.abs(RNG.normal(0.17, 0.05, len(arms)))
    df = pd.DataFrame({
        "arm": arms,
        "n": RNG.integers(80, 420, len(arms)),
        "estimate": est,
        "ci_lo": (est - half).round(3),
        "ci_hi": (est + half).round(3),
    })
    df.to_csv(os.path.join(HERE, "trial_effects.csv"), index=False,
              encoding="utf-8")
    return "trial_effects.csv", len(df)


def main():
    for fn in (sales, clinical_gbk, trial_effects):
        name, rows = fn()
        size = os.path.getsize(os.path.join(HERE, name))
        print(f"wrote {name:<22} {rows:>5} rows  {size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
