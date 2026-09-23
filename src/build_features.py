#!/usr/bin/env python
"""
build_features.py
=================
Construct the underutilization target and the ML feature matrix.

Row key: (psgc10, decision_year).
  decision_year = t   ->  features from year t
  target        = ur at year t+1

Inputs
------
  data/processed/panel.parquet                    (SRE + PSGC + poverty)
  data/processed/population_lgu_annual.parquet    (pop, land area)
  data/processed/psgc_lgu_master.parquet          (structural attributes)

Outputs
-------
  data/processed/panel_ml.parquet
  data/processed/panel_ml_qa.txt
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_PANEL = Path("data/processed/panel.parquet")
DEFAULT_POP   = Path("data/processed/population_lgu_annual.parquet")
DEFAULT_PSGC  = Path("data/processed/psgc_lgu_master.parquet")
DEFAULT_OUT   = Path("data/processed/panel_ml.parquet")

NATIONAL_ELECTIONS = {1992, 1995, 1998, 2001, 2004, 2007,
                      2010, 2013, 2016, 2019, 2022, 2025}

LOG = logging.getLogger("build_features")


def _winsorize(s: pd.Series, lo=0.0, hi=1.0) -> pd.Series:
    return s.astype(float).clip(lower=lo, upper=hi)


def add_ur_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numerator candidate 1: detailed CO + Capex split (populated 2009+).
    spent_detailed = (df["total_current_operating_exp"].fillna(0)
                      + df["total_capital_investment_exp"].fillna(0))

    # Numerator candidate 2: top-line total expenditures (populated across
    # all eras — BOS, SIE, and SRE).  Used where the detailed split is
    # missing, mainly SIE 2001-2008 where the harmonizer never mapped the
    # CO/Capex line items.
    if "total_expenditures" in df.columns:
        spent_total = df["total_expenditures"].fillna(0)
    else:
        spent_total = pd.Series(0.0, index=df.index)

    # Prefer the detailed split when non-zero; otherwise fall back.
    spent = spent_detailed.where(spent_detailed > 0, spent_total)
    df["spent_used"]   = spent
    df["spent_source"] = np.where(spent_detailed > 0, "detailed", "total")

    avail_a = df["fund_cash_available"].where(df["fund_cash_available"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ur_a"] = _winsorize(1.0 - spent / avail_a)

    avail_b = df["total_current_operating_income"].where(
        df["total_current_operating_income"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ur_b"] = _winsorize(1.0 - spent / avail_b)

    return df


def add_lags_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lags and (t+1) targets.

    IMPORTANT: `ur_b_current` / `ur_a_current` are the SAME-YEAR UR values —
    i.e. UR at year t.  They are what the paper's naive persistence
    baseline uses:  predict UR(t+1) = UR(t).  Without them, the naive
    baseline would use the 2-year lag and understate its own accuracy.

    Also note the observation-window shift for lagged features:
        row (LGU, t) carries:
            ur_b_lag1   = UR_b at t-1
            ur_b_lag2   = UR_b at t-2
            target_ur_b = UR_b at t+1
    """
    df = df.sort_values(["psgc10", "fiscal_year"]).copy()
    g = df.groupby("psgc10", sort=False)

    # Same-year UR — used by naive persistence AND available to ML models.
    df["ur_a_current"] = df["ur_a"]
    df["ur_b_current"] = df["ur_b"]

    # Lagged features (from t-1, t-2) for the ML models.
    for lag in (1, 2):
        df[f"ur_a_lag{lag}"] = g["ur_a"].shift(lag)
        df[f"ur_b_lag{lag}"] = g["ur_b"].shift(lag)
        df[f"nta_ira_lag{lag}"] = g["nta_ira"].shift(lag)
        df[f"fund_cash_balance_end_lag{lag}"] = g["fund_cash_balance_end"].shift(lag)
        df[f"total_current_operating_income_lag{lag}"] = \
            g["total_current_operating_income"].shift(lag)

    # Target = UR one year ahead (t+1).
    df["target_ur_a"] = g["ur_a"].shift(-1)
    df["target_ur_b"] = g["ur_b"].shift(-1)

    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    log_cols = [
        "nta_ira", "total_current_operating_income",
        "total_local_sources", "total_external_sources",
        "fund_cash_balance_end", "total_capital_investment_exp",
        "population", "land_area_sqkm",
    ]
    for c in log_cols:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    denom = df["total_current_operating_income"].replace(0, np.nan)
    df["nta_dependency"]      = df["nta_ira"] / denom
    df["local_share"]         = df["total_local_sources"] / denom
    df["external_share"]      = df["total_external_sources"] / denom
    df["fund_balance_share"]  = df["fund_cash_balance_end"] / denom
    df["capex_share_avail"]   = (df["total_capital_investment_exp"]
                                  / df["fund_cash_available"].replace(0, np.nan))

    # Revenue-concentration HHI on local-source components.
    comps = ["rpt", "business_tax", "other_taxes",
             "regulatory_fees", "service_charges", "econ_enterprise"]
    present = [c for c in comps if c in df.columns]
    if present:
        comp_sum = df[present].sum(axis=1, skipna=True)
        shares = df[present].div(comp_sum.replace(0, np.nan), axis=0)
        df["revenue_hhi"] = (shares ** 2).sum(axis=1)
    return df


def add_structural(df: pd.DataFrame) -> pd.DataFrame:
    if "population" in df.columns:
        df["pop_density"] = df["population"] / df["land_area_sqkm"].replace(0, np.nan)
    df["election_year"] = df["fiscal_year"].isin(NATIONAL_ELECTIONS).astype(int)
    if "income_class" in df.columns:
        df["income_class_num"] = pd.to_numeric(df["income_class"], errors="coerce")
    return df


def load_and_merge(panel_path: Path, pop_path: Path) -> pd.DataFrame:
    LOG.info("Loading panel: %s", panel_path)
    panel = pd.read_parquet(panel_path)
    LOG.info("  %d rows x %d cols", *panel.shape)

    if "in_sre" in panel.columns:
        panel = panel[panel["in_sre"]].copy()
        LOG.info("  %d rows with in_sre=True", len(panel))

    LOG.info("Loading population: %s", pop_path)
    pop = pd.read_parquet(pop_path)
    LOG.info("  %d rows", len(pop))

    panel = panel.merge(
        pop[["psgc10", "fiscal_year", "population", "land_area_sqkm"]],
        on=["psgc10", "fiscal_year"], how="left")
    n_pop = int(panel["population"].notna().sum())
    LOG.info("  after pop merge: %d rows; population non-null on %d (%.1f%%)",
             len(panel), n_pop, 100 * n_pop / max(len(panel), 1))
    return panel


def write_qa(df: pd.DataFrame, out_path: Path) -> None:
    L = []
    L.append("=" * 78)
    L.append("PANEL_ML QA")
    L.append("=" * 78)
    L.append(f"Rows: {len(df):,}   Cols: {len(df.columns)}")
    L.append(f"Years: {df['fiscal_year'].min()}-{df['fiscal_year'].max()}")
    L.append(f"LGUs:  {df['psgc10'].nunique()}")
    L.append("")

    L.append("-- UR coverage --")
    for col in ("ur_a", "ur_b",
                "ur_a_current", "ur_b_current",
                "target_ur_a", "target_ur_b"):
        if col in df.columns:
            n = int(df[col].notna().sum())
            L.append(f"  {col:18s} non-null {n:>7,}  "
                     f"({100 * n / max(len(df), 1):5.1f}%)  "
                     f"median={df[col].median(skipna=True):.3f}")
    L.append("")

    L.append("-- UR_b by year --")
    yr = (df.groupby("fiscal_year")["ur_b"]
            .agg(n="count", mean="mean", median="median")
            .round(3))
    for y, row in yr.iterrows():
        L.append(f"  {int(y)}: n={int(row['n']):>5,}  "
                 f"mean={row['mean']:.3f}  median={row['median']:.3f}")
    L.append("")

    L.append("-- UR_b by template era --")
    if "template_era" in df.columns:
        era = (df.groupby("template_era")["ur_b"]
                 .agg(n="count", mean="mean", median="median")
                 .round(3))
        for k, row in era.iterrows():
            L.append(f"  {k:18s} n={int(row['n']):>5,}  "
                     f"mean={row['mean']:.3f}  median={row['median']:.3f}")
    L.append("")

    L.append("-- spent_source by era --")
    if "spent_source" in df.columns and "template_era" in df.columns:
        src = (df.groupby(["template_era", "spent_source"])
                 .size()
                 .unstack(fill_value=0))
        for era, row in src.iterrows():
            parts = "  ".join(f"{k}={int(v):>6,}" for k, v in row.items())
            L.append(f"  {era:18s} {parts}")
    L.append("")

    L.append("-- UR by decade (both definitions) --")
    for lo, hi in ((1992, 1999), (2000, 2009), (2010, 2019), (2020, 2024)):
        sub = df[df["fiscal_year"].between(lo, hi)]
        if not len(sub):
            continue
        L.append(f"  {lo}-{hi}: n={len(sub):>6,}  "
                 f"ur_a mean={sub['ur_a'].mean():.3f} "
                 f"non-null={sub['ur_a'].notna().mean():.1%}  | "
                 f"ur_b mean={sub['ur_b'].mean():.3f} "
                 f"non-null={sub['ur_b'].notna().mean():.1%}")
    L.append("")

    L.append("-- Feature non-null rates --")
    prefixes = ("ur_", "log_", "target_", "nta_dependency",
                "local_share", "external_share", "fund_balance_",
                "capex_share_", "revenue_hhi", "pop_density")
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    for c in sorted(feature_cols):
        L.append(f"  {c:38s} {df[c].notna().mean():.1%}")

    out_path.write_text("\n".join(L), encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--pop",   type=Path, default=DEFAULT_POP)
    p.add_argument("--out",   type=Path, default=DEFAULT_OUT)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    df = load_and_merge(args.panel, args.pop)
    df = add_ur_targets(df)
    df = add_lags_and_targets(df)
    df = add_log_features(df)
    df = add_ratio_features(df)
    df = add_structural(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    LOG.info("Wrote %s (%d rows x %d cols)", args.out, *df.shape)

    write_qa(df, args.out.parent / "panel_ml_qa.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())