#!/usr/bin/env python
"""
shap_report.py
==============
Global, per-group, and cluster-level SHAP analysis for the XGBoost
underutilization model.

Uses XGBoost's native TreeSHAP (pred_contribs=True) so no extra
dependency on the `shap` package is required.

Inputs
------
  data/processed/panel_ml.parquet
  data/processed/xgb_model.json
  data/processed/xgb_study.pkl    (for cat_codes)

Outputs
-------
  data/processed/shap_global.txt
  data/processed/shap_by_group.txt
  data/processed/shap_clusters.txt
  data/processed/shap_values.parquet    (per-row SHAP matrix)
  data/processed/shap_lgu_summary.parquet  (per-LGU avg SHAP + cluster label)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_IN     = Path("data/processed/panel_ml.parquet")
DEFAULT_MODEL  = Path("data/processed/xgb_model.json")
DEFAULT_STUDY  = Path("data/processed/xgb_study.pkl")
DEFAULT_OUTDIR = Path("data/processed")

LOG = logging.getLogger("shap_report")


# ---------------------------------------------------------------------------
# Feature configuration — MUST match xgboost_pipeline.py
# ---------------------------------------------------------------------------

FEATURES_NUMERIC = [
    "ur_b_current",
    "ur_b_lag1", "ur_b_lag2",
    "log_nta_ira", "log_total_local_sources",
    "nta_dependency", "revenue_hhi",
    "income_class_num",
    "log_population", "log_land_area_sqkm", "pop_density",
    "fiscal_year", "election_year",
    "pi_2018",
]
FEATURES_CATEGORICAL = ["lgu_type", "region", "urban_rural"]
FEATURES_ALL = FEATURES_NUMERIC + FEATURES_CATEGORICAL


def build_X(df: pd.DataFrame, cat_codes: dict) -> pd.DataFrame:
    X = df[FEATURES_NUMERIC].copy()
    for c in FEATURES_CATEGORICAL:
        codes = pd.Categorical(df[c], categories=cat_codes[c]).codes
        X[c] = codes
    return X


# ---------------------------------------------------------------------------
# Human-readable feature labels for the report
# ---------------------------------------------------------------------------

FEATURE_LABEL = {
    "ur_b_current":             "Underutilization ratio, current year (t)",
    "ur_b_lag1":                "Underutilization ratio, lag 1 (t-1)",
    "ur_b_lag2":                "Underutilization ratio, lag 2 (t-2)",
    "log_nta_ira":              "log(1 + NTA / IRA received)",
    "log_total_local_sources":  "log(1 + total own-source revenue)",
    "nta_dependency":           "NTA as share of total income",
    "revenue_hhi":              "Revenue concentration (HHI)",
    "income_class_num":         "Income class (1=highest, 6=lowest)",
    "log_population":           "log(1 + population)",
    "log_land_area_sqkm":       "log(1 + land area)",
    "pop_density":              "Population density",
    "fiscal_year":              "Fiscal year",
    "election_year":            "Election-year indicator",
    "pi_2018":                  "Poverty incidence (2018 SAE)",
    "lgu_type":                 "LGU type (city / mun / prov)",
    "region":                   "Region (categorical)",
    "urban_rural":              "Urban / rural",
}


# ---------------------------------------------------------------------------
# Compute SHAP
# ---------------------------------------------------------------------------

def compute_shap(model: xgb.XGBRegressor,
                 X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (shap_values, base_value).
        shap_values : (n_rows, n_features)  — feature contributions
        base_value  : scalar               — expected value (bias)
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]
    base = float(contribs[0, -1])
    return shap_vals, base


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_global(shap_vals: np.ndarray, X: pd.DataFrame,
                  out_path: Path, top_n: int = 20) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    L = []
    L.append("=" * 78)
    L.append("GLOBAL SHAP IMPORTANCE")
    L.append("=" * 78)
    L.append("Mean |SHAP| across all rows (higher = more important).")
    L.append("")
    L.append(f"{'rank':>4} {'feature':<30} {'mean|SHAP|':>11} "
             f"{'share':>7}  description")
    L.append("-" * 78)
    total = mean_abs.sum()
    for rank, idx in enumerate(order[:top_n], 1):
        f = X.columns[idx]
        share = mean_abs[idx] / total if total > 0 else 0
        L.append(f"{rank:>4} {f:<30} {mean_abs[idx]:>11.5f} "
                 f"{share:>6.2%}  {FEATURE_LABEL.get(f, '')}")
    L.append("")
    L.append(f"Total mean |SHAP| (all features): {total:.5f}")
    out_path.write_text("\n".join(L), encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)


def report_by_group(shap_vals: np.ndarray, X: pd.DataFrame,
                    meta: pd.DataFrame, out_path: Path) -> None:
    """
    Per-group mean |SHAP| for each feature, grouped by:
      * LGU type
      * income-class bucket (1-3 vs 4-6)
      * region (NCR vs rest)
      * era (SIE 2001-2008, SRE 2009-2013, SRE 2014-2019, SRE 2020-2024)
    """
    abs_shap = np.abs(shap_vals)
    feat_names = list(X.columns)

    def _group_stats(mask, label, lines):
        if not mask.any():
            return
        sub = abs_shap[mask]
        mean_abs = sub.mean(axis=0)
        order = np.argsort(-mean_abs)[:10]
        lines.append(f"\n-- {label}  (n={int(mask.sum()):,}) --")
        for idx in order:
            f = feat_names[idx]
            lines.append(f"  {f:<30} {mean_abs[idx]:>11.5f}  "
                         f"{FEATURE_LABEL.get(f, '')}")

    L = []
    L.append("=" * 78)
    L.append("SHAP IMPORTANCE BY SUBGROUP")
    L.append("=" * 78)
    L.append("Top 10 features per group by mean |SHAP|.")

    # --- LGU type ---
    for lvl in ("province", "city", "municipality"):
        _group_stats(meta["lgu_type"].values == lvl, f"LGU type = {lvl}", L)

    # --- Income class buckets ---
    ic = meta["income_class_num"].values
    _group_stats(np.isin(ic, [1, 2, 3]),
                 "Income class 1-3 (higher income)", L)
    _group_stats(np.isin(ic, [4, 5, 6]),
                 "Income class 4-6 (lower income)", L)

    # --- NCR vs rest ---
    _group_stats(meta["region"].values == "National Capital Region",
                 "NCR", L)
    _group_stats(meta["region"].values != "National Capital Region",
                 "Outside NCR", L)

    # --- Era buckets ---
    fy = meta["fiscal_year"].values
    _group_stats((fy >= 1992) & (fy <= 2000), "BOS era 1992-2000", L)
    _group_stats((fy >= 2001) & (fy <= 2008), "SIE era 2001-2008", L)
    _group_stats((fy >= 2009) & (fy <= 2013), "SRE early 2009-2013", L)
    _group_stats((fy >= 2014) & (fy <= 2019), "SRE pre-COVID 2014-2019", L)
    _group_stats((fy >= 2020) & (fy <= 2024), "SRE post-COVID 2020-2024", L)

    out_path.write_text("\n".join(L), encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)


def report_clusters(shap_vals: np.ndarray, X: pd.DataFrame,
                    meta: pd.DataFrame, out_path: Path,
                    k: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Cluster LGUs by their *average* SHAP profile across the panel.
    Saves a summary and returns a per-LGU frame with cluster labels.
    """
    df_shap = pd.DataFrame(shap_vals, columns=[f"shap_{c}" for c in X.columns])
    df_shap["psgc10"] = meta["psgc10"].values

    # Per-LGU average SHAP vector (one row per LGU).
    lgu_shap = df_shap.groupby("psgc10", sort=False).mean()

    # Cluster on standardized SHAP vectors.
    scaler = StandardScaler()
    Z = scaler.fit_transform(lgu_shap.values)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Z)
    lgu_shap["cluster"] = labels

    # Attach descriptive attributes per LGU (mode of categorical, mean of numeric).
    lgu_meta = (meta.groupby("psgc10")
                    .agg(lgu_name=("lgu_name", "first"),
                         lgu_type=("lgu_type", "first"),
                         region=("region", "first"),
                         income_class=("income_class_num", "median"),
                         mean_ur=("ur_b", "mean"),
                         mean_nta_dep=("nta_dependency", "mean"),
                         mean_pop=("log_population", "mean"),
                         mean_year=("fiscal_year", "mean")))

    summary = lgu_shap[["cluster"]].join(lgu_meta)

    L = []
    L.append("=" * 78)
    L.append(f"K-MEANS SHAP CLUSTERING  (k={k})")
    L.append("=" * 78)
    L.append("Each cluster groups LGUs whose average SHAP vectors are similar,")
    L.append("i.e. LGUs driven by the same combination of predictive factors.")
    L.append("")

    # Cluster sizes + dominant attributes
    L.append("-- Cluster sizes and dominant attributes --")
    L.append(f"{'cluster':>7} {'n_lgus':>8} {'mean_ur':>8} {'mean_nta_dep':>13} "
             f"{'mean_year':>10}  top_type   top_region")
    L.append("-" * 78)
    for c in range(k):
        sub = summary[summary["cluster"] == c]
        if len(sub) == 0:
            continue
        top_type = sub["lgu_type"].value_counts().idxmax()
        top_region = sub["region"].value_counts().idxmax()
        L.append(f"{c:>7} {len(sub):>8} "
                 f"{sub['mean_ur'].mean():>8.3f} "
                 f"{sub['mean_nta_dep'].mean():>13.3f} "
                 f"{sub['mean_year'].mean():>10.1f}  "
                 f"{top_type:<10} {top_region}")

    L.append("")
    L.append("-- Cluster signatures: mean SHAP of the features that most")
    L.append("   distinguish each cluster from the global mean --")
    global_mean = lgu_shap.drop(columns=["cluster"]).mean(axis=0)
    for c in range(k):
        sub = lgu_shap[lgu_shap["cluster"] == c].drop(columns=["cluster"])
        if len(sub) == 0:
            continue
        cluster_mean = sub.mean(axis=0)
        diff = cluster_mean - global_mean
        order = np.argsort(-np.abs(diff))[:8]
        L.append(f"\n  Cluster {c}  (n={len(sub)}):")
        for idx in order:
            f = diff.index[idx]
            L.append(f"    {f:<30} delta={diff.iloc[idx]:+8.5f}  "
                     f"{FEATURE_LABEL.get(f, '')}")

    L.append("")
    L.append("-- Representative LGUs per cluster (highest mean UR) --")
    for c in range(k):
        sub = summary[summary["cluster"] == c].nlargest(5, "mean_ur")
        if len(sub) == 0:
            continue
        L.append(f"\n  Cluster {c}:")
        for psgc, row in sub.iterrows():
            L.append(f"    {psgc}  {row['lgu_name'][:40]:<40}  "
                     f"UR={row['mean_ur']:.3f}  "
                     f"{row['lgu_type']:<12}  {row['region']}")

    out_path.write_text("\n".join(L), encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)

    return summary.reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in",       dest="inp",   type=Path, default=DEFAULT_IN)
    p.add_argument("--model",    type=Path,    default=DEFAULT_MODEL)
    p.add_argument("--study",    type=Path,    default=DEFAULT_STUDY)
    p.add_argument("--outdir",   type=Path,    default=DEFAULT_OUTDIR)
    p.add_argument("--k",        type=int,     default=5)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    LOG.info("Loading panel: %s", args.inp)
    df = pd.read_parquet(args.inp)
    LOG.info("  %d rows x %d cols", *df.shape)

    LOG.info("Loading model: %s", args.model)
    model = xgb.XGBRegressor()
    model.load_model(str(args.model))

    LOG.info("Loading study (for cat_codes): %s", args.study)
    with open(args.study, "rb") as f:
        study_pkl = pickle.load(f)
    cat_codes = study_pkl["cat_codes"]

    # Use ALL rows (including 2020-2024) so SHAP reflects the deployment
    # population, not just the training sample.
    df_eval = df[df["target_ur_b"].notna()].copy()
    LOG.info("  evaluating SHAP on %d rows", len(df_eval))

    X = build_X(df_eval, cat_codes)
    LOG.info("  X shape: %s", X.shape)

    LOG.info("Computing TreeSHAP...")
    shap_vals, base = compute_shap(model, X)
    LOG.info("  base value (expected model output): %.4f", base)
    LOG.info("  shap_vals shape: %s", shap_vals.shape)

        # Persist raw SHAP for downstream use.
    # Prefix feature columns so they can never collide with metadata names
    # (fiscal_year, lgu_type, region all appear on both sides).
    shap_df = pd.DataFrame(shap_vals,
                           columns=[f"shap_{c}" for c in X.columns])
    out_shap = pd.concat([
        df_eval[["psgc10", "fiscal_year", "lgu_name", "lgu_type",
                 "region", "ur_b", "target_ur_b"]].reset_index(drop=True),
        shap_df.reset_index(drop=True),
    ], axis=1)
    assert len(out_shap.columns) == len(set(out_shap.columns)), \
        f"Duplicate columns after concat: {list(out_shap.columns)}"
    out_shap.to_parquet(args.outdir / "shap_values.parquet", index=False)
    LOG.info("Wrote shap_values.parquet (%d rows)", len(out_shap))

    # Reports.
    args.outdir.mkdir(parents=True, exist_ok=True)
    report_global(shap_vals, X,
                  args.outdir / "shap_global.txt")
    report_by_group(shap_vals, X, df_eval.reset_index(drop=True),
                    args.outdir / "shap_by_group.txt")

    lgu_summary = report_clusters(shap_vals, X, df_eval.reset_index(drop=True),
                                   args.outdir / "shap_clusters.txt",
                                   k=args.k)
    lgu_summary.to_parquet(args.outdir / "shap_lgu_summary.parquet",
                            index=False)
    LOG.info("Wrote shap_lgu_summary.parquet")

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())