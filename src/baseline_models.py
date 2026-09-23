#!/usr/bin/env python
"""
baseline_models.py
==================
Naive persistence + Ridge + Random Forest baselines for one-year-ahead
UR prediction, using the paper's time-series CV fold structure.

Folds (expanding window):
    F1: train 1992-2005  test 2006-2008
    F2: train 1992-2008  test 2009-2011
    F3: train 1992-2011  test 2012-2014
    F4: train 1992-2014  test 2015-2017
    F5: train 1992-2017  test 2018-2019
    H : train 1992-2019  test 2020-2024   (final holdout)

The naive persistence model — predict next year's UR = this year's UR — is the
benchmark the paper says the ML model must beat.

Inputs
------
  data/processed/panel_ml.parquet

Outputs
-------
  data/processed/baseline_results.txt
  data/processed/baseline_predictions.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_IN = Path("data/processed/panel_ml.parquet")
DEFAULT_OUTDIR = Path("data/processed")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

LOG = logging.getLogger("baseline_models")


# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    name: str
    train_years: tuple[int, int]
    test_years:  tuple[int, int]

    def __str__(self) -> str:
        return (f"{self.name}: train {self.train_years[0]}-{self.train_years[1]} "
                f"| test {self.test_years[0]}-{self.test_years[1]}")


FOLDS = [
    Fold("F1", (1992, 2005), (2006, 2008)),
    Fold("F2", (1992, 2008), (2009, 2011)),
    Fold("F3", (1992, 2011), (2012, 2014)),
    Fold("F4", (1992, 2014), (2015, 2017)),
    Fold("F5", (1992, 2017), (2018, 2019)),
    Fold("H",  (1992, 2019), (2020, 2024)),   # final holdout
]


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

FEATURES_NUMERIC = [
    "ur_b_current",          # <-- ADD THIS LINE
    "ur_b_lag1", "ur_b_lag2",
    "log_nta_ira", "log_total_local_sources",
    "nta_dependency", "revenue_hhi",
    "income_class_num",
    "log_population", "log_land_area_sqkm", "pop_density",
    "fiscal_year", "election_year",
    "pi_2018",
]

FEATURES_CATEGORICAL = ["lgu_type", "region", "urban_rural"]


@dataclass
class BaselineResult:
    fold: str
    model: str
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    rmse_ratio_vs_naive: float = float("nan")
    preds: np.ndarray = field(default_factory=lambda: np.array([]))
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def make_ridge() -> Pipeline:
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer([
        ("num", num_pipe, FEATURES_NUMERIC),
        ("cat", cat_pipe, FEATURES_CATEGORICAL),
    ])
    return Pipeline([("prep", ct), ("model", Ridge(alpha=1.0, random_state=0))])


def make_rf() -> Pipeline:
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer([
        ("num", num_pipe, FEATURES_NUMERIC),
        ("cat", cat_pipe, FEATURES_CATEGORICAL),
    ])
    return Pipeline([
        ("prep", ct),
        ("model", RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=1,                    # ← the fix
            random_state=0)),
    ])

def try_make_lgbm():
    try:
        from lightgbm import LGBMRegressor  # noqa: WPS433
    except ImportError:
        return None
    return LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=0, verbose=-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, target: str, outdir: Path) -> pd.DataFrame:
    feature_cols = FEATURES_NUMERIC + FEATURES_CATEGORICAL

    # Drop rows with no target.
    df = df[df[target].notna()].copy()
    LOG.info("Rows with non-null %s: %d", target, len(df))

    # Ensure all feature cols exist; warn about any missing.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")

    lgbm = try_make_lgbm()
    if lgbm is None:
        LOG.warning("lightgbm not installed — skipping LGBM baseline")

    results: list[BaselineResult] = []
    pred_frames: list[pd.DataFrame] = []

    for fold in FOLDS:
        LOG.info("")
        LOG.info("=" * 60)
        LOG.info("%s", fold)
        LOG.info("=" * 60)

        tr = df[df["fiscal_year"].between(*fold.train_years)].copy()
        te = df[df["fiscal_year"].between(*fold.test_years)].copy()
        if len(tr) == 0 or len(te) == 0:
            LOG.warning("  empty train or test — skipping")
            continue
        LOG.info("  train n=%d, test n=%d", len(tr), len(te))

        X_tr = tr[feature_cols]
        y_tr = tr[target].values
        X_te = te[feature_cols]
        y_te = te[target].values

        # ---- Naive persistence ------------------------------------------
        naive_pred = te["ur_b_current"].fillna(y_tr.mean()).values
        r, m, r2 = _metrics(y_te, naive_pred)
        results.append(BaselineResult(fold.name, "naive_persistence",
                                      len(tr), len(te), r, m, r2))
        pred_frames.append(pd.DataFrame({
            "psgc10":     te["psgc10"].values,
            "fiscal_year": te["fiscal_year"].values,
            "fold":       fold.name,
            "model":      "naive_persistence",
            "y_true":     y_te,
            "y_pred":     naive_pred,
        }))
        LOG.info("  naive_persistence  RMSE=%.4f  MAE=%.4f  R2=%.4f", r, m, r2)

        # ---- Fold mean (floor) ------------------------------------------
        mean_pred = np.full_like(y_te, y_tr.mean(), dtype=float)
        r, m, r2 = _metrics(y_te, mean_pred)
        results.append(BaselineResult(fold.name, "fold_mean",
                                      len(tr), len(te), r, m, r2))
        LOG.info("  fold_mean          RMSE=%.4f  MAE=%.4f  R2=%.4f", r, m, r2)

        # ---- Zero predictor (also a floor) ------------------------------
        zero_pred = np.zeros_like(y_te, dtype=float)
        r, m, r2 = _metrics(y_te, zero_pred)
        results.append(BaselineResult(fold.name, "zero",
                                      len(tr), len(te), r, m, r2))
        LOG.info("  zero               RMSE=%.4f  MAE=%.4f  R2=%.4f", r, m, r2)

        # ---- Ridge ------------------------------------------------------
        ridge = make_ridge()
        ridge.fit(X_tr, y_tr)
        pred = ridge.predict(X_te)
        pred = np.clip(pred, 0.0, 1.0)
        r, m, r2 = _metrics(y_te, pred)
        results.append(BaselineResult(fold.name, "ridge",
                                      len(tr), len(te), r, m, r2))
        pred_frames.append(pd.DataFrame({
            "psgc10":     te["psgc10"].values,
            "fiscal_year": te["fiscal_year"].values,
            "fold":       fold.name,
            "model":      "ridge",
            "y_true":     y_te,
            "y_pred":     pred,
        }))
        LOG.info("  ridge              RMSE=%.4f  MAE=%.4f  R2=%.4f", r, m, r2)

        # ---- Random Forest ----------------------------------------------
        rf = make_rf()
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        pred = np.clip(pred, 0.0, 1.0)
        r, m, r2 = _metrics(y_te, pred)
        results.append(BaselineResult(fold.name, "random_forest",
                                      len(tr), len(te), r, m, r2))
        pred_frames.append(pd.DataFrame({
            "psgc10":     te["psgc10"].values,
            "fiscal_year": te["fiscal_year"].values,
            "fold":       fold.name,
            "model":      "random_forest",
            "y_true":     y_te,
            "y_pred":     pred,
        }))
        LOG.info("  random_forest      RMSE=%.4f  MAE=%.4f  R2=%.4f", r, m, r2)

        # ---- LightGBM (optional) ----------------------------------------
        if lgbm is not None:
            lgbm_fresh = try_make_lgbm()
            # LightGBM handles NaN natively, but categorical columns still
            # need to be numeric.
            X_tr_l = X_tr.copy()
            X_te_l = X_te.copy()
            for c in FEATURES_CATEGORICAL:
                X_tr_l[c] = X_tr_l[c].astype("category")
                X_te_l[c] = pd.Categorical(X_te_l[c],
                                            categories=X_tr_l[c].cat.categories)
            lgbm_fresh.fit(X_tr_l, y_tr,
                           categorical_feature=FEATURES_CATEGORICAL)
            pred = lgbm_fresh.predict(X_te_l)
            pred = np.clip(pred, 0.0, 1.0)
            r, m, r2 = _metrics(y_te, pred)
            results.append(BaselineResult(fold.name, "lightgbm",
                                          len(tr), len(te), r, m, r2))
            pred_frames.append(pd.DataFrame({
                "psgc10":     te["psgc10"].values,
                "fiscal_year": te["fiscal_year"].values,
                "fold":       fold.name,
                "model":      "lightgbm",
                "y_true":     y_te,
                "y_pred":     pred,
            }))
            LOG.info("  lightgbm           RMSE=%.4f  MAE=%.4f  R2=%.4f",
                     r, m, r2)

    # Post-hoc: annotate RMSE ratio vs. naive for each (fold, model).
    df_res = pd.DataFrame([{
        "fold": r.fold, "model": r.model,
        "n_train": r.n_train, "n_test": r.n_test,
        "rmse": r.rmse, "mae": r.mae, "r2": r.r2,
    } for r in results])

    naive_rmse = df_res[df_res["model"] == "naive_persistence"] \
                    .set_index("fold")["rmse"]
    df_res["rmse_ratio_vs_naive"] = df_res.apply(
        lambda r: r["rmse"] / naive_rmse.get(r["fold"], np.nan), axis=1)

    preds = pd.concat(pred_frames, ignore_index=True)
    return df_res, preds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(df_res: pd.DataFrame, out_path: Path,
                 target: str) -> None:
    L = []
    L.append("=" * 90)
    L.append("BASELINE MODEL RESULTS")
    L.append("=" * 90)
    L.append(f"Target: {target}")
    L.append("")
    L.append("RMSE ratio < 1.0 means the model beats naive persistence;")
    L.append("ratio >= 1.0 means naive persistence is at least as good.")
    L.append("")
    L.append("-- Per-fold metrics --")
    L.append(f"{'fold':<5} {'model':<20} {'n_train':>8} {'n_test':>7} "
             f"{'RMSE':>8} {'MAE':>8} {'R2':>8} {'RMSE_ratio':>11}")
    L.append("-" * 90)
    for _, r in df_res.sort_values(["fold", "model"]).iterrows():
        L.append(f"{r['fold']:<5} {r['model']:<20} {r['n_train']:>8,} "
                 f"{r['n_test']:>7,} {r['rmse']:>8.4f} "
                 f"{r['mae']:>8.4f} {r['r2']:>8.4f} "
                 f"{r['rmse_ratio_vs_naive']:>11.3f}")

    L.append("")
    L.append("-- Cross-fold mean (averaging across the 5 CV folds only, "
             "excluding H) --")
    cv = df_res[df_res["fold"] != "H"]
    for model in sorted(cv["model"].unique()):
        sub = cv[cv["model"] == model]
        L.append(f"  {model:<20} "
                 f"RMSE={sub['rmse'].mean():.4f}  "
                 f"MAE={sub['mae'].mean():.4f}  "
                 f"R2={sub['r2'].mean():.4f}  "
                 f"RMSE_ratio={sub['rmse_ratio_vs_naive'].mean():.3f}")

    L.append("")
    L.append("-- Final holdout (H: 2020-2024) --")
    h = df_res[df_res["fold"] == "H"]
    for _, r in h.iterrows():
        L.append(f"  {r['model']:<20} "
                 f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  "
                 f"R2={r['r2']:.4f}  "
                 f"RMSE_ratio={r['rmse_ratio_vs_naive']:.3f}")

    L.append("")
    L.append("-- Verdict --")
    cv_by_model = cv.groupby("model")["rmse"].mean()
    best_baseline = cv_by_model.drop(labels=["naive_persistence",
                                             "fold_mean", "zero"],
                                     errors="ignore").idxmin()
    best_rmse = cv_by_model[best_baseline]
    naive_rmse = cv_by_model["naive_persistence"]
    ratio = best_rmse / naive_rmse
    L.append(f"  Best non-trivial baseline:  {best_baseline}")
    L.append(f"  Its CV mean RMSE:           {best_rmse:.4f}")
    L.append(f"  Naive persistence RMSE:     {naive_rmse:.4f}")
    L.append(f"  Ratio:                      {ratio:.3f}")
    if ratio < 0.95:
        L.append("  => Baseline beats naive by >5%. XGBoost has a target.")
    elif ratio < 1.0:
        L.append("  => Baseline beats naive by <5%. XGBoost may squeeze "
                 "a bit more.")
    else:
        L.append("  => No baseline beats naive persistence. Re-examine the "
                 "feature set before investing in XGBoost.")

    out_path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--target", default="target_ur_b",
                   choices=["target_ur_a", "target_ur_b"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    LOG.info("Loading %s", args.inp)
    df = pd.read_parquet(args.inp)
    LOG.info("  %d rows x %d cols", *df.shape)

    df_res, preds = evaluate(df, args.target, args.outdir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(args.outdir / "baseline_results.csv", index=False)
    preds.to_parquet(args.outdir / "baseline_predictions.parquet",
                     index=False)
    write_report(df_res, args.outdir / "baseline_results.txt", args.target)
    LOG.info("Wrote baseline_results.txt / .csv and baseline_predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())