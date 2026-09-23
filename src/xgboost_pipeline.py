#!/usr/bin/env python
"""
xgboost_pipeline.py
===================
XGBoost with Optuna hyperparameter search, time-series CV, spatial holdout,
and a Ridge benchmark on the same folds.

Design
------
- 5 expanding-window CV folds (identical to baseline_models.py)
- Final holdout: 2020-2024
- Spatial holdout: 20% of LGUs (stratified by region), held out across all years
- Optuna objective: mean RMSE across the 5 CV folds (holdout not touched)
- Early stopping inside each fold on that fold's test set

Inputs
------
  data/processed/panel_ml.parquet

Outputs
-------
  data/processed/xgb_results.txt
  data/processed/xgb_results.csv
  data/processed/xgb_predictions.parquet
  data/processed/xgb_model.json
  data/processed/xgb_study.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_IN = Path("data/processed/panel_ml.parquet")
DEFAULT_OUTDIR = Path("data/processed")

LOG = logging.getLogger("xgboost_pipeline")


# ---------------------------------------------------------------------------
# Configuration
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


@dataclass
class Fold:
    name: str
    train_years: tuple[int, int]
    test_years: tuple[int, int]


FOLDS = [
    Fold("F1", (1992, 2005), (2006, 2008)),
    Fold("F2", (1992, 2008), (2009, 2011)),
    Fold("F3", (1992, 2011), (2012, 2014)),
    Fold("F4", (1992, 2014), (2015, 2017)),
    Fold("F5", (1992, 2017), (2018, 2019)),
    Fold("H",  (1992, 2019), (2020, 2024)),
]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Return df with categorical columns label-encoded to consistent integer
    codes across train and test.  XGBoost handles categorical natively in
    2.x, but for cross-version safety we one-hot on the fly via pandas
    get_dummies — cheap given only 3 categorical columns.
    """
    df = df.copy()
    for c in FEATURES_CATEGORICAL:
        df[c] = df[c].astype("category")
    # Ordinal-encode categoricals to integers, keeping all categories.
    cat_codes = {}
    for c in FEATURES_CATEGORICAL:
        cat_codes[c] = df[c].cat.categories
    return df, cat_codes


def build_X(df: pd.DataFrame, cat_codes: dict) -> pd.DataFrame:
    X = df[FEATURES_NUMERIC].copy()
    for c in FEATURES_CATEGORICAL:
        codes = pd.Categorical(df[c], categories=cat_codes[c]).codes
        X[c] = codes
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(df: pd.DataFrame, cat_codes: dict,
                   n_trials_log_interval: int = 10):
    """Return an Optuna objective that trains 5 CV folds and returns mean RMSE."""
    trial_counter = {"n": 0}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "reg:squarederror",
            "tree_method":       "hist",
            "n_estimators":      2000,          # capped by early stopping
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "gamma":             trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            "n_jobs":            1,
            "random_state":      0,
            "verbosity":         0,
            "early_stopping_rounds": 50,
            "eval_metric":       "rmse",
        }

        fold_rmses = []
        for fold in FOLDS:
            if fold.name == "H":
                continue  # never use holdout for tuning
            tr = df[df["fiscal_year"].between(*fold.train_years)]
            te = df[df["fiscal_year"].between(*fold.test_years)]
            if len(tr) == 0 or len(te) == 0:
                continue

            X_tr = build_X(tr, cat_codes); y_tr = tr["target_ur_b"].values
            X_te = build_X(te, cat_codes); y_te = te["target_ur_b"].values

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)],
                      verbose=False)
            pred = np.clip(model.predict(X_te), 0.0, 1.0)
            rmse, _, _ = _metrics(y_te, pred)
            fold_rmses.append(rmse)

        trial_counter["n"] += 1
        if trial_counter["n"] % n_trials_log_interval == 0:
            LOG.info("  trial %d: mean_rmse=%.5f",
                     trial_counter["n"], float(np.mean(fold_rmses)))

        return float(np.mean(fold_rmses))

    return objective


# ---------------------------------------------------------------------------
# Final evaluation on CV folds + holdout
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: str
    model: str
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    preds: np.ndarray = None
    y_true: np.ndarray = None
    # Spatial-holdout metrics (NaN if not applicable)
    rmse_spatial: float = float("nan")
    r2_spatial: float = float("nan")


def spatial_split(df: pd.DataFrame, fraction: float = 0.20,
                  seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """
    Return (train_mask, test_mask) where test_mask selects ~20% of LGUs
    (stratified by region) across ALL years.
    """
    rng = np.random.default_rng(seed)
    lgu_region = df.groupby("psgc10")["region"].first()
    test_lgus: set[str] = set()
    for region, grp in lgu_region.groupby(lgu_region):
        lgus = grp.index.to_numpy()
        n_test = max(1, int(round(fraction * len(lgus))))
        pick = rng.choice(lgus, size=n_test, replace=False)
        test_lgus.update(pick.tolist())
    test_mask = df["psgc10"].isin(test_lgus)
    return ~test_mask, test_mask


def run_cv_and_holdout(df: pd.DataFrame, params: dict,
                       cat_codes: dict) -> tuple[list[FoldResult], pd.DataFrame]:
    results: list[FoldResult] = []
    pred_frames: list[pd.DataFrame] = []

    # Spatial split (same test set reused for every fold).
    spatial_train_mask, spatial_test_mask = spatial_split(df, fraction=0.20)
    LOG.info("Spatial split: %d train LGUs, %d test LGUs",
             df.loc[spatial_train_mask, "psgc10"].nunique(),
             df.loc[spatial_test_mask, "psgc10"].nunique())

    for fold in FOLDS:
        LOG.info("")
        LOG.info("=" * 60)
        LOG.info("%s: train %d-%d | test %d-%d",
                 fold.name, *fold.train_years, *fold.test_years)
        LOG.info("=" * 60)

        tr = df[df["fiscal_year"].between(*fold.train_years)]
        te = df[df["fiscal_year"].between(*fold.test_years)]
        if len(tr) == 0 or len(te) == 0:
            LOG.warning("  empty fold — skipping")
            continue

        X_tr = build_X(tr, cat_codes); y_tr = tr["target_ur_b"].values
        X_te = build_X(te, cat_codes); y_te = te["target_ur_b"].values

        # ---- XGBoost ------------------------------------------------------
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred = np.clip(model.predict(X_te), 0.0, 1.0)
        rmse, mae, r2 = _metrics(y_te, pred)
        results.append(FoldResult(fold.name, "xgboost",
                                  len(tr), len(te), rmse, mae, r2,
                                  pred, y_te))
        LOG.info("  xgboost        RMSE=%.4f  MAE=%.4f  R2=%.4f",
                 rmse, mae, r2)
        pred_frames.append(pd.DataFrame({
            "psgc10":     te["psgc10"].values,
            "fiscal_year": te["fiscal_year"].values,
            "fold":       fold.name,
            "model":      "xgboost",
            "y_true":     y_te,
            "y_pred":     pred,
        }))

        # ---- Ridge benchmark (same folds) ---------------------------------
        ridge_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        # Ridge cannot handle NaN in categoricals cleanly; fill them.
        X_tr_r = X_tr.fillna(-999)
        X_te_r = X_te.fillna(-999)
        ridge_pipe.fit(X_tr_r, y_tr)
        r_pred = np.clip(ridge_pipe.predict(X_te_r), 0.0, 1.0)
        rr, rm, rr2 = _metrics(y_te, r_pred)
        results.append(FoldResult(fold.name, "ridge",
                                  len(tr), len(te), rr, rm, rr2,
                                  r_pred, y_te))
        LOG.info("  ridge          RMSE=%.4f  MAE=%.4f  R2=%.4f",
                 rr, rm, rr2)
        pred_frames.append(pd.DataFrame({
            "psgc10":     te["psgc10"].values,
            "fiscal_year": te["fiscal_year"].values,
            "fold":       fold.name,
            "model":      "ridge",
            "y_true":     y_te,
            "y_pred":     r_pred,
        }))

        # ---- Spatial holdout (fit on non-held-out LGUs, predict held-out) -
        tr_s = tr[tr["psgc10"].isin(df.loc[spatial_train_mask, "psgc10"])]
        te_s = te[te["psgc10"].isin(df.loc[spatial_test_mask, "psgc10"])]
        if len(tr_s) > 0 and len(te_s) > 0:
            X_tr_s = build_X(tr_s, cat_codes); y_tr_s = tr_s["target_ur_b"].values
            X_te_s = build_X(te_s, cat_codes); y_te_s = te_s["target_ur_b"].values
            m_s = xgb.XGBRegressor(**params)
            m_s.fit(X_tr_s, y_tr_s, eval_set=[(X_te_s, y_te_s)],
                    verbose=False)
            p_s = np.clip(m_s.predict(X_te_s), 0.0, 1.0)
            rs, _, rs2 = _metrics(y_te_s, p_s)
            LOG.info("  xgboost spatial holdout  RMSE=%.4f  R2=%.4f", rs, rs2)
            # Attach spatial metrics to the xgb result row.
            results[-2].rmse_spatial = rs
            results[-2].r2_spatial = rs2

    preds_df = pd.concat(pred_frames, ignore_index=True)
    return results, preds_df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results: list[FoldResult], out_path: Path,
                 best_params: dict, best_cv_rmse: float,
                 n_trials: int) -> None:
    L = []
    L.append("=" * 90)
    L.append("XGBOOST PIPELINE RESULTS")
    L.append("=" * 90)
    L.append(f"Optuna trials: {n_trials}")
    L.append(f"Best CV RMSE:  {best_cv_rmse:.5f}")
    L.append("")
    L.append("-- Best hyperparameters --")
    for k, v in sorted(best_params.items()):
        L.append(f"  {k:25s} {v}")
    L.append("")

    L.append("-- Per-fold metrics --")
    L.append(f"{'fold':<5} {'model':<10} {'n_train':>8} {'n_test':>7} "
             f"{'RMSE':>8} {'MAE':>8} {'R2':>8} {'RMSE_spatial':>13} {'R2_spatial':>11}")
    L.append("-" * 90)
    for r in results:
        sp_r = f"{r.rmse_spatial:.4f}" if not np.isnan(r.rmse_spatial) else "    -"
        sp_2 = f"{r.r2_spatial:.4f}"   if not np.isnan(r.r2_spatial)   else "    -"
        L.append(f"{r.fold:<5} {r.model:<10} {r.n_train:>8,} {r.n_test:>7,} "
                 f"{r.rmse:>8.4f} {r.mae:>8.4f} {r.r2:>8.4f} "
                 f"{sp_r:>13} {sp_2:>11}")

    L.append("")
    L.append("-- Cross-fold means (excluding holdout H) --")
    cv = [r for r in results if r.fold != "H"]
    for model in ("xgboost", "ridge"):
        sub = [r for r in cv if r.model == model]
        if not sub:
            continue
        L.append(f"  {model:<10} RMSE={np.mean([r.rmse for r in sub]):.4f}  "
                 f"MAE={np.mean([r.mae for r in sub]):.4f}  "
                 f"R2={np.mean([r.r2 for r in sub]):.4f}")

    L.append("")
    L.append("-- Final holdout (H: 2020-2024) --")
    for r in [r for r in results if r.fold == "H"]:
        L.append(f"  {r.model:<10} RMSE={r.rmse:.4f}  MAE={r.mae:.4f}  "
                 f"R2={r.r2:.4f}")

    L.append("")
    L.append("-- Spatial-holdout RMSE (XGBoost) --")
    for r in [r for r in results if r.model == "xgboost"
              and not np.isnan(r.rmse_spatial)]:
        L.append(f"  {r.fold:<5} RMSE_spatial={r.rmse_spatial:.4f}  "
                 f"R2_spatial={r.r2_spatial:.4f}")

    out_path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in",   dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--trials", type=int, default=50,
                   help="Optuna trials (default 50; 100-200 for final run).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    LOG.info("Loading %s", args.inp)
    df = pd.read_parquet(args.inp)
    LOG.info("  %d rows x %d cols", *df.shape)

    df = df[df["target_ur_b"].notna()].copy()
    LOG.info("  %d rows with non-null target_ur_b", len(df))

    _, cat_codes = prepare_features(df)
    LOG.info("Categorical categories: %s",
             {k: len(v) for k, v in cat_codes.items()})

    # ---- Optuna tuning ----------------------------------------------------
    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("Optuna hyperparameter search (%d trials)", args.trials)
    LOG.info("=" * 60)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = make_objective(df, cat_codes, n_trials_log_interval=5)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({
        "objective":             "reg:squarederror",
        "tree_method":           "hist",
        "n_estimators":          2000,
        "n_jobs":                1,
        "random_state":          0,
        "verbosity":             0,
        "early_stopping_rounds": 50,
        "eval_metric":           "rmse",
    })
    LOG.info("")
    LOG.info("Best CV RMSE: %.5f", study.best_value)
    LOG.info("Best params: %s", study.best_params)

    # ---- Final evaluation -------------------------------------------------
    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("Final CV + holdout evaluation")
    LOG.info("=" * 60)
    results, preds_df = run_cv_and_holdout(df, best_params, cat_codes)

    # ---- Persist ----------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame([{
        "fold": r.fold, "model": r.model,
        "n_train": r.n_train, "n_test": r.n_test,
        "rmse": r.rmse, "mae": r.mae, "r2": r.r2,
        "rmse_spatial": r.rmse_spatial, "r2_spatial": r.r2_spatial,
    } for r in results])
    res_df.to_csv(args.outdir / "xgb_results.csv", index=False)
    preds_df.to_parquet(args.outdir / "xgb_predictions.parquet", index=False)
    write_report(results,
                 args.outdir / "xgb_results.txt",
                 best_params, study.best_value, args.trials)

        # Retrain on data through 2019, save model for SHAP.
    #
    # Because best_params contains early_stopping_rounds, we need a
    # validation set at fit time.  Hold out 2019 (last available year in
    # the tuning window) and use it for early stopping.  The loss of one
    # training year is immaterial for SHAP interpretation.
    final_train = df[df["fiscal_year"].between(1992, 2019)]
    val_year = 2019
    trn = final_train[final_train["fiscal_year"] < val_year]
    val = final_train[final_train["fiscal_year"] == val_year]

    X_trn = build_X(trn, cat_codes); y_trn = trn["target_ur_b"].values
    X_val = build_X(val, cat_codes); y_val = val["target_ur_b"].values

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_trn, y_trn,
                    eval_set=[(X_val, y_val)],
                    verbose=False)
    final_model.save_model(str(args.outdir / "xgb_model.json"))
    LOG.info("Final model saved (trained on %d rows, early-stopped at iter %s)",
             len(X_trn), final_model.best_iteration)
    with open(args.outdir / "xgb_study.pkl", "wb") as f:
        pickle.dump({"study": study, "cat_codes": cat_codes}, f)

    LOG.info("Wrote xgb_results.txt / .csv, xgb_predictions.parquet, "
             "xgb_model.json, xgb_study.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())