```markdown
# NTA Fiscal Panel — Dataset README

**Project:** Predicting Fiscal Underutilization in Philippine Local Governments  
**Panel:** LGU × fiscal year, 1992–2024  
**Unit of observation:** (psgc10, fiscal_year)  
**Row count (final ML panel):** 55,904  
**LGU count:** 1,724 (149 cities, 1,493 municipalities, 82 provinces)  
**Regions:** 18 (16 administrative regions + BARMM + NCR)

---

## Table of contents

1. [Dataset overview](#1-dataset-overview)
2. [Pipeline build order](#2-pipeline-build-order)
3. [Directory layout](#3-directory-layout)
4. [Master reference tables](#4-master-reference-tables)
5. [Source panels](#5-source-panels)
6. [Unified panel](#6-unified-panel)
7. [ML feature matrix](#7-ml-feature-matrix)
8. [Model outputs](#8-model-outputs)
9. [Known caveats and gotchas](#9-known-caveats-and-gotchas)
10. [Replication instructions](#10-replication-instructions)
11. [Citation and license](#11-citation-and-license)

---

## 1. Dataset overview

This dataset is an LGU-year panel assembled from four primary sources:

| Source | Provider | Coverage | Content |
|---|---|---|---|
| **Statement of Receipts and Expenditures (SRE)** | BLGF | 1992–2024 | Fiscal: revenues, expenditures, fund balances |
| **Philippine Standard Geographic Code (PSGC)** | PSA | 2Q 2026 vintage | LGU identity, income class, urban/rural, 2024 population |
| **Small Area Estimates (SAE) poverty incidence** | PSA | 2018, 2021, 2023 | Poverty incidence, CV, SE, CI |
| **Population and land area** | PSA PSY 2025 | 1992–2024 | Annual LGU population; land area |
| **Consumer Price Index (CPI)** | PSA OpenSTAT | 1992–2024 | Regional annual CPI deflator (2018=100) |

The final analytic panel (`panel_ml.parquet`) carries the UR target and ML feature matrix used in the paper. The core target variable is:

```
ur_b = 1 − (total_current_operating_exp + total_capital_investment_exp) / total_current_operating_income
```

winsorized to [0, 1]. For SIE-era rows (2001–2008) where the CO/Capex split is not populated by the harmonizer, the numerator falls back to `total_expenditures`. See [§9](#9-known-caveats-and-gotchas).

---

## 2. Pipeline build order

Scripts must be run in this order. Each step's output feeds the next.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  1. psgc_harmonizer.py    →  psgc_lgu_master.parquet                 │
   │     (raw PSGC workbook → canonical LGU reference)                    │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │  2. SRE_harmonizer.py     →  sre_panel.parquet                       │
   │     (raw BLGF BOS/SIE/SRE workbooks → canonical fiscal schema)       │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │  3. sre_psgc_crosswalk.py →  sre_panel_with_psgc.parquet             │
   │     (attach psgc10 to every SRE row via tiered fuzzy matching)       │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  4. load_psa_sae.py       →  psa_panel.parquet                       │
   │     (raw SAE workbook → tidy long poverty panel)                     │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │  5. psa_harmonizer.py     →  psa_panel_with_psgc.parquet             │
   │     (attach psgc10 to every SAE row via tiered fuzzy matching)       │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  6. fetch_population.py   →  population_lgu_annual.parquet           │
   │     (PSY 2025 CSVs → annual population + land area)                  │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  7. fetch_cpi.py          →  cpi_regional.parquet                    │
   │     (PSA OpenSTAT CSVs → annual regional CPI deflator)               │
   └──────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  8. build_panel.py        →  panel.parquet                           │
   │     (merge SRE + PSGC master + SAE poverty; base frame is SRE)       │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │  9. build_features.py     →  panel_ml.parquet                        │
   │     (merge population; construct UR targets, lags, ratios, logs)     │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 10. baseline_models.py    →  baseline_results.txt                    │
   │     (naive persistence + Ridge + RF + LGBM on TS-CV folds)           │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 11. xgboost_pipeline.py   →  xgb_model.json, xgb_results.txt         │
   │     (Optuna + TS-CV + spatial holdout)                               │
   └──────────────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────────────┐
   │ 12. shap_report.py        →  shap_*.txt, shap_values.parquet         │
   │     (TreeSHAP global, per-group, cluster analysis)                   │
   └──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory layout

```
data/
├── PSA/                                   raw inputs (not tracked in git)
│   ├── PSGC-2Q-2026-Publication-Datafile.xlsx
│   ├── 2_2023 SAE_with PSGC_noHUC_06Feb2026.xlsx
│   ├── 2025_T1_*.csv                      PSA PSY 2025 Tables 1.x
│   └── Consumer Price Index*.csv          PSA OpenSTAT CPI
├── SRE/                                   raw inputs (not tracked in git)
│   └── By-LGU-SRE-YYYY.xlsx               1992–2024 BLGF SRE/BOS/SIE
└── processed/                             all derived files
    ├── psgc_lgu_master.parquet
    ├── psgc_regions_master.parquet
    ├── psgc_provinces_master.parquet
    ├── psgc_cities_master.parquet
    ├── psgc_municipalities_master.parquet
    ├── psgc_submunicipalities_master.parquet
    ├── psgc_barangays_master.parquet
    ├── psgc_lgu_master.csv
    ├── psa_panel.parquet
    ├── psa_panel_with_psgc.parquet
    ├── sre_panel.parquet
    ├── sre_panel_with_psgc.parquet
    ├── population_lgu_annual.parquet
    ├── cpi_regional.parquet
    ├── panel.parquet
    ├── panel_ml.parquet
    ├── panel_ml_qa.txt
    ├── baseline_results.txt / .csv
    ├── baseline_predictions.parquet
    ├── xgb_model.json
    ├── xgb_study.pkl
    ├── xgb_results.txt / .csv
    ├── xgb_predictions.parquet
    ├── shap_global.txt
    ├── shap_by_group.txt
    ├── shap_clusters.txt
    ├── shap_values.parquet
    ├── shap_lgu_summary.parquet
    └── *.log                              per-script build logs
```

---

## 4. Master reference tables

### `psgc_lgu_master.parquet` — canonical LGU reference

**Rows:** 1,724 (82 provinces + 149 cities + 1,493 municipalities)  
**Primary key:** `psgc10`

| Column | Type | Description |
|---|---|---|
| `psgc10` | string(10) | 10-digit PSGC code. Primary key. |
| `psgc9` | string(9) | Legacy 9-digit correspondence code. |
| `level` | string | `province` \| `city` \| `municipality` |
| `name` | string | Official PSA name, e.g. `"City of Manila"`, `"Naga"` |
| `name_short` | string | Name with prefixes/suffixes stripped: `"Manila"`, `"Naga"` |
| `match_key` | string | Aggressive normalization (ASCII, lowercase, no punct) for fuzzy joins |
| `match_key_short` | string | `match_key` of `name_short` |
| `region_code` | string(10) | Region PSGC (first 2 digits + 8 zeros) |
| `region_name` | string | Canonical region name, e.g. `"Region I (Ilocos Region)"` |
| `province_code` | string(10) | Parent province PSGC. **Empty for provinces and HUCs/ICCs.** |
| `province_name` | string | Parent province name (or own name for provinces) |
| `is_independent_city` | bool | TRUE if city has no parent province (HUC or ICC) |
| `city_class` | string | `"HUC"` \| `"ICC"` \| `"Component"` \| NaN |
| `income_class` | string | `"1st"` … `"6th"` \| `"Special"` \| NaN |
| `urban_rural` | string | **Entirely NaN in the 2Q 2026 publication.** See §9. |
| `pop_2024` | Int64 | 2024 population from PSGC publication |
| `status` | string | PSA status field (rarely populated) |
| `old_names` | string | Historical names (rarely populated) |

---

## 5. Source panels

### `psa_panel_with_psgc.parquet` — poverty SAE

**Rows:** ~5,000 (one per LGU × 3 SAE years, plus 14 Manila districts before collapse)  
**Long format:** each row is a (LGU, SAE year) observation

| Column | Type | Description |
|---|---|---|
| `lgu_name` | string | LGU name as printed in SAE workbook |
| `province` | string | Province label from SAE workbook |
| `region` | string | Region label from SAE workbook |
| `lgu_type` | string | Inferred `city` \| `municipality` |
| `psgc10` | string(10) | Attached by `psa_harmonizer.py` |
| `pi_YYYY` | float | Poverty incidence for SAE year YYYY (2018/2021/2023) |
| `cv_YYYY` | float | Coefficient of variation |
| `se_YYYY` | float | Standard error |
| `ci_lo_YYYY` | float | Lower 95% confidence bound |
| `ci_hi_YYYY` | float | Upper 95% confidence bound |
| `matched` | bool | Did the harmonizer attach a psgc10? |
| `tier` | int | Match tier: 1=exact, 2=global-unique, 3=fuzzy-province, 4=global-fuzzy, 5=loose-fuzzy |

### `sre_panel_with_psgc.parquet` — SRE fiscal panel

**Rows:** ~55,000 LGU-years  
**Coverage:** 1992–2024, with three template eras (see §9)

| Column | Type | Description |
|---|---|---|
| `region` | string | Region label from SRE file |
| `province` | string | Province label from SRE file |
| `lgu_name` | string | LGU name from SRE file |
| `lgu_type` | string | `province` \| `city` \| `municipality` |
| `fiscal_year` | int | Fiscal year |
| `fund_type` | string | `"General Fund"` \| `"SEF"` \| etc. |
| `template_era` | string | `BOS_1992_2000` \| `SIE_2001_2008` \| `SRE_2009_2015` \| `SRE_2016_2017` \| `SRE_2018_2021` \| `SRE_2022_2024` |
| `psgc10` | string(10) | Attached by `sre_psgc_crosswalk.py` |
| `tier`, `score` | int, float | Match tier and fuzzy score |

**All fiscal columns are in millions of pesos (₱M).** The harmonizer divides by 10⁶ when it detects pesos.

#### Income columns

| Column | Description |
|---|---|
| `rpt` | Real property tax, total |
| `rpt_general_fund` | RPT accruing to General Fund |
| `rpt_sef` | RPT accruing to Special Education Fund |
| `business_tax` | Tax on business |
| `other_taxes` | Other local taxes |
| `total_tax_revenue` | Sum of local tax revenue |
| `regulatory_fees` | Regulatory fees (permits, licenses) |
| `service_charges` | Service / user charges |
| `econ_enterprise` | Receipts from economic enterprises |
| `other_non_tax` | Other non-tax receipts |
| `total_non_tax` | Sum of non-tax revenue |
| `total_local_sources` | Sum of local tax + non-tax |
| `nta_ira` | **National Tax Allotment (formerly IRA)** |
| `other_national_shares` | Other national wealth shares |
| `interlocal_transfers` | Transfers from other LGUs |
| `extraordinary_aids` | Grants, donations, aids from national |
| `total_external_sources` | Sum of all external transfers |
| `total_current_operating_income` | **TCOI** = local + external sources |

#### Expenditure columns

| Column | Description |
|---|---|
| `gps` | General public services |
| `education` | Education, culture, sports, manpower |
| `health` | Health, nutrition, population control |
| `labor` | Labor and employment |
| `housing` | Housing and community development |
| `social_welfare` | Social services and welfare |
| `total_social_services` | Sum of social sector |
| `economic_services` | Economic development |
| `debt_service_interest` | Debt service — interest |
| `other_current_exp` | Other current operating expenditures |
| `other_current_exp_2` | Additional other current expenditures |
| `total_current_operating_exp` | **Total CO expenditures** |

#### Non-income receipts

| Column | Description |
|---|---|
| `proceeds_sale_assets` | Proceeds from sale of assets |
| `proceeds_sale_debt_securities` | Proceeds from sale of debt securities |
| `collection_loans_receivables` | Collection of loan receivables |
| `total_capital_investment_receipts` | Total capital / investment receipts |
| `acquisition_loans` | Acquisition of loans |
| `issuance_bonds` | Issuance of bonds |
| `total_receipts_loans` | Total receipts from loans and borrowings |
| `other_non_income_receipts` | Other non-income receipts |
| `total_non_income_receipts` | Total non-income receipts |

#### Non-operating expenditures

| Column | Description |
|---|---|
| `capex_ppe` | Purchase/construction of PP&E |
| `investment_outlay_debt` | Purchase of debt securities of other entities |
| `investment_outlay_loans` | Grants / loans to other entities |
| `total_capital_investment_exp` | **Total capital / investment expenditures** |
| `payment_loan_amortization` | Payment of loan amortization |
| `retirement_bonds` | Retirement / redemption of bonds |
| `debt_service_principal` | Debt service — principal |
| `other_non_operating_exp` | Other non-operating |
| `total_non_operating_exp` | Total non-operating |

#### Fund balance / cash

| Column | Description |
|---|---|
| `net_operating_income` | Net operating income / (loss) from current ops |
| `net_increase_decrease_funds` | Net increase / (decrease) in funds |
| `cash_balance_beginning` | Cash balance, beginning of period |
| `fund_cash_available` | **Fund / cash available for operations** |
| `payment_prior_ap` | Less: payment of prior years' accounts payable |
| `continuing_appropriation` | Continuing appropriation |
| `fund_cash_balance_end` | Fund / cash balance, end of period |

#### Auxiliary

| Column | Description |
|---|---|
| `total_expenditures` | **Top-line total expenditures** (present across all eras) |

### `population_lgu_annual.parquet` — population and land area

**Rows:** 56,636 (1,724 LGUs × 33 years, less than full coverage for 8 SGA municipalities)  
**Key:** (psgc10, fiscal_year)

| Column | Type | Description |
|---|---|---|
| `psgc10` | string(10) | LGU PSGC |
| `fiscal_year` | int | Year |
| `population` | float | Population estimate (interpolated between census years 2000, 2007, 2010, 2015, 2020, 2024) |
| `land_area_sqkm` | float | Land area in square kilometers (province-level for cities/municipalities) |
| `pop_source` | string | One of: `province_direct`, `city_direct`, `lgu_direct`, `province_allocated`, `master_2024_only` |

### `cpi_regional.parquet` — CPI deflator

**Rows:** 594 (18 regions × 33 years)

| Column | Type | Description |
|---|---|---|
| `region` | string | Canonical region name (Region I … Region XIII, NCR, CAR, BARMM, Philippines) |
| `fiscal_year` | int | Year |
| `cpi` | float | Consumer Price Index (2018=100), annual average |
| `deflator_2024` | float | `cpi / cpi_2024`. Multiply 1992-peso amount by this to get 2024-real pesos. |

---

## 6. Unified panel

### `panel.parquet` — SRE + PSGC + poverty

**Rows:** 55,904  
**Cols:** 91  
**Key:** (psgc10, fiscal_year)

Built by `build_panel.py`. Base frame is the SRE panel; rows that failed to match a PSGC are dropped.

**Identifier columns** (authoritative values come from the PSGC master, not SRE):

| Column | Type | Description |
|---|---|---|
| `psgc10` | string(10) | LGU PSGC. Primary key (with `fiscal_year`). |
| `lgu_name` | string | Canonical name from master |
| `province` | string | Parent province from master |
| `region` | string | Region from master |
| `lgu_type` | string | `province` \| `city` \| `municipality` |
| `income_class` | string | `"1st"` … `"6th"` \| `"Special"` \| NaN |
| `city_class` | string | `"HUC"` \| `"ICC"` \| `"Component"` \| NaN |
| `urban_rural` | string | NaN (unused) |
| `is_independent_city` | bool | HUC / ICC flag |
| `pop_2024` | Int64 | 2024 population from master |
| `fiscal_year` | int | Year |
| `fund_type` | string | From SRE |
| `template_era` | string | SRE template era |
| `in_sre` | bool | Membership flag |
| `in_psa` | bool | Membership flag |

**Fiscal columns:** all columns from §5 SRE panel above, at their canonical names.

**Poverty columns:**

| Column | Type | Description |
|---|---|---|
| `poverty_pi` | float | Poverty incidence — **non-null only for fiscal_year ∈ {2018, 2021, 2023}** |
| `poverty_cv`, `poverty_se`, `poverty_ci_lo`, `poverty_ci_hi` | float | Corresponding SAE statistics |
| `pi_2018`, `pi_2021`, `pi_2023` | float | Broadcast form — non-null on every row of the same psgc10 |
| `cv_YYYY`, `se_YYYY`, `ci_lo_YYYY`, `ci_hi_YYYY` | float | Broadcast form |

---

## 7. ML feature matrix

### `panel_ml.parquet` — target + features

**Rows:** 55,904  
**Cols:** 128  
**Key:** (psgc10, fiscal_year)

Built by `build_features.py`. **This is the input to the modelling scripts.**

#### Target and current-year UR

| Column | Type | Description |
|---|---|---|
| `ur_a` | float ∈ [0, 1] | Fund-based UR = `1 − spent / fund_cash_available`. **Only populated 2010+ (~48%)** |
| `ur_b` | float ∈ [0, 1] | **Primary target.** Income-based UR = `1 − spent / total_current_operating_income`. ~98.9% populated |
| `ur_a_current` | float | Same-year alias, used by naive persistence baseline and as an ML feature |
| `ur_b_current` | float | Same-year alias |
| `target_ur_a` | float | `ur_a` shifted 1 year forward. Prediction target. |
| `target_ur_b` | float | **Primary prediction target** = `ur_b` at t+1 |
| `spent_used` | float (₱M) | Numerator actually used: CO+Capex split when available, else `total_expenditures` |
| `spent_source` | string | `"detailed"` \| `"total"` — which numerator was used |

#### Lagged features (available at time t for predicting t+1)

| Column | Description |
|---|---|
| `ur_a_lag1`, `ur_a_lag2` | UR_a at t−1, t−2 |
| `ur_b_lag1`, `ur_b_lag2` | UR_b at t−1, t−2 |
| `nta_ira_lag1`, `nta_ira_lag2` | NTA at t−1, t−2 |
| `fund_cash_balance_end_lag1`, `fund_cash_balance_end_lag2` | Fund balance at t−1, t−2 |
| `total_current_operating_income_lag1`, `total_current_operating_income_lag2` | TCOI at t−1, t−2 |

#### Log features

All are `log(1 + x)` where x is clipped to non-negative:

| Column | Description |
|---|---|
| `log_nta_ira` | log NTA received |
| `log_total_current_operating_income` | log TCOI |
| `log_total_local_sources` | log local revenue |
| `log_total_external_sources` | log external transfers |
| `log_fund_cash_balance_end` | log ending fund balance |
| `log_total_capital_investment_exp` | log capex |
| `log_population` | log population |
| `log_land_area_sqkm` | log land area |

#### Ratio features

| Column | Description |
|---|---|
| `nta_dependency` | `nta_ira / total_current_operating_income` |
| `local_share` | `total_local_sources / TCOI` |
| `external_share` | `total_external_sources / TCOI` |
| `fund_balance_share` | `fund_cash_balance_end / TCOI` |
| `capex_share_avail` | `total_capital_investment_exp / fund_cash_available` |
| `revenue_hhi` | Herfindahl index on 6 local revenue components |

#### Structural features

| Column | Description |
|---|---|
| `population` | From `population_lgu_annual.parquet` |
| `land_area_sqkm` | " |
| `pop_density` | `population / land_area_sqkm` |
| `election_year` | 1 if national election year (1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025) |
| `income_class_num` | Parsed numeric income class 1–6 (`"1st"` → 1). Null for `"Special"`. |
| `fiscal_year` | Year |
| `region`, `lgu_type`, `urban_rural` | Categorical identifiers |

---

## 8. Model outputs

### `baseline_results.txt` / `.csv`

Per-fold RMSE / MAE / R² for four baselines: `naive_persistence`, `fold_mean`, `zero`, `ridge`, `random_forest`, `lightgbm`.

### `xgb_results.txt` / `.csv`

Per-fold metrics for `xgboost` and `ridge` on the same 6 folds (F1–F5 + final holdout H), plus **spatial-holdout** RMSE / R² for XGBoost.

### `xgb_model.json`

Final XGBoost model (trained on 1992–2019, early-stopped on 2019). Load with:

```python
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model("data/processed/xgb_model.json")
```

### `xgb_study.pkl`

Pickle containing the Optuna study object and the `cat_codes` dictionary (mapping for `lgu_type`, `region`, `urban_rural` category codes).

### `xgb_predictions.parquet`

Per-row predictions on CV and holdout folds.

| Column | Description |
|---|---|
| `psgc10`, `fiscal_year` | Row identifier |
| `fold` | F1, F2, F3, F4, F5, or H |
| `model` | `xgboost` \| `ridge` |
| `y_true` | Actual `target_ur_b` |
| `y_pred` | Model prediction |

### `shap_values.parquet`

**Rows:** 53,650 (every row with non-null `target_ur_b`)  
**Cols:** metadata + 17 SHAP columns

| Column | Description |
|---|---|
| `psgc10`, `lgu_name`, `meta_fiscal_year`, `meta_lgu_type`, `meta_region` | Row identifier |
| `meta_ur_b`, `meta_target_ur_b` | Target values |
| `ur_b_current`, `ur_b_lag1`, `ur_b_lag2` | SHAP contributions (not raw features) |
| `log_nta_ira`, `log_total_local_sources`, `nta_dependency`, `revenue_hhi` | " |
| `income_class_num`, `log_population`, `log_land_area_sqkm`, `pop_density` | " |
| `fiscal_year`, `election_year`, `pi_2018` | " |
| `lgu_type`, `region`, `urban_rural` | " |

Each column is the SHAP contribution of that feature to that row's prediction.

### `shap_lgu_summary.parquet`

One row per LGU (1,724 rows). Aggregated SHAP signatures and cluster labels.

| Column | Description |
|---|---|
| `psgc10` | LGU identifier |
| `cluster` | k-means cluster label (0–4, k=5) |
| `lgu_name`, `lgu_type`, `region`, `income_class` | Descriptive attributes |
| `mean_ur` | LGU's mean UR_b across all years |
| `mean_nta_dep` | LGU's mean NTA dependency |
| `mean_pop` | LGU's mean log population |
| `mean_year` | LGU's mean observation year |

### Diagnostic reports

| File | Contents |
|---|---|
| `population_coverage.txt` | Match rates for population fetch |
| `panel_qa.txt` | Row counts, coverage by year, poverty match rate |
| `panel_ml_qa.txt` | UR coverage, UR by year / decade / era, feature non-null rates |
| `shap_global.txt` | Global mean \|SHAP\| ranking of all 17 features |
| `shap_by_group.txt` | Top-10 features for each LGU type, income bucket, region, era |
| `shap_clusters.txt` | Cluster sizes, signatures, representative LGUs |

---

## 9. Known caveats and gotchas

**Read this section before using the data.** Each item is a real limitation discovered during the build.

### 9.1 SRE template eras

BLGF changed its reporting template three times: BOS (1992–2000), SIE (2001–2008), SRE (2009–2024). The `template_era` column records which template each row came from. The harmonizer attempts to map each template's labels onto the canonical schema, but coverage of specific line items varies by era:

| Column | 1992–2000 | 2001–2008 | 2009–2024 |
|---|---|---|---|
| `total_current_operating_income` | ✓ | ✓ | ✓ |
| `total_expenditures` | ✓ | ✓ | ✓ |
| `total_current_operating_exp` | partial | ✗ | ✓ |
| `total_capital_investment_exp` | partial | ✗ | ✓ |
| `fund_cash_available` | ✗ | ✗ | ~2010+ |

### 9.2 UR numerator fallback for SIE rows

Because `total_current_operating_exp` and `total_capital_investment_exp` are entirely unpopulated for 2001–2008, the UR numerator falls back to `total_expenditures` for those years. This makes 2001–2008 UR_b **not strictly definitionally comparable** to UR_b in other eras. The `spent_source` column records which numerator was used, so analysts can filter or control for this.

### 9.3 UR definitions

- **`ur_a` (fund-based)** is only usable from ~2010 onward. It is a robustness check, not a primary target.
- **`ur_b` (income-based)** is the primary target and is populated for ~98.9% of LGU-years.

### 9.4 BARMM underutilization anomaly

BARMM municipalities report a mean `ur_b` of ~0.05–0.07 across all years, well below the national municipality median of ~0.14. The pattern is not explained by income class. Two interpretations:

1. **Genuine behavior** — small, low-capacity municipalities that spend their NTA on personnel and MOOE the same year they receive it.
2. **Reporting regime difference** — own-source revenue and fund balances are under-reported in BARMM.

The paper does not adjudicate between these but flags BARMM rows in the targeting framework as requiring SRE verification before intervention. Use the `region` column to filter if needed.

### 9.5 `urban_rural` is unused

The 2Q 2026 PSGC publication does not populate the `urban_rural` field. Every value in the panel is NaN. The feature slot exists in the model configuration but contributes zero SHAP.

### 9.6 `income_class_num` requires parsing

The master stores income class as strings (`"1st"`, `"2nd"`, …, `"Special"`). The `income_class_num` column in `panel_ml.parquet` is the numeric version (1–6, NaN for Special). Verify it is non-null before using it as a feature — there was a version where it was accidentally all-NaN due to a `to_numeric` on string input.

### 9.7 Eight SGA municipalities have only 2024 data

The 8 BARMM Special Geographic Area municipalities created in 2023 (PSGC 1999901000–1999908000) have exactly one row each in the panel: fiscal_year 2024. Any panel specification requiring balanced coverage across years should drop them or model them as entering in 2024.

### 9.8 Two pre-2022 BARMM provinces remain unmatched

`Maguindanao (excluding Cotabato City)` and `Special Geographic Area` do not map to any 2024 PSGC province. They are dropped from the panel. Total population impact: <1% of national.

### 9.9 Fiscal-year effect is strong

`fiscal_year` ranks #2 in SHAP importance (21.9% of mean |SHAP|). The model is fitting a year effect, not just an LGU effect. Predictions are conditioned on the year the prediction was made; an "at-risk LGU" list from 2018 will differ from one for 2023, even holding structural features fixed.

### 9.10 The model is an autocorrelation-dominated predictor

Autocorrelation features (`ur_b_current`, `ur_b_lag1`, `ur_b_lag2`) account for 63% of SHAP signal; fiscal-year features account for another 25%; all structural features combined account for ~11%. The model beats naive persistence by 21% on the final holdout, but the win comes largely from temporal structure rather than cross-sectional LGU characteristics. Section 5.6 of the paper should reflect this.

### 9.11 All monetary values are in millions of pesos

The SRE harmonizer divides raw values by 10⁶ when it detects pesos. A value of `100.0` in `nta_ira` means ₱100 million. Do not multiply by 10⁶ again.

### 9.12 Poverty incidence is only available for 3 years

`pi_2018`, `pi_2021`, `pi_2023` exist; there is no poverty data for any other year. The broadcast form (`pi_2018` on every row of the same LGU) treats PI as a time-invariant LGU characteristic, which is a deliberate simplification — poverty moves slowly and the 2018 snapshot is used as a structural proxy in the model.

---

## 10. Replication instructions

```powershell
# 0. Setup (once)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy pyarrow scikit-learn xgboost lightgbm optuna openpyxl

# 1. Reference tables
python src\psgc_harmonizer.py

# 2. SRE harmonization + PSGC attachment
python src\SRE_harmonizer.py --validate
python src\sre_psgc_crosswalk.py --diagnose

# 3. PSA SAE + PSGC attachment
python src\load_psa_sae.py
python src\psa_harmonizer.py

# 4. External data
python src\fetch_population.py
python src\fetch_cpi.py

# 5. Unified panel
python src\build_panel.py

# 6. ML features
python -u src\build_features.py

# 7. Baselines (optional reality check, ~5 min)
python -u src\baseline_models.py

# 8. Model + tuning (~50 min at --trials 100)
python -u src\xgboost_pipeline.py --trials 100

# 9. SHAP interpretation
python -u src\shap_report.py
```

**Windows note:** use `python -u` (unbuffered) to see log output live. `n_jobs=-1` triggers a joblib deadlock on Windows under sklearn's `Pipeline`; use `n_jobs=1`.

---

## 11. Citation and license

**Data sources:**

- Bureau of Local Government Finance (BLGF). *Statement of Receipts and Expenditures, 1992–2024.* Republic of the Philippines.
- Philippine Statistics Authority (PSA). *Philippine Standard Geographic Code, 2Q 2026 Publication.*
- Philippine Statistics Authority (PSA). *Small Area Estimates of Poverty Incidence, 2018/2021/2023.*
- Philippine Statistics Authority (PSA). *2025 Philippine Statistical Yearbook, Tables 1.x.*
- Philippine Statistics Authority (PSA). *Consumer Price Index by Region, 1992–2024 (2018=100).*

**This dataset:** Aggregated LGU-level fiscal and socioeconomic statistics. No individual-level or personally identifiable information. Redistribution subject to the terms of the upstream data providers.

**Contact:** jjlumingkit@up.edu.ph

---

*Generated: 2026-09-24. Pipeline version: NTA-Fiscal v0.1.*
```
