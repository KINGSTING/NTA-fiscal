"""
sre_harmonizer.py
=================
Harmonization engine for Philippine LGU fiscal data (1992–2024).

Transforms BOS (1992–2000), SIE (2001–2008), and SRE (2009–2024)
into a single canonical schema suitable for panel analysis.

Usage:
    # Inspect a raw file
    python src/sre_harmonizer.py --inspect data/SRE/By-LGU-SRE-2024.xlsx

    # Pilot run (one year)
    python src/sre_harmonizer.py --pattern "By-LGU-SRE-2024.xlsx" ^
        --output data/processed/sre_2024.parquet --validate

    # Full batch
    python src/sre_harmonizer.py --output data/processed/sre_panel.parquet --validate
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sre_harmonizer")


# ---------------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# CANONICAL SCHEMA
# ---------------------------------------------------------------------------

IDENTIFIER_COLS = [
    "region", "province", "lgu_name", "lgu_type",
    "fiscal_year", "fund_type", "template_era",
]

INCOME_COLS = [
    "rpt", "rpt_general_fund", "rpt_sef",
    "business_tax", "other_taxes", "total_tax_revenue",
    "regulatory_fees", "service_charges", "econ_enterprise",
    "other_non_tax", "total_non_tax",
    "total_local_sources",
    "nta_ira", "other_national_shares", "interlocal_transfers",
    "extraordinary_aids", "total_external_sources",
    "total_current_operating_income",
]

EXPENDITURE_COLS = [
    "gps", "education", "health", "labor", "housing",
    "social_welfare", "total_social_services",
    "economic_services", "debt_service_interest",
    "other_current_exp", "other_current_exp_2",
    "total_current_operating_exp",
]

NON_INCOME_COLS = [
    "proceeds_sale_assets", "proceeds_sale_debt_securities",
    "collection_loans_receivables", "total_capital_investment_receipts",
    "acquisition_loans", "issuance_bonds", "total_receipts_loans",
    "other_non_income_receipts", "total_non_income_receipts",
]

NON_OPERATING_COLS = [
    "capex_ppe", "investment_outlay_debt", "investment_outlay_loans",
    "total_capital_investment_exp",
    "payment_loan_amortization", "retirement_bonds",
    "debt_service_principal", "other_non_operating_exp",
    "total_non_operating_exp",
]

FUND_BALANCE_COLS = [
    "net_operating_income", "net_increase_decrease_funds",
    "cash_balance_beginning", "fund_cash_available",
    "payment_prior_ap", "continuing_appropriation",
    "fund_cash_balance_end",
]

AUX_COLS = ["total_expenditures"]

CANONICAL_COLS = (
    IDENTIFIER_COLS + INCOME_COLS + EXPENDITURE_COLS
    + NON_INCOME_COLS + NON_OPERATING_COLS
    + FUND_BALANCE_COLS + AUX_COLS
)


# ---------------------------------------------------------------------------
# LABEL NORMALIZATION
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_label(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return _WS_RE.sub(" ", str(s)).strip().lower()


# ---------------------------------------------------------------------------
# ORDERED ERA DETECTION
# ---------------------------------------------------------------------------

ERA_RULES: list[tuple[str, str]] = [
    ("national tax allotment",                              "SRE_2022_2024"),
    ("total non-income receipts",                           "SRE_2018_2021"),
    ("other receipts (other general income)",               "SRE_2016_2017"),
    ("current operating income",                            "SRE_2009_2015"),
    ("toll fees",                                           "SIE_2001_2008"),
    ("total shares from national tax collections",          "SIE_2001_2008"),
    ("aids and allotments",                                 "BOS_1992_2000"),
]

YEAR_TO_ERA: dict[int, str] = {}
for _y in range(1992, 2001): YEAR_TO_ERA[_y] = "BOS_1992_2000"
for _y in range(2001, 2009): YEAR_TO_ERA[_y] = "SIE_2001_2008"
for _y in range(2009, 2016): YEAR_TO_ERA[_y] = "SRE_2009_2015"
for _y in range(2016, 2018): YEAR_TO_ERA[_y] = "SRE_2016_2017"
for _y in range(2018, 2022): YEAR_TO_ERA[_y] = "SRE_2018_2021"
for _y in range(2022, 2025): YEAR_TO_ERA[_y] = "SRE_2022_2024"


def detect_era(header_labels: list[str], fiscal_year: int) -> tuple[str, str]:
    haystack = " || ".join(normalize_label(h) for h in header_labels)
    for pattern, era in ERA_RULES:
        if pattern in haystack:
            return era, "auto"
    return YEAR_TO_ERA.get(fiscal_year, "SRE_2009_2015"), "year-fallback"


# ---------------------------------------------------------------------------
# COLUMN MATCHING
# ---------------------------------------------------------------------------

COLUMN_PATTERNS: list[tuple[str, str]] = [
    # RPT with fund split
    ("real property tax | general fund",           "rpt_general_fund"),
    ("real property tax | special education fund", "rpt_sef"),
    ("real property tax | total",                  "rpt"),
    ("real property tax",                          "rpt"),

    # Business tax
    ("tax on business",                            "business_tax"),
    ("business tax",                               "business_tax"),

    # Other taxes
    ("other taxes",                                "other_taxes"),
    ("total tax revenue",                          "total_tax_revenue"),

    # Non-tax revenue
    ("regulatory fees",                            "regulatory_fees"),
    ("service/ user charges",                      "service_charges"),
    ("service/user charges",                       "service_charges"),
    ("receipts from economic enterprise",          "econ_enterprise"),
    ("other receipts (other general income)",      "other_non_tax"),
    ("other receipts",                             "other_non_tax"),
    ("toll fees",                                  "other_non_tax"),
    ("total non-tax revenue",                      "total_non_tax"),

    # Local sources
    ("total local sources",                        "total_local_sources"),

    # External sources
    ("national tax allotment",                     "nta_ira"),
    ("internal revenue allotment",                 "nta_ira"),
    ("other shares from national tax collection",  "other_national_shares"),
    ("other shares",                               "other_national_shares"),
    ("inter-local transfers",                      "interlocal_transfers"),
    ("interlocal transfers",                       "interlocal_transfers"),
    ("extraordinary receipts/ grants/ donations/ aids", "extraordinary_aids"),
    ("extraordinary recipts/ aids",                "extraordinary_aids"),
    ("extraordinary receipts/ aids",               "extraordinary_aids"),
    ("national aids",                              "extraordinary_aids"),
    ("national wealth",                            "other_national_shares"),
    ("total external sources",                     "total_external_sources"),
    ("aids and allotments",                        "total_external_sources"),

    # Total income
    ("total current operating income",             "total_current_operating_income"),
    ("total income",                               "total_current_operating_income"),

    # Expenditures
    ("general public services",                    "gps"),
    ("general government",                         "gps"),
    ("education, culture & sports/ manpower development", "education"),
    ("health, nutrition & population control",     "health"),
    ("labor and employment",                       "labor"),
    ("housing and community development",          "housing"),
    ("social services and social welfare",         "social_welfare"),
    ("social security /social services & welfare", "social_welfare"),
    ("public welfare & internal safety",           "social_welfare"),
    ("total social services",                      "total_social_services"),
    ("economic services",                          "economic_services"),
    ("economic development",                       "economic_services"),
    ("debt service (interest expense & other charges)", "debt_service_interest"),
    ("debt service",                               "debt_service_interest"),
    ("operation of economic enterprise",           "other_current_exp"),
    ("other charges",                              "other_current_exp_2"),
    ("other purposes",                             "other_current_exp"),
    ("total current operating expenditures",       "total_current_operating_exp"),
    ("current expenditures",                       "total_current_operating_exp"),

    # Non-income receipts
    ("proceeds from sale of assets",               "proceeds_sale_assets"),
    ("proceeds from sale of debt securities of other entities", "proceeds_sale_debt_securities"),
    ("collection of loans receivables",            "collection_loans_receivables"),
    ("total capital/investment receipts",          "total_capital_investment_receipts"),
    ("acquisition of loans",                       "acquisition_loans"),
    ("issuance of bonds",                          "issuance_bonds"),
    ("total receipts from loans and borrowings",   "total_receipts_loans"),
    ("loans and borrowings",                       "total_receipts_loans"),
    ("loans & borrowings",                         "total_receipts_loans"),
    ("other non-income receipts",                  "other_non_income_receipts"),
    ("total non-income receipts",                  "total_non_income_receipts"),

    # Non-operating expenditures
    ("purchase/ construct of property plant and equipment", "capex_ppe"),
    ("purchase/construct of property plant and equipment",  "capex_ppe"),
    ("purchase of debt securities of other entities",       "investment_outlay_debt"),
    ("grant/ make loan to other entities",                  "investment_outlay_loans"),
    ("total capital/ investment expenditures",              "total_capital_investment_exp"),
    ("payment of loan amortization",                        "payment_loan_amortization"),
    ("retirement/ redemption of bonds/ debt securities",    "retirement_bonds"),
    ("total debt service (principal cost)",                 "debt_service_principal"),
    ("other non-operating expenditures",                    "other_non_operating_exp"),
    ("total non-operating expenditures",                    "total_non_operating_exp"),
    ("capital outlay",                                      "capex_ppe"),

    # Fund balance
    ("net operating income/ (loss) from current operations", "net_operating_income"),
    ("net increase/ (decrease) in funds",                    "net_increase_decrease_funds"),
    ("add: cash balance, beginning",                         "cash_balance_beginning"),
    ("fund/ cash available",                                 "fund_cash_available"),
    ("less: payment of prior year/s accounts payable",       "payment_prior_ap"),
    ("continuing  appropriation",                            "continuing_appropriation"),
    ("continuing appropriation",                             "continuing_appropriation"),
    ("fund/ cash balance, end",                              "fund_cash_balance_end"),

    # BOS/SIE totals
    ("total expenditures",                         "total_expenditures"),
    ("excess (deficit) of income over expenditures", "net_operating_income"),
]


IDENTIFIER_EXACT: dict[str, str] = {
    "region": "region",
    "province": "province",
    "lgu name": "lgu_name",
    "lgu type": "lgu_type",
    "fund": "fund_type",
    "fund type": "fund_type",
}


def match_canonical(flat_label: str) -> str | None:
    label = normalize_label(flat_label)
    if label in IDENTIFIER_EXACT:
        return IDENTIFIER_EXACT[label]
    for pattern, canonical in COLUMN_PATTERNS:
        if pattern in label:
            return canonical
    return None


# ---------------------------------------------------------------------------
# HEADER KEYWORDS (used by both loading and detection)
# ---------------------------------------------------------------------------

_HEADER_KEYWORDS = {
    "LGU NAME", "LGU_NAME", "LGUNAME", "NAME OF LGU",
    "NAME OF LGU (CITY/MUNICIPALITY)", "LOCAL GOVERNMENT UNIT",
}

_SKIP_ROW_PATTERNS = re.compile(
    r"^(TOTAL|SUBTOTAL|GRAND\s*TOTAL|REGION\b|"
    r"LGU\s*NAME|PROVINCE|LGU\s*TYPE)\s*$",
    re.IGNORECASE,
)


def _row_has_lgu_name(values) -> bool:
    """Return True if any value in the row looks like 'LGU NAME'."""
    for v in values:
        if pd.isna(v):
            continue
        s = re.sub(r"\s+", " ", str(v)).strip().upper()
        s = s.replace("\u00a0", " ")
        if s in _HEADER_KEYWORDS:
            return True
        if "LGU" in s and "NAME" in s and len(s) < 30:
            return True
    return False


# ---------------------------------------------------------------------------
# LOADING (multi-sheet aware)
# ---------------------------------------------------------------------------

def load_raw_with_merges(path: Path) -> pd.DataFrame:
    """
    Load an .xlsx file, expand merged cells, and return the sheet that
    contains the 'LGU NAME' header.

    Scans ALL sheets because some BLGF files have a metadata cover sheet
    as the active sheet, with the actual SRE data on a second sheet.
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True, read_only=False)

    best_sheet_name: str | None = None
    best_df: pd.DataFrame | None = None
    sheet_report: list[str] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            sheet_report.append(f"{sheet_name}:empty")
            continue

        df = pd.DataFrame(rows)

        # Expand merged cells
        for merge_range in list(ws.merged_cells.ranges):
            r1, r2 = merge_range.min_row, merge_range.max_row
            c1, c2 = merge_range.min_col, merge_range.max_col
            val = df.iat[r1 - 1, c1 - 1]
            if pd.isna(val):
                continue
            for r in range(r1 - 1, min(r2, len(df))):
                for c in range(c1 - 1, min(c2, df.shape[1])):
                    df.iat[r, c] = val

        # Check first 40 rows for LGU NAME
        has_header = False
        for i in range(min(40, len(df))):
            if _row_has_lgu_name(df.iloc[i]):
                has_header = True
                break

        marker = "HAS_LGU" if has_header else "no_lgu"
        sheet_report.append(f"{sheet_name}:{marker}({df.shape[0]}x{df.shape[1]})")

        if has_header:
            # Prefer the active sheet; otherwise take first matching sheet
            if sheet_name == wb.active.title:
                best_sheet_name = sheet_name
                best_df = df
                break
            if best_df is None:
                best_sheet_name = sheet_name
                best_df = df

    wb.close()

    log.info("  Sheets: %s", " | ".join(sheet_report))

    if best_df is None:
        raise ValueError(
            f"No sheet in {path.name} contains an 'LGU NAME' header. "
            f"Sheets: {list(wb.sheetnames)}"
        )

    log.info("  Using sheet: %s", best_sheet_name)
    return best_df


def load_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        try:
            return load_raw_with_merges(path)
        except Exception as e:
            log.warning("Multi-sheet load failed (%s); falling back to active sheet", e)
            return pd.read_excel(path, header=None, dtype=object, engine="openpyxl")
    elif suffix == ".xls":
        return pd.read_excel(path, header=None, dtype=object, engine="xlrd")
    elif suffix == ".csv":
        return pd.read_csv(path, header=None, dtype=object, low_memory=False)
    raise ValueError(f"Unsupported extension: {suffix}")


# ---------------------------------------------------------------------------
# HEADER DETECTION + FLATTENING
# ---------------------------------------------------------------------------

@dataclass
class HeaderBlock:
    header_start: int
    data_start: int
    lgu_name_col: int
    header_labels: list[str]


def detect_header_block(raw: pd.DataFrame, max_scan: int = 40) -> HeaderBlock:
    header_start: int | None = None
    lgu_name_col: int | None = None

    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i]
        for j, val in enumerate(row):
            if pd.isna(val):
                continue
            s = re.sub(r"\s+", " ", str(val)).strip().upper()
            s = s.replace("\u00a0", " ")
            if s in _HEADER_KEYWORDS:
                header_start = i
                lgu_name_col = j
                break
            if "LGU" in s and "NAME" in s and len(s) < 30:
                header_start = i
                lgu_name_col = j
                break
        if header_start is not None:
            break

    if header_start is None or lgu_name_col is None:
        raise ValueError(f"Could not locate 'LGU NAME' header in first {max_scan} rows")

    data_start: int | None = None
    for i in range(header_start + 1, len(raw)):
        val = raw.iloc[i, lgu_name_col]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s:
            continue
        upper = s.upper()
        if upper in _HEADER_KEYWORDS:
            continue
        if _SKIP_ROW_PATTERNS.match(upper):
            continue
        data_start = i
        break

    if data_start is None:
        data_start = min(header_start + 6, len(raw) - 1)
        log.warning("Data start not found; using row %d", data_start)

    header_rows = raw.iloc[header_start:data_start]

    new_cols: list[str] = []
    for col in header_rows.columns:
        fragments: list[str] = []
        for _, hrow in header_rows.iterrows():
            v = hrow[col]
            if pd.isna(v):
                continue
            sv = str(v).strip()
            if not sv:
                continue
            if not fragments or fragments[-1].lower() != sv.lower():
                fragments.append(sv)
        new_cols.append(" | ".join(fragments))

    return HeaderBlock(
        header_start=header_start,
        data_start=data_start,
        lgu_name_col=lgu_name_col,
        header_labels=new_cols,
    )


# ---------------------------------------------------------------------------
# HARMONIZATION
# ---------------------------------------------------------------------------

@dataclass
class HarmonizationReport:
    fiscal_year: int
    template_era: str
    detection_method: str
    n_rows: int
    n_lgus: int
    header_start: int
    data_start: int
    n_mapped_columns: int
    n_unmapped_columns: int
    unit_conversion: str = "none"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def harmonize_year(raw: pd.DataFrame, fiscal_year: int) -> tuple[pd.DataFrame, HarmonizationReport]:
    log.info("Harmonizing FY%d ...", fiscal_year)

    block = detect_header_block(raw)
    log.info(
        "  Header rows %d–%d, data starts at row %d, LGU col %d",
        block.header_start, block.data_start - 1,
        block.data_start, block.lgu_name_col,
    )

    body = raw.iloc[block.data_start:].copy().reset_index(drop=True)
    body.columns = block.header_labels

    era, method = detect_era(block.header_labels, fiscal_year)
    log.info("  Template era: %s (%s)", era, method)

    rename_map: dict[str, str] = {}
    unmapped: list[str] = []
    seen: set[str] = set()
    for col in body.columns:
        canonical = match_canonical(col)
        if canonical and canonical not in seen:
            rename_map[col] = canonical
            seen.add(canonical)
        else:
            if not canonical:
                unmapped.append(col)

    mapped = body.rename(columns=rename_map)

    keep = [c for c in CANONICAL_COLS if c in mapped.columns]
    mapped = mapped[keep].copy()

    if "lgu_name" not in mapped.columns:
        raise ValueError(f"FY{fiscal_year}: no lgu_name column after mapping")

    mapped = mapped.dropna(subset=["lgu_name"])
    mapped["lgu_name"] = mapped["lgu_name"].astype(str).str.strip()
    mapped = mapped[mapped["lgu_name"] != ""]
    mapped = mapped[
        ~mapped["lgu_name"].str.upper().str.match(_SKIP_ROW_PATTERNS, na=False)
    ]

    # --- Numeric conversion ---
    for col in mapped.columns:
        if col in IDENTIFIER_COLS:
            continue
        mapped[col] = pd.to_numeric(mapped[col], errors="coerce")

    # --- Unit normalization: pesos → millions ---
    unit_note = "unknown"
    if "nta_ira" in mapped.columns:
        median_nta = mapped["nta_ira"].median(skipna=True)
        if pd.notna(median_nta):
            if median_nta > 1_000_000:
                unit_note = "pesos→millions"
                log.info("  Detected PESOS units; converting to MILLIONS")
                for col in mapped.columns:
                    if col in IDENTIFIER_COLS:
                        continue
                    if pd.api.types.is_numeric_dtype(mapped[col]):
                        mapped[col] = mapped[col] / 1_000_000
            else:
                unit_note = "already millions"
                log.info("  Values already in MILLIONS; no conversion")
    else:
        unit_note = "NTA missing"
        log.warning("  NTA column missing; skipping unit normalization")

    # --- Metadata ---
    mapped["fiscal_year"] = fiscal_year
    mapped["template_era"] = era
    if "fund_type" not in mapped.columns:
        mapped["fund_type"] = "General Fund"

    for col in ("region", "province", "lgu_name", "lgu_type", "fund_type"):
        if col in mapped.columns:
            mapped[col] = mapped[col].astype(str).str.strip()

    report = HarmonizationReport(
        fiscal_year=fiscal_year,
        template_era=era,
        detection_method=method,
        n_rows=len(mapped),
        n_lgus=mapped["lgu_name"].nunique(),
        header_start=block.header_start,
        data_start=block.data_start,
        n_mapped_columns=len(rename_map),
        n_unmapped_columns=len(unmapped),
        unit_conversion=unit_note,
    )
    report.issues, report.warnings = validate_year(mapped, fiscal_year)

    if unmapped:
        log.debug("  Unmapped columns: %s", unmapped[:10])

    return mapped, report


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

BLGF_NTA_LANDMARKS: dict[int, float] = {
    2010: 265_000, 2015: 428_000, 2018: 522_000,
    2020: 575_000, 2022: 871_000, 2024: 1_006_000,
}


def validate_year(df: pd.DataFrame, fiscal_year: int) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []

    for col in ("region", "province", "lgu_name", "lgu_type"):
        if col not in df.columns:
            issues.append(f"Missing identifier: {col}")

    if "nta_ira" in df.columns:
        neg = (df["nta_ira"] < 0).sum()
        if neg:
            issues.append(f"{neg} rows with negative NTA")
        zeros = (df["nta_ira"] == 0).sum()
        if zeros:
            warnings.append(f"{zeros} rows with zero NTA (reported as 'not captured')")

    needed = {"total_local_sources", "total_external_sources",
              "total_current_operating_income"}
    if needed.issubset(df.columns):
        computed = df["total_local_sources"].fillna(0) + df["total_external_sources"].fillna(0)
        actual = df["total_current_operating_income"].fillna(0)
        diff = (computed - actual).abs()
        denom = actual.abs().replace(0, np.nan)
        mismatch = (diff / denom > 0.01).sum()
        if mismatch:
            warnings.append(f"{mismatch} rows with TCOI mismatch > 1%")

    n = df["lgu_name"].nunique() if "lgu_name" in df.columns else 0
    if n < 1000:
        issues.append(f"Only {n} unique LGUs (expected >= 1,500)")

    if "nta_ira" in df.columns and fiscal_year in BLGF_NTA_LANDMARKS:
        total_nta = df["nta_ira"].sum()
        landmark = BLGF_NTA_LANDMARKS[fiscal_year]
        if landmark > 0 and total_nta > 0:
            rel = abs(total_nta - landmark) / landmark
            if rel > 0.30:
                warnings.append(
                    f"Total NTA {total_nta:,.0f} Mn vs landmark "
                    f"{landmark:,.0f} Mn ({rel:.1%} off) — likely due to "
                    f"LGUs with missing data in source file"
                )

    return issues, warnings


# ---------------------------------------------------------------------------
# YEAR EXTRACTION
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"((?:19|20)\d{2})\s*$")


def extract_year(path: Path) -> int | None:
    stem = path.stem.strip()
    m = _YEAR_RE.search(stem)
    if m:
        return int(m.group(1))
    m = re.search(r"(19|20)\d{2}", stem)
    if m:
        return int(m.group(0))
    return None


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path, pattern: str = "*.xls*",
) -> tuple[pd.DataFrame, list[HarmonizationReport]]:

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {input_dir}")

    log.info("Found %d files in %s", len(files), input_dir)

    frames: list[pd.DataFrame] = []
    reports: list[HarmonizationReport] = []

    for path in files:
        year = extract_year(path)
        if year is None:
            log.warning("Skipping %s (no year in name)", path.name)
            continue
        try:
            raw = load_raw(path)
            clean, report = harmonize_year(raw, year)
            frames.append(clean)
            reports.append(report)
            log.info(
                "  FY%d: %d rows, %d LGUs, %d mapped, %d unmapped, "
                "%d issues, %d warnings [%s]",
                year, report.n_rows, report.n_lgus,
                report.n_mapped_columns, report.n_unmapped_columns,
                len(report.issues), len(report.warnings),
                report.unit_conversion,
            )
        except Exception as e:
            log.exception("Failed to process %s: %s", path.name, e)

    if not frames:
        raise RuntimeError("No files were processed successfully")

    panel = pd.concat(frames, ignore_index=True)
    ordered = [c for c in CANONICAL_COLS if c in panel.columns]
    extra = [c for c in panel.columns if c not in ordered]
    panel = panel[ordered + extra]

    sort_cols = [c for c in ("region", "province", "lgu_name", "fiscal_year")
                 if c in panel.columns]
    panel = panel.sort_values(sort_cols).reset_index(drop=True)
    return panel, reports


# ---------------------------------------------------------------------------
# INSPECT MODE
# ---------------------------------------------------------------------------

def inspect_file(path: Path) -> None:
    print("=" * 78)
    print(f"INSPECT: {path}")
    print("=" * 78)

    # --- Show all sheets ---
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
            wb = load_workbook(path, data_only=True, read_only=True)
            print(f"Sheets: {wb.sheetnames}")
            print(f"Active: {wb.active.title}")
            wb.close()
        except Exception as e:
            print(f"Could not list sheets: {e}")
        print()

    try:
        raw = load_raw(path)
    except Exception as e:
        print(f"load_raw failed: {e}")
        return

    print(f"Selected sheet shape: {raw.shape}")
    print()

    for i in range(min(12, len(raw))):
        row = raw.iloc[i]
        cells = [str(v)[:35] for v in row[:12] if pd.notna(v)]
        print(f"Row {i:3d}: {cells}")

    print()
    try:
        block = detect_header_block(raw)
        print(f"Header start:  {block.header_start}")
        print(f"Data start:    {block.data_start}")
        print(f"LGU NAME col:  {block.lgu_name_col}")

        era, method = detect_era(block.header_labels, 0)
        print(f"Detected era:  {era} ({method})")
        print()

        print("Column mapping preview (first 30 cols):")
        n_mapped = 0
        n_unmapped = 0
        for i, lbl in enumerate(block.header_labels[:30]):
            canonical = match_canonical(lbl)
            short = lbl[:60] + ("..." if len(lbl) > 60 else "")
            if canonical:
                print(f"  [{i:2d}] -> {canonical:<25}  <- {short}")
                n_mapped += 1
            else:
                print(f"  [{i:2d}]    (unmapped)            <- {short}")
                n_unmapped += 1
        print()
        print(f"Mapped: {n_mapped}, Unmapped: {n_unmapped} (in first 30 cols)")
    except Exception as e:
        print(f"Detection failed: {e}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report_summary(reports: list[HarmonizationReport]) -> None:
    print()
    print("=" * 118)
    print("HARMONIZATION SUMMARY")
    print("=" * 118)
    print(f"{'Year':<6} {'Era':<18} {'Method':<14} {'Rows':>7} {'LGUs':>6} "
          f"{'Mapped':>7} {'Unmap':>6} {'Units':<18} {'Issues':>7} {'Warns':>6}")
    print("-" * 118)
    for r in reports:
        print(f"{r.fiscal_year:<6} {r.template_era:<18} {r.detection_method:<14} "
              f"{r.n_rows:>7} {r.n_lgus:>6} {r.n_mapped_columns:>7} "
              f"{r.n_unmapped_columns:>6} {r.unit_conversion:<18} "
              f"{len(r.issues):>7} {len(r.warnings):>6}")
    print("=" * 118)

    for r in reports:
        if r.issues or r.warnings:
            print(f"\nFY{r.fiscal_year}:")
            for i in r.issues:
                print(f"  [ISSUE] {i}")
            for w in r.warnings:
                print(f"  [WARN]  {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Harmonize Philippine LGU fiscal data (BOS/SIE/SRE)."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/SRE"),
                        help="Directory with raw files (default: data/SRE).")
    parser.add_argument("--pattern", default="*.xls*",
                        help="Glob pattern (default: *.xls*).")
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/sre_panel.parquet"),
                        help="Output path (.parquet or .csv).")
    parser.add_argument("--validate", action="store_true",
                        help="Print detailed validation report.")
    parser.add_argument("--inspect", type=Path, default=None,
                        help="Inspect one raw file and exit.")
    args = parser.parse_args(argv)

    if args.inspect is not None:
        inspect_file(resolve_path(args.inspect))
        return 0

    panel, reports = process_directory(
        resolve_path(args.input_dir), args.pattern,
    )

    out_path = resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        panel.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        panel.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    log.info("Wrote %d rows x %d cols to %s",
             len(panel), panel.shape[1], out_path)

    if args.validate:
        _print_report_summary(reports)

    return 0


if __name__ == "__main__":
    sys.exit(main())