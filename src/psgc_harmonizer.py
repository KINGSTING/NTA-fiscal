#!/usr/bin/env python
"""
psgc_harmonizer.py
==================
Harmonize the PSA Philippine Standard Geographic Code (PSGC) publication
file into a clean, analysis-ready master of Philippine LGUs.

This is the canonical reference table that every downstream dataset
(SRE panel, PSA population, poverty SAE, DILG SGLG) will be joined to.

Inputs
------
    data/PSA/PSGC-2Q-2026-Publication-Datafile.xlsx
        - sheet "PSGC"              : master LGU list (Reg/Prov/City/Mun/SubMun/Bgy)
        - sheet "National Summary"  : region-level totals (not used, kept for reference)

Outputs
-------
    data/processed/psgc_lgu_master.parquet        <- THE master (Prov + City + Mun)
    data/processed/psgc_regions_master.parquet
    data/processed/psgc_provinces_master.parquet
    data/processed/psgc_cities_master.parquet
    data/processed/psgc_municipalities_master.parquet
    data/processed/psgc_submunicipalities_master.parquet
    data/processed/psgc_barangays_master.parquet
    data/processed/psgc_lgu_master.csv            (for eyeballing)
    data/processed/psgc_harmonizer.log

Key design choices
------------------
* 10-digit PSGC is the primary key. The 9-digit Correspondence Code is
  preserved as a secondary key because many BLGF/SRE vintages still use
  the old 9-digit format.
* Hierarchy is derived, not trusted:
      region_code   = first 2 digits + "00000000"
      province_code = first 5 digits + "00000"
  If a city's derived province_code is not present in the province list,
  the city is flagged as independent (HUC or ICC).
* Three name forms are produced:
      name            - cleaned name as printed in PSA file
      name_short      - city/mun prefix and city suffix stripped
      match_key       - aggressive normalization (ASCII, lower, no punct)
                        for fuzzy joins to SRE / BLGF / DILG
* Sub-municipalities (Manila districts) are separated out so they cannot
  be accidentally double-counted against the parent city.

Usage
-----
    python src/psgc_harmonizer.py
    python src/psgc_harmonizer.py --input path/to/file.xlsx --outdir data/processed
    python src/psgc_harmonizer.py --no-barangay    # skip 42k-row barangay file
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = Path("data/PSA/PSGC-2Q-2026-Publication-Datafile.xlsx")
DEFAULT_OUTDIR = Path("data/processed")
DEFAULT_LOG    = Path("data/processed/psgc_harmonizer.log")

SHEET_MAIN = "PSGC"

# Expected counts for QA (from PSA 2Q 2026 publication)
EXPECTED = {
    "province":     82,
    "city":         149,
    "municipality": 1493,
}

LOG = logging.getLogger("psgc_harmonizer")

# ---------------------------------------------------------------------------
# Column / level maps
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "10-digit PSGC":                                   "psgc10",
    "Name":                                            "name_raw",
    "Correspondence Code":                             "psgc9",
    "Geographic Level":                                "level",
    "Old names":                                       "old_names",
    "City Class":                                      "city_class",
    "Income Classification (DOF DO No. 074.2024)":     "income_class",
    "Urban / Rural (based on 2020 CPH)":               "urban_rural",
    "2024 Population":                                 "pop_2024",
    "Status":                                          "status",
}

LEVEL_MAP = {
    "Reg":    "region",
    "Prov":   "province",
    "City":   "city",
    "Mun":    "municipality",
    "SubMun": "submunicipality",
    "Bgy":    "barangay",
}

CITY_PREFIXES = ("City of ", "Lungsod ng ", "Dakbayan sa ")
MUN_PREFIXES  = ("Municipality of ", "Mun. of ", "Bayan ng ", "Munisipalidad ng ")
CITY_SUFFIXES = (" City",)
MUN_SUFFIXES  = (" Municipality",)

LGU_COLUMNS = [
    "psgc10", "psgc9", "level",
    "name", "name_short", "match_key", "match_key_short",
    "region_code", "region_name",
    "province_code", "province_name",
    "is_independent_city",
    "city_class", "income_class", "urban_rural",
    "pop_2024",
    "status", "old_names",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Optional[Path] = None, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s | %(levelname)-7s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)

# ---------------------------------------------------------------------------
# Name cleaning
# ---------------------------------------------------------------------------

_WS_RE       = re.compile(r"\s+")
_FOOTNOTE_RE = re.compile(r"\*+$")
_PUNCT_RE    = re.compile(r"[^a-z0-9 ]+")

def normalize_name(name) -> str:
    """NFKC, collapse whitespace, strip trailing footnote markers, fix mojibake."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name)
    s = s.replace("牋", " ").replace("\u00a0", " ")   # mojibake / NBSP
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _FOOTNOTE_RE.sub("", s).strip()
    return s


def strip_prefix(name: str) -> str:
    for p in CITY_PREFIXES + MUN_PREFIXES:
        if name.startswith(p):
            return name[len(p):].strip()
    return name


def strip_suffix(name: str) -> str:
    for sfx in CITY_SUFFIXES + MUN_SUFFIXES:
        if name.endswith(sfx):
            return name[: -len(sfx)].strip()
    return name


def make_short(name: str) -> str:
    return strip_suffix(strip_prefix(name))


def make_match_key(name: str) -> str:
    """Aggressive normalization for cross-source fuzzy joins."""
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

# ---------------------------------------------------------------------------
# Code cleaning
# ---------------------------------------------------------------------------

def clean_code_column(s: pd.Series, width: int) -> pd.Series:
    """Return a fixed-width string column of digits (or '' for missing)."""
    txt = s.astype("string").str.strip()
    txt = txt.str.replace(r"\.0+$", "", regex=True)      # Excel floats
    txt = txt.str.replace(r"[^\d]", "", regex=True)      # digits only
    txt = txt.fillna("")
    has_digits = txt.str.len().fillna(0).astype(int) > 0
    out = pd.Series("", index=s.index, dtype=object)
    out.loc[has_digits] = txt[has_digits].str.zfill(width).astype(object)
    return out

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def find_header_row(path: Path, sheet: str,
                    sentinel: str = "10-digit PSGC", scan: int = 15) -> int:
    """Locate the header row by scanning for a sentinel string."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=scan, dtype=str)
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str)
        if row.str.contains(sentinel, case=False, na=False).any():
            return i
    raise ValueError(f"Could not find header row with '{sentinel}' in sheet '{sheet}'")


def read_psgc_sheet(path: Path, sheet: str = SHEET_MAIN) -> pd.DataFrame:
    header_row = find_header_row(path, sheet)
    LOG.info("Header row detected at index %d in sheet '%s'", header_row, sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype=str)
    df = df.dropna(how="all").reset_index(drop=True)
    LOG.info("Raw shape: %s", df.shape)
    LOG.info("Raw columns: %s", list(df.columns))
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_WS_RE.sub(" ", str(c)).strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    return df

# ---------------------------------------------------------------------------
# Core harmonization
# ---------------------------------------------------------------------------

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- required columns ---
    required = ["psgc10", "name_raw", "level"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    # --- codes ---
    df["psgc10"] = clean_code_column(df["psgc10"], 10)
    if "psgc9" in df.columns:
        df["psgc9"] = clean_code_column(df["psgc9"], 9)
    else:
        df["psgc9"] = ""

    # --- names ---
    df["name"] = df["name_raw"].map(normalize_name)

    # --- filter out junk rows ---
    df = df[df["psgc10"].str.len() == 10].copy()
    df = df[df["name"].str.len() > 0].copy()

    # --- levels ---
    df["level"] = (
        df["level"].astype("string").str.strip()
                    .map(LEVEL_MAP)
                    .fillna("")
                    .astype(object)
    )
    df = df[df["level"] != ""].copy()

    # --- region code ---
    df["region_code"] = df["psgc10"].str[:2] + "00000000"

    # --- province code (derived) ---
    prov_codes = set(df.loc[df["level"] == "province", "psgc10"])
    derived    = df["psgc10"].str[:5] + "00000"

    province_code = pd.Series("", index=df.index, dtype=object)
    is_prov = df["level"] == "province"
    province_code.loc[is_prov] = df.loc[is_prov, "psgc10"]

    is_sub = df["level"].isin(["city", "municipality"])
    valid  = derived.isin(prov_codes)
    province_code.loc[is_sub] = derived.loc[is_sub].where(valid.loc[is_sub], "")
    df["province_code"] = province_code

    # --- independent city flag ---
    df["is_independent_city"] = (df["level"] == "city") & (df["province_code"] == "")

    # --- region name lookup ---
    region_lookup = (
        df.loc[df["level"] == "region", ["psgc10", "name"]]
          .rename(columns={"psgc10": "region_code", "name": "region_name"})
          .set_index("region_code")["region_name"]
    )
    df["region_name"] = df["region_code"].map(region_lookup).fillna("")

    # --- province name lookup ---
    prov_lookup = (
        df.loc[df["level"] == "province", ["psgc10", "name"]]
          .rename(columns={"psgc10": "province_code", "name": "province_name"})
          .set_index("province_code")["province_name"]
    )
    df["province_name"] = df["province_code"].map(prov_lookup).fillna("")

    # --- short name + match keys ---
    df["name_short"]      = df["name"].map(make_short)
    df["match_key"]       = df["name"].map(make_match_key)
    df["match_key_short"] = df["name_short"].map(make_match_key)

    # --- population ---
    if "pop_2024" in df.columns:
        df["pop_2024"] = (
            pd.to_numeric(
                df["pop_2024"].astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            ).astype("Int64")
        )
    else:
        df["pop_2024"] = pd.NA

    # --- ensure optional columns exist ---
    for col in ["psgc9", "old_names", "city_class", "income_class",
                "urban_rural", "status"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    LOG.info("--- Validation ---")

    counts = df["level"].value_counts().to_dict()
    LOG.info("Level counts: %s", counts)

    for lvl, expected in EXPECTED.items():
        got = counts.get(lvl, 0)
        if got == expected:
            LOG.info("  ✓ %-13s %d", lvl, got)
        else:
            LOG.warning("  ✗ %-13s expected %d, got %d", lvl, expected, got)

    # Code uniqueness
    dup = int(df["psgc10"].duplicated().sum())
    if dup:
        LOG.warning("Duplicate psgc10 codes: %d", dup)
        LOG.warning("  examples: %s",
                    df.loc[df["psgc10"].duplicated(keep=False), "psgc10"]
                      .unique()[:5].tolist())
    else:
        LOG.info("  ✓ psgc10 is unique")

    # Missing region / province names
    bad_region = df[(df["level"] != "region") & (df["region_name"] == "")]
    if len(bad_region):
        LOG.warning("  ! %d non-region rows missing region_name", len(bad_region))

    orphan_cities = df[df["is_independent_city"]]
    LOG.info("  independent cities (HUC/ICC): %d", len(orphan_cities))

    # Quick LGU-level totals
    lgu = df[df["level"].isin(["province", "city", "municipality"])]
    LOG.info("  total LGU rows (Prov+City+Mun): %d", len(lgu))
    LOG.info("  total 2024 population (LGU rows): %s",
             f"{int(lgu['pop_2024'].fillna(0).sum()):,}")

# ---------------------------------------------------------------------------
# Split + write
# ---------------------------------------------------------------------------

def split_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    regions   = df[df["level"] == "region"].copy()
    provinces = df[df["level"] == "province"].copy()
    cities    = df[df["level"] == "city"].copy()
    muns      = df[df["level"] == "municipality"].copy()
    submins   = df[df["level"] == "submunicipality"].copy()
    bgy       = df[df["level"] == "barangay"].copy()

    lgu_master = pd.concat([provinces, cities, muns], ignore_index=True)

    def order(sub: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in LGU_COLUMNS if c in sub.columns]
        extra = [c for c in sub.columns if c not in cols]
        return sub[cols + extra]

    return {
        "lgu":               order(lgu_master),
        "regions":           order(regions),
        "provinces":         order(provinces),
        "cities":            order(cities),
        "municipalities":    order(muns),
        "submunicipalities": order(submins),
        "barangays":         order(bgy),
    }


def write_outputs(outputs: dict[str, pd.DataFrame],
                  outdir: Path,
                  write_barangay: bool = True,
                  write_csv: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for name, sub in outputs.items():
        if sub.empty:
            LOG.info("Skipping empty output: %s", name)
            continue
        if name == "barangays" and not write_barangay:
            LOG.info("Skipping barangays (--no-barangay)")
            continue
        fname = outdir / f"psgc_{name}_master.parquet"
        sub.to_parquet(fname, index=False)
        LOG.info("Wrote %s (%d rows)", fname, len(sub))

    if write_csv:
        csv_path = outdir / "psgc_lgu_master.csv"
        outputs["lgu"].to_csv(csv_path, index=False)
        LOG.info("Wrote %s", csv_path)


def preview(df: pd.DataFrame, n: int = 10) -> None:
    cols = ["psgc10", "psgc9", "level", "name", "region_name",
            "province_name", "is_independent_city", "pop_2024"]
    cols = [c for c in cols if c in df.columns]
    LOG.info("Preview (first %d LGU rows):\n%s", n,
             df[cols].head(n).to_string(index=False))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input",  type=Path, default=DEFAULT_INPUT,
                   help=f"PSGC workbook (default: {DEFAULT_INPUT})")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                   help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--log",    type=Path, default=DEFAULT_LOG,
                   help=f"Log file (default: {DEFAULT_LOG})")
    p.add_argument("--no-barangay", action="store_true",
                   help="Skip writing the barangay-level parquet")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip writing the LGU CSV preview")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.log, args.verbose)
    LOG.info("PSGC harmonizer starting")
    LOG.info("Input:  %s", args.input)
    LOG.info("Outdir: %s", args.outdir)

    if not args.input.exists():
        LOG.error("Input file not found: %s", args.input)
        return 1

    try:
        raw = read_psgc_sheet(args.input, SHEET_MAIN)
    except Exception as e:
        LOG.exception("Failed to read PSGC sheet: %s", e)
        return 2

    df = normalize_columns(raw)
    df = harmonize(df)

    validate(df)

    outputs = split_outputs(df)
    write_outputs(outputs,
                  args.outdir,
                  write_barangay=not args.no_barangay,
                  write_csv=not args.no_csv)

    preview(outputs["lgu"], n=10)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())