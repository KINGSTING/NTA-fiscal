#!/usr/bin/env python
"""
fetch_cpi.py
============
Normalize PSA OpenSTAT CPI CSVs into a tidy
(region, fiscal_year, cpi, deflator_2024) deflator table.

Input format (as emitted by PSA OpenSTAT):
    row 0 (title): "Consumer Price Index for All Income Households by
                    Commodity Group (2018=100): January 1994 - December 2017"
    row 1 (header): "Geolocation","Commodity Description","1994 Jan",...,"2017 Ave"
    row 2+: data, incl. region-level and province/city-level rows

We keep only rows whose Commodity Description is "All Items" (top-level CPI)
and only geolocations that map to a canonical region.  Annual CPI is read
from the "YYYY Ave" columns.

Inputs
------
  data/PSA/Consumer Price Index*.csv     (one or more files; a glob)

Output
------
  data/processed/cpi_regional.parquet
      region, fiscal_year, cpi, deflator_2024
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SRC_GLOB = "data/PSA/Consumer Price Index*.csv"
DEFAULT_OUT      = Path("data/processed/cpi_regional.parquet")

LOG = logging.getLogger("fetch_cpi")

_AVE_RE   = re.compile(r"^(\d{4})\s+Ave\s*$")
_NORM_RE  = re.compile(r"[^a-z0-9]+")
_PUNCT_RE = re.compile(r"[^a-z0-9]+")

# ---------------------------------------------------------------------------
# Region canonicalization
# ---------------------------------------------------------------------------

_REGION_CANON = {
    "philippines":                                      "Philippines",
    "ncr":                                              "National Capital Region",
    "national capital region":                          "National Capital Region",
    "metro manila":                                     "National Capital Region",
    "cordillera administrative region":                 "Cordillera Administrative Region",
    "car":                                              "Cordillera Administrative Region",
    "region i":                                         "Region I",
    "region i ilocos region":                           "Region I",
    "ilocos region":                                    "Region I",
    "region ii":                                        "Region II",
    "region ii cagayan valley":                         "Region II",
    "cagayan valley":                                   "Region II",
    "region iii":                                       "Region III",
    "region iii central luzon":                         "Region III",
    "central luzon":                                    "Region III",
    "region iv a":                                      "Region IV-A",
    "region iv a calabarzon":                           "Region IV-A",
    "calabarzon":                                       "Region IV-A",
    "region iv b":                                      "Region IV-B",
    "region iv b mimaropa":                             "Region IV-B",
    "mimaropa":                                         "Region IV-B",
    "region v":                                         "Region V",
    "region v bicol region":                            "Region V",
    "bicol region":                                     "Region V",
    "bicol":                                            "Region V",
    "region vi":                                        "Region VI",
    "region vi western visayas":                        "Region VI",
    "western visayas":                                  "Region VI",
    "region vii":                                       "Region VII",
    "region vii central visayas":                       "Region VII",
    "central visayas":                                  "Region VII",
    "region viii":                                      "Region VIII",
    "region viii eastern visayas":                      "Region VIII",
    "eastern visayas":                                  "Region VIII",
    "region ix":                                        "Region IX",
    "region ix zamboanga peninsula":                    "Region IX",
    "zamboanga peninsula":                              "Region IX",
    "region x":                                         "Region X",
    "region x northern mindanao":                       "Region X",
    "northern mindanao":                                "Region X",
    "region xi":                                        "Region XI",
    "region xi davao region":                           "Region XI",
    "davao region":                                     "Region XI",
    "region xii":                                       "Region XII",
    "region xii soccsksargen":                          "Region XII",
    "soccsksargen":                                     "Region XII",
    "region xiii":                                      "Region XIII",
    "region xiii caraga":                               "Region XIII",
    "caraga":                                           "Region XIII",
    "bangsamoro":                                       "BARMM",
    "bangsamoro autonomous region in muslim mindanao":  "BARMM",
    "barmm":                                            "BARMM",
    "armm":                                             "BARMM",
    "autonomous region in muslim mindanao":             "BARMM",
}


def _norm(s) -> str:
    return _NORM_RE.sub(" ", str(s).lower()).strip()


def canon_region(r) -> str | None:
    """
    Map a PSA geolocation string to a canonical region name.

    Handles:
        'National Capital Region'
        'National Capital Region (NCR)'
        'NCR'
        'Region I - Ilocos Region'
        '..Region I'
        'MIMAROPA Region'
        'Region IV-B - MIMAROPA'
        'Cordillera Administrative Region'
        'CAR'
        'Bangsamoro Autonomous Region in Muslim Mindanao'
        'BARMM'
    """
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return None
    raw = str(r).strip()
    if not raw:
        return None

    # Strip leading dots (PSA hierarchy indent: '', '..', '....', '......')
    raw = re.sub(r"^[.\s]+", "", raw)
    # Strip trailing parentheticals: "(NCR)", "(CAR)", "(BARMM)"
    without_parens = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()

    for candidate in (raw, without_parens):
        k = _norm(candidate)
        if not k:
            continue

        # 1. Exact match
        if k in _REGION_CANON:
            return _REGION_CANON[k]

        # 2. Prefix match: 'national capital region ncr'
        #    starts with 'national capital region '
        for key in sorted(_REGION_CANON, key=len, reverse=True):
            if k.startswith(key + " "):
                return _REGION_CANON[key]

        # 3. Suffix match: 'mimaropa region' ends with ' mimaropa'
        for key in sorted(_REGION_CANON, key=len, reverse=True):
            if k.endswith(" " + key):
                return _REGION_CANON[key]

        # 4. 'Region <roman>' — accepts trailing content
        m = re.match(r"^region\s+(i{1,3}[ab]?|iv\s*[ab]|1[0-3]?|[1-9])\b", k)
        if m:
            token = m.group(1).replace(" ", "").upper()
            return f"Region {token}"

    return None


# ---------------------------------------------------------------------------
# Commodity matching — deliberately permissive
# ---------------------------------------------------------------------------

def _is_all_items(s) -> bool:
    """
    Match the top-level 'All Items' row, tolerating:
        'All Items'
        '0 - ALL ITEMS'
        '00 - All Items'
        'ALL ITEMS '
        'All Items (2018=100)'
    Strategy: strip any leading 'NN - ' / 'NN.NN - ' code prefix, then
    normalize to lowercase alnum, and require exactly 'all items'.
    """
    x = str(s or "").strip()
    if not x:
        return False
    # Drop a leading numeric code prefix like "0 - ", "01 - ", "01.1 - ".
    x = re.sub(r"^[\d.\s]+\s*[-–—]\s*", "", x)
    # Normalize punctuation and whitespace, lowercase.
    x = _PUNCT_RE.sub(" ", x.lower())
    x = re.sub(r"\s+", " ", x).strip()
    return x == "all items"


# ---------------------------------------------------------------------------
# Column filter for read_csv
# ---------------------------------------------------------------------------

def _cpi_usecols(col) -> bool:
    c  = str(col).strip()
    cl = c.lower()
    if cl == "geolocation":
        return True
    if "commodity" in cl:
        return True
    if _AVE_RE.match(c):
        return True
    return False


# ---------------------------------------------------------------------------
# Header row discovery (csv.reader — no pandas, no tokenization errors)
# ---------------------------------------------------------------------------

def _peek_header_row(path: Path, max_rows: int = 10) -> int:
    """
    Return the 0-based index of the row containing the real header
    (matched only on the exact cell 'geolocation').

    The title line above the header contains the substring 'Commodity',
    so we must NOT loosen this check.
    """
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    cells = [c.strip().lower() for c in row]
                    if any(c == "geolocation" for c in cells):
                        return i
            return -1
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(f"Could not open {path}: {last_err}")


# ---------------------------------------------------------------------------
# Per-file loader
# ---------------------------------------------------------------------------

def load_one(path: Path) -> pd.DataFrame:
    header_row = _peek_header_row(path)
    if header_row < 0:
        raise SystemExit(
            f"{path.name}: could not locate 'Geolocation' header in first "
            f"10 rows.")
    LOG.debug("  %s: header at raw row %d", path.name, header_row)

    df = None
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str,
                             skiprows=header_row,
                             usecols=_cpi_usecols,
                             low_memory=False)
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        raise SystemExit(f"Could not read {path}: {last_err}")

    # --- Locate Geolocation and Commodity Description columns ---
    region_col = next((c for c in df.columns if _norm(c) == "geolocation"),
                      None)
    commodity_cols = [c for c in df.columns if "commodity" in _norm(c)]
    commodity_col = None
    if commodity_cols:
        commodity_col = next((c for c in commodity_cols
                              if "description" in _norm(c)),
                             commodity_cols[0])

    if region_col is None:
        raise SystemExit(
            f"{path.name}: no 'Geolocation' column. "
            f"Columns: {list(df.columns)[:8]}")
    if commodity_col is None:
        raise SystemExit(
            f"{path.name}: no 'Commodity Description' column. "
            f"Columns: {list(df.columns)[:8]}")

    # --- Filter to All Items ---
    mask = df[commodity_col].map(_is_all_items)
    n_all = int(mask.sum())
    if n_all == 0:
        uniq = (df[commodity_col].dropna().astype(str).str.strip()
                  .unique()[:25])
        raise SystemExit(
            f"{path.name}: no 'All Items' rows found.\n"
            f"  First 25 unique Commodity Descriptions: "
            f"{list(uniq)}")
    df = df[mask].copy()
    LOG.info("  %s: %d All-Items rows kept", path.name, n_all)

    # --- Melt year-Ave columns ---
    ave_cols = [(c, int(_AVE_RE.match(c.strip()).group(1)))
                for c in df.columns if _AVE_RE.match(c.strip())]
    if not ave_cols:
        raise SystemExit(f"{path.name}: no 'YYYY Ave' columns found")

    long = df.melt(id_vars=[region_col],
                   value_vars=[c for c, _ in ave_cols],
                   var_name="col", value_name="cpi")
    year_map = {c: y for c, y in ave_cols}
    long["fiscal_year"] = long["col"].map(year_map).astype("Int64")
    long = long.rename(columns={region_col: "region"})
    long = long.drop(columns=["col"])

    long["cpi"] = pd.to_numeric(
        long["cpi"].astype(str).str.replace(",", "", regex=False),
        errors="coerce")
    long = long.dropna(subset=["fiscal_year", "cpi"])
    return long[["region", "fiscal_year", "cpi"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src-glob", default=DEFAULT_SRC_GLOB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    from glob import glob
    paths = sorted(Path(x) for x in glob(args.src_glob))
    if not paths:
        LOG.error("No files matched %r", args.src_glob)
        return 1
    LOG.info("Reading %d file(s):", len(paths))
    for q in paths:
        LOG.info("  %s  (%.1f MB)", q.name, q.stat().st_size / 1e6)

    frames = []
    for q in paths:
        LOG.info("Loading %s ...", q.name)
        frames.append(load_one(q))
    long = pd.concat(frames, ignore_index=True)

    # --- Canonicalize region names ---
    long["region_canon"] = long["region"].map(canon_region)
    n_bad = int(long["region_canon"].isna().sum())
    if n_bad:
        bad = sorted(long.loc[long["region_canon"].isna(), "region"]
                     .dropna().unique())
        LOG.info("  %d rows dropped (not a canonical region). "
                 "Unique unmapped labels: %d", n_bad, len(bad))
        for b in bad[:30]:
            LOG.debug("    %r", b)
    long = long[long["region_canon"].notna()].copy()
    long = long.drop(columns=["region"]).rename(
        columns={"region_canon": "region"})

    # --- Dedupe overlapping (region, year) at the backcast/current boundary ---
    before = len(long)
    long = (long.sort_values(["region", "fiscal_year"])
                .drop_duplicates(subset=["region", "fiscal_year"], keep="last")
                .reset_index(drop=True))
    if before != len(long):
        LOG.info("  Deduped %d overlapping rows", before - len(long))

    # --- Restrict to panel window, then reindex to full 1992-2024 grid ---
    long = long[long["fiscal_year"].between(1992, 2024)].copy()

    all_years = list(range(1992, 2025))
    full_idx = pd.MultiIndex.from_product(
        [sorted(long["region"].unique()), all_years],
        names=["region", "fiscal_year"])
    long = (long.set_index(["region", "fiscal_year"])
                .reindex(full_idx)
                .reset_index())

    # --- 1992-1993 filled from 1994 (backcast); safety ffill for gaps ---
    long = long.sort_values(["region", "fiscal_year"]).reset_index(drop=True)
    long["cpi"] = long.groupby("region")["cpi"].transform(
        lambda s: s.ffill().bfill())

    # --- Per-region 2024 deflator = cpi / cpi_2024 ---
    base = (long[long["fiscal_year"] == 2024]
            .set_index("region")["cpi"].to_dict())
    missing_base = sorted(set(long["region"]) - set(base))
    if missing_base:
        LOG.warning("  No 2024 CPI for: %s", missing_base)
    long["deflator_2024"] = long.apply(
        lambda r: r["cpi"] / base[r["region"]] if r["region"] in base else np.nan,
        axis=1)

    long = (long[["region", "fiscal_year", "cpi", "deflator_2024"]]
                .sort_values(["region", "fiscal_year"])
                .reset_index(drop=True))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(args.out, index=False)
    LOG.info("Wrote %s (%d rows, %d regions, years %d-%d)",
             args.out, len(long), long["region"].nunique(),
             int(long["fiscal_year"].min()), int(long["fiscal_year"].max()))
    LOG.info("Regions covered:\n%s",
             "\n".join(f"  {r}" for r in sorted(long["region"].unique())))
    return 0


if __name__ == "__main__":
    sys.exit(main())