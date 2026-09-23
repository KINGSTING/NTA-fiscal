#!/usr/bin/env python
"""
fetch_population.py
===================
Build annual LGU-level population and land-area estimates for the
1992-2024 panel from PSA 2025 Philippine Statistical Yearbook CSVs.

Sources
-------
  data/PSA/2025_T1_*.csv                    — PSA PSY 2025, Tables 1.x
  data/processed/psgc_lgu_master.parquet    — LGU identities + 2024 pop

Outputs
-------
  data/processed/population_lgu_annual.parquet
      psgc10, fiscal_year, population, land_area_sqkm, pop_source
  data/processed/population_coverage.txt    — diagnostic report

Usage
-----
    python src/fetch_population.py
    python src/fetch_population.py --input-dir data/PSA --inspect
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT_DIR = Path("data/PSA")
DEFAULT_MASTER    = Path("data/processed/psgc_lgu_master.parquet")
DEFAULT_OUTDIR    = Path("data/processed")

CENSUS_YEARS = [2000, 2007, 2010, 2015, 2020, 2024]
PANEL_START  = 1992
PANEL_END    = 2024
MIN_POP_YEAR_COLS = 4

_WS_RE    = re.compile(r"\s+")
_YEAR_RE  = re.compile(r"\b(?:19|20)\d{2}\b")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")

# Footnote-marker lines: 'a2 Due to...', 'h Excluding...', 'j Not a Province/HUC.'
_FOOTNOTE_LINE_RE = re.compile(r"^[a-z]\d?\s+\S")

# Footnote markers PSA appends to province / city names:
#   'Benguet i'                     -> 'Benguet'
#   'Negros Occidental 1,i'         -> 'Negros Occidental'
#   'Maguindanao Del Norte 9'       -> 'Maguindanao Del Norte'
#   'Baliwag 1'                     -> 'Baliwag'
_FOOTNOTE_TAIL_RE = re.compile(r"(?:[\s,]+[A-Za-z0-9]{1,2})+$")

# Any parenthetical chunk, anywhere in the string:
#   'Samar (Western Samar)'                -> 'Samar'
#   'Davao De Oro (Compostela Valley) 2'   -> 'Davao De Oro 2'
_PAREN_ANY_RE = re.compile(r"\s*\([^)]*\)")

_CITY_OF_RE  = re.compile(r"^(?:City|Municipality)\s+of\s+", re.IGNORECASE)
_CITY_SUF_RE = re.compile(r"\s+(?:City|Municipality)\s*$", re.IGNORECASE)
_PROV_SUF_RE = re.compile(r"\s+Province\s*$", re.IGNORECASE)

# Common PSA -> PSGC spelling differences for city names.
_CITY_NAME_ALIASES: dict[str, str] = {
    "ozamis": "ozamiz",
}

LOG = logging.getLogger("fetch_population")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(outdir: Path, verbose: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "fetch_population.log",
                                mode="w", encoding="utf-8"),
        ],
        force=True,
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _clean(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    x = str(s).replace("\u00a0", " ").replace("牋", " ")
    x = unicodedata.normalize("NFKC", x)
    return _WS_RE.sub(" ", x).strip()


def _key(s) -> str:
    x = unicodedata.normalize("NFKD", _clean(s))
    x = x.encode("ascii", "ignore").decode("ascii").lower()
    x = _PUNCT_RE.sub(" ", x)
    x = _WS_RE.sub(" ", x).strip()
    return x


def _strip_footnote_markers(name: str) -> str:
    """Strip trailing PSA footnote markers (' i', ' 1,i', ' 8', ' 10,j')."""
    return _FOOTNOTE_TAIL_RE.sub("", name).strip()


def _is_metadata_row(name: str) -> bool:
    """Rows that are table-continuations, notes, or prose — never data."""
    if not name:
        return True
    n = name.strip()
    nl = n.lower()
    if nl in ("nan", "none"):
        return True
    if nl.startswith("table 1."):
        return True
    if nl in ("notes:", "note:", "source:", "sources:"):
        return True
    if nl.startswith(("note:", "source:", "sources:")):
        return True
    if "republic act" in nl or "in accordance" in nl:
        return True
    # Footnote-marker lines: 'a2 Due to...', 'h Excluding...', 'j Not a Province/HUC.'
    if _FOOTNOTE_LINE_RE.match(n):
        return True
    # Prose / source lines always end with a period; LGU names never do.
    if n.endswith("."):
        return True
    # Long sentences are definitely not LGU names.
    if len(n) > 90:
        return True
    return False


# ---------------------------------------------------------------------------
# Name variants
# ---------------------------------------------------------------------------

def _province_name_variants(name: str) -> list[str]:
    """
    Candidate forms of a province label, in priority order.

    Handles footnote markers, parenthetical qualifiers anywhere,
    and ' Province' suffixes.
    """
    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in out:
            out.append(s)

    base = _clean(name)
    add(base)

    # Strip trailing footnote markers.
    no_fn = _strip_footnote_markers(base)
    add(no_fn)

    # Strip ALL parenthetical chunks (anywhere in the string).
    no_par = _clean(_PAREN_ANY_RE.sub("", base))
    add(no_par)
    add(_strip_footnote_markers(no_par))

    # Strip ' Province' suffix from everything we've produced.
    for v in list(out):
        x = _PROV_SUF_RE.sub("", v).strip()
        if x and x != v:
            add(x)

    return out


def _split_city_province_hint(name: str) -> tuple[str, str | None]:
    """
    'San Carlos (Negros Occidental)' -> ('San Carlos', 'Negros Occidental')
    'Tarlac (Capital)'               -> ('Tarlac', 'Capital')
    'Ozamis'                         -> ('Ozamis', None)
    """
    m = re.match(r"^(.+?)\s*\((.+?)\)\s*$", name.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return name.strip(), None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, header=None, dtype=str,
                               encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path} with any encoding")


def _parse_number(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return float("nan")
    x = str(s).strip()
    if x == "" or x.lower() in ("nan", "none", "-", "n/a", "…"):
        return float("nan")
    x = x.replace(",", "").replace(" ", "").replace("\u00a0", "")
    x = re.sub(r"[^\d.\-]", "", x)
    try:
        return float(x)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def _find_census_row(df: pd.DataFrame) -> int | None:
    """
    Find the row that has census years spread across multiple DIFFERENT
    columns.  Rejects title rows that mention several years in one cell.
    """
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        year_cols: set[int] = set()
        for j, v in enumerate(row.tolist()):
            if v is None:
                continue
            s = str(v)
            for y in CENSUS_YEARS:
                if str(y) in s:
                    year_cols.add(j)
                    break
        if len(year_cols) >= 4:
            return i
    return None


def _find_data_start(df: pd.DataFrame, header_row: int) -> int:
    for i in range(header_row + 1, min(header_row + 15, len(df))):
        v = df.iloc[i, 0]
        if pd.isna(v):
            continue
        s = _clean(v)
        if not s or s.lower() in ("nan", "none"):
            continue
        if _YEAR_RE.fullmatch(s):
            continue
        return i
    return header_row + 1


def _detect_column_groups(df: pd.DataFrame, header_row: int) -> dict:
    """Read year labels in the census-year row to map year -> column."""
    census = [str(v) for v in df.iloc[header_row].tolist()]

    year_to_col: dict[int, int] = {}
    for j, v in enumerate(census):
        if not v:
            continue
        m = re.search(r"\b((?:19|20)\d{2})\b", v)
        if not m:
            continue
        y = int(m.group(1))
        if y in CENSUS_YEARS and y not in year_to_col:
            year_to_col[y] = j

    main_header_idx = max(0, header_row - 1)
    main = [str(v) for v in df.iloc[main_header_idx].tolist()]
    land_col: int | None = None
    for j, v in enumerate(main):
        if v and "land area" in v.lower():
            land_col = j
            break
    if land_col is None and main_header_idx > 0:
        main2 = [str(v) for v in df.iloc[main_header_idx - 1].tolist()]
        for j, v in enumerate(main2):
            if v and "land area" in v.lower():
                land_col = j
                break

    if land_col is None:
        pop_years  = dict(year_to_col)
        dens_years: dict[int, int] = {}
    else:
        years_left  = {y: c for y, c in year_to_col.items() if c < land_col}
        years_right = {y: c for y, c in year_to_col.items() if c > land_col}
        if len(years_left) >= len(years_right):
            pop_years, dens_years = years_left, years_right
        else:
            pop_years, dens_years = years_right, years_left

    return {
        "pop_year_to_col":  pop_years,
        "dens_year_to_col": dens_years,
        "land_col":         land_col,
    }


# ---------------------------------------------------------------------------
# Level detection
# ---------------------------------------------------------------------------

def _level_from_title(title: str) -> str | None:
    t = title.lower()
    if "by city" in t or "by municipality" in t:
        return "lgu"
    if "by region and province" in t or "by region, province" in t:
        return "province"
    return None


def _looks_like_lgu_row(name: str) -> bool:
    """Strict — explicit LGU markers only."""
    n = name.lower()
    if n.startswith(("city of ", "municipality of ")):
        return True
    if n.endswith((" city", " municipality")):
        return True
    return False


def parse_psa_table(path: Path) -> dict | None:
    try:
        df = _load_csv(path)
    except Exception as e:
        LOG.debug("  %s: unreadable (%s)", path.name, e)
        return None
    if df.empty:
        return None

    title = _clean(df.iloc[0, 0]) if len(df) else ""
    census_row = _find_census_row(df)
    if census_row is None:
        LOG.debug("  %s: no census-year header found", path.name)
        return None

    cols = _detect_column_groups(df, census_row)
    if not cols["pop_year_to_col"]:
        LOG.debug("  %s: no population columns detected", path.name)
        return None
    if len(cols["pop_year_to_col"]) < MIN_POP_YEAR_COLS:
        LOG.debug("  %s: only %d population-year column(s) detected "
                  "(need >= %d) — skipping",
                  path.name, len(cols["pop_year_to_col"]),
                  MIN_POP_YEAR_COLS)
        return None

    data_start = _find_data_start(df, census_row)

    rows = []
    for i in range(data_start, len(df)):
        name = _clean(df.iloc[i, 0])
        if not name or name.lower() in ("nan", "none"):
            continue
        if _is_metadata_row(name):
            continue
        nl = name.lower()
        if nl in ("philippines", "total", "grand total",
                  "region", "province", "city", "municipality"):
            continue
        if _YEAR_RE.fullmatch(name):
            continue
        if name in ("Region and province", "Region and Province"):
            continue

        rec = {"name": name}
        for year, col in cols["pop_year_to_col"].items():
            if col >= df.shape[1]:
                continue
            rec[f"pop_{year}"] = _parse_number(df.iloc[i, col])

        if cols["land_col"] is not None and cols["land_col"] < df.shape[1]:
            rec["land_area_sqkm"] = _parse_number(
                df.iloc[i, cols["land_col"]])

        rows.append(rec)

    if not rows:
        LOG.debug("  %s: no data rows parsed", path.name)
        return None

    wide = pd.DataFrame(rows)

    title_level = _level_from_title(title)
    if title_level == "lgu":
        is_lgu_level = True
    elif title_level == "province":
        is_lgu_level = False
    else:
        lgu_hits = sum(1 for n in wide["name"] if _looks_like_lgu_row(n))
        is_lgu_level = (lgu_hits / len(wide)) >= 0.60

    return {
        "path":         path,
        "name":         title,
        "is_lgu_level": is_lgu_level,
        "wide":         wide,
    }


def scan_tables(input_dir: Path) -> list[dict]:
    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs in {input_dir}")
    LOG.info("Scanning %d CSVs in %s", len(csvs), input_dir)
    tables = []
    for path in csvs:
        result = parse_psa_table(path)
        if result is None:
            continue
        tables.append(result)
        LOG.info("  %-26s  rows=%-5d  lgu_level=%s  title=%s",
                 path.name, len(result["wide"]),
                 "YES" if result["is_lgu_level"] else " no",
                 result["name"][:50])
    return tables


# ---------------------------------------------------------------------------
# Key maps from master
# ---------------------------------------------------------------------------

def _build_province_key_map(master: pd.DataFrame) -> dict[str, str]:
    """
    {normalized_name -> province_psgc10}, indexed by all reasonable
    name variants (raw, paren-stripped, ' Province'-stripped).
    """
    provs = master[master["level"] == "province"]
    m: dict[str, str] = {}
    for row in provs.itertuples(index=False):
        psgc = row.psgc10
        n = row.name
        for cand in _province_name_variants(n):
            k = _key(cand)
            if k and k not in m:
                m[k] = psgc
    return m


def _build_all_lgu_map(master: pd.DataFrame) -> dict[str, str]:
    """
    {normalized_name -> psgc10} for every city AND municipality in master,
    indexed with the same alias forms used elsewhere ('City of X' -> 'X',
    'X City' -> 'X').  Used to catch LGU rows that PSA lists alongside
    provinces (e.g. Pateros, the sole NCR municipality).
    """
    m: dict[str, str] = {}
    rows = master[master["level"].isin(("city", "municipality"))]
    for row in rows.itertuples(index=False):
        psgc = row.psgc10
        n = row.name
        for k in (_key(n),
                  _key(_CITY_OF_RE.sub("", n)),
                  _key(_CITY_SUF_RE.sub("", n))):
            if k and k not in m:
                m[k] = psgc
    return m


def _build_city_key_maps(master: pd.DataFrame,
                         level: str) -> tuple[dict[str, list[str]],
                                              dict[tuple[str, str], list[str]]]:
    """
    Returns:
      by_key:          city_key -> [psgc10]
      by_city_prov:    (city_key, province_key) -> [psgc10]

    Both maps include aliases: 'City of X' -> 'X', 'X City' -> 'X'.
    """
    rows = master[master["level"] == level]
    by_key: dict[str, list[str]] = {}
    by_city_prov: dict[tuple[str, str], list[str]] = {}

    for row in rows.itertuples(index=False):
        psgc = row.psgc10
        n = row.name
        variants = {
            _key(n),
            _key(_CITY_OF_RE.sub("", n)),
            _key(_CITY_SUF_RE.sub("", n)),
        }
        for k in variants:
            if not k:
                continue
            lst = by_key.setdefault(k, [])
            if psgc not in lst:
                lst.append(psgc)

        prov = getattr(row, "province_name", None)
        if prov:
            pk = _key(prov)
            ck = _key(_CITY_OF_RE.sub("", n))
            if pk and ck:
                lst = by_city_prov.setdefault((ck, pk), [])
                if psgc not in lst:
                    lst.append(psgc)

    return by_key, by_city_prov


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_province_pop(prov_wide: pd.DataFrame,
                          master: pd.DataFrame) -> pd.DataFrame:
    """
    Read the province-level table.

    Rows are attributed in priority order:
      1. province match -> province_direct
      2. non-province LGU match (e.g. Pateros) -> lgu_direct
      3. otherwise -> unmatched (logged, then dropped)
    """
    prov_map = _build_province_key_map(master)
    lgu_map  = _build_all_lgu_map(master)

    rows = []
    unmatched = []
    for _, r in prov_wide.iterrows():
        raw = r["name"]

        if _is_metadata_row(raw):
            continue

        # HUCs and component cities appear in T1_1 but are captured by T1_3.
        if _looks_like_lgu_row(raw):
            continue

        nm_lower = raw.lower()
        if "region" in nm_lower or nm_lower.startswith("national "):
            continue

        psgc: str | None = None
        src: str = ""

        for cand in _province_name_variants(raw):
            k = _key(cand)
            if k in prov_map:
                psgc, src = prov_map[k], "province_direct"
                break

        if psgc is None:
            # Not a province.  Maybe a top-level LGU that PSA lists
            # alongside provinces — e.g. Pateros, the sole municipality
            # of NCR (NCR is a region, not a province).
            lgu_psgc = lgu_map.get(_key(raw))
            if lgu_psgc is not None:
                psgc, src = lgu_psgc, "lgu_direct"

        if psgc is None:
            unmatched.append(raw)
            continue

        for year in CENSUS_YEARS:
            col = f"pop_{year}"
            if col in r and pd.notna(r[col]):
                rows.append({
                    "psgc10":     psgc,
                    "year":       year,
                    "population": float(r[col]),
                    "pop_source": src,
                })

    if unmatched:
        LOG.info("Province rows that did NOT match master (%d):",
                 len(unmatched))
        for nm in unmatched[:25]:
            LOG.info("    %r", nm)

    return pd.DataFrame(rows)


def _extract_lgu_pop(lgu_wide: pd.DataFrame,
                     master: pd.DataFrame,
                     level: str) -> pd.DataFrame:
    by_key, by_city_prov = _build_city_key_maps(master, level)

    rows = []
    unmatched = []
    for _, r in lgu_wide.iterrows():
        raw = r["name"]
        if _is_metadata_row(raw):
            continue

        nm = _strip_footnote_markers(re.sub(r"\*+$", "", raw).strip())
        base, hint = _split_city_province_hint(nm)
        k = _key(base)
        kh = _key(hint) if hint else None

        hits: list[str] = []
        if kh:
            hits = by_city_prov.get((k, kh), [])
        if not hits:
            hits = by_key.get(k, [])
        if not hits:
            alt = _CITY_NAME_ALIASES.get(k)
            if alt:
                hits = by_key.get(alt, [])

        if len(hits) != 1:
            unmatched.append(raw)
            continue
        psgc10 = hits[0]

        for year in CENSUS_YEARS:
            col = f"pop_{year}"
            if col in r and pd.notna(r[col]):
                rows.append({
                    "psgc10":     psgc10,
                    "year":       year,
                    "population": float(r[col]),
                    "pop_source": f"{level}_direct",
                })

    if unmatched:
        LOG.info("%s rows that did NOT match master (%d):",
                 level.capitalize(), len(unmatched))
        for nm in unmatched[:25]:
            LOG.info("    %r", nm)

    return pd.DataFrame(rows)


def _allocate_municipalities(province_pop: pd.DataFrame,
                             master: pd.DataFrame) -> pd.DataFrame:
    muns = master[master["level"] == "municipality"].copy()
    muns["_prov_psgc"] = muns["province_code"]
    by_prov = {p: sub for p, sub in muns.groupby("_prov_psgc")
               if p is not None and p != ""}

    rows = []
    for prov_psgc, g in province_pop.groupby("psgc10"):
        members = by_prov.get(prov_psgc)
        if members is None or len(members) == 0:
            continue
        pops = pd.to_numeric(members["pop_2024"], errors="coerce").fillna(0)
        if pops.sum() == 0:
            shares = pd.Series(1.0 / len(members), index=members.index)
        else:
            shares = pops / pops.sum()
        for _, r in g.iterrows():
            year = r["year"]
            prov_pop = r["population"]
            for idx, share in shares.items():
                rows.append({
                    "psgc10":     members.loc[idx, "psgc10"],
                    "year":       year,
                    "population": float(prov_pop) * float(share),
                    "pop_source": "province_allocated",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Annual interpolation
# ---------------------------------------------------------------------------

def interpolate_to_annual(census_long: pd.DataFrame,
                          start: int = PANEL_START,
                          end: int = PANEL_END) -> pd.DataFrame:
    all_years = list(range(start, end + 1))
    rows = []
    for psgc10, g in census_long.groupby("psgc10"):
        g = g.sort_values("year").reset_index(drop=True)
        years = g["year"].tolist()
        pops  = g["population"].astype(float).tolist()
        src   = g["pop_source"].iloc[0]
        if len(years) == 0:
            continue
        for y in all_years:
            if y <= years[0]:
                p = pops[0]
            elif y >= years[-1]:
                p = pops[-1]
            else:
                for k in range(len(years) - 1):
                    y0, y1 = years[k], years[k + 1]
                    if y0 <= y <= y1:
                        p0, p1 = pops[k], pops[k + 1]
                        if p0 > 0 and p1 > 0:
                            t = (y - y0) / (y1 - y0)
                            p = np.exp(np.log(p0) * (1 - t) + np.log(p1) * t)
                        else:
                            t = (y - y0) / (y1 - y0)
                            p = p0 * (1 - t) + p1 * t
                        break
                else:
                    p = pops[-1]
            rows.append({
                "psgc10":      psgc10,
                "fiscal_year": y,
                "population":  p,
                "pop_source":  src,
            })
    return pd.DataFrame(rows)


def _backfill_missing_2024(annual: pd.DataFrame,
                           master: pd.DataFrame) -> pd.DataFrame:
    """
    For any LGU in master that received no population value from the PSA
    tables, add a single 2024 row using master's pop_2024.  Covers the 8
    BARMM Special Geographic Area municipalities, which were created in
    2023 and have no historical PSA PSY row.
    """
    if "pop_2024" not in master.columns:
        return annual
    have = set(annual["psgc10"].unique())
    extra = []
    for row in master.itertuples(index=False):
        if row.psgc10 in have:
            continue
        pop = getattr(row, "pop_2024", None)
        if pop is None or pd.isna(pop):
            continue
        extra.append({
            "psgc10":         row.psgc10,
            "fiscal_year":    2024,
            "population":     float(pop),
            "land_area_sqkm": np.nan,
            "pop_source":     "master_2024_only",
        })
    if not extra:
        return annual
    LOG.info("Backfilled %d LGU(s) with master pop_2024 only", len(extra))
    for e in extra:
        LOG.info("    %s", e["psgc10"])
    return pd.concat([annual, pd.DataFrame(extra)], ignore_index=True)


# ---------------------------------------------------------------------------
# Land area
# ---------------------------------------------------------------------------

def build_land_area(master: pd.DataFrame,
                    prov_wide: pd.DataFrame | None) -> pd.DataFrame:
    if prov_wide is not None and "land_area_sqkm" in prov_wide.columns:
        prov_map = _build_province_key_map(master)
        area_by_prov: dict[str, float] = {}
        for _, r in prov_wide.iterrows():
            area = r.get("land_area_sqkm")
            if pd.isna(area):
                continue
            for cand in _province_name_variants(r["name"]):
                k = _key(cand)
                if k in prov_map:
                    area_by_prov[prov_map[k]] = float(area)
                    break
    else:
        area_by_prov = {}

    rows = []
    for row in master.itertuples(index=False):
        psgc10 = row.psgc10
        area = np.nan
        if row.level == "province" and psgc10 in area_by_prov:
            area = area_by_prov[psgc10]
        elif row.level in ("city", "municipality"):
            area = area_by_prov.get(row.province_code, np.nan)
        rows.append({"psgc10": psgc10, "land_area_sqkm": area})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

def write_qa(out_annual: pd.DataFrame, out_path: Path,
             n_tables: int, n_lgu_tables: int,
             n_matched: int, n_total: int) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("POPULATION FETCH QA")
    lines.append("=" * 78)
    lines.append(f"CSVs scanned:                    {n_tables}")
    lines.append(f"  LGU-level tables identified:   {n_lgu_tables}")
    lines.append(f"LGUs matched to master:          {n_matched} / {n_total}")
    lines.append("")
    lines.append(f"Rows produced:                   {len(out_annual):,}")
    lines.append(f"Distinct psgc10:                 "
                 f"{out_annual['psgc10'].nunique()}")
    lines.append(f"Years covered:                   "
                 f"{out_annual['fiscal_year'].min()} – "
                 f"{out_annual['fiscal_year'].max()}")
    lines.append("")
    lines.append("-- pop_source distribution --")
    src = out_annual.groupby("pop_source")["psgc10"].nunique()
    for k, v in src.items():
        lines.append(f"  {k:25s}  {v} LGUs")
    lines.append("")
    lines.append("-- Sample (2024) --")
    sample = (out_annual[out_annual["fiscal_year"] == 2024].head(10))
    lines.append(sample.to_string(index=False))
    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--master",    type=Path, default=DEFAULT_MASTER)
    p.add_argument("--outdir",    type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--out",       type=Path,
                   default=DEFAULT_OUTDIR / "population_lgu_annual.parquet")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.outdir, verbose=args.verbose)

    tables = scan_tables(args.input_dir)
    if not tables:
        LOG.error("No population tables found in %s", args.input_dir)
        return 1

    if args.inspect:
        LOG.info("Inspect mode — printing wide head of each detected table")
        for t in tables:
            print(f"\n--- {t['path'].name} ---")
            print(f"title: {t['name']}")
            print(f"lgu_level: {t['is_lgu_level']}")
            print(t["wide"].head(10).to_string(index=False))
        return 0

    prov_wide = None
    city_wide = None
    for t in tables:
        title_lower = t["name"].lower()
        if "by city" in title_lower and city_wide is None:
            city_wide = t["wide"]
            LOG.info("Using city-level table: %s (%d rows)",
                     t["path"].name, len(city_wide))
        elif ("by region and province" in title_lower
              or "by region, province" in title_lower):
            if prov_wide is None or len(t["wide"]) > len(prov_wide):
                prov_wide = t["wide"]
                LOG.info("Using province-level table: %s (%d rows)",
                         t["path"].name, len(prov_wide))

    if prov_wide is None:
        for t in tables:
            if prov_wide is None or len(t["wide"]) > len(prov_wide):
                prov_wide = t["wide"]
        LOG.info("Fallback province table: %d rows", len(prov_wide))

    prov_wide_for_land = prov_wide

    master = pd.read_parquet(args.master)
    LOG.info("Loaded master: %d rows", len(master))

    province_pop = _extract_province_pop(prov_wide, master)
    LOG.info("Province-level population rows: %d", len(province_pop))

    city_pop = pd.DataFrame()
    if city_wide is not None:
        city_pop = _extract_lgu_pop(city_wide, master, level="city")
        LOG.info("City-level population rows: %d", len(city_pop))

    mun_pop = _allocate_municipalities(province_pop, master)
    LOG.info("Allocated municipality rows: %d", len(mun_pop))

    census_long = pd.concat(
        [province_pop, city_pop, mun_pop], ignore_index=True
    ).drop_duplicates(subset=["psgc10", "year"], keep="first")
    LOG.info("Total census-year rows after merge: %d", len(census_long))

    if census_long.empty:
        LOG.error("No census population extracted")
        return 1

    LOG.info("Interpolating to annual (%d–%d)", PANEL_START, PANEL_END)
    annual = interpolate_to_annual(census_long, PANEL_START, PANEL_END)
    LOG.info("Annual rows: %d", len(annual))

    LOG.info("Building land area lookup")
    land = build_land_area(master, prov_wide_for_land)
    annual = annual.merge(land, on="psgc10", how="left")

    annual = annual[["psgc10", "fiscal_year",
                     "population", "land_area_sqkm", "pop_source"]]

    annual = _backfill_missing_2024(annual, master)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    annual.to_parquet(args.out, index=False)
    LOG.info("Wrote %s (%d rows x %d cols)",
             args.out, len(annual), annual.shape[1])

    n_matched = annual["psgc10"].nunique()
    n_total   = master["psgc10"].nunique()
    n_lgu_tables = sum(1 for t in tables if t["is_lgu_level"])
    write_qa(annual,
             args.outdir / "population_coverage.txt",
             n_tables=len(tables),
             n_lgu_tables=n_lgu_tables,
             n_matched=n_matched,
             n_total=n_total)

    return 0


if __name__ == "__main__":
    sys.exit(main())