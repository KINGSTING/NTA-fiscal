#!/usr/bin/env python
"""
build_panel.py
==============
Build the unified LGU-year panel: SRE fiscal data (1992–2024) joined to
PSA SAE poverty snapshots (2018, 2021, 2023), keyed on 10-digit PSGC.

Unit of observation: (psgc10, fiscal_year)

Design choices
--------------
* Base frame is the SRE panel (longest coverage).  Rows unmatched to a
  PSGC (orphans like "PROVINCIAL TOTAL") are dropped.
* Canonical identifiers (lgu_name / province / region / lgu_type /
  income_class / city_class / urban_rural / pop_2024) come from the PSGC
  master, NOT from either source panel.  This is authoritative and also
  fixes source-side labelling errors (e.g. the Surigao del Norte/Sur
  mix-up in the PSA SAE sheet).
* Poverty (PI, CV, SE, CI) is merged twice:
      poverty_pi / poverty_cv / ...          — non-null only on
                                                fiscal_year ∈ {2018,2021,2023}
      pi_2018 / pi_2021 / pi_2023 / ...      — broadcast, non-null on every
                                                row of the same psgc10
  Analysts who want a current-year specification use the first set;
  those who want a fixed LGU characteristic use the second.
* Manila's 14 legislative districts (all matched to 1380600000) are
  collapsed into ONE row per SAE year before the merge, using an
  unweighted mean for PI and NaN for CV/SE/CI (which cannot be pooled
  without population weights).  Pass --keep-manila-districts to skip
  this — you'll get 14 duplicate psgc10 rows per SAE year and it will be
  your job to reconcile them.

Inputs
------
    data/processed/sre_panel_with_psgc.parquet
    data/processed/psa_panel_with_psgc.parquet
    data/processed/psgc_lgu_master.parquet

Outputs
-------
    data/processed/panel.parquet
    data/processed/panel.csv          (optional, --csv)
    data/processed/panel_qa.txt
    data/processed/panel_build.log

Usage
-----
    python src/build_panel.py
    python src/build_panel.py --keep-manila-districts --csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_SRE    = Path("data/processed/sre_panel_with_psgc.parquet")
DEFAULT_PSA    = Path("data/processed/psa_panel_with_psgc.parquet")
DEFAULT_PSGC   = Path("data/processed/psgc_lgu_master.parquet")
DEFAULT_OUTDIR = Path("data/processed")
DEFAULT_OUT    = DEFAULT_OUTDIR / "panel.parquet"

SAE_YEARS = (2018, 2021, 2023)
MANILA_PSGC = "1380600000"

# Non-LGU placeholder names that occasionally leak through the SRE side.
ORPHAN_NAMES = {
    "provincial total", "regional total", "national total",
    "total", "subtotal", "grand total",
    "", "`", "-", "n/a", "na", "none", "null",
}

LOG = logging.getLogger("build_panel")


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
            logging.FileHandler(outdir / "panel_build.log",
                                mode="w", encoding="utf-8"),
        ],
        force=True,
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _key(s) -> str:
    """Cheap lowercase-strip normalizer for orphan detection."""
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# Stage 1 — clean SRE
# ---------------------------------------------------------------------------

def load_sre(path: Path) -> pd.DataFrame:
    LOG.info("Loading SRE panel: %s", path)
    sre = pd.read_parquet(path)
    LOG.info("  %d rows, %d cols", len(sre), len(sre.columns))

    if "psgc10" not in sre.columns:
        raise SystemExit("SRE panel has no psgc10 column — did the crosswalk run?")

    before = len(sre)

    # Drop unmatched (orphan) rows.
    sre = sre[sre["psgc10"].notna()].copy()
    LOG.info("  dropped %d rows with null psgc10", before - len(sre))

    # Drop placeholder names.
    sre["_orphan"] = sre["lgu_name"].map(_key).isin(ORPHAN_NAMES)
    n_orphan = int(sre["_orphan"].sum())
    if n_orphan:
        LOG.warning("  dropped %d orphan rows (PROVINCIAL TOTAL / blank names)",
                    n_orphan)
        LOG.debug("  orphan examples: %s",
                  sre.loc[sre["_orphan"], "lgu_name"].head(5).tolist())
    sre = sre[~sre["_orphan"]].drop(columns=["_orphan"])

    # Ensure fiscal_year is int.
    if "fiscal_year" not in sre.columns:
        raise SystemExit("SRE panel has no fiscal_year column")
    sre["fiscal_year"] = sre["fiscal_year"].astype(int)

        # --- Dedupe (psgc10, fiscal_year) ----------------------------------
    # When two source rows were matched to the same PSGC (e.g. the same
    # LGU appears twice with different province labels, one correct and
    # one garbage), we keep the row with the *lowest tier* — the best
    # match.  We do NOT sum: the duplicates carry the same fund_type, so
    # summing would double-count.
    dup_mask = sre.duplicated(subset=["psgc10", "fiscal_year"], keep=False)
    if dup_mask.any():
        n_dup_rows = int(dup_mask.sum())
        n_dup_groups = (sre.loc[dup_mask, ["psgc10", "fiscal_year"]]
                        .drop_duplicates()
                        .shape[0])
        LOG.warning("  %d rows share a (psgc10, fiscal_year) key across "
                    "%d groups — keeping lowest-tier row, dropping rest",
                    n_dup_rows, n_dup_groups)

        # Safety check: if any group has multiple fund_types, the caller
        # should know we're losing those rows.
        multi_fund = (sre.loc[dup_mask]
                      .groupby(["psgc10", "fiscal_year"])["fund_type"]
                      .nunique()
                      .gt(1))
        if multi_fund.any():
            LOG.warning("  %d groups have >1 fund_type — summing would be "
                        "correct there.  Manual review recommended.",
                        int(multi_fund.sum()))

        sre = (sre.sort_values(["psgc10", "fiscal_year", "tier"],
                               kind="stable")
                  .drop_duplicates(subset=["psgc10", "fiscal_year"],
                                   keep="first")
                  .reset_index(drop=True))
        LOG.info("  after tier-preferred dedup: %d rows", len(sre))

    return sre


# ---------------------------------------------------------------------------
# Stage 2 — Manila collapse on the PSA side
# ---------------------------------------------------------------------------

def collapse_manila(psa: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Collapse the 14 Manila district rows (all psgc10 == 1380600000) into
    a single row.  PI is averaged unweighted; CV/SE/CI are dropped.
    Returns (new_psa, n_districts_collapsed).
    """
    mask = psa["psgc10"] == MANILA_PSGC
    n = int(mask.sum())
    if n <= 1:
        return psa, 0

    LOG.info("Collapsing %d Manila district rows -> 1 (psgc10=%s)",
             n, MANILA_PSGC)

    manila = psa[mask]
    rest = psa[~mask].copy()

    row = {
        "lgu_name": "City of Manila",
        "province": "",
        "region": manila["region"].iloc[0],
        "lgu_type": "city",
        "psgc10": MANILA_PSGC,
        "tier": 1,
        "score": 1.0,
        "matched_name": f"aggregated from {n} Manila districts",
        "matched": True,
        "orphan": False,
    }
    # PI: unweighted mean across the districts.
    # CV / SE / CI: NaN (cannot be pooled without population weights).
    for y in SAE_YEARS:
        pi_col = f"pi_{y}"
        if pi_col in manila.columns:
            row[pi_col] = manila[pi_col].mean(skipna=True)
        for prefix in ("cv", "se", "ci_lo", "ci_hi"):
            col = f"{prefix}_{y}"
            if col in manila.columns:
                row[col] = np.nan

    new_row = pd.DataFrame([row])
    # Match the surrounding column order.
    for c in psa.columns:
        if c not in new_row.columns:
            new_row[c] = np.nan
    new_row = new_row[psa.columns]

    out = pd.concat([rest, new_row], ignore_index=True)
    return out, n


# ---------------------------------------------------------------------------
# Stage 3 — PSA long + wide forms
# ---------------------------------------------------------------------------

def psa_to_long(psa: pd.DataFrame) -> pd.DataFrame:
    """
    For each SAE year, produce a frame keyed (psgc10, fiscal_year) with
    columns poverty_pi, poverty_cv, poverty_se, poverty_ci_lo, poverty_ci_hi.
    """
    frames = []
    for y in SAE_YEARS:
        cols = [f"pi_{y}", f"cv_{y}", f"se_{y}", f"ci_lo_{y}", f"ci_hi_{y}"]
        missing = [c for c in cols if c not in psa.columns]
        if missing:
            LOG.warning("  PSA missing %s for year %d — skipping year",
                        missing, y)
            continue
        sub = psa[["psgc10"] + cols].copy()
        sub.columns = ["psgc10", "poverty_pi", "poverty_cv",
                       "poverty_se", "poverty_ci_lo", "poverty_ci_hi"]
        sub["fiscal_year"] = y
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    # If a psgc10 somehow appears twice per year (shouldn't after Manila
    # collapse), keep the first.
    long = long.drop_duplicates(subset=["psgc10", "fiscal_year"], keep="first")
    return long


def psa_to_wide(psa: pd.DataFrame) -> pd.DataFrame:
    """Broadcast form: one row per psgc10, one column per (field, year)."""
    cols = ["psgc10"] + [
        f"{field}_{y}"
        for y in SAE_YEARS
        for field in ("pi", "cv", "se", "ci_lo", "ci_hi")
        if f"{field}_{y}" in psa.columns
    ]
    wide = psa[cols].drop_duplicates(subset=["psgc10"], keep="first").copy()
    return wide


# ---------------------------------------------------------------------------
# Stage 4 — PSGC master identifiers
# ---------------------------------------------------------------------------

def load_master_ids(path: Path) -> pd.DataFrame:
    master = pd.read_parquet(path)
    cols_keep = ["psgc10", "name", "level",
                 "province_name", "region_name",
                 "income_class", "city_class", "urban_rural",
                 "is_independent_city", "pop_2024"]
    cols_keep = [c for c in cols_keep if c in master.columns]
    m = master[cols_keep].copy()
    m = m.rename(columns={
        "name":              "lgu_name",
        "level":             "lgu_type",
        "province_name":     "province",
        "region_name":       "region",
    })
    # For a province row, master.province_name is the province itself or blank.
    # Prefer the province's own name when self-referencing is missing.
    if "lgu_type" in m.columns and "province" in m.columns:
        is_prov = m["lgu_type"] == "province"
        m.loc[is_prov & (m["province"] == ""), "province"] = \
            m.loc[is_prov & (m["province"] == ""), "lgu_name"]
    return m


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

def write_qa(panel: pd.DataFrame, out_path: Path,
             n_manila_collapsed: int) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("PANEL QA")
    lines.append("=" * 78)
    lines.append(f"Rows: {len(panel):,}")
    lines.append(f"Cols: {len(panel.columns)}")
    lines.append("")

    lines.append("-- Years --")
    yr = panel["fiscal_year"].value_counts().sort_index()
    lines.append(f"  min={yr.index.min()}  max={yr.index.max()}  "
                 f"n_years={len(yr)}")
    lines.append(f"  rows/year  min={yr.min()}  median={int(yr.median())}  "
                 f"max={yr.max()}")
    lines.append("")

    lines.append("-- LGUs --")
    lines.append(f"  unique psgc10: {panel['psgc10'].nunique()}")
    if "lgu_type" in panel.columns:
        for lvl, n in panel.groupby("lgu_type")["psgc10"].nunique().items():
            lines.append(f"    {lvl:15s} {n}")
    if "region" in panel.columns:
        lines.append(f"  unique regions: {panel['region'].nunique()}")
    lines.append("")

    lines.append("-- Source coverage --")
    if "in_sre" in panel.columns and "in_psa" in panel.columns:
        both = ((panel["in_sre"]) & (panel["in_psa"])).sum()
        only_sre = ((panel["in_sre"]) & (~panel["in_psa"])).sum()
        only_psa = ((~panel["in_sre"]) & (panel["in_psa"])).sum()
        lines.append(f"  in both: {both:,}")
        lines.append(f"  only SRE: {only_sre:,}")
        lines.append(f"  only PSA: {only_psa:,}")
    lines.append("")

    lines.append("-- Poverty coverage --")
    if "poverty_pi" in panel.columns:
        n_pov = int(panel["poverty_pi"].notna().sum())
        lines.append(f"  poverty_pi non-null: {n_pov:,} "
                     f"({100 * n_pov / len(panel):.2f}%)")
        for y in SAE_YEARS:
            sub = panel[panel["fiscal_year"] == y]
            n = int(sub["poverty_pi"].notna().sum())
            lines.append(f"    {y}: {n:,} LGUs with PI")
    if "pi_2018" in panel.columns:
        n_bc = int(panel["pi_2018"].notna().sum())
        lines.append(f"  pi_2018 broadcast non-null: {n_bc:,}")
    lines.append("")

    if n_manila_collapsed:
        lines.append(f"-- Manila --")
        lines.append(f"  collapsed {n_manila_collapsed} district rows "
                     f"-> 1 (psgc10={MANILA_PSGC})")
        lines.append("")

    lines.append("-- Top-of-file sample --")
    show_cols = [c for c in ("psgc10", "lgu_name", "province", "region",
                             "lgu_type", "fiscal_year",
                             "nta_ira", "poverty_pi")
                 if c in panel.columns]
    with pd.option_context("display.width", 160,
                           "display.max_columns", 20):
        lines.append(panel[show_cols].head(10).to_string(index=False))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sre",  type=Path, default=DEFAULT_SRE)
    p.add_argument("--psa",  type=Path, default=DEFAULT_PSA)
    p.add_argument("--psgc", type=Path, default=DEFAULT_PSGC)
    p.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--keep-manila-districts", action="store_true",
                   help="Do NOT collapse the 14 Manila district rows. "
                        "You will get 14 rows sharing psgc10=1380600000 "
                        "on each SAE year.")
    p.add_argument("--csv", action="store_true",
                   help="Also write panel.csv (large, ~50 MB).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(args.outdir, verbose=args.verbose)

    for label, path in (("SRE", args.sre), ("PSA", args.psa),
                        ("PSGC", args.psgc)):
        if not path.exists():
            LOG.error("%s input not found: %s", label, path)
            return 1

    # --- 1. Load & clean SRE -------------------------------------------
    sre = load_sre(args.sre)
    sre["in_sre"] = True

    # --- 2. Load & clean PSA -------------------------------------------
    LOG.info("Loading PSA panel: %s", args.psa)
    psa = pd.read_parquet(args.psa)
    LOG.info("  %d rows, %d cols", len(psa), len(psa.columns))
    psa = psa[psa["psgc10"].notna()].copy()

    n_manila = 0
    if not args.keep_manila_districts:
        psa, n_manila = collapse_manila(psa)
    else:
        LOG.info("Keeping Manila districts as-is (%d rows)",
                 (psa["psgc10"] == MANILA_PSGC).sum())

    # --- 3. PSA long + wide --------------------------------------------
    LOG.info("Pivoting PSA to long form")
    psa_long = psa_to_long(psa)
    LOG.info("  %d (psgc10, fiscal_year) poverty rows", len(psa_long))

    LOG.info("Building PSA wide (broadcast) form")
    psa_wide = psa_to_wide(psa)
    LOG.info("  %d LGUs with a broadcast poverty block", len(psa_wide))

    # --- 4. PSGC master identifiers ------------------------------------
    LOG.info("Loading PSGC master: %s", args.psgc)
    master = load_master_ids(args.psgc)
    LOG.info("  %d master rows", len(master))

    # --- 5. Merge -------------------------------------------------------
    LOG.info("Merging: SRE base + master ids + PSA long + PSA wide")

    panel = sre.merge(master, on="psgc10", how="left", suffixes=("_sre", ""))
    # After this merge: 'lgu_name', 'province', 'region', 'lgu_type' come
    # from the master.  The SRE copies got '_sre' suffixed.
    LOG.info("  after master merge: %d rows", len(panel))

    panel = panel.merge(psa_long, on=["psgc10", "fiscal_year"], how="left")
    LOG.info("  after PSA long merge: %d rows", len(panel))

    panel = panel.merge(psa_wide, on="psgc10", how="left")
    LOG.info("  after PSA wide merge: %d rows", len(panel))

    # Flag PSA membership (any of the three SAE years with non-null PI).
    pi_cols = [f"pi_{y}" for y in SAE_YEARS if f"pi_{y}" in panel.columns]
    panel["in_psa"] = panel[pi_cols].notna().any(axis=1)

    # --- 6. Tidy columns ------------------------------------------------
    # Canonical column order.
    id_cols = ["psgc10", "lgu_name", "province", "region", "lgu_type",
               "income_class", "city_class", "urban_rural",
               "is_independent_city", "pop_2024",
               "fiscal_year", "fund_type", "template_era",
               "in_sre", "in_psa"]

    # SRE-specific fiscal columns (present in panel after the merge).
    fiscal_cols = [c for c in panel.columns
                   if c not in id_cols
                   and not c.endswith("_sre")
                   and c not in (
                       "tier", "score", "matched", "orphan",
                       "matched_name", "source_file",
                   )
                   and not c.startswith("poverty_")
                   and not any(c == f"{fld}_{y}"
                               for y in SAE_YEARS
                               for fld in ("pi", "cv", "se",
                                           "ci_lo", "ci_hi"))]

    poverty_current = ["poverty_pi", "poverty_cv", "poverty_se",
                       "poverty_ci_lo", "poverty_ci_hi"]
    poverty_broadcast = [f"{fld}_{y}"
                         for y in SAE_YEARS
                         for fld in ("pi", "cv", "se", "ci_lo", "ci_hi")
                         if f"{fld}_{y}" in panel.columns]

    ordered = []
    for grp in (id_cols, fiscal_cols, poverty_current, poverty_broadcast):
        for c in grp:
            if c in panel.columns and c not in ordered:
                ordered.append(c)
    # Append any remaining columns at the end.
    for c in panel.columns:
        if c not in ordered:
            ordered.append(c)

    panel = panel[ordered]

    # Drop "_sre"-suffixed duplicates of identifier columns (they're now
    # redundant — the master values are authoritative).
    drop_sre = [c for c in panel.columns
                if c.endswith("_sre")
                and c[:-4] in ("lgu_name", "province", "region", "lgu_type")]
    if drop_sre:
        LOG.debug("Dropping redundant SRE-side identifiers: %s", drop_sre)
        panel = panel.drop(columns=drop_sre)

    # Sort.
    panel = panel.sort_values(["psgc10", "fiscal_year"]).reset_index(drop=True)

    # --- 7. Write -------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    LOG.info("Wrote %s (%d rows x %d cols)", args.out,
             len(panel), len(panel.columns))

    if args.csv:
        csv_path = args.out.with_suffix(".csv")
        panel.to_csv(csv_path, index=False)
        LOG.info("Wrote %s", csv_path)

    write_qa(panel, args.outdir / "panel_qa.txt", n_manila)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())