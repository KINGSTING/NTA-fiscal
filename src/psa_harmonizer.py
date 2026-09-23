#!/usr/bin/env python
"""
psa_harmonizer.py
=================
Attach a PSGC code to every row of the PSA panel.

Changes vs. previous version
----------------------------
1. NCR province normalization.
2. Manila-district fast path — try City of Manila first for known
   districts in NCR rows, bypassing the fuzzy matcher.
3. Province-scoped aliases (Carmen, Surigao del Norte → Del Carmen).
4. NCR-tuple diagnostic line.
5. Removed a dead duplicate body of `match_with_fallback` that was
   accidentally left below the first return statement.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sre_psgc_crosswalk import (          # noqa: E402
    DEFAULT_PSGC,
    DEFAULT_OUT,
    LOG as _SRE_LOG,
    build_psgc_lookups,
    make_key,
    match_one,
    diagnose_unmatched,
    lgu_type_norm,   # noqa: F401
)

DEFAULT_PSA = Path("data/processed/psa_panel.parquet")

LOG = logging.getLogger("psa_harmonizer")


# ---------------------------------------------------------------------------
# NCR helpers
# ---------------------------------------------------------------------------

MANILA_DISTRICTS = {
    "tondo", "binondo", "quiapo", "san nicolas", "santa cruz",
    "sampaloc", "san miguel", "ermita", "intramuros", "malate",
    "paco", "pandacan", "port area", "santa ana",
}

MANILA_CITY_ALIASES = ("City of Manila", "Manila")

# Province-scoped name aliases applied before match_one.
# Keys are (make_key(lgu_name), make_key(province)).
SCOPED_NAME_ALIASES = {
    ("carmen", "surigao del norte"): "Del Carmen",
}


def is_ncr(region: str) -> bool:
    r = str(region).lower()
    return "ncr" in r or "national capital" in r


def normalize_province_for_ncr(province: str, region: str) -> str:
    if not is_ncr(region):
        return province
    p = str(province).strip().lower()
    if not p or p in {"nan", "none"} or "district" in p or "continued" in p:
        return ""
    return province


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------

LGU_COL_CANDIDATES = [
    "lgu_name", "municipality", "city_mun", "citymun", "municipality_city",
    "city_municipality", "geo_name", "name",
]
PROV_COL_CANDIDATES = ["province", "province_name", "prov_name", "prov"]
REGION_COL_CANDIDATES = ["region", "region_name", "reg_name", "reg"]
TYPE_COL_CANDIDATES = [
    "lgu_type", "level", "geo_level", "type", "classification",
]
PSGC_COL_CANDIDATES = ["psgc10", "psgc", "psgc_code", "geocode", "geo_code"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def resolve_schema(df, lgu_col, prov_col, region_col, type_col, psgc_col) -> dict:
    if psgc_col is None:
        psgc_col = _find_col(df, PSGC_COL_CANDIDATES)

    if psgc_col is not None:
        LOG.info("PSA panel already has a PSGC column: %r — will use it directly.",
                 psgc_col)
        return {"lgu_col": lgu_col, "prov_col": prov_col,
                "region_col": region_col, "type_col": type_col,
                "psgc_col": psgc_col}

    if lgu_col is None:
        lgu_col = _find_col(df, LGU_COL_CANDIDATES)
    if prov_col is None:
        prov_col = _find_col(df, PROV_COL_CANDIDATES)
    if region_col is None:
        region_col = _find_col(df, REGION_COL_CANDIDATES)
    if type_col is None:
        type_col = _find_col(df, TYPE_COL_CANDIDATES)

    missing = [name for name, val in
               (("lgu-name", lgu_col), ("province", prov_col)) if val is None]
    if missing:
        raise SystemExit(
            f"Could not auto-detect column(s): {', '.join(missing)}.\n"
            f"PSA columns present: {list(df.columns)}\n"
            f"Re-run with --lgu-col / --province-col / --region-col / --type-col."
        )

    return {"lgu_col": lgu_col, "prov_col": prov_col,
            "region_col": region_col, "type_col": type_col,
            "psgc_col": None}


# ---------------------------------------------------------------------------
# Orphan handling
# ---------------------------------------------------------------------------

ORPHAN_NAMES = {
    "provincial total", "regional total", "national total",
    "total", "subtotal", "grand total",
    "", "`", "-", "n/a", "na", "none", "null",
}


def flag_orphans(df: pd.DataFrame, lgu_col: str) -> pd.Series:
    keys = df[lgu_col].map(make_key)
    return keys.isin(ORPHAN_NAMES)


# ---------------------------------------------------------------------------
# Normalized key frame + fallback match
# ---------------------------------------------------------------------------

def build_norm_frame(psa: pd.DataFrame, schema: dict) -> pd.DataFrame:
    n = len(psa)
    blank = pd.Series([""] * n, index=psa.index)

    norm = pd.DataFrame({
        "lgu_name": psa[schema["lgu_col"]].astype(str),
        "province": psa[schema["prov_col"]].astype(str) if schema["prov_col"] else blank,
        "region":   psa[schema["region_col"]].astype(str) if schema["region_col"] else blank,
        "lgu_type": psa[schema["type_col"]].astype(str) if schema["type_col"] else blank.replace("", "municipality"),
    })

    ncr_mask = norm["region"].map(is_ncr)
    if ncr_mask.any():
        norm.loc[ncr_mask, "province"] = [
            normalize_province_for_ncr(p, r)
            for p, r in zip(norm.loc[ncr_mask, "province"],
                            norm.loc[ncr_mask, "region"])
        ]
    return norm


def match_with_fallback(lgu_name, province, lgu_type, region, L) -> dict:
    """
    Order of operations:
      1. Province-scoped alias (Carmen, Surigao del Norte → Del Carmen).
      2. Manila district fast path (Tondo, Binondo, ... → City of Manila).
      3. Standard tiered match.
    """
    # 1. Scoped alias
    alias_key = (make_key(lgu_name), make_key(province))
    if alias_key in SCOPED_NAME_ALIASES:
        lgu_name = SCOPED_NAME_ALIASES[alias_key]

    # 2. Manila district fast path
    if is_ncr(region) and make_key(lgu_name) in MANILA_DISTRICTS:
        for cand in MANILA_CITY_ALIASES:
            m = match_one(cand, "", "city", L)
            if m.get("psgc10") is not None:
                return {
                    **m,
                    "matched_name": f"{cand} (via Manila district {lgu_name})",
                }

    # 3. Standard tiered match
    return match_one(lgu_name, province, lgu_type, L)


# ---------------------------------------------------------------------------
# Crosswalk driver
# ---------------------------------------------------------------------------

def build_crosswalk(norm: pd.DataFrame, L: dict) -> pd.DataFrame:
    uniq = norm.drop_duplicates().reset_index(drop=True)
    LOG.info("Distinct LGU tuples to match: %d", len(uniq))

    rows = []
    for i, r in uniq.iterrows():
        m = match_with_fallback(r["lgu_name"], r["province"],
                                r["lgu_type"], r["region"], L)
        rows.append({
            "lgu_name": r["lgu_name"], "province": r["province"],
            "region": r["region"], "lgu_type": r["lgu_type"],
            "psgc10": m.get("psgc10"), "tier": m.get("tier"),
            "score": m.get("score"), "matched_name": m.get("matched_name"),
        })
        if (i + 1) % 500 == 0:
            LOG.info("  matched %d / %d", i + 1, len(uniq))

    xw = pd.DataFrame(rows)
    xw["matched"] = xw["psgc10"].notna()
    xw["orphan"]  = flag_orphans(xw, "lgu_name")
    xw.loc[xw["orphan"], "tier"] = -1
    xw.loc[xw["orphan"], "matched"] = False
    return xw


# ---------------------------------------------------------------------------
# Diagnostic wrapper
# ---------------------------------------------------------------------------

def diagnose(psa_unmatched, L, out_path: Path, n_per: int = 8) -> None:
    diag = diagnose_unmatched(psa_unmatched, L, n_per=n_per)
    out_path.write_text(diag, encoding="utf-8")
    LOG.info("Wrote %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psa",    type=Path, default=DEFAULT_PSA)
    p.add_argument("--psgc",   type=Path, default=DEFAULT_PSGC)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)

    p.add_argument("--lgu-col",      type=str, default=None)
    p.add_argument("--province-col", type=str, default=None)
    p.add_argument("--region-col",   type=str, default=None)
    p.add_argument("--type-col",     type=str, default=None)
    p.add_argument("--psgc-col",     type=str, default=None)

    p.add_argument("--diagnose", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.outdir / "psa_psgc_crosswalk.log",
                                mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    _SRE_LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    LOG.info("Loading PSA panel: %s", args.psa)
    psa = pd.read_parquet(args.psa)
    LOG.info("  %d rows, %d columns", len(psa), len(psa.columns))
    LOG.debug("PSA columns: %s", list(psa.columns))

    schema = resolve_schema(psa, args.lgu_col, args.province_col,
                            args.region_col, args.type_col, args.psgc_col)
    LOG.info("Schema resolved: %s", schema)

    LOG.info("Loading PSGC master: %s", args.psgc)
    psgc = pd.read_parquet(args.psgc)
    LOG.info("  %d LGU rows", len(psgc))

    # Fast path: already has PSGC
    if schema["psgc_col"] is not None:
        psgc_col = schema["psgc_col"]
        out = psa.copy()
        out["psgc10"] = out[psgc_col].astype(str).str.zfill(10)
        out["matched"] = out["psgc10"].isin(psgc["psgc10"])
        out["tier"] = np.where(out["matched"], 0, -1)
        out["score"] = np.where(out["matched"], 1.0, 0.0)
        out["matched_name"] = None
        LOG.info("Pass-through match rate: %.2f%%", out["matched"].mean() * 100)
        out.to_parquet(args.outdir / "psa_panel_with_psgc.parquet", index=False)
        out.to_csv(args.outdir / "psa_panel_with_psgc.csv", index=False)
        LOG.info("Wrote psa_panel_with_psgc.parquet / .csv")
        return 0

    norm = build_norm_frame(psa, schema)
    L  = build_psgc_lookups(psgc)
    xw = build_crosswalk(norm, L)

    real = xw[~xw["orphan"]]
    LOG.info("Distinct tuple match rate (all):      %.2f%%", xw["matched"].mean() * 100)
    LOG.info("Distinct tuple match rate (real LGU): %.4f%%  (%d / %d)",
             real["matched"].mean() * 100,
             int(real["matched"].sum()), len(real))
    LOG.info("Tier distribution:\n%s",
             xw["tier"].value_counts().sort_index().to_string())

    ncr_xw = xw[xw["region"].map(is_ncr)]
    if len(ncr_xw):
        LOG.info("NCR tuples: %d  matched: %d  (%.2f%%)",
                 len(ncr_xw), int(ncr_xw["matched"].sum()),
                 ncr_xw["matched"].mean() * 100)
        LOG.debug("NCR crosswalk:\n%s",
                  ncr_xw[["lgu_name", "province", "psgc10", "tier",
                          "matched_name"]].to_string(index=False))

    xw.to_parquet(args.outdir / "psa_psgc_crosswalk.parquet", index=False)
    xw.to_csv(args.outdir / "psa_psgc_crosswalk.csv", index=False)

    unmatched = xw[~xw["matched"] & ~xw["orphan"]].copy()
    unmatched.to_csv(args.outdir / "psa_unmatched.csv", index=False)
    LOG.info("Unmatched (real-LGU) tuples: %d", len(unmatched))

    if len(unmatched):
        rank = (norm
                .merge(unmatched[["lgu_name", "province", "region", "lgu_type"]],
                       on=["lgu_name", "province", "region", "lgu_type"],
                       how="inner")
                .groupby(["lgu_name", "province", "region", "lgu_type"])
                .size().reset_index(name="n_rows")
                .sort_values("n_rows", ascending=False))
        rank.to_csv(args.outdir / "psa_unmatched_ranked.csv", index=False)
        LOG.info("Top 30 unmatched:\n%s", rank.head(30).to_string(index=False))

        if args.diagnose:
            diagnose(unmatched, L,
                     args.outdir / "psa_unmatched_diagnostic.txt", n_per=8)

    LOG.info("Merging back to panel...")
    merge_cols = ["lgu_name", "province", "region", "lgu_type",
                  "psgc10", "tier", "score", "matched_name", "matched", "orphan"]

    merged = psa.copy()
    for c in ("psgc10", "tier", "score", "matched_name", "matched", "orphan"):
        merged[c] = None

    left = norm.merge(xw[merge_cols],
                      on=["lgu_name", "province", "region", "lgu_type"],
                      how="left")
    for c in ("psgc10", "tier", "score", "matched_name", "matched", "orphan"):
        merged[c] = left[c].values

    real_rows = merged[~merged["orphan"].fillna(False)]
    row_match = real_rows["psgc10"].notna().mean() * 100
    LOG.info("Panel row match rate (real LGU): %.4f%%", row_match)

    merged.to_parquet(args.outdir / "psa_panel_with_psgc.parquet", index=False)
    merged.to_csv(args.outdir / "psa_panel_with_psgc.csv", index=False)
    LOG.info("Wrote psa_panel_with_psgc.parquet / .csv (%d rows)", len(merged))

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())