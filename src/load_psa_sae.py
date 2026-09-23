#!/usr/bin/env python
"""
load_psa_sae.py
===============
Convert the PSA 2023 SAE workbook (old 6-digit PSGC codes) into a
harmonizer-friendly parquet file.

Changes vs. previous version
----------------------------
1. NCR province blanked.
2. infer_type() reads KNOWN_CITY_NAMES.
3. Continuation-marker filter ( "(Continued)" page breaks ).
4. Region-boundary reset in province forward-fill.  Prevents a province
   header from leaking past the end of its region into the next one
   (e.g. "Surigao del Norte" bleeding into the Surigao del Sur block).
5. --debug-rows N.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SRC = Path("data/PSA/2_2023 SAE_with PSGC_noHUC_06Feb2026.xlsx")
DEFAULT_OUT = Path("data/processed/psa_panel.parquet")

COLS = [
    "psgc", "region_or_province", "municipality_city",
    "pi_2018", "pi_2021", "pi_2023",
    "cv_2018", "cv_2021", "cv_2023",
    "se_2018", "se_2021", "se_2023",
    "ci_lo_2018", "ci_hi_2018",
    "ci_lo_2021", "ci_hi_2021",
    "ci_lo_2023", "ci_hi_2023",
]
HEADER_ROWS = 5


# ---------------------------------------------------------------------------
# LGU type inference
# ---------------------------------------------------------------------------

KNOWN_CITY_NAMES = {
    "caloocan", "las pinas", "makati", "malabon", "mandaluyong",
    "manila", "marikina", "muntinlupa", "navotas", "paranaque",
    "pasay", "pasig", "quezon", "san juan", "taguig", "valenzuela",
    "angeles", "antipolo", "bacoor", "balanga", "batac", "binan",
    "biñan", "cabanatuan", "cabuyao", "calamba", "candon", "cauayan",
    "dagupan", "dasmarinas", "gapan", "ilagan", "imus", "laoag",
    "legazpi", "lucena", "mabalacat", "malolos", "meycauayan",
    "munoz", "olongapo", "palayan", "san carlos", "san fernando",
    "san jose", "san jose del monte", "san pablo", "santa rosa",
    "santiago", "tabuk", "tacloban", "tagaytay", "tarlac",
    "trece martires", "tuguegarao", "urdaneta", "vigan",
    "bacolod", "bago", "bayawan", "baybay", "bogo", "cadiz",
    "canlaon", "carcar", "danao", "dumaguete", "escalante",
    "guihulngan", "himamaylan", "iloilo", "kabankalan", "la carlota",
    "lapu-lapu", "maasin", "mandaue", "naga", "oroc", "passi",
    "rojas", "sagay", "silay", "sipalay", "talisay", "tagbilaran",
    "toledo", "victorias",
    "butuan", "cagayan de oro", "cotabato", "davao", "digos",
    "dipolog", "general santos", "gingoog", "iligan", "isabela",
    "kidapawan", "koronadal", "marawi", "pagadian", "panabo",
    "samal", "surigao", "tacurong", "tagum", "tangub", "valencia",
    "zamboanga",
}


def infer_type(name: str) -> str:
    n = str(name).strip().lower()
    if not n:
        return "municipality"
    if n.startswith("city of ") or n.endswith(" city") or " city " in n:
        return "city"
    key = n.replace(" city", "").strip()
    if key in KNOWN_CITY_NAMES:
        return "city"
    return "municipality"


# ---------------------------------------------------------------------------
# NCR + continuation helpers
# ---------------------------------------------------------------------------

def is_ncr(region: str) -> bool:
    r = str(region).lower()
    return "ncr" in r or "national capital" in r


CONTINUATION_TOKENS = ("continued", "(cont", "cont.)")


def is_continuation(val) -> bool:
    s = str(val).strip().lower()
    if not s:
        return False
    return any(tok in s for tok in CONTINUATION_TOKENS)


# ---------------------------------------------------------------------------
# Sheet picker
# ---------------------------------------------------------------------------

def pick_sheet(src: Path):
    xl = pd.ExcelFile(src)
    best, best_n = xl.sheet_names[0], -1
    for name in xl.sheet_names:
        try:
            n = len(pd.read_excel(src, sheet_name=name, header=None))
        except Exception:
            continue
        if n > best_n:
            best, best_n = name, n
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--sheet", default=None,
                    help="Sheet name or index. Default: largest sheet.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--debug-rows", type=int, default=0,
                    help="Dump the first N raw rows (cols A..C) for inspection.")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source workbook not found: {args.src}")

    sheet = args.sheet if args.sheet is not None else pick_sheet(args.src)
    print(f"Reading sheet: {sheet!r} from {args.src}")

    raw = pd.read_excel(args.src, sheet_name=sheet, header=None,
                        skiprows=HEADER_ROWS)
    raw = raw.iloc[:, :len(COLS)]
    raw.columns = COLS

    if args.debug_rows:
        print(f"\n--- raw head ({args.debug_rows} rows, cols A..C) ---")
        with pd.option_context("display.max_colwidth", 60):
            print(raw[["psgc", "region_or_province", "municipality_city"]]
                  .head(args.debug_rows).to_string(index=False))
        print("--- end raw head ---\n")

    df = raw.dropna(how="all").copy()

    for c in ("psgc", "region_or_province", "municipality_city"):
        df[c] = df[c].fillna("").astype(str).str.strip()
        df[c] = df[c].replace({"nan": "", "None": ""})

    # Region divider rows
    is_region_row = (df["psgc"] == "") & (df["municipality_city"] == "")

    df["region"] = ""
    df.loc[is_region_row, "region"] = df.loc[is_region_row, "region_or_province"]
    df["region"] = df["region"].replace("", pd.NA).ffill().fillna("")

    # Defensive: blank continuation tokens misclassified as region dividers
    cont_mask = df["region"].map(is_continuation)
    if cont_mask.any():
        df.loc[cont_mask, "region"] = pd.NA
        df["region"] = df["region"].ffill().fillna("")

    region_set = set(df.loc[is_region_row, "region"].unique())

    # --- Forward-fill province -------------------------------------------
    # Column B carries the province on the first row of each group.
    # Two guards:
    #   * reset cur when we cross into a new region, so a province header
    #     cannot leak past its region boundary (Surigao del Norte → Sur);
    #   * skip continuation markers and NCR (whose column B is a district).
    prov: list[str] = []
    cur = ""
    region_at_cur = ""
    for b, r in zip(df["region_or_province"], df["region"]):
        if r != region_at_cur:
            cur = ""
            region_at_cur = r
        if is_ncr(r):
            prov.append("")
            continue
        if b and b not in region_set and not is_continuation(b):
            cur = b
        prov.append(cur)
    df["province"] = prov

    # Keep real LGU rows only
    df = df[df["psgc"].str.match(r"^\d{4,10}$")].copy()
    df = df.drop(columns=["psgc", "region_or_province"])

    df["lgu_type"] = df["municipality_city"].map(infer_type)
    df = df.rename(columns={"municipality_city": "lgu_name"})

    first = ["lgu_name", "province", "region", "lgu_type"]
    df = df[first + [c for c in df.columns if c not in first]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    # --- Diagnostics -----------------------------------------------------
    print(f"\nWrote {args.out}")
    print(f"  rows: {len(df)}")
    print(f"  cols: {list(df.columns)}")
    print(f"\nHead:\n{df.head(8).to_string(index=False)}")
    print(f"\nlgu_type distribution:\n{df['lgu_type'].value_counts().to_string()}")
    print(f"\nregion distribution:\n{df['region'].value_counts().to_string()}")

    leaked = df[df["province"].str.contains("continued", case=False, na=False)]
    if len(leaked):
        print("\n!! WARNING: continuation tokens leaked into province:")
        for p in leaked["province"].unique():
            print(f"     {p!r}  ({len(leaked[leaked['province'] == p])} rows)")

    ncr = df[df["region"].map(is_ncr)]
    print(f"\nNCR rows: {len(ncr)}  (17 = 16 cities + Pateros; "
          f"14 = Manila districts only)")
    if len(ncr) < 17:
        print("  !! NCR is under-represented (expected for *_noHUC_* workbook).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())