"""
diagnose_2024.py — Inspect raw SRE file and harmonized panel side by side.
Run from project root: python src\diagnose_2024.py
"""
from pathlib import Path
import pandas as pd

# --- Resolve paths relative to project root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "SRE" / "By-LGU-SRE-2024.xlsx"
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "sre_2024.parquet"

# --- Load raw file ---
print("=" * 78)
print("RAW FILE DIAGNOSTICS")
print("=" * 78)
raw = pd.read_excel(RAW_PATH, header=None)
print(f"Shape: {raw.shape}")

# --- Last 5 rows (look for grand total) ---
print("\n--- Last 5 rows ---")
for i in range(len(raw) - 5, len(raw)):
    row = raw.iloc[i]
    cells = [str(v)[:40] for v in row if pd.notna(v)]
    print(f"Row {i}: {cells[:8]}")

# --- Load harmonized panel ---
print("\n" + "=" * 78)
print("HARMONIZED PANEL DIAGNOSTICS")
print("=" * 78)
df = pd.read_parquet(PANEL_PATH)
print(f"Shape: {df.shape}")
print(f"Rows: {len(df)}")

# --- NTA stats ---
print("\n--- NTA (raw, in whatever units the file used) ---")
print(f"Non-null: {df['nta_ira'].notna().sum()} / {len(df)}")
print(f"Null:     {df['nta_ira'].isna().sum()}")
print(f"Zero:     {(df['nta_ira'] == 0).sum()}")
print(f"Negative: {(df['nta_ira'] < 0).sum()}")
print(f"Sum:      {df['nta_ira'].sum():,.2f}")
print(f"Median:   {df['nta_ira'].median():,.2f}")
print(f"Min:      {df['nta_ira'].min():,.2f}")
print(f"Max:      {df['nta_ira'].max():,.2f}")

# --- By LGU type ---
print("\n--- NTA by LGU type ---")
if "lgu_type" in df.columns:
    print(df.groupby("lgu_type")["nta_ira"].agg(["count", "sum"]).to_string())

# --- By region ---
print("\n--- NTA by region (all regions) ---")
if "region" in df.columns:
    by_region = df.groupby("region")["nta_ira"].agg(["count", "sum"])
    print(by_region.sort_values("sum", ascending=False).to_string())

# --- Other shares check ---
if "other_national_shares" in df.columns:
    other = df["other_national_shares"].sum()
    print(f"\nOther Shares from National Tax Collection: {other:,.2f}")

# --- Row composition ---
print("\n--- Row accounting ---")
if "lgu_type" in df.columns:
    lt = df["lgu_type"].str.lower()
    print(f"Provinces:      {(lt == 'province').sum()}")
    print(f"Cities:         {(lt == 'city').sum()}")
    print(f"Municipalities: {(lt == 'municipality').sum()}")
    print(f"Other:          {len(df) - lt.isin(['province', 'city', 'municipality']).sum()}")

# --- Duplicate rows check ---
print("\n--- Duplicate (region, province, lgu_name, lgu_type) rows ---")
key_cols = ["region", "province", "lgu_name", "lgu_type"]
if all(c in df.columns for c in key_cols):
    dup = df.duplicated(subset=key_cols, keep=False).sum()
    print(f"Duplicates: {dup}")
    if dup > 0:
        print("\nFirst 10 duplicates:")
        print(df[df.duplicated(subset=key_cols, keep=False)]
              .sort_values(key_cols)
              .head(10)[key_cols + ["fund_type", "nta_ira"]].to_string())

# --- Show first 5 rows ---
print("\n--- First 5 rows of panel ---")
show_cols = ["region", "province", "lgu_name", "lgu_type", "nta_ira",
             "total_current_operating_income"]
show_cols = [c for c in show_cols if c in df.columns]
print(df[show_cols].head(5).to_string())

print("\n" + "=" * 78)