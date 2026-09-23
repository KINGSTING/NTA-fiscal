"""One-off diagnostic: what are the duplicate (psgc10, fiscal_year) rows?"""
import pandas as pd

sre = pd.read_parquet(r"data/processed/sre_panel_with_psgc.parquet")
sre = sre[sre["psgc10"].notna()].copy()
sre["fiscal_year"] = sre["fiscal_year"].astype(int)

dup = sre[sre.duplicated(subset=["psgc10", "fiscal_year"], keep=False)]

print(f"Total duplicate rows:          {len(dup)}")
print(f"Distinct (psgc10, fy) groups:  {dup.groupby(['psgc10','fiscal_year']).ngroups}")
print()

id_cols = [c for c in ("psgc10", "fiscal_year", "lgu_name", "fund_type",
                       "province", "lgu_type", "template_era", "source_file")
           if c in dup.columns]

print("--- First 15 duplicate groups ---")
for (pg, fy), g in list(dup.groupby(["psgc10", "fiscal_year"]))[:15]:
    print(f"\n=== psgc10={pg}  fiscal_year={fy} ===")
    print(g[id_cols].to_string(index=False))

print("\n--- Distribution of fund_type within duplicate groups ---")
by_group = (dup.groupby(["psgc10", "fiscal_year"])["fund_type"]
              .apply(lambda s: " | ".join(sorted(s.astype(str).unique())))
              .value_counts())
print(by_group.head(20).to_string())