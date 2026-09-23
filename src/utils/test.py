import pandas as pd
sre = pd.read_csv("data/processed/sre_panel.csv", dtype=str, nrows=200_000)

print("lgu_type values:", sre["lgu_type"].value_counts(dropna=False).to_dict())
print("rows:", len(sre))

print("\nunique regions:", sre["region"].nunique())
print("unique provinces:", sre["province"].nunique())
print("unique lgu_name:", sre["lgu_name"].nunique())

print("\n20 sample lgu_name by type:")
for t in sre["lgu_type"].dropna().unique():
    print(f"\n--- {t} ---")
    print(sre.loc[sre["lgu_type"] == t, "lgu_name"].dropna().sample(
        min(15, (sre["lgu_type"] == t).sum()), random_state=1).tolist())

print("\n5 sample provinces:")
print(sre["province"].dropna().unique()[:20].tolist())

print("\n5 sample regions:")
print(sre["region"].dropna().unique()[:20].tolist())

print("\nrows per fiscal_year:")
print(sre["fiscal_year"].value_counts().sort_index().to_dict())