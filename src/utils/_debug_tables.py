"""Quick diagnostic for the PSA tables that fetch_population.py skipped or partly matched."""
import re
import sys
from pathlib import Path
from unicodedata import normalize

import pandas as pd

CENSUS = [2000, 2007, 2010, 2015, 2020, 2024]
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def clean(s) -> str:
    if s is None:
        return ""
    x = str(s).replace("\u00a0", " ").replace("牋", " ")
    return _WS.sub(" ", x).strip()


def key(s) -> str:
    x = normalize("NFKD", clean(s)).encode("ascii", "ignore").decode().lower()
    x = _PUNCT.sub(" ", x)
    return _WS.sub(" ", x).strip()


def load(path: Path) -> pd.DataFrame:
    """Same encoding fallback as fetch_population._load_csv."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, header=None, dtype=str,
                               encoding=enc, on_bad_lines="skip")
        except Exception as e:
            print(f"  [load] {path.name}: {enc} failed ({type(e).__name__})")
    raise RuntimeError(f"Could not read {path}")


# ---------------------------------------------------------------------------
# 1. What does T1_3 actually look like?
# ---------------------------------------------------------------------------
print("=" * 78)
print("T1_3 HEADER LAYOUT")
print("=" * 78)
t13 = Path("data/PSA/2025_T1_3.csv")
df3 = load(t13)
print(f"{t13.name}:  shape = {df3.shape}\n")

print("-- first 15 rows, all columns --")
with pd.option_context("display.max_columns", None,
                       "display.width", 250,
                       "display.max_colwidth", 22):
    print(df3.head(15).to_string())

print("\n-- rows 0-25 containing any census year --")
for i in range(min(26, len(df3))):
    joined = " | ".join(str(v) for v in df3.iloc[i].tolist())
    hits = sum(1 for y in CENSUS if str(y) in joined)
    if hits:
        print(f"  row {i:2d}: {hits} year-hits | {joined[:200]}")


# ---------------------------------------------------------------------------
# 2. Which provinces in T1_1 failed to match the master?
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PROVINCE MATCHING CHECK (T1_1 vs master)")
print("=" * 78)

master = pd.read_parquet("data/processed/psgc_lgu_master.parquet")
provs = master[master["level"] == "province"].copy()
print(f"master provinces: {len(provs)}")

t11 = load(Path("data/PSA/2025_T1_1.csv"))

master_keys: dict[str, str] = {}
for _, r in provs.iterrows():
    k_full = key(r["name"])
    k_nostrip = key(re.sub(r"\s+Province\s*$", "", r["name"], flags=re.IGNORECASE))
    master_keys.setdefault(k_full, r["psgc10"])
    master_keys.setdefault(k_nostrip, r["psgc10"])

# T1_1 province-like rows
t11_rows = []
for i in range(2, len(t11)):
    nm = clean(t11.iloc[i, 0])
    if not nm:
        continue
    nl = nm.lower()
    if "region" in nl or nl.startswith("national "):
        continue
    if nm.lower() in ("philippines", "total", "grand total"):
        continue
    t11_rows.append(nm)

matched, unmatched = [], []
for nm in t11_rows:
    k1 = key(nm)
    k2 = key(re.sub(r"\s+Province\s*$", "", nm, flags=re.IGNORECASE))
    if k1 in master_keys or k2 in master_keys:
        matched.append(nm)
    else:
        unmatched.append(nm)

print(f"\nT1_1 province-like names that MATCH:  {len(matched)}")
print(f"T1_1 province-like names that DON'T match ({len(unmatched)}):")
for nm in unmatched:
    print(f"  {nm!r}   key={key(nm)!r}")

print("\n-- sample of 10 matched names --")
for nm in matched[:10]:
    print(f"  {nm!r}")