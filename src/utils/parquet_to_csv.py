"""Convert every .parquet in a folder to .csv. Usage: python src/utils/parquet_to_csv.py data/processed"""
import sys
from pathlib import Path
import pandas as pd

def convert(folder: Path) -> None:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        print(f"No .parquet files in {folder}")
        return
    for pq in files:
        csv = pq.with_suffix(".csv")
        df = pd.read_parquet(pq)
        df.to_csv(csv, index=False)
        print(f"{pq.name:45s} -> {csv.name}  ({len(df):,} rows)")

if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
    convert(folder)