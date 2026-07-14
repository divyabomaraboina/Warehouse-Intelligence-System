import pandas as pd
from pathlib import Path
from datetime import datetime

#--------------------Paths
RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
#Creates parent directories if they don't exist
#exist_ok=True means if the folder already exists — don't throw an error,
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

#BLOCK 3:open the CSV, add timestamp, add source name
def load_raw_data(filename: str) -> pd.DataFrame:
    """Load raw data from a CSV file."""
    filepath = RAW_DIR / filename
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    df["_ingested_at"] = datetime.now().isoformat()
    df["_source_file"] = filename
    return df
def profile_dataset(df: pd.DataFrame):
    """Profile the dataset."""
    print("\n" + "="*50)
    print("  DATA PROFILE REPORT")
    print("="*50)
    print(f"  Rows:        {len(df):,}")
    print(f"  Columns:     {len(df.columns)}")
    print(f"  Duplicates:  {df.duplicated().sum():,}")
    print(f"  Memory:      {round(df.memory_usage(deep=True).sum() / 1e6, 2)} MB")
    print(f"\n  {'Column':<35} {'Type':<15} {'Nulls %'}")
    print(f"  {'-'*55}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = round(df[col].isnull().sum() / len(df) * 100, 1)
        if null_pct > 5:
            flag = " ⚠ HIGH"
        else:
            flag = ""

        print(f"  {col:<35} {dtype:<15} {null_pct}%{flag}")
    print("="*50)
def save_to_bronze(df: pd.DataFrame, filename: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"bronze_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to bronze: {out_path}")

if __name__ == "__main__":
    filename = "Warehouse_and_Retail_Sales.csv"
    df = load_raw_data(filename)
    profile_dataset(df)
    save_to_bronze(df, filename)