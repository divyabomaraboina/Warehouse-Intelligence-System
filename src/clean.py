from datetime import datetime
from duckdb import df
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
GOLD_DIR = Path("data/processed")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# ── Clean data ───────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix 1 → Fix data types first
    df['YEAR'] = df['YEAR'].astype(int)
    df['MONTH'] = df['MONTH'].astype(int)
    # combine YEAR + MONTH into DATE
    df['DATE'] = pd.to_datetime({'year': df['YEAR'],
                                 'month': df['MONTH'],
                                 'day': 1})
    # Fix 2 → Fill missing SUPPLIER with "UNKNOWN"
    df['SUPPLIER'] = df['SUPPLIER'].fillna("UNKNOWN")

    # Fix 3 → Standardise SUPPLIER
    df['SUPPLIER'] = df['SUPPLIER'].str.strip().str.title()
    # Fix 4 → Standardise ITEM DESCRIPTION
    df['ITEM DESCRIPTION'] = df['ITEM DESCRIPTION'].str.strip().str.title()
    # Fix 5 → Standardise ITEM TYPE
    df['ITEM TYPE'] = df['ITEM TYPE'].str.strip().str.title()

    # Fix 6 → Validate
    print(df["DATE"].dtype)
    print(df["SUPPLIER"].isnull().sum())
    print(df["SUPPLIER"].str.strip())
    print(df['ITEM DESCRIPTION'].str.strip())
    print(df['ITEM TYPE'].str.strip())
    print(df.head())

    # Fix 7 → return cleaned dataframe
    return df

# ── Load bronze ──────────────────────────────────────────────────
def load_bronze() -> pd.DataFrame:
    # find all bronze files in data/raw/
    files = list(RAW_DIR.glob("bronze_*.parquet"))
    # pick the most recent file
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    # print which file we are loading
    print(f"Loading: {latest_file}")
    # load the parquet file
    df = pd.read_parquet(latest_file)
    # return the dataframe
    return df

# ── Save to silver ───────────────────────────────────────────────
def save_to_silver(df: pd.DataFrame):
    # create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # build output path in silver layer
    out_path = CLEANED_DIR / f"silver_{timestamp}.parquet"
    # save as parquet
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to silver: {out_path}")

# ── Clean for modelling ──────────────────────────────────────────
def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1 → remove suspicious row 254069
    df = df.drop(index=254069)

    # Step 2 → drop YEAR and MONTH — DATE column replaces them
    df = df.drop(columns=['YEAR', 'MONTH'])

    # Step 3 → exclude non-product item types
    df = df[~df['ITEM TYPE'].isin(['Ref', 'Dunnage', 'Str_Supplies'])]
    # Fill missing ITEM TYPE based on item description
    df.loc[df['ITEM TYPE'].isna(), 'ITEM TYPE'] = 'Wine'

    # Step 4 → save to gold layer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = GOLD_DIR / f"gold_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to gold: {out_path}")
    return df

# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_bronze()
    cleaned_df = clean_data(df)
    save_to_silver(cleaned_df)
    gold_df = clean_for_modeling(cleaned_df)