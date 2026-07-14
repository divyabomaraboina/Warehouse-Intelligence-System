from datetime import datetime
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

GOLD_DIR = Path("data/processed")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

def load_gold() -> pd.DataFrame:
    files = list(GOLD_DIR.glob("gold_*.parquet"))
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading: {latest}")
    return pd.read_parquet(latest)

def build_features(df: pd.DataFrame):
    # Step 0 → Split
   # ← ADD HERE FIRST — before split
    median = df['WAREHOUSE SALES'].median()
    df['IS_HIGH_SALES'] = (df['WAREHOUSE SALES'] > median).astype(int)

    # ← REPLACE old DATE split with this
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(df, df['IS_HIGH_SALES']):
        train = df.iloc[train_idx].copy()
        test  = df.iloc[test_idx].copy()


    # Step 1 → Extract
    train['MONTH']   = train['DATE'].dt.month
    test['MONTH']    = test['DATE'].dt.month
    train['QUARTER'] = train['DATE'].dt.quarter
    test['QUARTER']  = test['DATE'].dt.quarter

    # Step 2 → Transform
    # encode SUPPLIER
    train['SUPPLIER_ENCODED'], supplier_index = pd.factorize(train['SUPPLIER'])
    supplier_map = {v: i for i, v in enumerate(supplier_index)}
    test['SUPPLIER_ENCODED'] = test['SUPPLIER'].map(supplier_map).fillna(-1).astype(int)

    # encode ITEM TYPE
    train['ITEM_TYPE_ENCODED'], item_index = pd.factorize(train['ITEM TYPE'])
    item_map = {v: i for i, v in enumerate(item_index)}
    test['ITEM_TYPE_ENCODED'] = test['ITEM TYPE'].map(item_map).fillna(-1).astype(int)

    # Step 3 → Construct
    train['IS_SUMMER']  = train['MONTH'].apply(lambda x: 1 if x in [5,6,7,8] else 0)
    test['IS_SUMMER']   = test['MONTH'].apply(lambda x: 1 if x in [5,6,7,8] else 0)
    train['IS_HOLIDAY'] = train['MONTH'].apply(lambda x: 1 if x in [11,12] else 0)
    test['IS_HOLIDAY']  = test['MONTH'].apply(lambda x: 1 if x in [11,12] else 0)

    train['LAST_MONTH_SALES'] = train.groupby(['SUPPLIER','ITEM CODE'])['WAREHOUSE SALES'].shift(1)
    test['LAST_MONTH_SALES']  = test.groupby(['SUPPLIER','ITEM CODE'])['WAREHOUSE SALES'].shift(1)

    train['LAST_3_MONTHS_SALES'] = train.groupby(['SUPPLIER','ITEM CODE'])['WAREHOUSE SALES'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    test['LAST_3_MONTHS_SALES'] = test.groupby(['SUPPLIER','ITEM CODE'])['WAREHOUSE SALES'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # Fix NaN in lag features — fill with train median
    median_lag = train['LAST_MONTH_SALES'].median()
    train['LAST_MONTH_SALES'] = train['LAST_MONTH_SALES'].fillna(median_lag)
    test['LAST_MONTH_SALES']  = test['LAST_MONTH_SALES'].fillna(median_lag)

    median_rolling = train['LAST_3_MONTHS_SALES'].median()
    train['LAST_3_MONTHS_SALES'] = train['LAST_3_MONTHS_SALES'].fillna(median_rolling)
    test['LAST_3_MONTHS_SALES']  = test['LAST_3_MONTHS_SALES'].fillna(median_rolling)

    # Sanity checks
    print("\n=== SANITY CHECKS ===")
    print(f"Train size: {len(train):,}")
    print(f"Test size:  {len(test):,}")
    print(f"\nIS_HIGH_SALES balance — Train:")
    print(train['IS_HIGH_SALES'].value_counts(normalize=True).round(2))
    print(f"\nIS_HIGH_SALES balance — Test:")
    print(test['IS_HIGH_SALES'].value_counts(normalize=True).round(2))
    print(f"\nItem types in Train: {train['ITEM TYPE'].unique()}")
    print(f"Item types in Test:  {test['ITEM TYPE'].unique()}")
    print("=== END CHECKS ===\n")

    return train, test

def save_featured_data(train: pd.DataFrame, test: pd.DataFrame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = GOLD_DIR / f"featured_train_{timestamp}.parquet"
    test_path  = GOLD_DIR / f"featured_test_{timestamp}.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"Saved train: {train_path}")
    print(f"Saved test:  {test_path}")
def load_featured_data():
    # find latest featured_train file
    train_files = list(GOLD_DIR.glob("featured_train_*.parquet"))
    latest_train = max(train_files, key=lambda f: f.stat().st_mtime)
    
    # find latest featured_test file
    test_files = list(GOLD_DIR.glob("featured_test_*.parquet"))
    latest_test = max(test_files, key=lambda f: f.stat().st_mtime)
    
    print(f"Loading train: {latest_train}")
    print(f"Loading test:  {latest_test}")
    
    train = pd.read_parquet(latest_train)
    test  = pd.read_parquet(latest_test)
    
    return train, test

if __name__ == "__main__":
    df = load_gold()
    train, test = build_features(df)
    save_featured_data(train, test)
