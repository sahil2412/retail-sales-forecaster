"""
scripts/build_sample.py
-----------------------
Run once after downloading M5 raw CSVs from Kaggle:
  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Expected files in data/:
  - sales_train_evaluation.csv
  - calendar.csv
  - sell_prices.csv

Usage:
  python scripts/build_sample.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forecasting.preprocessing import build_m5_sample

if __name__ == "__main__":
    print("Building M5 sample...")
    df = build_m5_sample(
        sales_path="data/sales_train_evaluation.csv",
        calendar_path="data/calendar.csv",
        prices_path="data/sell_prices.csv",
        n_items=10,
        store_id="CA_1",
    )
    print(f"Done. {len(df):,} rows, {df['item_id'].nunique()} items saved to data/m5_sample.parquet")
    print(df.head())
