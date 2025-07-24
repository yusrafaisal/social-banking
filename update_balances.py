#!/usr/bin/env python3
"""
update_balances.py

Replaces the (possibly stale) current balances in users.xlsx
with the balances from the most‑recent transaction per account
found in transactions.xlsx.
"""

import pandas as pd
from pathlib import Path

# ──────────────────────────────
# 1)  File locations – edit me!
# ──────────────────────────────
TRANSACTIONS_FILE = Path("social_banking/final_transactions_database.xlsx")   # all_transactions.xlsx, etc.
USERS_FILE        = Path("social_banking/users_data_updated.xlsx")          # user_master.xlsx, etc.
OUTPUT_EXCEL      = Path("users_updated.xlsx")
OUTPUT_CSV        = Path("users_updated.csv")

# ──────────────────────────────
# 2)  Load data
# ──────────────────────────────
# Read as strings so account numbers with leading zeros stay intact
tx = pd.read_excel(TRANSACTIONS_FILE, dtype={"account_number": str})
users = pd.read_excel(USERS_FILE, dtype={"account_number": str})

# Make sure the date column is datetime
tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

# ──────────────────────────────
# 3)  Pick the latest transaction per account
# ──────────────────────────────
# idxmax() returns the index of the row with the latest date for each account
latest_idx = tx.groupby("account_number")["date"].idxmax()
latest_tx  = tx.loc[latest_idx, ["account_number",
                                 "balance_usd",
                                 "balance_pkr"]].copy()

# Rename columns to match the users sheet
latest_tx.rename(columns={"balance_usd": "current_balance_usd",
                          "balance_pkr": "current_balance_pkr"},
                 inplace=True)

# ──────────────────────────────
# 4)  Merge & update
# ──────────────────────────────
# We keep any user rows that have no matching transaction (left join)
merged = users.merge(latest_tx,
                     on="account_number",
                     how="left",
                     suffixes=("", "_new"))

# For accounts that *did* match, copy the new balances over;
# otherwise leave the old value alone
for col in ("current_balance_usd", "current_balance_pkr"):
    merged[col] = merged[f"{col}_new"].combine_first(merged[col])
    merged.drop(columns=f"{col}_new", inplace=True)

# ──────────────────────────────
# 5)  Save results
# ──────────────────────────────
merged.to_excel(OUTPUT_EXCEL, index=False)  # Excel copy
merged.to_csv(OUTPUT_CSV,   index=False)    # CSV  copy
print(f"✔ Updated balances written to:\n   {OUTPUT_EXCEL}\n   {OUTPUT_CSV}")

# ──────────────────────────────
# 6)  One‑liner function (optional importable API)
# ──────────────────────────────
def update_balances(tx_path, users_path, out_xlsx="users_updated.xlsx", out_csv="users_updated.csv"):
    """Convenience wrapper around the logic above."""
    globals().update(
        TRANSACTIONS_FILE=Path(tx_path),
        USERS_FILE=Path(users_path),
        OUTPUT_EXCEL=Path(out_xlsx),
        OUTPUT_CSV=Path(out_csv),
    )
    # rerun the script’s main body
    import importlib, sys, pathlib
    module = sys.modules[__name__]
    importlib.reload(module)

