import pandas as pd
from pymongo import MongoClient

# --- 1️⃣ Load Excel file with names and clean dates ---
transactions_df = pd.read_excel("final_dataset.xlsx")

# Convert key fields to string to ensure consistency
transactions_df["name"] = transactions_df["name"].astype(str)
transactions_df["cnic"] = transactions_df["cnic"].astype(str)
transactions_df["account_number"] = transactions_df["account_number"].astype(str)

print("Dataset shape:", transactions_df.shape)
print("Columns:", transactions_df.columns.tolist())

# Verify dataset structure
print("\nDataset verification:")
print(f"Unique names: {transactions_df['name'].nunique()}")
print(f"Unique CNICs: {transactions_df['cnic'].nunique()}")
print(f"Unique account numbers: {transactions_df['account_number'].nunique()}")

# Check CNIC to account mapping
print("\nCNIC to Account mapping:")
cnic_accounts = transactions_df.groupby('cnic')['account_number'].nunique()
print("Accounts per CNIC:")
print(cnic_accounts)

# Show sample data
print("\nSample data:")
print(transactions_df[['name', 'cnic', 'account_number', 'date', 'type', 'description']].head())

# --- 2️⃣ Connect to MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["bank_database"]

# --- 3️⃣ Define collection ---
transactions_col = db["transactions"]

# --- 4️⃣ Clean insert (remove existing data) ---
transactions_col.delete_many({})

# --- 5️⃣ Insert data into MongoDB ---
transactions_col.insert_many(transactions_df.to_dict(orient="records"))

print(f"\n✅ {len(transactions_df)} transaction records successfully loaded to MongoDB.")

# Verify data in MongoDB
print("\nMongoDB verification:")
total_docs = transactions_col.count_documents({})
print(f"Total documents in MongoDB: {total_docs}")

# Show unique CNICs in MongoDB
cnics_in_db = transactions_col.distinct("cnic")
print(f"Unique CNICs in MongoDB: {cnics_in_db}")

# Show accounts per CNIC
for cnic in cnics_in_db:
    accounts = transactions_col.distinct("account_number", {"cnic": cnic})
    name = transactions_col.find_one({"cnic": cnic})["name"]
    print(f"CNIC {cnic} ({name}): {len(accounts)} accounts - {accounts}")

print("\n✅ Database setup complete!")