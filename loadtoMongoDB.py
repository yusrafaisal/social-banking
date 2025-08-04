import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1️⃣ Load Excel file with names and clean dates ---
transactions_df = pd.read_excel("social-banking/final_dataset.xlsx")

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

# --- 2️⃣ Connect to MongoDB Atlas (CHANGED THIS PART) ---
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("❌ Error: MONGODB_URI not found in environment variables")
    print("Make sure your .env file contains:")
    print("MONGODB_URI=mongodb+srv://yusrafaisal68:GMbeD6kRXdC2sDT2@cluster0.z0o6f47.mongodb.net/bank_database?retryWrites=true&w=majority&appName=Cluster0")
    exit(1)

print(f"Connecting to MongoDB Atlas...")
client = MongoClient(MONGODB_URI)

# Test connection
try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB Atlas: {e}")
    exit(1)

db = client["bank_database"]

# --- 3️⃣ Define collection ---
transactions_col = db["transactions"]

# --- 4️⃣ Clean insert (remove existing data) ---
print("🗑️ Removing existing data...")
delete_result = transactions_col.delete_many({})
print(f"Deleted {delete_result.deleted_count} existing documents")

# --- 5️⃣ Insert data into MongoDB ---
print("📤 Uploading data to MongoDB Atlas...")
insert_result = transactions_col.insert_many(transactions_df.to_dict(orient="records"))
print(f"✅ {len(insert_result.inserted_ids)} transaction records successfully loaded to MongoDB Atlas.")

# Verify data in MongoDB
print("\nMongoDB Atlas verification:")
total_docs = transactions_col.count_documents({})
print(f"Total documents in MongoDB Atlas: {total_docs}")

# Show unique CNICs in MongoDB
cnics_in_db = transactions_col.distinct("cnic")
print(f"Unique CNICs in MongoDB Atlas: {cnics_in_db}")

# Show accounts per CNIC
for cnic in cnics_in_db:
    accounts = transactions_col.distinct("account_number", {"cnic": cnic})
    name = transactions_col.find_one({"cnic": cnic})["name"]
    print(f"CNIC {cnic} ({name}): {len(accounts)} accounts - {accounts}")

# Check if test CNIC exists
test_cnic = "42522-5110826-6"
test_docs = transactions_col.count_documents({"cnic": test_cnic})
print(f"\n🔍 Test CNIC {test_cnic} found: {test_docs} documents")

print("\n✅ Database upload to MongoDB Atlas complete!")