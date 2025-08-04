import os
from pymongo import MongoClient

# --- 1️⃣ Connect to MongoDB ---
# client = MongoClient("mongodb://localhost:27017/")
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

# --- 2️⃣ Access database and collections ---
db = client["bank_database"]
# Updated to use transactions collection with new structure (name, cnic, account_number)
transactions = db["transactions"]

print(db.list_collection_names())  # Optional: Print collection names to verify connection