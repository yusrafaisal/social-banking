# import os
# from pymongo import MongoClient

# # --- 1️⃣ Connect to MongoDB ---
# # client = MongoClient("mongodb://localhost:27017/")
# MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
# client = MongoClient(MONGO_URI)

# # --- 2️⃣ Access database and collections ---
# db = client["bank_database"]
# # Updated to use transactions collection with new structure (name, cnic, account_number)
# transactions = db["transactions"]

# print(db.list_collection_names())  # Optional: Print collection names to verify connection

import os
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)

# Use environment variable for MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

try:
    # Create MongoDB client with the Atlas connection string
    client = MongoClient(MONGODB_URI)
    
    # Get database name from URI or use default
    if "mongodb+srv://" in MONGODB_URI or "mongodb://" in MONGODB_URI:
        # Extract database name from URI or use default
        db_name = MONGODB_URI.split('/')[-1].split('?')[0] if '/' in MONGODB_URI else 'bank_database'
        if not db_name or db_name == '':
            db_name = 'bank_database'
        db = client[db_name]
    else:
        db = client.bank_database
    
    # Test connection but don't fail startup if it doesn't work
    try:
        client.admin.command('ping')
        logger.info("✅ MongoDB connected successfully")
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
    except Exception as ping_error:
        logger.warning(f"⚠️ MongoDB connection test failed: {ping_error}")
        # Don't print the full error in production to avoid exposing credentials
    
    # Initialize collections
    transactions = db.transactions
    users = db.users
    
except Exception as e:
    logger.error(f"❌ MongoDB initialization error: {e}")
    # Create None objects to prevent import errors
    client = None
    db = None
    transactions = None
    users = None