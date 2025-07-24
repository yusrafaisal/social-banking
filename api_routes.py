from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mongo import transactions
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId
import json
import re
import logging
from ai_agent import BankingAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize AI agent instance
ai_agent = BankingAIAgent()

class CNICVerifyRequest(BaseModel):
    cnic: str

class AccountSelectionRequest(BaseModel):
    cnic: str
    account_number: str

class UserBalanceQuery(BaseModel):
    account_number: str

class MoneyTransferRequest(BaseModel):
    from_account: str
    to_recipient: str
    amount: float
    currency: str = "PKR"  # Default to PKR since most transactions are in PKR

class PipelineQuery(BaseModel):
    account_number: str
    pipeline: List[Dict[str, Any]]

class ProcessQueryRequest(BaseModel):
    user_message: str
    account_number: str
    first_name: str

class ProcessQueryResponse(BaseModel):
    status: str
    response: str
    error: Optional[str] = None

def convert_objectid_to_string(doc):
    """Recursively convert ObjectId to string in documents."""
    if isinstance(doc, dict):
        return {k: convert_objectid_to_string(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [convert_objectid_to_string(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

def process_pipeline_dates(pipeline):
    """Process pipeline to handle date objects properly."""
    processed_pipeline = []
    for stage in pipeline:
        processed_stage = {}
        for key, value in stage.items():
            if isinstance(value, dict):
                processed_stage[key] = process_dict_dates(value)
            else:
                processed_stage[key] = value
        processed_pipeline.append(processed_stage)
    return processed_pipeline

def process_dict_dates(obj):
    """Recursively process dictionary to handle date objects."""
    if isinstance(obj, dict):
        processed = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                if "$date" in v:
                    try:
                        processed[k] = datetime.fromisoformat(v["$date"].replace("Z", "+00:00"))
                    except ValueError as e:
                        logger.error({
                            "action": "process_dict_dates",
                            "error": f"Invalid date format: {v['$date']}"
                        })
                        processed[k] = v  # Keep original value to avoid breaking pipeline
                elif "$gte" in v or "$lte" in v or "$lt" in v or "$gt" in v:
                    processed[k] = {}
                    for op, date_val in v.items():
                        if isinstance(date_val, dict) and "$date" in date_val:
                            try:
                                processed[k][op] = datetime.fromisoformat(date_val["$date"].replace("Z", "+00:00"))
                            except ValueError as e:
                                logger.error({
                                    "action": "process_dict_dates",
                                    "error": f"Invalid date format: {date_val['$date']}"
                                })
                                processed[k][op] = date_val
                        else:
                            processed[k][op] = date_val
                else:
                    processed[k] = process_dict_dates(v)
            elif isinstance(v, list):
                processed[k] = [process_dict_dates(item) for item in v]
            else:
                processed[k] = v
        return processed
    else:
        return obj

@router.post("/verify_cnic")
def verify_cnic(data: CNICVerifyRequest):
    """Verify user by CNIC and return available accounts."""
    try:
        # Clean and format CNIC
        cnic = data.cnic.strip()
        
        logger.info(f"🔍 Verifying CNIC: {cnic}")
        
        # Find user by CNIC in transactions collection
        user_record = transactions.find_one({"cnic": cnic})
        
        if user_record:
            # Get all unique accounts for this CNIC
            accounts = list(transactions.distinct("account_number", {"cnic": cnic}))
            
            logger.info(f"✅ CNIC verified. Found {len(accounts)} accounts for {user_record['name']}")
            
            return {
                "status": "success",
                "user": {
                    "name": user_record["name"],
                    "cnic": cnic,
                    "accounts": accounts
                }
            }
        else:
            logger.warning(f"❌ CNIC not found: {cnic}")
            return {"status": "fail", "reason": "CNIC not found"}
            
    except Exception as e:
        logger.error(f"CNIC verification error: {e}")
        return {"status": "fail", "reason": "Verification failed"}

@router.post("/select_account")
def select_account(data: AccountSelectionRequest):
    """Confirm account selection for verified CNIC."""
    try:
        cnic = data.cnic.strip()
        account_number = data.account_number.strip()
        
        logger.info(f"🔍 Verifying account selection: CNIC {cnic}, Account {account_number}")
        
        # Verify that this account belongs to this CNIC
        account_record = transactions.find_one({
            "cnic": cnic,
            "account_number": account_number
        })
        
        if account_record:
            logger.info(f"✅ Account selection verified for {account_record['name']}")
            
            return {
                "status": "success",
                "user": {
                    "name": account_record["name"],
                    "cnic": cnic,
                    "selected_account": account_number
                }
            }
        else:
            logger.warning(f"❌ Account {account_number} not found for CNIC {cnic}")
            return {"status": "fail", "reason": "Account not found for this CNIC"}
            
    except Exception as e:
        logger.error(f"Account selection error: {e}")
        return {"status": "fail", "reason": "Account selection failed"}

@router.post("/user_balance")
async def get_user_balance(data: UserBalanceQuery):
    """Get user's current balance for selected account."""
    try:
        account_number = data.account_number.strip()
        
        # Get user info from latest transaction
        user_record = transactions.find_one({"account_number": account_number})
        if not user_record:
            return {"status": "fail", "reason": "Account not found"}
        
        # Get latest transaction for most current balance
        latest_txn = transactions.find_one(
            {"account_number": account_number},
            sort=[("date", -1), ("_id", -1)]
        )
        
        if latest_txn:
            # Extract balance from account_balance field
            current_balance = latest_txn.get("account_balance", 0)
            account_currency = latest_txn.get("account_currency", "pkr")
        else:
            current_balance = 0
            account_currency = "pkr"
        
        # Format balance by currency
        if account_currency.lower() == "usd":
            balance_usd = current_balance
            balance_pkr = 0
        else:
            balance_usd = 0 
            balance_pkr = current_balance
        
        return {
            "status": "success",
            "user": {
                "first_name": user_record["name"].split()[0],  # Get first name
                "last_name": user_record["name"].split()[-1] if len(user_record["name"].split()) > 1 else "",
                "account_number": account_number,
                "current_balance_usd": balance_usd,
                "current_balance_pkr": balance_pkr,
                "account_currency": account_currency
            }
        }
    except Exception as e:
        logger.error(f"Balance error: {e}")
        return {"status": "fail", "error": str(e)}

@router.post("/execute_pipeline")
async def execute_pipeline(data: PipelineQuery):
    """Execute a dynamic MongoDB aggregation pipeline."""
    try:
        # Validate input
        if not data.pipeline:
            return {"status": "fail", "reason": "Empty pipeline provided"}
        
        if not data.account_number:
            return {"status": "fail", "reason": "Account number is required"}
        
        # Process pipeline to handle date objects
        processed_pipeline = process_pipeline_dates(data.pipeline)
        
        logger.info(f"Executing pipeline for account {data.account_number}: {processed_pipeline}")
        
        # Execute pipeline on transactions collection
        result = list(transactions.aggregate(processed_pipeline))
        
        # Convert ObjectId to string for JSON serialization
        result = convert_objectid_to_string(result)
        
        logger.info(f"Pipeline execution successful. Returned {len(result)} documents")
        
        return {
            "status": "success",
            "data": result,
            "count": len(result)
        }
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        return {"status": "fail", "error": str(e)}

@router.post("/transfer_money")
async def transfer_money(data: MoneyTransferRequest):
    """Handle money transfer with new currency structure."""
    try:
        # Validate input
        if data.amount <= 0:
            return {"status": "fail", "reason": "Transfer amount must be positive"}
        
        if data.currency.upper() not in ["USD", "PKR"]:
            return {"status": "fail", "reason": "Currency must be USD or PKR"}
        
        # Get account info
        account_record = transactions.find_one({"account_number": data.from_account})
        if not account_record:
            return {"status": "fail", "reason": "Sender account not found"}
        
        # Get current balance from latest transaction
        latest_txn = transactions.find_one(
            {"account_number": data.from_account},
            sort=[("date", -1), ("_id", -1)]
        )
        
        if latest_txn:
            current_balance = latest_txn.get("account_balance", 0)
            account_currency = latest_txn.get("account_currency", "pkr")
        else:
            return {"status": "fail", "reason": "No transaction history found"}
        
        # Check if currencies match
        if data.currency.upper() != account_currency.upper():
            return {
                "status": "fail", 
                "reason": f"Cannot transfer {data.currency} from {account_currency} account"
            }
        
        # Check sufficient balance
        if current_balance < data.amount:
            return {
                "status": "fail", 
                "reason": f"Insufficient {data.currency} balance. Available: {current_balance:.2f}, Required: {data.amount:.2f}"
            }
        
        # Calculate new balance
        new_balance = current_balance - data.amount
        
        # Create transfer transaction
        transfer_txn = {
            "name": account_record["name"],
            "cnic": account_record["cnic"],
            "account_number": data.from_account,
            "date": datetime.now(),
            "type": "debit",
            "description": f"Transfer to {data.to_recipient}",
            "category": "Transfer",
            "account_currency": account_currency.lower(),
            "amount_deducted_from_account": data.amount,
            "transaction_amount": data.amount,
            "transaction_currency": data.currency.lower(),
            "account_balance": new_balance
        }
        
        # Insert transaction
        txn_result = transactions.insert_one(transfer_txn)
        
        logger.info(f"Transfer successful: {data.amount} {data.currency} from {data.from_account} to {data.to_recipient}")
        logger.info(f"Updated balance: {new_balance} {account_currency}")
        
        return {
            "status": "success",
            "message": f"Successfully transferred {data.amount} {data.currency} to {data.to_recipient}",
            "transaction_id": str(txn_result.inserted_id),
            "new_balance": new_balance,
            "currency": account_currency,
            "transfer_details": {
                "amount": data.amount,
                "currency": data.currency,
                "recipient": data.to_recipient,
                "timestamp": transfer_txn["date"].isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Transfer error: {e}")
        return {"status": "fail", "error": str(e)}

@router.post("/process_query", response_model=ProcessQueryResponse)
async def process_query(data: ProcessQueryRequest):
    """Process user banking queries using AI agent with contextual awareness."""
    try:
        logger.info({
            "action": "api_process_query_start",
            "user_message": data.user_message,
            "account_number": data.account_number,
            "first_name": data.first_name
        })
        
        # Use the AI agent to process the query
        response = await ai_agent.process_query(
            user_message=data.user_message,
            account_number=data.account_number,
            first_name=data.first_name
        )
        
        logger.info({
            "action": "api_process_query_success",
            "account_number": data.account_number,
            "response_length": len(response)
        })
        
        return ProcessQueryResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        logger.error({
            "action": "api_process_query_error",
            "error": str(e),
            "account_number": data.account_number,
            "user_message": data.user_message
        })
        
        return ProcessQueryResponse(
            status="error",
            response="Sorry, an error occurred while processing your request. Please try again.",
            error=str(e)
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "enhanced_banking_ai_backend",
        "authentication": "cnic_otp_smart_account_selection",
        "features": {
            "greeting_detection": "enabled",
            "varied_responses": "enabled",
            "context_filtering": "enabled",
            "otp_verification": "enabled",
            "smart_account_selection": "enabled", 
            "transfer_confirmation": "enabled"
        }
    }