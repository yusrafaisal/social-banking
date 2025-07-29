# Updated webhook.py - LLM-First approach with OTP, flexible CNIC, smart account selection, and transfer confirmation
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
from prompts import account_selection_prompt
 
# Import the prompt (add this import at the top of webhook.py)
from prompts import account_selection_prompt
from ai_agent import llm


import os
import requests
import re
from typing import Dict, Any, List
from translation_service import translation_service
from state import (
    authenticated_users, processed_messages, periodic_cleanup,
    VERIFICATION_STAGES, get_user_verification_stage, set_user_verification_stage,
    is_fully_authenticated, get_user_account_info, clear_user_state,
    is_otp_pending, is_transfer_otp_pending, is_valid_otp, extract_cnic_from_text,
    get_pending_transfer_info, set_pending_transfer_info, clear_pending_transfer_info,
    is_transfer_confirmation_pending, get_user_accounts_with_details, set_user_accounts_with_details,
    set_user_language, get_user_language, get_user_last_language, clear_user_language
)
import time
import logging
from datetime import datetime
from ai_agent import BankingAIAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPGiZCdRaVk6KrTAiQYclW4ZCZC9e8FiC4EqdOU0zN2gLDaVC1UtXeDXYT7VtnKPyr5NV3TZAgChtsMiDhzgZBsqk6eHZA8IKUQjqlORPXIatiTbs9OekNOeFxL16xOpEM2gJKMgJLR7yo70dPCHWBTyILXZAiBLEzQt9KfZBdOYCIEGyOVDdzMDM9aey"

BACKEND_URL = "http://localhost:8000"

# Initialize AI Agent for natural responses
ai_agent = BankingAIAgent()

@app.get("/webhook")
async def webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Invalid verification token.")

@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
    
    if "entry" not in data:
        return JSONResponse(content={"status": "ok"})

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            message_id = messaging_event.get("message", {}).get("mid")
            sender_id = messaging_event["sender"]["id"]

            if message_id and message_id in processed_messages:
                continue

            if "message" in messaging_event:
                if message_id:
                    processed_messages.add(message_id)
                
                user_message = messaging_event["message"].get("text", "")
                
                if user_message.strip():
                    # Handle translation and process message
                    response_text = await process_multilingual_message(sender_id, user_message)
                    send_message(sender_id, response_text)

    if len(processed_messages) % 100 == 0:
        periodic_cleanup()

    return JSONResponse(content={"status": "ok"})

async def process_multilingual_message(sender_id: str, user_message: str) -> str:
    """Process message with language detection and translation support."""
    
    try:
        # Detect language of incoming message
        detected_language = translation_service.detect_language_smart(
            user_message, 
            sender_id, 
            get_user_last_language
        )
        
        # Store the detected language for this user
        set_user_language(sender_id, detected_language)
        
        logger.info({
            "action": "language_detected",
            "sender_id": sender_id,
            "detected_language": detected_language,
            "original_message": user_message
        })
        
        # Translate to English for processing if needed
        if detected_language != 'en':
            english_message = translation_service.translate_to_english(user_message, detected_language)
            logger.info({
                "action": "message_translated_to_english",
                "sender_id": sender_id,
                "original": user_message,
                "translated": english_message,
                "source_language": detected_language
            })
        else:
            english_message = user_message
        
        # Process the English message through existing flow
        english_response = await process_user_message(sender_id, english_message)
        
        # Translate response back to user's language if needed
        if detected_language != 'en':
            final_response = translation_service.translate_from_english(english_response, detected_language)
            logger.info({
                "action": "response_translated_to_user_language",
                "sender_id": sender_id,
                "english_response": english_response[:100] + "...",
                "translated_response": final_response[:100] + "...",
                "target_language": detected_language
            })
        else:
            final_response = english_response
        
        return final_response
        
    except Exception as e:
        logger.error({
            "action": "multilingual_processing_error",
            "sender_id": sender_id,
            "error": str(e),
            "user_message": user_message
        })
        
        # Fallback to English processing
        return await process_user_message(sender_id, user_message)
    
user_last_message_time = {}

def is_greeting_message(message: str) -> bool:
    """Check if the message is a greeting."""
    greeting_words = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "good day", "howdy", "what's up", "whats up", "sup",
        "hola", "bonjour", "namaste", "salaam", "salam", "assalam", "start"
    ]
    
    message_lower = message.lower().strip()
    
    # Check if message is exactly a greeting or starts with greeting
    for greeting in greeting_words:
        if message_lower == greeting or message_lower.startswith(greeting + " "):
            return True
    
    # Check for common greeting patterns
    greeting_patterns = [
        r'^hi+$',  # hi, hii, hiii
        r'^hey+$',  # hey, heyy
        r'^hello+$',  # hello, helloo
        r'^good (morning|afternoon|evening|day)',
        r'^how are you',
        r'^what\'?s up'
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, message_lower):
            return True
    
    return False

async def get_account_details_from_backend(accounts: List[str]) -> List[Dict]:
    """Get detailed account information including currency from backend."""
    account_details = []
    
    for account in accounts:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/user_balance",
                    json={"account_number": account}
                )
                if response.status_code == 200:
                    result = response.json()
                    if result["status"] == "success":
                        user_info = result["user"]
                        account_details.append({
                            "account_number": account,
                            "currency": user_info.get("account_currency", "pkr").upper(),
                            "balance_usd": user_info.get("current_balance_usd", 0),
                            "balance_pkr": user_info.get("current_balance_pkr", 0)
                        })
        except Exception as e:
            logger.error(f"Error getting account details for {account}: {e}")
            # Add account with default info if API fails
            account_details.append({
                "account_number": account,
                "currency": "UNKNOWN",
                "balance_usd": 0,
                "balance_pkr": 0
            })
    
    return account_details

async def smart_account_selection(user_input: str, account_details: List[Dict]) -> str:
    """LLM-based smart account selection for natural language understanding."""
    try:
        # Format account details for the prompt
        account_info = []
        for i, account in enumerate(account_details, 1):
            account_info.append(
                f"{i}. Account: {account['account_number']}, Currency: {account['currency']}, "
                f"Balance: {account.get('balance_pkr', 0)} PKR / {account.get('balance_usd', 0)} USD"
            )
        
        account_details_str = "\n".join(account_info)
       

        # Use LLM to understand the selection
        response = await ai_agent.llm.ainvoke([
            {
                "role": "system", 
                "content": account_selection_prompt.format(
                    user_input=user_input,
                    account_details=account_details_str
                )
            }
        ])
        
        selected_account = response.content.strip()
        
        # Validate the response
        if selected_account == "NO_MATCH":
            logger.info(f"LLM could not match account selection: {user_input}")
            return None
            
        # Verify the account exists in our list
        for account in account_details:
            if account["account_number"] == selected_account:
                logger.info(f"LLM selected account: {selected_account} for input: {user_input}")
                return selected_account
        
        # Fallback: try partial matching if LLM returned partial number
        if selected_account.isdigit():
            for account in account_details:
                if account["account_number"].endswith(selected_account):
                    logger.info(f"LLM partial match: {account['account_number']} for input: {user_input}")
                    return account["account_number"]
        
        logger.warning(f"LLM returned invalid account: {selected_account} for input: {user_input}")
        return None
        
    except Exception as e:
        logger.error(f"Error in LLM account selection: {e}")
        # Fallback to original logic for critical errors
        return smart_account_selection_fallback(user_input, account_details)

def smart_account_selection_fallback(user_input: str, account_details: List[Dict]) -> str:
    """Fallback account selection logic."""
    user_input_clean = user_input.strip()
    
    # Check if it's a full account number match first
    for account in account_details:
        if account["account_number"] == user_input_clean:
            logger.info(f"Fallback: Exact account number match: {user_input_clean}")
            return account["account_number"]
    
    # Check for partial number matches (last 4, 5, 6+ digits)
    if user_input_clean.isdigit() and len(user_input_clean) >= 4:
        for account in account_details:
            if account["account_number"].endswith(user_input_clean):
                logger.info(f"Fallback: Partial match {user_input_clean} -> {account['account_number']}")
                return account["account_number"]
    
    user_input_lower = user_input_clean.lower()
    
    # Currency-based selection
    if "usd" in user_input_lower or "dollar" in user_input_lower:
        usd_accounts = [acc for acc in account_details if acc["currency"] == "USD"]
        if usd_accounts:
            logger.info(f"Fallback: USD currency match -> {usd_accounts[0]['account_number']}")
            return usd_accounts[0]["account_number"]
    
    if "pkr" in user_input_lower or "rupee" in user_input_lower:
        pkr_accounts = [acc for acc in account_details if acc["currency"] == "PKR"]
        if pkr_accounts:
            logger.info(f"Fallback: PKR currency match -> {pkr_accounts[0]['account_number']}")
            return pkr_accounts[0]["account_number"]
    
    # Position-based selection
    if any(word in user_input_lower for word in ["first", "1st", "one"]):
        if account_details:
            logger.info(f"Fallback: First account -> {account_details[0]['account_number']}")
            return account_details[0]["account_number"]
    
    if any(word in user_input_lower for word in ["second", "2nd", "two"]):
        if len(account_details) > 1:
            logger.info(f"Fallback: Second account -> {account_details[1]['account_number']}")
            return account_details[1]["account_number"]
    
    logger.warning(f"Fallback: No match found for input: {user_input_clean}")
    return None




def is_confirmation_positive(message: str) -> bool:
    """Check if user message is a positive confirmation."""
    message_lower = message.lower().strip()
    
    positive_words = [
        "yes", "y", "yeah", "yep", "yup", "ok", "okay", "confirm", "proceed", 
        "go ahead", "continue", "sure", "definitely", "absolutely", "correct",
        "right", "true", "confirm it", "do it", "send it", "transfer it"
    ]
    
    return any(word in message_lower for word in positive_words)

def is_confirmation_negative(message: str) -> bool:
    """Check if user message is a negative confirmation."""
    message_lower = message.lower().strip()
    
    negative_words = [
        "no", "n", "nope", "cancel", "stop", "abort", "don't", "dont", 
        "not", "wrong", "incorrect", "false", "refuse", "decline", "back"
    ]
    
    return any(word in message_lower for word in negative_words)



async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message with enhanced features."""
    
    current_time = time.time()
    
    # Rate limiting
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 2:
            return "I appreciate your enthusiasm! Please give me just a moment to process your previous message before sending another. 😊"
    
    user_last_message_time[sender_id] = current_time

        # Check for exit command first# Check for exit command first
    if user_message.strip().lower() == "exit":
        logger.info({
            "action": "exit_command_detected",
            "sender_id": sender_id
        })
        
        user_info = get_user_account_info(sender_id)
        first_name = user_info.get("name", "").split()[0] if user_info.get("name") else ""
        account_number = user_info.get("account_number", "")
        
        # FORCE CLEAR ALL MEMORY FOR THIS USER
        logger.info(f"🧠 DEBUG: account_number = '{account_number}'")
        
        if account_number:
            ai_agent.clear_user_memory(account_number)
            logger.info(f"✅ Force cleared memory for account: {account_number}")
        else:
            # Clear memory for all possible accounts if account_number is empty
            logger.info("⚠️ No account_number found, clearing all user memories")
            ai_agent.user_memories.clear()  # Nuclear option - clear everything
            logger.info("✅ Cleared all AI agent memories")
        
        clear_user_state(sender_id)
        
        logger.info({
            "action": "session_terminated",
            "sender_id": sender_id
        })
        
        return await ai_agent.handle_session_end(account_number, first_name)
    





    # Get current verification stage
    verification_stage = get_user_verification_stage(sender_id)
    
    logger.info({
        "action": "processing_user_message",
        "sender_id": sender_id,
        "verification_stage": verification_stage,
        "user_message": user_message,
        "enhanced_features": "flexible_cnic_smart_account_transfer_confirmation"
    })

    # Handle different verification stages
    if verification_stage == VERIFICATION_STAGES["NOT_VERIFIED"]:
        return await handle_cnic_verification(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["CNIC_VERIFIED"]:
        return await handle_otp_verification(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["OTP_VERIFIED"]:
        return await handle_account_selection(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["ACCOUNT_SELECTED"]:
        return await handle_banking_queries(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["TRANSFER_OTP_PENDING"]:
        return await handle_transfer_otp_verification(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["TRANSFER_CONFIRMATION_PENDING"]:
        return await handle_transfer_confirmation(sender_id, user_message)
    
    else:
        return await ai_agent.handle_session_start()

async def handle_cnic_verification(sender_id: str, user_message: str) -> str:
    """Handle CNIC verification with flexible input format."""
    
    user_message_clean = user_message.strip()
    
    # Check if this is a greeting first
    if is_greeting_message(user_message_clean):
        logger.info({
            "action": "initial_greeting_detected",
            "sender_id": sender_id,
            "message": user_message_clean
        })
        
        return await ai_agent.handle_initial_greeting()
    
    # Try to extract CNIC from natural language
    extracted_cnic = extract_cnic_from_text(user_message_clean)
    
    if not extracted_cnic:
        # Use AI agent for natural invalid format response
        return await ai_agent.handle_invalid_cnic_format(user_message_clean)
    
    try:
        # Verify CNIC with backend
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/verify_cnic",
                json={"cnic": extracted_cnic}
            )
            result = response.json()
        
        if result["status"] == "success":
            user_data = result["user"]
            
            # Store CNIC verification data
            set_user_verification_stage(
                sender_id, 
                VERIFICATION_STAGES["CNIC_VERIFIED"],
                cnic=user_data["cnic"],
                name=user_data["name"],
                accounts=user_data["accounts"]
            )
            
            logger.info({
                "action": "cnic_verified_successfully_flexible",
                "sender_id": sender_id,
                "extracted_cnic": extracted_cnic,
                "original_message": user_message_clean,
                "name": user_data["name"],
                "accounts_count": len(user_data["accounts"]),
                "next_step": "otp_verification"
            })
            
            return await ai_agent.handle_otp_request(user_data["name"].split()[0])
        
        else:
            logger.warning({
                "action": "cnic_verification_failed_flexible",
                "sender_id": sender_id,
                "extracted_cnic": extracted_cnic,
                "original_message": user_message_clean,
                "reason": result.get("reason", "Unknown")
            })
            
            return await ai_agent.handle_cnic_verification_failure(extracted_cnic)
    
    except Exception as e:
        logger.error({
            "action": "cnic_verification_error_flexible",
            "sender_id": sender_id,
            "error": str(e)
        })
        
        return await ai_agent.handle_error_gracefully(e, user_message_clean, "", "cnic_verification")

async def handle_otp_verification(sender_id: str, user_message: str) -> str:
    """Handle OTP verification after CNIC verification."""
    
    user_data = authenticated_users[sender_id]
    first_name = user_data.get("name", "").split()[0]
    
    if is_valid_otp(user_message.strip()):
        # Get detailed account information for smart selection
        accounts = user_data.get("accounts", [])
        account_details = await get_account_details_from_backend(accounts)
        
        # Store account details for smart selection
        set_user_accounts_with_details(sender_id, account_details)
        
        set_user_verification_stage(
            sender_id,
            VERIFICATION_STAGES["OTP_VERIFIED"],
            cnic=user_data["cnic"],
            name=user_data["name"],
            accounts=user_data["accounts"]
        )
        
        logger.info({
            "action": "otp_verified_successfully",
            "sender_id": sender_id,
            "otp_entered": user_message.strip(),
            "account_details_loaded": len(account_details),
            "next_step": "smart_account_selection"
        })
        
        return await ai_agent.handle_otp_success(user_data["name"], user_data["accounts"])
    
    else:
        logger.warning({
            "action": "invalid_otp_format",
            "sender_id": sender_id,
            "otp_entered": user_message.strip()
        })
        
        return await ai_agent.handle_otp_failure(user_message.strip(), first_name)

async def handle_account_selection(sender_id: str, user_message: str) -> str:
    """Handle smart account selection with natural language support."""
    
    user_data = authenticated_users[sender_id]
    accounts = user_data.get("accounts", [])
    first_name = user_data.get("name", "").split()[0]
    
    # Get detailed account information
    account_details = get_user_accounts_with_details(sender_id)
    # Add this logging block:
    logger.info(f"Account selection debug - User input: '{user_message}'")
    logger.info(f"Available accounts: {[acc['account_number'] for acc in account_details]}")
    
    
    # If no account details stored, fetch them
    if not account_details:
        account_details = await get_account_details_from_backend(accounts)
        set_user_accounts_with_details(sender_id, account_details)
    
    # Use smart account selection
    selected_account = await smart_account_selection(user_message, account_details)
    
    if selected_account:
        try:
            # Verify account selection with backend
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/select_account",
                    json={
                        "cnic": user_data["cnic"],
                        "account_number": selected_account
                    }
                )
                result = response.json()
            
            if result["status"] == "success":
                # Update to final verification stage
                set_user_verification_stage(
                    sender_id,
                    VERIFICATION_STAGES["ACCOUNT_SELECTED"],
                    cnic=user_data["cnic"],
                    name=user_data["name"],
                    selected_account=selected_account
                )
                
                logger.info({
                    "action": "smart_account_selected_successfully",
                    "sender_id": sender_id,
                    "user_input": user_message,
                    "selected_account": selected_account,
                    "selection_method": "smart_selection",
                    "fully_authenticated": True
                })
                
                return await ai_agent.handle_account_confirmation(selected_account, user_data["name"])
            
            else:
                return await ai_agent.handle_error_gracefully(
                    Exception("Account selection failed"), 
                    user_message, 
                    first_name, 
                    "account_selection"
                )
        
        except Exception as e:
            logger.error({
                "action": "smart_account_selection_error",
                "sender_id": sender_id,
                "error": str(e)
            })
            return await ai_agent.handle_error_gracefully(e, user_message, first_name, "account_selection")
    
    else:
        # No smart selection match, provide guidance with LLM help
        try:
            # Use AI agent to provide helpful guidance
            context_state = "User account selection unclear, providing helpful guidance with available options"
            account_list = [f"{i+1}. {acc['account_number']} ({acc['currency']})" for i, acc in enumerate(account_details)]
            data = {
                "available_accounts": account_list,
                "user_input": user_message,
                "selection_examples": ["my USD account", "first account", "1234 (last 4 digits)", "PKR account"]
            }
            conversation_history = ""
            return await ai_agent.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
        except:
            return await ai_agent.handle_account_selection(user_message, accounts, first_name)
        

async def handle_banking_queries(sender_id: str, user_message: str) -> str:
    """Handle banking queries for fully authenticated users."""
    
    user_info = get_user_account_info(sender_id)
    account_number = user_info["account_number"]
    first_name = user_info["name"].split()[0]
    
    try:
        logger.info({
            "action": "processing_banking_query",
            "sender_id": sender_id,
            "account_number": account_number,
            "user_message": user_message,
            "approach": "llm_first_with_transfer_confirmation"
        })
        
        # Make API call to backend process_query endpoint
        response = await call_process_query_api(
            user_message=user_message,
            account_number=account_number,
            first_name=first_name
        )
        
        # Check if response indicates transfer OTP is required
        if response.startswith("OTP_REQUIRED|"):
            # Parse transfer details from response
            parts = response.split("|")
            if len(parts) == 4:
                amount = float(parts[1])
                currency = parts[2]
                recipient = parts[3]
                
                # Store transfer details and set OTP pending
                set_pending_transfer_info(sender_id, amount, currency, recipient)
                set_user_verification_stage(
                    sender_id,
                    VERIFICATION_STAGES["TRANSFER_OTP_PENDING"],
                    cnic=user_info["cnic"],
                    name=user_info["name"],
                    selected_account=account_number
                )
                
                logger.info({
                    "action": "transfer_otp_required",
                    "sender_id": sender_id,
                    "amount": amount,
                    "currency": currency,
                    "recipient": recipient
                })
                
                return await ai_agent.handle_transfer_otp_request(amount, currency, recipient, first_name)
            else:
                return "Sorry, there was an error processing your transfer request. Please try again."
        
        logger.info({
            "action": "banking_query_processed_successfully",
            "sender_id": sender_id,
            "response_length": len(response)
        })
        
        return response
        
    except Exception as e:
        logger.error({
            "action": "banking_query_error",
            "sender_id": sender_id,
            "error": str(e),
            "user_message": user_message
        })
        return await ai_agent.handle_error_gracefully(e, user_message, first_name, "banking_query")

async def handle_transfer_otp_verification(sender_id: str, user_message: str) -> str:
    """Handle OTP verification for money transfer - now leads to confirmation step."""
    
    try:
        user_data = authenticated_users.get(sender_id, {})
        
        if not user_data:
            logger.error({
                "action": "transfer_otp_no_user_data",
                "sender_id": sender_id
            })
            return "Session expired. Please start over by sending 'hi'."
        
        user_name = user_data.get("name", "")
        first_name = user_name.split()[0] if user_name else "there"
        account_number = user_data.get("selected_account", "")
        cnic = user_data.get("cnic", "")
        
        if not account_number:
            logger.error({
                "action": "transfer_otp_no_account",
                "sender_id": sender_id
            })
            return "Account information missing. Please restart your session."
        
        # Check if OTP is valid (1-5 digits)
        if is_valid_otp(user_message.strip()):
            # OTP is valid, move to confirmation step instead of executing transfer
            transfer_info = get_pending_transfer_info(sender_id)
            
            if not transfer_info or not all([
                transfer_info.get("amount"), 
                transfer_info.get("currency"), 
                transfer_info.get("recipient")
            ]):
                logger.error({
                    "action": "transfer_otp_no_pending_transfer",
                    "sender_id": sender_id,
                    "transfer_info": transfer_info
                })
                return "No pending transfer found. Please start the transfer process again."
            
            amount = transfer_info["amount"]
            currency = transfer_info["currency"]
            recipient = transfer_info["recipient"]
            
            # Move to confirmation stage instead of executing transfer
            set_user_verification_stage(
                sender_id,
                VERIFICATION_STAGES["TRANSFER_CONFIRMATION_PENDING"],
                cnic=cnic,
                name=user_name,
                selected_account=account_number
            )
            
            logger.info({
                "action": "transfer_otp_verified_requesting_confirmation",
                "sender_id": sender_id,
                "amount": amount,
                "currency": currency,
                "recipient": recipient,
                "next_step": "transfer_confirmation"
            })
            
            # Ask for confirmation using AI agent
            return await ai_agent.handle_transfer_confirmation_request(amount, currency, recipient, first_name)
        
        else:
            logger.warning({
                "action": "invalid_transfer_otp_format",
                "sender_id": sender_id,
                "otp_entered": user_message.strip()
            })
            
            return await ai_agent.handle_otp_failure(user_message.strip(), first_name)
    
    except Exception as e:
        logger.error({
            "action": "transfer_otp_verification_error",
            "sender_id": sender_id,
            "error": str(e),
            "user_message": user_message
        })
        
        return "Sorry, there was an error processing your transfer OTP. Please try again or restart the transfer process."

async def handle_transfer_confirmation(sender_id: str, user_message: str) -> str:
    """Handle transfer confirmation step - NEW FUNCTION."""
    
    try:
        user_data = authenticated_users.get(sender_id, {})
        user_name = user_data.get("name", "")
        first_name = user_name.split()[0] if user_name else "there"
        account_number = user_data.get("selected_account", "")
        cnic = user_data.get("cnic", "")
        
        transfer_info = get_pending_transfer_info(sender_id)
        
        if not transfer_info or not all([
            transfer_info.get("amount"), 
            transfer_info.get("currency"), 
            transfer_info.get("recipient")
        ]):
            logger.error({
                "action": "transfer_confirmation_no_pending_transfer",
                "sender_id": sender_id
            })
            return "No pending transfer found. Please start the transfer process again."
        
        amount = transfer_info["amount"]
        currency = transfer_info["currency"]
        recipient = transfer_info["recipient"]
        
        # Check user's confirmation response
        if is_confirmation_positive(user_message):
            # User confirmed - proceed with transfer
            
            # Clear pending transfer info and reset to fully authenticated
            clear_pending_transfer_info(sender_id)
            set_user_verification_stage(
                sender_id,
                VERIFICATION_STAGES["ACCOUNT_SELECTED"],
                cnic=cnic,
                name=user_name,
                selected_account=account_number
            )
            
            logger.info({
                "action": "transfer_confirmed_proceeding",
                "sender_id": sender_id,
                "amount": amount,
                "currency": currency,
                "recipient": recipient,
                "user_confirmation": user_message
            })
            
            # Execute the transfer using AI agent
            memory = ai_agent.get_user_memory(account_number)
            response = await ai_agent.execute_verified_transfer(
                account_number, amount, currency, recipient, first_name, memory
            )
            
            return response
            
        elif is_confirmation_negative(user_message):
            # User cancelled - clear transfer and return to normal state
            
            clear_pending_transfer_info(sender_id)
            set_user_verification_stage(
                sender_id,
                VERIFICATION_STAGES["ACCOUNT_SELECTED"],
                cnic=cnic,
                name=user_name,
                selected_account=account_number
            )
            
            logger.info({
                "action": "transfer_cancelled_by_user",
                "sender_id": sender_id,
                "amount": amount,
                "currency": currency,
                "recipient": recipient,
                "user_response": user_message
            })
            
            return await ai_agent.handle_transfer_cancellation(amount, currency, recipient, first_name)
            
        else:
            # Unclear response - ask for clarification
            logger.info({
                "action": "transfer_confirmation_unclear_response",
                "sender_id": sender_id,
                "user_response": user_message
            })
            
            return await ai_agent.handle_transfer_confirmation_clarification(amount, currency, recipient, first_name)
    
    except Exception as e:
        logger.error({
            "action": "transfer_confirmation_error",
            "sender_id": sender_id,
            "error": str(e),
            "user_message": user_message
        })
        
        return "Sorry, there was an error processing your confirmation. Please try again."

async def call_process_query_api(user_message: str, account_number: str, first_name: str) -> str:
    """Make API call to backend process_query endpoint."""
    try:
        payload = {
            "user_message": user_message,
            "account_number": account_number,
            "first_name": first_name
        }
        
        logger.info({
            "action": "calling_process_query_api",
            "payload": payload
        })
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/process_query",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"] == "success":
                logger.info({
                    "action": "process_query_api_success",
                    "account_number": account_number,
                    "response_preview": result["response"][:100] + "..."
                })
                return result["response"]
            else:
                logger.error({
                    "action": "process_query_api_error",
                    "error": result.get("error", "Unknown error"),
                    "account_number": account_number
                })
                return result.get("response", "Sorry, I couldn't process your request. Please try again.")
                
    except httpx.TimeoutException:
        logger.error({
            "action": "process_query_api_timeout",
            "account_number": account_number,
            "user_message": user_message
        })
        return "Request timed out. Please try again with a simpler query."
        
    except httpx.HTTPStatusError as e:
        logger.error({
            "action": "process_query_api_http_error",
            "status_code": e.response.status_code,
            "account_number": account_number,
            "error": str(e)
        })
        return "Backend service error. Please try again later."
        
    except Exception as e:
        logger.error({
            "action": "process_query_api_unexpected_error",
            "error": str(e),
            "account_number": account_number
        })
        return "Unexpected error occurred. Please try again."

def send_message(recipient_id, message_text):
    """Send response to Facebook Messenger."""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info({
            "action": "message_sent_successfully",
            "recipient_id": recipient_id,
            "response_status": response.status_code
        })
    except requests.exceptions.RequestException as e:
        logger.error({
            "action": "send_message_error",
            "recipient_id": recipient_id,
            "error": str(e)
        })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            backend_healthy = response.status_code == 200
            backend_info = response.json() if backend_healthy else {}
    except:
        backend_healthy = False
        backend_info = {}
    
    # Check translation service status
    translation_healthy = hasattr(translation_service, 'use_llm') and translation_service.use_llm
    
    return {
        "status": "healthy",
        "backend_connection": "healthy" if backend_healthy else "unhealthy",
        "translation_service": "llm_enabled" if translation_healthy else "fallback_only",
        "backend_approach": backend_info.get("approach", "unknown"),
        "timestamp": time.time(),
        "service": "banking_webhook_enhanced",
        "features": {
            "flexible_cnic_input": "enabled",
            "smart_account_selection": "enabled", 
            "transfer_confirmation": "enabled",
            "multilingual_support": "enabled",
            "language_detection": "llm_enhanced" if translation_healthy else "basic",
            "authentication_flow": "cnic_otp_smart_account",
            "response_system": "ai_agent_natural_language",
            "exit_functionality": "enabled",
            "greeting_detection": "enabled",
            "llm_first_approach": "enabled",
            "otp_support": "enabled"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)