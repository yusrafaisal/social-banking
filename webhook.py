# Updated webhook.py - Fixed greeting flow and natural responses
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
import re
from typing import Dict, Any
from state import (
    authenticated_users, processed_messages, periodic_cleanup,
    VERIFICATION_STAGES, get_user_verification_stage, set_user_verification_stage,
    is_fully_authenticated, get_user_account_info, clear_user_state
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
                    response_text = await process_user_message(sender_id, user_message)
                    send_message(sender_id, response_text)

    if len(processed_messages) % 100 == 0:
        periodic_cleanup()

    return JSONResponse(content={"status": "ok"})

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

async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message with CNIC-based authentication flow using AI agent responses."""
    
    current_time = time.time()
    
    # Rate limiting
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 2:
            return "I appreciate your enthusiasm! Please give me just a moment to process your previous message before sending another. 😊"
    
    user_last_message_time[sender_id] = current_time

    # 🚪 CHECK FOR EXIT COMMAND FIRST (before any other processing)
    if user_message.strip().lower() == "exit":
        logger.info({
            "action": "exit_command_detected",
            "sender_id": sender_id
        })
        
        # Get user info for personalized goodbye
        user_info = get_user_account_info(sender_id)
        first_name = user_info.get("name", "").split()[0] if user_info.get("name") else ""
        account_number = user_info.get("account_number", "")
        
        # Clear user session completely
        clear_user_state(sender_id)
        
        logger.info({
            "action": "session_terminated",
            "sender_id": sender_id
        })
        
        # Use AI agent for natural session end response
        return await ai_agent.handle_session_end(account_number, first_name)

    # Get current verification stage
    verification_stage = get_user_verification_stage(sender_id)
    
    logger.info({
        "action": "processing_user_message",
        "sender_id": sender_id,
        "verification_stage": verification_stage,
        "user_message": user_message
    })

    # Handle different verification stages
    if verification_stage == VERIFICATION_STAGES["NOT_VERIFIED"]:
        return await handle_cnic_verification(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["CNIC_VERIFIED"]:
        return await handle_account_selection(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["ACCOUNT_SELECTED"]:
        return await handle_banking_queries(sender_id, user_message)
    
    else:
        # Fallback to AI agent session start
        return await ai_agent.handle_session_start()

async def handle_cnic_verification(sender_id: str, user_message: str) -> str:
    """Handle CNIC verification step with proper greeting detection."""
    
    user_message_clean = user_message.strip()
    
    # 🔧 FIX 1: Check if this is a greeting first, before checking CNIC format
    if is_greeting_message(user_message_clean):
        logger.info({
            "action": "initial_greeting_detected",
            "sender_id": sender_id,
            "message": user_message_clean
        })
        
        # Use AI agent for proper initial greeting and CNIC request
        return await ai_agent.handle_initial_greeting()
    
    # Now check if message looks like a CNIC
    cnic_pattern = r'^\d{5}-\d{7}-\d$'
    
    if not re.match(cnic_pattern, user_message_clean):
        # Use AI agent for natural invalid format response
        return await ai_agent.handle_invalid_cnic_format(user_message_clean)
    
    try:
        # Verify CNIC with backend
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/verify_cnic",
                json={"cnic": user_message_clean}
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
                "action": "cnic_verified_successfully",
                "sender_id": sender_id,
                "cnic": user_data["cnic"],
                "name": user_data["name"],
                "accounts_count": len(user_data["accounts"])
            })
            
            # Use AI agent for natural verification success response
            return await ai_agent.handle_cnic_verification_success(
                user_data["name"], 
                user_data["accounts"], 
                user_data["cnic"]
            )
        
        else:
            logger.warning({
                "action": "cnic_verification_failed",
                "sender_id": sender_id,
                "cnic": user_message_clean,
                "reason": result.get("reason", "Unknown")
            })
            
            # Use AI agent for natural verification failure response
            return await ai_agent.handle_cnic_verification_failure(user_message_clean)
    
    except Exception as e:
        logger.error({
            "action": "cnic_verification_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        
        # Use AI agent for natural error response
        return await ai_agent.handle_error_gracefully(e, user_message_clean, "", "cnic_verification")

async def handle_account_selection(sender_id: str, user_message: str) -> str:
    """Handle account selection step using last 4 digits and AI agent responses."""
    
    user_data = authenticated_users[sender_id]
    accounts = user_data.get("accounts", [])
    first_name = user_data.get("name", "").split()[0]
    
    # Check if user entered 4 digits (last 4 digits of account)
    selection = user_message.strip()
    
    if selection.isdigit() and len(selection) == 4:
        try:
            # Find account that ends with these 4 digits
            selected_account = None
            for account in accounts:
                if account.endswith(selection):
                    selected_account = account
                    break
            
            if selected_account:
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
                        "action": "account_selected_successfully",
                        "sender_id": sender_id,
                        "cnic": user_data["cnic"],
                        "selected_account": selected_account
                    })
                    
                    # Use AI agent for natural account confirmation response
                    return await ai_agent.handle_account_confirmation(selected_account, user_data["name"])
                
                else:
                    # Use AI agent for natural error response
                    return await ai_agent.handle_error_gracefully(
                        Exception("Account selection failed"), 
                        user_message, 
                        first_name, 
                        "account_selection"
                    )
            
            else:
                # No matching account found - use AI agent for natural response
                return await ai_agent.handle_account_selection(selection, accounts, first_name)
        
        except Exception as e:
            logger.error({
                "action": "account_selection_error",
                "sender_id": sender_id,
                "error": str(e)
            })
            return await ai_agent.handle_error_gracefully(e, user_message, first_name, "account_selection")
    
    else:
        # Invalid format - use AI agent for natural guidance response
        return await ai_agent.handle_account_selection(selection, accounts, first_name)

async def handle_banking_queries(sender_id: str, user_message: str) -> str:
    """Handle banking queries for fully authenticated users."""
    
    user_info = get_user_account_info(sender_id)
    account_number = user_info["account_number"]
    first_name = user_info["name"].split()[0]  # Get first name
    
    try:
        logger.info({
            "action": "processing_banking_query",
            "sender_id": sender_id,
            "account_number": account_number,
            "user_message": user_message
        })
        
        # Make API call to backend process_query endpoint
        response = await call_process_query_api(
            user_message=user_message,
            account_number=account_number,
            first_name=first_name
        )
        
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
        # Use AI agent for natural error response
        return await ai_agent.handle_error_gracefully(e, user_message, first_name, "banking_query")

async def call_process_query_api(user_message: str, account_number: str, first_name: str) -> str:
    """Make API call to backend process_query endpoint."""
    try:
        payload = {
            "user_message": user_message,
            "account_number": account_number,
            "first_name": first_name
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/process_query",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"] == "success":
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
    except:
        backend_healthy = False
    
    return {
        "status": "healthy",
        "backend_connection": "healthy" if backend_healthy else "unhealthy",
        "timestamp": time.time(),
        "service": "banking_webhook_natural_responses",
        "authentication_flow": "cnic_verification_last_4_digits",
        "response_system": "ai_agent_natural_language",
        "account_selection": "last_4_digits",
        "exit_functionality": "enabled",
        "greeting_detection": "enabled",
        "varied_responses": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)