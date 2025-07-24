# Enhanced webhook.py - OTP authentication, smart account selection, and transfer confirmation
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
import re
from typing import Dict, Any, Tuple
from state import (
    authenticated_users, processed_messages, periodic_cleanup,
    VERIFICATION_STAGES, TRANSFER_STATES, pending_transfers,
    get_user_verification_stage, set_user_verification_stage,
    is_fully_authenticated, get_user_account_info, clear_user_state,
    is_valid_otp, set_pending_transfer, get_pending_transfer, 
    clear_pending_transfer, is_in_transfer_flow, get_transfer_stage,
    restart_user_session, is_banking_related_query
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

def is_restart_command(message: str) -> bool:
    """Check if message is a restart command."""
    restart_commands = [
        "restart", "start over", "begin again", "reset", "start fresh", 
        "new session", "start new", "fresh start", "reload"
    ]
    
    message_lower = message.lower().strip()
    return any(cmd in message_lower for cmd in restart_commands)

async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message with enhanced authentication flow including OTP and smart account selection."""
    
    current_time = time.time()
    
    # Rate limiting
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 2:
            return "I appreciate your enthusiasm! Please give me just a moment to process your previous message before sending another. 😊"
    
    user_last_message_time[sender_id] = current_time

    # 🚪 CHECK FOR RESTART COMMAND (equivalent to page refresh)
    if is_restart_command(user_message):
        logger.info({
            "action": "restart_command_detected",
            "sender_id": sender_id
        })
        
        # Clear all user state (equivalent to fresh page load)
        restart_user_session(sender_id)
        
        # Return to initial greeting
        return await ai_agent.handle_initial_greeting()

    # 🚪 CHECK FOR EXIT COMMAND (before any other processing)
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

    # Check if user is in transfer confirmation flow
    if is_in_transfer_flow(sender_id):
        return await handle_transfer_flow(sender_id, user_message)

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
        return await handle_otp_verification(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["OTP_REQUIRED"]:
        return await handle_account_selection(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["ACCOUNT_SELECTED"]:
        return await handle_banking_queries(sender_id, user_message)
    
    else:
        # Fallback to AI agent session start
        return await ai_agent.handle_initial_greeting()

async def handle_cnic_verification(sender_id: str, user_message: str) -> str:
    """Handle CNIC verification step with proper greeting detection."""
    
    user_message_clean = user_message.strip()
    
    # Check if this is a greeting first, before checking CNIC format
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
            
            # Store CNIC verification data and move to OTP stage
            set_user_verification_stage(
                sender_id, 
                VERIFICATION_STAGES["CNIC_VERIFIED"],
                cnic=user_data["cnic"],
                name=user_data["name"],
                accounts=user_data["accounts"]
            )
            
            logger.info({
                "action": "cnic_verified_successfully_requesting_otp",
                "sender_id": sender_id,
                "cnic": user_data["cnic"],
                "name": user_data["name"],
                "accounts_count": len(user_data["accounts"])
            })
            
            # Use AI agent for OTP request after CNIC verification
            first_name = user_data["name"].split()[0]
            return await ai_agent.handle_otp_request(first_name)
        
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

async def handle_otp_verification(sender_id: str, user_message: str) -> str:
    """Handle OTP verification step (NEW)."""
    
    user_data = authenticated_users[sender_id]
    first_name = user_data.get("name", "").split()[0]
    otp_input = user_message.strip()
    
    # Validate OTP format (1-5 digits)
    if not is_valid_otp(otp_input):
        logger.info({
            "action": "invalid_otp_format",
            "sender_id": sender_id,
            "otp_attempt": otp_input
        })
        
        return await ai_agent.handle_invalid_otp(otp_input, first_name)
    
    try:
        # OTP is valid - move to account selection stage
        set_user_verification_stage(
            sender_id,
            VERIFICATION_STAGES["OTP_REQUIRED"],  # Ready for account selection
            cnic=user_data["cnic"],
            name=user_data["name"],
            accounts=user_data["accounts"],
            otp_verified=True
        )
        
        logger.info({
            "action": "otp_verified_successfully",
            "sender_id": sender_id,
            "otp": otp_input,
            "name": user_data["name"]
        })
        
        # Generate OTP success message and show account options
        otp_success_msg = await ai_agent.handle_otp_verification_success(otp_input, first_name)
        account_display_msg = await ai_agent.handle_account_display(user_data["accounts"], first_name)
        
        # Combine both messages
        return f"{otp_success_msg}\n\n{account_display_msg}"
        
    except Exception as e:
        logger.error({
            "action": "otp_verification_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        return await ai_agent.handle_error_gracefully(e, user_message, first_name, "otp_verification")

async def handle_account_selection(sender_id: str, user_message: str) -> str:
    """Handle smart account selection step (ENHANCED)."""
    
    user_data = authenticated_users[sender_id]
    accounts = user_data.get("accounts", [])
    first_name = user_data.get("name", "").split()[0]
    
    try:
        # Use smart account selection
        selected_account, selection_method = await ai_agent.handle_smart_account_selection(
            user_message, accounts, first_name, BACKEND_URL
        )
        
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
                    "selected_account": selected_account,
                    "selection_method": selection_method
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
            # Invalid selection - provide guidance
            return await ai_agent.generate_smart_account_selection_response(
                None, None, selection_method, first_name, accounts
            )
    
    except Exception as e:
        logger.error({
            "action": "account_selection_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        return await ai_agent.handle_error_gracefully(e, user_message, first_name, "account_selection")

async def handle_banking_queries(sender_id: str, user_message: str) -> str:
    """Handle banking queries for fully authenticated users with transfer flow support."""
    
    user_info = get_user_account_info(sender_id)
    account_number = user_info["account_number"]
    first_name = user_info["name"].split()[0]  # Get first name
    
    try:
        # Check if this is a transfer request
        if is_transfer_request(user_message):
            return await initiate_transfer_flow(sender_id, user_message, account_number, first_name)
        
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

def is_transfer_request(user_message: str) -> bool:
    """Check if user message is a transfer request."""
    transfer_keywords = [
        "transfer", "send money", "send", "pay", "wire", "remit", 
        "i want to transfer", "transfer money", "move money"
    ]
    
    message_lower = user_message.lower()
    return any(keyword in message_lower for keyword in transfer_keywords)

async def initiate_transfer_flow(sender_id: str, user_message: str, account_number: str, first_name: str) -> str:
    """Initiate the enhanced transfer flow with OTP and confirmation (NEW)."""
    try:
        # Parse transfer details using AI agent
        transfer_prompt = f"""
        Extract transfer details from this query:
        Query: "{user_message}"
        
        Extract:
        - amount: number (null if not specified)
        - currency: "PKR" or "USD" (default PKR if not specified)  
        - recipient: string (null if not specified)
        - has_amount: boolean
        - has_recipient: boolean
        
        Return JSON: {{"amount": number, "currency": string, "recipient": string, "has_amount": boolean, "has_recipient": boolean}}
        """
        
        # Use the AI agent's LLM to parse transfer details
        from langchain_core.messages import SystemMessage
        from ai_agent import llm
        
        response = await llm.ainvoke([SystemMessage(content=transfer_prompt)])
        transfer_details = ai_agent.extract_json_from_response(response.content)
        
        if not transfer_details:
            return "I couldn't understand the transfer details. Please specify the amount and recipient clearly."
        
        # Check what information is missing
        missing_parts = []
        
        if not transfer_details.get("has_amount") or not transfer_details.get("amount"):
            missing_parts.append("amount")
        
        if not transfer_details.get("has_recipient") or not transfer_details.get("recipient"):
            missing_parts.append("recipient")
        
        # Handle missing information
        if missing_parts:
            missing_info_msg = "I need more information for the transfer:\n"
            if "amount" in missing_parts:
                missing_info_msg += "• Amount (e.g., 1000 PKR, $50)\n"
            if "recipient" in missing_parts:
                missing_info_msg += "• Recipient name (e.g., to Ahmed)\n"
            missing_info_msg += "\nPlease provide the complete transfer details."
            return missing_info_msg
        
        # All information available - start OTP flow
        amount = transfer_details.get("amount")
        currency = transfer_details.get("currency", "PKR")
        recipient = transfer_details.get("recipient")
        
        if amount <= 0:
            return f"The transfer amount must be positive. Please specify a valid amount."
        
        # Store pending transfer details
        complete_transfer_details = {
            "amount": amount,
            "currency": currency,
            "recipient": recipient,
            "account_number": account_number,
            "from_user": first_name
        }
        
        set_pending_transfer(sender_id, complete_transfer_details, TRANSFER_STATES["OTP_REQUIRED"])
        
        logger.info({
            "action": "transfer_initiated",
            "sender_id": sender_id,
            "amount": amount,
            "currency": currency,
            "recipient": recipient
        })
        
        # Request OTP for transfer
        return await ai_agent.handle_transfer_otp_request(complete_transfer_details, first_name)
        
    except Exception as e:
        logger.error({
            "action": "initiate_transfer_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        return await ai_agent.handle_error_gracefully(e, user_message, first_name, "transfer_initiation")

async def handle_transfer_flow(sender_id: str, user_message: str) -> str:
    """Handle the transfer confirmation flow (NEW)."""
    try:
        transfer_data = get_pending_transfer(sender_id)
        transfer_stage = transfer_data.get("stage", "")
        transfer_details = transfer_data.get("transfer_details", {})
        
        user_info = get_user_account_info(sender_id)
        first_name = user_info.get("name", "").split()[0]
        
        if transfer_stage == TRANSFER_STATES["OTP_REQUIRED"]:
            # Handle OTP verification for transfer
            otp_input = user_message.strip()
            
            if not is_valid_otp(otp_input):
                return await ai_agent.handle_invalid_otp(otp_input, first_name)
            
            # OTP valid - move to confirmation stage
            set_pending_transfer(sender_id, transfer_details, TRANSFER_STATES["AWAITING_CONFIRMATION"])
            
            logger.info({
                "action": "transfer_otp_verified",
                "sender_id": sender_id,
                "otp": otp_input
            })
            
            # Request final confirmation
            return await ai_agent.handle_transfer_confirmation_request(transfer_details, first_name)
        
        elif transfer_stage == TRANSFER_STATES["AWAITING_CONFIRMATION"]:
            # Handle final confirmation
            user_response = user_message.lower().strip()
            
            if user_response in ["yes", "y", "confirm", "proceed", "ok", "okay"]:
                # Execute the transfer
                result = await execute_transfer(transfer_details)
                
                # Clear pending transfer
                clear_pending_transfer(sender_id)
                
                logger.info({
                    "action": "transfer_confirmed_and_executed",
                    "sender_id": sender_id,
                    "result": result.get("status", "unknown")
                })
                
                if result.get("status") == "success":
                    return f"✅ Transfer completed successfully!\n\n{result.get('message', '')}\n\nYour new balance: {result.get('new_balance', 0)} {result.get('currency', 'PKR')}"
                else:
                    return f"❌ Transfer failed: {result.get('reason', 'Unknown error')}"
            
            elif user_response in ["no", "n", "cancel", "stop", "abort"]:
                # Cancel the transfer
                clear_pending_transfer(sender_id)
                
                logger.info({
                    "action": "transfer_cancelled_by_user",
                    "sender_id": sender_id
                })
                
                return await ai_agent.handle_transfer_cancellation(transfer_details, first_name)
            
            else:
                # Invalid response
                return "Please respond with 'Yes' to proceed with the transfer or 'No' to cancel it."
        
        else:
            # Unknown transfer stage - clear and restart
            clear_pending_transfer(sender_id)
            return "Transfer session expired. Please start a new transfer request."
            
    except Exception as e:
        logger.error({
            "action": "handle_transfer_flow_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        
        # Clear pending transfer on error
        clear_pending_transfer(sender_id)
        
        user_info = get_user_account_info(sender_id)
        first_name = user_info.get("name", "").split()[0]
        return await ai_agent.handle_error_gracefully(e, user_message, first_name, "transfer_flow")

async def execute_transfer(transfer_details: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the actual money transfer."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/transfer_money",
                json={
                    "from_account": transfer_details["account_number"],
                    "to_recipient": transfer_details["recipient"],
                    "amount": transfer_details["amount"],
                    "currency": transfer_details["currency"]
                }
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Transfer execution error: {e}")
        return {"status": "fail", "reason": f"Transfer execution failed: {str(e)}"}

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
        "service": "enhanced_banking_webhook",
        "authentication_flow": "cnic_otp_smart_account_selection",
        "features": {
            "otp_verification": "enabled",
            "smart_account_selection": "enabled", 
            "transfer_confirmation": "enabled",
            "context_filtering": "enabled",
            "restart_functionality": "enabled",
            "exit_functionality": "enabled",
            "greeting_detection": "enabled",
            "varied_responses": "enabled"
        },
        "account_selection_methods": [
            "USD account", "PKR account", 
            "1st account", "2nd account", "3rd account",
            "last 4 digits"
        ],
        "transfer_flow": "OTP -> Confirmation -> Execution"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)