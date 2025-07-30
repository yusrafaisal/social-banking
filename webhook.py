# Updated webhook.py - LLM-First approach with constants migration
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
from prompts import account_selection_prompt
from langchain_core.messages import SystemMessage
import aiohttp
from openai import OpenAI

from ai_agent import llm

# Import constants
from constants import (
    VerificationStages, GreetingWords, ConfirmationWords, ExitCommands,
    Limits, WebhookConfig, RegexPatterns, Currencies, StatusMessages,
    TransferSignals, DatabaseFields, LLMConfig
)

import os
import requests
import re
from typing import Dict, Any, List
from translation_service import translation_service
from state import (
    authenticated_users, processed_messages, periodic_cleanup,
    get_user_verification_stage, set_user_verification_stage,
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

VERIFY_TOKEN = WebhookConfig.VERIFY_TOKEN

PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPAP0Nhv5ZBWpiL6hboNQXrbnp1NKardW6jeCyJbXVKNR0MCooUu0H7xaINNxHLsgM6ZAddqPHTp7J17Kjl9AjqZAZB0xlyNLIIsA9q3VVtCIva0MZA3OQKKUZAZBP4qZBZBfVMr3KID33fAmI7aS4FUxDUczo6HrSW1MbOmoyIw3F2yEAITdwMTt6yYZCw"

# Add this after your other global variables (around line 40-50, after BACKEND_URL)
BACKEND_URL = WebhookConfig.BACKEND_URL

# Initialize AI Agent for natural responses
ai_agent = BankingAIAgent()

# Add voice message deduplication cache
voice_message_last_time = {}

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

# Add to webhook.py constants
VOICE_MESSAGE_COOLDOWN = 15  # seconds

# In receive_message function
webhook_message_cache = {}

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
            sender_id = messaging_event["sender"]["id"]

            if "message" in messaging_event:
                # Add webhook-level deduplication
                message_id = messaging_event["message"].get("mid")
                if message_id and message_id in webhook_message_cache:
                    logger.info(f"🔄 WEBHOOK: Duplicate message blocked at webhook level: {message_id}")
                    continue
                
                if message_id:
                    webhook_message_cache[message_id] = time.time()
                    
                # Clean old cache entries (keep last 100)
                if len(webhook_message_cache) > 100:
                    old_entries = sorted(webhook_message_cache.items(), key=lambda x: x[1])[:50]
                    for old_id, _ in old_entries:
                        del webhook_message_cache[old_id]

                # Handle text messages
                if "text" in messaging_event["message"]:
                    user_message = messaging_event["message"]["text"]
                    response_text = await process_multilingual_message(sender_id, user_message)
                    send_message(sender_id, response_text)

                # Handle voice messages with rate limiting
                elif "attachments" in messaging_event["message"]:
                    for attachment in messaging_event["message"]["attachments"]:
                        if attachment["type"] == "audio":
                            # Check voice message rate limit
                            current_time = time.time()
                            
                            if sender_id in voice_message_last_time:
                                if current_time - voice_message_last_time[sender_id] < VOICE_MESSAGE_COOLDOWN:
                                    logger.info(f"🎤 Voice message rate limited for {sender_id}")
                                    send_message(sender_id, "Please wait a moment before sending another voice message. ")
                                    continue
                            
                            # Update voice message timestamp
                            voice_message_last_time[sender_id] = current_time
                            
                            audio_url = attachment["payload"]["url"]
                            response_text = await handle_voice_message(sender_id, audio_url)
                            send_message(sender_id, response_text)

    return JSONResponse(content={"status": "ok"})

voice_message_cache = {}
async def handle_voice_message(sender_id: str, audio_url: str) -> str:
    """Handle voice messages with deduplication and proper language handling."""
    try:
        current_time = time.time()
        
        # Check for recent voice message processing
        cache_key = f"{sender_id}:voice"
        if cache_key in voice_message_cache:
            last_time, last_response = voice_message_cache[cache_key]
            if current_time - last_time < 10:  # 10 second cooldown
                logger.info(f"🎤 Voice message too soon, returning cached response for {sender_id}")
                return "I'm still processing your previous voice message. Please wait a moment."
        
        # Download audio
        async with aiohttp.ClientSession() as session:
            async with session.get(audio_url) as response:
                if response.status != 200:
                    raise Exception("Failed to download audio file")
                audio_data = await response.read()

        # Create unique filename to avoid conflicts
        audio_file_path = f"temp_audio_{sender_id}_{int(current_time)}.mp3"
        
        try:
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_data)

            # Transcribe
            transcription = await transcribe_audio(audio_file_path)
            
            # Check for duplicate transcription
            transcription_hash = hash(transcription.strip().lower())
            recent_key = f"{sender_id}:transcription:{transcription_hash}"
            
            if recent_key in voice_message_cache:
                last_time, last_response = voice_message_cache[recent_key]
                if current_time - last_time < 30:  # 30 seconds for same transcription
                    logger.info(f"🎤 Duplicate transcription detected: '{transcription[:50]}...'")
                    return last_response

            # Detect language of transcription
            detected_language = translation_service.detect_language_smart(
                transcription, sender_id, get_user_last_language
            )
            
            logger.info(f"🎤 VOICE: Detected language '{detected_language}' for transcription: '{transcription}'")
            
            # Store the detected language BEFORE processing
            set_user_language(sender_id, detected_language)
            
            # Translate to English for processing if needed
            if detected_language != "en":
                english_transcription = translation_service.translate_to_english(transcription, detected_language)
                logger.info(f"🎤 VOICE: Translated '{transcription}' to '{english_transcription}'")
            else:
                english_transcription = transcription

            # Process the English transcription directly (not through multilingual processor)
            english_response = await process_user_message(sender_id, english_transcription)
            
            # Translate response back to user's ORIGINAL language
            if detected_language != "en":
                final_response = translation_service.translate_from_english(english_response, detected_language)
                logger.info(f"🎤 VOICE: Final response translated to {detected_language}: '{final_response[:100]}...'")
            else:
                final_response = english_response
            
            # Cache the result
            voice_message_cache[cache_key] = (current_time, final_response)
            voice_message_cache[recent_key] = (current_time, final_response)
            
            return final_response

        finally:
            # Safe cleanup
            try:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                    logger.info(f"🗑️ Cleaned up audio file: {audio_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")

    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        return "Sorry, I couldn't process your voice message. Please try typing your question instead."
    

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) 

async def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio using OpenAI's new SDK (>=1.0.0)"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"  # ensures raw string instead of JSON
            )
        return response
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise Exception("Failed to transcribe audio")

async def process_multilingual_message(sender_id: str, user_message: str) -> str:
    """Process message with language detection and translation support."""
    
    try:
        current_time = time.time()
        message_hash = hash(user_message.strip().lower())
        translation_cache_key = f"{sender_id}:translation:{message_hash}"
        
        if translation_cache_key in user_request_cache:
            last_time, last_response = user_request_cache[translation_cache_key]
            if current_time - last_time < 30:
                logger.info(f"🌐 TRANSLATION DUPLICATE BLOCKED: '{user_message[:50]}...' (sent {current_time - last_time:.1f}s ago)")
                return last_response
        
        # Detect language of incoming message - SAVE THIS FOR LATER USE
        original_detected_language = translation_service.detect_language_smart(
            user_message, 
            sender_id, 
            get_user_last_language
        )
        
        # Store the detected language for this user
        set_user_language(sender_id, original_detected_language)
        
        logger.info({
            "action": "language_detected",
            "sender_id": sender_id,
            "detected_language": original_detected_language,
            "original_message": user_message
        })
        
        # Translate to English for processing if needed
        if original_detected_language != 'en':
            english_message = translation_service.translate_to_english(user_message, original_detected_language)
            logger.info({
                "action": "message_translated_to_english",
                "sender_id": sender_id,
                "original": user_message,
                "translated": english_message,
                "source_language": original_detected_language
            })
        else:
            english_message = user_message
        
        # Process the English message through existing flow
        english_response = await process_user_message(sender_id, english_message)
        
        # Translate response back to user's ORIGINAL language (not re-detected)
        if original_detected_language != 'en':
            final_response = translation_service.translate_from_english(english_response, original_detected_language)
            logger.info({
                "action": "response_translated_to_user_language",
                "sender_id": sender_id,
                "english_response": english_response[:100] + "...",
                "translated_response": final_response[:100] + "...",
                "target_language": original_detected_language
            })
        else:
            final_response = english_response
        
        # Cache the final translated response
        user_request_cache[translation_cache_key] = (current_time, final_response)
        
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
    message_lower = message.lower().strip()
    
    # Check if message is exactly a greeting or starts with greeting
    for greeting in GreetingWords.BASIC_GREETINGS:
        if message_lower == greeting or message_lower.startswith(greeting + " "):
            return True
    
    # Check for common greeting patterns
    for pattern in RegexPatterns.GREETING_PATTERNS:
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
                    json={DatabaseFields.ACCOUNT_NUMBER: account}
                )
                if response.status_code == 200:
                    result = response.json()
                    if result["status"] == StatusMessages.SUCCESS:
                        user_info = result["user"]
                        account_details.append({
                            DatabaseFields.ACCOUNT_NUMBER: account,
                            "currency": user_info.get(DatabaseFields.ACCOUNT_CURRENCY, Currencies.PKR_LOWER).upper(),
                            "balance_usd": user_info.get("current_balance_usd", 0),
                            "balance_pkr": user_info.get("current_balance_pkr", 0)
                        })
        except Exception as e:
            logger.error(f"Error getting account details for {account}: {e}")
            # Add account with default info if API fails
            account_details.append({
                DatabaseFields.ACCOUNT_NUMBER: account,
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
                f"{i}. Account: {account[DatabaseFields.ACCOUNT_NUMBER]}, Currency: {account['currency']}, "
                f"Balance: {account.get('balance_pkr', 0)} {Currencies.PKR} / {account.get('balance_usd', 0)} {Currencies.USD}"
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
            if account[DatabaseFields.ACCOUNT_NUMBER] == selected_account:
                logger.info(f"LLM selected account: {selected_account} for input: {user_input}")
                return selected_account
        
        # Fallback: try partial matching if LLM returned partial number
        if selected_account.isdigit():
            for account in account_details:
                if account[DatabaseFields.ACCOUNT_NUMBER].endswith(selected_account):
                    logger.info(f"LLM partial match: {account[DatabaseFields.ACCOUNT_NUMBER]} for input: {user_input}")
                    return account[DatabaseFields.ACCOUNT_NUMBER]
        
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
        if account[DatabaseFields.ACCOUNT_NUMBER] == user_input_clean:
            logger.info(f"Fallback: Exact account number match: {user_input_clean}")
            return account[DatabaseFields.ACCOUNT_NUMBER]
    
    # Check for partial number matches (last 4, 5, 6+ digits)
    if user_input_clean.isdigit() and len(user_input_clean) >= 4:
        for account in account_details:
            if account[DatabaseFields.ACCOUNT_NUMBER].endswith(user_input_clean):
                logger.info(f"Fallback: Partial match {user_input_clean} -> {account[DatabaseFields.ACCOUNT_NUMBER]}")
                return account[DatabaseFields.ACCOUNT_NUMBER]
    
    user_input_lower = user_input_clean.lower()
    
    # Currency-based selection
    if Currencies.USD_LOWER in user_input_lower or "dollar" in user_input_lower:
        usd_accounts = [acc for acc in account_details if acc["currency"] == Currencies.USD]
        if usd_accounts:
            logger.info(f"Fallback: USD currency match -> {usd_accounts[0][DatabaseFields.ACCOUNT_NUMBER]}")
            return usd_accounts[0][DatabaseFields.ACCOUNT_NUMBER]
    
    if Currencies.PKR_LOWER in user_input_lower or "rupee" in user_input_lower:
        pkr_accounts = [acc for acc in account_details if acc["currency"] == Currencies.PKR]
        if pkr_accounts:
            logger.info(f"Fallback: PKR currency match -> {pkr_accounts[0][DatabaseFields.ACCOUNT_NUMBER]}")
            return pkr_accounts[0][DatabaseFields.ACCOUNT_NUMBER]
    
    # Position-based selection
    if any(word in user_input_lower for word in ["first", "1st", "one"]):
        if account_details:
            logger.info(f"Fallback: First account -> {account_details[0][DatabaseFields.ACCOUNT_NUMBER]}")
            return account_details[0][DatabaseFields.ACCOUNT_NUMBER]
    
    if any(word in user_input_lower for word in ["second", "2nd", "two"]):
        if len(account_details) > 1:
            logger.info(f"Fallback: Second account -> {account_details[1][DatabaseFields.ACCOUNT_NUMBER]}")
            return account_details[1][DatabaseFields.ACCOUNT_NUMBER]
    
    logger.warning(f"Fallback: No match found for input: {user_input_clean}")
    return None

def is_confirmation_positive(message: str) -> bool:
    """Check if user message is a positive confirmation."""
    message_lower = message.lower().strip()
    return any(word in message_lower for word in ConfirmationWords.POSITIVE)

def is_confirmation_negative(message: str) -> bool:
    """Check if user message is a negative confirmation."""
    message_lower = message.lower().strip()
    return any(word in message_lower for word in ConfirmationWords.NEGATIVE)

user_request_cache = {}
async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message with enhanced LLM-based exit detection."""
    
    current_time = time.time()

    # CRITICAL: Check for duplicate messages first
    message_hash = hash(user_message.strip().lower())
    cache_key = f"{sender_id}:{message_hash}"
    
    if cache_key in user_request_cache:
        last_time, last_response = user_request_cache[cache_key]
        if current_time - last_time < 30:  # 30 seconds
            logger.info(f"🔄 DUPLICATE MESSAGE BLOCKED: '{user_message[:50]}...' (sent {current_time - last_time:.1f}s ago)")
            return last_response
    
    # Rate limiting (existing code) - FIXED: Remove duplicate rate limiting
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < Limits.MESSAGE_RATE_LIMIT_SECONDS:
            return "I appreciate your enthusiasm! Please give me just a moment to process your previous message before sending another."
    
    user_last_message_time[sender_id] = current_time

    # LLM-based exit command detection - NEW AND IMPROVED!
    if await ai_agent.detect_exit_intent_with_llm(user_message):
        logger.info({
            "action": "llm_exit_intent_detected",
            "sender_id": sender_id,
            "original_message": user_message,
            "detection_method": "llm_based"
        })
        
        user_info = get_user_account_info(sender_id)
        first_name = user_info.get(DatabaseFields.NAME, "").split()[0] if user_info.get(DatabaseFields.NAME) else ""
        account_number = user_info.get(DatabaseFields.ACCOUNT_NUMBER, "")
        
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
            "action": "session_terminated_via_llm_detection",
            "sender_id": sender_id,
            "exit_phrase": user_message
        })
        
        response = await ai_agent.handle_session_end(account_number, first_name)
        
        # Cache the exit response
        user_request_cache[cache_key] = (current_time, response)
        return response

    # Get current verification stage
    verification_stage = get_user_verification_stage(sender_id)
    
    logger.info({
        "action": "processing_user_message",
        "sender_id": sender_id,
        "verification_stage": verification_stage,
        "user_message": user_message,
        "enhanced_features": "flexible_cnic_smart_account_transfer_confirmation_llm_exit"
    })

    # Process based on verification stage and cache the response
    try:
        response = None
        
        # Handle different verification stages
        if verification_stage == VerificationStages.NOT_VERIFIED:
            response = await handle_cnic_verification(sender_id, user_message)
        
        elif verification_stage == VerificationStages.CNIC_VERIFIED:
            response = await handle_otp_verification(sender_id, user_message)
        
        elif verification_stage == VerificationStages.OTP_VERIFIED:
            response = await handle_account_selection(sender_id, user_message)
        
        elif verification_stage == VerificationStages.ACCOUNT_SELECTED:
            response = await handle_banking_queries(sender_id, user_message)
        
        elif verification_stage == VerificationStages.TRANSFER_OTP_PENDING:
            response = await handle_transfer_otp_verification(sender_id, user_message)
        
        elif verification_stage == VerificationStages.TRANSFER_CONFIRMATION_PENDING:
            response = await handle_transfer_confirmation(sender_id, user_message)
        
        else:
            response = await ai_agent.handle_session_start()
        
        # Cache the successful response
        if response:
            user_request_cache[cache_key] = (current_time, response)
            
            # Clean old cache entries (keep only last 100)
            if len(user_request_cache) > 100:
                old_keys = [k for k, (t, _) in user_request_cache.items() if current_time - t > 300]  # 5 minutes
                for k in old_keys:
                    user_request_cache.pop(k, None)
                logger.info(f"🧹 Cleaned {len(old_keys)} old cache entries")
        
        return response
        
    except Exception as e:
        # Don't cache errors
        logger.error(f"Processing error: {e}")
        return "Sorry, there was an error processing your request."

async def handle_cnic_verification(sender_id: str, user_message: str) -> str:
    """Handle CNIC verification with flexible input format and non-banking query protection."""
    
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
                json={DatabaseFields.CNIC: extracted_cnic}
            )
            result = response.json()
        
        if result["status"] == StatusMessages.SUCCESS:
            user_data = result["user"]
            
            # Store CNIC verification data
            set_user_verification_stage(
                sender_id, 
                VerificationStages.CNIC_VERIFIED,
                cnic=user_data[DatabaseFields.CNIC],
                name=user_data[DatabaseFields.NAME],
                accounts=user_data["accounts"]
            )
            
            logger.info({
                "action": "cnic_verified_successfully_flexible",
                "sender_id": sender_id,
                "extracted_cnic": extracted_cnic,
                "original_message": user_message_clean,
                "name": user_data[DatabaseFields.NAME],
                "accounts_count": len(user_data["accounts"]),
                "next_step": "otp_verification"
            })
            
            return await ai_agent.handle_otp_request(user_data[DatabaseFields.NAME].split()[0])
        
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
    first_name = user_data.get(DatabaseFields.NAME, "").split()[0]
    
    # CRITICAL: Check for non-banking queries BEFORE OTP processing
    if ai_agent.is_clearly_non_banking_query(user_message.strip(), ""):
        logger.info({
            "action": "non_banking_query_blocked_during_otp_verification",
            "sender_id": sender_id,
            "blocked_query": user_message.strip()
        })
        return await ai_agent.handle_non_banking_query(user_message.strip(), first_name)
    
    if is_valid_otp(user_message.strip()):
        # Get detailed account information for smart selection
        accounts = user_data.get("accounts", [])
        account_details = await get_account_details_from_backend(accounts)
        
        # Store account details for smart selection
        set_user_accounts_with_details(sender_id, account_details)
        
        set_user_verification_stage(
            sender_id,
            VerificationStages.OTP_VERIFIED,
            cnic=user_data[DatabaseFields.CNIC],
            name=user_data[DatabaseFields.NAME],
            accounts=user_data["accounts"]
        )
        
        logger.info({
            "action": "otp_verified_successfully",
            "sender_id": sender_id,
            "otp_entered": user_message.strip(),
            "account_details_loaded": len(account_details),
            "next_step": "smart_account_selection"
        })
        
        return await ai_agent.handle_otp_success(user_data[DatabaseFields.NAME], user_data["accounts"])
    
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
    first_name = user_data.get(DatabaseFields.NAME, "").split()[0]
    
    # Get detailed account information
    account_details = get_user_accounts_with_details(sender_id)
    
    logger.info(f"Account selection debug - User input: '{user_message}'")
    logger.info(f"Available accounts: {[acc[DatabaseFields.ACCOUNT_NUMBER] for acc in account_details]}")
    

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
                        DatabaseFields.CNIC: user_data[DatabaseFields.CNIC],
                        DatabaseFields.ACCOUNT_NUMBER: selected_account
                    }
                )
                result = response.json()
            
            if result["status"] == StatusMessages.SUCCESS:
                # Update to final verification stage
                set_user_verification_stage(
                    sender_id,
                    VerificationStages.ACCOUNT_SELECTED,
                    cnic=user_data[DatabaseFields.CNIC],
                    name=user_data[DatabaseFields.NAME],
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
                
                return await ai_agent.handle_account_confirmation(selected_account, user_data[DatabaseFields.NAME])
            
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
            account_list = [f"{i+1}. {acc[DatabaseFields.ACCOUNT_NUMBER]} ({acc['currency']})" for i, acc in enumerate(account_details)]
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
    account_number = user_info[DatabaseFields.ACCOUNT_NUMBER]
    first_name = user_info[DatabaseFields.NAME].split()[0]
    
    # EARLY NON-BANKING FILTER - Check before processing
    try:
        if ai_agent.is_clearly_non_banking_query(user_message.strip(), ""):
            logger.info({
                "action": "non_banking_query_blocked_at_webhook",
                "sender_id": sender_id,
                "blocked_query": user_message.strip()
            })
            return await ai_agent.handle_non_banking_query(user_message.strip(), first_name)
    except Exception as filter_error:
        logger.error(f"Non-banking filter error: {filter_error}")
        # Continue processing if filter fails
    
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
        if response.startswith(f"{TransferSignals.OTP_REQUIRED_PREFIX}{TransferSignals.SEPARATOR}"):
            # Parse transfer details from response
            parts = response.split(TransferSignals.SEPARATOR)
            if len(parts) == 4:
                amount = float(parts[1])
                currency = parts[2]
                recipient = parts[3]
                
                # Store transfer details and set OTP pending
                set_pending_transfer_info(sender_id, amount, currency, recipient)
                set_user_verification_stage(
                    sender_id,
                    VerificationStages.TRANSFER_OTP_PENDING,
                    cnic=user_info[DatabaseFields.CNIC],
                    name=user_info[DatabaseFields.NAME],
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
        
        user_name = user_data.get(DatabaseFields.NAME, "")
        first_name = user_name.split()[0] if user_name else "there"
        account_number = user_data.get("selected_account", "")
        cnic = user_data.get(DatabaseFields.CNIC, "")

        
        if not account_number:
            logger.error({
                "action": "transfer_otp_no_account",
                "sender_id": sender_id
            })
            return "Account information missing. Please restart your session."
        
        # Check if OTP is valid
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
                VerificationStages.TRANSFER_CONFIRMATION_PENDING,
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
        user_name = user_data.get(DatabaseFields.NAME, "")
        first_name = user_name.split()[0] if user_name else "there"
        account_number = user_data.get("selected_account", "")
        cnic = user_data.get(DatabaseFields.CNIC, "")
        
        user_message_lower = user_message.lower().strip()
        if not any(word in user_message_lower for word in ["yes", "no", "confirm", "cancel", "ok", "sure", "proceed"]):
            if ai_agent.is_clearly_non_banking_query(user_message.strip(), ""):
                logger.info({
                    "action": "non_banking_query_blocked_during_transfer_confirmation",
                    "sender_id": sender_id,
                    "blocked_query": user_message.strip()
                })
                return await ai_agent.handle_non_banking_query(user_message.strip(), first_name)
        
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
                VerificationStages.ACCOUNT_SELECTED,
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
                VerificationStages.ACCOUNT_SELECTED,
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
        
        async with httpx.AsyncClient(timeout=LLMConfig.TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{BACKEND_URL}/process_query",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"] == StatusMessages.SUCCESS:
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
    url = f"https://graph.facebook.com/{WebhookConfig.FACEBOOK_API_VERSION}/me/messages?access_token={PAGE_ACCESS_TOKEN}"
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
        "status": StatusMessages.HEALTHY,
        "backend_connection": StatusMessages.HEALTHY if backend_healthy else "unhealthy",
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