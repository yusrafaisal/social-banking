# state.py - Updated to use constants
import time
import re
from typing import Dict, Set, Optional
import logging
from constants import VerificationStages, RegexPatterns, Limits

logger = logging.getLogger(__name__)

# Global state storage
authenticated_users: Dict[str, Dict] = {}
processed_messages: Set[str] = set()
user_languages = {}
user_last_languages = {}

def set_user_language(sender_id: str, language: str):
    """Set the current language for user's message."""
    user_languages[sender_id] = language
    user_last_languages[sender_id] = language
    logger.info(f"Set language for user {sender_id}: {language}")

def get_user_language(sender_id: str) -> str:
    """Get the current language for user."""
    return user_languages.get(sender_id, 'en')

def get_user_last_language(sender_id: str) -> str:
    """Get the last detected language for user (useful for number-only messages)."""
    return user_last_languages.get(sender_id, 'en')

def clear_user_language(sender_id: str):
    """Clear language data for user."""
    if sender_id in user_languages:
        del user_languages[sender_id]
    if sender_id in user_last_languages:
        del user_last_languages[sender_id]

def cleanup_old_user_languages():
    """Clean up old user language data."""
    active_users = set(authenticated_users.keys())
    languages_to_remove = []
    last_languages_to_remove = []
    
    for sender_id in user_languages.keys():
        if sender_id not in active_users:
            languages_to_remove.append(sender_id)
    
    for sender_id in user_last_languages.keys():
        if sender_id not in active_users:
            last_languages_to_remove.append(sender_id)
    
    for sender_id in languages_to_remove:
        del user_languages[sender_id]
    
    for sender_id in last_languages_to_remove:
        del user_last_languages[sender_id]
    
    if languages_to_remove or last_languages_to_remove:
        logger.info(f"Cleaned up language data for {len(languages_to_remove)} users")

def get_user_verification_stage(sender_id: str) -> str:
    """Get current verification stage for user."""
    user_data = authenticated_users.get(sender_id, {})
    return user_data.get("verification_stage", VerificationStages.NOT_VERIFIED)

def set_user_verification_stage(sender_id: str, stage: str, **kwargs) -> None:
    """Set verification stage and additional data for user."""
    if sender_id not in authenticated_users:
        authenticated_users[sender_id] = {}
    
    authenticated_users[sender_id]["verification_stage"] = stage
    authenticated_users[sender_id]["last_activity"] = time.time()
    
    # Store additional data
    for key, value in kwargs.items():
        authenticated_users[sender_id][key] = value

def is_valid_otp(otp: str) -> bool:
    """Check if OTP is valid (any 1-5 digit number)."""
    otp = otp.strip()
    return bool(re.match(RegexPatterns.OTP_PATTERN, otp))

def extract_cnic_from_text(text: str) -> Optional[str]:
    """Extract CNIC from natural language text."""
    match = re.search(RegexPatterns.CNIC_PATTERN, text)
    if match:
        return match.group(1)
    return None

def is_fully_authenticated(sender_id: str) -> bool:
    """Check if user is fully authenticated (includes OTP verification)."""
    return get_user_verification_stage(sender_id) == VerificationStages.ACCOUNT_SELECTED

def is_otp_pending(sender_id: str) -> bool:
    """Check if user is waiting for OTP verification."""
    return get_user_verification_stage(sender_id) == VerificationStages.OTP_PENDING

def is_transfer_otp_pending(sender_id: str) -> bool:
    """Check if user is waiting for transfer OTP verification."""
    return get_user_verification_stage(sender_id) == VerificationStages.TRANSFER_OTP_PENDING

def is_transfer_confirmation_pending(sender_id: str) -> bool:
    """Check if user is waiting for transfer confirmation."""
    return get_user_verification_stage(sender_id) == VerificationStages.TRANSFER_CONFIRMATION_PENDING

def get_user_account_info(sender_id: str) -> Dict:
    """Get user account information."""
    user_data = authenticated_users.get(sender_id, {})
    
    if (get_user_verification_stage(sender_id) in [
        VerificationStages.ACCOUNT_SELECTED, 
        VerificationStages.TRANSFER_OTP_PENDING,
        VerificationStages.TRANSFER_CONFIRMATION_PENDING
    ]):
        return {
            "name": user_data.get("name", ""),
            "cnic": user_data.get("cnic", ""),
            "account_number": user_data.get("selected_account", ""),
            "verification_stage": user_data.get("verification_stage", "")
        }
    
    return {}

def get_user_accounts_with_details(sender_id: str) -> list:
    """Get user accounts with detailed information for smart selection."""
    user_data = authenticated_users.get(sender_id, {})
    return user_data.get("accounts_with_details", [])

def set_user_accounts_with_details(sender_id: str, accounts_details: list) -> None:
    """Store detailed account information for smart selection."""
    if sender_id in authenticated_users:
        authenticated_users[sender_id]["accounts_with_details"] = accounts_details

def get_pending_transfer_info(sender_id: str) -> Dict:
    """Get pending transfer information."""
    user_data = authenticated_users.get(sender_id, {})
    return {
        "amount": user_data.get("pending_transfer_amount"),
        "currency": user_data.get("pending_transfer_currency"),
        "recipient": user_data.get("pending_transfer_recipient"),
        "account_number": user_data.get("selected_account", "")
    }

def set_pending_transfer_info(sender_id: str, amount: float, currency: str, recipient: str) -> None:
    """Store pending transfer information."""
    if sender_id in authenticated_users:
        authenticated_users[sender_id]["pending_transfer_amount"] = amount
        authenticated_users[sender_id]["pending_transfer_currency"] = currency
        authenticated_users[sender_id]["pending_transfer_recipient"] = recipient

def clear_pending_transfer_info(sender_id: str) -> None:
    """Clear pending transfer information."""
    if sender_id in authenticated_users:
        keys_to_remove = ["pending_transfer_amount", "pending_transfer_currency", "pending_transfer_recipient"]
        for key in keys_to_remove:
            authenticated_users[sender_id].pop(key, None)

def clear_user_state(sender_id: str) -> None:
    """Clear all state for a user (logout/exit)."""
    if sender_id in authenticated_users:
        del authenticated_users[sender_id]
    
    clear_user_language(sender_id)

def should_translate_for_user(sender_id: str) -> bool:
    """Check if user needs translation based on their language preference."""
    user_language = get_user_language(sender_id)
    return user_language != 'en'

def get_user_preferred_language(sender_id: str) -> str:
    """Get user's preferred language for responses."""
    return get_user_language(sender_id)

def periodic_cleanup() -> None:
    """Clean up old sessions and processed messages."""
    current_time = time.time()
    
    # Clean up old user sessions (older than 1 hour)
    expired_users = []
    for sender_id, user_data in authenticated_users.items():
        last_activity = user_data.get("last_activity", 0)
        if current_time - last_activity > Limits.SESSION_TIMEOUT_SECONDS:
            expired_users.append(sender_id)
    
    for sender_id in expired_users:
        del authenticated_users[sender_id]
    
    # Clean up old processed messages
    if len(processed_messages) > Limits.MAX_PROCESSED_MESSAGES:
        oldest_messages = sorted(processed_messages)[:len(processed_messages) - Limits.MAX_PROCESSED_MESSAGES]
        processed_messages.difference_update(oldest_messages)
    
    cleanup_old_user_languages()
    
    if expired_users:
        logger.info(f"Cleaned up {len(expired_users)} expired user sessions")

def get_session_stats() -> Dict:
    """Get current session statistics."""
    return {
        "active_users": len(authenticated_users),
        "processed_messages": len(processed_messages),
        "fully_authenticated_users": sum(1 for user in authenticated_users.values() 
                                       if user.get("verification_stage") == VerificationStages.ACCOUNT_SELECTED),
        "otp_pending_users": sum(1 for user in authenticated_users.values() 
                               if user.get("verification_stage") in [
                                   VerificationStages.OTP_PENDING, 
                                   VerificationStages.TRANSFER_OTP_PENDING
                               ]),
        "transfer_confirmation_pending": sum(1 for user in authenticated_users.values() 
                                           if user.get("verification_stage") == VerificationStages.TRANSFER_CONFIRMATION_PENDING)
    }