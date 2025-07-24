# Enhanced state.py with OTP support and transfer confirmation
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Authentication states
authenticated_users = {}  # sender_id -> {cnic, name, verification_stage}
user_sessions = {} 
processed_messages = set()

# Enhanced CNIC-based authentication stages with OTP
VERIFICATION_STAGES = {
    "NOT_VERIFIED": "not_verified",
    "CNIC_VERIFIED": "cnic_verified",
    "OTP_REQUIRED": "otp_required",  # NEW: OTP verification stage
    "ACCOUNT_SELECTED": "account_selected"
}

# Transfer confirmation states
TRANSFER_STATES = {
    "COLLECTING_DETAILS": "collecting_details",
    "OTP_REQUIRED": "transfer_otp_required",  # NEW: OTP for transfer
    "AWAITING_CONFIRMATION": "awaiting_confirmation",  # NEW: Confirmation step
    "PROCESSING": "processing"
}

# Store pending transfers for confirmation flow
pending_transfers = {}  # sender_id -> {transfer_details, stage}

def cleanup_old_processed_messages():
    """Clean up old processed messages to prevent memory leaks."""
    # Keep only last 1000 message IDs
    if len(processed_messages) > 1000:
        # Convert to list, keep last 1000, convert back to set
        recent_messages = list(processed_messages)[-1000:]
        processed_messages.clear()
        processed_messages.update(recent_messages)
        logger.info("Cleaned up old processed messages")

def cleanup_old_sessions():
    """Clean up old user sessions to prevent memory leaks."""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session_data in user_sessions.items():
        if session_data.get('timestamp') and (current_time - session_data['timestamp']).seconds > 7200:  # 2 hours
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del user_sessions[session_id]
        logger.info(f"Cleaned up old session: {session_id}")

def cleanup_old_authenticated_users():
    """Clean up old authenticated users to prevent memory leaks."""
    current_time = datetime.now()
    users_to_remove = []
    
    for sender_id, user_data in authenticated_users.items():
        if user_data.get('timestamp') and (current_time - user_data['timestamp']).seconds > 3600:  # 1 hour
            users_to_remove.append(sender_id)
    
    for sender_id in users_to_remove:
        del authenticated_users[sender_id]
        logger.info(f"Cleaned up old authenticated user: {sender_id}")

def cleanup_old_pending_transfers():
    """Clean up old pending transfers to prevent memory leaks."""
    current_time = datetime.now()
    transfers_to_remove = []
    
    for sender_id, transfer_data in pending_transfers.items():
        if transfer_data.get('timestamp') and (current_time - transfer_data['timestamp']).seconds > 1800:  # 30 minutes
            transfers_to_remove.append(sender_id)
    
    for sender_id in transfers_to_remove:
        del pending_transfers[sender_id]
        logger.info(f"Cleaned up old pending transfer: {sender_id}")

# Periodic cleanup (you might want to call this periodically in your webhook)
def periodic_cleanup():
    """Run periodic cleanup of old data."""
    cleanup_old_processed_messages()
    cleanup_old_sessions()
    cleanup_old_authenticated_users()
    cleanup_old_pending_transfers()  # NEW: Clean pending transfers
    logger.info("Completed periodic cleanup")

# Helper functions for CNIC authentication
def get_user_verification_stage(sender_id: str) -> str:
    """Get current verification stage for user."""
    if sender_id not in authenticated_users:
        return VERIFICATION_STAGES["NOT_VERIFIED"]
    return authenticated_users[sender_id].get("verification_stage", VERIFICATION_STAGES["NOT_VERIFIED"])

def set_user_verification_stage(sender_id: str, stage: str, **additional_data):
    """Set verification stage for user with additional data."""
    if sender_id not in authenticated_users:
        authenticated_users[sender_id] = {}
    
    authenticated_users[sender_id].update({
        "verification_stage": stage,
        "timestamp": datetime.now(),
        **additional_data
    })
    
    logger.info({
        "action": "set_verification_stage",
        "sender_id": sender_id,
        "stage": stage,
        "additional_data": additional_data
    })

def is_fully_authenticated(sender_id: str) -> bool:
    """Check if user is fully authenticated (CNIC verified AND OTP verified AND account selected)."""
    return get_user_verification_stage(sender_id) == VERIFICATION_STAGES["ACCOUNT_SELECTED"]

def get_user_account_info(sender_id: str) -> Dict[str, Any]:
    """Get user's selected account information."""
    if sender_id in authenticated_users and is_fully_authenticated(sender_id):
        user_data = authenticated_users[sender_id]
        return {
            "cnic": user_data.get("cnic"),
            "name": user_data.get("name"),
            "account_number": user_data.get("selected_account"),
            "verification_stage": user_data.get("verification_stage")
        }
    return {}

# NEW: Transfer management functions
def set_pending_transfer(sender_id: str, transfer_details: Dict[str, Any], stage: str):
    """Store pending transfer details for confirmation flow."""
    pending_transfers[sender_id] = {
        "transfer_details": transfer_details,
        "stage": stage,
        "timestamp": datetime.now()
    }
    
    logger.info({
        "action": "set_pending_transfer",
        "sender_id": sender_id,
        "stage": stage,
        "amount": transfer_details.get("amount"),
        "recipient": transfer_details.get("recipient")
    })

def get_pending_transfer(sender_id: str) -> Dict[str, Any]:
    """Get pending transfer details."""
    return pending_transfers.get(sender_id, {})

def clear_pending_transfer(sender_id: str):
    """Clear pending transfer after completion or cancellation."""
    if sender_id in pending_transfers:
        del pending_transfers[sender_id]
        logger.info(f"Cleared pending transfer for user {sender_id}")

def is_in_transfer_flow(sender_id: str) -> bool:
    """Check if user is currently in transfer confirmation flow."""
    return sender_id in pending_transfers

def get_transfer_stage(sender_id: str) -> str:
    """Get current transfer stage."""
    transfer_data = pending_transfers.get(sender_id, {})
    return transfer_data.get("stage", "")

# NEW: OTP validation function
def is_valid_otp(otp_input: str) -> bool:
    """Validate OTP input - accept any 1-5 digit number."""
    return otp_input.isdigit() and 1 <= len(otp_input) <= 5

# NEW: Smart account selection helper
def find_account_by_currency(accounts: list, currency: str) -> str:
    """Find account by currency preference (requires backend query)."""
    # This will be implemented in webhook.py with actual currency lookup
    return None

def find_account_by_position(accounts: list, position: int) -> str:
    """Find account by position (1st, 2nd, etc.)."""
    if 1 <= position <= len(accounts):
        return accounts[position - 1]
    return None

def find_account_by_last_digits(accounts: list, digits: str) -> str:
    """Find account by last 4 digits."""
    for account in accounts:
        if account.endswith(digits):
            return account
    return None

# Session restart functionality
def restart_user_session(sender_id: str):
    """Restart user session (equivalent to fresh page load)."""
    clear_user_state(sender_id)
    logger.info(f"Restarted session for user {sender_id}")

def clear_user_state(sender_id: str):
    """Clear all state data for a user (for logout/session termination/restart)."""
    if sender_id in authenticated_users:
        del authenticated_users[sender_id]
    
    if sender_id in pending_transfers:
        del pending_transfers[sender_id]
    
    logger.info(f"Cleared complete state for user {sender_id}")

# NEW: Banking context validation
def is_banking_related_query(user_message: str) -> bool:
    """Check if query is banking-related to enforce domain boundaries."""
    banking_keywords = [
        "balance", "transaction", "transfer", "money", "account", "spend", "spent", "spending",
        "payment", "deposit", "withdraw", "history", "statement", "save", "saving", "savings",
        "debit", "credit", "currency", "amount", "pkr", "usd", "dollar", "rupee", "bank",
        "financial", "finance", "cash", "fund", "funds", "expense", "expenses", "income",
        "budget", "cost", "costs", "paid", "pay", "purchase", "purchased", "buying", "bought",
        "investment", "loan", "interest", "fee", "charges", "tax", "atm", "card", "otp"
    ]
    
    # Clear non-banking indicators that should be declined
    non_banking_indicators = [
        "president", "weather", "recipe", "movie", "movies", "sports", "news", "politics",
        "celebrity", "celebrities", "music", "song", "songs", "game", "games", "food",
        "restaurant", "travel", "vacation", "holiday", "birthday", "wedding", "school",
        "college", "university", "job", "work", "office", "meeting", "appointment",
        "doctor", "hospital", "medicine", "health", "fitness", "exercise", "diet",
        "shopping", "clothes", "fashion", "technology", "phone", "computer", "internet",
        "social media", "facebook", "instagram", "twitter", "youtube", "netflix",
        "what is", "who is", "define", "meaning", "explain", "tell me about",
        "how to cook", "how to make", "recipe for", "directions to", "weather in",
        "time in", "population of", "capital of", "largest", "smallest", "tallest"
    ]
    
    message_lower = user_message.lower().strip()
    
    # If clearly non-banking, reject
    if any(indicator in message_lower for indicator in non_banking_indicators):
        return False
    
    # If has banking keywords, accept
    if any(keyword in message_lower for keyword in banking_keywords):
        return True
    
    # For ambiguous queries, be conservative and assume banking if unclear
    # Common banking phrases that might not have explicit keywords
    banking_phrases = [
        "how much", "can i afford", "do i have enough", "what can i do",
        "show me", "give me", "i want to", "i need to", "help me with",
        "last month", "this month", "recent", "latest", "current", "right now"
    ]
    
    if any(phrase in message_lower for phrase in banking_phrases):
        return True
    
    # Default to non-banking for safety
    return False

# Note: Individual conversation memory cleanup is now handled by LangChain
# ConversationBufferMemory will automatically manage conversation history
# Memory persists only for the session duration of the BankingAIAgent instance