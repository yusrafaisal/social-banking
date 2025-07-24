# Enhanced state.py with CNIC-based authentication and account selection
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Authentication states
authenticated_users = {}  # sender_id -> {cnic, name, verification_stage}
user_sessions = {} 
processed_messages = set()

# New CNIC-based authentication stages
VERIFICATION_STAGES = {
    "NOT_VERIFIED": "not_verified",
    "CNIC_VERIFIED": "cnic_verified", 
    "ACCOUNT_SELECTED": "account_selected"
}

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

# Periodic cleanup (you might want to call this periodically in your webhook)
def periodic_cleanup():
    """Run periodic cleanup of old data."""
    cleanup_old_processed_messages()
    cleanup_old_sessions()
    cleanup_old_authenticated_users()
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
    """Check if user is fully authenticated (CNIC verified AND account selected)."""
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

# Note: Individual conversation memory cleanup is now handled by LangChain
# ConversationBufferMemory will automatically manage conversation history
# Memory persists only for the session duration of the BankingAIAgent instance


def clear_user_state(sender_id: str):
    """Clear all state data for a user (for logout/session termination)."""
    if sender_id in authenticated_users:
        del authenticated_users[sender_id]
        print(f"Cleared state for user {sender_id}")