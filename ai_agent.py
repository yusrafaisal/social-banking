import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import jsonschema
import re
from pymongo import MongoClient

# Import constants
from constants import (
    BankingIntents, TransactionTypes, Currencies, DatabaseFields, Limits, 
    ResponseFormats, ContextStates, ConfirmationWords, GreetingWords,
    ExitCommands, Months, RegexPatterns, BalanceKeywords, TransactionKeywords,
    LLMConfig, MongoConfig, WebhookConfig, StatusMessages, TransferSignals
)

from prompts import (
    filter_extraction_prompt,
    pipeline_generation_prompt,
    response_prompt,
    query_prompt,
    intent_prompt,
    transfer_prompt
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LangChain LLM
llm = ChatOpenAI(
    model=LLMConfig.MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=LLMConfig.TEMPERATURE
)

# MongoDB pipeline schema for validation
PIPELINE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "$match": {"type": "object"},
            "$group": {"type": "object"},
            "$sort": {"type": "object"},
            "$limit": {"type": "integer", "minimum": 1},
            "$project": {"type": "object"},
            "$facet": {"type": "object"}
        },
        "additionalProperties": False
    }
}

# === DATA MODELS ====
class FilterExtraction(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = None
    month: Optional[str] = None
    year: Optional[int] = None
    transaction_type: Optional[str] = None
    amount_range: Optional[Dict[str, float]] = None
    date_range: Optional[Dict[str, str]] = None
    limit: Optional[int] = None
    currency: Optional[str] = None
    intent_hint: Optional[str] = None

class QueryResult(BaseModel):
    intent: str = Field(default=BankingIntents.GENERAL)
    pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    response_format: str = Field(default="natural_language")
    filters: Optional[FilterExtraction] = None

class ContextualQuery(BaseModel):
    """Result of contextual query analysis."""
    needs_context: bool = False
    has_reference: bool = False
    is_complete: bool = True
    missing_info: List[str] = Field(default_factory=list)
    clarification_needed: Optional[str] = None
    resolved_query: Optional[str] = None

# === HELPER FUNCTIONS ====
def month_to_number(month: str) -> int:
    """Convert month name to number."""
    return Months.NAMES_TO_NUMBERS.get(month.lower(), 1)

def month_days(month: str, year: int) -> int:
    """Get number of days in a month."""
    month_num = month_to_number(month)
    if month_num in Months.THIRTY_DAY_MONTHS:
        return 30
    elif month_num == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    else:
        return Months.DAYS_IN_MONTH.get(month_num, 31)
        
_BRACE_RE = re.compile(RegexPatterns.JSON_BRACE_PATTERN)

def _find_json_span(text: str) -> Tuple[int, int]:
    """Return (start, end) indices of the first JSON value in text."""
    m = _BRACE_RE.search(text)
    if not m:
        raise ValueError("No '{' or '[' found")
    start = m.start()
    stack = [text[start]]
    for i in range(start + 1, len(text)):
        ch = text[i]
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if not stack:
                break
            open_ch = stack.pop()
            if not stack:
                return start, i + 1
    raise ValueError("Unbalanced brackets")

def _json_fix(raw: str) -> str:
    """Best‑effort clean‑ups that keep strict JSON subset."""
    fixed = raw.strip()
    fixed = re.sub(r"'", '"', fixed)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    fixed = fixed.replace('NaN', 'null')
    fixed = fixed.replace('Infinity', '1e308')
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed)
    return fixed


class BankingAIAgent:
    def __init__(self, mongodb_uri: str = MongoConfig.DEFAULT_URI, db_name: str = MongoConfig.DEFAULT_DB_NAME):
        """Initialize the Banking AI Agent with MongoDB connection."""
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db[MongoConfig.TRANSACTIONS_COLLECTION]
        self.backend_url = WebhookConfig.BACKEND_URL
        # Use LangChain memory directly without ConversationChain
        self.user_memories: Dict[str, ConversationBufferMemory] = {}
        self.llm = llm  # Make the global llm accessible as instance attribute

    def extract_json_from_response(self, raw: str) -> Optional[Any]:
        """Extract the first JSON value from an LLM reply."""
        try:
            start, end = _find_json_span(raw)
            candidate = raw[start:end]
        except ValueError as e:
            logger.error({"action": "extract_json_span_fail", "error": str(e)})
            return None

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        candidate = _json_fix(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.error({"action": "extract_json_parse_fail", "error": str(e), "candidate": candidate[:200]})
            return None

        
    # === MEMORY MANAGEMENT METHODS ===
    def get_user_memory(self, account_number: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a user account."""
        if account_number not in self.user_memories:
            self.user_memories[account_number] = ConversationBufferMemory(
                return_messages=True
            )
        return self.user_memories[account_number]

    def clear_user_memory(self, account_number: str) -> None:
        """Clear conversation memory for a specific user account."""
        if account_number in self.user_memories:
            del self.user_memories[account_number]
            logger.info(f"Cleared memory for account: {account_number}")
    
    def _get_context_summary(self, chat_history: List) -> str:
        """Get an enhanced summary of recent conversation with structured data extraction."""
        if not chat_history:
            return "No previous conversation."
        
        # Get more messages for better context (increased from default)
        context_window = min(len(chat_history), Limits.MAX_RECENT_MESSAGES * 2)  # Double the window
        recent_messages = chat_history[-context_window:]
        
        # Separate user and assistant messages for better analysis
        user_messages = []
        assistant_messages = []
        conversation_flow = []
        
        for i, msg in enumerate(recent_messages):
            speaker = "Human" if i % 2 == 0 else "Assistant"
            content = msg.content
            
            # Preserve important banking data in full
            if len(content) > Limits.MAX_RESPONSE_CONTENT:
                # Don't truncate if it contains important banking keywords
                banking_keywords = ["balance", "transaction", "transfer", "$", "PKR", "USD", "spent", "amount"]
                if any(keyword in content.lower() for keyword in banking_keywords):
                    # Keep full content for banking data, but limit extremely long responses
                    content = content[:Limits.MAX_RESPONSE_CONTENT * 2] + "..." if len(content) > Limits.MAX_RESPONSE_CONTENT * 2 else content
                else:
                    content = content[:Limits.MAX_RESPONSE_CONTENT] + "..."
            
            conversation_flow.append(f"{speaker}: {content}")
            
            if speaker == "Human":
                user_messages.append(content)
            else:
                assistant_messages.append(content)
        
        full_context = "\n".join(conversation_flow)
        
        # Enhanced summarization that preserves banking context
        if len(full_context) > Limits.MAX_CONTEXT_LENGTH:
            try:
                enhanced_summary_prompt = f"""
                Create a structured summary of this banking conversation that preserves ALL important context for future queries.

                CONVERSATION:
                {full_context}

                Create a summary with these sections:
                1. RECENT BALANCE INFO: Any balance amounts, account numbers mentioned
                2. TRANSACTION DATA: Recent transactions shown (amounts, descriptions, dates)
                3. SPENDING ANALYSIS: Any spending totals, categories, comparisons discussed
                4. TRANSFER CONTEXT: Any transfer amounts, recipients, percentages mentioned
                5. USER PATTERNS: What the user typically asks about or references
                6. PENDING ACTIONS: Any incomplete requests or multi-step processes

                Focus on preserving:
                - Exact amounts, percentages, and calculations
                - Specific transaction details and timeframes
                - Any "that amount", "those transactions" references
                - Transfer recipients and amounts
                - Categories and spending breakdowns

                Keep it detailed enough for contextual query resolution.
                """
                response = llm.invoke([SystemMessage(content=enhanced_summary_prompt)])
                return response.content.strip()
            except Exception as e:
                logger.error(f"Error in enhanced summarization: {e}")
                return full_context[:Limits.MAX_CONTEXT_LENGTH] + "..."
        
        return full_context
    

    def resolve_contextual_query(self, user_message: str, conversation_history: str) -> str:
        """Enhanced contextual query resolution with multi-turn conversation analysis."""
        try:
            if not conversation_history or len(conversation_history.strip()) < 10:
                return user_message  # No context to work with
            
            # Enhanced prompt with clear structure and examples
            enhanced_contextual_prompt = f"""
            You are a banking conversation resolver. Your job is to analyze the conversation history and resolve contextual references in the current user message.

            CONVERSATION HISTORY:
            {conversation_history}

            CURRENT USER MESSAGE: "{user_message}"

            STEP 1: IDENTIFY THE CONVERSATION TYPE

            Check what type of conversation this is:

            A) TRANSFER CLARIFICATION CONVERSATION
            - Look for assistant messages like "I found multiple [vendor] transactions. Which one?"
            - Look for assistant messages listing transaction options
            - Look for assistant messages asking "Which transaction would you like?"

            B) TRANSFER CONFIRMATION CONVERSATION  
            - Look for assistant messages asking "Do you want to confirm the transfer?"
            - Look for assistant messages requesting OTP
            - Look for assistant messages about transfer details

            C) AMBIGUOUS TRANSFER REQUEST
            - User mentions a vendor/merchant but conversation history shows multiple transactions
            - Example: History shows 3 McDonald transactions, user says "transfer 10% of the mcdonald transaction"

            D) NORMAL CONTEXTUAL CONVERSATION
            - Regular banking conversation with contextual references
            - User referring to "that amount", "those transactions", etc.

            STEP 2: APPLY RESOLUTION RULES

            **RULE A: TRANSFER CLARIFICATION RESOLUTION**
            If Type A (Transfer Clarification) AND current message is a selection:

            Selection Patterns:
            - "17.25 one" = User selecting the $17.25 transaction
            - "9.37 one" = User selecting the $9.37 transaction  
            - "first one" = User selecting the first transaction
            - "second one" = User selecting the second transaction
            - "latest one" = User selecting the most recent transaction
            - "May 14 one" = User selecting the May 14 transaction
            - "the $24.10 transaction" = User selecting specific amount

            Resolution Process:
            1. Find the original transfer request in conversation history
            2. Extract: percentage/amount, recipient, and original context
            3. Find the selected transaction details from the clarification list
            4. Combine into complete transfer request

            Example:
            History: "transfer 10% of the mcdonald transaction to Hamza"
            History: "I found multiple McDonald transactions: 
                1. June 22: $17.25 USD 
                2. June 15: $20.89 USD"
            which one are you referring to?

            Current: "17.25 one"
            → RESOLVE TO: "transfer 10% of the June 22 $17.25 USD McDonald transaction to Hamza"

            **RULE B: TRANSFER CONFIRMATION RESOLUTION**
            If Type B (Transfer Confirmation) AND current message is confirmation:

            Confirmation Patterns:
            - "yes" = User confirming transfer
            - "confirm" = User confirming transfer
            - "proceed" = User confirming transfer
            - "ok" = User confirming transfer

            Resolution Process:
            1. Find the transfer details being confirmed
            2. Resolve to confirmation format

            Example:
            History: "Do you want to confirm transfer of $15.54 USD to Ahmed?"
            Current: "yes"
            → RESOLVE TO: "yes confirm transfer $15.54 USD to Ahmed"

            **RULE C: AMBIGUOUS TRANSFER BLOCKING**
            If Type C (Ambiguous Transfer) - DO NOT RESOLVE, KEEP UNCHANGED

            When to Block:
            - Multiple transactions from same vendor AND user refers ambiguously
            - Example: 3 McDonald transactions exist, user says "the mcdonald transaction"

            Block Examples:
            - "transfer 10% of the mcdonald transaction" (when multiple McDonalds exist)
            - "send 5% of the netflix payment" (when multiple Netflix exist)

            **RULE D: NORMAL CONTEXTUAL RESOLUTION**
            If Type D (Normal Contextual):

            Normal Resolution Patterns:
            - "that balance" → find most recent balance mentioned
            - "1% of that" → calculate percentage of most recent amount
            - "those transactions" → find transaction list from conversation
            - "from that account" → identify account number in context

            STEP 3: EXECUTE RESOLUTION

            Based on the type identified above:

            1. If Type A + Selection Pattern: Create complete transfer request
            2. If Type B + Confirmation Pattern: Create confirmation statement  
            3. If Type C + Ambiguous Reference: Return query UNCHANGED
            4. If Type D: Apply normal contextual resolution

            STEP 4: FORMAT OUTPUT

            Return ONLY the resolved query - no explanations, no reasoning, just the final standalone query.

            CRITICAL EXAMPLES:

            Example 1 (Transfer Clarification):
            History: "transfer 15 percent of foodpanda transaction to Sophiya"
            History: "I found multiple Foodpanda transactions. Which one? 
            1. May 20: $10.49 USD 
            2. May 4: $9.37 USD"
            Current: "9.37 one"
            Output: "transfer 15 percent of the May 4 $9.37 USD Foodpanda transaction to Sophiya"

            Example 2 (Transfer Confirmation):
            History: "Do you want to confirm the transfer of $1.41 USD to Sophiya?"
            Current: "yes"
            Output: "yes confirm transfer $1.41 USD to Sophiya"

            Example 3 (Ambiguous - Keep Unchanged):
            History: "McDonald $17.25, McDonald $20.89, McDonald $12.65"
            Current: "transfer 10% of the mcdonald transaction to Ahmed"
            Output: "transfer 10% of the mcdonald transaction to Ahmed"

            Example 4 (Normal Contextual):
            History: "Your balance is $1,554.41 USD"
            Current: "transfer 1% of that to John"
            Output: "transfer 1% of $1,554.41 USD ($15.54) to John"

            RESOLVED QUERY:
            """
            
            response = llm.invoke([SystemMessage(content=enhanced_contextual_prompt)])
            resolved_query = response.content.strip()
            
            # Remove any quotes or extra formatting
            if resolved_query.startswith('"') and resolved_query.endswith('"'):
                resolved_query = resolved_query[1:-1]
            
            # Remove any explanation text that might have been added
            if "Output:" in resolved_query:
                resolved_query = resolved_query.split("Output:")[-1].strip()
            
            logger.info(f"Enhanced contextual query resolved: '{user_message}' → '{resolved_query}'")
            return resolved_query
            
        except Exception as e:
            logger.error(f"Error in enhanced contextual query resolution: {e}")
            return user_message

    def _extract_banking_entities_from_history(self, conversation_history: str) -> Dict[str, Any]:
        """Extract banking entities and context from conversation history for better resolution."""
        try:
            entity_extraction_prompt = f"""
            Extract key banking entities and context from this conversation history for contextual query resolution.

            CONVERSATION HISTORY:
            {conversation_history}

            Extract and return JSON with:
            {{
                "balances": [
                    {{"amount": 1500, "currency": "USD", "date": "recent", "context": "current balance"}}
                ],
                "transactions": [
                    {{"description": "Netflix", "amount": 15, "type": "debit", "mentioned_when": "recent"}}
                ],
                "amounts_mentioned": [
                    {{"value": 1000, "currency": "USD", "context": "affordability question"}}
                ],
                "recipients": ["john", "ali raza"],
                "categories": [
                    {{"name": "food", "amount": 85, "context": "spending breakdown"}}
                ],
                "timeframes": ["june", "last month", "recent"],
                "pending_transfers": [
                    {{"amount": 100, "recipient": "john", "status": "needs_confirmation"}}
                ],
                "user_focus": "last thing user was asking about",
                "conversation_flow": "summary of how conversation evolved"
            }}

            Focus on preserving exact amounts, names, and contextual relationships.
            """
            
            response = llm.invoke([SystemMessage(content=entity_extraction_prompt)])
            entities = self.extract_json_from_response(response.content)
            return entities if entities else {}
            
        except Exception as e:
            logger.error(f"Error extracting banking entities: {e}")
            return {}

    def _resolve_with_entity_context(self, user_message: str, entities: Dict[str, Any]) -> str:
        """Resolve query using extracted banking entities."""
        try:
            entity_resolution_prompt = f"""
            Resolve this contextual query using extracted banking entities from conversation history.

            USER QUERY: "{user_message}"
            
            AVAILABLE CONTEXT:
            {json.dumps(entities, indent=2)}

            Resolution rules:
            1. Map references like "that", "those", "it" to specific entities
            2. Calculate percentages using actual amounts from context
            3. Combine incomplete information (amount + recipient, etc.)
            4. Reference specific timeframes, categories, or transactions
            5. Preserve user's intent while making query standalone

            Return the resolved standalone query:
            """
            
            response = llm.invoke([SystemMessage(content=entity_resolution_prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in entity-based resolution: {e}")
            return user_message


    async def detect_exit_intent_with_llm(self, user_message: str) -> bool:
        """Use LLM to detect if user wants to exit/logout, even in natural language."""
        try:
            exit_detection_prompt = f"""
            You are an exit intent detector for a banking application. Analyze if the user wants to exit, logout, end session, or quit.

            User message: "{user_message}"

            Exit Intent Indicators:
            - Direct commands: "exit", "quit", "logout", "end", "bye", "goodbye"
            - Natural phrases: "ok exit", "I want to exit", "please logout", "end session", "log me out"
            - Contextual: "I'm done", "that's all", "finish", "close", "stop"
            - Polite: "thank you, exit", "thanks, bye", "good day, logout"

            NOT Exit Intent:
            - Banking queries: "exit strategy", "quit spending", "end of month"
            - General conversation: "exit the building", "quit my job"
            - Questions: "how do I exit?", "what does exit mean?"

            Rules:
            1. If the message contains clear exit/logout/quit/end intent → return "YES"
            2. If it's a banking query or general conversation → return "NO"
            3. When in doubt, lean towards "NO" for security

            Return ONLY "YES" or "NO".
            """

            response = await self.llm.ainvoke([SystemMessage(content=exit_detection_prompt)])
            result = response.content.strip().upper()
            
            # Log the detection for debugging
            logger.info(f"Exit intent detection: '{user_message}' → {result}")
            
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Error in LLM exit detection: {e}")
            # Safe fallback - only exact matches to prevent false positives
            safe_exit_commands = ["exit", "quit", "logout", "bye", "goodbye"]
            user_lower = user_message.strip().lower()
            
            # Check for exact matches only
            is_exit = user_lower in safe_exit_commands
            
            if is_exit:
                logger.info(f"Fallback exit detection: '{user_message}' matched exact command")
            
            return is_exit
    
    async def detect_cancel_transfer_intent_with_llm(self, user_message: str) -> bool:
        """Use LLM to detect if user wants to cancel the current transfer process."""
        try:
            cancel_detection_prompt = f"""
            You are analyzing if a user wants to cancel their current money transfer process.

            User message: "{user_message}"

            Cancel Transfer Intent Indicators:
            - Direct commands: "cancel", "cancel transfer", "cancel transaction", "stop", "abort"
            - Natural phrases: "leave this", "forget it", "never mind", "don't want to", "changed my mind"
            - Contextual: "go back", "exit transfer", "stop transfer", "cancel this process"
            - Polite: "please cancel", "can you cancel", "I don't want to proceed"

            NOT Cancel Intent:
            - Banking queries: "cancel my card", "cancel subscription" (different context)
            - Asking questions: "how do I cancel?", "what happens if I cancel?"
            - Transfer details: "cancel the Netflix payment" (referring to other transactions)

            Rules:
            1. If the message contains clear intent to stop/cancel the CURRENT transfer process → return "YES"
            2. If it's asking about other cancellations or just questions → return "NO"
            3. When in doubt about transfer cancellation, lean towards "YES" for user safety

            Return ONLY "YES" or "NO".
            """

            response = await self.llm.ainvoke([SystemMessage(content=cancel_detection_prompt)])
            result = response.content.strip().upper()
            
            logger.info(f"Cancel transfer intent detection: '{user_message}' → {result}")
            
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Error in LLM cancel transfer detection: {e}")
            # Safe fallback - check for obvious cancel words
            cancel_words = ["cancel", "stop", "abort", "leave this", "forget it", "never mind"]
            user_lower = user_message.strip().lower()
            
            for word in cancel_words:
                if word in user_lower:
                    logger.info(f"Fallback cancel detection: '{user_message}' matched '{word}'")
                    return True
            
            return False
        
    # === SESSION MANAGEMENT METHODS ===
    async def handle_session_start(self, first_name: str = "", last_name: str = "") -> str:
        """Handle session start with natural greeting."""
        try:
            context_state = ContextStates.SESSION_STARTING
            data = {
                "session_status": "starting",
                "authentication_required": "cnic_verification",
                "security_level": "high",
                "next_step": "cnic_input"
            }
            
            return await self.generate_natural_response(context_state, data, "", first_name or "there")
            
        except Exception as e:
            logger.error(f"Error in session start: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(context_state, {"error": str(e)}, "", first_name or "there")

    async def handle_session_end(self, account_number: str, first_name: str) -> str:
        """Handle session termination with natural response."""
        try:
            # Clear user memory for security
            if account_number in self.user_memories:
                del self.user_memories[account_number]
            
            context_state = ContextStates.SESSION_ENDING
            data = {
                "session_ended": True,
                "security_cleared": True,
                "memory_cleared": True,
                "restart_instructions": True
            }
            
            return await self.generate_natural_response(context_state, data, "exit", first_name)
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(context_state, {"error": str(e)}, "exit", first_name)

    # === INITIAL GREETING HANDLING ===
    async def handle_initial_greeting(self) -> str:
        """Handle initial user greeting and ask for CNIC verification with bold formatting and clear structure."""
        try:
            greeting_prompt = """
            You are Kora, a friendly banking assistant. A user just greeted you (said hi, hello, etc.) to start a new banking session.

            Your task:
            1. Greet them warmly and introduce yourself as their banking assistant
            2. Explain that you can help with their banking needs
            3. Ask them to provide their **CNIC** for secure verification
            4. Mention the **CNIC format** (**12345-1234567-1**)
            5. Keep it friendly and welcoming

            **CRITICAL FORMATTING RULES:**
            - **BOLD** all important words and actions: **CNIC**, **banking assistant**, **secure**, **verification**, **format**, **account**, **help**
            - Use line breaks for clarity and structure
            - Make the next step ("please provide your CNIC") stand out clearly
            - Structure the message so it's easy to read and follow

            Generate a natural, welcoming response that guides them to the next step, using proper bold formatting and clear structure.
            """
            response = await llm.ainvoke([SystemMessage(content=greeting_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating initial greeting: {e}")
            return (
                "Hello! I'm **Kora**, your **banking assistant**. I'm here to help you with all your **banking needs**.\n\n"
                "To get started **securely**, could you please provide your **CNIC** in the format **12345-1234567-1**?"
            )
    def _is_simple_greeting_or_general(self, user_message: str) -> bool:
        """Check if message is a simple greeting or general question."""
        user_lower = user_message.lower().strip()
        
        # Simple greetings
        if any(user_lower.startswith(greeting) for greeting in GreetingWords.BASIC_GREETINGS):
            return True
            
        # General questions
        if any(phrase in user_lower for phrase in GreetingWords.GENERAL_PHRASES):
            return True
            
        return False
    
    # === OTP HANDLING ===
    async def handle_otp_request(self, first_name: str = "") -> str:
        """Handle OTP request after CNIC verification with bold formatting and clear structure."""
        try:
            otp_prompt = f"""You are Kora, a banking assistant. A user has just had their **CNIC** verified successfully and now needs to provide an **OTP** for additional security.

            Your task:
            1. Explain that for additional security, they need to provide an **OTP**
            2. Tell them the **OTP** is a number between **{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits** sent to their **mobile phone**
            3. Ask them to enter their **OTP**
            4. **Bold** all important words and actions: **OTP**, **CNIC**, **security**, **mobile phone**, **digits**, **verification**
            5. Use line breaks for clarity and structure
            6. Make the next step ("please provide your OTP") stand out clearly

            Generate a natural, security-focused response asking for **OTP**, using proper bold formatting and clear structure.
            """
            response = await llm.ainvoke([SystemMessage(content=otp_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating OTP request: {e}")
            return (
                f"Great! For additional **security**, {first_name}, please provide your **OTP**. "
                f"You should have received an **OTP** (**{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits**) on your **mobile phone**."
            )
    async def handle_otp_success(self, user_name: str, accounts: List[str]) -> str:
        """Handle successful OTP verification - should only ask for account selection with bold formatting and clear structure."""
        try:
            first_name = user_name.split()[0]

            # Format accounts for selection with bold last 4 digits
            account_list = []
            for i, account in enumerate(accounts, 1):
                # Mask account except last 4 digits and bold them
                masked = f"***-***-**<b>{account[-4:]}</b>"
                account_list.append(f"{i}. {masked}")

            accounts_text = "\n".join(account_list)

            return (
                f"Hello <b>{first_name}</b>! <b>OTP verified successfully.</b>\n\n"
                f"<b>Please select your account:</b>\n"
                f"{accounts_text}\n\n"
                f"You can select by typing the <b>account number</b>, <b>last 4 digits</b>, or saying <b>'first account'</b>, <b>'second account'</b>, etc."
            )

        except Exception as e:
            logger.error(f"Error in OTP success: {e}")
            return (
                "<b>OTP verified successfully!</b> Now please select your account by entering the <b>last 4 digits</b>."
            )
        
    async def handle_otp_failure(self, user_input: str, first_name: str = "") -> str:
        """Handle failed OTP verification."""
        try:
            context_state = ContextStates.OTP_VERIFICATION_FAILED
            data = {
                "otp_failed": True,
                "user_input": user_input,
                "required_format": f"{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits"
            }
            
            return await self.generate_natural_response(context_state, data, user_input, first_name)
            
        except Exception as e:
            logger.error(f"Error in OTP failure: {e}")
            return f"Sorry {first_name}, that OTP format isn't valid. Please enter a number between {Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits."

    # === CNIC VERIFICATION METHODS ===

    async def handle_cnic_verification_success(self, user_name: str, accounts: List[str], cnic: str) -> str:
        """Handle successful CNIC verification with enhanced formatting."""
        try:
            context_state = ContextStates.CNIC_VERIFICATION_SUCCESS
            data = {
                "verification_status": StatusMessages.SUCCESS,
                "user_name": user_name,
                "accounts_found": len(accounts),
                "accounts": accounts,
                "next_step": "otp_verification",
                "otp_format": f"{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits"
            }
            
            # Enhanced prompt for CNIC success with formatting
            cnic_success_prompt = f"""You are Kora, a banking assistant. The user's **CNIC** has been successfully verified.

            USER: **{user_name}**
            ACCOUNTS FOUND: {len(accounts)}
            NEXT STEP: **OTP verification**

            Your response should:
            1. Congratulate them on successful **CNIC verification**
            2. Explain they need to provide an **OTP** (**{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits**)
            3. **Bold** all important terms like **OTP**, **CNIC**, **verification**
            4. Make it clear what they need to do next
            5. Keep it welcoming and secure

            Generate a well-formatted response with proper **bold highlighting**."""
            
            response = await llm.ainvoke([SystemMessage(content=cnic_success_prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in CNIC verification success: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(context_state, {"error": str(e)}, cnic, user_name.split()[0])



    async def handle_cnic_verification_failure(self, cnic: str, first_name: str = "") -> str:
        """Handle failed CNIC verification with natural response."""
        try:
            context_state = ContextStates.CNIC_VERIFICATION_FAILED
            data = {
                "verification_status": StatusMessages.FAIL,
                "provided_cnic": cnic,
                "required_format": "12345-1234567-1",
                "retry_needed": True
            }
            
            return await self.generate_natural_response(context_state, data, cnic, first_name or "there")
            
        except Exception as e:
            logger.error(f"Error in CNIC verification failure: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(context_state, {"error": str(e)}, cnic, first_name or "there")

    async def handle_invalid_cnic_format(self, user_input: str, first_name: str = "") -> str:
        """Handle invalid CNIC format with natural response."""
        try:
            context_state = ContextStates.INVALID_CNIC_FORMAT
            data = {
                "input_provided": user_input,
                "format_error": True,
                "required_format": "12345-1234567-1",
                "format_rules": {
                    "part1": "5 digits",
                    "separator1": "dash (-)",
                    "part2": "7 digits", 
                    "separator2": "dash (-)",
                    "part3": "1 digit"
                }
            }
            
            return await self.generate_natural_response(context_state, data, user_input, first_name or "there")
            
        except Exception as e:
            logger.error(f"Error in invalid CNIC format handling: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_input, first_name or "there")
        
    # === ACCOUNT SELECTION METHODS ===

    async def handle_account_selection(self, selection: str, accounts: List[str], first_name: str) -> str:
        """Handle account selection with natural response."""
        try:
            context_state = ContextStates.ACCOUNT_SELECTION_UNCLEAR
            data = {
                "available_accounts": accounts,
                "user_input": selection,
                "expected_format": "last_4_digits",
                "selection_method": "last_4_digits"
            }
            
            return await self.generate_natural_response(context_state, data, selection, first_name)
            
        except Exception as e:
            logger.error(f"Error in account selection: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(context_state, {"error": str(e)}, selection, first_name)

        
    async def handle_account_confirmation(self, account_number: str, user_name: str) -> str:
        """Handle account selection confirmation with enhanced formatting."""
        try:
            first_name = user_name.split()[0]
            masked_account = f"***-***-{account_number[-4:]}"
            
            return f"""Hello {first_name}! Account confirmed! 

    Welcome, you're now connected to
    account {masked_account}.

    I can help you with:
    • Check your account balance
    • View transaction history  
    • Analyze your spending patterns
    • Transfer money to others
    • Financial planning assistance

    What can I help you with today?"""
                    
        except Exception as e:
            logger.error(f"Error in account confirmation: {e}")
            return f"Hello! Account confirmed successfully. What can I help you with today?"
            
    # === QUERY PIPELINE FLOW ===
    def detect_intent_from_filters(self, user_message: str, filters: FilterExtraction) -> str:
        """Detect intent using LLM for more flexible understanding."""
        try:
            intent_hint = getattr(filters, 'intent_hint', None)
            
            # Check hint-based detection first
            if intent_hint == "transaction_list":
                return BankingIntents.TRANSACTION_HISTORY
            elif intent_hint == "spending_total":
                return BankingIntents.SPENDING_ANALYSIS
            elif intent_hint == "balance_query":
                return BankingIntents.BALANCE_INQUIRY
            
            # CRITICAL FIX: Add transfer keyword detection BEFORE LLM call
            user_lower = user_message.lower().strip()
            
            # Transfer keywords
            transfer_keywords = [
                "transfer", "send", "pay", "give", "move money", "wire",
                "% of", "percent of", "% from", "percent from",
                " to ", "send to", "transfer to", "pay to"
            ]
            
            # Check for transfer intent based on keywords
            if any(keyword in user_lower for keyword in transfer_keywords):
                logger.info(f"Transfer keywords detected in: '{user_message}'")
                return BankingIntents.TRANSFER_MONEY
            
            # Check for recipient patterns (to [name])
            import re
            if re.search(r'\bto\s+[a-zA-Z]+', user_lower):
                logger.info(f"Recipient pattern detected in: '{user_message}'")
                return BankingIntents.TRANSFER_MONEY
            
            # Check for confirmation words in transfer context
            confirmation_words = ["yes", "yess", "confirm", "proceed", "ok", "sure"]
            if user_lower in confirmation_words:
                # This should be handled by conversation context
                # For now, let LLM decide based on context
                logger.info(f"Confirmation word detected: '{user_message}' - letting LLM analyze context")
            
            # Use LLM for complex cases
            response = llm.invoke([
                SystemMessage(content=intent_prompt.format(
                    user_message=user_message,
                    filters=json.dumps(filters.dict())
                ))
            ])
            
            detected_intent = response.content.strip().lower()
            
            valid_intents = [
                BankingIntents.BALANCE_INQUIRY,
                BankingIntents.TRANSACTION_HISTORY, 
                BankingIntents.SPENDING_ANALYSIS,
                BankingIntents.CATEGORY_SPENDING,
                BankingIntents.TRANSFER_MONEY,
                BankingIntents.GENERAL
            ]
            
            if detected_intent in valid_intents:
                return detected_intent
            else:
                for intent in valid_intents:
                    if intent in detected_intent:
                        return intent
                return BankingIntents.GENERAL
                
        except Exception as e:
            logger.error({
                "action": "llm_intent_classification",
                "error": str(e),
                "user_message": user_message
            })
            return BankingIntents.GENERAL

    def extract_filters_with_llm(self, user_message: str) -> FilterExtraction:
        """Use LLM to extract filters from user query with enhanced date handling."""
        try:
            response = llm.invoke([SystemMessage(content=filter_extraction_prompt.format(
                user_message=user_message,
                current_date=datetime.now().strftime("%Y-%m-%d")
            ))])
            
            try:
                filters_obj = self.extract_json_from_response(response.content)
                if filters_obj is None:
                    raise ValueError("Could not parse filter JSON")
                filters = FilterExtraction(**filters_obj)
                logger.info(f"LLM extracted filters: {filters.dict()}")
                return filters
            
            except (json.JSONDecodeError, TypeError) as e:
                logger.error({
                    "action": "filter_extraction_parse_error",
                    "error": str(e),
                    "raw_response": response.content
                })
                return FilterExtraction()
                
        except Exception as e:
            logger.error({
                "action": "extract_filters_with_llm",
                "error": str(e)
            })
            return FilterExtraction()

    def generate_pipeline_from_filters(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate MongoDB pipeline from extracted filters using LLM."""
        try:
            response = llm.invoke([
                SystemMessage(content=pipeline_generation_prompt.format(
                    filters=json.dumps(filters.dict()),
                    intent=intent,
                    account_number=account_number
                ))
            ])
            
            cleaned_response = self.extract_json_from_response(response.content)
        
            if not cleaned_response:
                return self._generate_fallback_pipeline(filters, intent, account_number)
            
            pipeline = cleaned_response
            jsonschema.validate(pipeline, PIPELINE_SCHEMA)
            return pipeline
            
        except Exception as e:
            logger.error({
                "action": "generate_pipeline_from_filters",
                "error": str(e)
            })
            return self._generate_fallback_pipeline(filters, intent, account_number)

    async def _execute_llm_pipeline(self, account_number: str, pipeline: List[Dict[str, Any]], 
                          user_message: str, first_name: str, memory: ConversationBufferMemory, 
                          intent: str, is_balance_query: bool = False) -> str:
        """Execute LLM-generated pipeline and format response naturally."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/execute_pipeline",
                    json={"account_number": account_number, "pipeline": pipeline}
                )
                response.raise_for_status()
                data = response.json()
                
                # Set appropriate context state for balance queries
                if is_balance_query:
                    context_state = ContextStates.BALANCE_INQUIRY
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    response_text = await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                    
                    # ADD THIS: Update memory for balance queries - CRITICAL FIX
                    memory.chat_memory.add_user_message(user_message)
                    memory.chat_memory.add_ai_message(response_text)
                    
                    return response_text
                else:
                    # Use contextual banking response for other queries
                    return await self.generate_contextual_banking_response(data, user_message, first_name, memory, intent) 
                
        except Exception as e:
            logger.error(f"Error executing LLM pipeline: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            
            # ADD THIS: Update memory for errors too
            response_text = await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response_text)
            
            return response_text


    async def process_query(self, user_message: str, account_number: str, first_name: str) -> str:
        """Enhanced process query with contextual awareness (exit detection handled at webhook level)."""
        memory = self.get_user_memory(account_number)

        logger.info({
            "action": "process_query_start",
            "approach": "context_aware_llm_first_no_exit_check",
            "user_message": user_message,
            "account_number": account_number,
            "note": "exit_detection_handled_by_webhook"
        })

        # Get conversation history for context
        conversation_history = self._get_context_summary(memory.chat_memory.messages)

        # Handle greetings and simple queries FIRST
        if self._is_simple_greeting_or_general(user_message):
            context_state = ContextStates.GENERAL_BANKING
            data = {"greeting": True, "services_available": True}
            response = await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response
        
        # Check for currency conversion requests BEFORE non-banking filter
        if self.detect_currency_conversion_intent(user_message, conversation_history):
            logger.info("Currency conversion intent detected for: " + user_message)
            response = await self.handle_currency_conversion(user_message, conversation_history, first_name, memory)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # NOW check for non-banking queries (after allowing banking assistant features)
        if self.is_clearly_non_banking_query(user_message, conversation_history):
            logger.info(f"🚫 STRICT filter blocked non-banking query: {user_message}")
            response = await self.handle_non_banking_query(user_message, first_name)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # Resolve contextual queries into standalone queries
        original_message = user_message
        resolved_query = self.resolve_contextual_query(user_message, conversation_history)
        
        if resolved_query != original_message:
            logger.info(f"Using resolved query for processing: '{resolved_query}'")
            processing_message = resolved_query
        else:
            processing_message = user_message

        # PRIMARY PATH: LLM-FIRST APPROACH with resolved query
        try:
            logger.info("Attempting context-aware LLM-first approach")
            
            # Step 1: Extract filters using resolved query
            filters = self.extract_filters_with_llm(processing_message)
            logger.info(f"LLM extracted filters from resolved query: {filters.dict()}")
            
            # Step 2: Detect intent using resolved query
            intent = self.detect_intent_from_filters(processing_message, filters)
            logger.info(f"LLM detected intent: {intent}")
            
            # Step 3: Handle based on intent
            if intent == BankingIntents.BALANCE_INQUIRY:
                # Generate pipeline for balance inquiry
                pipeline = self.generate_pipeline_from_filters(filters, intent, account_number)
                return await self._execute_llm_pipeline(
                    account_number=account_number,
                    pipeline=pipeline,
                    user_message=user_message,
                    first_name=first_name,
                    memory=memory,
                    intent=intent,
                    is_balance_query=True
                )
            
            elif intent == BankingIntents.TRANSFER_MONEY:
                response = await self._handle_money_transfer_with_otp(account_number, original_message, first_name, memory)
                memory.chat_memory.add_user_message(original_message)
                memory.chat_memory.add_ai_message(response)
                return response
            
            elif intent in [BankingIntents.TRANSACTION_HISTORY, BankingIntents.SPENDING_ANALYSIS, BankingIntents.CATEGORY_SPENDING]:
                # Generate pipeline using resolved query
                pipeline = self.generate_pipeline_from_filters(filters, intent, account_number)
                logger.info(f"LLM generated pipeline from resolved query: {pipeline}")
                
                # Determine if this is a balance query based on original message
                is_balance_query = any(keyword in original_message.lower() for keyword in BalanceKeywords.KEYWORDS)
                
                # Execute with original message for natural response
                if pipeline:
                    try:
                        jsonschema.validate(pipeline, PIPELINE_SCHEMA)
                        # Pass is_balance_query flag to execution method
                        response = await self._execute_llm_pipeline(account_number, pipeline, original_message, first_name, memory, intent, is_balance_query)
                        memory.chat_memory.add_user_message(original_message)
                        memory.chat_memory.add_ai_message(response)
                        return response
                    except jsonschema.ValidationError as e:
                        logger.warning(f"LLM pipeline validation failed: {e}")
                        raise Exception("Pipeline validation failed")
                else:
                    raise Exception("Empty pipeline generated")
            
            else:
                # General or contextual query
                context_state = "General banking assistance or contextual query with history"
                response = await self.generate_natural_response(context_state, {"resolved_query": resolved_query}, original_message, first_name, conversation_history)
                memory.chat_memory.add_user_message(original_message)
                memory.chat_memory.add_ai_message(response)
                return response
                
        except Exception as llm_error:
            logger.warning(f"Context-aware LLM approach failed: {llm_error}. Falling back.")
            
            # FALLBACK PATH with resolved query
            try:
                logger.info("Using contextual fallback approach")
                reasoning_result = await self._reason_about_query(processing_message, memory, account_number, first_name)
                
                # Handle with fallback methods using original message for response
                if reasoning_result.get("action_needed") == "balance_check":
                    response = await self._handle_balance_inquiry(account_number, first_name, original_message, memory)
                elif reasoning_result.get("action_needed") == "transaction_history":
                    response = await self._handle_transaction_history(original_message, account_number, first_name, reasoning_result, memory)
                elif reasoning_result.get("action_needed") == "sophisticated_analysis":
                    response = await self._handle_sophisticated_analysis(original_message, account_number, first_name, reasoning_result, memory)
                else:
                    context_state = "Fallback - providing contextual assistance"
                    response = await self.generate_natural_response(context_state, {"resolved_query": resolved_query}, original_message, first_name, conversation_history)
                
                memory.chat_memory.add_user_message(original_message)
                memory.chat_memory.add_ai_message(response)
                return response
                
            except Exception as fallback_error:
                logger.error(f"Both context-aware LLM and fallback failed: {fallback_error}")
                context_state = ContextStates.ERROR_OCCURRED
                response = await self.generate_natural_response(context_state, {"error": str(fallback_error), "resolved_query": resolved_query}, original_message, first_name, conversation_history)
                memory.chat_memory.add_user_message(original_message)
                memory.chat_memory.add_ai_message(response)
                return response

    async def handle_transfer_cancellation_during_process(self, first_name: str, stage: str) -> str:
        """Handle transfer cancellation during the transfer process (OTP or confirmation stage)."""
        try:
            context_state = f"User cancelled transfer during {stage} stage, providing cancellation confirmation and returning to normal banking"
            data = {
                "transfer_cancelled": True,
                "cancellation_stage": stage,
                "account_secure": True,
                "ready_for_new_requests": True,
                "user_initiated_cancellation": True
            }
            
            return await self.generate_natural_response(context_state, data, f"cancel transfer during {stage}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer cancellation during process: {e}")
            return f"Transfer cancelled, {first_name}. You're back to your main banking menu. What can I help you with?"


    # === HANDLE MONEY TRANSFERS ===

    async def handle_transfer_otp_request(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle OTP request for money transfer with enhanced formatting."""
        try:
            transfer_otp_prompt = f"""You are Sage, a banking assistant. The user wants to transfer money and needs to provide an **OTP** for security.

            TRANSFER DETAILS:
            - Amount: **{amount} {currency}**
            - Recipient: **{recipient}**
            - User: **{first_name}**

            Your response should:
            1. Confirm the transfer details clearly
            2. Request **OTP** for security verification
            3. Explain OTP format (**{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits**)
            4. **Bold** all important elements like amounts, recipient name, **OTP**
            5. Make it clear this is for security

            Generate a well-formatted, secure response."""
            
            response = await llm.ainvoke([SystemMessage(content=transfer_otp_prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error in transfer OTP request: {e}")
            return f"To complete the transfer of **{amount} {currency}** to **{recipient}**, please provide an **OTP** (any number between **{Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits**) for security verification."
        
    async def handle_transfer_otp_success(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle successful transfer OTP verification."""
        try:
            context_state = ContextStates.TRANSFER_OTP_VERIFIED
            data = {
                "otp_verified": True,
                "transfer_amount": amount,
                "transfer_currency": currency,
                "transfer_recipient": recipient,
                "ready_to_transfer": True
            }
            
            return await self.generate_natural_response(context_state, data, "transfer_otp_verified", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer OTP success: {e}")
            return f"OTP verified! I'll now proceed with transferring {amount} {currency} to {recipient}."
        
    async def _handle_money_transfer_with_otp(self, account_number: str, user_message: str, 
                                        first_name: str, memory: ConversationBufferMemory) -> str:
        """Handle money transfer with OTP requirement (returns 'OTP_REQUIRED' for webhook to handle)."""
        try:
            # Get conversation history for contextual transfers
            conversation_history = self._get_context_summary(memory.chat_memory.messages)

            # FIXED: Enhanced transfer prompt with STRICT clarification rules
            enhanced_transfer_prompt = f"""
                    Extract transfer details from the query, using conversation history for context. Handle multi-turn transfer conversations.

                    CONVERSATION HISTORY:
                    {conversation_history}

                    CURRENT TRANSFER REQUEST: "{user_message}"

                    **CRITICAL CLARIFICATION RULE - MUST FOLLOW:**
                    
                    1. **SCAN CONVERSATION HISTORY** for transactions from the SAME vendor/merchant
                    2. **COUNT** how many transactions exist for that vendor
                    3. **IF 2+ transactions from same vendor** AND user refers to "the [vendor] transaction" → **MUST ASK FOR CLARIFICATION**
                    
                    **CLARIFICATION REQUIRED PATTERNS:**
                    - "the mcdonald transaction" (when multiple McDonald's exist)
                    - "the netflix payment" (when multiple Netflix exist)  
                    - "the uber transaction" (when multiple Uber exist)
                    - "the foodpanda transaction" (when multiple Foodpanda exist)
                    
                    **CLARIFICATION NOT REQUIRED:**
                    - "the latest mcdonald transaction" (user specified which one)
                    - "the $17.25 mcdonald transaction" (user specified amount)
                    - "the june 22 mcdonald transaction" (user specified date)
                    - Only 1 transaction from that vendor exists
                    
                    **STEP 1: CHECK FOR CLARIFICATION NEED**
                    
                    Look in conversation history for vendor mentioned in current request:
                    - If user says "mcdonald" → count McDonald's transactions
                    - If user says "netflix" → count Netflix transactions  
                    - If user says "uber" → count Uber transactions
                    - If user says "foodpanda" → count Foodpanda transactions
                    
                    **STEP 2: APPLY STRICT RULE**
                    
                    IF (vendor_count >= 2 AND user_refers_ambiguously):
                        return {{"clarification_needed": true, "matching_transactions": [list_of_matching_transactions], "message": "Which [vendor] transaction?"}}
                    ELSE:
                        calculate transfer normally
                    
                    **EXAMPLES OF CLARIFICATION REQUIRED:**
                    
                    History shows: 
                    "McDonald's $17.25,
                    McDonald's $20.89, 
                    McDonald's $12.65"
                    Current: "transfer 10% of the mcdonald transaction to Yusra"
                    → {{"clarification_needed": true, "matching_transactions": [...], "message": "Which McDonald's transaction?"}}
                    
                    History shows: 
                    "Foodpanda $10.49, 
                    Foodpanda $9.37"  
                    Current: "transfer 15% of the foodpanda transaction to John"
                    → {{"clarification_needed": true, "matching_transactions": [...], "message": "Which Foodpanda transaction?"}}
                    
                    **EXAMPLES WHERE NO CLARIFICATION NEEDED:**
                    
                    History shows: "Netflix $15.99" (only 1 Netflix)
                    Current: "transfer 5% of netflix to Sarah" 
                    → Calculate normally: {{"amount": 0.80, "currency": "USD", "recipient": "Sarah"}}
                    
                    History shows: "McDonald's $17.25, McDonald's $20.89"
                    Current: "transfer 10% of the latest mcdonald transaction to Yusra"
                    → Calculate normally (user specified "latest")
                    
                    **STEP 3: EXTRACT MATCHING TRANSACTIONS**
                    
                    If clarification is needed, extract ALL matching transactions from conversation history:
                    - Same vendor/merchant name (case-insensitive)
                    - Include: date, description, amount, currency, type
                    - Return in "matching_transactions" array
                    
                    **STEP 4: RETURN FORMAT**
                    
                    For clarification: {{"amount": null, "currency": null, "recipient": string, "clarification_needed": true, "matching_transactions": array, "message": "Which [vendor] transaction?"}}
                    
                    For normal transfer: {{"amount": number, "currency": string, "recipient": string, "clarification_needed": false, "matching_transactions": []}}

                    **BE STRICT**: When in doubt about multiple transactions, ALWAYS ask for clarification.
                    """
            
            logger.info(f"🔍 TRANSFER DEBUG - Using STRICT clarification prompt for: {user_message}")

            response = await llm.ainvoke([SystemMessage(content=enhanced_transfer_prompt)])

            logger.info(f"🔍 TRANSFER DEBUG - LLM response: {response.content}") 

            transfer_details = self.extract_json_from_response(response.content)

            # CRITICAL FIX: Check for clarification_needed FIRST, before any fallback logic
            if transfer_details and transfer_details.get("clarification_needed"):
                logger.info("🔍 CLARIFICATION NEEDED: Multiple transactions match user's request")
                
                # Format the clarification response
                matching_transactions = transfer_details.get("matching_transactions", [])
                
                context_state = "Multiple transactions found, asking user to specify which one"
                data = {
                    "clarification_needed": True,
                    "matching_transactions": matching_transactions,
                    "transaction_count": len(matching_transactions),
                    "user_request": user_message,
                    "recipient": transfer_details.get("recipient"),
                    "percentage": "10%"  # Extract from user message if needed
                }
                
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)

            # If transfer details are incomplete, try to complete from conversation history
            if not transfer_details or not all([transfer_details.get("amount"), transfer_details.get("recipient")]):
                logger.info("Transfer details incomplete, checking conversation history")
                historical_details = self._extract_incomplete_transfer_from_history(conversation_history, user_message)
                
                if historical_details:
                    # Merge current details with historical details
                    if not transfer_details:
                        transfer_details = {}
                    
                    for key in ["amount", "currency", "recipient"]:
                        if not transfer_details.get(key) and historical_details.get(key):
                            transfer_details[key] = historical_details[key]
                            logger.info(f"Completed {key} from conversation history: {historical_details[key]}")

            logger.info(f"Final transfer details after context merge: {transfer_details}")
            
            if not transfer_details:
                context_state = "Transfer details could not be understood, need clarification"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, {"missing": "transfer_details"}, user_message, first_name, conversation_history)

            # Check completeness
            missing_parts = []
            if not transfer_details.get("amount") or transfer_details.get("amount") <= 0:
                missing_parts.append("amount")
            if not transfer_details.get("recipient"):
                missing_parts.append("recipient")
            
            if missing_parts:
                context_state = f"Transfer request incomplete, missing: {', '.join(missing_parts)}"
                data = {
                    "missing_info": missing_parts,
                    "provided_amount": transfer_details.get("amount"),
                    "provided_recipient": transfer_details.get("recipient"),
                    "currency": transfer_details.get("currency", Currencies.PKR)
                }
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # If transfer details are complete, return special signal for OTP
            return f"{TransferSignals.OTP_REQUIRED_PREFIX}{TransferSignals.SEPARATOR}{transfer_details.get('amount')}{TransferSignals.SEPARATOR}{transfer_details.get('currency')}{TransferSignals.SEPARATOR}{transfer_details.get('recipient')}"
                
        except Exception as e:
            logger.error(f"Error in money transfer with OTP: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)


    def _extract_incomplete_transfer_from_history(self, conversation_history: str, current_message: str) -> Dict[str, Any]:
        """Extract transfer details from recent conversation history to complete current request."""
        try:
            extraction_prompt = f"""
            Analyze recent conversation to find incomplete transfer details that can complete the current request.

            CONVERSATION HISTORY:
            {conversation_history}

            CURRENT MESSAGE: "{current_message}"

            Task: If current message provides partial transfer info (just recipient or just amount), 
            look in conversation history for the missing pieces.

            Examples:
            History: "transfer 1% of 1554.41 USD which is 15.54 USD"
            Current: "to ahmed abrar" 
            → Complete transfer: {{"amount": 15.54, "currency": "USD", "recipient": "ahmed abrar"}}

            History: "I need transfer details" 
            Current: "to john"
            → Incomplete: {{"amount": null, "currency": null, "recipient": "john"}}

            Return JSON with completed transfer details or null for missing fields:
            {{"amount": number_or_null, "currency": string_or_null, "recipient": string_or_null}}
            """
            
            response = self.llm.invoke([SystemMessage(content=extraction_prompt)])
            result = self.extract_json_from_response(response.content)
            return result if result else {}
            
        except Exception as e:
            logger.error(f"Error extracting transfer context: {e}")
            return {}
        
    async def execute_verified_transfer(self, account_number: str, amount: float, currency: str, recipient: str, first_name: str, memory: ConversationBufferMemory) -> str:
        """Execute the transfer after OTP verification."""
        try:
            logger.info({
                "action": "EXECUTING_VERIFIED_TRANSFER_START",
                "account_number": account_number,
                "amount": amount,
                "currency": currency,
                "recipient": recipient,
                "backend_url": self.backend_url
            })
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/transfer_money",
                    json={
                        "from_account": account_number,
                        "to_recipient": recipient,
                        "amount": amount,
                        "currency": currency
                    }
                )
                response.raise_for_status()
                transfer_result = response.json()
                
                logger.info({
                    "action": "TRANSFER_API_RESPONSE",
                    "status": transfer_result.get("status"),
                    "result": transfer_result
                })
                
                # CRITICAL: Only report success if backend confirms success
                if transfer_result.get("status") == StatusMessages.SUCCESS:
                    context_state = ContextStates.TRANSFER_EXECUTED_SUCCESS
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    return await self.generate_natural_response(context_state, transfer_result, f"transfer {amount} {currency} to {recipient}", first_name, conversation_history)
                else:
                    context_state = ContextStates.TRANSFER_EXECUTED_FAILED
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    return await self.generate_natural_response(context_state, transfer_result, f"transfer {amount} {currency} to {recipient}", first_name, conversation_history)
                    
        except Exception as e:
            logger.error({
                "action": "TRANSFER_EXECUTION_FAILED",
                "error": str(e),
                "account_number": account_number,
                "amount": amount,
                "recipient": recipient
            })
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, f"transfer {amount} {currency} to {recipient}", first_name, conversation_history)
     
    async def handle_transfer_confirmation_request(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle transfer confirmation request after OTP verification."""
        try:
            context_state = ContextStates.TRANSFER_OTP_VERIFIED
            data = {
                "otp_verified": True,
                "transfer_amount": amount,
                "transfer_currency": currency,
                "transfer_recipient": recipient,
                "confirmation_required": True,
                "confirmation_options": ConfirmationWords.POSITIVE + ConfirmationWords.NEGATIVE
            }
            
            return await self.generate_natural_response(context_state, data, f"confirm transfer {amount} {currency} to {recipient}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer confirmation request: {e}")
            return f"OTP verified! Do you want to confirm the transfer of {amount} {currency} to {recipient}? Please reply with 'yes' to confirm or 'no' to cancel."

    async def handle_transfer_cancellation(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle transfer cancellation when user declines confirmation."""
        try:
            context_state = ContextStates.TRANSFER_CANCELLED
            data = {
                "transfer_cancelled": True,
                "cancelled_amount": amount,
                "cancelled_currency": currency,
                "cancelled_recipient": recipient,
                "account_secure": True,
                "ready_for_new_requests": True
            }
            
            return await self.generate_natural_response(context_state, data, f"cancel transfer {amount} {currency} to {recipient}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer cancellation: {e}")
            return f"Transfer cancelled, {first_name}. The transfer of {amount} {currency} to {recipient} has been stopped and your account is secure. Is there anything else I can help you with?"

    async def handle_transfer_confirmation_clarification(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle unclear confirmation response and ask for clarification."""
        try:
            context_state = "User provided unclear confirmation response, requesting clear yes/no answer"
            data = {
                "confirmation_unclear": True,
                "transfer_amount": amount,
                "transfer_currency": currency,
                "transfer_recipient": recipient,
                "clarification_needed": True,
                "clear_options": ["yes/confirm", "no/cancel"]
            }
            
            return await self.generate_natural_response(context_state, data, f"clarify transfer confirmation {amount} {currency} to {recipient}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer confirmation clarification: {e}")
            return f"I want to make sure, {first_name}. Do you want to proceed with transferring {amount} {currency} to {recipient}? Please reply with 'yes' to confirm or 'no' to cancel."
            
    # == CURRENCY CONVERSION HANDLING ===
    def detect_currency_conversion_intent(self, user_message: str, conversation_history: str) -> bool:
        """Detect if user wants to convert currency amounts."""
        try:
            from prompts import currency_conversion_intent_prompt
            
            response = llm.invoke([SystemMessage(content=currency_conversion_intent_prompt.format(
                user_message=user_message,
                conversation_history=conversation_history
            ))])
            
            result = response.content.strip().upper()
            logger.info(f"Currency conversion intent detection: {result} for message: '{user_message}'")
            
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Error detecting currency conversion intent: {e}")
            return False

    async def handle_currency_conversion(self, user_message: str, conversation_history: str, first_name: str, memory: ConversationBufferMemory) -> str:
        """Handle currency conversion requests."""
        try:
            from prompts import currency_extraction_prompt
            from currency_service import currency_converter
            
            # Extract conversion details using LLM
            response = await llm.ainvoke([SystemMessage(content=currency_extraction_prompt.format(
                user_message=user_message,
                conversation_history=conversation_history
            ))])
            
            conversion_details = self.extract_json_from_response(response.content)
            
            if not conversion_details:
                context_state = "Could not understand currency conversion request, asking for clarification"
                return await self.generate_natural_response(
                    context_state, 
                    {"error": "conversion_details_unclear"}, 
                    user_message, 
                    first_name, 
                    conversation_history
                )
            
            # Validate required fields
            amount = conversion_details.get("amount")
            from_currency = conversion_details.get("from_currency")
            to_currency = conversion_details.get("to_currency")
            context = conversion_details.get("context", "amount")
            
            if not all([amount, from_currency, to_currency]):
                context_state = "Currency conversion request missing required information"
                data = {
                    "missing_fields": [k for k in ["amount", "from_currency", "to_currency"] 
                                    if not conversion_details.get(k)],
                    "provided_details": conversion_details
                }
                return await self.generate_natural_response(
                    context_state, data, user_message, first_name, conversation_history
                )
            
            # Perform currency conversion
            conversion_result = await currency_converter.convert_currency(amount, from_currency, to_currency)
            
            if conversion_result and conversion_result.get("conversion_successful"):
                context_state = f"Successfully converted {context} from {from_currency} to {to_currency}"
                data = {
                    "conversion_result": conversion_result,
                    "context": context,
                    "user_requested": True
                }
                
                response_text = await self.generate_natural_response(
                    context_state, data, user_message, first_name, conversation_history
                )
                
                # Add to memory
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response_text)
                
                return response_text
            else:
                context_state = "Currency conversion failed, providing error response"
                data = {
                    "conversion_failed": True,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "amount": amount
                }
                return await self.generate_natural_response(
                    context_state, data, user_message, first_name, conversation_history
                )
                
        except Exception as e:
            logger.error(f"Error in currency conversion: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            return await self.generate_natural_response(
                context_state, {"error": str(e)}, user_message, first_name, conversation_history
            )

    # === NATURAL LANGUAGE RESPONSE GENERATION ===
    async def _determine_response_format_with_llm(self, user_message: str, data: Any, context_state: str) -> str:
        """Use LLM to determine appropriate response format based on query complexity and data."""

        data_summary = self._summarize_data_for_format_analysis(data)
        
        format_analysis_prompt = f"""
            You are a response format analyzer. Analyze the user query, available data, and context to determine the best response format.

            USER QUERY: "{user_message}"
            CONTEXT: {context_state}
            DATA AVAILABLE: {data_summary}

            Determine the most appropriate response format:

            1. **{ResponseFormats.CONCISE_ONE_LINER}**: For simple, direct questions wanting a single piece of information
            - Examples: "What's my balance?", "How much did I spend on Netflix?", "Current balance?"
            - When data contains single values: one balance, one amount, one simple answer

            2. **{ResponseFormats.STRUCTURED_LIST}**: For queries requesting multiple items or detailed breakdowns
            - Examples: "Show my transactions", "List my spending by category", "Transaction history"
            - When data contains: multiple transactions, category breakdowns, lists of items

            3. **{ResponseFormats.DETAILED_EXPLANATION}**: For complex queries, comparisons, or when context/explanation is needed
            - Examples: "Compare my spending", "Why is my spending high?", "Analyze my patterns"
            - When data needs interpretation, contains errors, or requires explanation

            4. **{ResponseFormats.HELPFUL_GUIDANCE}**: For error states, missing information, or clarification requests
            - When context indicates errors, missing data, or user needs guidance

            Analyze the query intent, data complexity, and user expectation to choose the best format.

            Return ONLY one of: {ResponseFormats.CONCISE_ONE_LINER}, {ResponseFormats.STRUCTURED_LIST}, {ResponseFormats.DETAILED_EXPLANATION}, or {ResponseFormats.HELPFUL_GUIDANCE}
            """
        try:
            response = await llm.ainvoke([SystemMessage(content=format_analysis_prompt)])
            format_type = response.content.strip().upper()
            
            # Simplified format instructions - let LLM format naturally
            format_instructions = {
                ResponseFormats.CONCISE_ONE_LINER: """
            Provide a direct, single sentence answer with the specific information requested.
            Keep it brief and natural.
            """,
                ResponseFormats.STRUCTURED_LIST: """
            Present data in a clear, organized format. 
            Show all relevant details naturally.
            Include helpful context where appropriate.
            """,
                ResponseFormats.DETAILED_EXPLANATION: """
            Provide thorough explanation with context.
            Include all relevant information and insights.
            Help user understand patterns or findings.
            """,
                ResponseFormats.HELPFUL_GUIDANCE: """
            Provide clear, supportive guidance.
            Explain what's needed and offer next steps.
            Keep it friendly and helpful.
            """
            }
            
            return format_instructions.get(format_type, format_instructions[ResponseFormats.DETAILED_EXPLANATION])
            
        except Exception as e:
            logger.error(f"Error in LLM format analysis: {e}")
            return "Provide a natural, helpful response based on the available data."


    def _summarize_data_for_format_analysis(self, data: Any) -> str:
        """Create a summary of available data for format analysis."""
        if not data:
            return "No data available"
        
        if isinstance(data, dict):
            summary_parts = []
            
            # Check for single values
            single_values = ["current_balance", "total_spent", "total_amount"]
            for key in single_values:
                if key in data:
                    summary_parts.append(f"Single value: {key}")
            
            # Check for lists/multiple items
            if "transactions" in data:
                count = len(data["transactions"]) if isinstance(data["transactions"], list) else 0
                summary_parts.append(f"Transaction list: {count} items")
            
            if "category_breakdown" in data:
                count = len(data["category_breakdown"]) if isinstance(data["category_breakdown"], list) else 0
                summary_parts.append(f"Category breakdown: {count} categories")
            
            # Check for error states
            if "error" in data or "missing" in str(data):
                summary_parts.append("Error or missing information")
            
            # Check for complex analysis data
            complex_keys = ["comparison", "analysis", "breakdown", "facet"]
            if any(key in data for key in complex_keys):
                summary_parts.append("Complex analysis data")
            
            return "; ".join(summary_parts) if summary_parts else "Simple data object"
        
        return f"Data type: {type(data).__name__}"
    
    async def generate_natural_response(self, context_state: str, data: Any, user_message: str, first_name: str, conversation_history: str = "") -> str:
        """Generate natural LLM responses with bold formatting and highlighting for important keywords."""
        
        # Special handling for non-banking queries
        if "non-banking question" in context_state or data and data.get("query_type") == "non_banking":
            non_banking_prompt = f"""You are Kora, a banking assistant. The user {first_name} just asked a non-banking question that you must politely but firmly decline to answer.

            Their question: "{user_message}"

            Response rules:
            1. Be polite but firm - you ONLY handle banking queries
            2. Don't apologize excessively - be confident in your banking focus
            3. Clearly state you cannot help with non-banking topics
            4. Redirect to specific banking services you can help with
            5. Keep it concise and professional
            6. **BOLD** important banking services you offer

            Generate a response that maintains boundaries while being helpful about banking topics."""

            try:
                response = await llm.ainvoke([SystemMessage(content=non_banking_prompt)])
                return response.content.strip()
            except:
                return f"I'm a banking assistant, {first_name}, and I can only help with your account-related questions like **checking balances**, **viewing transactions**, **analyzing spending**, or **transferring money**. I don't have information about topics outside of banking. What **banking question** can I help you with today?"

        # CRITICAL FIX: Handle clarification needed for transfers
        if "clarification_needed" in str(data) and data.get("clarification_needed"):
            clarification_prompt = f"""You are Kora, a banking assistant. The user **{first_name}** asked to transfer money but there are multiple matching transactions.

            USER REQUEST: "{user_message}"
            MATCHING TRANSACTIONS: {json.dumps(data.get("matching_transactions", []))}
            RECIPIENT: {data.get("recipient", "unknown")}

            Your task:
            1. Explain that you found multiple matching transactions
            2. List each transaction clearly with **date**, **amount**, and **currency**
            3. Ask them to specify which one they want to use
            4. **Bold** all important information
            5. Make it easy for them to identify and select

            Example format:
            "I found **multiple Foodpanda transactions**, **{first_name}**. Which one would you like to transfer 10% from?

            **1.** June 19: **$18.71 USD**
            **2.** May 14: **$24.10 USD** 
            **3.** May 10: **$21.49 USD**
            **4.** April 6: **$16.10 USD**

            Please specify by saying **'the first one'**, **'the $24.10 transaction'**, **'the May 14 one'**, etc."

            Generate a clear clarification request with proper **bold formatting**.
            """
            
            try:
                response = await llm.ainvoke([SystemMessage(content=clarification_prompt)])
                return response.content.strip()
            except Exception as e:
                logger.error(f"Error generating clarification response: {e}")
                # Fallback clarification
                transactions = data.get("matching_transactions", [])
                transaction_list = []
                for i, tx in enumerate(transactions, 1):
                    date = tx.get("date", "Unknown date")
                    amount = tx.get("amount", 0)
                    currency = tx.get("currency", "USD")
                    transaction_list.append(f"**{i}.** {date}: **${amount} {currency}**")
                
                transactions_text = "\n".join(transaction_list)
                
                return f"I found **multiple Foodpanda transactions**, **{first_name}**. Which one would you like to transfer from?\n\n{transactions_text}\n\nPlease specify which transaction you'd like to use."

        # Use LLM to determine response approach
        response_format_instruction = await self._determine_response_format_with_llm(user_message, data, context_state)
        
        system_prompt = f"""You are Kora, a professional banking assistant. Provide helpful, natural responses to banking queries with proper formatting and highlighting.

        {response_format_instruction}

        CURRENT CONTEXT: {context_state}
        USER'S MESSAGE: "{user_message}"

        CONVERSATION HISTORY (Your memory):
        {conversation_history}
        DATA AVAILABLE: {json.dumps(data) if data else "No data available"}

        **CRITICAL FORMATTING RULES - ALWAYS APPLY:**

        1. **BOLD FORMATTING** for important keywords and actions:
        - **CNIC** (when requesting CNIC verification)
        - **OTP** (when requesting OTP)
        - **Account numbers** (like ***-***-1234)
        - **Important amounts** ($1,554.41, PKR 25,000)
        - **Action words** (transfer, send, confirm, verify, check)
        - **Banking terms** (balance, transactions, spending, account)
        - **Currency codes** (**USD**, **PKR**, **CAD**)
        - **Recipient names** when transferring money
        - **Next steps** user needs to take

        2. **STRUCTURE RESPONSES** clearly:
        - Use line breaks for readability
        - Separate different types of information
        - Highlight key numbers and amounts
        - Make action items stand out

        3. **EXAMPLES OF PROPER FORMATTING:**

        CNIC Request: "Please provide your **CNIC** in the format **12345-1234567-1**"
        
        OTP Request: "Please enter your **OTP** (**4-6 digits**) to verify the transfer"
        
        Balance Response: "Your current account balance is **$1,554.41 USD** as of July 30th"
        
        Transfer Confirmation: "Transfer **$15.54 USD** to **Ahmed Abrar**? Reply **yes** to confirm or **no** to cancel"
        
        Account Selection: "Account ***-***-1234** confirmed! Welcome **{first_name}**!"

        4. **TABULAR DATA FORMATTING RULES:**
        When presenting data that would be clearer in tabular format, use markdown tables:

        For TRANSACTIONS:
        | Date | Description | Type | Amount | Currency | Balance |
        |------|-------------|------|--------|----------|---------|
        | 2025-07-30 | Netflix | Debit | **15.99** | **USD** | **1,234.56** |
        | 2025-07-29 | Salary | Credit | **3,000.00** | **USD** | **1,250.55** |

        For SPENDING BREAKDOWN:
        | Category | Amount | Percentage | Count |
        |----------|--------|------------|-------|
        | Food | **450.00 USD** | **35%** | 12 |
        | Entertainment | **200.00 USD** | **15%** | 5 |

        For CURRENCY CONVERSION:
        | From | To | Amount | Converted | Rate |
        |------|----|---------|---------|----- |
        | **USD** | **PKR** | **100** | **27,800** | **278.00** |

        Use tables when:
        - Showing multiple transactions (more than 2-3)
        - Displaying spending breakdowns by category
        - Presenting currency conversions
        - Comparing data across periods
        - Any structured data that benefits from column organization

        For single values (like balance inquiries), use natural text format with bold highlighting.

        5. **CONTEXTUAL RESPONSE RULES:**
        1. Reference Previous Data: If you previously showed transactions, balances, or spending - reference them specifically
        2. Build on Previous Context: Make it feel like a continuous conversation
        3. Use Specific Numbers/Details: Reference exact amounts, dates, descriptions from previous messages
        4. Natural Conversation Flow: 
        - If user asks follow-up questions, acknowledge the connection
        - If showing new data, relate it to previous context when relevant
        - Make responses feel like you remember everything

        6. Contextual Greetings: 
        - Don't always greet the same way
        - Sometimes skip greetings for follow-ups
        - For continuing conversations: "**{first_name}**, looking at that data..."

        7. Balance Query Responses:
        - If context mentions "balance at specific date", focus on the date and balance amount
        - If context mentions "average balance", explain the calculation period
        - Always include currency information for balance responses
        - Reference the specific date or period requested

        8. NO TRANSACTIONS OR AMOUNT IS ZERO:
        - If there are no transactions or if total amount is equal to 0, provide a clear message like "**No transactions/amount found** for this period/category."

        9. **HIGHLIGHT NEXT STEPS:**
        - Make it crystal clear what the user needs to do next
        - Use bold formatting for action items
        - Example: "**Next step:** Please enter your **OTP** to proceed"
        - Example: "**To continue:** Select your **account number** or say **'first account'**"

        **PERSONALITY GUIDELINES:**
        - Be conversational like you're having an ongoing chat
        - Show you remember previous parts of conversation
        - Reference specific data points naturally
        - Make each response build on the previous conversation
        - Avoid repetitive patterns - vary your language
        - **ALWAYS** use proper bold formatting for important elements

        Generate a natural, well-formatted response that feels like a continuation of your ongoing conversation with **{first_name}**."""
        
        try:
            response = await llm.ainvoke([SystemMessage(content=system_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            return f"I'm having some technical difficulties right now, **{first_name}**. Could you try that again?"

    async def generate_contextual_banking_response(self, query_result: Any, user_message: str, first_name: str, memory: ConversationBufferMemory, intent: str) -> str:
        """Generate natural banking responses with bold formatting and highlighting."""
        
        conversation_history = self._get_context_summary(memory.chat_memory.messages)

        banking_context_prompt = f"""You are Kora, a professional banking assistant. Generate natural, helpful responses to banking queries with proper formatting.

            **CRITICAL FORMATTING RULES - ALWAYS APPLY:**

            1. **BOLD FORMATTING** for important elements:
            - **Account numbers** (***-***-1234)
            - **Important amounts** (**$1,554.41**, **PKR 25,000**)
            - **Currency codes** (**USD**, **PKR**, **CAD**)
            - **Action words** (**transfer**, **check**, **analyze**)
            - **Banking terms** (**balance**, **transactions**, **spending**)
            - **Next steps** and action items
            - **Recipient names** in transfers
            - **Dates** and **timeframes**

            2. **TABULAR DATA FORMATTING RULES:**
            When presenting data that would be clearer in tabular format, use markdown tables:

            For TRANSACTIONS:
            | Date | Description | Type | Amount | Currency | Balance |
            |------|-------------|------|--------|----------|---------|
            | 2025-07-30 | Netflix | Debit | **15.99** | **USD** | **1,234.56** |
            | 2025-07-29 | Salary | Credit | **3,000.00** | **USD** | **1,250.55** |

            For SPENDING BREAKDOWN:
            | Category | Amount | Percentage | Count |
            |----------|--------|------------|-------|
            | Food | **450.00 USD** | **35%** | 12 |
            | Entertainment | **200.00 USD** | **15%** | 5 |

            Use tables when showing multiple transactions, spending breakdowns, or structured data.

            3. **STRUCTURE RESPONSES** clearly with line breaks and highlighting

            CONVERSATION HISTORY: {conversation_history}
            USER'S REQUEST: "{user_message}"
            INTENT: {intent}
            QUERY RESULTS: {json.dumps(query_result) if query_result else "No data"}

            **RESPONSE RULES:**
            1. Be Natural: Present information in a conversational way
            2. Reference Context: Connect to previous conversation when relevant
            3. Be Helpful: Provide useful follow-up suggestions
            4. Stay Professional: Maintain banking assistant professionalism
            5. **FORMAT PROPERLY**: Use bold formatting for all important elements
            6. **HIGHLIGHT NUMBERS**: Make amounts, balances, and counts stand out
            7. **CLEAR ACTIONS**: Bold any next steps or action items

            Generate a natural, well-formatted response that addresses **{first_name}**'s banking needs."""
        
        try:
            response = await llm.ainvoke([SystemMessage(content=banking_context_prompt)])
            
            # Add to memory
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response.content.strip())
            
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating contextual banking response: {e}")
            return await self.generate_natural_response(ContextStates.ERROR_OCCURRED, {"error": str(e)}, user_message, first_name, conversation_history)

                        
    def is_clearly_non_banking_query(self, user_message: str, conversation_history: str = "") -> bool:
        """Enhanced non-banking detection with proper blocking logic."""
        try:
            # CRITICAL: Never block banking credentials
            if re.match(r'^\d{5}-\d{7}-\d{1}$', user_message.strip()):
                return False
                
            if re.match(r'^\d+$', user_message.strip()):
                return False

            user_lower = user_message.lower().strip()
            
            # TIER 1: ALWAYS ALLOW - Core banking operations
            banking_core_keywords = [
                "balance", "transaction", "transfer", "money", "account", "spending",
                "pay", "deposit", "withdraw", "statement", "bill", "payment", 
                "bank", "cash", "fund", "amount", "rupee", "dollar", "pkr", "usd",
                "debit", "credit", "cnic", "otp"
            ]
            
            if any(keyword in user_lower for keyword in banking_core_keywords):
                logger.info(f"✅ TIER 1 ALLOWED: Core banking keyword detected - '{user_message}'")
                return False

            # TIER 2: ALLOW - Banking assistant features (currency conversion, etc.)
            banking_assistant_keywords = [
                "convert", "conversion", "exchange rate", "currency", "cad", "gbp", "eur",
                "pounds", "euros", "canadian", "what is", "tell me", "show me", 
                "how much", "calculate", "in dollars", "in pkr", "in usd",
                "translate", "translation", "in urdu", "in arabic", "in english",
                "urdu me", "arabic me", "english me", "language", "bolo", "kaho",
                "write in", "respond in", "reply in", "answer in"
            ]
            
            # Check if it's a banking-related conversion/calculation
            if any(keyword in user_lower for keyword in banking_assistant_keywords):
                # If it also contains banking context or numbers, allow it
                if (any(banking_word in user_lower for banking_word in ["balance", "spent", "amount", "money", "$", "pkr", "usd"]) or
                    re.search(r'\d+', user_message)):
                    logger.info(f"✅ TIER 2 ALLOWED: Banking assistant feature - '{user_message}'")
                    return False

            # TIER 3: KEYWORD-BASED BLOCKING - Check for clearly non-banking topics
            tier3_result = self._tier3_keyword_analysis(user_message, user_lower)
            
            # TIER 4: LLM-BASED BLOCKING - Use AI to analyze unclear cases
            tier4_result = self._tier4_llm_analysis(user_message, conversation_history) if len(user_message.strip()) > 10 else False
            
            # CHANGED: Use OR logic instead of AND logic for blocking
            # If EITHER tier wants to block, then block it
            if tier3_result and tier4_result:
                logger.info(f"🚫 NON-BANKING QUERY BLOCKED: Tier3={tier3_result}, Tier4={tier4_result} - '{user_message}'")
                return True
            else:
                logger.info(f"✅ BANKING QUERY ALLOWED: Both tiers agreed to allow - '{user_message}'")
                return False
                    
        except Exception as e:
            logger.error(f"Error in non-banking detection: {e}")
            # Safe fallback - allow if error (better for banking users)
            return False

    def _tier3_keyword_analysis(self, user_message: str, user_lower: str) -> bool:
        """Tier 3: Keyword-based analysis for clearly non-banking topics."""
        try:
            # TIER 3: BLOCK - Clearly non-banking topics
            non_banking_blacklist = [
                # Technology/Programming
                "python", "programming", "code", "ai", "technology", "computer", "software",
                "algorithm", "machine learning", "data science", "javascript", "html",
                
                # General knowledge
                "president", "ceo", "apple", "google", "microsoft", "amazon", "facebook", "tesla",
                "steve jobs", "bill gates", "elon musk", "mark zuckerberg",
                
                # Entertainment
                "weather", "sports", "movie", "celebrity", "politics", "news",
                "music", "song", "album", "actor", "actress", "film", "tv show",
                "joke", "sing", "entertainment",
                
                # Health/Lifestyle
                "health", "doctor", "medicine", "recipe", "cook", "food recipe",
                "workout", "exercise", "diet", "travel", "vacation", "hotel",
                
                # Education/Career
                "job", "career", "university", "college", "education", "study",
                "homework", "assignment", "research", "thesis"
            ]
            
            # Check for strong non-banking indicators
            strong_non_banking_count = sum(1 for keyword in non_banking_blacklist if keyword in user_lower)
            
            if strong_non_banking_count >= 1:
                logger.info(f"🔍 TIER 3: Found {strong_non_banking_count} non-banking keywords in '{user_message}'")
                return True
            
            # Additional context checks
            # Check if it's asking about companies in non-financial context
            company_context_phrases = [
                "who is ceo", "who founded", "when was founded", "headquarters of",
                "what does company do", "company history", "company products",
                "who is president", "who is the president"
            ]
            
            if any(phrase in user_lower for phrase in company_context_phrases):
                logger.info(f"🔍 TIER 3: Company context phrase detected in '{user_message}'")
                return True
            
            # Check for general knowledge patterns
            general_knowledge_patterns = [
                "what is the capital of", "who invented", "when did", "where is",
                "how to make", "recipe for", "weather in", "temperature in",
                "tell me about", "what do you know about"
            ]
            
            if any(pattern in user_lower for pattern in general_knowledge_patterns):
                logger.info(f"🔍 TIER 3: General knowledge pattern detected in '{user_message}'")
                return True
            
            logger.info(f"✅ TIER 3: No clear non-banking indicators found in '{user_message}'")
            return False
            
        except Exception as e:
            logger.error(f"Error in Tier 3 keyword analysis: {e}")
            return False

    def _tier4_llm_analysis(self, user_message: str, conversation_history: str) -> bool:
        """Tier 4: LLM-based analysis with stricter blocking."""
        try:
            enhanced_llm_prompt = f"""
            You are a banking query analyzer. Determine if this query is clearly non-banking and should be blocked.

            CONVERSATION CONTEXT:
            {conversation_history[-500:] if conversation_history else "No context"}

            USER MESSAGE: "{user_message}"

            **DEFINITELY ALLOW (return "ALLOW"):**
            ✅ Account operations: balance, transactions, transfer, payments, spending analysis
            ✅ Financial calculations: currency conversion of amounts, budgeting
            ✅ Banking assistance: "what can you do", general banking help
            ✅ Language support: translation requests for banking responses

            **DEFINITELY BLOCK (return "BLOCK"):**
            ❌ General knowledge: "Who is president", "Weather today", "Sports scores"
            ❌ Entertainment: "Tell jokes", "Sing songs", "Recommend movies"
            ❌ Technology help: "How to program", "Fix computer"
            ❌ Personal advice: "Health tips", "Dating advice"
            ❌ Academic help: "Solve homework", "Write essay"
            ❌ Company info (non-financial): "Apple's history", "Google's products"

            **CRITICAL DECISION RULE:**
            - If it's clearly about general knowledge, entertainment, technology, etc. → "BLOCK"
            - If it's banking-related or unclear → "ALLOW"

            **EXAMPLES:**
            "Who is CEO of Apple?" → BLOCK
            "What's the weather?" → BLOCK  
            "Who is president of USA?" → BLOCK
            "Convert $100 to PKR" → ALLOW
            "What can you do?" → ALLOW
            "How much is my balance?" → ALLOW

            Return ONLY "BLOCK" or "ALLOW".
            """

            response = llm.invoke([SystemMessage(content=enhanced_llm_prompt)])
            result = response.content.strip().upper()
            
            is_blocked = result == "BLOCK"
            
            if is_blocked:
                logger.info(f"🔍 TIER 4 (LLM): BLOCK decision for '{user_message}'")
            else:
                logger.info(f"🔍 TIER 4 (LLM): ALLOW decision for '{user_message}'")
            
            return is_blocked
            
        except Exception as e:
            logger.error(f"Error in Tier 4 LLM analysis: {e}")
            # Safe fallback - don't block on LLM errors
            return False

    async def handle_non_banking_query(self, user_message: str, first_name: str) -> str:
        """Handle non-banking queries with enhanced formatting."""
        try:
            non_banking_prompt = f"""You are Kora, a strict banking assistant. The user **{first_name}** asked a non-banking question that you must firmly refuse to answer.

            Their question: "{user_message}"

            Response rules:
            1. Be polite but VERY firm - you **ONLY** handle **banking account queries**
            2. Don't provide ANY information about the non-banking topic
            3. Immediately redirect to **banking services** only
            4. **Bold** all banking services you offer
            5. Keep response short and direct
            6. **Highlight** what you CAN do for them

            Banking services to **bold**:
            - **Check account balance**
            - **View transaction history**
            - **Transfer money**
            - **Currency conversion**
            - **Financial planning**

            Generate a firm but polite refusal with proper **bold formatting**."""
            
            try:
                response = await llm.ainvoke([SystemMessage(content=non_banking_prompt)])
                return response.content.strip()
            except:
                return f"I'm a **banking assistant**, **{first_name}**, and I can only help with your **bank account questions** like **checking your balance**, **viewing transactions**, **analyzing spending**, or **transferring money**. I don't provide information about other topics. **What banking question can I help you with?**"
                
        except Exception as e:
            logger.error(f"Error in non-banking response: {e}")
            return f"I'm a **banking assistant**, **{first_name}**, and I can only help with **banking questions** related to your account. **What banking question can I help you with today?**"

    # === FALLBACK METHODS ===
    async def _reason_about_query(self, user_message: str, memory: ConversationBufferMemory, 
                            account_number: str, first_name: str) -> Dict[str, Any]:
        """Natural reasoning layer - let the LLM understand intent intelligently."""
        
        chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        context_summary = self._get_context_summary(chat_history)
        
        reasoning_prompt = f"""
        You are analyzing a banking query to understand what the user really wants. Think about their intent naturally.

        Previous conversation context:
        {context_summary}
        
        User's current message: "{user_message}"
        
        Based on your understanding, what does the user want? Think about it like a human would:

        - Do they want to see their transaction history or recent transactions? (transaction_history)
        - Do they want to understand what caused high spending in a specific period? (spending_breakdown)
        - Do they want to compare spending across different months? (monthly_comparison)  
        - Do they want to know about cutting specific expenses or categories? (category_analysis)
        - Do they want help with savings goals or financial planning? (savings_planning)
        - Are they asking about general spending patterns or trends? (spending_patterns)
        - Do they want their account balance? (balance_check)
        - Is this a simple follow-up about a previous transaction? (simple_contextual)
        - Is this just a greeting or general question? (direct_answer)

        Think about the context and what a human would naturally understand from this query.

        IMPORTANT: 
        - "show me my transactions", "last X transactions", "recent transactions", "transaction history", "transactions in May", "transactions for June" = transaction_history
        - "what did I spend on X", "spending analysis" = spending_breakdown or spending_patterns
        - "balance", "how much money", "can I afford" = balance_check
        - "hello", "hi", "what can you do" = direct_answer

        Return JSON:
        {{
            "action_needed": "transaction_history | sophisticated_analysis | simple_contextual | direct_answer | balance_check",
            "analysis_type": "transaction_history | spending_breakdown | monthly_comparison | category_analysis | savings_planning | spending_patterns",
            "reasoning": "explain your natural understanding of what the user wants"
        }}
        """
        
        try:
            response = await llm.ainvoke([SystemMessage(content=reasoning_prompt)])
            reasoning = self.extract_json_from_response(response.content)
            
            if not reasoning:
                # Check for transaction keywords as fallback
                if any(keyword in user_message.lower() for keyword in TransactionKeywords.KEYWORDS):
                    return {
                        "action_needed": "transaction_history",
                        "analysis_type": "transaction_history",
                        "reasoning": "Fallback detection found transaction keywords"
                    }
                
                # Fallback to sophisticated analysis if parsing fails
                return {
                    "action_needed": "sophisticated_analysis",
                    "analysis_type": "spending_patterns",
                    "reasoning": "Could not parse LLM reasoning, defaulting to sophisticated analysis"
                }
            
            logger.info(f"LLM Understanding: {reasoning.get('reasoning', '')}")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in LLM reasoning: {e}")
            # Check for transaction keywords as fallback
            if any(keyword in user_message.lower() for keyword in TransactionKeywords.KEYWORDS):
                return {
                    "action_needed": "transaction_history",
                    "analysis_type": "transaction_history",
                    "reasoning": "Error fallback detected transaction keywords"
                }
            
            return {
                "action_needed": "sophisticated_analysis", 
                "analysis_type": "spending_patterns",
                "reasoning": f"Error in LLM reasoning: {e}"
            }
        
    def _generate_fallback_pipeline(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate a basic pipeline when LLM fails."""
        match_stage = {"$match": {DatabaseFields.ACCOUNT_NUMBER: account_number}}
        
        if intent == BankingIntents.TRANSACTION_HISTORY:
            pipeline = [match_stage, {"$sort": {DatabaseFields.DATE: -1, "_id": -1}}]
            if filters.limit:
                pipeline.append({"$limit": filters.limit})
            return pipeline
        
        elif intent in [BankingIntents.SPENDING_ANALYSIS, BankingIntents.CATEGORY_SPENDING]:
            if filters.transaction_type:
                match_stage["$match"][DatabaseFields.TYPE] = filters.transaction_type
            
            if filters.description:
                match_stage["$match"][DatabaseFields.DESCRIPTION] = {
                    "$regex": filters.description,
                    "$options": "i"
                }
            
            if filters.category:
                match_stage["$match"][DatabaseFields.CATEGORY] = {
                    "$regex": filters.category,
                    "$options": "i"
                }
            
            pipeline = [
                match_stage,
                {
                    "$group": {
                        "_id": None,
                        "total_amount": {"$sum": f"${DatabaseFields.AMOUNT_DEDUCTED_FROM_ACCOUNT}"},
                        "currency": {"$first": f"${DatabaseFields.ACCOUNT_CURRENCY}"}
                    }
                }
            ]
            return pipeline
        
        return [match_stage, {"$sort": {DatabaseFields.DATE: -1, "_id": -1}}, {"$limit": Limits.DEFAULT_TRANSACTION_LIMIT}]

    async def _handle_transaction_history(self, user_message: str, account_number: str, 
                                         first_name: str, reasoning: Dict[str, Any], 
                                         memory: ConversationBufferMemory) -> str:
        """Handle transaction history requests with real database queries."""
        try:
            logger.info(f"Handling transaction history request: {user_message}")
            
            # Extract limit from the message using LLM
            limit_prompt = f"""
            Extract the number of transactions requested from this message: "{user_message}"
            
            Examples:
            - "last 10 transactions" -> 10
            - "recent 5 transactions" -> 5
            - "show me my transactions" -> {Limits.DEFAULT_TRANSACTION_LIMIT} (default)
            - "transaction history" -> {Limits.DEFAULT_TRANSACTION_LIMIT} (default)
            - "transactions in May" -> {Limits.MONTH_TRANSACTION_LIMIT} (show all for month)
            
            Return only the number as an integer. If no specific number mentioned, return {Limits.DEFAULT_TRANSACTION_LIMIT}.
            If asking for a specific month, return {Limits.MONTH_TRANSACTION_LIMIT}.
            """
            
            try:
                limit_response = await llm.ainvoke([SystemMessage(content=limit_prompt)])
                limit = int(limit_response.content.strip())
                if limit <= 0 or limit > Limits.MAX_TRANSACTION_LIMIT:
                    limit = Limits.DEFAULT_TRANSACTION_LIMIT
            except:
                limit = Limits.DEFAULT_TRANSACTION_LIMIT
            
            logger.info(f"Using limit: {limit} transactions")
            
            # Check if user specified a month
            month_filter = None
            user_message_lower = user_message.lower()
            for month_name, month_num in Months.NAMES_TO_NUMBERS.items():
                if month_name in user_message_lower:
                    month_filter = month_num
                    logger.info(f"Found month filter: {month_name} ({month_num})")
                    break
            
            # Build MongoDB query for transaction history
            query = {
                DatabaseFields.ACCOUNT_NUMBER: account_number
            }
            
            # Add month filter if specified
            if month_filter:
                current_year = datetime.now().year
                days_in_month = month_days(list(Months.NAMES_TO_NUMBERS.keys())[month_filter-1], current_year)
                
                query[DatabaseFields.DATE] = {
                    "$gte": datetime(current_year, month_filter, 1),
                    "$lte": datetime(current_year, month_filter, days_in_month, 23, 59, 59)
                }
                logger.info(f"Added date filter for month {month_filter}: {query[DatabaseFields.DATE]}")
            
            # Query the database directly using the collection
            transactions = list(self.collection.find(query).sort(DatabaseFields.DATE, -1).limit(limit))
            
            logger.info(f"Found {len(transactions)} transactions for account {account_number}")
            
            if not transactions:
                if month_filter:
                    month_name = list(Months.NAMES_TO_NUMBERS.keys())[month_filter-1].title()
                    context_state = f"No transactions found for {month_name} for user account"
                    data = {"transaction_count": 0, DatabaseFields.ACCOUNT_NUMBER: account_number, "month": month_name}
                else:
                    context_state = "No transaction history found for user account"
                    data = {"transaction_count": 0, DatabaseFields.ACCOUNT_NUMBER: account_number}
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # Format transactions for response - CONVERT DATETIME TO STRING
            formatted_transactions = []
            for tx in transactions:
                # Convert datetime to string to avoid JSON serialization error
                date_obj = tx.get(DatabaseFields.DATE)
                if isinstance(date_obj, datetime):
                    date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = str(date_obj) if date_obj else "Unknown"
                
                formatted_tx = {
                    DatabaseFields.DATE: date_str,
                    DatabaseFields.DESCRIPTION: tx.get(DatabaseFields.DESCRIPTION, ""),
                    DatabaseFields.CATEGORY: tx.get(DatabaseFields.CATEGORY, ""),
                    DatabaseFields.TYPE: tx.get(DatabaseFields.TYPE, ""),
                    "amount": tx.get(DatabaseFields.TRANSACTION_AMOUNT, 0),
                    "currency": tx.get(DatabaseFields.TRANSACTION_CURRENCY, Currencies.PKR_LOWER),
                    "balance": tx.get(DatabaseFields.ACCOUNT_BALANCE, 0)
                }
                formatted_transactions.append(formatted_tx)
            
            # Set context based on whether month filter was used
            if month_filter:
                month_name = list(Months.NAMES_TO_NUMBERS.keys())[month_filter-1].title()
                context_state = f"User requested transaction history for {month_name}, providing {len(transactions)} transactions"
            else:
                context_state = f"User requested transaction history, providing {len(transactions)} recent transactions"
            
            data = {
                "transactions": formatted_transactions,
                "transaction_count": len(transactions),
                "requested_limit": limit,
                DatabaseFields.ACCOUNT_NUMBER: account_number,
                "month_filter": list(Months.NAMES_TO_NUMBERS.keys())[month_filter-1].title() if month_filter else None
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Transaction history error: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _handle_sophisticated_analysis(self, user_message: str, account_number: str, 
                                           first_name: str, reasoning: Dict[str, Any], 
                                           memory: ConversationBufferMemory) -> str:
        """Handle sophisticated financial analysis with real transaction data."""
        try:
            analysis_type = reasoning.get("analysis_type", "spending_patterns")
            
            logger.info(f"Handling sophisticated analysis: {analysis_type} for query: {user_message}")
            
            if analysis_type == "transaction_history":
                return await self._handle_transaction_history(user_message, account_number, first_name, reasoning, memory)
            elif analysis_type == "category_analysis":
                return await self._analyze_category_spending(user_message, account_number, first_name, reasoning, memory)
            elif analysis_type == "monthly_comparison":
                return await self._analyze_monthly_comparison(user_message, account_number, first_name, reasoning, memory)
            elif analysis_type == "savings_planning":
                return await self._analyze_savings_planning(user_message, account_number, first_name, reasoning, memory)
            elif analysis_type == "spending_breakdown":
                logger.info(f"Routing to spending breakdown analysis")
                return await self._analyze_spending_breakdown(user_message, account_number, first_name, reasoning, memory)
            else:
                logger.info(f"Defaulting to spending patterns analysis")
                return await self._analyze_spending_patterns(user_message, account_number, first_name, reasoning, memory)
                
        except Exception as e:
            logger.error(f"Error in sophisticated analysis: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _analyze_spending_breakdown(self, user_message: str, account_number: str, 
                                     first_name: str, reasoning: Dict[str, Any], 
                                     memory: ConversationBufferMemory) -> str:
        """Analyze what drove high spending in a specific month."""
        try:
            logger.info(f"Starting spending breakdown analysis for: {user_message}")
            
            # Extract month and year from user message or conversation
            target_month = None
            target_year = datetime.now().year
            
            # Check conversation history for month context
            chat_history = memory.chat_memory.messages
            
            # Look in recent messages for month mentions
            for msg in reversed(chat_history[-Limits.MAX_RECENT_MESSAGES:] if len(chat_history) > Limits.MAX_RECENT_MESSAGES else chat_history):
                content = msg.content.lower()
                for month_name, month_num in Months.NAMES_TO_NUMBERS.items():
                    if month_name in content:
                        target_month = month_num
                        logger.info(f"Found month {month_name} ({month_num}) in conversation history")
                        break
                if target_month:
                    break
            
            # If still no month found, default to previous month
            if target_month is None:
                current_date = datetime.now()
                target_month = current_date.month - 1 if current_date.month > 1 else 12
                target_year = target_year if current_date.month > 1 else target_year - 1
                logger.info(f"No month found, defaulting to month {target_month}")
            
            month_name = datetime(target_year, target_month, 1).strftime("%B")
            
            # Get days in month
            days_in_month = Months.DAYS_IN_MONTH.get(target_month, 31)
            if target_month in Months.THIRTY_DAY_MONTHS:
                days_in_month = 30
            elif target_month == 2:
                days_in_month = 29 if target_year % 4 == 0 else 28
            
            # Query all transactions for the target month
            start_date = datetime(target_year, target_month, 1)
            end_date = datetime(target_year, target_month, days_in_month, 23, 59, 59)
            
            query = {
                DatabaseFields.ACCOUNT_NUMBER: account_number,
                DatabaseFields.TYPE: TransactionTypes.DEBIT,
                DatabaseFields.DATE: {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            logger.info(f"Querying transactions for {month_name} {target_year}: {start_date} to {end_date}")
            
            transactions = list(self.collection.find(query))
            logger.info(f"Found {len(transactions)} transactions")
            
            if not transactions:
                context_state = f"No spending data found for {month_name}, explaining this naturally"
                data = {"month": month_name, "year": target_year, "total_spent": 0}
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # Calculate total spending and group by category
            total_spent = 0
            category_spending = {}
            
            for tx in transactions:
                # Use the correct field based on database schema
                amount = tx.get(DatabaseFields.AMOUNT_DEDUCTED_FROM_ACCOUNT, 0)
                if amount == 0:  # Fallback to alternative field name
                    amount = tx.get(DatabaseFields.TRANSACTION_AMOUNT, 0)
                
                total_spent += amount
                
                category = tx.get(DatabaseFields.CATEGORY, "Other")
                if not category or category.strip() == "":
                    category = "Other"
                
                if category not in category_spending:
                    category_spending[category] = {
                        "total": 0, 
                        "count": 0, 
                        "percentage": 0,
                        "largest_transaction": 0,
                        "description": ""
                    }
                
                category_spending[category]["total"] += amount
                category_spending[category]["count"] += 1
                
                # Track largest transaction in this category
                if amount > category_spending[category]["largest_transaction"]:
                    category_spending[category]["largest_transaction"] = amount
                    category_spending[category]["description"] = tx.get(DatabaseFields.DESCRIPTION, "")
            
            # Calculate percentages
            for category in category_spending:
                if total_spent > 0:
                    category_spending[category]["percentage"] = (category_spending[category]["total"] / total_spent) * 100
            
            # Sort categories by spending amount (highest first)
            sorted_categories = sorted(category_spending.items(), key=lambda x: x[1]["total"], reverse=True)
            
            logger.info(f"Spending breakdown complete: {total_spent} total, {len(sorted_categories)} categories")
            
            # Prepare data for natural response
            context_state = f"User asked what caused high spending in {month_name}, providing detailed breakdown with real transaction data by category"
            data = {
                "month": month_name,
                "year": target_year,
                "total_spent": total_spent,
                "category_breakdown": sorted_categories,
                "top_categories": sorted_categories[:5],  # Top 5 spending categories
                "transaction_count": len(transactions),
                "currency": transactions[0].get(DatabaseFields.TRANSACTION_CURRENCY, Currencies.USD) if transactions else Currencies.USD
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Spending breakdown error: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _handle_balance_inquiry(self, account_number: str, first_name: str, user_message: str, memory: ConversationBufferMemory) -> str:
        """Enhanced balance inquiry with natural responses."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/user_balance",
                    json={DatabaseFields.ACCOUNT_NUMBER: account_number}
                )
                response.raise_for_status()
                data = response.json()
                
                # Check if this is a savings goal query
                if "target" in user_message.lower() or "goal" in user_message.lower() or "1000000" in user_message or "million" in user_message:
                    context_state = "User asking about balance in context of savings goal"
                    
                    # Extract target amount
                    import re
                    amount_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', user_message.replace(',', ''))
                    target_amount = float(amount_match.group(1).replace(',', '')) if amount_match else 1000000
                    
                    balance_info = data.get("user", {})
                    current_balance = balance_info.get("current_balance_usd", 0) or balance_info.get("current_balance_pkr", 0)
                    needed = target_amount - current_balance
                    
                    goal_data = {
                        "current_balance": current_balance,
                        "target_amount": target_amount,
                        "amount_needed": needed,
                        "has_enough": needed <= 0
                    }
                    
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    response_text = await self.generate_natural_response(context_state, goal_data, user_message, first_name, conversation_history)
                else:
                    # Regular balance inquiry
                    context_state = ContextStates.BALANCE_INQUIRY
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    response_text = await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                
                return response_text
                
        except Exception as e:
            logger.error({"action": "handle_balance_inquiry", "error": str(e)})
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    # Add missing methods with stubs
    async def _analyze_category_spending(self, user_message: str, account_number: str, first_name: str, reasoning: Dict[str, Any], memory: ConversationBufferMemory) -> str:
        """Analyze spending by category."""
        context_state = ContextStates.SPENDING_ANALYSIS
        return await self.generate_natural_response(context_state, {"analysis": "category_spending"}, user_message, first_name)

    async def _analyze_monthly_comparison(self, user_message: str, account_number: str, first_name: str, reasoning: Dict[str, Any], memory: ConversationBufferMemory) -> str:
        """Analyze monthly spending comparison."""
        context_state = ContextStates.SPENDING_ANALYSIS
        return await self.generate_natural_response(context_state, {"analysis": "monthly_comparison"}, user_message, first_name)

    async def _analyze_savings_planning(self, user_message: str, account_number: str, first_name: str, reasoning: Dict[str, Any], memory: ConversationBufferMemory) -> str:
        """Analyze savings planning."""
        context_state = ContextStates.SPENDING_ANALYSIS
        return await self.generate_natural_response(context_state, {"analysis": "savings_planning"}, user_message, first_name)

    async def _analyze_spending_patterns(self, user_message: str, account_number: str, first_name: str, reasoning: Dict[str, Any], memory: ConversationBufferMemory) -> str:
        """Analyze general spending patterns."""
        context_state = ContextStates.SPENDING_ANALYSIS
        return await self.generate_natural_response(context_state, {"analysis": "spending_patterns"}, user_message, first_name)

    async def handle_error_gracefully(self, error: Exception, user_message: str, first_name: str, context: str) -> str:
        """Handle errors gracefully with natural response."""
        context_state = ContextStates.ERROR_OCCURRED
        data = {"error": str(error), "context": context}
        return await self.generate_natural_response(context_state, data, user_message, first_name)


class BankingSession:
    """Session manager for banking interactions."""
    
    def __init__(self, ai_agent: BankingAIAgent):
        self.ai_agent = ai_agent
        self.current_user = None
        self.current_account = None
        self.session_active = False
    
    async def start_session(self, user_data: Dict) -> str:
        """Start a new banking session."""
        self.current_user = user_data
        self.session_active = True
        
        first_name = user_data.get("first_name", "")
        last_name = user_data.get("last_name", "")
        
        return await self.ai_agent.handle_session_start(first_name, last_name)
    
    async def set_active_account(self, account_number: str) -> None:
        """Set the active account for the session."""
        self.current_account = account_number
    
    async def process_message(self, message: str) -> str:
        """Process a user message within the session context."""
        if not self.session_active or not self.current_user or not self.current_account:
            return "Session not properly initialized. Please restart your banking session."
        
        first_name = self.current_user.get("first_name", "")
        
        # Check for exit command
        if message.lower().strip() in ExitCommands.COMMANDS:
            response = await self.ai_agent.handle_session_end(self.current_account, first_name)
            self.end_session()
            return response
        
        # Process regular message
        return await self.ai_agent.process_query(message, self.current_account, first_name)
    
    def end_session(self):
        """End the current session."""
        if self.current_account:
            self.ai_agent.clear_user_memory(self.current_account)
        
        self.current_user = None
        self.current_account = None
        self.session_active = False
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.session_active