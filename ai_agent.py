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
        """Get an enhanced summary of recent conversation for context."""
        if not chat_history:
            return "No previous conversation."
        
        # Get last messages for better context
        recent_messages = chat_history[-Limits.MAX_RECENT_MESSAGES:] if len(chat_history) > Limits.MAX_RECENT_MESSAGES else chat_history
        context_lines = []
        
        for i, msg in enumerate(recent_messages):
            speaker = "Human" if i % 2 == 0 else "Assistant"
            content = msg.content
            
            # Keep more content for better context but limit extremely long responses
            if len(content) > Limits.MAX_RESPONSE_CONTENT:
                content = content[:Limits.MAX_RESPONSE_CONTENT] + "..."
            
            context_lines.append(f"{speaker}: {content}")
        
        full_context = "\n".join(context_lines)
        
        # If context is very long, summarize it
        if len(full_context) > Limits.MAX_CONTEXT_LENGTH:
            try:
                summary_prompt = f"""
        Summarize this banking conversation context in 5-6 sentences, preserving key banking data:

        {full_context}

        Focus on: transaction amounts, spending categories, timeframes, specific entities and any other important data shown to the user.
        """
                response = llm.invoke([SystemMessage(content=summary_prompt)])
                return response.content.strip()
            except:
                return full_context[:Limits.MAX_CONTEXT_LENGTH] + "..."
        
        return full_context

    def resolve_contextual_query(self, user_message: str, conversation_history: str) -> str:
        """Convert contextual queries into standalone queries using conversation history."""
        try:
            if not conversation_history or len(conversation_history.strip()) < 10:
                logger.info(f"🔧 CONTEXTUAL: No sufficient history for: '{user_message}'")
                return user_message  # No context to work with
            
            # Log the conversation history being used
            logger.info(f"🔧 CONTEXTUAL INPUT: Message='{user_message}', History='{conversation_history[:200]}...'")
            
            contextual_resolution_prompt = f"""
            You are a query resolver. Convert contextual queries into standalone, complete queries using conversation history.

            CONVERSATION HISTORY:
            {conversation_history}

            CURRENT CONTEXTUAL QUERY: "{user_message}"

            Your task: If the current query is contextual (refers to previous data), create a standalone query that includes all necessary context.

            EXAMPLES:

            History: "Here are your last 4 transactions: 1. Hamza transfer $35, 2. Grocery Store $77.23, 3. Car Loan $350.69, 4. Withdrawal $117.19"
            Query: "which one is most expensive" 
            → "show me the most expensive transaction from my recent transaction history"

            History: "You spent $77.23 at Grocery Store on June 29"
            Query: "mujhey inka total kite batao in usd"
            → "what is my total spending amount in USD"

            History: "Your June spending: Food $100, Travel $200, Shopping $150"
            Query: "food ka kitna tha"
            → "how much did I spend on food in June"

            History: "Your balance is $4023.90"
            Query: "kya main 5000 afford kar sakta hun"
            → "can I afford $5000 based on my current balance"

            History: "Your last 4 transactions: Foodpanda $14.01, Grocery Store $53.44, Car Loan $350.69, Withdrawal $117.19"
            Query: "mujhey inka total kite batao in usd"
            → "what is the total amount of my recent transactions in USD"

            # ADD THESE NEW EXAMPLES FOR BALANCE TRANSFERS:
            History: "Here's your current account balance: USD 1,554.41 as of July 29, 2025"
            Query: "ok transfer 1% of that"
            → "transfer 1% of 1554.41 USD which is 15.54 USD"

            History: "Account Balance: PKR 245,600.00 As of: 29th July 2025"
            Query: "transfer 2% of that to john"
            → "transfer 2% of 245600 PKR which is 4912 PKR to john"

            History: "Your current balance is $2,341.50"
            Query: "send 5% of that amount"
            → "send 5% of 2341.50 USD which is 117.08 USD"

            RULES:
            1. If the query is already standalone (contains complete context), return it unchanged
            2. If the query references previous data ("which one", "inka total", "that transaction", "that balance", "that amount"), resolve it with context
            3. Preserve the user's intent while making it standalone
            4. Convert to English if needed for processing
            5. Include relevant timeframes, amounts, or categories from history
            6. For percentage calculations, calculate the actual amount and include both percentage and calculated amount
            7. When "that" refers to a balance, extract the exact amount and calculate percentages

            RESOLVED STANDALONE QUERY:
            """
            
            response = llm.invoke([SystemMessage(content=contextual_resolution_prompt)])
            resolved_query = response.content.strip()
            
            # Remove any quotes or extra formatting
            if resolved_query.startswith('"') and resolved_query.endswith('"'):
                resolved_query = resolved_query[1:-1]
            
            # Enhanced logging (ADDED)
            if resolved_query != user_message:
                logger.info(f"🔧 CONTEXTUAL SUCCESS: '{user_message}' → '{resolved_query}'")
            else:
                logger.info(f"🔧 CONTEXTUAL: No resolution needed for '{user_message}'")
            
            # Keep original logging line too
            logger.info(f"Contextual query resolved: '{user_message}' → '{resolved_query}'")
            return resolved_query
            
        except Exception as e:
            logger.error(f"🔧 CONTEXTUAL ERROR: {e} for message '{user_message}'")  # Enhanced
            logger.error(f"Error in contextual query resolution: {e}")  # Original
            return user_message  # Return original query if resolution fails


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
        """Handle initial user greeting and ask for CNIC verification."""
        try:
            greeting_prompt = """You are Sage, a friendly banking assistant. A user just greeted you (said hi, hello, etc.) to start a new banking session.

        Your task:
        1. Greet them warmly and introduce yourself as their banking assistant
        2. Explain that you can help with their banking needs
        3. Ask them to provide their CNIC for secure verification
        4. Mention the CNIC format (12345-1234567-1) 
        5. Keep it friendly and welcoming

        Generate a natural, welcoming response that guides them to the next step."""

            response = await llm.ainvoke([SystemMessage(content=greeting_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating initial greeting: {e}")
            return "Hello! I'm Sage, your banking assistant. I'm here to help you with all your banking needs. To get started securely, could you please provide your CNIC in the format 12345-1234567-1?"
        
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
        """Handle OTP request after CNIC verification."""
        try:
            otp_prompt = f"""You are Sage, a banking assistant. A user has just had their CNIC verified successfully and now needs to provide an OTP for additional security.

        Your task:
        1. Explain that for additional security, they need to provide an OTP
        2. Tell them the OTP is a number between {Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits sent to your mobile phone
        3. Ask them to enter their OTP
        4. Keep it simple and secure-sounding

        Generate a natural, security-focused response asking for OTP."""

            response = await llm.ainvoke([SystemMessage(content=otp_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating OTP request: {e}")
            return f"Great! For additional security, {first_name}, please provide an OTP. You should have received an OTP ({Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits) on your mobile phone."

    async def handle_otp_success(self, user_name: str, accounts: List[str]) -> str:
        """Handle successful OTP verification - should only ask for account selection."""
        try:
            first_name = user_name.split()[0]
            
            # Format accounts for selection
            account_list = []
            for i, account in enumerate(accounts, 1):
                account_list.append(f"{i}. {account}")
            
            accounts_text = "\n".join(account_list)
            
            return f"Hello {first_name}! OTP verified successfully.\n\nPlease select your account:\n{accounts_text}\n\nYou can select by typing the account number, last 4 digits, or saying 'first account', 'second account', etc."
            
        except Exception as e:
            logger.error(f"Error in OTP success: {e}")
            return "OTP verified successfully! Now please select your account by entering the last 4 digits."
        
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
        """Handle successful CNIC verification with natural response."""
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
            
            return await self.generate_natural_response(context_state, data, cnic, user_name.split()[0])
            
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
        """Handle account selection confirmation with natural response."""
        try:
            first_name = user_name.split()[0]
            masked_account = f"***-***-{account_number[-4:]}"
            
            return f"""Hello {first_name}! Account confirmed! Welcome, you're now connected to account {masked_account}.

    I can help you with:
    - Check your account balance
    - View transaction history  
    - Analyze your spending patterns
    - Transfer money to others
    - Financial planning assistance

    What can I help you with today?"""
            
        except Exception as e:
            logger.error(f"Error in account confirmation: {e}")
            return f"Hello! Account confirmed successfully. What can I help you with today?"
        
            
    # === QUERY PIPELINE FLOW ===
    def detect_intent_from_filters(self, user_message: str, filters: FilterExtraction) -> str:
        """Detect intent using LLM for more flexible understanding."""
        try:
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
                    return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                else:
                    # Use contextual banking response for other queries
                    return await self.generate_contextual_banking_response(data, user_message, first_name, memory, intent) 
                
        except Exception as e:
            logger.error(f"Error executing LLM pipeline: {e}")
            context_state = ContextStates.ERROR_OCCURRED
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

   

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

        # CRITICAL FIX: Resolve contextual queries FIRST, before non-banking detection
        original_message = user_message
        resolved_query = self.resolve_contextual_query(user_message, conversation_history)
        
        if resolved_query != original_message:
            logger.info(f"🔧 CONTEXTUAL RESOLUTION: '{original_message}' → '{resolved_query}'")
            processing_message = resolved_query
        else:
            processing_message = user_message

        # NOW check for non-banking queries using the RESOLVED query
        if self.is_clearly_non_banking_query(processing_message, conversation_history):
            logger.info(f"🚫 STRICT filter blocked non-banking query: {processing_message} (original: {original_message})")
            response = await self.handle_non_banking_query(original_message, first_name)
            memory.chat_memory.add_user_message(original_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # Handle greetings and simple queries
        if self._is_simple_greeting_or_general(user_message):
            context_state = "User sent a greeting or general question, no specific banking data needed"
            response = await self.generate_natural_response(context_state, None, user_message, first_name, conversation_history)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response
        
        # Check for currency conversion requests
        if self.detect_currency_conversion_intent(user_message, conversation_history):
            logger.info(f"Currency conversion intent detected for: {user_message}")
            return await self.handle_currency_conversion(user_message, conversation_history, first_name, memory)

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
                
    # === HANDLE MONEY TRANSFERS ===
    async def handle_transfer_otp_request(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle OTP request for money transfer."""
        try:
            context_state = ContextStates.TRANSFER_DETAILS_COLLECTED
            data = {
                "transfer_amount": amount,
                "transfer_currency": currency,
                "transfer_recipient": recipient,
                "otp_required": True
            }
            
            return await self.generate_natural_response(context_state, data, f"transfer {amount} {currency} to {recipient}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer OTP request: {e}")
            return f"To complete the transfer of {amount} {currency} to {recipient}, please provide an OTP (any number between {Limits.OTP_MIN_DIGITS}-{Limits.OTP_MAX_DIGITS} digits) for security verification."

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

            # Enhanced transfer prompt with context
            enhanced_transfer_prompt = f"""
                    Extract transfer details from the query, using conversation history for context. Handle multi-turn transfer conversations.

                    CONVERSATION HISTORY:
                    {conversation_history}

                    CURRENT TRANSFER REQUEST: "{user_message}"

                    Rules:
                    - If message mentions percentages of "that" or "it", calculate based on amounts in conversation history
                    - Extract exact amount, currency, and recipient
                    - For "1% of that $1,554.41" → amount should be 15.54, currency USD
                    - For "transfer 10% of that PKR amount" → calculate 10% of the PKR amount mentioned
                    - If only recipient is provided ("to ahmed abrar"), look for amount/percentage in recent conversation
                    - If only amount is provided, look for recipient in recent conversation
                    - Combine information from multiple recent messages to complete transfer details

                    Enhanced Examples:
                    History: "Here's your current account balance: USD 1,554.41"
                    Current: "ok transfer 1% of that" → {{"amount": 15.54, "currency": "USD", "recipient": null}}

                    History: "transfer 1% of 1554.41 USD which is 15.54 USD"  
                    Current: "to ahmed abrar" → {{"amount": 15.54, "currency": "USD", "recipient": "ahmed abrar"}}

                    History: "send 100 USD"
                    Current: "to john smith" → {{"amount": 100, "currency": "USD", "recipient": "john smith"}}

                    History: "Assistant: I need more details. User: transfer 50 PKR"
                    Current: "to sarah" → {{"amount": 50, "currency": "PKR", "recipient": "sarah"}}

                    Multi-turn completion rules:
                    - Look at last 3-4 conversation turns for missing transfer details
                    - If current message only has recipient, search history for amount
                    - If current message only has amount, search history for recipient
                    - Prioritize most recent complete transfer attempt

                    Return JSON: {{"amount": number, "currency": string, "recipient": string}}
                    Set null for missing fields, but try to complete from conversation history first.
                    """
            logger.info(f"🔍 TRANSFER DEBUG - Using enhanced prompt for: {user_message}")

            response = await llm.ainvoke([SystemMessage(content=enhanced_transfer_prompt)])

            logger.info(f"🔍 TRANSFER DEBUG - LLM response: {response.content}") 

            transfer_details = self.extract_json_from_response(response.content)
            transfer_details = self.extract_json_from_response(response.content)

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
            # The webhook will handle the OTP flow
            return f"{TransferSignals.OTP_REQUIRED_PREFIX}{TransferSignals.SEPARATOR}{transfer_details['amount']}{TransferSignals.SEPARATOR}{transfer_details.get('currency', Currencies.PKR)}{TransferSignals.SEPARATOR}{transfer_details['recipient']}"
                
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
                
                context_state = ContextStates.TRANSFER_EXECUTED_SUCCESS if transfer_result.get("status") == StatusMessages.SUCCESS else ContextStates.TRANSFER_EXECUTED_FAILED
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                
                return await self.generate_natural_response(context_state, transfer_result, f"transfer {amount} {currency} to {recipient}", first_name, conversation_history)
                
        except Exception as e:
            logger.error(f"Error executing verified transfer: {e}")
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
            
            # Map LLM decision to formatting instructions
            format_instructions = {
                ResponseFormats.CONCISE_ONE_LINER: """
            FORMAT: CONCISE ONE-LINER
            - Give a direct, single sentence answer with the specific number
            - Reference previous context naturally but keep it brief
            - No bullet points or lists needed
            - Example: "Your current balance is $1,234.56" or "You spent $45.20 on Netflix in April"
            """,
                        ResponseFormats.STRUCTURED_LIST: """
            FORMAT: STRUCTURED DATA PRESENTATION
            - Use bullet points (•) or numbering for multiple items/transactions
            - Organize data clearly with proper formatting
            - Include all relevant details, amounts, dates, and descriptions
            - Group related information logically
            - End with a helpful summary line if appropriate
            - Preserve ALL data - never omit important details
            """,
                        ResponseFormats.DETAILED_EXPLANATION: """
            FORMAT: DETAILED EXPLANATION WITH CONTEXT
            - Provide thorough explanation with context
            - Include all relevant numbers and comparisons
            - Use paragraphs and logical flow
            - Reference previous conversation context
            - Help user understand patterns or insights
            - Structure with natural breaks between topics
            """,
                        ResponseFormats.HELPFUL_GUIDANCE: """
            FORMAT: HELPFUL GUIDANCE
            - Provide clear, supportive guidance
            - Explain what's needed or what went wrong
            - Offer specific next steps
            - Keep it friendly and helpful
            - Include examples if relevant
            """
            }
            
            return format_instructions.get(format_type, format_instructions[ResponseFormats.DETAILED_EXPLANATION])
            
        except Exception as e:
            logger.error(f"Error in LLM format analysis: {e}")
            # Fallback to default detailed format
            return """
        FORMAT: CONVERSATIONAL RESPONSE
        - Provide appropriate detail based on the data available
        - Reference context naturally
        - Include all important information
        - Make it helpful and clear
        """

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

    # UPDATE ai_agent.py - Replace your generate_natural_response method (NO EMOJIS)

    async def generate_natural_response(self, context_state: str, data: Any, user_message: str, first_name: str, conversation_history: str = "") -> str:
        """Generate contextual LLM responses with ChatGPT-style direct, structured formatting (no emojis)."""
        
        # Special handling for non-banking queries (keep existing)
        if "non-banking question" in context_state or data and data.get("query_type") == "non_banking":
            non_banking_prompt = f"""You are Sage, a banking assistant. The user {first_name} just asked a non-banking question that you must politely but firmly decline to answer.

            Their question: "{user_message}"

            Response rules:
            1. Be polite but firm - you ONLY handle banking queries
            2. Don't apologize excessively - be confident in your banking focus
            3. Clearly state you cannot help with non-banking topics
            4. Redirect to specific banking services you can help with
            5. Keep it concise and professional

            Generate a response that maintains boundaries while being helpful about banking topics."""
            
            try:
                response = await llm.ainvoke([SystemMessage(content=non_banking_prompt)])
                return response.content.strip()
            except:
                return f"I'm a banking assistant, {first_name}, and I can only help with your account-related questions like checking balances, viewing transactions, analyzing spending, or transferring money. I don't have information about topics outside of banking. What banking question can I help you with today?"

        # Use LLM to determine response format for banking queries (keep existing)
        response_format_instruction = await self._determine_response_format_with_llm(user_message, data, context_state)
        
        # ENHANCED system prompt - CHATGPT DIRECT STYLE (NO ASTERISKS)
        system_prompt = f"""You are Sage, a professional banking assistant. Generate responses in the exact ChatGPT banking style: direct, structured, and to-the-point. NO asterisks (*) anywhere.

        CRITICAL RESPONSE FORMAT:
        **FIRST LINE MUST ALWAYS BE:** "Hello [FirstName]! [brief context-appropriate greeting]"

        **RESPONSE FORMAT RULES (CRITICAL):**
        {response_format_instruction}

        **CHATGPT-STYLE FORMATTING RULES (MANDATORY - NO ASTERISKS):**

        **TRANSACTION HISTORY (DIRECT BULLET POINT STYLE):**
        - After greeting, header: "Here are your last [X] transactions:"
        - Use bullet points (•) for each transaction
        - Format: "• [Date] | [Description] | [Type] | [Amount] [Currency] | Balance: [Balance]"
        - NO narrative explanation of each transaction
        - End with: "Let me know if you'd like a filter (e.g., only credits, only food-related) or if you want to send money, download a statement, or something else."

        **BALANCE RESPONSES (DIRECT STYLE):**
        - After greeting: "Account Balance: [Currency] [Amount] As of: [Date]"
        - Add helpful follow-up: "Is there anything else I can help you with?"

        **VERIFICATION RESPONSES (STRUCTURED):**
        - After greeting: "Verification successful. Thank you, your identity has been verified."
        - "Please answer any two of the following questions:"
        - Number the verification steps clearly

        **TRANSFER RESPONSES (DIRECT CONFIRMATION):**
        - After greeting: "Success! You have successfully transferred [Currency] [Amount] to [Recipient]."
        - "Reference: [reference] Date: [date] Updated Balance: [Currency] [Amount]"
        - "Would you like to do anything else? (View transactions, check spending, send more money?)"

        **ACCOUNT CONFIRMATION (STRUCTURED):**
        - After greeting: "Account confirmed! Welcome [name], you're now connected to account [masked_account]"
        - List services with bullet points
        - "What can I help you with today?"

        **SPENDING ANALYSIS (STRUCTURED DATA):**
        - After greeting, use clear headers with bold formatting
        - Present data in structured format with bullet points
        - Include totals and percentages where relevant
        - No long explanations, just the data

        **ERROR HANDLING (PROFESSIONAL & BRIEF):**
        - After greeting: "Error: [Brief error explanation]"
        - Provide clear next steps
        - Keep it concise and professional

        **GENERAL STYLE RULES:**
        1. **ALWAYS START WITH GREETING:** "Hello [FirstName]! [context]"
        2. **Be Direct**: Don't explain every detail, just present the data
        3. **Use Structure**: Bullet points, clear headers, organized layout
        4. **Bold Important Info**: Use **bold** for amounts, names, status
        5. **End with Options**: Always provide helpful follow-up choices
        6. **No Narrative**: Don't tell a story, just show the information
        7. **NO ASTERISKS**: Use only text and **bold** formatting - NEVER use * symbols
        8. **Consistent Format**: Same structure for similar types of responses

        CURRENT CONTEXT: {context_state}
        USER'S MESSAGE: "{user_message}"
        AVAILABLE DATA: {json.dumps(data) if data else "No specific data"}

        CONVERSATION HISTORY (Your memory):
        {conversation_history}

        CONTEXTUAL RESPONSE RULES (CRITICAL):
            1. **Reference Previous Data**: If you previously showed transactions, balances, or spending - reference them specifically
            - "Looking at those 5 transactions I showed you..."
            - "From your June spending that we just discussed..."
            - "Referring back to your balance of $2,341..."

            2. **Build on Previous Context**: Make it feel like a continuous conversation
            - "Now looking at that data..." 
            - "Based on what we just saw..."
            - "Following up on those transactions..."

            3. **Use Specific Numbers/Details**: Reference exact amounts, dates, descriptions from previous messages
            - Instead of: "Your spending was high"
            - Say: "Your $550 spending on groceries that I mentioned"

            4. **Natural Conversation Flow**: 
            - If user asks follow-up questions, acknowledge the connection
            - If showing new data, relate it to previous context when relevant
            - Make responses feel like you remember everything

            5. **Contextual Greetings**: 
            - Don't always greet the same way
            - Sometimes skip greetings for follow-ups: "That most expensive transaction was..."
            - For continuing conversations: "{first_name}, looking at that data..."

            6. **Balance Query Responses**:
            - If context mentions "balance at specific date", focus on the date and balance amount
            - If context mentions "average balance", explain the calculation period
            - Always include currency information for balance responses
            - Reference the specific date or period requested

            PERSONALITY GUIDELINES:
            - Be conversational like you're having an ongoing chat
            - Show you remember previous parts of conversation
            - Reference specific data points naturally
            - Make each response build on the previous conversation
            - Avoid repetitive patterns - vary your language

            Generate a contextual response that feels like a natural continuation of your ongoing conversation with {first_name}."""
        
        try:
            response = await llm.ainvoke([SystemMessage(content=system_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return f"I'm having some technical difficulties right now, {first_name}. Could you try that again?"


    # ALSO UPDATE your generate_contextual_banking_response method (NO EMOJIS):

    async def generate_contextual_banking_response(self, query_result: Any, user_message: str, first_name: str, memory: ConversationBufferMemory, intent: str) -> str:
        """Generate banking responses with ChatGPT-style direct, structured formatting (no emojis)."""
        
        conversation_history = self._get_context_summary(memory.chat_memory.messages)
        
        # Enhanced banking context prompt with CHATGPT DIRECT STYLE (NO EMOJIS)
        banking_context_prompt = f"""You are Sage, a professional banking assistant. Generate responses in ChatGPT banking style: direct, structured, to-the-point. DO NOT use any emojis.

        
            
            **CHATGPT-STYLE FORMATTING (MANDATORY - NO EMOJIS):**

            **TRANSACTION LISTS (DIRECT BULLET STYLE):**
            Header: "Here are your **last [X] transactions**:"
            Format: "• [Date] | [Description] | [Type] | [Amount] [Currency] | Balance: [Balance]"
            Footer: "Let me know if you'd like a filter or if you want to send money, download a statement, or something else."

            **BALANCE FORMAT:** 
            "**Account Balance:** PKR **245,600.00** **As of:** **29th July 2025**"

            **TRANSFER FORMAT:** 
            "**Success!** You have successfully transferred **PKR 1,000** to **Ali Raza**"

            **VERIFICATION FORMAT:** 
            "**Verification successful.** Thank you, your identity has been verified."

            **SPENDING ANALYSIS:** 
            Use structured data presentation with clear headers and bullet points

            CONVERSATION HISTORY: {conversation_history}
            USER'S REQUEST: "{user_message}"
            INTENT: {intent}
            QUERY RESULTS: {json.dumps(query_result) if query_result else "No data"}

            RESPONSE RULES:
            1. **Be Direct**: Present data clearly without narrative explanation
            2. **Use Structure**: Bullet points, clear headers, organized layout
            3. **Reference Context**: Connect to previous conversation when relevant
            4. **End with Options**: Provide helpful follow-up choices
            5. **ChatGPT Style**: Professional, structured, to-the-point
            6. **NO EMOJIS**: Use only text and **bold** formatting

            Generate a direct, structured response that presents the banking information clearly in ChatGPT style. DO NOT use any emojis."""

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
        """Detect clearly non-banking queries and block them."""
        try:
            # CRITICAL: Don't block CNIC numbers - they are essential for banking authentication
            if re.match(r'^\d{5}-\d{7}-\d{1}$', user_message.strip()):
                logger.info(f"✅ CNIC format detected, allowing: '{user_message}'")
                return False
                
            # Don't block other number patterns that could be banking related
            if re.match(r'^\d+$', user_message.strip()) or re.match(r'^\d{4,}$', user_message.strip()):
                logger.info(f"✅ Number pattern detected, allowing: '{user_message}'")
                return False
                
            non_banking_detection_prompt = f"""
                You are a STRICT banking query filter. Your job is to BLOCK any query that is NOT directly related to banking operations.

                User message: "{user_message}"
                Recent conversation history: {conversation_history[:1500]}

                CONTEXT-AWARE ANALYSIS:
                - Check if recent conversation mentions transfers, amounts, or banking operations
                - If recent context shows transfer in progress, allow recipient names and transfer-related follow-ups
                - If recent context shows balance inquiry, allow transfer requests referring to "that amount"

                ALLOWED BANKING QUERIES ONLY:
                - Account balance checks ("balance", "current balance", "how much money")
                - Transaction history and details
                - Money transfers and payments (including recipient names in transfer context)
                - Spending analysis and budgets
                - Currency conversions of banking amounts
                - Account management questions
                - CNIC verification requests (format: 12345-1234567-1)
                - Banking authentication steps (OTPs, account numbers)
                - Account numbers, CNIC numbers, banking credentials
                - Follow-up messages in banking context (recipient names after transfer requests)

                SPECIAL TRANSFER CONTEXT RULES:
                - If conversation history mentions "transfer", "send money", or percentages → allow recipient names
                - If history shows balance inquiry → allow "transfer X% of that"  
                - If history shows incomplete transfer → allow "to [person name]"
                - Names like "ahmed", "john", "sarah" are ALLOWED if recent context is about transfers

                BLOCK EVERYTHING ELSE (return "YES" to block):
                - Entertainment: "Netflix movies", "TV shows", "music recommendations"
                - Job applications: "I want a job at Amazon"
                - General knowledge: "Who is the CEO of Apple", "What is the price of X"
                - Weather, sports, entertainment: "Weather today", "Football scores"
                - Technology: "How does AI work", "What is Python"
                - Shopping: "Where to buy X", "Price of Y"
                - Health/Medical advice
                - Travel information
                - News and current events
                - Personal relationships (unless in transfer context)
                - Education questions
                - Cooking/recipes
                - Movie/TV recommendations
                - ANY topic not directly about the user's bank account

                CRITICAL RULES:
                1. If recent conversation history contains transfer/balance context → be MORE LENIENT for follow-ups
                2. NEVER block CNIC numbers (format: 12345-1234567-1) - these are banking credentials
                3. NEVER block account numbers, OTPs, or banking authentication data
                4. In transfer context, allow recipient names and amounts
                5. When in doubt about banking vs non-banking in transfer context, ALLOW (return "NO")
                6. Check last 2-3 conversation turns for banking context before blocking

                Return "YES" to BLOCK non-banking queries, "NO" for genuine banking queries and banking follow-ups.
                """


            response = self.llm.invoke([SystemMessage(content=non_banking_detection_prompt)])
            result = response.content.strip().upper()
            
            is_non_banking = result == "YES"
            
            if is_non_banking:
                logger.info(f"🚫 BLOCKED non-banking query: '{user_message}'")
            else:
                logger.info(f"✅ ALLOWED banking query: '{user_message}'")
            
            return is_non_banking
            
        except Exception as e:
            logger.error(f"Error in non-banking detection: {e}")
            # Fallback: be more conservative with banking-related patterns
            user_lower = user_message.lower().strip()
            
            # NEVER block CNIC format
            if re.match(r'^\d{5}-\d{7}-\d{1}$', user_message.strip()):
                logger.info(f"✅ Fallback: CNIC format allowed: '{user_message}'")
                return False
                
            # NEVER block number-only inputs (could be account numbers, OTPs)
            if re.match(r'^\d+$', user_message.strip()):
                logger.info(f"✅ Fallback: Number input allowed: '{user_message}'")
                return False
            
            # Banking keywords - ALLOW these
            banking_keywords = [
                "balance", "transaction", "transfer", "money", "account", "spending", 
                "pay", "currency", "cnic", "otp", "deposit", "withdraw", "statement",
                "bill", "payment", "bank", "cash", "fund", "amount", "rupee", "dollar"
            ]
            
            # Non-banking keywords - BLOCK these
            non_banking_keywords = [
                "ceo", "weather", "sports", "movie", "celebrity", "politics", 
                "job", "amazon", "apple", "google", "microsoft", "hire", "career",
                "recipe", "cook", "health", "doctor", "medicine", "travel",
                "python", "programming", "ai", "technology", "computer",
                "netflix", "tv", "entertainment", "music", "songs", "game"
            ]
            
            # If contains banking keywords, allow
            if any(keyword in user_lower for keyword in banking_keywords):
                logger.info(f"✅ Fallback: Banking keyword detected, allowed: '{user_message}'")
                return False
                
            # If contains non-banking keywords, block
            if any(keyword in user_lower for keyword in non_banking_keywords):
                logger.info(f"🚫 Fallback: Non-banking keyword detected, blocked: '{user_message}'")
                return True
                
            # For uncertain cases during authentication flow, be more lenient
            logger.info(f"⚠️ Fallback: Uncertain query, allowing for safety: '{user_message}'")
            return False  # Changed from True to False for authentication flow
   
            
        except Exception as e:
            logger.error(f"Error in non-banking detection: {e}")
            # Fallback: allow banking-related keywords, block obvious non-banking
            non_banking_keywords = ["ceo", "weather", "sports", "movie", "celebrity", "politics"]
            banking_keywords = ["balance", "transaction", "transfer", "money", "account", "spending"]
            
            user_lower = user_message.lower()
            
            # If contains banking keywords, allow
            if any(keyword in user_lower for keyword in banking_keywords):
                return False
                
            # If contains non-banking keywords, block
            if any(keyword in user_lower for keyword in non_banking_keywords):
                logger.info(f"Fallback blocked non-banking query: '{user_message}'")
                return True
                
            return False  # Default to allowing if uncertain
        
    async def handle_non_banking_query(self, user_message: str, first_name: str) -> str:
        """Handle clearly non-banking related queries with polite but firm decline."""
        try:
            non_banking_prompt = f"""You are Sage, a strict banking assistant. The user {first_name} asked a non-banking question that you must firmly refuse to answer.

            Their question: "{user_message}"

            Response rules:
            1. Be polite but VERY firm - you ONLY handle banking account queries
            2. Don't provide ANY information about the non-banking topic
            3. Clearly state you cannot and will not help with non-banking topics
            4. Redirect to ONLY banking services: balance, transactions, spending, transfers
            5. Keep it short and redirect immediately

            Generate a firm refusal that maintains strict banking boundaries."""
            
            try:
                response = await llm.ainvoke([SystemMessage(content=non_banking_prompt)])
                return response.content.strip()
            except:
                return f"I'm a banking assistant, {first_name}, and I can only help with your bank account questions like checking your balance, viewing transactions, analyzing spending, or transferring money. I cannot provide information about other topics. What banking question can I help you with?"
                
        except Exception as e:
            logger.error(f"Error in non-banking response: {e}")
            return f"I'm sorry {first_name}, but I can only help with banking questions related to your account. I don't have access to information about other topics. What banking question can I help you with today?"
    
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
                    current_balance = balance_info.get("current_balance_pkr", 0)
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