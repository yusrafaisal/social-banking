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
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3  # Balanced for natural but accurate responses
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
    intent: str = Field(default="general")
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

def month_to_number(month: str) -> int:
    """Convert month name to number."""
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    return months.get(month.lower(), 1)

def month_days(month: str, year: int) -> int:
    """Get number of days in a month."""
    month_num = month_to_number(month)
    if month_num in [4, 6, 9, 11]:
        return 30
    elif month_num == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    else:
        return 31
        
_BRACE_RE = re.compile(r'[{[]')

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
    def __init__(self, mongodb_uri: str = "mongodb://localhost:27017/", db_name: str = "bank_database"):
        """Initialize the Banking AI Agent with MongoDB connection."""
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db["transactions"]
        self.backend_url = "http://localhost:8000"
        # Use LangChain memory directly without ConversationChain
        self.user_memories: Dict[str, ConversationBufferMemory] = {}
        
    def get_user_memory(self, account_number: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a user account."""
        if account_number not in self.user_memories:
            self.user_memories[account_number] = ConversationBufferMemory(
                return_messages=True
            )
        return self.user_memories[account_number]

    def is_clearly_non_banking_query(self, user_message: str) -> bool:
        """Check if user query is clearly non-banking using keyword detection and LLM as backup."""
        try:
            user_message_lower = user_message.lower().strip()
            
            # Quick keyword-based filtering for obvious non-banking topics
            obvious_non_banking_keywords = [
                "president", "weather", "temperature", "rain", "sunny", "cloudy",
                "sports", "football", "cricket", "basketball", "soccer",
                "movies", "film", "actor", "actress", "celebrity",
                "news", "politics", "government", "election", "vote",
                "cooking", "recipe", "food recipe", "how to cook",
                "travel destination", "vacation spot", "tourism",
                "joke", "tell me a joke", "funny story",
                "science", "physics", "chemistry", "biology",
                "technology news", "latest tech", "gadgets",
                "health tips", "medical advice", "symptoms",
                "astrology", "horoscope", "zodiac",
                "music", "song", "singer", "band",
                "book", "novel", "author", "literature"
            ]
            
            # Check for obvious non-banking keywords
            for keyword in obvious_non_banking_keywords:
                if keyword in user_message_lower:
                    logger.info(f"Blocked obvious non-banking query: '{user_message}' (keyword: {keyword})")
                    return True
            
            # Common non-banking question patterns
            non_banking_patterns = [
                r'^who is (the )?president',
                r'^what.*(weather|temperature)',
                r'^tell me a joke',
                r'^how to cook',
                r'^what happened in the news',
                r'^latest news',
                r'^(who|what|when|where|how).*(movie|film|actor)',
                r'^(who|what).*(singer|song|music)',
                r'^what time is it',
                r'^how old is'
            ]
            
            for pattern in non_banking_patterns:
                if re.match(pattern, user_message_lower):
                    logger.info(f"Blocked non-banking query by pattern: '{user_message}' (pattern: {pattern})")
                    return True
            
            # If no obvious non-banking indicators, use LLM as backup for edge cases
            if len(user_message.split()) > 3:  # Only use LLM for longer queries
                non_banking_filter_prompt = f"""
                You are a banking assistant filter. Determine if the following query is CLEARLY about non-banking topics.

                ONLY respond "YES" if the query is OBVIOUSLY about:
                - World events, politics, news, current affairs
                - Weather, climate, forecasts  
                - Entertainment (movies, music, celebrities, sports)
                - General knowledge (science, history, geography)
                - Technology unrelated to banking
                - Health, cooking, travel, jokes, personal advice

                If there's ANY chance it could be banking-related or if it's ambiguous, respond "NO".
                
                Examples of CLEAR non-banking (respond "YES"):
                - "Who is the president of USA?"
                - "What's the weather today?"
                - "Tell me a joke"
                - "Who won the football game?"
                
                Examples to ALLOW (respond "NO"):  
                - "check 3 instead" (could be contextual banking)
                - "what about last month" (could be banking context)
                - "show me more" (could be banking follow-up)
                - "how much" (could be banking amount)
                - Numbers, dates, amounts (could be banking)

                Query: "{user_message}"

                Respond with only "YES" (clearly non-banking) or "NO" (allow processing).
                """

                response = llm.invoke([SystemMessage(content=non_banking_filter_prompt)])
                result = response.content.strip().upper()
                
                if result == "YES":
                    logger.info(f"LLM blocked non-banking query: '{user_message}'")
                    return True
                else:
                    logger.info(f"LLM allowed query (banking or ambiguous): '{user_message}'")
                    return False
            
            # Default: allow processing (err on the side of allowing)
            logger.info(f"Allowing query by default: '{user_message}'")
            return False
            
        except Exception as e:
            logger.error(f"Error in non-banking filter: {e}")
            # If filter fails, err on the side of allowing the query
            logger.info(f"Filter error - allowing query: '{user_message}'")
            return False

    async def generate_natural_response(self, context_state: str, data: Any, user_message: str, first_name: str, conversation_history: str = "") -> str:
        """Generate natural LLM responses with varied greetings and natural conversation flow."""
        
        # Enhanced system prompt for varied, natural responses
        system_prompt = f"""You are Sage, a conversational and intelligent personal banking assistant. You maintain a natural, helpful personality throughout all interactions.

CURRENT CONTEXT: {context_state}
USER'S NAME: {first_name}
USER'S MESSAGE: "{user_message}"
CONVERSATION HISTORY: {conversation_history}
AVAILABLE DATA: {json.dumps(data) if data else "No specific data"}

PERSONALITY GUIDELINES:
- Be conversational and natural like a knowledgeable friend
- Keep responses balanced in length - not too brief, not too verbose  
- No bullet points or excessive formatting unless truly natural
- Maintain consistent Sage personality
- Be helpful and accurate with data
- Handle state transitions smoothly

GREETING VARIATION INSTRUCTIONS (VERY IMPORTANT):
- NEVER always start with "Hey {first_name}!" - this is robotic and unnatural
- Vary your greetings naturally: "Hi {first_name}!", "Hello {first_name}!", "Good to see you {first_name}!", "Alright {first_name},", "{first_name},", "Hope you're doing well {first_name}!", "Nice to hear from you {first_name}!"
- Sometimes skip greetings entirely and dive straight into the response
- Use different greeting styles: casual ("What's up {first_name}!"), formal ("Good day {first_name}"), friendly ("Hope you're well {first_name}!")
- Match the greeting energy to the context and conversation flow
- If continuing a conversation thread, often skip greetings and just respond naturally
- For follow-up questions or contextual queries, usually start directly with the answer

RESPONSE GUIDELINES:
- If presenting financial data, be accurate but conversational
- If no data exists, explain naturally without being apologetic
- Understand conversation flow and respond appropriately
- Be engaging but professional
- Use natural language structure
- Make each response feel unique and human-like
- Avoid repetitive patterns at all costs

Generate a natural response that fits the context and data provided. Remember to vary your greeting style and don't always use the same pattern."""

        try:
            response = await llm.ainvoke([SystemMessage(content=system_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            return f"I'm having some technical difficulties right now, {first_name}. Could you try that again?"

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

    async def handle_otp_request(self, first_name: str = "") -> str:
        """Handle OTP request after CNIC verification."""
        try:
            otp_prompt = f"""You are Sage, a banking assistant. A user has just had their CNIC verified successfully and now needs to provide an OTP for additional security.

Your task:
1. Explain that for additional security, they need to provide an OTP
2. Tell them the OTP can be any number between 1-5 digits
3. Ask them to enter their OTP
4. Keep it simple and secure-sounding

Generate a natural, security-focused response asking for OTP."""

            response = await llm.ainvoke([SystemMessage(content=otp_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating OTP request: {e}")
            return f"Great! For additional security, {first_name}, please provide an OTP. You can enter any number between 1-5 digits as your OTP."

    async def handle_otp_success(self, user_name: str, accounts: List[str]) -> str:
        """Handle successful OTP verification."""
        try:
            first_name = user_name.split()[0]
            context_state = "OTP verification successful, user can now select account"
            data = {
                "otp_verified": True,
                "user_name": user_name,
                "accounts": accounts,
                "next_step": "account_selection"
            }
            
            return await self.generate_natural_response(context_state, data, "otp_verified", first_name)
            
        except Exception as e:
            logger.error(f"Error in OTP success: {e}")
            return "OTP verified successfully! Now please select your account by entering the last 4 digits."

    async def handle_otp_failure(self, user_input: str, first_name: str = "") -> str:
        """Handle failed OTP verification."""
        try:
            context_state = "OTP verification failed, need valid 1-5 digit number"
            data = {
                "otp_failed": True,
                "user_input": user_input,
                "required_format": "1-5 digits"
            }
            
            return await self.generate_natural_response(context_state, data, user_input, first_name)
            
        except Exception as e:
            logger.error(f"Error in OTP failure: {e}")
            return f"Sorry {first_name}, that OTP format isn't valid. Please enter a number between 1-5 digits."

    async def handle_transfer_otp_request(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle OTP request for money transfer."""
        try:
            context_state = "Transfer details collected, requesting OTP for security"
            data = {
                "transfer_amount": amount,
                "transfer_currency": currency,
                "transfer_recipient": recipient,
                "otp_required": True
            }
            
            return await self.generate_natural_response(context_state, data, f"transfer {amount} {currency} to {recipient}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer OTP request: {e}")
            return f"To complete the transfer of {amount} {currency} to {recipient}, please provide an OTP (any number between 1-5 digits) for security verification."

    async def handle_transfer_otp_success(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle successful transfer OTP verification."""
        try:
            context_state = "Transfer OTP verified, ready to proceed with transfer"
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

    async def handle_transfer_confirmation_request(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle transfer confirmation request after OTP verification."""
        try:
            context_state = "Transfer OTP verified, now requesting user confirmation before executing transfer"
            data = {
                "otp_verified": True,
                "transfer_amount": amount,
                "transfer_currency": currency,
                "transfer_recipient": recipient,
                "confirmation_required": True,
                "confirmation_options": ["yes", "no", "confirm", "cancel"]
            }
            
            return await self.generate_natural_response(context_state, data, f"confirm transfer {amount} {currency} to {recipient}", first_name)
            
        except Exception as e:
            logger.error(f"Error in transfer confirmation request: {e}")
            return f"OTP verified! Do you want to confirm the transfer of {amount} {currency} to {recipient}? Please reply with 'yes' to confirm or 'no' to cancel."

    async def handle_transfer_cancellation(self, amount: float, currency: str, recipient: str, first_name: str) -> str:
        """Handle transfer cancellation when user declines confirmation."""
        try:
            context_state = "User cancelled transfer during confirmation step, providing cancellation confirmation"
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

    async def handle_non_banking_query(self, user_message: str, first_name: str) -> str:
        """Handle clearly non-banking related queries with polite decline."""
        try:
            context_state = "User asked clearly non-banking question, politely redirecting to banking services"
            data = {
                "query_type": "non_banking",
                "available_services": ["balance inquiry", "transaction history", "spending analysis", "money transfer"]
            }
            
            return await self.generate_natural_response(context_state, data, user_message, first_name)
            
        except Exception as e:
            logger.error(f"Error in non-banking response: {e}")
            return f"I'm sorry {first_name}, but I can only help with banking and financial questions. I don't have information about topics outside of banking in my database. Is there anything related to your banking needs I can help you with instead?"

    async def process_query(self, user_message: str, account_number: str, first_name: str) -> str:
        """Enhanced process query with LLM-first approach, OTP support, and less restrictive filtering."""
        memory = self.get_user_memory(account_number)

        logger.info({
            "action": "process_query_start",
            "approach": "llm_first_with_fallback_otp_less_restrictive",
            "user_message": user_message,
            "account_number": account_number
        })

        # Check for exit command first
        if user_message.lower().strip() in ['exit', 'quit', 'logout', 'end']:
            response = await self.handle_session_end(account_number, first_name)
            return response

        # Handle greetings and simple queries
        if self._is_simple_greeting_or_general(user_message):
            context_state = "User sent a greeting or general question, no specific banking data needed"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            response = await self.generate_natural_response(context_state, None, user_message, first_name, conversation_history)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # NEW: Check if query is CLEARLY non-banking (less restrictive)
        if self.is_clearly_non_banking_query(user_message):
            logger.info(f"Clearly non-banking query detected: {user_message}")
            response = await self.handle_non_banking_query(user_message, first_name)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # PRIMARY PATH: LLM-FIRST APPROACH (for banking/ambiguous queries)
        try:
            logger.info("Attempting LLM-first approach for banking/contextual query")
            
            # Step 1: Extract filters using LLM
            filters = self.extract_filters_with_llm(user_message)
            logger.info(f"LLM extracted filters: {filters.dict()}")
            
            # Step 2: Detect intent using LLM
            intent = self.detect_intent_from_filters(user_message, filters)
            logger.info(f"LLM detected intent: {intent}")
            
            # Step 3: Handle based on intent
            if intent == "balance_inquiry":
                response = await self._handle_balance_inquiry(account_number, first_name, user_message, memory)
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response)
                return response
            
            elif intent == "transfer_money":
                # Enhanced transfer with OTP support
                response = await self._handle_money_transfer_with_otp(account_number, user_message, first_name, memory)
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response)
                return response
            
            elif intent in ["transaction_history", "spending_analysis", "category_spending"]:
                # Step 4: Generate pipeline using LLM
                pipeline = self.generate_pipeline_from_filters(filters, intent, account_number)
                logger.info(f"LLM generated pipeline: {pipeline}")
                
                # Step 5: Validate and execute pipeline
                if pipeline:
                    try:
                        jsonschema.validate(pipeline, PIPELINE_SCHEMA)
                        response = await self._execute_llm_pipeline(account_number, pipeline, user_message, first_name, memory, intent)
                        memory.chat_memory.add_user_message(user_message)
                        memory.chat_memory.add_ai_message(response)
                        return response
                    except jsonschema.ValidationError as e:
                        logger.warning(f"LLM pipeline validation failed: {e}")
                        raise Exception("Pipeline validation failed")
                else:
                    raise Exception("Empty pipeline generated")
            
            else:
                # General or unclear intent - handle naturally (allow contextual queries)
                context_state = "General banking assistance or contextual query"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                response = await self.generate_natural_response(context_state, None, user_message, first_name, conversation_history)
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response)
                return response
                
        except Exception as llm_error:
            logger.warning(f"LLM approach failed: {llm_error}. Falling back to hardcoded methods.")
            
            # FALLBACK PATH: HARDCODED METHODS
            try:
                logger.info("Using hardcoded fallback approach")
                
                # Use the original natural reasoning as fallback
                reasoning_result = await self._reason_about_query(user_message, memory, account_number, first_name)
                
                logger.info(f"Fallback - Query: '{user_message}' -> Action: {reasoning_result.get('action_needed')} -> Analysis: {reasoning_result.get('analysis_type', 'N/A')}")
                
                # Handle with original hardcoded methods
                if reasoning_result.get("action_needed") == "balance_check":
                    response = await self._handle_balance_inquiry(account_number, first_name, user_message, memory)
                    
                elif reasoning_result.get("action_needed") == "transaction_history":
                    response = await self._handle_transaction_history(user_message, account_number, first_name, reasoning_result, memory)
                    
                elif reasoning_result.get("action_needed") == "sophisticated_analysis":
                    response = await self._handle_sophisticated_analysis(user_message, account_number, first_name, reasoning_result, memory)
                    
                else:
                    # Default to natural response
                    context_state = "Fallback - providing general assistance"
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    response = await self.generate_natural_response(context_state, None, user_message, first_name, conversation_history)
                
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response)
                return response
                
            except Exception as fallback_error:
                logger.error(f"Both LLM and fallback failed: {fallback_error}")
                context_state = "Both primary and fallback systems failed, providing error response"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                response = await self.generate_natural_response(context_state, {"error": str(fallback_error)}, user_message, first_name, conversation_history)
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response)
                return response

    async def _handle_money_transfer_with_otp(self, account_number: str, user_message: str, 
                                            first_name: str, memory: ConversationBufferMemory) -> str:
        """Handle money transfer with OTP requirement (returns 'OTP_REQUIRED' for webhook to handle)."""
        try:
            # Use sophisticated transfer prompt
            response = await llm.ainvoke([SystemMessage(content=transfer_prompt.format(user_message=user_message))])
            transfer_details = self.extract_json_from_response(response.content)
            
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
                    "currency": transfer_details.get("currency", "PKR")
                }
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # If transfer details are complete, return special signal for OTP
            # The webhook will handle the OTP flow
            return f"OTP_REQUIRED|{transfer_details['amount']}|{transfer_details.get('currency', 'PKR')}|{transfer_details['recipient']}"
                
        except Exception as e:
            logger.error(f"Error in money transfer with OTP: {e}")
            context_state = "Error occurred during money transfer processing"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

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
                
                context_state = "Transfer executed successfully after OTP verification" if transfer_result.get("status") == "success" else "Transfer failed after OTP verification"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                
                return await self.generate_natural_response(context_state, transfer_result, f"transfer {amount} {currency} to {recipient}", first_name, conversation_history)
                
        except Exception as e:
            logger.error(f"Error executing verified transfer: {e}")
            context_state = "Error occurred during transfer execution"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, f"transfer {amount} {currency} to {recipient}", first_name, conversation_history)

    def _is_simple_greeting_or_general(self, user_message: str) -> bool:
        """Check if message is a simple greeting or general question."""
        user_lower = user_message.lower().strip()
        
        # Simple greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(user_lower.startswith(greeting) for greeting in greetings):
            return True
            
        # General questions
        general_phrases = ["what can you do", "help me", "what do you do", "how can you help"]
        if any(phrase in user_lower for phrase in general_phrases):
            return True
            
        return False

    async def _execute_llm_pipeline(self, account_number: str, pipeline: List[Dict[str, Any]], 
                                  user_message: str, first_name: str, memory: ConversationBufferMemory, 
                                  intent: str) -> str:
        """Execute LLM-generated pipeline and format response naturally."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/execute_pipeline",
                    json={"account_number": account_number, "pipeline": pipeline}
                )
                response.raise_for_status()
                data = response.json()
                
                # Format response using LLM with sophisticated prompt
                formatted_response = await llm.ainvoke([
                    SystemMessage(content=response_prompt.format(
                        user_message=user_message,
                        intent=intent,
                        data=json.dumps(data)
                    ))
                ])
                
                logger.info(f"LLM pipeline executed successfully for intent: {intent}")
                return formatted_response.content.strip()
                
        except Exception as e:
            logger.error(f"Error executing LLM pipeline: {e}")
            context_state = "Error occurred while executing database query"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    # Keep all existing methods for fallback and session management...
    
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
                transaction_keywords = ["transaction", "transactions", "last", "recent", "history", "may", "june", "july"]
                if any(keyword in user_message.lower() for keyword in transaction_keywords):
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
            transaction_keywords = ["transaction", "transactions", "last", "recent", "history", "may", "june", "july"]
            if any(keyword in user_message.lower() for keyword in transaction_keywords):
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
                "balance_inquiry",
                "transaction_history", 
                "spending_analysis",
                "category_spending",
                "transfer_money",
                "general"
            ]
            
            if detected_intent in valid_intents:
                return detected_intent
            else:
                for intent in valid_intents:
                    if intent in detected_intent:
                        return intent
                return "general"
                
        except Exception as e:
            logger.error({
                "action": "llm_intent_classification",
                "error": str(e),
                "user_message": user_message
            })
            return self._rule_based_intent_fallback(user_message, filters)

    def _rule_based_intent_fallback(self, user_message: str, filters: FilterExtraction) -> str:
        """Enhanced rule-based intent detection."""
        user_message_lower = user_message.lower()
        
        # Enhanced keywords for better detection
        balance_keywords = ["balance", "money", "amount", "funds", "account", "cash", "afford", "target", "save", "purchase", "buy", "enough", "capacity"]
        transaction_keywords = ["transaction", "history", "recent", "last", "show", "list", "activities"]
        spending_keywords = ["spend", "spent", "spending", "expenditure", "expense", "expenses", 
                            "cost", "costs", "paid", "pay", "payment", "purchase", "purchased", 
                            "buying", "bought", "money went", "charged", "compare", "more than", 
                            "less than", "patterns", "habits", "analysis", "right now"]
        transfer_keywords = ["transfer", "send", "wire", "remit", "move money", "i want to transfer", 
                            "transfer money", "send money", "pay"]
        planning_keywords = ["planning", "target", "goal", "save", "afford", "can i", "what can i do"]
        
        # Check for transfer intent first (highest priority for explicit transfer requests)
        if any(keyword in user_message_lower for keyword in transfer_keywords):
            return "transfer_money"
        
        # Financial planning and affordability questions
        if any(keyword in user_message_lower for keyword in planning_keywords):
            return "balance_inquiry"
        
        # Spending comparisons and analysis
        if any(keyword in user_message_lower for keyword in ["more than", "less than", "compared to", "compare", "vs", "versus"]):
            return "spending_analysis"
            
        # Traditional keyword matching
        if any(keyword in user_message_lower for keyword in balance_keywords):
            return "balance_inquiry"
        elif any(keyword in user_message_lower for keyword in transaction_keywords) or filters.limit:
            return "transaction_history"
        elif any(keyword in user_message_lower for keyword in spending_keywords):
            if filters.category:
                return "category_spending"
            else:
                return "spending_analysis"
        else:
            return "general"

    def _generate_fallback_pipeline(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate a basic pipeline when LLM fails."""
        match_stage = {"$match": {"account_number": account_number}}
        
        if intent == "transaction_history":
            pipeline = [match_stage, {"$sort": {"date": -1, "_id": -1}}]
            if filters.limit:
                pipeline.append({"$limit": filters.limit})
            return pipeline
        
        elif intent in ["spending_analysis", "category_spending"]:
            if filters.transaction_type:
                match_stage["$match"]["type"] = filters.transaction_type
            
            if filters.description:
                match_stage["$match"]["description"] = {
                    "$regex": filters.description,
                    "$options": "i"
                }
            
            if filters.category:
                match_stage["$match"]["category"] = {
                    "$regex": filters.category,
                    "$options": "i"
                }
            
            pipeline = [
                match_stage,
                {
                    "$group": {
                        "_id": None,
                        "total_amount": {"$sum": "$transaction_amount"},
                        "currency": {"$first": "$transaction_currency"}
                    }
                }
            ]
            return pipeline
        
        return [match_stage, {"$sort": {"date": -1, "_id": -1}}, {"$limit": 10}]

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

    def _get_context_summary(self, chat_history: List) -> str:
        """Get a summary of recent conversation for context."""
        if not chat_history:
            return "No previous conversation."
        
        # Get last 4 messages for context
        recent_messages = chat_history[-4:] if len(chat_history) > 4 else chat_history
        context_lines = []
        
        for i, msg in enumerate(recent_messages):
            speaker = "Human" if i % 2 == 0 else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context_lines.append(f"{speaker}: {content}")
        
        return "\n".join(context_lines)

    # Keep ALL the existing fallback methods unchanged for compatibility...
    
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
            - "show me my transactions" -> 20 (default)
            - "transaction history" -> 20 (default)
            - "transactions in May" -> 50 (show all for month)
            
            Return only the number as an integer. If no specific number mentioned, return 20.
            If asking for a specific month, return 50.
            """
            
            try:
                limit_response = await llm.ainvoke([SystemMessage(content=limit_prompt)])
                limit = int(limit_response.content.strip())
                if limit <= 0 or limit > 100:  # Reasonable bounds
                    limit = 20
            except:
                limit = 20  # Default fallback
            
            logger.info(f"Using limit: {limit} transactions")
            
            # Check if user specified a month
            month_filter = None
            months = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            
            user_message_lower = user_message.lower()
            for month_name, month_num in months.items():
                if month_name in user_message_lower:
                    month_filter = month_num
                    logger.info(f"Found month filter: {month_name} ({month_num})")
                    break
            
            # Build MongoDB query for transaction history
            query = {
                "account_number": account_number
            }
            
            # Add month filter if specified
            if month_filter:
                current_year = datetime.now().year
                days_in_month = month_days(list(months.keys())[month_filter-1], current_year)
                
                query["date"] = {
                    "$gte": datetime(current_year, month_filter, 1),
                    "$lte": datetime(current_year, month_filter, days_in_month, 23, 59, 59)
                }
                logger.info(f"Added date filter for month {month_filter}: {query['date']}")
            
            # Query the database directly using the collection
            transactions = list(self.collection.find(query).sort("date", -1).limit(limit))
            
            logger.info(f"Found {len(transactions)} transactions for account {account_number}")
            
            if not transactions:
                if month_filter:
                    month_name = list(months.keys())[month_filter-1].title()
                    context_state = f"No transactions found for {month_name} for user account"
                    data = {"transaction_count": 0, "account_number": account_number, "month": month_name}
                else:
                    context_state = "No transaction history found for user account"
                    data = {"transaction_count": 0, "account_number": account_number}
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # Format transactions for response - CONVERT DATETIME TO STRING
            formatted_transactions = []
            for tx in transactions:
                # Convert datetime to string to avoid JSON serialization error
                date_obj = tx.get("date")
                if isinstance(date_obj, datetime):
                    date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = str(date_obj) if date_obj else "Unknown"
                
                formatted_tx = {
                    "date": date_str,  # Now a string, not datetime object
                    "description": tx.get("description", ""),
                    "category": tx.get("category", ""),
                    "type": tx.get("type", ""),
                    "amount": tx.get("transaction_amount", 0),
                    "currency": tx.get("transaction_currency", "PKR"),
                    "balance": tx.get("account_balance", 0)
                }
                formatted_transactions.append(formatted_tx)
            
            # Set context based on whether month filter was used
            if month_filter:
                month_name = list(months.keys())[month_filter-1].title()
                context_state = f"User requested transaction history for {month_name}, providing {len(transactions)} transactions"
            else:
                context_state = f"User requested transaction history, providing {len(transactions)} recent transactions"
            
            data = {
                "transactions": formatted_transactions,
                "transaction_count": len(transactions),
                "requested_limit": limit,
                "account_number": account_number,
                "month_filter": list(months.keys())[month_filter-1].title() if month_filter else None
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Transaction history error: {e}")
            context_state = "Error occurred while retrieving transaction history"
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
            context_state = "Technical error occurred during analysis"
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
            months = {
                "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            
            # Look in recent messages for month mentions
            for msg in reversed(chat_history[-6:] if len(chat_history) > 6 else chat_history):
                content = msg.content.lower()
                for month_name, month_num in months.items():
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
            days_in_month = 31
            if target_month in [4, 6, 9, 11]:
                days_in_month = 30
            elif target_month == 2:
                days_in_month = 29 if target_year % 4 == 0 else 28
            
            # Query all transactions for the target month
            start_date = datetime(target_year, target_month, 1)
            end_date = datetime(target_year, target_month, days_in_month, 23, 59, 59)
            
            query = {
                "account_number": account_number,
                "type": "debit",
                "date": {
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
                amount = tx.get("transaction_amount", 0)
                total_spent += amount
                
                category = tx.get("category", "Other")
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
                    category_spending[category]["description"] = tx.get("description", "")
            
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
                "currency": transactions[0].get("transaction_currency", "USD") if transactions else "USD"
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Spending breakdown error: {e}")
            context_state = "Error occurred while analyzing spending breakdown"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    # Keep all other existing methods...
    async def _handle_balance_inquiry(self, account_number: str, first_name: str, user_message: str, memory: ConversationBufferMemory) -> str:
        """Enhanced balance inquiry with natural responses."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/user_balance",
                    json={"account_number": account_number}
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
                    context_state = "User requesting current account balance"
                    conversation_history = self._get_context_summary(memory.chat_memory.messages)
                    response_text = await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                
                return response_text
                
        except Exception as e:
            logger.error({"action": "handle_balance_inquiry", "error": str(e)})
            context_state = "Error occurred while retrieving balance information"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    # Add all other session management and utility methods...
    async def handle_session_start(self, first_name: str = "", last_name: str = "") -> str:
        """Handle session start with natural greeting."""
        try:
            context_state = "New user starting banking session, need CNIC verification to begin"
            data = {
                "session_status": "starting",
                "authentication_required": "cnic_verification",
                "security_level": "high",
                "next_step": "cnic_input"
            }
            
            return await self.generate_natural_response(context_state, data, "", first_name or "there")
            
        except Exception as e:
            logger.error(f"Error in session start: {e}")
            context_state = "Error occurred during session initialization"
            return await self.generate_natural_response(context_state, {"error": str(e)}, "", first_name or "there")

    async def handle_session_end(self, account_number: str, first_name: str) -> str:
        """Handle session termination with natural response."""
        try:
            # Clear user memory for security
            if account_number in self.user_memories:
                del self.user_memories[account_number]
            
            context_state = "User ending banking session, providing secure farewell and cleanup confirmation"
            data = {
                "session_ended": True,
                "security_cleared": True,
                "memory_cleared": True,
                "restart_instructions": True
            }
            
            return await self.generate_natural_response(context_state, data, "exit", first_name)
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            context_state = "Error occurred while ending session securely"
            return await self.generate_natural_response(context_state, {"error": str(e)}, "exit", first_name)

    async def handle_cnic_verification_success(self, user_name: str, accounts: List[str], cnic: str) -> str:
        """Handle successful CNIC verification with natural response."""
        try:
            context_state = "CNIC verification successful, now requesting OTP for additional security"
            data = {
                "verification_status": "success",
                "user_name": user_name,
                "accounts_found": len(accounts),
                "accounts": accounts,
                "next_step": "otp_verification",
                "otp_format": "1-5 digits"
            }
            
            return await self.generate_natural_response(context_state, data, cnic, user_name.split()[0])
            
        except Exception as e:
            logger.error(f"Error in CNIC verification success: {e}")
            context_state = "Error occurred after successful CNIC verification"
            return await self.generate_natural_response(context_state, {"error": str(e)}, cnic, user_name.split()[0])

    async def handle_cnic_verification_failure(self, cnic: str, first_name: str = "") -> str:
        """Handle failed CNIC verification with natural response."""
        try:
            context_state = "CNIC verification failed, user needs to try again with correct format"
            data = {
                "verification_status": "failed",
                "provided_cnic": cnic,
                "required_format": "12345-1234567-1",
                "retry_needed": True
            }
            
            return await self.generate_natural_response(context_state, data, cnic, first_name or "there")
            
        except Exception as e:
            logger.error(f"Error in CNIC verification failure: {e}")
            context_state = "Error occurred during CNIC verification failure handling"
            return await self.generate_natural_response(context_state, {"error": str(e)}, cnic, first_name or "there")

    async def handle_invalid_cnic_format(self, user_input: str, first_name: str = "") -> str:
        """Handle invalid CNIC format with natural response."""
        try:
            context_state = "User provided invalid CNIC format, need guidance on correct format"
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
            context_state = "Error occurred while handling invalid CNIC format"
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_input, first_name or "there")

    async def handle_account_selection(self, selection: str, accounts: List[str], first_name: str) -> str:
        """Handle account selection with natural response."""
        try:
            context_state = "User provided account selection input, providing guidance"
            data = {
                "available_accounts": accounts,
                "user_input": selection,
                "expected_format": "last_4_digits",
                "selection_method": "last_4_digits"
            }
            
            return await self.generate_natural_response(context_state, data, selection, first_name)
            
        except Exception as e:
            logger.error(f"Error in account selection: {e}")
            context_state = "Error occurred during account selection"
            return await self.generate_natural_response(context_state, {"error": str(e)}, selection, first_name)

    async def handle_account_confirmation(self, account_number: str, user_name: str) -> str:
        """Handle account selection confirmation with natural response."""
        try:
            first_name = user_name.split()[0]
            context_state = "Account successfully selected and confirmed, user ready for full banking operations"
            data = {
                "account_confirmed": True,
                "account_number": account_number,
                "masked_account": f"***-***-{account_number[-4:]}",
                "user_name": user_name,
                "ready_for_banking": True,
                "available_services": [
                    "balance_inquiry",
                    "spending_analysis", 
                    "transaction_history",
                    "money_transfer",
                    "financial_planning"
                ]
            }
            
            return await self.generate_natural_response(context_state, data, account_number, first_name)
            
        except Exception as e:
            logger.error(f"Error in account confirmation: {e}")
            context_state = "Error occurred during account confirmation"
            return await self.generate_natural_response(context_state, {"error": str(e)}, account_number, user_name.split()[0])

    async def handle_error_gracefully(self, error: Exception, user_message: str, first_name: str, context: str = "general") -> str:
        """Handle any error gracefully with natural response."""
        logger.error(f"Error in {context}: {str(error)}")
        
        context_state = f"Technical error occurred during {context}, providing helpful alternative"
        data = {
            "error_type": context,
            "user_can_retry": True,
            "alternative_suggestions": True
        }
        
        return await self.generate_natural_response(context_state, data, user_message, first_name)

    def clear_user_memory(self, account_number: str) -> None:
        """Clear conversation memory for a specific user account."""
        if account_number in self.user_memories:
            del self.user_memories[account_number]
            logger.info(f"Cleared memory for account: {account_number}")

    def get_conversation_summary(self, account_number: str) -> str:
        """Get a summary of the current conversation."""
        memory = self.get_user_memory(account_number)
        return self._get_context_summary(memory.chat_memory.messages)

    def __del__(self):
        """Cleanup resources when agent is destroyed."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


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
        if message.lower().strip() in ['exit', 'quit', 'logout', 'end']:
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