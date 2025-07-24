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
            "$project": {"type": "object"}
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

    async def generate_natural_response(self, context_state: str, data: Any, user_message: str, first_name: str, conversation_history: str = "") -> str:
        """Generate natural LLM responses with varied greetings and natural conversation flow."""
        
        # 🔧 FIX 2: Enhanced system prompt for varied, natural responses
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

    # 🔧 FIX 1: New method for handling initial greetings properly
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

    def _extract_merchant(self, user_message: str) -> str:
        """Extract merchant name from contextual queries."""
        user_lower = user_message.lower()
        
        # Look for "was this on X" or "was that on X"
        import re
        patterns = [
            r'was (?:this|that) on (\w+)',
            r'from (\w+)',
            r'at (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_lower)
            if match:
                return match.group(1)
        
        return ""

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

    async def _handle_simple_contextual_query(self, user_message: str, account_number: str, 
                                            first_name: str, reasoning: Dict[str, Any], 
                                            memory: ConversationBufferMemory) -> str:
        """Handle simple contextual queries without complex pipelines."""
        try:
            merchant_filter = reasoning.get("merchant_filter", "")
            
            logger.info(f"Handling simple contextual query for merchant: {merchant_filter}")
            
            # For "was this on X?" queries, look for recent transactions with that merchant
            if merchant_filter and "was this on" in user_message.lower():
                # Get date context from recent conversation if possible
                chat_history = memory.chat_memory.messages
                date_context = self._extract_date_from_context(chat_history)
                
                # Build simple query
                query = {
                    "account_number": account_number,
                    "description": {"$regex": merchant_filter, "$options": "i"}
                }
                
                # Add date filter if we found date context
                if date_context:
                    query["date"] = date_context
                
                # Simple database query (not complex pipeline)
                transactions = list(self.collection.find(query).sort("date", -1).limit(3))
                
                # Generate natural response
                if not transactions:
                    context_state = f"User asked if previous transaction was with {merchant_filter}, but no transactions found"
                    data = {"merchant": merchant_filter, "transactions": []}
                else:
                    context_state = f"User asked if previous transaction was with {merchant_filter}, found matching transactions"
                    data = {"merchant": merchant_filter, "transactions": transactions}
                
                conversation_history = self._get_context_summary(chat_history)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # For "show me other X" queries
            elif "other" in user_message.lower() and merchant_filter:
                query = {
                    "account_number": account_number,
                    "description": {"$regex": merchant_filter, "$options": "i"}
                }
                
                transactions = list(self.collection.find(query).sort("date", -1).limit(10))
                
                context_state = f"User asked for other {merchant_filter} transactions"
                data = {"merchant": merchant_filter, "transactions": transactions, "total_found": len(transactions)}
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
            # Fallback to complex analysis if we can't handle it simply
            return None
            
        except Exception as e:
            logger.error(f"Error in simple contextual query: {e}")
            return None

    def _extract_date_from_context(self, chat_history: List) -> Optional[Dict[str, Any]]:
        """Extract date context from recent conversation."""
        if not chat_history:
            return None
        
        # Look for date mentions in recent AI responses
        for msg in reversed(chat_history[-4:]):
            content = msg.content.lower()
            
            # Look for specific dates like "may 15th", "may 15"
            import re
            date_patterns = [
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',
                r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
                r'may\s+15',  # Specific to your example
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    # For May 15th example, return specific date filter
                    if "may" in content and "15" in content:
                        return {
                            "$gte": datetime(2025, 5, 15),
                            "$lte": datetime(2025, 5, 15, 23, 59, 59)
                        }
        
        return None

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

    async def process_query(self, user_message: str, account_number: str, first_name: str) -> str:
        """Enhanced process query with natural reasoning layer."""
        memory = self.get_user_memory(account_number)

        # Natural reasoning layer first
        reasoning_result = await self._reason_about_query(user_message, memory, account_number, first_name)
        
        logger.info(f"Query: '{user_message}' -> Action: {reasoning_result.get('action_needed')} -> Analysis: {reasoning_result.get('analysis_type', 'N/A')}")
        
        # Handle simple contextual queries directly
        if reasoning_result.get("action_needed") == "simple_contextual":
            simple_response = await self._handle_simple_contextual_query(
                user_message, account_number, first_name, reasoning_result, memory
            )
            if simple_response:  # If we successfully handled it simply
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(simple_response)
                return simple_response
        
        # Handle direct answers without database queries
        if reasoning_result.get("action_needed") == "direct_answer":
            context_state = "User sent a greeting or general question, no specific banking data needed"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            response = await self.generate_natural_response(context_state, None, user_message, first_name, conversation_history)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response
        
        # Handle balance checks
        if reasoning_result.get("action_needed") == "balance_check":
            response = await self._handle_balance_inquiry(account_number, first_name, user_message, memory)
            return response

        # Handle transaction history requests
        if reasoning_result.get("action_needed") == "transaction_history":
            response = await self._handle_transaction_history(user_message, account_number, first_name, reasoning_result, memory)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # Handle sophisticated analysis with real data
        if reasoning_result.get("action_needed") == "sophisticated_analysis":
            logger.info(f"Routing to sophisticated analysis: {reasoning_result.get('analysis_type')}")
            response = await self._handle_sophisticated_analysis(
                user_message, account_number, first_name, reasoning_result, memory
            )
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response

        # Fallback to existing sophisticated logic for edge cases
        logger.info(f"Falling back to original complex pipeline system")
        contextual_analysis = self.analyze_contextual_query(user_message, account_number)
        
        if not contextual_analysis.is_complete and contextual_analysis.clarification_needed:
            # Generate natural clarification response
            context_state = "User's request is incomplete, need more information"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            response = await self.generate_natural_response(context_state, {"missing_info": contextual_analysis.missing_info}, user_message, first_name, conversation_history)
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            return response
        
        # Use resolved query if available, otherwise use original
        query_to_process = contextual_analysis.resolved_query or user_message

        # Analyze the query (resolved or original)
        query_analysis = await self._analyze_query(query_to_process, account_number)

        # Validate pipeline
        if query_analysis.pipeline:
            try:
                jsonschema.validate(query_analysis.pipeline, PIPELINE_SCHEMA)
            except jsonschema.ValidationError as e:
                context_state = "Pipeline validation error occurred"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                response = await self.generate_natural_response(context_state, {"error": "pipeline_validation"}, user_message, first_name, conversation_history)
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response)
                return response

        # Execute appropriate action
        response = None
        
        if query_analysis.intent == "balance_inquiry":
            response = await self._handle_balance_inquiry(account_number, first_name, user_message, memory)
        elif query_analysis.intent in ["transaction_history", "spending_analysis", "category_spending"]:
            response = await self._handle_data_query(account_number, query_analysis, query_to_process, first_name, memory)
        elif query_analysis.intent == "transfer_money":
            response = await self._handle_money_transfer(account_number, query_analysis, query_to_process, first_name, memory)
        else:
            context_state = "General banking assistance needed"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            response = await self.generate_natural_response(context_state, None, query_to_process, first_name, conversation_history)

        # Add conversation to memory
        if response:
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)

        return response

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

    async def _analyze_spending_patterns(self, user_message: str, account_number: str, 
                                       first_name: str, reasoning: Dict[str, Any], 
                                       memory: ConversationBufferMemory) -> str:
        """Default sophisticated spending pattern analysis."""
        try:
            # Get current date dynamically
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            
            # Get recent spending trends
            monthly_data = {}
            
            for month_offset in range(3):  # Last 3 months
                target_month = current_month - month_offset
                target_year = current_year
                
                if target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
                month_name = datetime(target_year, target_month, 1).strftime("%B")
                
                days_in_month = 31 if target_month in [1,3,5,7,8,10,12] else 30
                if target_month == 2:
                    days_in_month = 29 if target_year % 4 == 0 else 28
                
                query = {
                    "account_number": account_number,
                    "type": "debit",
                    "date": {
                        "$gte": datetime(target_year, target_month, 1),
                        "$lte": datetime(target_year, target_month, days_in_month, 23, 59, 59)
                    }
                }
                
                transactions = list(self.collection.find(query))
                month_total = sum(tx["transaction_amount"] for tx in transactions)
                monthly_data[month_name] = month_total
            
            context_state = "User asking for general spending analysis, providing recent spending trends"
            data = {
                "monthly_data": monthly_data,
                "months_analyzed": len(monthly_data)
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Spending patterns error: {e}")
            context_state = "Error occurred while analyzing spending patterns"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _analyze_monthly_comparison(self, user_message: str, account_number: str, 
                                        first_name: str, reasoning: Dict[str, Any], 
                                        memory: ConversationBufferMemory) -> str:
        """Compare spending across multiple months with real data."""
        try:
            # Get current date dynamically
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            
            # Get spending for last 6 months
            monthly_spending = {}
            
            for month_offset in range(6):
                # Calculate the month (working backwards from current month)
                target_month = current_month - month_offset
                target_year = current_year
                
                if target_month <= 0:
                    target_month += 12
                    target_year -= 1
                
                month_name = datetime(target_year, target_month, 1).strftime("%B %Y")
                
                # Get days in month
                if target_month in [4, 6, 9, 11]:
                    days_in_month = 30
                elif target_month == 2:
                    days_in_month = 29 if target_year % 4 == 0 else 28
                else:
                    days_in_month = 31
                
                # Query for this month
                query = {
                    "account_number": account_number,
                    "type": "debit",
                    "date": {
                        "$gte": datetime(target_year, target_month, 1),
                        "$lte": datetime(target_year, target_month, days_in_month, 23, 59, 59)
                    }
                }
                
                transactions = list(self.collection.find(query))
                month_total = sum(tx["transaction_amount"] for tx in transactions)
                monthly_spending[month_name] = month_total
            
            # Prepare data for natural response
            spending_values = list(monthly_spending.values())
            avg_spending = sum(spending_values) / len(spending_values) if spending_values else 0
            
            # Find highest and lowest months
            max_month = max(monthly_spending.items(), key=lambda x: x[1]) if monthly_spending else ("Unknown", 0)
            min_month = min(monthly_spending.items(), key=lambda x: x[1]) if monthly_spending else ("Unknown", 0)
            
            context_state = "User requested monthly spending comparison, providing 6 months of spending data with analysis"
            data = {
                "monthly_spending": monthly_spending,
                "average": avg_spending,
                "highest_month": max_month,
                "lowest_month": min_month,
                "total_months": len(monthly_spending)
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Monthly comparison error: {e}")
            context_state = "Error occurred while analyzing monthly spending comparison"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _analyze_category_spending(self, user_message: str, account_number: str, 
                                       first_name: str, reasoning: Dict[str, Any], 
                                       memory: ConversationBufferMemory) -> str:
        """Analyze spending by specific categories with real data."""
        try:
            # Let LLM determine which categories to analyze based on the user message
            category_prompt = f"""
            The user asked: "{user_message}"
            
            Based on this query, which spending categories should I analyze? Common categories include:
            entertainment, dining, transport, food, shopping, travel, utilities, etc.
            
            If they're asking about cutting expenses, include the main spending categories.
            If they mention specific categories, focus on those.
            
            Return JSON: {{"categories": ["category1", "category2", "category3"]}}
            """
            
            try:
                category_response = await llm.ainvoke([SystemMessage(content=category_prompt)])
                category_result = self.extract_json_from_response(category_response.content)
                categories_needed = category_result.get("categories", ["entertainment", "dining", "transport"]) if category_result else ["entertainment", "dining", "transport"]
            except:
                categories_needed = ["entertainment", "dining", "transport"]  # fallback
            
            # Map user terms to database categories
            category_mapping = {
                "entertainment": ["Entertainment", "Streaming", "Movies", "Games"],
                "dining": ["Food", "Restaurant", "Fast Food", "Dining"],
                "uber": ["Transport", "Taxi", "Ride"],
                "transport": ["Transport", "Taxi", "Uber", "Careem", "Ride"],
                "food": ["Food", "Restaurant", "Grocery", "Dining"],
                "shopping": ["Shopping", "Retail", "Online", "Store"],
                "travel": ["Travel", "Hotel", "Flight", "Vacation"],
                "utilities": ["Utilities", "Electric", "Gas", "Water"]
            }
            
            # Get current date dynamically
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            
            # Get current month spending for each category
            current_month_data = {}
            total_monthly_spending = 0
            
            for user_category in categories_needed:
                db_categories = category_mapping.get(user_category.lower(), [user_category])
                
                category_total = 0
                for db_cat in db_categories:
                    # Query for current month
                    query = {
                        "account_number": account_number,
                        "type": "debit",
                        "date": {
                            "$gte": datetime(current_year, current_month, 1),
                            "$lte": datetime(current_year, current_month, 31, 23, 59, 59)
                        }
                    }
                    
                    # Add category or description filter
                    if user_category.lower() in ["uber", "careem"]:
                        query["description"] = {"$regex": user_category, "$options": "i"}
                    else:
                        query["category"] = {"$regex": db_cat, "$options": "i"}
                    
                    transactions = list(self.collection.find(query))
                    category_spending = sum(tx["transaction_amount"] for tx in transactions)
                    category_total += category_spending
                
                current_month_data[user_category] = category_total
                total_monthly_spending += category_total
            
            # Get total monthly spending for context
            total_query = {
                "account_number": account_number,
                "type": "debit",
                "date": {
                    "$gte": datetime(current_year, current_month, 1),
                    "$lte": datetime(current_year, current_month, 31, 23, 59, 59)
                }
            }
            all_transactions = list(self.collection.find(total_query))
            total_month_spending = sum(tx["transaction_amount"] for tx in all_transactions)
            
            # Determine context based on query type
            if "cut" in user_message.lower():
                context_state = "User asking about cutting expenses in specific categories, providing savings potential analysis"
            else:
                context_state = "User asking about category spending breakdown"
            
            data = {
                "category_spending": current_month_data,
                "total_category_spending": total_monthly_spending,
                "total_month_spending": total_month_spending,
                "categories_analyzed": categories_needed,
                "is_cutting_query": "cut" in user_message.lower()
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                
        except Exception as e:
            logger.error(f"Category analysis error: {e}")
            context_state = "Error occurred while analyzing category spending"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _analyze_savings_planning(self, user_message: str, account_number: str, 
                                      first_name: str, reasoning: Dict[str, Any], 
                                      memory: ConversationBufferMemory) -> str:
        """Provide intelligent savings planning based on real spending patterns."""
        try:
            # Get current balance
            async with httpx.AsyncClient() as client:
                balance_response = await client.post(
                    f"{self.backend_url}/user_balance",
                    json={"account_number": account_number}
                )
                balance_data = balance_response.json()
                current_balance = balance_data["user"]["current_balance_pkr"]
            
            # Extract target amount from user message
            import re
            amount_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', user_message.replace(',', ''))
            target_amount = float(amount_match.group(1).replace(',', '')) if amount_match else 1000000
            
            # Get current date and calculate time to target month
            current_date = datetime.now()
            current_month = current_date.month
            current_year = current_date.year
            
            # Get current month spending patterns
            current_month_query = {
                "account_number": account_number,
                "type": "debit",
                "date": {
                    "$gte": datetime(current_year, current_month, 1),
                    "$lte": datetime(current_year, current_month, 31, 23, 59, 59)
                }
            }
            
            current_transactions = list(self.collection.find(current_month_query))
            current_monthly_spending = sum(tx["transaction_amount"] for tx in current_transactions)
            
            # Calculate what's needed
            amount_needed = target_amount - current_balance
            
            # Calculate months to target (dynamically based on target mentioned)
            months_to_target = 1  # Default
            if "august" in user_message.lower():
                if current_month == 7:  # July
                    months_to_target = 1
                elif current_month < 8:
                    months_to_target = 8 - current_month
                else:
                    months_to_target = (12 - current_month) + 8  # Next year
            
            # Analyze by category for cutting suggestions
            category_spending = {}
            categories = ["Entertainment", "Food", "Transport", "Shopping", "Utilities"]
            
            for category in categories:
                cat_transactions = [tx for tx in current_transactions 
                                 if category.lower() in tx.get("category", "").lower()]
                cat_total = sum(tx["transaction_amount"] for tx in cat_transactions)
                if cat_total > 0:
                    category_spending[category] = cat_total
            
            context_state = "User asking for savings planning to reach financial target, providing comprehensive analysis"
            data = {
                "current_balance": current_balance,
                "target_amount": target_amount,
                "amount_needed": amount_needed,
                "months_to_target": months_to_target,
                "current_monthly_spending": current_monthly_spending,
                "category_breakdown": category_spending,
                "is_achievable": amount_needed <= current_monthly_spending * months_to_target
            }
            
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
            
        except Exception as e:
            logger.error(f"Savings planning error: {e}")
            context_state = "Error occurred while analyzing savings plan"
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
            # Fallback: Try to get basic spending data for the month at least
            try:
                # Simple fallback query
                current_date = datetime.now()
                target_month = 5  # May
                target_year = current_date.year
                
                simple_query = {
                    "account_number": account_number,
                    "type": "debit",
                    "date": {
                        "$gte": datetime(target_year, target_month, 1),
                        "$lte": datetime(target_year, target_month, 31, 23, 59, 59)
                    }
                }
                
                simple_transactions = list(self.collection.find(simple_query))
                simple_total = sum(tx.get("transaction_amount", 0) for tx in simple_transactions)
                
                context_state = "Error occurred during detailed analysis but providing basic spending information"
                data = {
                    "month": "May",
                    "total_spent": simple_total,
                    "error_occurred": True,
                    "transaction_count": len(simple_transactions)
                }
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                context_state = "Error occurred while analyzing spending breakdown"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

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
                
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(response_text)
                return response_text
                
        except Exception as e:
            logger.error({"action": "handle_balance_inquiry", "error": str(e)})
            context_state = "Error occurred while retrieving balance information"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    # Keep existing methods for compatibility with fallback system
    def analyze_contextual_query(self, user_message: str, account_number: str) -> ContextualQuery:
        """Analyze if query needs context using conversation history."""
        memory = self.get_user_memory(account_number)
        
        # Get conversation history
        chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        
        # First, use LLM to detect if this is a contextual query
        contextual_detection_result = self._detect_contextual_reference_with_llm(user_message, chat_history)
    
        # If no contextual reference detected, analyze as standalone query
        if not contextual_detection_result["is_contextual"]:
            return self._analyze_standalone_query(user_message)
    
        # If contextual reference detected but no previous conversation exists
        if not chat_history:
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="I don't have any previous conversation to reference. Could you please provide the complete information for your request?"
            )
    
        # Try to resolve the query with context
        try:
            resolved_query = self._resolve_contextual_query_with_llm(user_message, chat_history)
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=True,
                resolved_query=resolved_query
            )
        except Exception as e:
            logger.error(f"Error resolving contextual query: {e}")
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="I couldn't understand the context. Could you please provide the complete information for your request?"
            )

    def _detect_contextual_reference_with_llm(self, user_message: str, chat_history: List) -> Dict[str, Any]:
        """Use LLM to detect if user query references previous context."""
        context_summary = "No previous conversation available"
        if chat_history:
            recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
            context_summary = "\n".join([
                f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}" 
                for i, msg in enumerate(recent_messages)
            ])
    
        contextual_detection_prompt = f"""
        Analyze if the current user query is referencing or building upon previous conversation context.

        Current Query: "{user_message}"
        Previous Conversation: {context_summary}

        A query is contextual if it:
        1. References previous results (e.g., "from this", "from that data", "those transactions")
        2. Uses pronouns that refer to previous content (e.g., "them", "these", "it")
        3. Asks for filtering/drilling down into previous results
        4. Uses relative terms that depend on previous context
        5. Asks follow-up questions that only make sense with previous context

        Return JSON: {{"is_contextual": true/false, "confidence": 0.0-1.0, "reasoning": "explanation"}}
        """
    
        try:
            response = llm.invoke([SystemMessage(content=contextual_detection_prompt)])
            result = self.extract_json_from_response(response.content)
        
            if result and isinstance(result, dict):
                return result
            else:
                return {"is_contextual": False, "confidence": 0.5, "reasoning": "Could not parse LLM response"}
            
        except Exception as e:
            logger.error(f"Error in contextual detection with LLM: {e}")
            return self._fallback_trigger_word_detection(user_message)

    def _fallback_trigger_word_detection(self, user_message: str) -> Dict[str, Any]:
        """Fallback method using trigger words if LLM fails."""
        context_phrases = [
            "from this", "from that", "out of this", "out of that", "from the above",
            "from these", "from those", "of this", "of that", "in this", "in that",
            "them", "these", "those", "it", "they", "break it down", "filter them",
            "show me the", "which ones", "the highest", "the lowest", "the recent ones"
        ]
    
        has_reference = any(phrase in user_message.lower() for phrase in context_phrases)
    
        return {
            "is_contextual": has_reference,
            "confidence": 0.7 if has_reference else 0.8,
            "reasoning": f"Trigger word detection: {'found' if has_reference else 'not found'} contextual phrases"
        }

    def _resolve_contextual_query_with_llm(self, user_message: str, chat_history: List) -> str:
        """Enhanced contextual query resolution using LLM."""
        recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
        conversation_context = "\n".join([
            f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}" 
            for i, msg in enumerate(recent_messages)
        ])
    
        resolution_prompt = f"""
        You are helping resolve a contextual banking query. The user is referencing previous conversation context.
        Current User Query: "{user_message}"
        Previous Conversation: {conversation_context}
        
        Create a complete, standalone query that combines current query with previous context.
        Return ONLY the resolved query as a plain string, no JSON or formatting.
        """
    
        try:
            response = llm.invoke([SystemMessage(content=resolution_prompt)])
            resolved_query = response.content.strip().strip('"\'')
            return resolved_query
        except Exception as e:
            logger.error(f"Error resolving contextual query with LLM: {e}")
            raise e

    def _analyze_standalone_query(self, user_message: str) -> ContextualQuery:
        """Simplified analysis - only check transfers for completeness."""
        if any(word in user_message.lower() for word in ["transfer", "send", "pay", "wire", "remit"]):
            completeness_prompt = f"""
            Analyze this transfer query for completeness:
            Query: "{user_message}"
            
            Check if the query has:
            1. Amount (e.g., 500, 1000 PKR, 50 USD, half of that, etc.)
            2. Recipient (e.g., "to John", "to account 1234", etc.)
            
            Return JSON: {{"is_complete": true/false, "missing_info": ["amount", "recipient"], "clarification_needed": "What to ask for"}}
            """
        
            try:
                response = llm.invoke([SystemMessage(content=completeness_prompt)])
                result = self.extract_json_from_response(response.content)
            
                if result:
                    return ContextualQuery(
                        needs_context=False,
                        has_reference=False,
                        is_complete=result.get("is_complete", True),
                        missing_info=result.get("missing_info", []),
                        clarification_needed=result.get("clarification_needed")
                    )
            except Exception as e:
                logger.error(f"Error analyzing transfer completeness: {e}")
    
        return ContextualQuery(is_complete=True)
    
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
    
    def extract_filters_with_llm(self, user_message: str) -> FilterExtraction:
        """Use LLM to extract filters from user query with new dataset structure."""
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

    def detect_intent_fallback(self, user_message: str) -> tuple[str, List[Dict[str, Any]]]:
        """Improved fallback intent detection using LLM filter extraction."""
        filters = self.extract_filters_with_llm(user_message)
        intent = self.detect_intent_from_filters(user_message, filters)
        pipeline = self.generate_pipeline_from_filters(filters, intent, "{{account_number}}")
        
        return intent, pipeline

    def replace_account_number_in_pipeline(self, pipeline: List[Dict[str, Any]], account_number: str) -> List[Dict[str, Any]]:
        """Recursively replace {{account_number}} placeholder in pipeline."""
        def replace_in_dict(obj):
            if isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace("{{account_number}}", account_number)
            else:
                return obj
        
        return replace_in_dict(pipeline)

    async def _analyze_query(self, user_message: str, account_number: str) -> QueryResult:
        """Use LLM to analyze query and generate MongoDB pipeline."""
        try:
            intent, pipeline = self.detect_intent_fallback(user_message)
            if intent != "general":
                pipeline = self.replace_account_number_in_pipeline(pipeline, account_number)
                return QueryResult(intent=intent, pipeline=pipeline)
        except Exception as e:
            logger.error({
                "action": "fallback_intent_detection",
                "error": str(e),
                "user_message": user_message
            })

        try:
            response = llm.invoke([
                SystemMessage(content=query_prompt.format(
                    user_message=user_message,
                    current_date=datetime.now().strftime("%Y-%m-%d")
                ))
            ])

            result = self.extract_json_from_response(response.content)
            if result is None:
                return QueryResult(intent="general", pipeline=[])

            if not isinstance(result, dict) or "intent" not in result:
                return QueryResult(intent="general", pipeline=[])

            pipeline = self.replace_account_number_in_pipeline(result.get("pipeline", []), account_number)

            query_result = QueryResult(
                intent=result.get("intent", "general"),
                pipeline=pipeline,
                response_format=result.get("response_format", "natural_language")
            )
            return query_result
        except Exception as e:
            logger.error({
                "action": "analyze_query",
                "error": str(e),
                "user_message": user_message
            })
            return QueryResult(intent="general", pipeline=[])

    async def _handle_data_query(self, account_number: str, query_analysis: QueryResult, user_message: str, 
                                first_name: str, memory: ConversationBufferMemory) -> str:
        """Enhanced data query handling with natural responses."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/execute_pipeline",
                    json={"account_number": account_number, "pipeline": query_analysis.pipeline}
                )
                response.raise_for_status()
                data = response.json()
                
                context_state = f"User requested {query_analysis.intent}, executed database query and returning results"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                
                formatted_response_content = await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)
                return formatted_response_content
                
        except Exception as e:
            logger.error({"action": "handle_data_query", "error": str(e)})
            context_state = "Error occurred while processing data query"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    async def _handle_money_transfer(self, account_number: str, query_analysis: QueryResult, user_message: str, 
                                   first_name: str, memory: ConversationBufferMemory) -> str:
        """Enhanced money transfer with natural conversation flow."""
        try:
            # Enhanced transfer parsing with contextual amount support
            transfer_prompt_enhanced = f"""
            Extract transfer details from this query, handling contextual references:
            Query: "{user_message}"
            
            Context: Previous conversation may have mentioned amounts. Handle phrases like:
            - "I want to transfer 1000 PKR" (amount specified, recipient missing)
            - "transfer half of that" (where "that" refers to a previous amount)
            - "send 50% of PKR 134,761.10"
            - "transfer $20 to John"
            
            Extract:
            - amount: number (calculate if percentage/fraction given)
            - currency: "PKR" or "USD" (default PKR if not specified)
            - recipient: string (null if not mentioned)
            - has_amount: boolean (true if amount specified)
            - has_recipient: boolean (true if recipient specified)
            
            Return JSON: {{"amount": number, "currency": string, "recipient": string, "has_amount": boolean, "has_recipient": boolean}}
            """
            
            response = llm.invoke([SystemMessage(content=transfer_prompt_enhanced)])
            transfer_details = self.extract_json_from_response(response.content)
            
            if transfer_details is None:
                context_state = "Transfer details could not be understood, need clarification"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, {"missing": "transfer_details"}, user_message, first_name, conversation_history)

            # Check what information is missing
            missing_parts = []
            
            if not transfer_details.get("has_amount") or not transfer_details.get("amount"):
                missing_parts.append("amount")
            
            if not transfer_details.get("has_recipient") or not transfer_details.get("recipient"):
                missing_parts.append("recipient")
            
            # Handle missing information naturally
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
            
            # All information available - process the transfer
            amount = transfer_details.get("amount")
            currency = transfer_details.get("currency", "PKR")
            recipient = transfer_details.get("recipient")
            
            if amount <= 0:
                context_state = "Transfer amount invalid (zero or negative)"
                data = {"amount": amount, "recipient": recipient}
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                return await self.generate_natural_response(context_state, data, user_message, first_name, conversation_history)

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
                
                context_state = "Transfer processed successfully" if transfer_result.get("status") == "success" else "Transfer failed"
                conversation_history = self._get_context_summary(memory.chat_memory.messages)
                
                return await self.generate_natural_response(context_state, transfer_result, user_message, first_name, conversation_history)
                
        except Exception as e:
            logger.error({"action": "handle_money_transfer", "error": str(e)})
            context_state = "Error occurred during money transfer"
            conversation_history = self._get_context_summary(memory.chat_memory.messages)
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_message, first_name, conversation_history)

    # Session Management Methods
    async def handle_account_selection(self, user_input: str, available_accounts: List[str], first_name: str) -> str:
        """Handle account selection with natural LLM response."""
        try:
            # Check if user entered last 4 digits of account
            if user_input.isdigit() and len(user_input) == 4:
                # Find account that ends with these 4 digits
                selected_account = None
                for account in available_accounts:
                    if account.endswith(user_input):
                        selected_account = account
                        break
                
                if selected_account:
                    context_state = "User successfully selected account by last 4 digits, ready for banking operations"
                    data = {
                        "selected_account": selected_account,
                        "account_number": selected_account,
                        "input_method": "last_4_digits",
                        "masked_number": f"***-***-{selected_account[-4:]}"
                    }
                    
                    return await self.generate_natural_response(context_state, data, user_input, first_name)
                else:
                    context_state = "User entered 4 digits but no matching account found"
                    data = {"available_accounts": available_accounts, "invalid_digits": user_input}
                    return await self.generate_natural_response(context_state, data, user_input, first_name)
            else:
                context_state = "User provided invalid account selection format"
                data = {"available_accounts": available_accounts, "expected_format": "last_4_digits"}
                return await self.generate_natural_response(context_state, data, user_input, first_name)
                
        except Exception as e:
            logger.error(f"Error in account selection: {e}")
            context_state = "Error occurred during account selection"
            return await self.generate_natural_response(context_state, {"error": str(e)}, user_input, first_name)

    async def handle_account_display(self, accounts: List[str], first_name: str) -> str:
        """Display available accounts with natural LLM response."""
        try:
            context_state = "Showing user their available accounts for selection using last 4 digits"
            data = {
                "accounts": accounts,
                "total_accounts": len(accounts),
                "selection_method": "last_4_digits",
                "account_options": [{"full": acc, "last_4": acc[-4:]} for acc in accounts]
            }
            
            return await self.generate_natural_response(context_state, data, "", first_name)
            
        except Exception as e:
            logger.error(f"Error displaying accounts: {e}")
            context_state = "Error occurred while displaying accounts"
            return await self.generate_natural_response(context_state, {"error": str(e)}, "", first_name)

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
            context_state = "CNIC verification successful, user has multiple accounts, need account selection"
            data = {
                "verification_status": "success",
                "user_name": user_name,
                "accounts_found": len(accounts),
                "accounts": accounts,
                "next_step": "account_selection",
                "selection_method": "last_4_digits"
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

    # Utility Methods
    def clear_user_memory(self, account_number: str) -> None:
        """Clear conversation memory for a specific user account."""
        if account_number in self.user_memories:
            del self.user_memories[account_number]
            logger.info(f"Cleared memory for account: {account_number}")

    def get_conversation_summary(self, account_number: str) -> str:
        """Get a summary of the current conversation."""
        memory = self.get_user_memory(account_number)
        return self._get_context_summary(memory.chat_memory.messages)

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