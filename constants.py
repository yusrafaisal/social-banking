"""
Constants for Banking AI Assistant
Centralizes all magic strings, numbers, and configuration values
"""

# ===== VERIFICATION STAGES =====
class VerificationStages:
    NOT_VERIFIED = "not_verified"
    CNIC_VERIFIED = "cnic_verified"
    OTP_PENDING = "otp_pending"
    OTP_VERIFIED = "otp_verified"
    ACCOUNT_SELECTED = "account_selected"
    TRANSFER_OTP_PENDING = "transfer_otp_pending"
    TRANSFER_CONFIRMATION_PENDING = "transfer_confirmation_pending"

# ===== BANKING INTENTS =====
class BankingIntents:
    BALANCE_INQUIRY = "balance_inquiry"
    TRANSACTION_HISTORY = "transaction_history"
    SPENDING_ANALYSIS = "spending_analysis"
    CATEGORY_SPENDING = "category_spending"
    TRANSFER_MONEY = "transfer_money"
    GENERAL = "general"

# ===== TRANSACTION TYPES =====
class TransactionTypes:
    DEBIT = "debit"
    CREDIT = "credit"

# ===== CURRENCIES =====
class Currencies:
    PKR = "PKR"
    USD = "USD"
    
    # Lowercase versions for database
    PKR_LOWER = "pkr"
    USD_LOWER = "usd"
    
    # All supported currencies
    SUPPORTED = [PKR, USD]
    SUPPORTED_LOWER = [PKR_LOWER, USD_LOWER]

# ===== DATABASE FIELDS =====
class DatabaseFields:
    # User fields
    NAME = "name"
    CNIC = "cnic"
    ACCOUNT_NUMBER = "account_number"
    
    # Transaction fields
    DATE = "date"
    TYPE = "type"
    DESCRIPTION = "description"
    CATEGORY = "category"
    ACCOUNT_CURRENCY = "account_currency"
    AMOUNT_DEDUCTED_FROM_ACCOUNT = "amount_deducted_from_account"
    TRANSACTION_AMOUNT = "transaction_amount"
    TRANSACTION_CURRENCY = "transaction_currency"
    ACCOUNT_BALANCE = "account_balance"

# ===== SYSTEM LIMITS =====
class Limits:
    # Memory and context
    MAX_RECENT_MESSAGES = 6
    MAX_CONTEXT_LENGTH = 5000
    MAX_RESPONSE_CONTENT = 500
    
    # Transaction limits
    DEFAULT_TRANSACTION_LIMIT = 20
    MAX_TRANSACTION_LIMIT = 100
    MONTH_TRANSACTION_LIMIT = 50
    
    # OTP limits
    OTP_MIN_DIGITS = 1
    OTP_MAX_DIGITS = 5
    
    # Rate limiting
    MESSAGE_RATE_LIMIT_SECONDS = 2
    
    # Session limits
    SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
    MAX_PROCESSED_MESSAGES = 1000

# ===== RESPONSE FORMATS =====
class ResponseFormats:
    CONCISE_ONE_LINER = "CONCISE_ONE_LINER"
    STRUCTURED_LIST = "STRUCTURED_LIST"
    DETAILED_EXPLANATION = "DETAILED_EXPLANATION"
    HELPFUL_GUIDANCE = "HELPFUL_GUIDANCE"

# ===== CONTEXT STATES =====
class ContextStates:
    # Session management
    SESSION_STARTING = "New user starting banking session, need CNIC verification to begin"
    SESSION_ENDING = "User ending banking session, providing secure farewell and cleanup confirmation"
    
    # Authentication
    CNIC_VERIFICATION_SUCCESS = "CNIC verification successful, now requesting OTP for additional security"
    CNIC_VERIFICATION_FAILED = "CNIC verification failed, user needs to try again with correct format"
    INVALID_CNIC_FORMAT = "User provided invalid CNIC format, need guidance on correct format"
    
    # OTP
    OTP_VERIFICATION_SUCCESS = "OTP verification successful, user can now select account"
    OTP_VERIFICATION_FAILED = "OTP verification failed, need valid 1-5 digit number"
    
    # Account selection
    ACCOUNT_SELECTED = "Account successfully selected and confirmed, user ready for full banking operations"
    ACCOUNT_SELECTION_UNCLEAR = "User account selection unclear, providing helpful guidance with available options"
    
    # Transfer
    TRANSFER_DETAILS_COLLECTED = "Transfer details collected, requesting OTP for security"
    TRANSFER_OTP_VERIFIED = "Transfer OTP verified, now requesting user confirmation before executing transfer"
    TRANSFER_EXECUTED_SUCCESS = "Transfer executed successfully after OTP verification"
    TRANSFER_EXECUTED_FAILED = "Transfer failed after OTP verification"
    TRANSFER_CANCELLED = "User cancelled transfer during confirmation step, providing cancellation confirmation"
    
    # General banking
    BALANCE_INQUIRY = "User requesting current account balance"
    TRANSACTION_HISTORY = "User requested transaction history, providing recent transactions"
    SPENDING_ANALYSIS = "User requested spending analysis, providing spending data from pipeline results"
    
    # Errors
    ERROR_OCCURRED = "Error occurred while processing request"

# ===== CONFIRMATION WORDS =====
class ConfirmationWords:
    POSITIVE = [
        "yes", "y", "yeah", "yep", "yup", "ok", "okay", "confirm", "proceed",
        "go ahead", "continue", "sure", "definitely", "absolutely", "correct",
        "right", "true", "confirm it", "do it", "send it", "transfer it"
    ]
    
    NEGATIVE = [
        "no", "n", "nope", "cancel", "stop", "abort", "don't", "dont",
        "not", "wrong", "incorrect", "false", "refuse", "decline", "back"
    ]

# ===== GREETING WORDS =====
class GreetingWords:
    BASIC_GREETINGS = [
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "good day", "howdy", "what's up", "whats up", "sup",
        "hola", "bonjour", "namaste", "salaam", "salam", "assalam", "start"
    ]
    
    GENERAL_PHRASES = [
        "what can you do", "help me", "what do you do", "how can you help"
    ]

# ===== EXIT COMMANDS =====
class ExitCommands:
    COMMANDS = ['exit', 'quit', 'logout', 'end']

# ===== MONTHS =====
class Months:
    NAMES_TO_NUMBERS = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    DAYS_IN_MONTH = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    
    THIRTY_DAY_MONTHS = [4, 6, 9, 11]

# ===== REGEX PATTERNS =====
class RegexPatterns:
    CNIC_PATTERN = r'\b(\d{5}-\d{7}-\d)\b'
    OTP_PATTERN = r'^\d{1,5}$'
    JSON_BRACE_PATTERN = r'[{[]'
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r'^hi+$',  # hi, hii, hiii
        r'^hey+$',  # hey, heyy
        r'^hello+$',  # hello, helloo
        r'^good (morning|afternoon|evening|day)',
        r'^how are you',
        r'^what\'?s up'
    ]

# ===== BALANCE QUERY KEYWORDS =====
class BalanceKeywords:
    KEYWORDS = ["balance", "money", "amount", "funds"]

# ===== TRANSACTION KEYWORDS =====
class TransactionKeywords:
    KEYWORDS = ["transaction", "transactions", "last", "recent", "history", "may", "june", "july"]

# ===== LLM CONFIGURATION =====
class LLMConfig:
    MODEL_NAME = "gpt-4o"
    TEMPERATURE = 0.3
    MAX_TOKENS_OTP = 10
    TEMPERATURE_TRANSLATION = 0.1
    TIMEOUT_SECONDS = 60.0

# ===== API ENDPOINTS =====
class APIEndpoints:
    VERIFY_CNIC = "/verify_cnic"
    SELECT_ACCOUNT = "/select_account"
    USER_BALANCE = "/user_balance"
    EXECUTE_PIPELINE = "/execute_pipeline"
    TRANSFER_MONEY = "/transfer_money"
    PROCESS_QUERY = "/process_query"
    HEALTH = "/health"

# ===== HTTP STATUS MESSAGES =====
class StatusMessages:
    SUCCESS = "success"
    FAIL = "fail"
    ERROR = "error"
    HEALTHY = "healthy"

# ===== TRANSFER SIGNALS =====
class TransferSignals:
    OTP_REQUIRED_PREFIX = "OTP_REQUIRED"
    SEPARATOR = "|"

# # ===== SUPPORTED LANGUAGES =====
class Languages:
    ENGLISH = "en"
    URDU_ROMAN = "ur-roman"
    URDU_ARABIC = "ur"
    
    URDU_VARIANTS = [URDU_ROMAN, URDU_ARABIC]

# ===== MONGODB CONFIGURATION =====
class MongoConfig:
    DEFAULT_URI = "mongodb://localhost:27017/"
    DEFAULT_DB_NAME = "bank_database"
    TRANSACTIONS_COLLECTION = "transactions"
    
# ===== WEBHOOK CONFIGURATION =====
class WebhookConfig:
    VERIFY_TOKEN = "helloworld3"
    BACKEND_URL = "http://localhost:8000"
    FACEBOOK_API_VERSION = "v18.0"
    FACEBOOK_MESSAGES_ENDPOINT = "me/messages"



class BankingEmojis:
    """Emojis for consistent banking interface formatting."""
    # Security & Verification
    SECURITY = "🔒"
    VERIFICATION = "🔐" 
    SUCCESS = "✅"
    FAILURE = "❌"
    WARNING = "⚠️"
    
    # Banking & Money
    ACCOUNT = "💼"
    MONEY = "💰"
    CARD = "💳"
    TRANSFER = "💸"
    BANK = "🏦"
    
    # Data & Information
    DATE = "📅"
    CHART = "📊"
    LIST = "📋"
    TARGET = "🎯"
    LABEL = "🏷️"
    
    # Communication & Process
    PHONE = "📱"
    MESSAGE = "💬"
    PROCESSING = "🔄"
    LIGHTNING = "⚡"
