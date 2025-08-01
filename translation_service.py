from langdetect import detect
from googletrans import Translator, LANGUAGES
import logging
from dotenv import load_dotenv
import os
import re

# Import constants
from constants import (
    Languages, LLMConfig, RegexPatterns
)

logger = logging.getLogger(__name__)

load_dotenv()

class TranslationService:
    def __init__(self):
        self.translator = Translator()
        
        # Initialize OpenAI client only if API key is available
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                self.use_llm = True
                logger.info("OpenAI client initialized for language detection and translation")
            else:
                self.openai_client = None
                self.use_llm = False
                logger.warning("OPENAI_API_KEY not found, using fallback detection only")
        except ImportError:
            self.openai_client = None
            self.use_llm = False
            logger.warning("OpenAI package not installed, using fallback detection only")
    
    def detect_language_with_llm(self, text: str) -> str:
        """Use LLM to accurately detect language including Roman Urdu."""
        if not self.use_llm or not self.openai_client:
            return self.fallback_detection(text)
            
        try:
            # STRICT ENGLISH CHECK: If 80% of words are English, classify as English
            words = text.lower().split()
            total_words = len(words)
            
            if total_words > 0:
                # Common English words list (you can expand this)
                common_english_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'among', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
                    'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
                    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
                    'what', 'where', 'when', 'why', 'how', 'who', 'which', 'all', 'any', 'some', 'each',
                    'every', 'no', 'not', 'only', 'own', 'other', 'such', 'same', 'so', 'than', 'too',
                    'very', 'just', 'now', 'here', 'there', 'then', 'more', 'also', 'back', 'well',
                    'get', 'go', 'come', 'take', 'give', 'make', 'see', 'know', 'think', 'say', 'tell',
                    'want', 'need', 'like', 'work', 'look', 'use', 'find', 'feel', 'try', 'ask', 'seem',
                    'show', 'check', 'balance', 'account', 'transaction', 'transfer', 'money', 'send',
                    'last', 'recent', 'history', 'statement', 'payment', 'deposit', 'withdraw', 'bank',
                    'percent', 'percentage', 'food', 'restaurant', 'grocery', 'store', 'amazon', 'uber',
                    'foodpanda', 'mcdonalds'
                }
                
                # Check how many words are recognizably English
                english_word_count = 0
                for word in words:
                    # Remove punctuation for checking
                    clean_word = re.sub(r'[^\w]', '', word.lower())
                    
                    # Check if word is in common English words
                    if clean_word in common_english_words:
                        english_word_count += 1
                    # Check if word follows English patterns (contains common English letter combinations)
                    elif re.match(r'^[a-z]+$', clean_word) and len(clean_word) > 2:
                        # Check for common English endings and patterns
                        if (clean_word.endswith(('ing', 'tion', 'ed', 'er', 'ly', 'ty', 'ness', 'ment')) or
                            clean_word.startswith(('un', 'pre', 'dis', 'mis', 'over', 'under', 'out')) or
                            any(pattern in clean_word for pattern in ['th', 'ck', 'sh', 'ch', 'gh'])):
                            english_word_count += 1
                
                english_percentage = english_word_count / total_words
                
                # If 80% or more words are English, classify as English
                if english_percentage >= 0.8:
                    logger.info(f"STRICT ENGLISH CHECK: {english_percentage:.2%} English words detected, classifying as English: '{text}'")
                    return Languages.ENGLISH

            prompt = f"""You are a language detection expert. Analyze the following text and determine its language.

            Text: "{text}"

            Language Detection Rules:
            1. If text is STANDARD ENGLISH (proper English words and grammar) → return "{Languages.ENGLISH}"
            2. If text is ROMAN URDU (Urdu/Hindi words written in English letters) → return "{Languages.URDU_ROMAN}" 
            3. If text is URDU in Arabic script → return "{Languages.URDU_ARABIC}"
            4. If text is any other language → return the 2-letter ISO code (de, fr, es, ar, etc.)

            IMPORTANT: Even if the text contains non-English names (like "kainat", "ahmed", "hassan"), if the sentence structure and majority of words are English, classify it as ENGLISH.

            Examples:
            - "what are my last 3 transactions" → "{Languages.ENGLISH}" (Standard English)
            - "check my balance please" → "{Languages.ENGLISH}" (Standard English)  
            - "show me transaction history" → "{Languages.ENGLISH}" (Standard English)
            - "transfer 10 percent of the foodpanda transaction to kainat" → "{Languages.ENGLISH}" (English with name)
            - "send money to ahmed from my account" → "{Languages.ENGLISH}" (English with name)
            - "mera balance kya hai" → "{Languages.URDU_ROMAN}" (Roman Urdu)
            - "meine last mahine kitna khracha kiya" → "{Languages.URDU_ROMAN}" (Roman Urdu)
            - "account me kitna paisa hai" → "{Languages.URDU_ROMAN}" (Roman Urdu)
            - "balance check karo" → "{Languages.URDU_ROMAN}" (Roman Urdu)
            - "میرا بیلنس کیا ہے" → "{Languages.URDU_ARABIC}" (Arabic script Urdu)
            - "wie geht es dir" → "de" (German)

            Response format: Return ONLY the language code ({Languages.ENGLISH}, {Languages.URDU_ROMAN}, {Languages.URDU_ARABIC}, de, fr, etc.). Nothing else."""

            response = self.openai_client.chat.completions.create(
                model=LLMConfig.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLMConfig.MAX_TOKENS_OTP,
                temperature=0
            )
            
            detected_lang = response.choices[0].message.content.strip().lower()
            
            if detected_lang in LANGUAGES or detected_lang in Languages.URDU_VARIANTS:
                logger.info(f"LLM detected language '{detected_lang}' for text: '{text}'")
                return detected_lang
            else:
                logger.warning(f"LLM returned invalid language code '{detected_lang}', falling back")
                return self.fallback_detection(text)
                
        except Exception as e:
            logger.error(f"LLM language detection failed: {e}, falling back")
            return self.fallback_detection(text)
    
    def translate_with_llm(self, text: str, source_lang: str, target_lang: str) -> str:
        """Use LLM for accurate translation, especially for Roman Urdu."""
        if not self.use_llm or not self.openai_client:
            return self.translate_with_google(text, source_lang, target_lang)
        
        try:
            # Special handling for Roman Urdu to English
            if source_lang == Languages.URDU_ROMAN and target_lang == Languages.ENGLISH:
                prompt = f"""You are an expert Roman Urdu to English translator. Translate this text accurately while preserving all numbers, names, and banking terms.

                Text to translate: "{text}"

                Translation Rules:
                1. Keep ALL numbers EXACTLY as they are (8 stays 8, never change to 3 or any other number)
                2. Keep banking terms in English: transactions, balance, account, transfer, etc.
                3. Keep proper nouns and names unchanged
                4. Preserve CNIC format: 12345-1234567-1 stays exactly the same
                5. Common Roman Urdu translations:
                - "mera/meri" = "my"
                - "pichli/pichle/akhri" = "last/previous" 
                - "batao/dikhao" = "tell me/show me"
                - "kya hai" = "what is"
                - "kitna" = "how much"
                - "karo" = "do"
                - "check" = "check" (keep as is)

                Examples:
                - "meri pichli 8 transactions batao" → "tell me my last 8 transactions"
                - "balance check karo" → "check my balance"
                - "account me kitna paisa hai" → "how much money is in my account"

                Return ONLY the English translation, nothing else."""

            elif source_lang == Languages.ENGLISH and target_lang == Languages.URDU_ROMAN:
                prompt = f"""You are an expert English to Roman Urdu translator. Translate this COMPLETE English text to Roman Urdu (Urdu written in English letters) while preserving all numbers and technical terms. TRANSLATE THE ENTIRE TEXT - DO NOT TRUNCATE.

            Text to translate: "{text}"

            Translation Rules:
            1. Keep ALL numbers EXACTLY as they are
            2. Keep banking terms in English when commonly used: balance, account, transaction, grocery store, Amazon, Uber, etc.
            3. Use Roman Urdu (English letters) - DO NOT use Arabic script
            4. Preserve proper nouns and CNIC formats
            5. TRANSLATE THE COMPLETE TEXT - translate every single sentence and detail

            Examples:
            - "tell me my last 8 transactions" → "meri pichli 8 transactions batao"
            - "check my balance" → "mera balance check karo"
            - "Your current balance is $1,234.56" → "Aapka current balance $1,234.56 hai"
            - "On June 29, you spent $77.23" → "June 29 ko aap ne $77.23 kharch kiya"

            IMPORTANT: 
            - Use ONLY English letters (Roman script)
            - Translate the ENTIRE text completely
            - Keep it natural and conversational in Roman Urdu

            Return ONLY the complete Roman Urdu translation in English letters, nothing else."""

            elif source_lang == Languages.URDU_ARABIC and target_lang == Languages.ENGLISH:
                # Arabic script Urdu to English
                prompt = f"""You are an expert Urdu to English translator. Translate this Arabic script Urdu text accurately while preserving all numbers, names, and banking terms.

            Text to translate: "{text}"

            Translation Rules:
            1. Keep ALL numbers EXACTLY as they are
            2. Keep banking terms in English: transactions, balance, account, transfer, etc.
            3. Keep proper nouns and names unchanged
            4. Preserve CNIC format exactly

            Return ONLY the English translation, nothing else."""

            elif source_lang == Languages.ENGLISH and target_lang == Languages.URDU_ARABIC:
                # English to Arabic script Urdu
                prompt = f"""You are an expert English to Urdu translator. Translate this COMPLETE English text to natural Urdu in Arabic script while preserving all numbers and technical terms. TRANSLATE THE ENTIRE TEXT - DO NOT TRUNCATE.

                Text to translate: "{text}"

                Translation Rules:
                1. Keep ALL numbers EXACTLY as they are
                2. Keep banking terms in English when commonly used: balance, account, transaction, etc.
                3. Use natural Urdu grammar and vocabulary in Arabic script
                4. Preserve proper nouns and CNIC formats
                5. TRANSLATE THE COMPLETE TEXT - translate every single sentence and detail

                IMPORTANT: Translate the ENTIRE text completely. Do not stop in the middle.

                Return ONLY the complete Urdu translation in Arabic script, nothing else."""

            else:
                # For other language pairs, use a general prompt
                prompt = f"""Translate this COMPLETE text from {source_lang} to {target_lang}. Keep all numbers, names, and technical terms exactly as they are. TRANSLATE THE ENTIRE TEXT.

            Text: "{text}"

            Return only the complete translation."""

            response = self.openai_client.chat.completions.create(
                model=LLMConfig.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLMConfig.TEMPERATURE_TRANSLATION
            )
            
            translated = response.choices[0].message.content.strip()
            
            # Remove quotes if LLM adds them
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            
            logger.info(f"LLM translated '{text[:50]}...' from {source_lang} to {target_lang}: '{translated[:100]}...'")
            return translated
            
        except Exception as e:
            logger.error(f"LLM translation failed: {e}, falling back to Google")
            return self.translate_with_google(text, source_lang, target_lang)
    
    def translate_with_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """Fallback Google translation."""
        try:
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            return text
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Enhanced translation to English with LLM priority."""
        try:
            if source_lang == Languages.ENGLISH:
                return text
            
            # Don't translate number-only text
            if self.is_number_only_text(text):
                logger.info(f"Skipping translation for number-only text: '{text}'")
                return text
            
            # Use LLM for better translation, especially Roman Urdu and Arabic Urdu
            if self.use_llm and source_lang in Languages.URDU_VARIANTS:
                return self.translate_with_llm(text, source_lang, Languages.ENGLISH)
            else:
                return self.translate_with_google(text, source_lang, Languages.ENGLISH)
            
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Enhanced translation from English with LLM priority."""
        try:
            if target_lang == Languages.ENGLISH:
                return text
            
            # Use LLM for better translation, especially to Urdu variants
            if self.use_llm and target_lang in Languages.URDU_VARIANTS:
                return self.translate_with_llm(text, Languages.ENGLISH, target_lang)
            else:
                return self.translate_with_google(text, Languages.ENGLISH, target_lang)
            
        except Exception as e:
            logger.error(f"Translation from English failed: {e}")
            return text
    
    def detect_language_smart(self, text: str, sender_id: str = None, get_last_language_func=None) -> str:
        """Smart language detection with LLM priority and number handling."""
        try:
            if len(text.strip()) < 3:
                if sender_id and get_last_language_func:
                    last_lang = get_last_language_func(sender_id)
                    if last_lang != Languages.ENGLISH:
                        logger.info(f"Short text detected, using last language: {last_lang}")
                        return last_lang
                return Languages.ENGLISH
            
            # Check if text is number-only
            if self.is_number_only_text(text):
                if sender_id and get_last_language_func:
                    last_lang = get_last_language_func(sender_id)
                    logger.info(f"Number-only text detected: '{text}', using last language: {last_lang}")
                    return last_lang
                else:
                    return Languages.ENGLISH
            
            # Use LLM for detection if available
            if self.use_llm:
                detected = self.detect_language_with_llm(text)
                logger.info(f"LLM detection result: '{detected}' for text: '{text}'")
            else:
                detected = self.fallback_detection(text)
                logger.info(f"Fallback detection result: '{detected}' for text: '{text}'")
            
            if detected in LANGUAGES or detected in Languages.URDU_VARIANTS:
                return detected
            else:
                logger.warning(f"Detected language '{detected}' not supported, defaulting to English")
                return Languages.ENGLISH
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return Languages.ENGLISH
        
    def fallback_detection(self, text: str) -> str:
        """Simple fallback using langdetect only."""
        try:
            detected = detect(text)
            if detected in LANGUAGES:
                logger.info(f"Fallback detected language '{detected}' for text: '{text}'")
                return detected
            else:
                return Languages.ENGLISH
        except Exception as e:
            logger.warning(f"Fallback detection failed: {e}, defaulting to English")
            return Languages.ENGLISH

    def is_number_only_text(self, text: str) -> bool:
        """Check if text contains only numbers, spaces, and basic punctuation."""
        clean_text = re.sub(r'[\s\-\.\,\(\)\/]+', '', text)
        return clean_text.isdigit() and len(clean_text) > 0
    
    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name."""
        if lang_code == Languages.URDU_ARABIC:
            return 'Urdu (Arabic Script)'
        elif lang_code == Languages.URDU_ROMAN:
            return 'Roman Urdu (English Letters)'
        return LANGUAGES.get(lang_code, lang_code.title())
    
    def get_supported_languages(self) -> dict:
        """Get all supported languages."""
        return LANGUAGES
    
    def detect_language(self, text: str) -> str:
        """Backward compatibility."""
        return self.detect_language_smart(text)

# Global instance
translation_service = TranslationService()