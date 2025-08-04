from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import aiofiles
import os
import time
from openai import OpenAI

from api_routes import router
from webhook import process_multilingual_message

# Import constants
from constants import (
    VerificationStages, BankingIntents, WebhookConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Banking AI Assistant - Full Authentication & Context System",
    description=f"""
    Advanced CNIC-based banking assistant with comprehensive authentication and intelligent context management.
    
    Features:
    • CNIC Verification + OTP Authentication
    • Smart Account Selection (USD/PKR, 1st/2nd, Last 4 digits)
    • Enhanced Transfer Flow (OTP + Confirmation)
    • Banking Context Filtering (Politely declines non-banking queries)
    • Intelligent Context Memory System
    • Session Restart Functionality (Page refresh simulation)
    • Natural Language Processing with LangChain
    • Varied Response Generation (No repetitive patterns)
    
    Authentication Flow: CNIC → OTP → Smart Account Selection → Full Banking Access
    Transfer Flow: Details → OTP → Confirmation → Execution
    Context Management: Multi-layer memory with intelligent reference resolution
    
    Verification Stages:
    • {VerificationStages.NOT_VERIFIED}: Initial state
    • {VerificationStages.CNIC_VERIFIED}: CNIC verified, awaiting OTP
    • {VerificationStages.OTP_VERIFIED}: OTP verified, awaiting account selection
    • {VerificationStages.ACCOUNT_SELECTED}: Fully authenticated
    • {VerificationStages.TRANSFER_OTP_PENDING}: Transfer OTP required
    • {VerificationStages.TRANSFER_CONFIRMATION_PENDING}: Transfer confirmation required
    
    Banking Intents:
    • {BankingIntents.BALANCE_INQUIRY}: Check account balance
    • {BankingIntents.TRANSACTION_HISTORY}: View transaction history
    • {BankingIntents.SPENDING_ANALYSIS}: Analyze spending patterns
    • {BankingIntents.CATEGORY_SPENDING}: Category-based spending analysis
    • {BankingIntents.TRANSFER_MONEY}: Money transfer operations
    • {BankingIntents.GENERAL}: General banking assistance
    """,
    version="3.0.0"
)

# Initialize OpenAI client for transcription
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://social-banking.vercel.app",  # Remove trailing slash
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat message model
class ChatMsg(BaseModel):
    sender_id: str
    message: str

# Add chat endpoints
@app.post("/api/chat")
async def chat(msg: ChatMsg):
    logger.info({"react_chat": msg.dict()})
    reply = await process_multilingual_message(msg.sender_id, msg.message)
    return {"reply": reply}

@app.post("/api/voice")
async def handle_voice_message(
    audio: UploadFile = File(...),
    sender_id: str = Form(...)
):
    """Handle voice message uploads from React frontend."""
    try:
        logger.info(f"🎤 Received voice message from {sender_id}")
        
        # Create unique filename to avoid conflicts
        current_time = int(time.time())
        audio_file_path = f"temp_audio_web_{sender_id}_{current_time}.webm"
        
        try:
            # Save uploaded file
            async with aiofiles.open(audio_file_path, 'wb') as audio_file:
                content = await audio.read()
                await audio_file.write(content)
            
            logger.info(f"🎤 Saved audio file: {audio_file_path} ({len(content)} bytes)")
            
            # Transcribe using OpenAI Whisper
            transcription = await transcribe_audio_file(audio_file_path)
            logger.info(f"🎤 Transcribed: '{transcription}'")
            
            # Process transcribed text through existing pipeline
            reply = await process_multilingual_message(sender_id, transcription)
            
            return {"reply": reply, "transcription": transcription}
            
        finally:
            # Clean up audio file
            try:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                    logger.info(f"🗑️ Cleaned up: {audio_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return {"reply": "Sorry, I couldn't process your voice message. Please try typing instead."}

async def transcribe_audio_file(audio_file_path: str) -> str:
    """Transcribe audio file using OpenAI Whisper."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response.strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise Exception("Failed to transcribe audio")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "features": {
            "text_chat": "enabled",
            "voice_chat": "enabled",
            "transcription": "whisper-1"
        }
    }

# Include existing API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)