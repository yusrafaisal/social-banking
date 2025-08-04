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
    description="Advanced CNIC-based banking assistant with comprehensive authentication.",
    version="3.0.0"
)

# Initialize OpenAI client for transcription
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://social-banking.vercel.app",  # Your Vercel URL
        "http://localhost:3000",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat message model
class ChatMsg(BaseModel):
    sender_id: str
    message: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Kora AI Banking Assistant API",
        "status": "running",
        "version": "3.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "voice": "/api/voice", 
            "health": "/health",
            "docs": "/docs"
        }
    }

# Chat endpoints (from chat_api.py)
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
        
        current_time = int(time.time())
        audio_file_path = f"temp_audio_web_{sender_id}_{current_time}.webm"
        
        try:
            async with aiofiles.open(audio_file_path, 'wb') as audio_file:
                content = await audio.read()
                await audio_file.write(content)
            
            transcription = await transcribe_audio_file(audio_file_path)
            reply = await process_multilingual_message(sender_id, transcription)
            
            return {"reply": reply, "transcription": transcription}
            
        finally:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        
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

# Include existing API routes (webhook, etc.)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)