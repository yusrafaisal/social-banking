from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import aiofiles
import os
import time
from openai import OpenAI

# import the SAME helper used by the Facebook webhook
from webhook import process_multilingual_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize OpenAI client for transcription
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://social-banking-agent.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMsg(BaseModel):
    sender_id: str
    message: str

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)