# chat_api.py  (sits next to webhook.py)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# import the SAME helper used by the Facebook webhook
from webhook import process_multilingual_message   # relative import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)