import os 
import requests
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
#from main import verify_user
from mongo import users_col
from state import authenticated_users, user_sessions

router = APIRouter()

VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPJFK1Vpnhqpcv4z7gdijoUXXi1lqdyllBamRzdPVjPmxpONRdzTz18hXiRvJvpXarDX1ZAy71Lp6RXZC0rfZAOhFtY7cJGFxAw6Qabq3XBRjCcvCC8LRoYC7ad7zTjJdfZBY1fcD9ZAwZCLb2EmLBDa25ZAZBcZCWBZA7V0M6ZCewER6akH8QxYZAcAZBTR4B"


@router.get("/webhook")
async def webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Invalid verification token.")
    
@router.post("/webhook") # Handle incoming webhook events
async def receive_message(request: Request):
    body = await request.json()
    print("📩 Incoming FB Message: ", body)

    if body.get("object") == "page":
        for entry in body.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender_id = messaging_event["sender"]["id"]
                if "message" in messaging_event:
                    user_message = messaging_event["message"]["text"]

                    # Simple response logic 
                    if user_message.lower().startswith("verify"):
                        response_text = "Please send your account number, DOB, mother's name and place of birth in this format:\n\naccount_number, dob, mother_name, place_of_birth"
                    elif "," in user_message:
                        parts = [p.strip() for p in user_message.split(",")]
                        if len(parts) == 4:
                            acc, dob, mom, pob = parts
                            user = users_col.find_one({
                                "account_number": acc,
                                "dob": dob,
                                "mother_name": mom,
                                "place_of_birth": pob
                            })
                            if user:
                                authenticated_users.add(acc)
                                
                                # Store session info after successful verification
                                user_sessions[sender_id] = {
                                        "account_number": acc,
                                        "first_name": user["first_name"],
                                        "verified": True,
                                        "last_filter": None
                                        }

                                response_text = f"✅ Hi {user['first_name']}! You're now verified. You can now ask about your transactions."
                                #response_text = "✅ You are verified!"



                            else: 
                                response_text = "❌ Verification failed. Please check your info."
                        else: 
                            response_text = "❌ Format invalid. Use: account_number, dob, mother_name, place_of_birth"
                    else:
                        response_text = "👋 Welcome! Type 'verify' to start verification."

                    send_message(sender_id, response_text)
    return JSONResponse(content={"status": "success"}, status_code=200)  


# Message sender function
def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v17.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}

    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
