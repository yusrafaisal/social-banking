from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your existing apps
from app import app as main_app
from chat_api import app as chat_app

# Create a new combined app
app = FastAPI(title="Combined Banking App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount both apps
app.mount("/api", chat_app)
app.mount("/main", main_app)

# Add a health check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Banking app is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)