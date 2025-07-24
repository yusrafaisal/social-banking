from fastapi import FastAPI
from api_routes import router

app = FastAPI(
    title="Banking AI Assistant - CNIC Authentication",
    description="CNIC-based banking assistant with account selection and LangChain memory",
    version="2.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)