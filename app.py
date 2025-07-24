from fastapi import FastAPI
from api_routes import router

app = FastAPI(
    title="Enhanced Banking AI Assistant - Full Authentication & Context System",
    description="""
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
    """,
    version="3.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)