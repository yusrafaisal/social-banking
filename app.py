from fastapi import FastAPI
from api_routes import router
from fastapi.middleware.cors import CORSMiddleware

# Import constants
from constants import (
    VerificationStages, BankingIntents, WebhookConfig
)

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://social-banking.vercel.app/",
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)