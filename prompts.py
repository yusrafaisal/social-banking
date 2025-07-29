from langchain.prompts import PromptTemplate

filter_extraction_prompt = PromptTemplate(
    input_variables=["user_message", "current_date"],
    template="""
    You are a banking AI assistant. Extract relevant filters from the user's query for MongoDB aggregation.
    
    Current date: {current_date}
    
    Available database fields for new dataset structure:
    - name (string: user's full name)
    - cnic (string: National ID)
    - account_number (string: bank account number)
    - date (Date: transaction date without time)
    - type (string: "debit" or "credit")
    - description (string: McDonald, Foodpanda, Careem, JazzCash, Amazon, Uber, Netflix, etc.)
    - category (string: Food, Travel, Telecom, Shopping, Finance, Utilities, Income, Entertainment, etc.)
    - account_currency (string: "pkr" or "usd")
    - amount_deducted_from_account (number)
    - transaction_amount (number)
    - transaction_currency (string: "pkr" or "usd")
    - account_balance (number)
    
    Extract the following filters from the user query and return as JSON:
    {{
        "description": "description name if mentioned (e.g., Netflix, Uber, Amazon)",
        "category": "category if mentioned (e.g., Food, Entertainment, Travel)",
        "month": "month name if mentioned (e.g., january, june, december)",
        "year": "year if mentioned (default to 2025 if not specified)",
        "transaction_type": "debit or credit if specified",
        "amount_range": {{"min": number, "max": number}} if amount range mentioned,
        "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} if specific date range,
        "limit": number if specific count mentioned (e.g., last 10 transactions),
        "currency": "pkr or usd if specified"
    }}
    
    Rules:
    - Only include fields that are explicitly mentioned or can be inferred
    - For description names, extract the exact name mentioned (case-insensitive matching will be handled later)
    - For months, use lowercase full names (january, february, etc.)
    - For spending queries, default transaction_type to "debit"
    - If "last X transactions" mentioned, set limit to X
    - If no year specified but month is mentioned, assume 2025
    - Return null for fields not mentioned
    - Currency can be PKR or USD based on the account type
    
    Examples:
    
    Query: "how much did i spend on netflix in june"
    Response: {{
        "description": "netflix",
        "category": null,
        "month": "june",
        "year": 2025,
        "transaction_type": "debit",
        "amount_range": null,
        "date_range": null,
        "limit": null,
        "currency": null
    }}
    
    Query: "show my last 10 transactions"
    Response: {{
        "description": null,
        "category": null,
        "month": null,
        "year": null,
        "transaction_type": null,
        "amount_range": null,
        "date_range": null,
        "limit": 10,
        "currency": null
    }}
    
    Query: "how much did i spend on food last month"
    Response: {{
        "description": null,
        "category": "Food",
        "month": "june",
        "year": 2025,
        "transaction_type": "debit",
        "amount_range": null,
        "date_range": null,
        "limit": null,
        "currency": null
    }}

    Query: "what was my balance on june 15th"
    Response: {{
        "description": null,
        "category": null,
        "month": null,
        "year": null,
        "transaction_type": null,
        "amount_range": null,
        "date_range": {{"start": "2025-06-15", "end": "2025-06-15"}},
        "limit": null,
        "currency": null
    }}

    Query: "average balance in may"
    Response: {{
        "description": null,
        "category": null,
        "month": "may",
        "year": 2025,
        "transaction_type": null,
        "amount_range": null,
        "date_range": null,
        "limit": null,
        "currency": null
    }}
    
    User query: {user_message}
    ### RESPONSE FORMAT – READ CAREFULLY
    Return **exactly one** valid JSON value that fits the schema above.
    • No Markdown, no ``` fences, no comments, no keys other than the schema.
    • Do not pretty‑print; a single‑line minified object/array is required.
    • If a value is unknown, use null.
    Your entire reply must be parsable by `json.loads`.
    """
)

pipeline_generation_prompt = PromptTemplate(
    input_variables=["filters", "intent", "account_number"],
    template="""
    Generate a MongoDB aggregation pipeline based on the extracted filters and intent for the new dataset structure.

    IMPORTANT: Return ONLY the JSON array, no explanatory text, no markdown formatting.
    CRITICAL: Use proper JSON format - do NOT use ISODate() syntax. Use {{"$date": "YYYY-MM-DDTHH:mm:ss.sssZ"}} format instead.

    Account Number: {account_number}
    Intent: {intent}
    Extracted Filters: {filters}

    New Dataset Structure:
    - name: user's full name
    - cnic: National ID
    - account_number: bank account number
    - date: transaction date (Date type)
    - type: "debit" or "credit"
    - description: merchant/service name
    - category: transaction category
    - account_currency: "pkr" or "usd"
    - amount_deducted_from_account: number (THIS IS THE ACTUAL AMOUNT TAKEN FROM ACCOUNT - USE THIS FOR SPENDING ANALYSIS)
    - transaction_amount: number (original transaction amount in its currency)
    - transaction_currency: "pkr" or "usd"
    - account_balance: current balance

    CRITICAL FIELD USAGE RULES:
    - For ALL spending analysis, use "amount_deducted_from_account" NOT "transaction_amount"
    - For transaction history display, show both amounts but emphasize "amount_deducted_from_account"
    - "amount_deducted_from_account" represents the actual impact on the user's account
    - "transaction_amount" is just the original transaction value which may be in different currency or the account currency if that is what the user transacted in

    BALANCE QUERY RULES (CRITICAL):
    - Detect balance queries by keywords: "balance", "money", "amount", "funds" WITHOUT spending words
    - Balance queries need the account_balance field, NOT spending calculations
    - For "balance on [date]": Get the last transaction up to that date
    - For "average balance in [period]": Calculate average of account_balance in that period, apply avg function
    - Never filter by transaction type for balance queries
    - Never sum amounts for balance queries - use the account_balance field directly
    
    SPECIAL HANDLING FOR COMPARATIVE QUERIES:
    - For "spending more/less than" queries, create separate groups for each time period
    - Use $facet to compare multiple time periods in one pipeline
    - Current date context: July 2025

    Generate a pipeline array with the following stages as needed:
    1. $match - for filtering documents
    2. $facet - for comparing multiple time periods
    3. $group - for aggregating data (spending analysis, category totals)
    4. $sort - for ordering results
    5. $limit - for limiting results
    6. $project - for selecting specific fields

    
    CRITICAL DATE HANDLING RULES:
    - If filters contain 'date_range' with start and end dates, use EXACT date range with proper JSON format
    - If filters contain BOTH 'month' and 'year' (both not null), use full month range
    - If filters contain ONLY 'year' without month or date_range, DO NOT add any date filter
    - If filters contain null/empty month AND null/empty date_range, DO NOT add any date filter regardless of year
    - ALWAYS prioritize date_range over month/year when both are present
    - Use proper JSON date format: {{"$date": "YYYY-MM-DDTHH:mm:ss.sssZ"}}
    - For "right now" or "current" spending, use July 2025 data

    Month to date range mapping (use JSON format):
    - january: {{"$gte": {{"$date": "2025-01-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-01-31T23:59:59.999Z"}}}}
    - february: {{"$gte": {{"$date": "2025-02-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-02-28T23:59:59.999Z"}}}}
    - march: {{"$gte": {{"$date": "2025-03-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-03-31T23:59:59.999Z"}}}}
    - april: {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}
    - may: {{"$gte": {{"$date": "2025-05-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-05-31T23:59:59.999Z"}}}}
    - june: {{"$gte": {{"$date": "2025-06-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59.999Z"}}}}
    - july: {{"$gte": {{"$date": "2025-07-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-07-31T23:59:59.999Z"}}}}
    - august: {{"$gte": {{"$date": "2025-08-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-08-31T23:59:59.999Z"}}}}
    - september: {{"$gte": {{"$date": "2025-09-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-09-30T23:59:59.999Z"}}}}
    - october: {{"$gte": {{"$date": "2025-10-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-10-31T23:59:59.999Z"}}}}
    - november: {{"$gte": {{"$date": "2025-11-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-11-30T23:59:59.999Z"}}}}
    - december: {{"$gte": {{"$date": "2025-12-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-12-31T23:59:59.999Z"}}}}

    General Rules:
    - Always include account_number in $match
    - For description and category matching, use $regex with case-insensitive option
    - For spending analysis or category_spending, group by null and sum amount_deducted_from_account (NOT transaction_amount)
    - For transaction history, sort by date descending and _id descending
    - Handle currency filtering when specified
    - NEVER use ISODate() syntax - always use {{"$date": "ISO-string"}} format
    - For queries asking average balance, average spending, or total spending, use $group with $avg or $sum as appropriate

    Examples:

    Intent: spending_analysis, Query: "did I spend more in may than in april"
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "type": "debit"}}}},
        {{"$facet": {{
            "may_spending": [
                {{"$match": {{"date": {{"$gte": {{"$date": "2025-05-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-05-31T23:59:59.999Z"}}}}}}}},
                {{"$group": {{"_id": null, "total": {{"$sum": "$amount_deducted_from_account"}}, "currency": {{"$first": "$account_currency"}}}}}}
            ],
            "april_spending": [
                {{"$match": {{"date": {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}}}}},
                {{"$group": {{"_id": null, "total": {{"$sum": "$amount_deducted_from_account"}}, "currency": {{"$first": "$account_currency"}}}}}}
            ]
        }}}}
    ]

    Intent: spending_analysis, Query: "am I spending more right now than in april"
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "type": "debit"}}}},
        {{"$facet": {{
            "current_spending": [
                {{"$match": {{"date": {{"$gte": {{"$date": "2025-07-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-07-31T23:59:59.999Z"}}}}}}}},
                {{"$group": {{"_id": null, "total": {{"$sum": "$amount_deducted_from_account"}}, "currency": {{"$first": "$account_currency"}}}}}}
            ],
            "april_spending": [
                {{"$match": {{"date": {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}}}}},
                {{"$group": {{"_id": null, "total": {{"$sum": "$amount_deducted_from_account"}}, "currency": {{"$first": "$account_currency"}}}}}}
            ]
        }}}}
    ]

    Intent: spending_analysis, Filters: {{"description": "netflix", "month": "april", "year": 2025, "transaction_type": "debit"}}
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "type": "debit", "description": {{"$regex": "netflix", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}}}}},
        {{"$group": {{"_id": null, "total_amount": {{"$sum": "$amount_deducted_from_account"}}, "currency": {{"$first": "$account_currency"}}}}}}
    ]

    Intent: transaction_history, Filters: {{"limit": 10}} (set limit according to user query)
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}"}}}},
        {{"$sort": {{"date": -1, "_id": -1}}}},
        {{"$limit": 10}}
    ]

    Intent: category_spending, Filters: {{"category": "Food", "month": "april", "year": 2025, "transaction_type": "debit"}}
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "type": "debit", "category": {{"$regex": "Food", "$options": "i"}}, "date": {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}}}}},
        {{"$group": {{"_id": null, "total_amount": {{"$sum": "$amount_deducted_from_account"}}, "currency": {{"$first": "$account_currency"}}}}}}
    ]

    Intent: balance_inquiry , Query: "what was my balance on may 15th" (BALANCE QUERY)
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "date": {{"$lte": {{"$date": "2025-05-15T23:59:59.999Z"}}}}}}}},
        {{"$sort": {{"date": -1, "_id": -1}}}},
        {{"$limit": 1}},
        {{"$project": {{"account_balance": 1, "date": 1, "account_currency": 1}}}}
    ]

    Intent: balance_inquiry, Query: "average balance in june" (BALANCE QUERY)
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "date": {{"$gte": {{"$date": "2025-06-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59.999Z"}}}}}}}},
        {{"$group": {{"_id": null, "average_balance": {{"$avg": "$account_balance"}}, "currency": {{"$first": "$account_currency"}}}}}}
    ]

    Return only the JSON array pipeline.
    ### RESPONSE FORMAT – READ CAREFULLY
    Return **exactly one** valid JSON value that fits the schema above.
    • No Markdown, no ``` fences, no comments, no keys other than the schema.
    • Do not pretty‑print; a single‑line minified object/array is required.
    • If a value is unknown, use null.
    • NEVER use ISODate() - always use {{"$date": "ISO-string"}} format.
    Your entire reply must be parsable by `json.loads`.
    """
)

response_prompt = PromptTemplate(
    input_variables=["user_message", "data", "intent"],
    template="""
    You are Sage, a professional banking AI assistant. Format the API response data into a natural language answer with excellent presentation.

    🎯 **MANDATORY PROFESSIONAL BANKING FORMATTING:**
    
    💼 **BALANCE RESPONSES:**
    - "💼 **Account Balance:** [CURRENCY] **[AMOUNT]** 📅 As of: **[DATE]**"
    
    📊 **TRANSACTION FORMATTING:**
    📅 Date | 🏷️ Description | 💳 Type | 💰 Amount ([CURRENCY]) | 💼 Balance After
    [Show each transaction in this format]
    
    🔐 **VERIFICATION/SECURITY:**
    - "✅ **Verification successful.** [message]"
    - Use 🔒 🔐 ✅ ❌ emojis appropriately
    
    💸 **TRANSFER RESPONSES:**
    - "🔄 **Processing...** ✅ **Success!**"
    - Include all transfer details with emojis and **bold** formatting
    
    📋 **SPENDING ANALYSIS:**
    - Use structured breakdowns with category emojis
    - **Bold** all amounts and percentages
    - Include clear summaries with proper formatting

    ⚠️ **ERROR HANDLING:**
    - Use warning emojis and professional tone
    - Provide clear next steps

    User query: {user_message}
    Intent: {intent}
    API response data: {data}

    Guidelines for dataset structure:
    - For balance_inquiry: Use the mandatory balance format above
    - For transaction_history: Use the mandatory transaction table format
    - For spending_analysis: Use structured analysis with emojis and bold formatting
    - For category_spending: Provide formatted category breakdowns
    - For transfer_money: Use the mandatory transfer format
    - For general: Provide helpful responses with proper formatting
    - Handle errors with professional formatting using warning emojis
    - Handle both PKR and USD currencies with consistent formatting
    - Use transaction_amount and transaction_currency appropriately

    **Apply the mandatory formatting rules above to create a professional banking experience.**
    Convert the data into a finished, professionally formatted message.
    """
)


query_prompt = PromptTemplate(
    input_variables=["user_message", "current_date"],
    template="""
    You are a banking AI assistant. Analyze the user's query and return a valid JSON response with:
    1. "intent" - one of: balance_inquiry, transaction_history, spending_analysis, category_spending, transfer_money, general
    2. "pipeline" - MongoDB aggregation pipeline to fetch the required data
    3. "response_format" - "natural_language"

    Current date: {current_date}

    MongoDB collection structure (transactions):
    {{
        "name": "string (user's full name)",
        "cnic": "string (National ID)",
        "account_number": "string (bank account number)",
        "date": "Date (transaction date)",
        "type": "string (debit/credit)",
        "description": "string (merchant/service)",
        "category": "string (transaction category)",
        "account_currency": "string (pkr/usd)",
        "amount_deducted_from_account": "number",
        "transaction_amount": "number",
        "transaction_currency": "string (pkr/usd)",
        "account_balance": "number (current balance)"
    }}

    Guidelines:
    - For balance_inquiry, get latest transaction for account balance. Set pipeline to get most recent transaction.
    - For transaction_history, use $match, $sort, and optional $limit in the pipeline.
    - For spending_analysis, use $match and $group to aggregate transaction_amount by description, category, or date range.
    - For category_spending, use $match and $group for category aggregation.
    - For transfer_money, set pipeline to [] and handle via API.
    - IMPORTANT: Use proper JSON date format: {{"$date": "YYYY-MM-DDTHH:mm:ss.sssZ"}} NOT ISODate() syntax.
    - For relative dates (e.g., "last month"), calculate appropriate date ranges based on {current_date}.
    - Ensure the pipeline is valid MongoDB syntax and safe to execute.
    - Handle both PKR and USD currencies appropriately.

    User query: {user_message}
    ### RESPONSE FORMAT – READ CAREFULLY
    Return **exactly one** valid JSON value that fits the schema above.
    • No Markdown, no ``` fences, no comments, no keys other than the schema.
    • Do not pretty‑print; a single‑line minified object/array is required.
    • If a value is unknown, use null.
    • NEVER use ISODate() - always use {{"$date": "ISO-string"}} format.
    Your entire reply must be parsable by `json.loads`.
    """
)

intent_prompt = PromptTemplate(
    input_variables=["user_message", "filters"],
    template="""
    You are a banking AI assistant. Analyze the user's query and classify it into one of these intents:

    Available intents:
    1. "balance_inquiry" - User wants to check their account balance or financial capacity
    Examples: "What's my balance?", "How much money do I have?", "Can I afford X?", "Do I have enough for X?", "What can I do to hit that target?", "How can I save for X?"
    
    2. "transaction_history" - User wants to see their transaction history/list
    Examples: "Show my transactions", "List my recent purchases", "What are my last 10 transactions?", "Transaction history"
    
    3. "spending_analysis" - User wants to analyze their spending patterns, compare periods, or get spending insights
    Examples: "How much did I spend on Netflix?", "Am I spending more than last month?", "Compare my spending", "spending patterns", "My spending habits", "How much do I spend on average?", "Am I spending more right now than I did in April?"
    
    4. "category_spending" - User wants to analyze spending by specific categories
    Examples: "How much did I spend on food?", "My entertainment expenses", "Food spending last month", "How much on groceries?"
    
    5. "transfer_money" - User wants to transfer money to someone
    Examples: "Transfer money to John", "Send 100 PKR to Alice", "Pay my friend", "Transfer funds"
    
    6. "general" - Only for greetings, unclear requests, or questions about bot capabilities
    Examples: "Hello", "Hi", "What can you do?", "Help me understand your features"

    IMPORTANT Classification Guidelines:
    - Financial planning questions ("can I afford", "what can I do to hit target", "how to save") → "balance_inquiry"
    - Spending comparisons ("spending more than", "compared to last month") → "spending_analysis"  
    - Any spending pattern analysis → "spending_analysis"
    - Purchase planning with amounts → "balance_inquiry"
    - Time-based spending questions → "spending_analysis"
    - Be aggressive in classifying as banking intents rather than "general"

    Consider these extracted filters to help classification:
    - If filters.limit is set → likely "transaction_history"
    - If filters.description is set → likely "spending_analysis"
    - If filters.category is set → likely "category_spending"
    - If filters.transaction_type is "debit" and specific merchant → likely "spending_analysis"

    User query: "{user_message}"
    Extracted filters: {filters}

    Respond with only the intent name (e.g., "balance_inquiry", "spending_analysis", etc.)
    """
)

transfer_prompt = PromptTemplate(
    input_variables=["user_message"],
    template="""
    Extract transfer details from the query for the new dataset structure:
    - amount: number
    - currency: "PKR" or "USD" (default to "PKR" if not specified, since most transactions are in PKR)
    - recipient: string
    
    Return JSON: {{"amount": number, "currency": string, "recipient": string}}
    
    Examples:
    "Transfer 500 to John" → {{"amount": 500, "currency": "PKR", "recipient": "John"}}
    "Send 50 USD to Alice" → {{"amount": 50, "currency": "USD", "recipient": "Alice"}}
    "Pay 1000 PKR to Ahmed" → {{"amount": 1000, "currency": "PKR", "recipient": "Ahmed"}}
    
    Query: {user_message}
    """
)



account_selection_prompt = PromptTemplate(
    input_variables=["user_input", "account_details"],
    template="""
    You are a banking assistant helping a user select their account. Analyze the user's input and return the exact account number they want to select.

    User input: "{user_input}"
    
    Available accounts:
    {account_details}

    The user can specify accounts in various ways:
    - Currency: "usd account", "pkr account", "my USD account", "pakistani rupee account"
    - Position: "first account", "1st account", "second account", "2nd account", "third", etc.
    - Account number: full account number or last 4/5/6 digits
    - Account type: "savings account", "current account", "checking account"
    - Natural language: "my dollar account", "rupee wala account", etc.

    Rules:
    1. If user mentions USD/dollar/$ → select USD account
    2. If user mentions PKR/rupee/pakistani/rs → select PKR account  
    3. If user mentions position (first, second, 1st, 2nd) → select by order in the list
    4. If user provides digits → match against account numbers (exact match or ending digits)
    5. If ambiguous → select the first account that matches the criteria
    6. If no clear match → return "NO_MATCH"

    Examples:
    - "my usd account" → return the USD account number
    - "1st account" → return the first account in the list
    - "second" → return the second account in the list  
    - "1234" → return account ending in 1234
    - "12345678" → return exact account number match
    - "savings" → return PKR account (more common for savings)

    Return ONLY the exact account number (e.g., "1234567890") or "NO_MATCH" if no clear selection can be made.
    """
)


currency_conversion_intent_prompt = PromptTemplate(
    input_variables=["user_message", "conversation_history"],
    template="""
    You are analyzing if a user wants to convert currency amounts from their banking conversation.

    Conversation History (last few messages):
    {conversation_history}

    Current User Message: "{user_message}"

    Currency conversion indicators:
    - "convert this to [currency]"
    - "what is this in USD/GBP/EUR"  
    - "show me in dollars/pounds/euros"
    - "convert to [currency]"
    - "in GBP please"
    - "what's that in USD"

    Return "YES" if user wants currency conversion, "NO" otherwise.
    """
)

currency_extraction_prompt = PromptTemplate(
    input_variables=["user_message", "conversation_history"],
    template="""
    Extract currency conversion details from the conversation.

    Conversation History:
    {conversation_history}

    User Message: "{user_message}"

    Find:
    1. The amount to convert (look in recent conversation history)
    2. Source currency (currency of the amount mentioned previously)
    3. Target currency (what user wants to convert to)

    Return JSON:
    {{
        "amount": number (the amount to convert from conversation),
        "from_currency": "string (PKR/USD/EUR/GBP etc.)",
        "to_currency": "string (target currency user wants)",
        "context": "brief description of what amount is being converted"
    }}

    Examples:
    History: "Your balance is 50,000 PKR"
    Query: "convert this to USD" 
    → {{"amount": 50000, "from_currency": "PKR", "to_currency": "USD", "context": "account balance"}}

    History: "You spent $125.50 on groceries"  
    Query: "what's that in GBP"
    → {{"amount": 125.50, "from_currency": "USD", "to_currency": "GBP", "context": "grocery spending"}}

    History: "Your balance is 150,000 KES"
    Query: "convert to USD"
    → {{"amount": 150000, "from_currency": "KES", "to_currency": "USD", "context": "account balance"}}
    
    History: "You sent 500,000 UGX to John"
    Query: "what's that in KES"
    → {{"amount": 500000, "from_currency": "UGX", "to_currency": "KES", "context": "money transfer"}}
    
    History: "Transaction of 1,200 ETB for groceries"
    Query: "show me in dollars"
    → {{"amount": 1200, "from_currency": "ETB", "to_currency": "USD", "context": "grocery spending"}}

    Return only valid JSON.
    """
)