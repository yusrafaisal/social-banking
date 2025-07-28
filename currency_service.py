import httpx
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CurrencyConverter:
    def __init__(self):
        # Using exchangerate-api.com (free tier: 1500 requests/month)
        self.base_url = "https://api.exchangerate-api.com/v4/latest"
        
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get exchange rate between two currencies."""
        try:
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()
            
            if from_currency == to_currency:
                return 1.0
                
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/{from_currency}")
                response.raise_for_status()
                
                data = response.json()
                rates = data.get("rates", {})
                
                if to_currency in rates:
                    rate = rates[to_currency]
                    logger.info(f"Exchange rate {from_currency} to {to_currency}: {rate}")
                    return rate
                else:
                    logger.error(f"Currency {to_currency} not found in rates")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting exchange rate: {e}")
            return None
    
    async def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Optional[Dict]:
        """Convert amount from one currency to another."""
        try:
            rate = await self.get_exchange_rate(from_currency, to_currency)
            
            if rate is None:
                return None
                
            converted_amount = amount * rate
            
            return {
                "original_amount": amount,
                "from_currency": from_currency.upper(),
                "converted_amount": round(converted_amount, 2),
                "to_currency": to_currency.upper(),
                "exchange_rate": rate,
                "conversion_successful": True
            }
            
        except Exception as e:
            logger.error(f"Error converting currency: {e}")
            return None

    def get_supported_currencies(self) -> list:
        """Get list of commonly supported currencies."""
        return [
        "USD", "EUR", "GBP", "PKR", "INR", "CAD", "AUD", "JPY", 
        "CHF", "CNY", "SAR", "AED", "SGD", "HKD", "NZD"
        ]

# Global instance
currency_converter = CurrencyConverter()