import os
import dotenv
import requests
from datetime import datetime

dotenv.load_dotenv(override=True)

fmp_base_url = "https://financialmodelingprep.com"
finage_base_url = "https://api.finage.co.uk"


def _call_api(endpoint: str, params: dict = {}):
    response = requests.get(
        endpoint,
        params=params
    )
    response.raise_for_status()
    return response.json()

def _call_fmp_api(function_name: str, params: dict = {}):
    fmp_api_key = os.getenv("FMP_API_KEY")
    if not fmp_api_key:
        raise ValueError("FMP_API_KEY not found in environment variables.")
    _params = {**params, "apikey": fmp_api_key}
    return _call_api(
        f"{fmp_base_url}/{function_name}",
        _params
    )

def _call_finage_api(function_name_and_param: str):
    finage_api_key = os.getenv("FINAGE_API_KEY")
    if not finage_api_key:
        raise ValueError("FINAGE_API_KEY not found in environment variables.")
    
    _params = {"apikey": finage_api_key}
    return _call_api(
        f"{finage_base_url}/{function_name_and_param}",
        _params
    )

def get_company_financial_metrics(ticker_symbol: str):
    return _call_fmp_api(
        f"/stable/income-statement",
        {"symbol": ticker_symbol}
    )
    
def get_historical_eods_for_ticker(ticker_symbol: str, from_date: datetime, to_date: datetime):
    return _call_finage_api(
        f"agg/stock/{ticker_symbol}/1/day/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
    )