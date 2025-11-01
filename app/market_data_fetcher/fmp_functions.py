import os
import dotenv
import requests

dotenv.load_dotenv(override=True)

fmp_base_url = os.getenv("FMP_BASE_URL")


def call_fmp_api(function_name: str, params: dict = {}):
    fmp_api_key = os.getenv("FMP_API_KEY")
    if not fmp_api_key:
        raise ValueError("FMP_API_KEY not found in environment variables.")
    params["apikey"] = fmp_api_key
    response = requests.get(
        f"{fmp_base_url}/{function_name}",
        params=params
    )
    response.raise_for_status()
    return response.json()




def get_historical_data_for_company(ticker_symbol: str, start_date: str, end_date: str):
    return call_fmp_api(
        "historical-price-eod/light",
        {
            "symbol": ticker_symbol,
            "from": start_date,
            "to": end_date
        }
    )